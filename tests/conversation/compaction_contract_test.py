"""Verify strict compact state, replay boundaries, and explicit forks."""

from ast import Attribute, Call, Name, parse, walk
from asyncio import CancelledError, gather
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from inspect import getsource
from json import dumps
from textwrap import dedent
from typing import cast

import httpx
import pytest
from native_openai_compaction_test import (
    _binding_with_limits as _native_binding_with_limits,
)
from native_openai_compaction_test import (
    _limits as _native_limits,
)
from native_openai_compaction_test import (
    _message as _native_message,
)
from native_openai_compaction_test import (
    _response as _native_response,
)
from native_openai_compaction_test import (
    _stateless_provider as _native_stateless_provider,
)
from phase2_fixtures import authority, binding, empty_stateless_plan, retention

import avalan
import avalan.conversation as conversation
from avalan.conversation import coordinator as coordinator_module
from avalan.conversation import items as items_module
from avalan.conversation import sdk as sdk_module
from avalan.conversation.providers import openai as openai_provider_module
from avalan.conversation.providers import (
    openai_stored as openai_stored_provider_module,
)
from avalan.types import JsonValue

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run compact contract tests on asyncio only."""
    return "asyncio"


def _ignore_acceptance_evidence(name: str, value: object) -> None:
    """Validate and discard delegated acceptance evidence."""
    assert name == "conversation_acceptance_evidence"
    assert isinstance(value, str)


def _message(
    lane_id: conversation.ProviderLaneId,
    order: int,
    name: str,
) -> conversation.ProviderItem:
    return conversation.ProviderItem(
        item_id=conversation.ProviderItemId(name),
        lane_id=lane_id,
        model_call_id=conversation.ConversationModelCallId(f"call-{name}"),
        kind=conversation.ProviderItemKind.MESSAGE,
        order=conversation.ProviderItemOrder(order),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.FINAL,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input={
            "content": (
                {
                    "annotations": (),
                    "text": name,
                    "type": "output_text",
                },
            ),
            "id": name,
            "role": "assistant",
            "status": "completed",
            "type": "message",
        },
        normalization_version=conversation.ConversationCodecVersion(1),
    )


def _compaction(
    lane_id: conversation.ProviderLaneId,
    order: int,
    name: str,
) -> conversation.ProviderItem:
    return conversation.ProviderItem(
        item_id=conversation.ProviderItemId(name),
        lane_id=lane_id,
        model_call_id=conversation.ConversationModelCallId(f"call-{name}"),
        kind=conversation.ProviderItemKind.COMPACTION,
        order=conversation.ProviderItemOrder(order),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.COMPACTION,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input={
            "created_by": "provider-test",
            "id": name,
            "type": "compaction",
        },
        normalization_version=conversation.ConversationCodecVersion(1),
        opaque_state=conversation.OpaqueProviderState(
            _value=f"opaque-{name}".encode()
        ),
    )


def _caller_message(
    lane_id: conversation.ProviderLaneId,
    order: int,
    name: str,
) -> conversation.ProviderItem:
    return conversation.ProviderItem(
        item_id=conversation.ProviderItemId(name),
        lane_id=lane_id,
        model_call_id=conversation.ConversationModelCallId(f"call-{name}"),
        kind=conversation.ProviderItemKind.MESSAGE,
        order=conversation.ProviderItemOrder(order),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.INPUT,
        caller=conversation.ProviderItemCaller.CALLER,
        canonical_input={
            "content": ({"text": name, "type": "input_text"},),
            "role": "user",
            "type": "message",
        },
        normalization_version=conversation.ConversationCodecVersion(1),
    )


def _tool_pair(
    lane_id: conversation.ProviderLaneId,
    order: int,
    name: str,
) -> tuple[conversation.ProviderItem, conversation.ProviderItem]:
    call_id = conversation.ProviderCallId(f"tool-{name}")
    model_call_id = conversation.ConversationModelCallId(f"call-{name}")
    call = conversation.ProviderItem(
        item_id=conversation.ProviderItemId(f"{name}-call"),
        lane_id=lane_id,
        model_call_id=model_call_id,
        kind=conversation.ProviderItemKind.FUNCTION_CALL,
        order=conversation.ProviderItemOrder(order),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.ASSISTANT,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input={
            "arguments": "{}",
            "call_id": call_id,
            "id": f"{name}-call",
            "name": "lookup",
            "status": "completed",
            "type": "function_call",
        },
        normalization_version=conversation.ConversationCodecVersion(1),
        call_id=call_id,
    )
    output = conversation.ProviderItem(
        item_id=conversation.ProviderItemId(f"{name}-output"),
        lane_id=lane_id,
        model_call_id=model_call_id,
        kind=conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT,
        order=conversation.ProviderItemOrder(order + 1),
        provider_index=conversation.ProviderItemIndex(1),
        phase=conversation.ProviderItemPhase.TOOL,
        caller=conversation.ProviderItemCaller.TOOL,
        canonical_input={
            "call_id": call_id,
            "output": "result",
            "type": "function_call_output",
        },
        normalization_version=conversation.ConversationCodecVersion(1),
        call_id=call_id,
    )
    return call, output


def _ledger(
    items: tuple[conversation.ProviderItem, ...],
    lane_id: conversation.ProviderLaneId,
) -> conversation.ProviderItemLedger:
    return conversation.ProviderItemLedger(
        lane_id=lane_id,
        normalization_version=conversation.ConversationCodecVersion(1),
        items=items,
    )


def _verify_latest_boundary_and_tool_adjacency() -> None:
    """Select only the latest compaction item and its unchanged suffix."""
    lane_id = conversation.ProviderLaneId("lane-boundaries")
    tool_call, tool_output = _tool_pair(lane_id, 2, "cycle")
    items = (
        _message(lane_id, 0, "prefix"),
        _compaction(lane_id, 1, "compact-one"),
        tool_call,
        tool_output,
        _compaction(lane_id, 4, "compact-two"),
        _message(lane_id, 5, "suffix"),
    )
    ledger = _ledger(items, lane_id)

    replay = conversation.provider_replay_items(ledger)
    assert replay == items[4:]
    assert replay[0] is items[4]
    assert replay[1] is items[5]
    assert replay[0].opaque_state is items[4].opaque_state
    no_boundary = _ledger((_message(lane_id, 0, "only"),), lane_id)
    assert conversation.provider_replay_items(no_boundary) is (
        no_boundary.items
    )
    suffix_call, suffix_output = _tool_pair(lane_id, 1, "suffix-cycle")
    suffix_ledger = _ledger(
        (
            _compaction(lane_id, 0, "suffix-boundary"),
            suffix_call,
            suffix_output,
        ),
        lane_id,
    )
    assert conversation.provider_replay_items(suffix_ledger) == (
        suffix_ledger.items
    )
    boundary = conversation.CompactionBoundary(
        boundary_item_id=items[4].item_id,
        boundary_order=items[4].order,
        retained_suffix=(items[5].item_id,),
    )
    boundary.validate_latest(ledger)


def test_latest_boundary_retains_exact_suffix_and_tool_adjacency(
    record_property: Callable[[str, object], None],
) -> None:
    """Select only the latest compaction item and its unchanged suffix."""
    record_property("conversation_acceptance_evidence", "contract")
    _verify_latest_boundary_and_tool_adjacency()


def _verify_standalone_canonical_context() -> (
    tuple[conversation.ProviderItem, ...]
):
    """Replay a standalone provider context item-for-item without pruning."""
    lane_id = conversation.ProviderLaneId("lane-standalone-context")
    caller = conversation.ProviderItem(
        item_id=conversation.ProviderItemId("standalone-user"),
        lane_id=lane_id,
        model_call_id=conversation.ConversationModelCallId("standalone-call"),
        kind=conversation.ProviderItemKind.MESSAGE,
        order=conversation.ProviderItemOrder(0),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.INPUT,
        caller=conversation.ProviderItemCaller.CALLER,
        canonical_input={
            "content": ({"text": "retained", "type": "input_text"},),
            "role": "user",
            "type": "message",
        },
        normalization_version=conversation.ConversationCodecVersion(1),
    )
    compact = _compaction(lane_id, 1, "standalone-boundary")
    compact = conversation.ProviderItem(
        item_id=compact.item_id,
        lane_id=compact.lane_id,
        model_call_id=caller.model_call_id,
        kind=compact.kind,
        order=compact.order,
        provider_index=conversation.ProviderItemIndex(1),
        phase=compact.phase,
        caller=compact.caller,
        canonical_input=compact.canonical_input,
        normalization_version=compact.normalization_version,
        opaque_state=compact.opaque_state,
    )
    ledger = _ledger((caller, compact), lane_id)
    replay = conversation.provider_replay_items(ledger)
    assert replay == ledger.items
    return replay


def test_standalone_canonical_context_keeps_provider_returned_user_messages(
    record_property: Callable[[str, object], None],
) -> None:
    """Replay a standalone provider context item-for-item without pruning."""
    record_property("conversation_acceptance_evidence", "contract")
    _verify_standalone_canonical_context()


def test_standalone_canonical_context_preserves_provider_items(
    record_property: Callable[[str, object], None],
) -> None:
    """Preserve the exact provider-returned standalone context."""
    record_property("conversation_acceptance_evidence", "contract")
    assert tuple(
        item.item_id for item in _verify_standalone_canonical_context()
    ) == (
        conversation.ProviderItemId("standalone-user"),
        conversation.ProviderItemId("standalone-boundary"),
    )


def test_latest_boundary_rejects_broken_tool_suffix() -> None:
    """Reject a compact boundary inserted inside one call/output pair."""
    lane_id = conversation.ProviderLaneId("lane-broken-adjacency")
    call, output = _tool_pair(lane_id, 0, "split")
    compact = _compaction(lane_id, 1, "compact-split")
    output = conversation.ProviderItem(
        item_id=output.item_id,
        lane_id=output.lane_id,
        model_call_id=output.model_call_id,
        kind=output.kind,
        order=conversation.ProviderItemOrder(2),
        provider_index=output.provider_index,
        phase=output.phase,
        caller=output.caller,
        canonical_input=output.canonical_input,
        normalization_version=output.normalization_version,
        call_id=output.call_id,
    )
    ledger = _ledger((call, compact, output), lane_id)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.provider_replay_items(ledger)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.provider_replay_items(
            cast(conversation.ProviderItemLedger, object())
        )
    duplicate_boundary = _compaction(lane_id, 0, "duplicate-boundary")
    with pytest.raises(conversation.ConversationValidationError):
        _ledger((call, duplicate_boundary), lane_id)
    skipped_boundary = _compaction(lane_id, 2, "skipped-boundary")
    with pytest.raises(conversation.ConversationValidationError):
        _ledger((call, skipped_boundary), lane_id)


def test_replay_rejects_corrupted_post_boundary_call_state() -> None:
    """Reject internally corrupted duplicate and unterminated tool suffixes."""
    lane_id = conversation.ProviderLaneId("lane-corrupt-replay")
    boundary = _compaction(lane_id, 0, "corrupt-boundary")
    call, output = _tool_pair(lane_id, 1, "corrupt-call")
    ledger = _ledger((boundary, call, output), lane_id)

    object.__setattr__(ledger, "items", (boundary, call))
    with pytest.raises(conversation.ConversationValidationError):
        conversation.provider_replay_items(ledger)

    duplicate = replace(
        call,
        item_id=conversation.ProviderItemId("duplicate-call"),
        order=conversation.ProviderItemOrder(2),
        provider_index=conversation.ProviderItemIndex(1),
        canonical_input={**call.canonical_input, "id": "duplicate-call"},
    )
    object.__setattr__(ledger, "items", (boundary, call, duplicate))
    with pytest.raises(conversation.ConversationValidationError):
        conversation.provider_replay_items(ledger)


def _client(
    results: tuple[conversation.ProviderResult, ...],
    *,
    lane_binding: conversation.ProviderLaneBinding | None = None,
    store: conversation.InMemoryConversationStore | None = None,
    namespace: str = "compact-explicit",
    provider_controller: (
        conversation.DeterministicFaultController | None
    ) = None,
    publisher: conversation.ConversationPublisher | None = None,
    observer: conversation.DeterministicFakeObserver | None = None,
    clock: conversation.DeterministicFakeClock | None = None,
    retention_limits: conversation.RetentionLimits | None = None,
) -> tuple[
    avalan.DirectConversationClient,
    conversation.InMemoryConversationStore,
    conversation.RunScopedConversationCoordinator,
    conversation.AuthorityScope,
]:
    lane = lane_binding or binding("lane-compact-explicit")
    scope = authority()
    selected_store = store or conversation.InMemoryConversationStore()
    coordinator = conversation.RunScopedConversationCoordinator(
        store=selected_store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=clock
        or conversation.DeterministicFakeClock(
            datetime(2026, 8, 2, tzinfo=UTC)
        ),
        publisher=publisher or conversation.DeterministicFakePublisher(),
        observer=observer or conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(
            conversation.ConversationLaneRuntime(
                binding=lane,
                capability_profile=conversation.fake_capability_profile(lane),
                provider_script=conversation.DeterministicFakeProviderScript(
                    results=results,
                    controller=provider_controller,
                ),
            ),
        ),
    )
    runtime = avalan.DirectConversationRuntime(
        coordinator=coordinator,
        store=selected_store,
        authority=scope,
        lane=lane,
        retention=retention_limits or retention(),
        id_namespace=namespace,
    )
    return (
        avalan.DirectConversationClient(runtime),
        selected_store,
        coordinator,
        scope,
    )


def _native_client(
    *,
    scope: conversation.AuthorityScope,
    lane: conversation.ProviderLaneBinding,
    provider: conversation.NativeOpenAIStatelessProvider,
    namespace: str,
) -> tuple[
    avalan.DirectConversationClient,
    conversation.InMemoryConversationStore,
    conversation.RunScopedConversationCoordinator,
]:
    """Return one direct client around an exact native provider lane."""
    store = conversation.InMemoryConversationStore()
    coordinator = conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(
            datetime(2026, 8, 2, tzinfo=UTC)
        ),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(
            conversation.NativeOpenAIConversationLaneRuntime(
                provider=provider
            ),
        ),
    )
    client = avalan.DirectConversationClient(
        avalan.DirectConversationRuntime(
            coordinator=coordinator,
            store=store,
            authority=scope,
            lane=lane,
            retention=retention(),
            id_namespace=namespace,
        )
    )
    return client, store, coordinator


def _checkpoint_bytes(
    checkpoint: conversation.ConversationCheckpoint,
) -> bytes:
    """Return canonical bytes for one parent-identity assertion."""
    return conversation.ConversationCheckpointCodec().encode(checkpoint)


def _assert_compaction_failure(
    coordinator: conversation.RunScopedConversationCoordinator,
    *,
    operation: conversation.CompactionOperation,
    boundary: conversation.FailureBoundary,
    error_code: conversation.ConversationErrorCode | str,
    cancelled: bool,
    committed: bool,
    streaming: bool,
    private_canary: str,
) -> None:
    """Assert one exact content-free coordinator failure record."""
    diagnostics = coordinator.diagnostics
    assert diagnostics.compaction_failure_count == 1
    assert len(diagnostics.compaction_failures) == 1
    assert diagnostics.compaction_failures[0] == (
        conversation.CompactionFailureRecord(
            operation=operation,
            boundary=boundary,
            error_code=error_code,
            cancelled=cancelled,
            committed=committed,
            streaming=streaming,
        )
    )
    assert private_canary not in repr(diagnostics)


def _sse_body(events: tuple[dict[str, object], ...]) -> str:
    """Return one exact test-only Responses event stream."""
    return (
        "".join(f"data: {dumps(event)}\n\n" for event in events)
        + "data: [DONE]\n\n"
    )


async def _committed_stream_result(
    stream: avalan.DirectConversationStream,
) -> avalan.DirectConversationResult:
    """Consume one stream and return its post-commit terminal result."""
    events = tuple([event async for event in stream])
    assert events
    terminal = events[-1]
    assert type(terminal) is avalan.DirectConversationStreamTerminal
    return terminal.result


async def _prepared_compaction(
    namespace: str,
    *,
    store: conversation.InMemoryConversationStore | None = None,
    clock: conversation.DeterministicFakeClock | None = None,
    retention_limits: conversation.RetentionLimits | None = None,
    named_head: bool = False,
) -> tuple[
    avalan.DirectConversationClient,
    conversation.InMemoryConversationStore,
    conversation.RunScopedConversationCoordinator,
    conversation.AuthorityScope,
    avalan.StandaloneCompactResult,
]:
    lane = binding("lane-compact-explicit")
    plan = empty_stateless_plan(lane)
    client, selected_store, coordinator, scope = _client(
        (
            conversation.fake_provider_result(plan, turn=1),
            conversation.fake_compaction_result(plan, turn=2),
        ),
        store=store,
        namespace=namespace,
        clock=clock,
        retention_limits=retention_limits,
    )
    first = await client.create(
        "compact parent",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    parent = avalan.StatelessParent(handle=first.handle)
    head_parent = None
    if named_head:
        head_id = conversation.NamedHeadId(f"{namespace}-head")
        await selected_store.create_head(
            conversation.NamedHeadSnapshot(
                head_id=head_id,
                revision=conversation.NamedHeadRevision(0),
                checkpoint_id=first.handle.checkpoint_id,
            ),
            scope,
        )
        head_parent = avalan.NamedHeadParent(
            head_id=head_id,
            expected_revision=conversation.NamedHeadRevision(0),
            parent=parent,
        )
    compacted = await client.compact(
        avalan.StandaloneCompactRequest(
            parent=parent,
            named_head=head_parent,
        )
    )
    return client, selected_store, coordinator, scope, compacted


async def test_direct_compact_rejects_named_head_metadata_drift() -> None:
    """Reject private compact handles that omit or replace bound head state."""
    client, _, coordinator, _, compacted = await _prepared_compaction(
        "compact-head-metadata-drift",
        named_head=True,
    )
    assert compacted.handle.head_id is not None
    forged_handles = (
        replace(
            compacted.handle,
            head_id=None,
            expected_head_revision=None,
        ),
        replace(
            compacted.handle,
            head_id=conversation.NamedHeadId("different-compact-head"),
        ),
    )
    for handle in forged_handles:
        with pytest.raises(conversation.ConversationValidationError):
            await client.commit_compact(replace(compacted, handle=handle))
    await coordinator.close()


def _compact_child_identity(
    source: conversation.ConversationCheckpoint,
    suffix: str,
) -> conversation.CheckpointIdentity:
    """Return one valid explicit child identity for private compact state."""
    return conversation.CheckpointIdentity(
        conversation_id=source.identity.conversation_id,
        logical_turn_id=conversation.LogicalTurnId(f"turn-{suffix}"),
        execution_segment_id=conversation.ExecutionSegmentId(
            f"segment-{suffix}"
        ),
        checkpoint_id=conversation.CheckpointId(f"checkpoint-{suffix}"),
        branch_id=source.identity.branch_id,
        sequence=conversation.CheckpointSequence(source.identity.sequence + 1),
        parent_checkpoint_id=source.identity.checkpoint_id,
        parent_sequence=source.identity.sequence,
    )


async def test_compact_commit_rejects_mismatched_head_advance() -> None:
    """Reject a coordinator head advance not bound to compact source state."""
    _, store, coordinator, scope, compacted = await _prepared_compaction(
        "compact-coordinator-head-drift",
        named_head=True,
    )
    source = await store.load(compacted.handle.checkpoint_id, scope)
    parent_id = source.identity.parent_checkpoint_id
    assert parent_id is not None
    with pytest.raises(conversation.ConversationValidationError):
        await coordinator.commit_compact_result(
            source,
            _compact_child_identity(source, "coordinator-head-drift"),
            scope,
            advance=conversation.NamedHeadAdvance(
                head_id=conversation.NamedHeadId("different-source-head"),
                parent_checkpoint_id=parent_id,
                expected_revision=conversation.NamedHeadRevision(0),
            ),
        )
    await coordinator.close()


@pytest.mark.parametrize(
    "recovery",
    ("success", "load-error", "mismatch"),
)
async def test_named_head_compact_commit_recovery_validates_advanced_head(
    recovery: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover lost success only when the exact named head also advanced."""
    client, store, coordinator, scope, compacted = await _prepared_compaction(
        f"compact-head-recovery-{recovery}",
        named_head=True,
    )
    original_create = (
        conversation.InMemoryConversationStore.create_with_named_head
    )
    original_load_head = conversation.InMemoryConversationStore.load_head

    async def create_then_lose_response(
        active: conversation.InMemoryConversationStore,
        candidate: conversation.CheckpointCandidate,
        advance: conversation.NamedHeadAdvance,
    ) -> conversation.ConversationCheckpoint:
        committed = await original_create(active, candidate, advance)
        assert committed.lifecycle is (
            conversation.CheckpointLifecycle.COMMITTED
        )
        if active is store:
            raise conversation.ConversationConflictError()
        return committed

    async def recovery_head(
        active: conversation.InMemoryConversationStore,
        head_id: conversation.NamedHeadId,
        authority_scope: conversation.AuthorityScope,
    ) -> conversation.NamedHeadSnapshot:
        if active is store and recovery == "load-error":
            raise conversation.ConversationStorageError()
        head = await original_load_head(active, head_id, authority_scope)
        if active is store and recovery == "mismatch":
            return replace(
                head,
                checkpoint_id=compacted.handle.parent_checkpoint_id,
            )
        return head

    monkeypatch.setattr(
        conversation.InMemoryConversationStore,
        "create_with_named_head",
        create_then_lose_response,
    )
    monkeypatch.setattr(
        conversation.InMemoryConversationStore,
        "load_head",
        recovery_head,
    )
    if recovery == "success":
        committed = await client.commit_compact(compacted)
        head_id = compacted.handle.head_id
        assert head_id is not None
        assert (await store.load_head(head_id, scope)).checkpoint_id == (
            committed.checkpoint_id
        )
    else:
        with pytest.raises(conversation.ConversationConflictError):
            await client.commit_compact(compacted)
    await coordinator.close()


async def test_in_memory_compact_head_commit_rejects_invalid_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed compact candidates before mutating head state."""
    client, store, coordinator, scope, compacted = await _prepared_compaction(
        "compact-store-head-validation",
        named_head=True,
    )
    source = await store.load(compacted.handle.checkpoint_id, scope)
    original_create = (
        conversation.InMemoryConversationStore.create_with_named_head
    )
    captured: list[
        tuple[
            conversation.CheckpointCandidate,
            conversation.NamedHeadAdvance,
        ]
    ] = []

    async def capture_candidate(
        active: conversation.InMemoryConversationStore,
        candidate: conversation.CheckpointCandidate,
        advance: conversation.NamedHeadAdvance,
    ) -> conversation.ConversationCheckpoint:
        if active is store:
            captured.append((candidate, advance))
            raise conversation.ConversationConflictError()
        return await original_create(active, candidate, advance)

    monkeypatch.setattr(
        conversation.InMemoryConversationStore,
        "create_with_named_head",
        capture_candidate,
    )
    with pytest.raises(conversation.ConversationConflictError):
        await client.commit_compact(compacted)
    assert len(captured) == 1
    candidate, advance = captured[0]
    assert type(candidate) is conversation.ExecutionSegmentCheckpointCandidate
    monkeypatch.setattr(
        conversation.InMemoryConversationStore,
        "create_with_named_head",
        original_create,
    )

    with pytest.raises(conversation.ConversationValidationError):
        await original_create(
            store,
            candidate,
            cast(conversation.NamedHeadAdvance, object()),
        )

    missing_head = conversation.ExecutionSegmentCheckpointCandidate(
        checkpoint=conversation.with_checkpoint_integrity(
            replace(candidate.checkpoint, head=None, integrity=None)
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        await original_create(store, missing_head, advance)

    root_id = source.identity.parent_checkpoint_id
    root_sequence = source.identity.parent_sequence
    assert root_id is not None and root_sequence is not None
    wrong_parent_identity = replace(
        candidate.checkpoint.identity,
        parent_checkpoint_id=root_id,
        parent_sequence=root_sequence,
        sequence=conversation.CheckpointSequence(root_sequence + 1),
    )
    ordinary_parent = conversation.ExecutionSegmentCheckpointCandidate(
        checkpoint=conversation.with_checkpoint_integrity(
            replace(
                candidate.checkpoint,
                identity=wrong_parent_identity,
                integrity=None,
            )
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        await original_create(
            store,
            ordinary_parent,
            replace(
                advance,
                parent_checkpoint_id=conversation.CheckpointId(
                    "unrelated-compact-grandparent"
                ),
            ),
        )
    await coordinator.close()


async def test_standalone_state_requires_explicit_commit_or_fork(
    record_property: Callable[[str, object], None],
) -> None:
    """Keep provider-private output non-parental until explicitly committed."""
    record_property("conversation_acceptance_evidence", "public")
    lane = binding("lane-compact-explicit")
    first_plan = empty_stateless_plan(lane)
    first_result = conversation.fake_provider_result(first_plan, turn=1)
    compact_result = conversation.fake_compaction_result(first_plan, turn=2)
    client, store, coordinator, scope = _client((first_result, compact_result))
    first = await client.create(
        "first",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    parent = avalan.StatelessParent(handle=first.handle)
    compacted = await client.compact(
        avalan.StandaloneCompactRequest(parent=parent)
    )
    assert type(compacted.handle) is avalan.StandaloneCompactHandle
    assert repr(compacted.handle) == "StandaloneCompactHandle(private=True)"
    assert "synthetic-compaction" not in repr(compacted)
    with pytest.raises(conversation.ConversationValidationError):
        await client.commit_compact(
            cast(avalan.StandaloneCompactResult, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        await client.fork_compact(
            cast(avalan.StandaloneCompactResult, object()),
            conversation.ConversationBranchId("invalid-compact-fork"),
        )
    with pytest.raises(conversation.ConversationValidationError):
        await client.fork_compact(compacted, compacted.handle.branch_id)
    with pytest.raises(conversation.ConversationValidationError):
        await client._commit_compact_result(
            cast(avalan.StandaloneCompactResult, object()),
            branch_id=conversation.ConversationBranchId(
                "invalid-private-compact"
            ),
            operation="invalid-private-compact",
            idempotency_key=None,
        )

    forged_parent = avalan.StatelessParent(
        handle=avalan.StatelessConversationHandle(
            conversation_id=compacted.handle.conversation_id,
            checkpoint_id=compacted.handle.checkpoint_id,
            branch_id=compacted.handle.branch_id,
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        await client.continue_conversation(
            "forged",
            avalan.StatelessConversationSettings(parent=forged_parent),
        )

    committed = await client.commit_compact(compacted)
    forked = await client.fork_compact(
        compacted,
        conversation.ConversationBranchId("compact-compact-fork"),
    )
    assert committed.branch_id == first.handle.branch_id
    assert forked.branch_id == "compact-compact-fork"
    assert (await store.load(committed.checkpoint_id, scope)).content == (
        await store.load(forked.checkpoint_id, scope)
    ).content
    original = await store.load(first.handle.checkpoint_id, scope)
    assert original.kind is conversation.CheckpointKind.COMPLETED_OUTWARD_TURN
    await coordinator.close()


async def test_named_head_compact_and_continue_race_has_one_cas_winner(
    record_property: Callable[[str, object], None],
) -> None:
    """Commit exactly one child when compact races ordinary continuation."""
    record_property("conversation_acceptance_evidence", "runtime")
    lane = binding("lane-compact-explicit")
    root_plan = empty_stateless_plan(lane)
    root_result = conversation.fake_provider_result(root_plan, turn=1)
    parent_plan = conversation.StatelessProviderPlan(
        binding=lane,
        ledger=_ledger(root_result.items, lane.lane_id),
        reasoning=root_result.reasoning,
    )
    client, store, coordinator, scope = _client(
        (
            root_result,
            conversation.fake_compaction_result(root_plan, turn=2),
            conversation.fake_provider_result(parent_plan, turn=3),
        ),
        namespace="compact-head-continue-race",
    )
    root = await client.create("root", avalan.StatelessConversationSettings())
    assert type(root.handle) is avalan.StatelessConversationHandle
    original = await store.load(root.handle.checkpoint_id, scope)
    head_id = conversation.NamedHeadId("compact-continue-head")
    await store.create_head(
        conversation.NamedHeadSnapshot(
            head_id=head_id,
            revision=conversation.NamedHeadRevision(0),
            checkpoint_id=root.handle.checkpoint_id,
        ),
        scope,
    )
    parent = avalan.StatelessParent(handle=root.handle)
    head_parent = avalan.NamedHeadParent(
        head_id=head_id,
        expected_revision=conversation.NamedHeadRevision(0),
        parent=parent,
    )
    compacted = await client.compact(
        avalan.StandaloneCompactRequest(
            parent=parent,
            named_head=head_parent,
        )
    )
    assert compacted.handle.head_id == head_id
    assert compacted.handle.expected_head_revision == 0
    assert await store.load_head(head_id, scope) == (
        conversation.NamedHeadSnapshot(
            head_id=head_id,
            revision=conversation.NamedHeadRevision(0),
            checkpoint_id=root.handle.checkpoint_id,
        )
    )

    outcomes = await gather(
        client.commit_compact(compacted),
        client.continue_conversation(
            "contender",
            avalan.StatelessConversationSettings(
                parent=parent,
                named_head=head_parent,
            ),
        ),
        return_exceptions=True,
    )
    successes = tuple(
        result for result in outcomes if not isinstance(result, BaseException)
    )
    conflicts = tuple(
        result
        for result in outcomes
        if type(result) is conversation.ConversationConflictError
    )
    assert len(successes) == 1
    assert len(conflicts) == 1
    winner = successes[0]
    winner_handle = (
        winner.handle
        if type(winner) is avalan.DirectConversationResult
        else winner
    )
    assert type(winner_handle) is avalan.StatelessConversationHandle
    advanced = await store.load_head(head_id, scope)
    assert advanced.revision == conversation.NamedHeadRevision(1)
    assert advanced.checkpoint_id == winner_handle.checkpoint_id
    assert await store.load(root.handle.checkpoint_id, scope) == original
    assert store.diagnostics.checkpoints == 3
    await coordinator.close()


async def test_two_named_head_compactions_have_one_cas_winner(
    record_property: Callable[[str, object], None],
) -> None:
    """Commit exactly one of two compact results from one head revision."""
    record_property("conversation_acceptance_evidence", "runtime")
    lane = binding("lane-compact-explicit")
    root_plan = empty_stateless_plan(lane)
    root_result = conversation.fake_provider_result(root_plan, turn=1)
    client, store, coordinator, scope = _client(
        (
            root_result,
            conversation.fake_compaction_result(root_plan, turn=2),
            conversation.fake_compaction_result(root_plan, turn=3),
        ),
        namespace="compact-head-compact-race",
    )
    root = await client.create("root", avalan.StatelessConversationSettings())
    assert type(root.handle) is avalan.StatelessConversationHandle
    original = await store.load(root.handle.checkpoint_id, scope)
    head_id = conversation.NamedHeadId("two-compact-head")
    await store.create_head(
        conversation.NamedHeadSnapshot(
            head_id=head_id,
            revision=conversation.NamedHeadRevision(0),
            checkpoint_id=root.handle.checkpoint_id,
        ),
        scope,
    )
    parent = avalan.StatelessParent(handle=root.handle)
    head_parent = avalan.NamedHeadParent(
        head_id=head_id,
        expected_revision=conversation.NamedHeadRevision(0),
        parent=parent,
    )
    first = await client.compact(
        avalan.StandaloneCompactRequest(
            parent=parent,
            named_head=head_parent,
        )
    )
    second = await client.compact(
        avalan.StandaloneCompactRequest(
            parent=parent,
            named_head=head_parent,
        )
    )
    unchanged = await store.load_head(head_id, scope)
    assert unchanged.revision == conversation.NamedHeadRevision(0)
    assert unchanged.checkpoint_id == root.handle.checkpoint_id

    outcomes = await gather(
        client.commit_compact(first),
        client.commit_compact(second),
        return_exceptions=True,
    )
    winners = tuple(
        result for result in outcomes if not isinstance(result, BaseException)
    )
    conflicts = tuple(
        result
        for result in outcomes
        if type(result) is conversation.ConversationConflictError
    )
    assert len(winners) == 1
    assert len(conflicts) == 1
    winner = winners[0]
    assert type(winner) is avalan.StatelessConversationHandle
    advanced = await store.load_head(head_id, scope)
    assert advanced.revision == conversation.NamedHeadRevision(1)
    assert advanced.checkpoint_id == winner.checkpoint_id
    assert await store.load(root.handle.checkpoint_id, scope) == original
    assert store.diagnostics.checkpoints == 4
    await coordinator.close()


async def test_direct_stream_and_nonstream_compaction_commit_identically(
    record_property: Callable[[str, object], None],
) -> None:
    """Reload equal checkpoints after stream and non-stream commits."""
    record_property("conversation_acceptance_evidence", "runtime")
    lane = binding("lane-direct-compact-parity", streaming=True)
    compact_item = _compaction(lane.lane_id, 0, "parity-boundary")
    message = _message(lane.lane_id, 1, "parity-visible")
    provider_result = conversation.ProviderResult(
        items=(compact_item, message),
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.AUTO,
            effective=conversation.EffectiveReasoningContext.CURRENT_TURN,
        ),
        usage=conversation.ProviderUsage(input_tokens=21, output_tokens=8),
    )
    nonstream_client, nonstream_store, nonstream_coordinator, scope = _client(
        (provider_result,),
        lane_binding=lane,
        namespace="direct-compact-parity",
    )
    stream_client, stream_store, stream_coordinator, _ = _client(
        (provider_result,),
        lane_binding=lane,
        namespace="direct-compact-parity",
    )
    settings = avalan.StatelessConversationSettings(
        compaction=avalan.InlineCompaction(compact_threshold=128)
    )

    nonstream = await nonstream_client.create("same input", settings)
    stream = await stream_client.create("same input", settings, stream=True)
    assert type(stream) is avalan.DirectConversationStream
    events = tuple([event async for event in stream])
    assert tuple(type(event) for event in events) == (
        avalan.DirectConversationOutputDelta,
        avalan.DirectConversationStreamTerminal,
    )
    terminal = events[-1]
    assert type(terminal) is avalan.DirectConversationStreamTerminal
    assert terminal.result == nonstream

    nonstream_checkpoint = await nonstream_store.load(
        nonstream.handle.checkpoint_id,
        scope,
    )
    stream_checkpoint = await stream_store.load(
        terminal.result.handle.checkpoint_id,
        scope,
    )
    assert stream_checkpoint == nonstream_checkpoint
    lane_snapshot = stream_checkpoint.content.lanes[0]
    assert type(lane_snapshot) is conversation.StatelessProviderLaneSnapshot
    assert lane_snapshot.ledger.items == (compact_item, message)
    assert lane_snapshot.compaction_boundary is not None
    assert lane_snapshot.compaction_boundary.boundary_item_id == (
        compact_item.item_id
    )
    assert stream.state is avalan.DirectConversationStreamState.COMMITTED
    await nonstream_coordinator.close()
    await stream_coordinator.close()


def test_compaction_execution_closure_has_no_local_substitute_calls(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject local summary, clipping, deletion, or reconstruction calls."""
    record_property("conversation_acceptance_evidence", "negative")
    _verify_latest_boundary_and_tool_adjacency()
    assert (
        openai_stored_provider_module.NativeOpenAIStoredExecution.__dataclass_fields__[
            "truncation"
        ].default
        == "disabled"
    )
    targets: tuple[Callable[..., object], ...] = (
        conversation.provider_replay_items,
        openai_provider_module._request_input_items,
        openai_provider_module._compact_provider_result,
        openai_provider_module.NativeOpenAIStatelessProvider.dispatch,
        openai_provider_module.NativeOpenAIStatelessProvider.compact,
        openai_provider_module.NativeOpenAIStatelessProvider.stream,
        (
            openai_provider_module.NativeOpenAIStatelessProvider.validate_compaction_request
        ),
        openai_stored_provider_module.NativeOpenAIStoredProvider.dispatch,
        openai_stored_provider_module.NativeOpenAIStoredProvider.stream,
        (
            openai_stored_provider_module.NativeOpenAIStoredProvider.validate_compaction_request
        ),
        coordinator_module.RunScopedConversationCoordinator._plan_lanes,
        (
            coordinator_module.RunScopedConversationCoordinator._validate_compact_outputs
        ),
        sdk_module.DirectConversationClient.compact,
    )
    forbidden_stems = (
        "summar",
        "truncat",
        "clip",
        "prune",
        "delete",
        "reconstruct",
    )
    violations: list[str] = []
    for target in targets:
        tree = parse(dedent(getsource(target)))
        for node in walk(tree):
            if type(node) is not Call:
                continue
            function = node.func
            name = (
                function.id
                if type(function) is Name
                else (function.attr if type(function) is Attribute else "")
            )
            if any(stem in name.lower() for stem in forbidden_stems):
                violations.append(f"{target.__qualname__}:{name}")
    assert violations == []


async def test_compaction_private_canary_is_absent_from_active_public_surfaces(
    caplog: pytest.LogCaptureFixture,
    record_property: Callable[[str, object], None],
) -> None:
    """Keep compact state out of presentation, telemetry, and failures."""
    record_property("conversation_acceptance_evidence", "security")
    canary = "compact-private-canary-6f859a"
    caplog.set_level(1)
    lane = binding("lane-compact-private-surfaces", streaming=True)
    compact_item = _compaction(lane.lane_id, 0, canary)
    message = _message(lane.lane_id, 1, "public-safe-message")
    inline_result = conversation.ProviderResult(
        items=(compact_item, message),
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.AUTO,
            effective=conversation.EffectiveReasoningContext.CURRENT_TURN,
        ),
        usage=conversation.ProviderUsage(input_tokens=18, output_tokens=4),
    )
    parent_plan = conversation.StatelessProviderPlan(
        binding=lane,
        ledger=_ledger(inline_result.items, lane.lane_id),
        reasoning=inline_result.reasoning,
    )
    standalone_result = conversation.fake_compaction_result(
        parent_plan,
        turn=2,
        opaque_state=canary.encode(),
    )
    observer = conversation.DeterministicFakeObserver()
    publisher = conversation.DeterministicFakePublisher()
    client, store, coordinator, scope = _client(
        (inline_result, standalone_result),
        lane_binding=lane,
        namespace="compact-private-surfaces",
        observer=observer,
        publisher=publisher,
    )
    stream = await client.create(
        "public-safe-input",
        avalan.StatelessConversationSettings(
            compaction=avalan.InlineCompaction(compact_threshold=128)
        ),
        stream=True,
    )
    assert type(stream) is avalan.DirectConversationStream
    events = tuple([event async for event in stream])
    terminal = events[-1]
    assert type(terminal) is avalan.DirectConversationStreamTerminal
    parent_handle = terminal.result.handle
    assert type(parent_handle) is avalan.StatelessConversationHandle
    checkpoint = await store.load(
        parent_handle.checkpoint_id,
        scope,
    )
    lane_snapshot = checkpoint.content.lanes[0]
    assert type(lane_snapshot) is conversation.StatelessProviderLaneSnapshot
    assert lane_snapshot.ledger.items[0] == compact_item
    assert compact_item.opaque_state is not None
    assert canary.encode() in compact_item.opaque_state._codec_bytes()
    assert items_module.public_provider_item_projection((compact_item,)) == ()

    compacted = await client.compact(
        avalan.StandaloneCompactRequest(
            parent=avalan.StatelessParent(handle=parent_handle)
        )
    )
    assert compacted.canonical_context.items[0].opaque_state is not None
    assert canary.encode() in (
        compacted.canonical_context.items[0].opaque_state._codec_bytes()
    )
    forged = replace(
        compacted,
        canonical_context_digest=conversation.IntegrityDigest("0" * 64),
    )
    with pytest.raises(conversation.ConversationValidationError) as error:
        await client.commit_compact(forged)
    with pytest.raises(
        conversation.ConversationValidationError
    ) as presentation_error:
        items_module.public_provider_item_projection(
            cast(tuple[conversation.ProviderItem, ...], (compacted,))
        )

    diagnostics = coordinator.fake_provider_diagnostics(lane.lane_id)
    surfaces = (
        events,
        stream,
        stream.terminal,
        terminal.result,
        checkpoint,
        checkpoint.content.visible_transcript,
        store.diagnostics,
        coordinator.diagnostics,
        diagnostics,
        observer.observations,
        tuple(item.to_mapping() for item in observer.observations),
        publisher.published,
        compacted,
        compacted.handle,
        error.value,
        presentation_error.value,
        items_module.public_provider_item_projection(
            compacted.canonical_context.items
        ),
    )
    for surface in surfaces:
        assert canary not in repr(surface)
        assert canary not in str(surface)
    assert canary not in caplog.text
    for record in caplog.records:
        assert canary not in record.getMessage()
        assert canary not in repr(record.args)
        assert canary not in repr(record.exc_info)
    for root_error in (error.value, presentation_error.value):
        pending: list[BaseException] = [root_error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            assert canary not in str(current)
            assert canary not in repr(current)
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
    assert tuple(
        event.text_delta
        for event in events
        if type(event) is avalan.DirectConversationOutputDelta
    ) == ("public-safe-message",)
    await coordinator.close()


async def _verify_matrix_validation_failure_is_predispatch() -> None:
    """Prove inline input-limit validation is predispatch and parent-safe."""
    private_canary = "private-validation-matrix"
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        return httpx.Response(
            200,
            json=_native_response(
                f"matrix-validation-response-{dispatches}",
                [
                    _native_message(
                        f"matrix-validation-message-{dispatches}",
                        f"matrix validation {dispatches}",
                    )
                ],
            ),
        )

    limits = replace(_native_limits(), max_input_items=1)
    scope = authority()
    lane = replace(
        _native_binding_with_limits(
            "lane-matrix-validation",
            limits,
        ),
        agent_id=scope.agent_id,
    )
    provider = _native_stateless_provider(lane, handler, limits=limits)
    client, store, coordinator = _native_client(
        scope=scope,
        lane=lane,
        provider=provider,
        namespace="matrix-validation",
    )
    root = await client.create(
        "root",
        avalan.StatelessConversationSettings(),
    )
    assert type(root.handle) is avalan.StatelessConversationHandle
    parent = avalan.StatelessParent(handle=root.handle)
    original = await store.load(root.handle.checkpoint_id, scope)
    original_bytes = _checkpoint_bytes(original)
    before = store.diagnostics
    assert dispatches == 1

    with pytest.raises(conversation.ConversationLimitError):
        await client.continue_conversation(
            private_canary,
            avalan.StatelessConversationSettings(
                parent=parent,
                compaction=avalan.InlineCompaction(compact_threshold=128),
            ),
        )
    assert dispatches == 1
    assert provider.diagnostics.request_count == 1
    assert store.diagnostics.checkpoints == before.checkpoints
    assert store.diagnostics.idempotency_records == before.idempotency_records
    assert (
        _checkpoint_bytes(await store.load(root.handle.checkpoint_id, scope))
        == original_bytes
    )
    _assert_compaction_failure(
        coordinator,
        operation=conversation.CompactionOperation.INLINE,
        boundary=conversation.FailureBoundary.VALIDATION_BEFORE_DISPATCH,
        error_code=conversation.ConversationErrorCode.LIMIT_EXCEEDED,
        cancelled=False,
        committed=False,
        streaming=False,
        private_canary=private_canary,
    )

    reused = await client.continue_conversation(
        "reused validation parent",
        avalan.StatelessConversationSettings(parent=parent),
    )
    assert type(reused.handle) is avalan.StatelessConversationHandle
    assert dispatches == 2
    assert reused.handle.checkpoint_id != root.handle.checkpoint_id
    assert (
        _checkpoint_bytes(await store.load(root.handle.checkpoint_id, scope))
        == original_bytes
    )
    await coordinator.close()


async def _verify_matrix_malformed_stream_failure_is_atomic() -> None:
    """Prove malformed streamed compaction records no partial child."""
    private_canary = "private-malformed-stream-matrix"
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        if dispatches == 2:
            return httpx.Response(
                200,
                text=_sse_body(
                    (
                        {
                            "type": "response.output_item.done",
                            "sequence_number": 0,
                            "output_index": 0,
                            "item": {
                                "created_by": f"{private_canary} invalid",
                                "encrypted_content": private_canary,
                                "id": "matrix-malformed-boundary",
                                "type": "compaction",
                            },
                        },
                    )
                ),
                headers={"content-type": "text/event-stream"},
            )
        output = [
            _native_message(
                f"matrix-stream-message-{dispatches}",
                f"matrix stream {dispatches}",
            )
        ]
        response = _native_response(
            f"matrix-stream-response-{dispatches}",
            output,
        )
        return httpx.Response(
            200,
            text=_sse_body(
                (
                    {
                        "type": "response.output_item.done",
                        "sequence_number": 0,
                        "output_index": 0,
                        "item": output[0],
                    },
                    {
                        "type": "response.completed",
                        "sequence_number": 1,
                        "response": response,
                    },
                )
            ),
            headers={"content-type": "text/event-stream"},
        )

    scope = authority()
    lane = replace(
        _native_binding_with_limits(
            "lane-matrix-malformed-stream",
            _native_limits(),
            streaming=True,
        ),
        agent_id=scope.agent_id,
    )
    provider = _native_stateless_provider(lane, handler)
    client, store, coordinator = _native_client(
        scope=scope,
        lane=lane,
        provider=provider,
        namespace="matrix-malformed-stream",
    )
    root_stream = await client.create(
        "root",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    root = await _committed_stream_result(root_stream)
    assert type(root.handle) is avalan.StatelessConversationHandle
    parent = avalan.StatelessParent(handle=root.handle)
    original_bytes = _checkpoint_bytes(
        await store.load(root.handle.checkpoint_id, scope)
    )
    before = store.diagnostics

    failed_stream = await client.continue_conversation(
        "malformed stream",
        avalan.StatelessConversationSettings(
            parent=parent,
            compaction=avalan.InlineCompaction(compact_threshold=128),
        ),
        stream=True,
    )
    with pytest.raises(conversation.ConversationError) as malformed_error:
        tuple([event async for event in failed_stream])
    assert (
        malformed_error.value.boundary
        is conversation.FailureBoundary.MALFORMED_STREAM_ITEM
    )
    assert dispatches == 2
    assert store.diagnostics.checkpoints == before.checkpoints
    assert store.diagnostics.idempotency_records == (
        before.idempotency_records + 1
    )
    assert (
        _checkpoint_bytes(await store.load(root.handle.checkpoint_id, scope))
        == original_bytes
    )
    _assert_compaction_failure(
        coordinator,
        operation=conversation.CompactionOperation.INLINE,
        boundary=conversation.FailureBoundary.MALFORMED_STREAM_ITEM,
        error_code=conversation.ConversationErrorCode.VALIDATION_FAILED,
        cancelled=False,
        committed=False,
        streaming=True,
        private_canary=private_canary,
    )

    reused_stream = await client.continue_conversation(
        "reused malformed-stream parent",
        avalan.StatelessConversationSettings(parent=parent),
        stream=True,
    )
    reused = await _committed_stream_result(reused_stream)
    assert type(reused.handle) is avalan.StatelessConversationHandle
    assert dispatches == 3
    assert reused.handle.checkpoint_id != root.handle.checkpoint_id
    assert (
        _checkpoint_bytes(await store.load(root.handle.checkpoint_id, scope))
        == original_bytes
    )
    await coordinator.close()


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("private-compact-commit-failure"),
        CancelledError(),
        conversation.ConversationValidationError(),
    ),
    ids=("failure", "cancellation", "domain-failure"),
)
async def _verify_explicit_compact_commit_failure_is_retryable_and_atomic(
    failure: BaseException,
    record_property: Callable[[str, object], None],
) -> None:
    """Leave no partial child and recover with the same explicit key."""
    record_property("conversation_acceptance_evidence", "negative")
    lane = binding("lane-compact-explicit")
    plan = empty_stateless_plan(lane)
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:create",
                exception=failure,
            ),
        )
    )
    store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(controller)
    )
    client, _, coordinator, scope = _client(
        (
            conversation.fake_provider_result(plan, turn=1),
            conversation.fake_compaction_result(plan, turn=2),
        ),
        store=store,
        namespace=f"compact-commit-{type(failure).__name__}",
    )
    first = await client.create(
        "parent", avalan.StatelessConversationSettings()
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    original_bytes = _checkpoint_bytes(
        await store.load(first.handle.checkpoint_id, scope)
    )
    compacted = await client.compact(
        avalan.StandaloneCompactRequest(
            parent=avalan.StatelessParent(handle=first.handle)
        )
    )
    compacted_bytes = _checkpoint_bytes(
        await store.load(compacted.handle.checkpoint_id, scope)
    )
    key = conversation.RequestIdempotencyKey("compact-explicit-commit-key")
    expected_error = (
        CancelledError
        if isinstance(failure, CancelledError)
        else (
            conversation.ConversationValidationError
            if isinstance(failure, conversation.ConversationValidationError)
            else conversation.ConversationCommitError
        )
    )
    with pytest.raises(expected_error):
        await client.commit_compact(compacted, idempotency_key=key)
    error_code: conversation.ConversationErrorCode | str
    if isinstance(failure, CancelledError):
        error_code = "conversation_cancelled"
    elif isinstance(failure, conversation.ConversationError):
        error_code = failure.code
    else:
        error_code = conversation.ConversationErrorCode.COMMIT_FAILED
    _assert_compaction_failure(
        coordinator,
        operation=conversation.CompactionOperation.STANDALONE,
        boundary=conversation.FailureBoundary.CHECKPOINT_COMMIT,
        error_code=error_code,
        cancelled=isinstance(failure, CancelledError),
        committed=False,
        streaming=False,
        private_canary="private-compact-commit-failure",
    )
    assert store.diagnostics.checkpoints == 2
    assert (
        _checkpoint_bytes(await store.load(first.handle.checkpoint_id, scope))
        == original_bytes
    )
    assert (
        _checkpoint_bytes(
            await store.load(compacted.handle.checkpoint_id, scope)
        )
        == compacted_bytes
    )

    committed = await client.commit_compact(compacted, idempotency_key=key)
    assert store.diagnostics.checkpoints == 3
    assert (
        await store.load(committed.checkpoint_id, scope)
    ).kind is conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
    assert (
        _checkpoint_bytes(await store.load(first.handle.checkpoint_id, scope))
        == original_bytes
    )
    assert (
        _checkpoint_bytes(
            await store.load(compacted.handle.checkpoint_id, scope)
        )
        == compacted_bytes
    )
    await coordinator.close()


async def test_explicit_compact_rejects_forged_state_and_receipts() -> None:
    """Reject every compact-result field bound by retained provider state."""
    client, _, coordinator, _, compacted = await _prepared_compaction(
        "compact-forged-state"
    )
    forged_digest = replace(
        compacted,
        canonical_context_digest=conversation.IntegrityDigest("0" * 64),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await client.commit_compact(forged_digest)

    forged_binding = replace(
        compacted,
        binding=replace(
            compacted.binding,
            model_configuration_revision=(
                conversation.ModelConfigurationRevision("forged-config")
            ),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await client.commit_compact(forged_binding)

    forged_usage = replace(
        compacted,
        usage=conversation.ProviderUsage(
            input_tokens=compacted.usage.input_tokens + 1,
            output_tokens=compacted.usage.output_tokens,
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await client.commit_compact(forged_usage)
    await coordinator.close()


@pytest.mark.parametrize("corruption", ("count", "lane"))
async def test_direct_compact_rejects_corrupted_coordinator_receipt(
    corruption: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject impossible coordinator receipts at the direct SDK boundary."""
    lane = binding("lane-compact-explicit")
    plan = empty_stateless_plan(lane)
    client, _, coordinator, _ = _client(
        (
            conversation.fake_provider_result(plan, turn=1),
            conversation.fake_compaction_result(plan, turn=2),
        ),
        namespace=f"compact-corrupt-receipt-{corruption}",
    )
    first = await client.create(
        "receipt parent", avalan.StatelessConversationSettings()
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    original_compact = conversation.RunScopedConversationCoordinator.compact

    async def corrupted_compact(
        active: conversation.RunScopedConversationCoordinator,
        request: conversation.ConversationRunRequest,
    ) -> conversation.AtomicCommitReceipt:
        receipt = await original_compact(active, request)
        if corruption == "count":
            object.__setattr__(receipt, "output_candidates", ())
        else:
            object.__setattr__(
                receipt.output_candidates[0],
                "reasoning",
                conversation.EffectiveReasoningMetadata(
                    requested=conversation.ReasoningContext.CURRENT_TURN,
                    effective=(
                        conversation.EffectiveReasoningContext.CURRENT_TURN
                    ),
                ),
            )
        return receipt

    monkeypatch.setattr(
        conversation.RunScopedConversationCoordinator,
        "compact",
        corrupted_compact,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await client.compact(
            avalan.StandaloneCompactRequest(
                parent=avalan.StatelessParent(handle=first.handle)
            )
        )
    await coordinator.close()


async def test_explicit_compact_rejects_invalid_and_expired_source() -> None:
    """Reject malformed explicit identity and expired private compact state."""
    now = datetime(2026, 8, 2, tzinfo=UTC)
    clock = conversation.DeterministicFakeClock(now)
    client, store, coordinator, scope, compacted = await _prepared_compaction(
        "compact-expired-source",
        clock=clock,
        retention_limits=retention(ttl=1),
    )
    source = await store.load(compacted.handle.checkpoint_id, scope)
    with pytest.raises(conversation.ConversationValidationError):
        await coordinator.commit_compact_result(
            source,
            cast(conversation.CheckpointIdentity, object()),
            scope,
        )
    clock.set(now + timedelta(seconds=2))
    with pytest.raises(conversation.ConversationTransitionError):
        await client.commit_compact(compacted)
    await coordinator.close()


@pytest.mark.parametrize("failure", ("conflict", "generic"))
async def test_compact_commit_recovers_exact_lost_success(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover only an exact committed child after a lost create response."""
    client, store, coordinator, scope, compacted = await _prepared_compaction(
        f"compact-lost-success-{failure}"
    )
    original_create = conversation.InMemoryConversationStore.create

    async def create_then_fail(
        active: conversation.InMemoryConversationStore,
        candidate: conversation.CheckpointCandidate,
    ) -> conversation.ConversationCheckpoint:
        committed = await original_create(active, candidate)
        if active is store:
            if failure == "conflict":
                raise conversation.ConversationConflictError()
            raise RuntimeError("private lost compact create response")
        return committed

    monkeypatch.setattr(
        conversation.InMemoryConversationStore,
        "create",
        create_then_fail,
    )
    committed = await client.commit_compact(compacted)
    assert (
        await store.load(committed.checkpoint_id, scope)
    ).kind is conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
    await coordinator.close()


async def test_compact_commit_rejects_nonmatching_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a conflict whose colliding child is not the exact candidate."""
    client, store, coordinator, _, compacted = await _prepared_compaction(
        "compact-mismatched-collision"
    )
    original_create = conversation.InMemoryConversationStore.create
    original_load = conversation.InMemoryConversationStore.load

    async def conflicting_create(
        active: conversation.InMemoryConversationStore,
        candidate: conversation.CheckpointCandidate,
    ) -> conversation.ConversationCheckpoint:
        if active is store:
            raise conversation.ConversationConflictError()
        return await original_create(active, candidate)

    async def colliding_load(
        active: conversation.InMemoryConversationStore,
        checkpoint_id: conversation.CheckpointId,
        scope: conversation.AuthorityScope,
    ) -> conversation.ConversationCheckpoint:
        if active is store and checkpoint_id != compacted.handle.checkpoint_id:
            return await original_load(
                active,
                compacted.handle.checkpoint_id,
                scope,
            )
        return await original_load(active, checkpoint_id, scope)

    monkeypatch.setattr(
        conversation.InMemoryConversationStore,
        "create",
        conflicting_create,
    )
    monkeypatch.setattr(
        conversation.InMemoryConversationStore,
        "load",
        colliding_load,
    )
    with pytest.raises(conversation.ConversationConflictError):
        await client.commit_compact(compacted)
    await coordinator.close()


async def test_streamed_standalone_dispatch_is_rejected_internally() -> None:
    """Reject a streamed standalone plan at the coordinator dispatch edge."""
    lane = binding("lane-compact-explicit")
    root_plan = empty_stateless_plan(lane)
    root_result = conversation.fake_provider_result(root_plan, turn=1)
    _, _, coordinator, _ = _client(
        (root_result,),
        namespace="compact-streamed-internal",
    )
    runtime = cast(
        conversation.ConversationLaneRuntime,
        coordinator._lanes[lane.lane_id],
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await coordinator._dispatch_complete_lane(
            runtime,
            conversation.StandaloneCompactProviderPlan(
                binding=lane,
                ledger=_ledger(root_result.items, lane.lane_id),
                reasoning=root_result.reasoning,
            ),
            streaming=True,
            progress=coordinator_module._DispatchProgress(),
            sink=None,
        )
    await coordinator.close()


@pytest.mark.parametrize("malformation", ("shape", "ledger"))
async def _verify_standalone_provider_result_validation_is_exact(
    malformation: str,
    record_property: Callable[[str, object], None],
) -> None:
    """Reject malformed standalone shape and canonical ledger sequence."""
    record_property("conversation_acceptance_evidence", "negative")
    lane = binding("lane-compact-explicit")
    root_plan = empty_stateless_plan(lane)
    root_result = conversation.fake_provider_result(root_plan, turn=1)
    if malformation == "shape":
        malformed = conversation.ProviderResult(
            items=(_message(lane.lane_id, 0, "not-compaction"),),
            reasoning=root_result.reasoning,
        )
    else:
        malformed = conversation.ProviderResult(
            items=(
                _caller_message(lane.lane_id, 0, "compact-input"),
                _compaction(lane.lane_id, 0, "invalid-order-boundary"),
            ),
            reasoning=root_result.reasoning,
        )
    client, _, coordinator, _ = _client(
        (root_result, malformed),
        namespace=f"compact-malformed-provider-{malformation}",
    )
    first = await client.create(
        "malformed parent", avalan.StatelessConversationSettings()
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    with pytest.raises(conversation.ConversationProviderResponseError):
        await client.compact(
            avalan.StandaloneCompactRequest(
                parent=avalan.StatelessParent(handle=first.handle)
            )
        )
    failure_record = coordinator.diagnostics.compaction_failures[-1]
    assert (
        failure_record.operation is conversation.CompactionOperation.STANDALONE
    )
    assert (
        failure_record.boundary
        is conversation.FailureBoundary.FAILURE_BEFORE_OUTPUT
    )
    assert failure_record.cancelled is False
    assert failure_record.committed is False
    await coordinator.close()


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("private-provider-compact-failure"),
        CancelledError(),
    ),
    ids=("failure", "cancellation"),
)
async def _verify_compact_provider_failure_leaves_parent_reusable(
    failure: BaseException,
    record_property: Callable[[str, object], None],
) -> None:
    """Keep the original parent usable after failed compact dispatch."""
    record_property("conversation_acceptance_evidence", "negative")
    lane = binding("lane-compact-explicit")
    root_plan = empty_stateless_plan(lane)
    root_result = conversation.fake_provider_result(root_plan, turn=1)
    parent_plan = conversation.StatelessProviderPlan(
        binding=lane,
        ledger=_ledger(root_result.items, lane.lane_id),
        reasoning=root_result.reasoning,
    )
    continued_result = conversation.fake_provider_result(parent_plan, turn=2)
    root_client, store, root_coordinator, scope = _client(
        (root_result,),
        namespace="compact-provider-root",
    )
    first = await root_client.create(
        "parent",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    original = await store.load(first.handle.checkpoint_id, scope)
    original_bytes = _checkpoint_bytes(original)
    before = store.diagnostics
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:dispatch",
                exception=failure,
            ),
        )
    )
    client, _, coordinator, _ = _client(
        (continued_result,),
        store=store,
        provider_controller=controller,
        namespace=f"compact-provider-{type(failure).__name__}",
    )
    expected_error = (
        CancelledError if isinstance(failure, CancelledError) else RuntimeError
    )
    with pytest.raises(expected_error):
        await client.compact(
            avalan.StandaloneCompactRequest(
                parent=avalan.StatelessParent(handle=first.handle)
            )
        )
    expected_boundary = (
        failure.boundary
        if isinstance(failure, conversation.ConversationError)
        else conversation.FailureBoundary.FAILURE_BEFORE_OUTPUT
    )
    expected_code: conversation.ConversationErrorCode | str = (
        failure.code
        if isinstance(failure, conversation.ConversationError)
        else (
            "conversation_cancelled"
            if isinstance(failure, CancelledError)
            else "conversation_internal_failure"
        )
    )
    _assert_compaction_failure(
        coordinator,
        operation=conversation.CompactionOperation.STANDALONE,
        boundary=expected_boundary,
        error_code=expected_code,
        cancelled=isinstance(failure, CancelledError),
        committed=False,
        streaming=False,
        private_canary="private-provider-compact-failure",
    )
    assert store.diagnostics.checkpoints == before.checkpoints
    ambiguous = not (
        isinstance(failure, conversation.ConversationError)
        and failure.boundary is conversation.FailureBoundary.PROVIDER_REJECTION
    )
    assert store.diagnostics.idempotency_records == (
        before.idempotency_records + int(ambiguous)
    )
    assert (
        _checkpoint_bytes(await store.load(first.handle.checkpoint_id, scope))
        == original_bytes
    )
    failed_provider = coordinator.fake_provider_diagnostics(lane.lane_id)
    assert len(failed_provider.plans) == 1

    continued = await client.continue_conversation(
        "still usable",
        avalan.StatelessConversationSettings(
            parent=avalan.StatelessParent(handle=first.handle)
        ),
    )
    assert type(continued.handle) is avalan.StatelessConversationHandle
    assert continued.handle.checkpoint_id != first.handle.checkpoint_id
    assert len(coordinator.fake_provider_diagnostics(lane.lane_id).plans) == 2
    assert (
        _checkpoint_bytes(await store.load(first.handle.checkpoint_id, scope))
        == original_bytes
    )
    await coordinator.close()
    await root_coordinator.close()


async def _verify_inline_publication_failure_replays_without_redispatch(
    record_property: Callable[[str, object], None],
) -> None:
    """Publish a committed inline boundary on retry without dispatching."""
    record_property("conversation_acceptance_evidence", "negative")
    lane = binding("lane-compact-explicit")
    root_plan = empty_stateless_plan(lane)
    root_result = conversation.fake_provider_result(root_plan, turn=1)
    inline_result = conversation.ProviderResult(
        items=(
            _compaction(lane.lane_id, 1, "inline-publication-boundary"),
            _message(lane.lane_id, 2, "inline-publication-message"),
        ),
        reasoning=root_result.reasoning,
        usage=conversation.ProviderUsage(input_tokens=20, output_tokens=5),
    )
    root_client, store, root_coordinator, scope = _client(
        (root_result,),
        namespace="inline-publication-root",
    )
    first = await root_client.create(
        "parent",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    original = await store.load(first.handle.checkpoint_id, scope)
    original_bytes = _checkpoint_bytes(original)
    before = store.diagnostics
    publisher_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="publisher:publish",
                exception=RuntimeError("private-inline-publication-failure"),
            ),
        )
    )
    publisher = conversation.DeterministicFakePublisher(publisher_controller)
    client, _, coordinator, _ = _client(
        (inline_result,),
        store=store,
        publisher=publisher,
        namespace="inline-publication",
    )
    settings = avalan.StatelessConversationSettings(
        parent=avalan.StatelessParent(handle=first.handle),
        compaction=avalan.InlineCompaction(compact_threshold=128),
    )
    key = conversation.RequestIdempotencyKey("inline-publication-key")
    with pytest.raises(conversation.ConversationPublicationError):
        await client.continue_conversation(
            "compact inline",
            settings,
            idempotency_key=key,
        )
    _assert_compaction_failure(
        coordinator,
        operation=conversation.CompactionOperation.INLINE,
        boundary=conversation.FailureBoundary.OUTWARD_PUBLICATION,
        error_code=conversation.ConversationErrorCode.PUBLICATION_FAILED,
        cancelled=False,
        committed=True,
        streaming=False,
        private_canary="private-inline-publication-failure",
    )
    after_failure = store.diagnostics
    assert after_failure.checkpoints == before.checkpoints + 1
    assert after_failure.public_responses == before.public_responses + 1
    assert after_failure.outbox_records == before.outbox_records + 1
    assert (
        _checkpoint_bytes(await store.load(first.handle.checkpoint_id, scope))
        == original_bytes
    )

    replay = await client.continue_conversation(
        "compact inline",
        settings,
        idempotency_key=key,
    )
    assert type(replay.handle) is avalan.StatelessConversationHandle
    assert replay.output == "inline-publication-message"
    assert "opaque-inline-publication-boundary" not in repr(replay)
    diagnostics = coordinator.fake_provider_diagnostics(lane.lane_id)
    assert isinstance(
        diagnostics,
        conversation.DeterministicFakeProviderDiagnostics,
    )
    assert len(diagnostics.plans) == 1
    assert len(publisher.published) == 1
    assert store.diagnostics.checkpoints == after_failure.checkpoints
    assert store.diagnostics.public_responses == after_failure.public_responses
    assert store.diagnostics.outbox_records == after_failure.outbox_records
    assert (
        _checkpoint_bytes(await store.load(first.handle.checkpoint_id, scope))
        == original_bytes
    )
    await coordinator.close()
    await root_coordinator.close()


async def _verify_compaction_failure_diagnostics_keep_a_bounded_tail(
    record_property: Callable[[str, object], None],
) -> bool:
    """Retain a fixed failure tail plus one monotonic total count."""
    record_property("conversation_acceptance_evidence", "audit")
    lane = binding("lane-compact-explicit")
    plan = empty_stateless_plan(lane)
    _, _, coordinator, _ = _client(
        (conversation.fake_provider_result(plan, turn=1),),
        namespace="compact-failure-tail",
    )
    private_error = RuntimeError("private-diagnostic-payload")
    coordinator._record_compaction_failure(
        conversation.CompactionOperation.INLINE,
        conversation.FailureBoundary.FAILURE_BEFORE_OUTPUT,
        private_error,
        committed=True,
        streaming=False,
    )
    for _ in range(127):
        coordinator._record_compaction_failure(
            conversation.CompactionOperation.STANDALONE,
            conversation.FailureBoundary.FAILURE_BEFORE_OUTPUT,
            private_error,
            committed=False,
            streaming=False,
        )
    boundary = coordinator.diagnostics
    assert boundary.compaction_failure_count == 128
    assert len(boundary.compaction_failures) == 128
    assert boundary.compaction_failures[0].committed is True

    coordinator._record_compaction_failure(
        conversation.CompactionOperation.STANDALONE,
        conversation.FailureBoundary.FAILURE_BEFORE_OUTPUT,
        private_error,
        committed=False,
        streaming=False,
    )
    overflow = coordinator.diagnostics
    assert overflow.compaction_failure_count == 129
    assert len(overflow.compaction_failures) == 128
    assert all(not item.committed for item in overflow.compaction_failures)
    assert "private-diagnostic-payload" not in repr(overflow)
    await coordinator.close()
    return (
        overflow.compaction_failure_count == 129
        and len(overflow.compaction_failures) == 128
    )


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("private-compact-commit-failure"),
        CancelledError(),
        conversation.ConversationValidationError(),
    ),
    ids=("failure", "cancellation", "domain-failure"),
)
async def test_explicit_compact_commit_failure_is_retryable_and_atomic(
    failure: BaseException,
    record_property: Callable[[str, object], None],
) -> None:
    """Leave no partial child and recover with the same explicit key."""
    await _verify_explicit_compact_commit_failure_is_retryable_and_atomic(
        failure,
        record_property,
    )


@pytest.mark.parametrize("malformation", ("shape", "ledger"))
async def test_standalone_provider_result_validation_is_exact(
    malformation: str,
    record_property: Callable[[str, object], None],
) -> None:
    """Reject malformed standalone shape and canonical ledger sequence."""
    await _verify_standalone_provider_result_validation_is_exact(
        malformation,
        record_property,
    )


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("private-provider-compact-failure"),
        CancelledError(),
    ),
    ids=("failure", "cancellation"),
)
async def test_compact_provider_failure_leaves_parent_reusable(
    failure: BaseException,
    record_property: Callable[[str, object], None],
) -> None:
    """Keep the original parent usable after failed compact dispatch."""
    await _verify_compact_provider_failure_leaves_parent_reusable(
        failure,
        record_property,
    )


async def test_inline_publication_failure_replays_without_redispatch(
    record_property: Callable[[str, object], None],
) -> None:
    """Publish a committed inline boundary on retry without dispatching."""
    await _verify_inline_publication_failure_replays_without_redispatch(
        record_property
    )


async def test_compaction_failure_diagnostics_keep_a_bounded_tail(
    record_property: Callable[[str, object], None],
) -> None:
    """Retain a fixed failure tail plus one monotonic total count."""
    await _verify_compaction_failure_diagnostics_keep_a_bounded_tail(
        record_property
    )


async def test_compaction_failure_records_cover_every_phase7_boundary(
    record_property: Callable[[str, object], None],
) -> None:
    """Prove every active Phase 7 failure-matrix cell directly."""
    record_property("conversation_acceptance_evidence", "negative")

    await _verify_matrix_validation_failure_is_predispatch()
    await _verify_compact_provider_failure_leaves_parent_reusable(
        conversation.ConversationError(
            conversation.ConversationErrorCode.VALIDATION_FAILED,
            boundary=conversation.FailureBoundary.PROVIDER_REJECTION,
        ),
        _ignore_acceptance_evidence,
    )
    await _verify_matrix_malformed_stream_failure_is_atomic()
    await _verify_compact_provider_failure_leaves_parent_reusable(
        CancelledError(),
        _ignore_acceptance_evidence,
    )
    await _verify_explicit_compact_commit_failure_is_retryable_and_atomic(
        RuntimeError("private-compact-commit-failure"),
        _ignore_acceptance_evidence,
    )
    await _verify_inline_publication_failure_replays_without_redispatch(
        _ignore_acceptance_evidence
    )
    assert await _verify_compaction_failure_diagnostics_keep_a_bounded_tail(
        _ignore_acceptance_evidence
    )


async def test_compact_and_continue_create_immutable_siblings() -> None:
    """Allow independent compact and continuation children of one parent."""
    lane = binding("lane-compact-explicit")
    root_plan = empty_stateless_plan(lane)
    root_result = conversation.fake_provider_result(root_plan, turn=1)
    root_client, store, root_coordinator, scope = _client(
        (root_result,),
        namespace="compact-race-root",
    )
    first = await root_client.create(
        "root",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    original = await store.load(first.handle.checkpoint_id, scope)
    parent = avalan.StatelessParent(handle=first.handle)
    parent_plan = conversation.StatelessProviderPlan(
        binding=lane,
        ledger=conversation.ProviderItemLedger(
            lane_id=lane.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=root_result.items,
        ),
        reasoning=root_result.reasoning,
    )
    compact_client, _, compact_coordinator, _ = _client(
        (
            conversation.fake_compaction_result(root_plan, turn=2),
            conversation.fake_compaction_result(root_plan, turn=5),
        ),
        store=store,
        namespace="compact-race-compact",
    )
    continue_client, _, continue_coordinator, _ = _client(
        (conversation.fake_provider_result(parent_plan, turn=3),),
        store=store,
        namespace="compact-race-continue",
    )

    compacted, continued = await gather(
        compact_client.compact(avalan.StandaloneCompactRequest(parent=parent)),
        continue_client.continue_conversation(
            "continued",
            avalan.StatelessConversationSettings(parent=parent),
        ),
    )
    assert type(compacted) is avalan.StandaloneCompactResult
    assert type(continued) is avalan.DirectConversationResult
    compact_checkpoint = await store.load(
        compacted.handle.checkpoint_id,
        scope,
    )
    continued_checkpoint = await store.load(
        continued.handle.checkpoint_id,
        scope,
    )
    assert compact_checkpoint.identity.parent_checkpoint_id == (
        first.handle.checkpoint_id
    )
    assert continued_checkpoint.identity.parent_checkpoint_id == (
        first.handle.checkpoint_id
    )
    assert compact_checkpoint.identity.checkpoint_id != (
        continued_checkpoint.identity.checkpoint_id
    )

    second_compact_client, _, second_compact_coordinator, _ = _client(
        (conversation.fake_compaction_result(root_plan, turn=4),),
        store=store,
        namespace="compact-race-compact-two",
    )
    compact_two, compact_three = await gather(
        compact_client.compact(
            avalan.StandaloneCompactRequest(parent=parent),
            idempotency_key=conversation.RequestIdempotencyKey(
                "compact-compact-race-one"
            ),
        ),
        second_compact_client.compact(
            avalan.StandaloneCompactRequest(parent=parent),
            idempotency_key=conversation.RequestIdempotencyKey(
                "compact-compact-race-two"
            ),
        ),
    )
    assert (
        compact_two.handle.checkpoint_id != compact_three.handle.checkpoint_id
    )
    assert await store.load(first.handle.checkpoint_id, scope) == original
    await compact_coordinator.close()
    await continue_coordinator.close()
    await second_compact_coordinator.close()
    await root_coordinator.close()


def _verify_compaction_models_and_limits_are_closed() -> None:
    """Reject unsafe thresholds, malformed handles, and stored parents."""
    failure_fields = {
        "operation": conversation.CompactionOperation.INLINE,
        "boundary": conversation.FailureBoundary.VALIDATION_BEFORE_DISPATCH,
        "error_code": conversation.ConversationErrorCode.VALIDATION_FAILED,
        "cancelled": False,
        "committed": False,
        "streaming": False,
    }
    with pytest.raises(conversation.ConversationValidationError):
        conversation.CompactionFailureRecord(
            **{
                **failure_fields,
                "operation": cast(
                    conversation.CompactionOperation,
                    object(),
                ),
            }
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.CompactionFailureRecord(
            **{
                **failure_fields,
                "error_code": "invalid-compaction-error",
            }
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.InlineCompaction(compact_threshold=2_147_483_648)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAICompactionLimits(
            min_compact_threshold=10,
            max_compact_threshold=9,
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.StandaloneCompactHandle(
            conversation_id=conversation.ConversationId("conversation"),
            checkpoint_id=conversation.CheckpointId("same"),
            branch_id=conversation.ConversationBranchId("branch"),
            parent_checkpoint_id=conversation.CheckpointId("same"),
        )
    handle_fields = {
        "conversation_id": conversation.ConversationId("conversation"),
        "checkpoint_id": conversation.CheckpointId("checkpoint"),
        "branch_id": conversation.ConversationBranchId("branch"),
        "parent_checkpoint_id": conversation.CheckpointId("parent"),
        "head_id": conversation.NamedHeadId("compact-head"),
    }
    with pytest.raises(conversation.ConversationValidationError):
        avalan.StandaloneCompactHandle(**handle_fields)
    with pytest.raises(conversation.ConversationValidationError):
        avalan.StandaloneCompactHandle(
            **handle_fields,
            expected_head_revision=conversation.NamedHeadRevision(-1),
        )
    lane = binding("lane-compact-closed-models")
    with pytest.raises(conversation.ConversationValidationError):
        conversation.StandaloneCompactProviderPlan(
            binding=lane,
            ledger=conversation.ProviderItemLedger(
                lane_id=lane.lane_id,
                normalization_version=conversation.ConversationCodecVersion(1),
                items=(),
            ),
            reasoning=empty_stateless_plan(lane).reasoning,
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.StoredConversationSettings(
            provider_storage_disclosed=True,
            compaction=cast(avalan.CompactionPolicy, object()),
        )
    stored = avalan.StoredParent(
        handle=avalan.StoredConversationHandle(
            conversation_id=conversation.ConversationId("conversation"),
            checkpoint_id=conversation.CheckpointId("stored"),
            branch_id=conversation.ConversationBranchId("branch"),
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.StandaloneCompactRequest(
            parent=cast(avalan.StatelessParent, stored)
        )
    nested: object = "leaf"
    for _ in range(34):
        nested = {"nested": nested}
    with pytest.raises(conversation.ConversationLimitError):
        conversation.ProviderItem(
            item_id=conversation.ProviderItemId("deep-compact-item"),
            lane_id=conversation.ProviderLaneId("deep-compact-lane"),
            model_call_id=conversation.ConversationModelCallId(
                "deep-compact-call"
            ),
            kind=conversation.ProviderItemKind.COMPACTION,
            order=conversation.ProviderItemOrder(0),
            provider_index=conversation.ProviderItemIndex(0),
            phase=conversation.ProviderItemPhase.COMPACTION,
            caller=conversation.ProviderItemCaller.PROVIDER,
            canonical_input=cast(dict[str, JsonValue], nested),
            normalization_version=conversation.ConversationCodecVersion(1),
            opaque_state=conversation.OpaqueProviderState(_value=b"private"),
        )


def test_compaction_models_and_limits_are_closed(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject unsafe thresholds, malformed handles, and stored parents."""
    record_property(
        "conversation_acceptance_evidence",
        "pre_dispatch_rejection",
    )
    _verify_compaction_models_and_limits_are_closed()


def test_compaction_model_and_handle_closure_remains_exact(
    record_property: Callable[[str, object], None],
) -> None:
    """Preserve exact model, handle, parent, and depth rejection."""
    record_property(
        "conversation_acceptance_evidence",
        "pre_dispatch_rejection",
    )
    _verify_compaction_models_and_limits_are_closed()
    assert (
        avalan.InlineCompaction(compact_threshold=128).compact_threshold == 128
    )
