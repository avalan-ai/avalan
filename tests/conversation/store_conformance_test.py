"""Verify the complete async in-memory conversation store contract."""

from asyncio import CancelledError
from copy import copy
from dataclasses import replace
from datetime import datetime, timedelta
from typing import cast

import pytest
from phase2_fixtures import (
    NOW,
    authority,
    binding,
    child_identity,
    coordinator,
    empty_stateless_plan,
    first_stored_plan,
    next_stateless_plan,
    request,
    root_identity,
)

import avalan.conversation as conversation

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run deterministic store races on asyncio only."""
    return "asyncio"


def _runtime(
    lane_binding: conversation.ProviderLaneBinding,
    results: tuple[conversation.ProviderResult, ...],
) -> conversation.ConversationLaneRuntime:
    return conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=conversation.fake_capability_profile(lane_binding),
        provider_script=conversation.DeterministicFakeProviderScript(
            results=results
        ),
    )


async def _seed(
    store: conversation.ConversationStore,
    *,
    suffix: str = "store",
) -> tuple[
    conversation.AuthorityScope,
    conversation.RunScopedConversationCoordinator,
    conversation.AtomicCommitReceipt,
    conversation.ProviderResult,
]:
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
    )
    receipt = await engine.execute(
        request(
            scope=scope,
            identity=root_identity(suffix),
            advance=conversation.FirstTurnAdvance(),
            response_suffix=suffix,
            key=f"key-{suffix}",
        )
    )
    return scope, engine, receipt, result


def _child_candidate(
    parent: conversation.ConversationCheckpoint,
    first_result: conversation.ProviderResult,
    *,
    suffix: str,
    turn: int = 2,
) -> conversation.ExecutionSegmentCheckpointCandidate:
    lane_binding = parent.content.lanes[0].binding
    plan = next_stateless_plan(lane_binding, first_result.items)
    result = conversation.fake_provider_result(plan, turn=turn, text=suffix)
    run = request(
        scope=parent.authority,
        identity=child_identity(parent, suffix),
        advance=conversation.OrdinaryChildAdvance(
            parent_checkpoint_id=parent.identity.checkpoint_id
        ),
        response_suffix=suffix,
        key=f"key-{suffix}",
        boundary=conversation.ConversationCommitBoundary.INTERNAL_SEGMENT,
    )
    execution_receipt = conversation.provider_lane_execution_receipt(
        authority=parent.authority,
        identity=run.identity,
        binding=lane_binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
        upstream_response_id=None,
    )
    lane = conversation.StatelessProviderLaneSnapshot(
        binding=lane_binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=lane_binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=first_result.items + result.items,
        ),
        reasoning=result.reasoning,
        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
        execution_receipt=execution_receipt,
    )
    candidate = conversation.build_checkpoint_candidate(
        run,
        parent=parent,
        completed_lanes=(lane,),
        created_at=NOW + timedelta(seconds=1),
    )
    assert isinstance(
        candidate, conversation.ExecutionSegmentCheckpointCandidate
    )
    return candidate


def _atomic_commit(
    suffix: str,
    *,
    scope: conversation.AuthorityScope | None = None,
) -> conversation.AtomicConversationCommit:
    scope = scope or authority()
    lane_binding = binding(
        f"lane-{suffix}",
        agent=str(scope.agent_id),
    )
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1, text=suffix)
    run = request(
        scope=scope,
        identity=root_identity(suffix),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(f"lane-{suffix}",),
        response_suffix=suffix,
        key=f"key-{suffix}",
    )
    execution_receipt = conversation.provider_lane_execution_receipt(
        authority=scope,
        identity=run.identity,
        binding=lane_binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
        upstream_response_id=None,
    )
    lane = conversation.StatelessProviderLaneSnapshot(
        binding=lane_binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=lane_binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=result.items,
        ),
        reasoning=result.reasoning,
        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
        execution_receipt=execution_receipt,
    )
    candidate = conversation.build_checkpoint_candidate(
        run,
        parent=None,
        completed_lanes=(lane,),
        created_at=NOW,
    )
    return conversation.AtomicConversationCommit(
        candidate=candidate,
        idempotency=conversation.RequestIdempotencyIdentity(
            authority=scope,
            operation=conversation.ConversationOperation.CREATE,
            key=run.idempotency_key,
            request_digest=conversation.CanonicalRequestDigest(
                f"digest-{suffix}"
            ),
        ),
        owner_token=f"unreserved-owner-{suffix}",
        output_candidates=(
            conversation.ProviderLaneOutputCandidate(
                lane_id=lane_binding.lane_id,
                binding=lane_binding,
                mode=conversation.ConversationMode.STATELESS,
                scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
                completed_items=result.items,
                reasoning=result.reasoning,
                usage=result.usage,
                execution_receipt=execution_receipt,
            ),
        ),
        committed_at=NOW,
        result_mode=conversation.ConversationMode.STATELESS,
        provisional_response_id=run.provisional_response_id,
        public_response_id=run.public_response_id,
        outbox_intent_id=f"outbox-{suffix}",
    )


def _two_item_atomic_commit(
    suffix: str,
) -> conversation.AtomicConversationCommit:
    base = _atomic_commit(suffix)
    output = base.output_candidates[0]
    lane = base.candidate.checkpoint.content.lanes[0]
    assert isinstance(lane, conversation.StatelessProviderLaneSnapshot)
    next_plan = next_stateless_plan(output.binding, output.completed_items)
    next_result = conversation.fake_provider_result(
        next_plan,
        turn=2,
        text=f"{suffix}-second",
    )
    items = output.completed_items + next_result.items
    usage = conversation.ProviderUsage(input_tokens=30, output_tokens=15)
    receipt = conversation.provider_lane_execution_receipt(
        authority=base.candidate.checkpoint.authority,
        identity=base.candidate.checkpoint.identity,
        binding=output.binding,
        mode=output.mode,
        scope=output.scope,
        completed_items=items,
        reasoning=next_result.reasoning,
        usage=usage,
        upstream_response_id=None,
    )
    updated_lane = replace(
        lane,
        ledger=conversation.ProviderItemLedger(
            lane_id=lane.lane_id,
            normalization_version=lane.binding.continuation_codec_version,
            items=items,
        ),
        reasoning=next_result.reasoning,
        execution_receipt=receipt,
    )
    checkpoint = conversation.with_checkpoint_integrity(
        replace(
            base.candidate.checkpoint,
            content=replace(
                base.candidate.checkpoint.content,
                lanes=(updated_lane,),
            ),
        )
    )
    return replace(
        base,
        candidate=replace(base.candidate, checkpoint=checkpoint),
        output_candidates=(
            replace(
                output,
                completed_items=items,
                reasoning=next_result.reasoning,
                usage=usage,
                execution_receipt=receipt,
            ),
        ),
    )


def _stored_atomic_commit(
    suffix: str,
) -> conversation.AtomicConversationCommit:
    scope = authority()
    lane_binding = binding(f"lane-{suffix}")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text=suffix,
    )
    assert result.upstream_response_id is not None
    run = request(
        scope=scope,
        identity=root_identity(suffix),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(lane_binding.lane_id,),
        modes=(conversation.ConversationMode.STORED,),
        response_suffix=suffix,
        key=f"key-{suffix}",
        stored_retention=True,
    )
    receipt = conversation.provider_lane_execution_receipt(
        authority=scope,
        identity=run.identity,
        binding=lane_binding,
        mode=conversation.ConversationMode.STORED,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
        upstream_response_id=result.upstream_response_id,
    )
    lane = conversation.StoredProviderLaneSnapshot(
        binding=lane_binding,
        upstream_response_id=result.upstream_response_id,
        reasoning=result.reasoning,
        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
        execution_receipt=receipt,
    )
    candidate = conversation.build_checkpoint_candidate(
        run,
        parent=None,
        completed_lanes=(lane,),
        created_at=NOW,
    )
    return conversation.AtomicConversationCommit(
        candidate=candidate,
        idempotency=conversation.RequestIdempotencyIdentity(
            authority=scope,
            operation=conversation.ConversationOperation.CREATE,
            key=run.idempotency_key,
            request_digest=conversation.CanonicalRequestDigest(
                f"digest-{suffix}"
            ),
        ),
        owner_token=f"unreserved-owner-{suffix}",
        output_candidates=(
            conversation.ProviderLaneOutputCandidate(
                lane_id=lane_binding.lane_id,
                binding=lane_binding,
                mode=conversation.ConversationMode.STORED,
                scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
                completed_items=result.items,
                reasoning=result.reasoning,
                usage=result.usage,
                execution_receipt=receipt,
                upstream_response_id=result.upstream_response_id,
            ),
        ),
        committed_at=NOW,
        result_mode=conversation.ConversationMode.STORED,
        provisional_response_id=run.provisional_response_id,
        public_response_id=run.public_response_id,
        outbox_intent_id=f"outbox-{suffix}",
    )


def _execution_reservation(
    commit: conversation.AtomicConversationCommit,
) -> conversation.ConversationExecutionReservation:
    return conversation.ConversationExecutionReservation(
        idempotency=commit.idempotency,
        identity=commit.candidate.checkpoint.identity,
        lanes=tuple(
            conversation.ProviderLaneExecutionReservation(
                binding=output.binding,
                mode=output.mode,
                scope=output.scope,
            )
            for output in commit.output_candidates
        ),
    )


async def _stage_atomic(
    store: conversation.InMemoryConversationStore,
    commit: conversation.AtomicConversationCommit,
    owner_token: str,
) -> tuple[conversation.ProviderLaneExecutionAttestation, ...]:
    return tuple(
        [
            await store.stage_execution(
                _execution_stage(commit, output, owner_token)
            )
            for output in commit.output_candidates
        ]
    )


def _execution_stage(
    commit: conversation.AtomicConversationCommit,
    output: conversation.ProviderLaneOutputCandidate,
    owner_token: str,
) -> conversation.ProviderLaneExecutionStage:
    """Return one exact public staging request for a prepared lane."""
    return conversation.ProviderLaneExecutionStage(
        idempotency=commit.idempotency,
        owner_token=owner_token,
        identity=commit.candidate.checkpoint.identity,
        binding=output.binding,
        mode=output.mode,
        scope=output.scope,
        completed_items=output.completed_items,
        reasoning=output.reasoning,
        usage=output.usage,
        execution_receipt=output.execution_receipt,
        upstream_response_id=output.upstream_response_id,
    )


async def _prepare_atomic(
    store: conversation.InMemoryConversationStore,
    commit: conversation.AtomicConversationCommit,
) -> conversation.AtomicConversationCommit:
    reservation = _execution_reservation(commit)
    resolution = await store.reserve_idempotency(
        commit.idempotency,
        execution=reservation,
    )
    assert resolution.owner_token is not None
    owned = replace(commit, owner_token=resolution.owner_token)
    assert commit.provisional_response_id is not None
    assert commit.public_response_id is not None
    try:
        await store.allocate_public_response(
            conversation.ProvisionalPublicResponse(
                provisional_response_id=commit.provisional_response_id,
                public_response_id=commit.public_response_id,
                owner_token=resolution.owner_token,
                authority_digest=str(
                    conversation.authority_digest(commit.idempotency.authority)
                ),
            )
        )
        attestations = await _stage_atomic(
            store,
            commit,
            resolution.owner_token,
        )
    except BaseException:
        await store.abandon_idempotency(
            commit.idempotency,
            resolution.owner_token,
            ambiguous=False,
        )
        raise
    return replace(owned, execution_attestations=attestations)


def _outbox_target(
    record: conversation.OutboxRecord,
    *,
    scope: conversation.AuthorityScope | None = None,
) -> conversation.OutboxClaimTarget:
    """Return the exact trusted target for one committed outbox record."""
    return conversation.OutboxClaimTarget(
        authority=scope or authority(),
        checkpoint_id=record.intent.checkpoint_id,
        public_response_id=record.intent.public_response_id,
        intent_id=record.intent.intent_id,
    )


def _claimed_record(
    resolution: conversation.OutboxClaimResolution,
) -> conversation.OutboxRecord:
    """Return the leased record from one successful closed claim."""
    assert (
        resolution.disposition is conversation.OutboxClaimDisposition.CLAIMED
    )
    assert resolution.record is not None
    return resolution.record


def _atomic_store_snapshot(
    store: conversation.InMemoryConversationStore,
) -> tuple[object, ...]:
    """Capture every authoritative in-memory allocation byte-for-byte."""
    return (
        dict(store._checkpoints),
        {key: set(value) for key, value in store._children.items()},
        dict(store._provisional),
        dict(store._public),
        dict(store._results),
        dict(store._outputs),
        dict(store._idempotency),
        dict(store._execution_staging),
        dict(store._execution_stage_keys),
        dict(store._heads),
        dict(store._outbox),
        dict(store._outbox_ready_order),
        dict(store._terminal),
        store._owner_sequence,
        store._execution_stage_sequence,
        store._outbox_ready_sequence,
        store.diagnostics,
    )


async def test_async_store_conformance_suite() -> None:
    """Exercise create/load/authorize/stage/head/list/lifecycle/close."""
    store: conversation.ConversationStore = (
        conversation.InMemoryConversationStore()
    )
    scope, _engine, root, first_result = await _seed(store)
    checkpoint_id = root.checkpoint.identity.checkpoint_id
    loaded = await store.load(checkpoint_id, scope)
    authorized = await store.authorize(checkpoint_id, scope)
    assert loaded == root.checkpoint == authorized

    candidate = _child_candidate(
        root.checkpoint, first_result, suffix="staged-child"
    )
    unit = await store.stage(candidate)
    async with unit:
        await unit.rollback()
    unit = await store.stage(candidate)
    async with unit:
        child = await unit.commit()
    assert child.identity.parent_checkpoint_id == checkpoint_id
    assert await store.branch_count(checkpoint_id, scope) == 1

    head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("main-phase2"),
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=checkpoint_id,
    )
    await store.create_head(head, scope)
    assert await store.load_head(head.head_id, scope) == head
    child_head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("child-phase2"),
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=child.identity.checkpoint_id,
    )
    await store.create_head(child_head, scope)
    page = await store.list_checkpoints(scope, cursor=None, limit=1)
    assert len(page.checkpoints) == 1
    assert page.next_cursor is not None
    next_page = await store.list_checkpoints(
        scope, cursor=page.next_cursor, limit=10
    )
    assert len(next_page.checkpoints) == 1
    assert next_page.next_cursor is None

    assert root.result is not None
    public_id = root.result.handle
    assert isinstance(public_id, conversation.StatelessConversationHandle)
    retrieved = await store.retrieve(
        conversation.PublicResponseId("response-store"), scope
    )
    assert retrieved == root.result
    tombstone = await store.tombstone(
        conversation.PublicResponseId("response-store"),
        scope,
        NOW + timedelta(minutes=1),
    )
    assert tombstone.lifecycle is conversation.CheckpointLifecycle.TOMBSTONED
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.retrieve(
            conversation.PublicResponseId("response-store"), scope
        )
    await store.delete(
        conversation.PublicResponseId("response-store"),
        scope,
        NOW + timedelta(minutes=2),
    )
    assert await store.load(child.identity.checkpoint_id, scope) == child
    assert await store.load_head(child_head.head_id, scope) == child_head
    await store.close()
    await store.close()
    with pytest.raises(conversation.ConversationStorageError):
        await store.load(child.identity.checkpoint_id, scope)


async def test_unavailable_state_has_constant_disclosure() -> None:
    """Use one stable error for absent, wrong-authority, and tombstoned IDs."""
    store = conversation.InMemoryConversationStore()
    scope, _engine, root, _result = await _seed(store, suffix="auth")
    wrong = authority("wrong-principal")
    failures: list[conversation.ConversationAuthorizationError] = []
    for checkpoint_id, candidate_scope in (
        (conversation.CheckpointId("absent-checkpoint"), scope),
        (root.checkpoint.identity.checkpoint_id, wrong),
    ):
        with pytest.raises(
            conversation.ConversationAuthorizationError
        ) as raised:
            await store.load(checkpoint_id, candidate_scope)
        failures.append(raised.value)
    await store.tombstone(
        conversation.PublicResponseId("response-auth"),
        scope,
        NOW + timedelta(seconds=1),
    )
    with pytest.raises(conversation.ConversationAuthorizationError) as raised:
        await store.load(root.checkpoint.identity.checkpoint_id, scope)
    failures.append(raised.value)
    assert {item.code for item in failures} == {
        conversation.ConversationErrorCode.AUTHORIZATION_FAILED
    }
    assert len({str(item) for item in failures}) == 1
    assert len({repr(item) for item in failures}) == 1


async def test_idempotency_reservation_is_exact_and_fenced() -> None:
    """Resolve execute/conflict/fenced/known-no-dispatch deterministically."""
    store = conversation.InMemoryConversationStore()
    scope = authority()
    identity = conversation.RequestIdempotencyIdentity(
        authority=scope,
        operation=conversation.ConversationOperation.CREATE,
        key=conversation.RequestIdempotencyKey("idempotency-phase2"),
        request_digest=conversation.CanonicalRequestDigest("digest-phase2"),
    )
    first = await store.reserve_idempotency(identity)
    assert first.disposition is conversation.IdempotencyDisposition.EXECUTE
    assert first.owner_token is not None
    duplicate = await store.reserve_idempotency(identity)
    assert duplicate.disposition is conversation.IdempotencyDisposition.FENCED
    conflict_identity = conversation.RequestIdempotencyIdentity(
        authority=scope,
        operation=identity.operation,
        key=identity.key,
        request_digest=conversation.CanonicalRequestDigest("different-digest"),
    )
    conflict = await store.reserve_idempotency(conflict_identity)
    assert conflict.disposition is conversation.IdempotencyDisposition.CONFLICT
    await store.fence_idempotency(identity, first.owner_token, ambiguous=False)
    retriable = await store.reserve_idempotency(identity)
    assert retriable.disposition is conversation.IdempotencyDisposition.EXECUTE
    assert retriable.owner_token is not None
    await store.fence_idempotency(
        identity, retriable.owner_token, ambiguous=True
    )
    fenced = await store.reserve_idempotency(identity)
    assert fenced.disposition is conversation.IdempotencyDisposition.FENCED

    waiter_store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(max_in_flight=1)
    )
    owner = await waiter_store.reserve_idempotency(identity)
    assert owner.owner_token is not None
    assert (
        await waiter_store.reserve_idempotency(identity)
    ).disposition is conversation.IdempotencyDisposition.FENCED
    other_identity = replace(
        identity,
        key=conversation.RequestIdempotencyKey("idempotency-other"),
    )
    with pytest.raises(conversation.ConversationLimitError):
        await waiter_store.reserve_idempotency(other_identity)
    assert waiter_store.diagnostics.idempotency_waiters == 0
    await waiter_store.abandon_idempotency(
        identity,
        owner.owner_token,
        ambiguous=False,
    )


async def test_idempotency_lease_fences_and_expires_abandoned_owner() -> None:
    """Fence active duplicates and expire an owner without caller time."""
    store_clock = conversation.DeterministicFakeClock(NOW)
    store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(idempotency_lease_seconds=1),
        clock=store_clock,
    )
    commit = _atomic_commit("wait-timeout")
    owner = await store.reserve_idempotency(commit.idempotency)
    assert owner.owner_token is not None
    assert commit.provisional_response_id is not None
    assert commit.public_response_id is not None
    await store.allocate_public_response(
        conversation.ProvisionalPublicResponse(
            provisional_response_id=commit.provisional_response_id,
            public_response_id=commit.public_response_id,
            owner_token=owner.owner_token,
            authority_digest=str(
                conversation.authority_digest(commit.idempotency.authority)
            ),
        )
    )

    active = await store.reserve_idempotency(commit.idempotency)
    assert active.disposition is conversation.IdempotencyDisposition.FENCED
    assert store.diagnostics.provisional_responses == 1
    store_clock.set(NOW + timedelta(seconds=2))
    expired = await store.reserve_idempotency(commit.idempotency)

    assert expired.disposition is conversation.IdempotencyDisposition.FENCED
    assert store.diagnostics.idempotency_waiters == 0
    assert store.diagnostics.provisional_responses == 0


async def test_authority_scopes_named_heads_and_binding_agents() -> None:
    """Isolate equal head names by authority and reject agent drift."""
    store = conversation.InMemoryConversationStore()
    scope_a = authority(
        "principal-a",
        tenant="tenant-a",
        agent="agent-a",
    )
    scope_b = authority(
        "principal-b",
        tenant="tenant-b",
        agent="agent-b",
    )
    binding_a = binding("lane-a", agent="agent-a")
    binding_b = binding("lane-b", agent="agent-b")
    root_plan_a = empty_stateless_plan(binding_a)
    root_plan_b = empty_stateless_plan(binding_b)
    root_result_a = conversation.fake_provider_result(root_plan_a, turn=1)
    root_result_b = conversation.fake_provider_result(root_plan_b, turn=1)
    child_result_a = conversation.fake_provider_result(
        next_stateless_plan(binding_a, root_result_a.items), turn=2
    )
    child_result_b = conversation.fake_provider_result(
        next_stateless_plan(binding_b, root_result_b.items), turn=2
    )
    engine_a = coordinator(
        store=store,
        scope=scope_a,
        runtimes=(_runtime(binding_a, (root_result_a, child_result_a)),),
    )
    engine_b = coordinator(
        store=store,
        scope=scope_b,
        runtimes=(_runtime(binding_b, (root_result_b, child_result_b)),),
    )
    root_a = await engine_a.execute(
        request(
            scope=scope_a,
            identity=root_identity("authority-a"),
            advance=conversation.FirstTurnAdvance(),
            lane_ids=("lane-a",),
            response_suffix="authority-a",
            key="key-authority-a",
        )
    )
    root_b = await engine_b.execute(
        request(
            scope=scope_b,
            identity=root_identity("authority-b"),
            advance=conversation.FirstTurnAdvance(),
            lane_ids=("lane-b",),
            response_suffix="authority-b",
            key="key-authority-b",
        )
    )
    head_id = conversation.NamedHeadId("main")
    head_a = conversation.NamedHeadSnapshot(
        head_id=head_id,
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=root_a.checkpoint.identity.checkpoint_id,
    )
    head_b = conversation.NamedHeadSnapshot(
        head_id=head_id,
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=root_b.checkpoint.identity.checkpoint_id,
    )
    await store.create_head(head_a, scope_a)
    await store.create_head(head_b, scope_b)
    assert await store.load_head(head_id, scope_a) == head_a
    assert await store.load_head(head_id, scope_b) == head_b

    child_a = request(
        scope=scope_a,
        identity=child_identity(root_a.checkpoint, "authority-child-a"),
        advance=conversation.NamedHeadAdvance(
            head_id=head_id,
            parent_checkpoint_id=root_a.checkpoint.identity.checkpoint_id,
            expected_revision=conversation.NamedHeadRevision(0),
        ),
        lane_ids=("lane-a",),
        response_suffix="authority-child-a",
        key="key-authority-child-a",
    )
    await engine_a.execute(child_a)
    assert (await store.load_head(head_id, scope_a)).revision == 1
    assert (await store.load_head(head_id, scope_b)).revision == 0

    child_b = request(
        scope=scope_b,
        identity=child_identity(root_b.checkpoint, "authority-child-b"),
        advance=conversation.NamedHeadAdvance(
            head_id=head_id,
            parent_checkpoint_id=root_b.checkpoint.identity.checkpoint_id,
            expected_revision=conversation.NamedHeadRevision(0),
        ),
        lane_ids=("lane-b",),
        response_suffix="authority-child-b",
        key="key-authority-child-b",
    )
    await engine_b.execute(child_b)
    assert (await store.load_head(head_id, scope_b)).revision == 1
    assert store.diagnostics.heads == 2
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load_head(
            head_id,
            authority(
                "principal-c",
                tenant="tenant-c",
                agent="agent-c",
            ),
        )

    mismatched = _atomic_commit("agent-mismatch", scope=scope_a)
    assert isinstance(
        mismatched.candidate, conversation.OutwardTurnCheckpointCandidate
    )
    mismatched_candidate = replace(
        mismatched.candidate,
        checkpoint=replace(
            mismatched.candidate.checkpoint,
            authority=scope_b,
        ),
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.create(mismatched_candidate)


async def test_limits_reject_before_store_growth() -> None:
    """Enforce checkpoint, branch, page, and concurrency bounds."""
    store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(
            max_checkpoints=1,
            max_checkpoint_bytes=8_388_608,
            max_provider_items=10,
            max_depth=1,
            max_children_per_parent=1,
            max_in_flight=1,
            max_outbox_records=2,
            max_terminal_metadata=1,
            max_page_size=1,
        )
    )
    scope, _engine, root, first_result = await _seed(store, suffix="limits")
    candidate = _child_candidate(
        root.checkpoint, first_result, suffix="too-many"
    )
    with pytest.raises(conversation.ConversationLimitError):
        await store.create(candidate)
    assert store.diagnostics.checkpoints == 1
    with pytest.raises(conversation.ConversationLimitError):
        await store.list_checkpoints(scope, cursor=None, limit=2)
    first_identity = conversation.RequestIdempotencyIdentity(
        authority=scope,
        operation=conversation.ConversationOperation.CONTINUE,
        key=conversation.RequestIdempotencyKey("limit-key-one"),
        request_digest=conversation.CanonicalRequestDigest("limit-digest-one"),
    )
    second_identity = conversation.RequestIdempotencyIdentity(
        authority=scope,
        operation=conversation.ConversationOperation.CONTINUE,
        key=conversation.RequestIdempotencyKey("limit-key-two"),
        request_digest=conversation.CanonicalRequestDigest("limit-digest-two"),
    )
    await store.reserve_idempotency(first_identity)
    with pytest.raises(conversation.ConversationLimitError):
        await store.reserve_idempotency(second_identity)

    head_store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(max_heads=1)
    )
    head_first = await head_store.create(
        _atomic_commit("head-limit-first").candidate
    )
    head_second = await head_store.create(
        _atomic_commit("head-limit-second").candidate
    )
    await head_store.create_head(
        conversation.NamedHeadSnapshot(
            head_id=conversation.NamedHeadId("head-limit-first"),
            revision=conversation.NamedHeadRevision(0),
            checkpoint_id=head_first.identity.checkpoint_id,
        ),
        scope,
    )
    with pytest.raises(conversation.ConversationLimitError):
        await head_store.create_head(
            conversation.NamedHeadSnapshot(
                head_id=conversation.NamedHeadId("head-limit-second"),
                revision=conversation.NamedHeadRevision(0),
                checkpoint_id=head_second.identity.checkpoint_id,
            ),
            scope,
        )


async def test_sweep_preserves_self_contained_descendants() -> None:
    """Expire and later delete a parent while retaining its committed child."""
    store = conversation.InMemoryConversationStore()
    scope, _engine, root, first_result = await _seed(store, suffix="sweep")
    child = await store.create(
        _child_candidate(root.checkpoint, first_result, suffix="sweep-child")
    )
    first = await store.sweep(NOW + timedelta(hours=2), limit=1)
    assert first.expired == 1
    assert first.deleted == 0
    second = await store.sweep(NOW + timedelta(hours=2), limit=1)
    assert second.deleted == 1
    assert await store.load(child.identity.checkpoint_id, scope) == child
    third = await store.sweep(NOW + timedelta(hours=2), limit=1)
    assert third.expired == 1
    fourth = await store.sweep(NOW + timedelta(hours=2), limit=1)
    assert fourth.deleted == 1
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(child.identity.checkpoint_id, scope)


async def test_sweep_ignores_unrelated_public_records() -> None:
    """Expire and delete one mapping without changing an unrelated mapping."""
    store = conversation.InMemoryConversationStore()
    first = _atomic_commit("sweep-public-first")
    second = _atomic_commit("sweep-public-second")
    first = await _prepare_atomic(store, first)
    second = await _prepare_atomic(store, second)
    await store.commit_atomic(first)
    second_receipt = await store.commit_atomic(second)

    expired = await store.sweep(NOW + timedelta(hours=2), limit=1)
    assert expired == conversation.SweepReceipt(expired=1, deleted=0)
    deleted = await store.sweep(NOW + timedelta(hours=2), limit=1)
    assert deleted == conversation.SweepReceipt(expired=0, deleted=1)
    assert second.public_response_id is not None
    assert (
        await store.retrieve(second.public_response_id, authority())
        == second_receipt.result
    )


async def test_outbox_claim_release_and_acknowledge_are_exact() -> None:
    """Settle one committed intent through explicit outbox states."""
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    commit = await _prepare_atomic(store, _atomic_commit("outbox"))
    receipt = await store.commit_atomic(commit)
    assert receipt.outbox is not None
    target = _outbox_target(receipt.outbox)
    claimed = _claimed_record(await store.claim_outbox(target))
    assert claimed.state is conversation.OutboxState.CLAIMED
    assert claimed.attempts == 1
    assert claimed.lease_owner is not None
    assert (await store.claim_outbox(target)).disposition is (
        conversation.OutboxClaimDisposition.ACTIVELY_LEASED
    )
    with pytest.raises(conversation.ConversationConflictError):
        await store.release_outbox(target, "wrong-owner")
    await store.release_outbox(
        target,
        claimed.lease_owner,
    )
    reclaimed = _claimed_record(await store.claim_outbox(target))
    assert reclaimed.state is conversation.OutboxState.CLAIMED
    assert reclaimed.attempts == 2
    assert reclaimed.lease_owner is not None
    assert reclaimed.lease_owner != claimed.lease_owner
    await store.acknowledge_outbox(
        target,
        reclaimed.lease_owner,
    )
    assert (await store.claim_outbox(target)).disposition is (
        conversation.OutboxClaimDisposition.ALREADY_PUBLISHED
    )
    assert store.diagnostics.outbox_records == 1
    assert await store.prune(NOW, limit=1) == conversation.PruneReceipt(
        outbox_records=1,
        idempotency_records=0,
    )
    assert store.diagnostics.outbox_records == 0


async def test_outbox_claim_can_target_one_committed_intent() -> None:
    """Keep a coordinator from publishing another run's pending intent."""
    clock = conversation.DeterministicFakeClock(NOW)
    store = conversation.InMemoryConversationStore(clock=clock)
    scope_a = authority("principal-a", tenant="tenant-a")
    scope_b = authority("principal-b", tenant="tenant-b")
    first = await _prepare_atomic(
        store,
        _atomic_commit("target-first", scope=scope_a),
    )
    second = await _prepare_atomic(
        store,
        _atomic_commit("target-second", scope=scope_b),
    )
    first_receipt = await store.commit_atomic(first)
    second_receipt = await store.commit_atomic(second)
    assert first_receipt.outbox is not None
    assert second_receipt.outbox is not None

    with pytest.raises(conversation.ConversationValidationError):
        await store.claim_outbox(
            cast(conversation.OutboxClaimTarget, object())
        )
    second_target = _outbox_target(second_receipt.outbox, scope=scope_b)
    missing_target = replace(
        second_target,
        intent_id="missing-intent",
    )
    assert (await store.claim_outbox(missing_target)).disposition is (
        conversation.OutboxClaimDisposition.NOT_FOUND_OR_UNAUTHORIZED
    )

    for unauthorized_target in (
        replace(second_target, authority=scope_a),
        replace(
            second_target,
            checkpoint_id=conversation.CheckpointId("wrong-checkpoint"),
        ),
        replace(
            second_target,
            public_response_id=conversation.PublicResponseId("wrong-response"),
        ),
    ):
        assert (await store.claim_outbox(unauthorized_target)).disposition is (
            conversation.OutboxClaimDisposition.NOT_FOUND_OR_UNAUTHORIZED
        )

    second_claim = _claimed_record(await store.claim_outbox(second_target))
    assert second_claim.intent == second_receipt.outbox.intent
    assert (await store.claim_outbox(second_target)).disposition is (
        conversation.OutboxClaimDisposition.ACTIVELY_LEASED
    )
    clock.set(NOW + timedelta(seconds=31))
    reclaimed_second = _claimed_record(await store.claim_outbox(second_target))
    assert reclaimed_second.lease_owner != second_claim.lease_owner

    first_claim = _claimed_record(
        await store.claim_outbox(
            _outbox_target(first_receipt.outbox, scope=scope_a)
        )
    )
    assert first_claim.intent == first_receipt.outbox.intent


async def test_outbox_and_rollback_cancellation_settle_store_ownership() -> (
    None
):
    """Mutate lease and rollback state before re-raising cancellation."""
    acknowledge_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:outbox_acknowledge",
                exception=CancelledError(),
            ),
        )
    )
    acknowledge_store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW),
        boundary_hook=conversation.FakeStoreBoundaryHook(
            acknowledge_controller
        ),
    )
    acknowledge_commit = await _prepare_atomic(
        acknowledge_store,
        _atomic_commit("cancel-acknowledge"),
    )
    acknowledge_receipt = await acknowledge_store.commit_atomic(
        acknowledge_commit
    )
    assert acknowledge_receipt.outbox is not None
    acknowledge_target = _outbox_target(acknowledge_receipt.outbox)
    acknowledged = _claimed_record(
        await acknowledge_store.claim_outbox(acknowledge_target)
    )
    assert acknowledged.lease_owner is not None
    with pytest.raises(CancelledError):
        await acknowledge_store.acknowledge_outbox(
            acknowledge_target,
            acknowledged.lease_owner,
        )
    assert (
        await acknowledge_store.claim_outbox(acknowledge_target)
    ).disposition is conversation.OutboxClaimDisposition.ALREADY_PUBLISHED
    assert (await acknowledge_store.prune(NOW, limit=1)).outbox_records == 1

    release_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:outbox_release",
                exception=CancelledError(),
            ),
        )
    )
    release_store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW),
        boundary_hook=conversation.FakeStoreBoundaryHook(release_controller),
    )
    release_commit = await _prepare_atomic(
        release_store,
        _atomic_commit("cancel-release"),
    )
    release_receipt = await release_store.commit_atomic(release_commit)
    assert release_receipt.outbox is not None
    release_target = _outbox_target(release_receipt.outbox)
    released = _claimed_record(
        await release_store.claim_outbox(release_target)
    )
    assert released.lease_owner is not None
    with pytest.raises(CancelledError):
        await release_store.release_outbox(
            release_target,
            released.lease_owner,
        )
    reclaimed = _claimed_record(
        await release_store.claim_outbox(release_target)
    )
    assert reclaimed.attempts == 2

    rollback_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:rollback",
                exception=CancelledError(),
            ),
        )
    )
    rollback_store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(rollback_controller)
    )
    rollback_commit = await _prepare_atomic(
        rollback_store,
        _atomic_commit("cancel-store-rollback"),
    )
    with pytest.raises(CancelledError):
        await rollback_store.abandon_idempotency(
            rollback_commit.idempotency,
            rollback_commit.owner_token,
            ambiguous=False,
        )
    assert rollback_store.diagnostics.idempotency_records == 0
    assert rollback_store.diagnostics.provisional_responses == 0

    rollback_attempt_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:rollback",
                exception=CancelledError(),
            ),
        )
    )
    rollback_attempt_store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(
            rollback_attempt_controller
        )
    )
    rollback_attempt_commit = await _prepare_atomic(
        rollback_attempt_store,
        _atomic_commit("cancel-rollback-attempt"),
    )
    with pytest.raises(CancelledError):
        await rollback_attempt_store.rollback_attempt(
            rollback_attempt_commit.owner_token
        )
    assert rollback_attempt_store.diagnostics.provisional_responses == 0
    assert rollback_attempt_store.diagnostics.idempotency_records == 1
    await rollback_attempt_store.abandon_idempotency(
        rollback_attempt_commit.idempotency,
        rollback_attempt_commit.owner_token,
        ambiguous=False,
    )

    settled_attempt_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:rollback_settled",
                exception=CancelledError(),
            ),
        )
    )
    settled_attempt_store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(
            settled_attempt_controller
        )
    )
    settled_attempt_commit = await _prepare_atomic(
        settled_attempt_store,
        _atomic_commit("cancel-rollback-attempt-settled"),
    )
    with pytest.raises(CancelledError):
        await settled_attempt_store.rollback_attempt(
            settled_attempt_commit.owner_token
        )
    assert settled_attempt_store.diagnostics.provisional_responses == 0
    await settled_attempt_store.abandon_idempotency(
        settled_attempt_commit.idempotency,
        settled_attempt_commit.owner_token,
        ambiguous=False,
    )

    fence_store = conversation.InMemoryConversationStore()
    fence_commit = _atomic_commit("cancel-fence")
    fence_owner = await fence_store.reserve_idempotency(
        fence_commit.idempotency
    )
    assert fence_owner.owner_token is not None
    fence_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:idempotency",
                exception=CancelledError(),
            ),
        )
    )
    fence_store._hook = conversation.FakeStoreBoundaryHook(fence_controller)
    with pytest.raises(CancelledError):
        await fence_store.fence_idempotency(
            fence_commit.idempotency,
            fence_owner.owner_token,
            ambiguous=False,
        )
    assert await fence_store.prune(NOW, limit=1) == (
        conversation.PruneReceipt(
            outbox_records=0,
            idempotency_records=1,
        )
    )

    reconcile_store = conversation.InMemoryConversationStore()
    reconcile_commit = _atomic_commit("cancel-reconcile")
    reconcile_owner = await reconcile_store.reserve_idempotency(
        reconcile_commit.idempotency
    )
    assert reconcile_owner.owner_token is not None
    reconcile_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:idempotency_reconcile",
                exception=CancelledError(),
            ),
        )
    )
    reconcile_store._hook = conversation.FakeStoreBoundaryHook(
        reconcile_controller
    )
    with pytest.raises(CancelledError):
        await reconcile_store.reconcile_idempotency(
            reconcile_commit.idempotency,
            reconcile_owner.owner_token,
            ambiguous=True,
        )
    assert (
        await reconcile_store.reserve_idempotency(reconcile_commit.idempotency)
    ).disposition is conversation.IdempotencyDisposition.FENCED
    await reconcile_store.reconcile_idempotency(
        reconcile_commit.idempotency,
        reconcile_owner.owner_token,
        ambiguous=True,
    )
    with pytest.raises(conversation.ConversationConflictError):
        await reconcile_store.reconcile_idempotency(
            reconcile_commit.idempotency,
            "wrong-owner",
            ambiguous=True,
        )


async def test_store_close_cancellation_settles_before_reraise() -> None:
    """Close every owned resource before propagating cancellation."""
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:close",
                exception=CancelledError(),
            ),
        )
    )
    store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(controller)
    )
    commit = await _prepare_atomic(store, _atomic_commit("cancel-close"))
    await store.commit_atomic(commit)
    with pytest.raises(CancelledError):
        await store.close()
    assert store.diagnostics.closed
    assert store.diagnostics.checkpoints == 0
    assert store.diagnostics.idempotency_waiters == 0
    assert not store.diagnostics.locked
    await store.close()
    store._hook = conversation.FakeStoreBoundaryHook(
        conversation.DeterministicFaultController(
            (
                conversation.FaultAction(
                    label="store:close",
                    exception=CancelledError(),
                ),
            )
        )
    )
    with pytest.raises(CancelledError):
        await store.close()


@pytest.mark.parametrize(
    ("operation", "boundary", "settled"),
    (
        ("abandon", "rollback_begin", False),
        ("abandon", "rollback_settled", True),
        ("reconcile", "idempotency_reconcile_begin", False),
        ("reconcile", "idempotency_reconcile_settled", True),
    ),
)
async def test_idempotency_cleanup_reports_pre_and_post_cancellation(
    operation: str,
    boundary: str,
    settled: bool,
) -> None:
    """Distinguish cancellation before mutation from settled cancellation."""
    store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(
            conversation.DeterministicFaultController(
                (
                    conversation.FaultAction(
                        label=f"store:{boundary}",
                        exception=CancelledError(),
                    ),
                )
            )
        )
    )
    commit = await _prepare_atomic(store, _atomic_commit(boundary))
    effect = (
        store.abandon_idempotency
        if operation == "abandon"
        else store.reconcile_idempotency
    )
    with pytest.raises(CancelledError):
        await effect(
            commit.idempotency,
            commit.owner_token,
            ambiguous=False,
        )
    resolution = await store.inspect_idempotency_settlement(
        commit.idempotency,
        commit.owner_token,
    )
    assert (
        resolution.disposition
        is conversation.IdempotencySettlementDisposition.SETTLED
    ) is settled
    if not settled:
        assert resolution.disposition is (
            conversation.IdempotencySettlementDisposition.LEASED
        )
        assert resolution.lease_expires_at is not None
        resolution = await store.reconcile_idempotency(
            commit.idempotency,
            commit.owner_token,
            ambiguous=False,
        )
        assert resolution.disposition is (
            conversation.IdempotencySettlementDisposition.SETTLED
        )
    assert store.diagnostics.provisional_responses == 0


@pytest.mark.parametrize(
    ("boundary", "closed"),
    (("close_begin", False), ("close_settled", True)),
)
async def test_store_close_reports_pre_and_post_cancellation(
    boundary: str,
    closed: bool,
) -> None:
    """Report close truthfully on either side of destructive settlement."""
    store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(
            conversation.DeterministicFaultController(
                (
                    conversation.FaultAction(
                        label=f"store:{boundary}",
                        exception=CancelledError(),
                    ),
                )
            )
        )
    )
    commit = await _prepare_atomic(store, _atomic_commit(boundary))
    await store.commit_atomic(commit)
    before = store.diagnostics
    with pytest.raises(CancelledError):
        await store.close()
    resolution = await store.inspect_close()
    assert (
        resolution.disposition is conversation.StoreCloseDisposition.CLOSED
    ) is closed
    assert store.diagnostics.closed is closed
    if closed:
        assert store.diagnostics.checkpoints == 0
    else:
        assert store.diagnostics.checkpoints == before.checkpoints
        await store.close()


async def test_generic_outbox_recovery_is_bounded_fair_and_isolated() -> None:
    """Recover oldest available work without cross-authority disclosure."""
    clock = conversation.DeterministicFakeClock(NOW)
    store = conversation.InMemoryConversationStore(
        clock=clock,
        limits=conversation.StoreLimits(max_page_size=2),
    )
    scope_a = authority("recovery-principal-a")
    scope_b = authority("recovery-principal-b")

    async def commit_outbox(
        suffix: str,
        scope: conversation.AuthorityScope,
        at: datetime,
    ) -> conversation.OutboxRecord:
        commit = replace(
            _atomic_commit(suffix, scope=scope),
            committed_at=at,
        )
        prepared = await _prepare_atomic(store, commit)
        receipt = await store.commit_atomic(prepared)
        assert receipt.outbox is not None
        return receipt.outbox

    a_oldest = await commit_outbox("recovery-a-oldest", scope_a, NOW)
    b_only = await commit_outbox(
        "recovery-b-only",
        scope_b,
        NOW + timedelta(seconds=1),
    )
    a_next = await commit_outbox(
        "recovery-a-next",
        scope_a,
        NOW + timedelta(seconds=2),
    )
    a_last = await commit_outbox(
        "recovery-a-last",
        scope_a,
        NOW + timedelta(seconds=3),
    )
    worker_a = store.create_outbox_recovery_worker(scope_a)
    worker_b = store.create_outbox_recovery_worker(scope_b)
    with pytest.raises(conversation.ConversationLimitError):
        await worker_a.claim(limit=0)
    with pytest.raises(conversation.ConversationLimitError):
        await worker_a.claim(limit=3)

    oldest_batch = await worker_a.claim(limit=1)
    assert oldest_batch.disposition is (
        conversation.OutboxRecoveryDisposition.CLAIMED
    )
    assert tuple(item.intent for item in oldest_batch.records) == (
        a_oldest.intent,
    )
    claimed_oldest = oldest_batch.records[0]
    assert claimed_oldest.lease_expires_at == NOW + timedelta(seconds=30)
    assert (
        await store.claim_outbox(_outbox_target(a_oldest, scope=scope_a))
    ).disposition is conversation.OutboxClaimDisposition.ACTIVELY_LEASED
    with pytest.raises(conversation.ConversationConflictError):
        await worker_b.acknowledge(claimed_oldest)

    b_batch = await worker_b.claim(limit=2)
    assert tuple(item.intent for item in b_batch.records) == (b_only.intent,)
    await worker_b.acknowledge(b_batch.records[0])
    assert (await worker_b.claim(limit=2)).disposition is (
        conversation.OutboxRecoveryDisposition.EMPTY
    )

    next_batch = await worker_a.claim(limit=1)
    assert tuple(item.intent for item in next_batch.records) == (
        a_next.intent,
    )
    await worker_a.release(next_batch.records[0])
    fair_batch = await worker_a.claim(limit=2)
    assert tuple(item.intent for item in fair_batch.records) == (
        a_last.intent,
        a_next.intent,
    )
    await worker_a.acknowledge(fair_batch.records[0])
    await worker_a.release(fair_batch.records[1])

    publisher = conversation.DeterministicFakePublisher()
    await publisher.publish(claimed_oldest.intent)
    await worker_a.acknowledge(claimed_oldest)
    assert publisher.published == (claimed_oldest.intent,)
    pending = await worker_a.claim(limit=2)
    assert tuple(item.intent for item in pending.records) == (a_next.intent,)
    assert pending.records[0].attempts == 3
    clock.set(NOW + timedelta(seconds=31))
    expired = await worker_a.claim(limit=1)
    assert tuple(item.intent for item in expired.records) == (a_next.intent,)
    assert expired.records[0].attempts == 4
    with pytest.raises(conversation.ConversationConflictError):
        await worker_a.release(pending.records[0])
    await publisher.publish(expired.records[0].intent)
    await worker_a.acknowledge(expired.records[0])
    await worker_a.acknowledge(expired.records[0])
    assert len(publisher.published) == 2
    assert (await worker_a.claim(limit=2)).disposition is (
        conversation.OutboxRecoveryDisposition.EMPTY
    )


async def test_generic_recovery_release_rotates_poison_behind_ready_work() -> (
    None
):
    """Serve every ready peer before retrying one released poison record."""
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW),
        limits=conversation.StoreLimits(max_page_size=4),
    )
    scope = authority("release-fairness-principal")

    async def commit(suffix: str) -> conversation.OutboxRecord:
        prepared = await _prepare_atomic(
            store,
            _atomic_commit(suffix, scope=scope),
        )
        receipt = await store.commit_atomic(prepared)
        assert receipt.outbox is not None
        return receipt.outbox

    poison, second, third, fourth = tuple(
        [
            await commit(f"release-fairness-{suffix}")
            for suffix in ("poison", "second", "third", "fourth")
        ]
    )
    first_worker = store.create_outbox_recovery_worker(scope)
    second_worker = store.create_outbox_recovery_worker(scope)
    first = await first_worker.claim(limit=1)
    assert tuple(item.intent for item in first.records) == (poison.intent,)
    await first_worker.release(first.records[0])

    peers = await second_worker.claim(limit=2)
    assert tuple(item.intent for item in peers.records) == (
        second.intent,
        third.intent,
    )
    publisher = conversation.DeterministicFakePublisher()
    for record in peers.records:
        await publisher.publish(record.intent)
        await second_worker.acknowledge(record)

    tail = await first_worker.claim(limit=2)
    assert tuple(item.intent for item in tail.records) == (
        fourth.intent,
        poison.intent,
    )
    assert tail.records[1].attempts == 2
    for record in tail.records:
        await publisher.publish(record.intent)
        await first_worker.acknowledge(record)
        await first_worker.acknowledge(record)
    assert publisher.published == (
        second.intent,
        third.intent,
        fourth.intent,
        poison.intent,
    )
    assert not store._outbox_ready_order
    assert (await first_worker.claim(limit=4)).disposition is (
        conversation.OutboxRecoveryDisposition.EMPTY
    )
    assert (await store.prune(NOW, limit=4)).outbox_records == 4
    assert store.diagnostics.outbox_records == 0
    assert not store._outbox_ready_order


async def test_generic_recovery_expiry_rotates_targeted_poison() -> None:
    """Rotate an expired targeted lease behind all generic ready peers."""
    clock = conversation.DeterministicFakeClock(NOW)
    store = conversation.InMemoryConversationStore(
        clock=clock,
        limits=conversation.StoreLimits(
            max_page_size=3,
            outbox_lease_seconds=1,
        ),
    )
    scope = authority("expiry-fairness-principal")

    async def commit(suffix: str) -> conversation.OutboxRecord:
        prepared = await _prepare_atomic(
            store,
            _atomic_commit(suffix, scope=scope),
        )
        receipt = await store.commit_atomic(prepared)
        assert receipt.outbox is not None
        return receipt.outbox

    poison, second, third = tuple(
        [
            await commit(f"expiry-fairness-{suffix}")
            for suffix in ("poison", "second", "third")
        ]
    )
    targeted = _claimed_record(
        await store.claim_outbox(_outbox_target(poison, scope=scope))
    )
    assert targeted.attempts == 1
    clock.set(NOW + timedelta(seconds=2))

    worker = store.create_outbox_recovery_worker(scope)
    peers = await worker.claim(limit=2)
    assert tuple(item.intent for item in peers.records) == (
        second.intent,
        third.intent,
    )
    for record in peers.records:
        await worker.acknowledge(record)
    retried = await worker.claim(limit=1)
    assert tuple(item.intent for item in retried.records) == (poison.intent,)
    assert retried.records[0].attempts == 2
    with pytest.raises(conversation.ConversationConflictError):
        await store.release_outbox(
            _outbox_target(poison, scope=scope),
            cast(str, targeted.lease_owner),
        )
    await worker.acknowledge(retried.records[0])
    assert (await worker.claim(limit=3)).disposition is (
        conversation.OutboxRecoveryDisposition.EMPTY
    )


async def test_generic_outbox_recovery_cancellation_is_pre_effect() -> None:
    """Leave pending work untouched when a generic recovery scan cancels."""
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:outbox_recovery_claim",
                exception=CancelledError(),
            ),
        )
    )
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW),
        boundary_hook=conversation.FakeStoreBoundaryHook(controller),
    )
    commit = await _prepare_atomic(store, _atomic_commit("recovery-cancel"))
    receipt = await store.commit_atomic(commit)
    assert receipt.outbox is not None
    with pytest.raises(conversation.ConversationValidationError):
        store.create_outbox_recovery_worker(
            cast(conversation.AuthorityScope, object())
        )
    worker = store.create_outbox_recovery_worker(authority())
    with pytest.raises(conversation.ConversationValidationError):
        await store._claim_pending_outbox(
            cast(conversation.AuthorityScope, object()),
            limit=1,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await worker.acknowledge(receipt.outbox)
    before = store.diagnostics
    with pytest.raises(CancelledError):
        await worker.claim(limit=1)
    assert store.diagnostics == before
    recovered = await worker.claim(limit=1)
    assert recovered.records[0].intent == receipt.outbox.intent
    assert recovered.records[0].attempts == 1
    await worker.release(recovered.records[0])
    await store.close()
    with pytest.raises(conversation.ConversationStorageError):
        await worker.claim(limit=1)


async def test_recovery_order_and_terminal_settlement_reject_corruption() -> (
    None
):
    """Reject missing recovery checkpoints and terminal provisional drift."""
    store = conversation.InMemoryConversationStore()
    prepared = await _prepare_atomic(
        store,
        _atomic_commit("recovery-order-corruption"),
    )
    receipt = await store.commit_atomic(prepared)
    assert receipt.outbox is not None
    checkpoint_id = receipt.outbox.intent.checkpoint_id
    stored = store._checkpoints.pop(checkpoint_id)
    with pytest.raises(conversation.ConversationStorageError):
        store._outbox_recovery_order_locked(receipt.outbox)
    store._checkpoints[checkpoint_id] = stored
    object.__setattr__(stored.checkpoint.timestamps, "committed_at", None)
    with pytest.raises(conversation.ConversationStorageError):
        store._outbox_recovery_order_locked(receipt.outbox)
    object.__setattr__(stored.checkpoint.timestamps, "committed_at", NOW)
    ready_order = store._outbox_ready_order.pop(
        receipt.outbox.intent.intent_id
    )
    with pytest.raises(conversation.ConversationStorageError):
        store._outbox_recovery_order_locked(receipt.outbox)
    store._outbox_ready_order[receipt.outbox.intent.intent_id] = ready_order
    with pytest.raises(conversation.ConversationStorageError):
        store._append_outbox_ready_locked(receipt.outbox.intent.intent_id)

    missing = store._outbox.pop(receipt.outbox.intent.intent_id)
    with pytest.raises(conversation.ConversationStorageError):
        store._requeue_outbox_ready_locked(receipt.outbox.intent.intent_id)
    store._outbox[receipt.outbox.intent.intent_id] = replace(
        missing,
        state=conversation.OutboxState.PUBLISHED,
        published_at=NOW,
    )
    with pytest.raises(conversation.ConversationStorageError):
        store._requeue_outbox_ready_locked(receipt.outbox.intent.intent_id)
    store._outbox[receipt.outbox.intent.intent_id] = missing

    assert prepared.provisional_response_id is not None
    assert prepared.public_response_id is not None
    store._provisional[prepared.provisional_response_id] = (
        conversation.ProvisionalPublicResponse(
            provisional_response_id=prepared.provisional_response_id,
            public_response_id=prepared.public_response_id,
            owner_token=prepared.owner_token,
            authority_digest=str(
                conversation.authority_digest(prepared.idempotency.authority)
            ),
        )
    )
    settlement = await store.inspect_idempotency_settlement(
        prepared.idempotency,
        prepared.owner_token,
    )
    assert settlement.disposition is (
        conversation.IdempotencySettlementDisposition.OWNERSHIP_CONFLICT
    )


async def test_reconcile_and_outbox_settlement_defensive_paths() -> None:
    """Keep reconciliation and repeated settlement closed and idempotent."""
    commit = _atomic_commit("reconcile-defensive")
    empty_store = conversation.InMemoryConversationStore()
    with pytest.raises(conversation.ConversationValidationError):
        await empty_store.inspect_idempotency_settlement(
            cast(conversation.RequestIdempotencyIdentity, object()),
            "owner",
        )
    with pytest.raises(conversation.ConversationValidationError):
        await empty_store.reconcile_idempotency(
            cast(conversation.RequestIdempotencyIdentity, object()),
            "owner",
            ambiguous=False,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await empty_store.reconcile_idempotency(
            commit.idempotency,
            "owner",
            ambiguous=cast(bool, 1),
        )
    await empty_store.reconcile_idempotency(
        commit.idempotency,
        "owner",
        ambiguous=False,
    )
    empty_store._hook = conversation.FakeStoreBoundaryHook(
        conversation.DeterministicFaultController(
            (
                conversation.FaultAction(
                    label="store:idempotency_reconcile",
                    exception=CancelledError(),
                ),
            )
        )
    )
    with pytest.raises(CancelledError):
        await empty_store.reconcile_idempotency(
            commit.idempotency,
            "owner",
            ambiguous=False,
        )

    owned_store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    owner = await owned_store.reserve_idempotency(commit.idempotency)
    assert owner.owner_token is not None
    drifted_identity = replace(
        commit.idempotency,
        request_digest=conversation.CanonicalRequestDigest(
            "reconcile-defensive-drift"
        ),
    )
    assert (
        await owned_store.inspect_idempotency_settlement(
            drifted_identity,
            owner.owner_token,
        )
    ).disposition is (
        conversation.IdempotencySettlementDisposition.OWNERSHIP_CONFLICT
    )
    await owned_store.reconcile_idempotency(
        commit.idempotency,
        owner.owner_token,
        ambiguous=False,
    )
    assert owned_store.diagnostics.idempotency_records == 0

    outbox_store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    prepared = await _prepare_atomic(
        outbox_store,
        _atomic_commit("settlement-defensive"),
    )
    receipt = await outbox_store.commit_atomic(prepared)
    assert receipt.outbox is not None
    target = _outbox_target(receipt.outbox)
    with pytest.raises(conversation.ConversationValidationError):
        await outbox_store.acknowledge_outbox(
            cast(conversation.OutboxClaimTarget, object()),
            "owner",
        )
    with pytest.raises(conversation.ConversationValidationError):
        await outbox_store.release_outbox(
            cast(conversation.OutboxClaimTarget, object()),
            "owner",
        )
    with pytest.raises(conversation.ConversationConflictError):
        await outbox_store.acknowledge_outbox(target, "owner")
    claimed = _claimed_record(await outbox_store.claim_outbox(target))
    assert claimed.lease_owner is not None
    await outbox_store.acknowledge_outbox(target, claimed.lease_owner)
    await outbox_store.acknowledge_outbox(target, claimed.lease_owner)

    outbox_store._hook = conversation.FakeStoreBoundaryHook(
        conversation.DeterministicFaultController(
            (
                conversation.FaultAction(
                    label="store:outbox_acknowledge",
                    exception=CancelledError(),
                ),
                conversation.FaultAction(
                    label="store:outbox_release",
                    exception=CancelledError(),
                ),
            )
        )
    )
    with pytest.raises(CancelledError):
        await outbox_store.acknowledge_outbox(target, claimed.lease_owner)
    with pytest.raises(CancelledError):
        await outbox_store.release_outbox(target, claimed.lease_owner)


async def test_max_one_operational_capacity_recovers_after_retirement() -> (
    None
):
    """Recover max-one slots and reject one-over atomically."""
    limits = conversation.StoreLimits(
        max_checkpoints=1,
        max_idempotency_records=1,
        max_provisional_responses=1,
        max_public_responses=1,
        max_outbox_records=1,
    )
    store = conversation.InMemoryConversationStore(
        limits=limits,
        clock=conversation.DeterministicFakeClock(NOW),
    )
    first = await _prepare_atomic(store, _atomic_commit("max-one-first"))
    first_receipt = await store.commit_atomic(first)
    with pytest.raises(conversation.ConversationLimitError):
        await _prepare_atomic(store, _atomic_commit("max-one-blocked"))
    assert store.diagnostics.idempotency_records == 1
    assert store.diagnostics.provisional_responses == 0
    assert first_receipt.outbox is not None
    target = _outbox_target(first_receipt.outbox)
    claimed = _claimed_record(await store.claim_outbox(target))
    assert claimed.lease_owner is not None
    await store.acknowledge_outbox(
        target,
        claimed.lease_owner,
    )
    assert (await store.prune(NOW, limit=1)).outbox_records == 1
    assert first.public_response_id is not None
    await store.tombstone(
        first.public_response_id,
        authority(),
        NOW + timedelta(seconds=1),
    )
    await store.delete(
        first.public_response_id,
        authority(),
        NOW + timedelta(seconds=2),
    )
    assert store.diagnostics.checkpoints == 0
    assert store.diagnostics.public_responses == 0
    assert store.diagnostics.idempotency_records == 0
    assert store.diagnostics.output_records == 0
    second = await _prepare_atomic(store, _atomic_commit("max-one-second"))
    await store.commit_atomic(second)
    assert store.diagnostics.checkpoints == 1

    one_over_store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(
            max_idempotency_records=2,
            max_provisional_responses=1,
            max_public_responses=2,
            max_outbox_records=2,
        )
    )
    owner = await _prepare_atomic(
        one_over_store,
        _atomic_commit("one-over-owner"),
    )
    with pytest.raises(conversation.ConversationLimitError):
        await _prepare_atomic(
            one_over_store,
            _atomic_commit("one-over-contender"),
        )
    assert one_over_store.diagnostics.provisional_responses == 1
    assert one_over_store.diagnostics.idempotency_records == 1
    await one_over_store.abandon_idempotency(
        owner.idempotency,
        owner.owner_token,
        ambiguous=False,
    )
    assert one_over_store.diagnostics.provisional_responses == 0
    assert one_over_store.diagnostics.idempotency_records == 0


@pytest.mark.parametrize(
    "boundary",
    tuple(
        item
        for item in conversation.StoreAwaitBoundary
        if item
        not in {
            conversation.StoreAwaitBoundary.IDEMPOTENCY_RECONCILE_BEGIN,
            conversation.StoreAwaitBoundary.ROLLBACK,
            conversation.StoreAwaitBoundary.ROLLBACK_BEGIN,
            conversation.StoreAwaitBoundary.ROLLBACK_SETTLED,
            conversation.StoreAwaitBoundary.IDEMPOTENCY_RECONCILE,
            conversation.StoreAwaitBoundary.IDEMPOTENCY_RECONCILE_SETTLED,
            conversation.StoreAwaitBoundary.IDEMPOTENCY_SETTLEMENT,
            conversation.StoreAwaitBoundary.OUTBOX_ACKNOWLEDGE,
            conversation.StoreAwaitBoundary.OUTBOX_RELEASE,
            conversation.StoreAwaitBoundary.CLOSE_BEGIN,
            conversation.StoreAwaitBoundary.CLOSE,
            conversation.StoreAwaitBoundary.CLOSE_SETTLED,
            conversation.StoreAwaitBoundary.CLOSE_STATUS,
        }
    ),
)
async def test_cancellation_at_every_store_boundary_is_state_safe(
    boundary: conversation.StoreAwaitBoundary,
) -> None:
    """Cancel every pre-effect store boundary without mutating state."""
    store = conversation.InMemoryConversationStore()
    base_commit = await _prepare_atomic(
        store,
        _atomic_commit(f"boundary-base-{boundary.value}"),
    )
    base = await store.commit_atomic(base_commit)
    assert base.result is not None
    output = base.output_candidates[0]
    first_result = conversation.ProviderResult(
        items=output.completed_items,
        reasoning=output.reasoning,
        usage=output.usage,
    )
    candidate = _child_candidate(
        base.checkpoint,
        first_result,
        suffix=f"boundary-child-{boundary.value}",
    )
    head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId(f"head-{boundary.value}"),
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=base.checkpoint.identity.checkpoint_id,
    )
    await store.create_head(head, authority())
    pending = _atomic_commit(f"boundary-pending-{boundary.value}")
    allocation: conversation.ProvisionalPublicResponse | None = None
    execution_stage: conversation.ProviderLaneExecutionStage | None = None
    if boundary is conversation.StoreAwaitBoundary.COMMIT_ATOMIC:
        pending = await _prepare_atomic(store, pending)
    elif boundary is conversation.StoreAwaitBoundary.EXECUTION_STAGE:
        reservation = _execution_reservation(pending)
        resolution = await store.reserve_idempotency(
            pending.idempotency,
            execution=reservation,
        )
        assert resolution.owner_token is not None
        output = pending.output_candidates[0]
        execution_stage = conversation.ProviderLaneExecutionStage(
            idempotency=pending.idempotency,
            owner_token=resolution.owner_token,
            identity=pending.candidate.checkpoint.identity,
            binding=output.binding,
            mode=output.mode,
            scope=output.scope,
            completed_items=output.completed_items,
            reasoning=output.reasoning,
            usage=output.usage,
            execution_receipt=output.execution_receipt,
            upstream_response_id=output.upstream_response_id,
        )
    elif boundary is conversation.StoreAwaitBoundary.ALLOCATE:
        resolution = await store.reserve_idempotency(pending.idempotency)
        assert resolution.owner_token is not None
        assert pending.provisional_response_id is not None
        assert pending.public_response_id is not None
        allocation = conversation.ProvisionalPublicResponse(
            provisional_response_id=pending.provisional_response_id,
            public_response_id=pending.public_response_id,
            owner_token=resolution.owner_token,
            authority_digest=str(
                conversation.authority_digest(pending.idempotency.authority)
            ),
        )
    if boundary is conversation.StoreAwaitBoundary.DELETE:
        assert base_commit.public_response_id is not None
        await store.tombstone(
            base_commit.public_response_id,
            authority(),
            NOW + timedelta(seconds=1),
        )

    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label=f"store:{boundary.value}",
                exception=CancelledError(),
            ),
        )
    )
    store._hook = conversation.FakeStoreBoundaryHook(controller)
    before = store.diagnostics
    with pytest.raises(CancelledError):
        match boundary:
            case conversation.StoreAwaitBoundary.CREATE:
                await store.create(candidate)
            case conversation.StoreAwaitBoundary.LOAD:
                await store.load(
                    base.checkpoint.identity.checkpoint_id, authority()
                )
            case conversation.StoreAwaitBoundary.AUTHORIZE:
                await store.authorize(
                    base.checkpoint.identity.checkpoint_id, authority()
                )
            case conversation.StoreAwaitBoundary.STAGE:
                await store.stage(candidate)
            case conversation.StoreAwaitBoundary.COMMIT:
                await store.commit(candidate)
            case conversation.StoreAwaitBoundary.COMMIT_ATOMIC:
                await store.commit_atomic(pending)
            case conversation.StoreAwaitBoundary.EXECUTION_STAGE:
                assert execution_stage is not None
                await store.stage_execution(execution_stage)
            case conversation.StoreAwaitBoundary.HEAD:
                await store.load_head(head.head_id, authority())
            case conversation.StoreAwaitBoundary.BRANCH:
                await store.branch_count(
                    base.checkpoint.identity.checkpoint_id, authority()
                )
            case conversation.StoreAwaitBoundary.IDEMPOTENCY:
                await store.reserve_idempotency(pending.idempotency)
            case conversation.StoreAwaitBoundary.ALLOCATE:
                assert allocation is not None
                await store.allocate_public_response(allocation)
            case conversation.StoreAwaitBoundary.RETRIEVE:
                assert base_commit.public_response_id is not None
                await store.retrieve(
                    base_commit.public_response_id, authority()
                )
            case conversation.StoreAwaitBoundary.RETRIEVE_OUTPUTS:
                await store.retrieve_output_candidates(
                    base.checkpoint.identity.checkpoint_id, authority()
                )
            case conversation.StoreAwaitBoundary.PREPARE_DELETE:
                assert base_commit.public_response_id is not None
                await store.prepare_deletion(
                    base_commit.public_response_id, authority()
                )
            case conversation.StoreAwaitBoundary.TOMBSTONE:
                assert base_commit.public_response_id is not None
                await store.tombstone(
                    base_commit.public_response_id,
                    authority(),
                    NOW + timedelta(seconds=1),
                )
            case conversation.StoreAwaitBoundary.DELETE:
                assert base_commit.public_response_id is not None
                await store.delete(
                    base_commit.public_response_id,
                    authority(),
                    NOW + timedelta(seconds=2),
                )
            case conversation.StoreAwaitBoundary.LIST:
                await store.list_checkpoints(authority(), cursor=None, limit=1)
            case conversation.StoreAwaitBoundary.SWEEP:
                await store.sweep(NOW, limit=1)
            case conversation.StoreAwaitBoundary.PRUNE:
                await store.prune(NOW, limit=1)
            case conversation.StoreAwaitBoundary.OUTBOX_CLAIM:
                assert base.outbox is not None
                await store.claim_outbox(_outbox_target(base.outbox))
            case conversation.StoreAwaitBoundary.OUTBOX_RECOVERY_CLAIM:
                worker = store.create_outbox_recovery_worker(authority())
                await worker.claim(limit=1)
            case _:
                raise AssertionError("special boundary has a dedicated test")
    assert store.diagnostics == before
    assert not store.diagnostics.locked


async def test_store_rejects_invalid_calls_and_state_transitions() -> None:
    """Fail closed for invalid values, duplicate state, and corrupt records."""
    commit = _atomic_commit("validation-store")
    candidate = commit.candidate
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore(
            limits=cast(conversation.StoreLimits, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore(
            codec=cast(conversation.ConversationCheckpointCodec, object())
        )

    store = conversation.InMemoryConversationStore()
    unit = await store.stage(candidate)
    async with unit:
        pass
    with pytest.raises(conversation.ConversationStorageError):
        await unit.__aenter__()
    unit = await store.stage(candidate)
    committed = await unit.commit()
    with pytest.raises(conversation.ConversationStorageError):
        await unit.commit()
    with pytest.raises(conversation.ConversationTransitionError):
        store._committed_checkpoint(committed, NOW)
    with pytest.raises(conversation.ConversationValidationError):
        await store.commit(cast(conversation.CheckpointCandidate, object()))
    with pytest.raises(conversation.ConversationValidationError):
        await store.commit_atomic(
            cast(conversation.AtomicConversationCommit, object())
        )
    with pytest.raises(conversation.ConversationConflictError):
        await store.commit_atomic(_atomic_commit("unreserved"))

    scope = authority()
    checkpoint_id = committed.identity.checkpoint_id
    head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("validation-head"),
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=checkpoint_id,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await store.create_head(
            cast(conversation.NamedHeadSnapshot, object()), scope
        )
    await store.create_head(head, scope)
    with pytest.raises(conversation.ConversationConflictError):
        await store.create_head(head, scope)
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load_head(conversation.NamedHeadId("missing-head"), scope)
    with pytest.raises(conversation.ConversationValidationError):
        await store.reserve_idempotency(
            cast(conversation.RequestIdempotencyIdentity, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        await store.fence_idempotency(
            commit.idempotency,
            commit.owner_token,
            ambiguous=cast(bool, 1),
        )
    with pytest.raises(conversation.ConversationConflictError):
        await store.fence_idempotency(
            commit.idempotency,
            "wrong-owner",
            ambiguous=False,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await store.abandon_idempotency(
            cast(conversation.RequestIdempotencyIdentity, object()),
            "owner",
            ambiguous=False,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await store.abandon_idempotency(
            commit.idempotency,
            "owner",
            ambiguous=cast(bool, 1),
        )
    with pytest.raises(conversation.ConversationConflictError):
        await store.abandon_idempotency(
            commit.idempotency,
            "wrong-owner",
            ambiguous=False,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await store.allocate_public_response(
            cast(conversation.ProvisionalPublicResponse, object())
        )
    assert commit.provisional_response_id is not None
    assert commit.public_response_id is not None
    resolution = await store.reserve_idempotency(commit.idempotency)
    assert resolution.owner_token is not None
    allocation = conversation.ProvisionalPublicResponse(
        provisional_response_id=commit.provisional_response_id,
        public_response_id=commit.public_response_id,
        owner_token=resolution.owner_token,
        authority_digest=str(conversation.authority_digest(scope)),
    )
    await store.allocate_public_response(allocation)
    with pytest.raises(conversation.ConversationConflictError):
        await store.allocate_public_response(allocation)
    with pytest.raises(conversation.ConversationConflictError):
        await store.allocate_public_response(
            replace(
                allocation,
                provisional_response_id=conversation.ProvisionalResponseId(
                    "wrong-authority-provisional"
                ),
                public_response_id=conversation.PublicResponseId(
                    "wrong-authority-response"
                ),
                authority_digest=str(
                    conversation.authority_digest(authority("wrong"))
                ),
            )
        )
    with pytest.raises(conversation.ConversationValidationError):
        await store.rollback_attempt("")
    await store.rollback_attempt(resolution.owner_token)
    assert store.diagnostics.provisional_responses == 0
    with pytest.raises(conversation.ConversationValidationError):
        await store.list_checkpoints(
            cast(conversation.AuthorityScope, object()), cursor=None, limit=1
        )
    with pytest.raises(conversation.ConversationLimitError):
        await store.sweep(NOW, limit=0)
    with pytest.raises(conversation.ConversationLimitError):
        await store.prune(NOW, limit=0)
    with pytest.raises(conversation.ConversationValidationError):
        await store.load(
            checkpoint_id,
            cast(conversation.AuthorityScope, object()),
        )
    with pytest.raises(conversation.ConversationValidationError):
        await store.tombstone(
            conversation.PublicResponseId("missing"),
            scope,
            datetime.min,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await store.delete(
            conversation.PublicResponseId("missing"),
            scope,
            datetime.min,
        )
    missing_target = conversation.OutboxClaimTarget(
        authority=scope,
        checkpoint_id=conversation.CheckpointId("missing-checkpoint"),
        public_response_id=conversation.PublicResponseId("missing-response"),
        intent_id="missing-outbox",
    )
    with pytest.raises(conversation.ConversationConflictError):
        await store.acknowledge_outbox(missing_target, "missing-owner")
    with pytest.raises(conversation.ConversationConflictError):
        await store.release_outbox(missing_target, "missing-owner")
    with pytest.raises(conversation.ConversationValidationError):
        store._authorize_entry_locked(
            checkpoint_id, cast(conversation.AuthorityScope, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        store._authorized_public_locked(
            conversation.PublicResponseId("missing"),
            cast(conversation.AuthorityScope, object()),
            allow_tombstone=False,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await store.load_head(
            head.head_id,
            cast(conversation.AuthorityScope, object()),
        )


async def test_corrupt_state_is_concealed_and_delete_needs_tombstone() -> None:
    """Conceal missing content and reject deletion before tombstoning."""
    scope = authority()
    store = conversation.InMemoryConversationStore()
    commit = _atomic_commit("corrupt-result")
    commit = await _prepare_atomic(store, commit)
    await store.commit_atomic(commit)
    assert commit.public_response_id is not None
    checkpoint_id = commit.candidate.checkpoint.identity.checkpoint_id
    store._outputs.pop(checkpoint_id)
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.retrieve_output_candidates(checkpoint_id, scope)
    store._results.pop(commit.public_response_id)
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.retrieve(commit.public_response_id, scope)

    store = conversation.InMemoryConversationStore()
    commit = _atomic_commit("delete-live")
    commit = await _prepare_atomic(store, commit)
    await store.commit_atomic(commit)
    assert commit.public_response_id is not None
    with pytest.raises(conversation.ConversationTransitionError):
        await store.delete(
            commit.public_response_id,
            scope,
            NOW + timedelta(seconds=1),
        )

    store = conversation.InMemoryConversationStore()
    commit = _atomic_commit("corrupt-lifecycle")
    commit = await _prepare_atomic(store, commit)
    receipt = await store.commit_atomic(commit)
    checkpoint_id = receipt.checkpoint.identity.checkpoint_id
    stored = store._checkpoints[checkpoint_id]
    store._checkpoints[checkpoint_id] = replace(
        stored,
        checkpoint=replace(
            stored.checkpoint,
            lifecycle=conversation.CheckpointLifecycle.EXPIRED,
        ),
    )
    assert commit.public_response_id is not None
    with pytest.raises(conversation.ConversationTransitionError):
        await store.tombstone(
            commit.public_response_id,
            scope,
            NOW + timedelta(seconds=1),
        )

    store = conversation.InMemoryConversationStore()
    commit = _atomic_commit("missing-tombstone")
    commit = await _prepare_atomic(store, commit)
    receipt = await store.commit_atomic(commit)
    assert commit.public_response_id is not None
    await store.tombstone(
        commit.public_response_id,
        scope,
        NOW + timedelta(seconds=1),
    )
    store._checkpoints.pop(receipt.checkpoint.identity.checkpoint_id)
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.delete(
            commit.public_response_id,
            scope,
            NOW + timedelta(seconds=2),
        )


async def test_child_deletion_retires_parent_graph_edge() -> None:
    """Remove a deleted outward child from its retained parent's graph."""
    store = conversation.InMemoryConversationStore()
    scope, _engine, root, first_result = await _seed(
        store,
        suffix="delete-child-root",
    )
    lane_binding = binding()
    next_plan = next_stateless_plan(lane_binding, first_result.items)
    child_results = (
        conversation.fake_provider_result(next_plan, turn=2),
        conversation.fake_provider_result(next_plan, turn=3),
    )
    child_engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(_runtime(lane_binding, child_results),),
    )
    child_runs = tuple(
        request(
            scope=scope,
            identity=child_identity(root.checkpoint, f"delete-child-{suffix}"),
            advance=conversation.OrdinaryChildAdvance(
                parent_checkpoint_id=root.checkpoint.identity.checkpoint_id
            ),
            response_suffix=f"delete-child-{suffix}",
            key=f"key-delete-child-{suffix}",
        )
        for suffix in ("one", "two")
    )
    for child_run in child_runs:
        await child_engine.execute(child_run)
    child_run = child_runs[0]
    assert child_run.public_response_id is not None
    await store.tombstone(
        child_run.public_response_id,
        scope,
        NOW + timedelta(seconds=1),
    )
    await store.delete(
        child_run.public_response_id,
        scope,
        NOW + timedelta(seconds=2),
    )
    assert (
        await store.branch_count(root.checkpoint.identity.checkpoint_id, scope)
        == 1
    )
    second_child = child_runs[1]
    assert second_child.public_response_id is not None
    await store.tombstone(
        second_child.public_response_id,
        scope,
        NOW + timedelta(seconds=3),
    )
    await store.delete(
        second_child.public_response_id,
        scope,
        NOW + timedelta(seconds=4),
    )
    assert (
        await store.branch_count(root.checkpoint.identity.checkpoint_id, scope)
        == 0
    )


async def test_store_write_limits_and_graph_conflicts_are_exact() -> None:
    """Enforce byte, item, depth, parent, child, and duplicate limits."""
    base_commit = _atomic_commit("duplicate-checkpoint")
    candidate = base_commit.candidate
    store = conversation.InMemoryConversationStore()
    await store.create(candidate)
    with pytest.raises(conversation.ConversationConflictError):
        await store.create(candidate)

    store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(max_checkpoint_bytes=1)
    )
    with pytest.raises(conversation.ConversationLimitError):
        await store.create(_atomic_commit("byte-limit").candidate)

    store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(max_provider_items=1)
    )
    scope, _engine, root, first_result = await _seed(
        store, suffix="item-limit"
    )
    item_child = _child_candidate(
        root.checkpoint, first_result, suffix="item-limit-child"
    )
    with pytest.raises(conversation.ConversationLimitError):
        await store.create(item_child)
    assert await store.load(root.checkpoint.identity.checkpoint_id, scope)

    store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(max_depth=1)
    )
    scope, _engine, root, first_result = await _seed(
        store, suffix="depth-limit"
    )
    child = await store.create(
        _child_candidate(root.checkpoint, first_result, suffix="depth-child")
    )
    child_lane = child.content.lanes[0]
    assert isinstance(child_lane, conversation.StatelessProviderLaneSnapshot)
    child_result = conversation.ProviderResult(
        items=child_lane.ledger.items,
        reasoning=child_lane.reasoning,
    )
    with pytest.raises(conversation.ConversationLimitError):
        await store.create(
            _child_candidate(
                child,
                child_result,
                suffix="depth-grandchild",
                turn=3,
            )
        )
    assert await store.load(root.checkpoint.identity.checkpoint_id, scope)

    missing_parent = _child_candidate(
        root.checkpoint, first_result, suffix="missing-parent"
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await conversation.InMemoryConversationStore().create(missing_parent)

    store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(max_children_per_parent=1)
    )
    scope, _engine, root, first_result = await _seed(
        store, suffix="child-limit"
    )
    await store.create(
        _child_candidate(root.checkpoint, first_result, suffix="child-one")
    )
    with pytest.raises(conversation.ConversationLimitError):
        await store.create(
            _child_candidate(root.checkpoint, first_result, suffix="child-two")
        )
    assert (
        await store.branch_count(root.checkpoint.identity.checkpoint_id, scope)
        == 1
    )


def test_output_candidate_commit_validation_is_exact() -> None:
    """Reject lane, mode, suffix, reasoning, and upstream output drift."""
    commit = _atomic_commit("output-validation")
    checkpoint = commit.candidate.checkpoint
    output = commit.output_candidates[0]
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore._validate_output_candidates(
            checkpoint,
            (output, output),
        )
    cumulative_scope = conversation.ProviderLaneOutputScope.CUMULATIVE
    cumulative_receipt = conversation.provider_lane_execution_receipt(
        authority=checkpoint.authority,
        identity=checkpoint.identity,
        binding=output.binding,
        mode=output.mode,
        scope=cumulative_scope,
        completed_items=output.completed_items,
        reasoning=output.reasoning,
        usage=output.usage,
        upstream_response_id=None,
    )
    lane = checkpoint.content.lanes[0]
    assert isinstance(lane, conversation.StatelessProviderLaneSnapshot)
    cumulative_checkpoint = conversation.with_checkpoint_integrity(
        replace(
            checkpoint,
            content=replace(
                checkpoint.content,
                lanes=(replace(lane, execution_receipt=cumulative_receipt),),
            ),
        )
    )
    conversation.InMemoryConversationStore._validate_output_candidates(
        cumulative_checkpoint,
        (
            replace(
                output,
                scope=cumulative_scope,
                execution_receipt=cumulative_receipt,
            ),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore._validate_output_candidates(
            checkpoint,
            (
                replace(
                    output,
                    reasoning=conversation.EffectiveReasoningMetadata(
                        requested=conversation.ReasoningContext.CURRENT_TURN,
                        effective=(
                            conversation.EffectiveReasoningContext.CURRENT_TURN
                        ),
                    ),
                ),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore._validate_output_candidates(
            checkpoint,
            (
                replace(
                    output,
                    mode=conversation.ConversationMode.STORED,
                    upstream_response_id=conversation.UpstreamResponseId(
                        "unexpected-upstream"
                    ),
                ),
            ),
        )
    incompatible_prior_lane = conversation.StoredProviderLaneSnapshot(
        binding=lane.binding,
        upstream_response_id=conversation.UpstreamResponseId(
            "prior-stored-upstream"
        ),
        reasoning=lane.reasoning,
        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    incompatible_parent = replace(
        checkpoint,
        content=replace(
            checkpoint.content,
            lanes=(incompatible_prior_lane,),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore._validate_output_candidates(
            checkpoint,
            (output,),
            parent=incompatible_parent,
        )
    wrong_suffix = conversation.fake_provider_result(
        empty_stateless_plan(lane.binding),
        turn=9,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore._validate_output_candidates(
            checkpoint,
            (replace(output, completed_items=wrong_suffix.items),),
        )
    wrong_items_receipt = conversation.provider_lane_execution_receipt(
        authority=checkpoint.authority,
        identity=checkpoint.identity,
        binding=output.binding,
        mode=output.mode,
        scope=output.scope,
        completed_items=wrong_suffix.items,
        reasoning=output.reasoning,
        usage=output.usage,
        upstream_response_id=None,
    )
    wrong_items_checkpoint = replace(
        checkpoint,
        content=replace(
            checkpoint.content,
            lanes=(replace(lane, execution_receipt=wrong_items_receipt),),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore._validate_output_candidates(
            wrong_items_checkpoint,
            (
                replace(
                    output,
                    completed_items=wrong_suffix.items,
                    execution_receipt=wrong_items_receipt,
                ),
            ),
        )
    unexpected_upstream = conversation.UpstreamResponseId(
        "stateless-unexpected-upstream"
    )
    wrong_mode_receipt = conversation.provider_lane_execution_receipt(
        authority=checkpoint.authority,
        identity=checkpoint.identity,
        binding=output.binding,
        mode=conversation.ConversationMode.STORED,
        scope=output.scope,
        completed_items=output.completed_items,
        reasoning=output.reasoning,
        usage=output.usage,
        upstream_response_id=unexpected_upstream,
    )
    wrong_mode_checkpoint = replace(
        checkpoint,
        content=replace(
            checkpoint.content,
            lanes=(replace(lane, execution_receipt=wrong_mode_receipt),),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore._validate_output_candidates(
            wrong_mode_checkpoint,
            (
                replace(
                    output,
                    mode=conversation.ConversationMode.STORED,
                    upstream_response_id=unexpected_upstream,
                    execution_receipt=wrong_mode_receipt,
                ),
            ),
        )

    stored_binding = binding("lane-output-stored")
    stored_plan = first_stored_plan(stored_binding)
    stored_result = conversation.fake_provider_result(stored_plan, turn=1)
    assert stored_result.upstream_response_id is not None
    stored_lane = conversation.StoredProviderLaneSnapshot(
        binding=stored_binding,
        upstream_response_id=stored_result.upstream_response_id,
        reasoning=stored_result.reasoning,
        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    stored_run = request(
        scope=authority(),
        identity=root_identity("output-stored"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("lane-output-stored",),
        modes=(conversation.ConversationMode.STORED,),
        response_suffix="output-stored",
        key="key-output-stored",
        stored_retention=True,
    )
    stored_receipt = conversation.provider_lane_execution_receipt(
        authority=stored_run.semantics.authority,
        identity=stored_run.identity,
        binding=stored_binding,
        mode=conversation.ConversationMode.STORED,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=stored_result.items,
        reasoning=stored_result.reasoning,
        usage=stored_result.usage,
        upstream_response_id=stored_result.upstream_response_id,
    )
    stored_lane = replace(
        stored_lane,
        execution_receipt=stored_receipt,
    )
    stored_candidate = conversation.build_checkpoint_candidate(
        stored_run,
        parent=None,
        completed_lanes=(stored_lane,),
        created_at=NOW,
    )
    different_upstream = conversation.UpstreamResponseId("different-upstream")
    stored_output = conversation.ProviderLaneOutputCandidate(
        lane_id=stored_binding.lane_id,
        binding=stored_binding,
        mode=conversation.ConversationMode.STORED,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=stored_result.items,
        reasoning=stored_result.reasoning,
        usage=stored_result.usage,
        execution_receipt=conversation.provider_lane_execution_receipt(
            authority=stored_run.semantics.authority,
            identity=stored_run.identity,
            binding=stored_binding,
            mode=conversation.ConversationMode.STORED,
            scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
            completed_items=stored_result.items,
            reasoning=stored_result.reasoning,
            usage=stored_result.usage,
            upstream_response_id=different_upstream,
        ),
        upstream_response_id=different_upstream,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore._validate_output_candidates(
            stored_candidate.checkpoint,
            (stored_output,),
        )
    stored_as_stateless_receipt = conversation.provider_lane_execution_receipt(
        authority=stored_candidate.checkpoint.authority,
        identity=stored_candidate.checkpoint.identity,
        binding=stored_binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=stored_result.items,
        reasoning=stored_result.reasoning,
        usage=stored_result.usage,
        upstream_response_id=None,
    )
    stored_as_stateless_checkpoint = replace(
        stored_candidate.checkpoint,
        content=replace(
            stored_candidate.checkpoint.content,
            lanes=(
                replace(
                    stored_lane,
                    execution_receipt=stored_as_stateless_receipt,
                ),
            ),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore._validate_output_candidates(
            stored_as_stateless_checkpoint,
            (
                replace(
                    stored_output,
                    mode=conversation.ConversationMode.STATELESS,
                    upstream_response_id=None,
                    execution_receipt=stored_as_stateless_receipt,
                ),
            ),
        )


def test_execution_receipt_binds_every_exact_execution_dimension() -> None:
    """Change the receipt for every identity, item, and usage mutation."""
    commit = _two_item_atomic_commit("receipt-dimensions")
    checkpoint = commit.candidate.checkpoint
    output = commit.output_candidates[0]
    first, second = output.completed_items

    def receipt(
        *,
        items: tuple[conversation.ProviderItem, ...] = output.completed_items,
        lane_binding: conversation.ProviderLaneBinding = output.binding,
        identity: conversation.CheckpointIdentity = checkpoint.identity,
        scope: conversation.ProviderLaneOutputScope = output.scope,
        usage: conversation.ProviderUsage = output.usage,
        reasoning: conversation.EffectiveReasoningMetadata = output.reasoning,
        mode: conversation.ConversationMode = output.mode,
        upstream_response_id: conversation.UpstreamResponseId | None = None,
        scope_authority: conversation.AuthorityScope = checkpoint.authority,
    ) -> conversation.ProviderLaneExecutionReceipt:
        return conversation.provider_lane_execution_receipt(
            authority=scope_authority,
            identity=identity,
            binding=lane_binding,
            mode=mode,
            scope=scope,
            completed_items=items,
            reasoning=reasoning,
            usage=usage,
            upstream_response_id=upstream_response_id,
        )

    substituted = conversation.fake_provider_result(
        empty_stateless_plan(output.binding),
        turn=1,
        text="receipt-substitution",
    ).items[0]
    changed_id = conversation.ProviderItemId("receipt-altered-item")
    changed_input = dict(first.canonical_input)
    changed_input["id"] = changed_id
    altered_id = replace(
        first,
        item_id=changed_id,
        canonical_input=changed_input,
    )
    altered_type = conversation.ProviderItem(
        item_id=conversation.ProviderItemId("receipt-altered-type"),
        lane_id=first.lane_id,
        model_call_id=first.model_call_id,
        kind=conversation.ProviderItemKind.COMPACTION_TRIGGER,
        order=first.order,
        provider_index=first.provider_index,
        phase=conversation.ProviderItemPhase.INPUT,
        caller=conversation.ProviderItemCaller.CALLER,
        canonical_input={"type": "compaction_trigger"},
        normalization_version=first.normalization_version,
    )
    drifted_binding = replace(
        output.binding,
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("receipt-drift")
        ),
    )
    variants = (
        receipt(items=(substituted, second)),
        receipt(items=(second, first)),
        receipt(items=(first,)),
        receipt(items=(first, second, substituted)),
        receipt(items=(altered_id, second)),
        receipt(
            items=(
                replace(
                    first,
                    order=conversation.ProviderItemOrder(9),
                ),
                second,
            )
        ),
        receipt(items=(altered_type, second)),
        receipt(lane_binding=drifted_binding),
        receipt(
            identity=replace(
                checkpoint.identity,
                logical_turn_id=conversation.LogicalTurnId(
                    "receipt-different-turn"
                ),
            )
        ),
        receipt(
            usage=conversation.ProviderUsage(
                input_tokens=output.usage.input_tokens + 1,
                output_tokens=output.usage.output_tokens,
            )
        ),
        receipt(
            reasoning=conversation.EffectiveReasoningMetadata(
                requested=conversation.ReasoningContext.ALL_TURNS,
                effective=conversation.EffectiveReasoningContext.ALL_TURNS,
            )
        ),
        receipt(scope=conversation.ProviderLaneOutputScope.CUMULATIVE),
        receipt(scope_authority=authority("receipt-other-principal")),
        receipt(
            mode=conversation.ConversationMode.STORED,
            upstream_response_id=conversation.UpstreamResponseId(
                "receipt-upstream"
            ),
        ),
    )
    assert all(item != output.execution_receipt for item in variants)
    assert len({item.digest for item in variants}) == len(variants)
    with pytest.raises(conversation.ConversationValidationError):
        receipt(
            items=(
                replace(
                    first,
                    lane_id=conversation.ProviderLaneId("receipt-other-lane"),
                ),
                second,
            )
        )

    stored = _stored_atomic_commit("receipt-upstream-dimension")
    stored_output = stored.output_candidates[0]
    changed_upstream = conversation.provider_lane_execution_receipt(
        authority=stored.candidate.checkpoint.authority,
        identity=stored.candidate.checkpoint.identity,
        binding=stored_output.binding,
        mode=stored_output.mode,
        scope=stored_output.scope,
        completed_items=stored_output.completed_items,
        reasoning=stored_output.reasoning,
        usage=stored_output.usage,
        upstream_response_id=conversation.UpstreamResponseId(
            "receipt-changed-upstream"
        ),
    )
    assert changed_upstream != stored_output.execution_receipt
    rendered = repr(stored_output.execution_receipt)
    assert "receipt-upstream-dimension" not in rendered
    assert "fake-upstream" not in rendered


async def test_execution_receipt_tampering_is_atomic() -> None:
    """Reject every stale receipt before any authoritative store mutation."""
    base = _two_item_atomic_commit("receipt-atomic")
    output = base.output_candidates[0]
    first, second = output.completed_items
    substitute = conversation.fake_provider_result(
        empty_stateless_plan(output.binding),
        turn=1,
        text="receipt-atomic-substitute",
    ).items[0]
    changed_id = conversation.ProviderItemId("receipt-atomic-id")
    changed_input = dict(first.canonical_input)
    changed_input["id"] = changed_id
    changed_type = conversation.ProviderItem(
        item_id=conversation.ProviderItemId("receipt-atomic-type"),
        lane_id=first.lane_id,
        model_call_id=first.model_call_id,
        kind=conversation.ProviderItemKind.COMPACTION_TRIGGER,
        order=first.order,
        provider_index=first.provider_index,
        phase=conversation.ProviderItemPhase.INPUT,
        caller=conversation.ProviderItemCaller.CALLER,
        canonical_input={"type": "compaction_trigger"},
        normalization_version=first.normalization_version,
    )
    third = conversation.fake_provider_result(
        next_stateless_plan(output.binding, output.completed_items),
        turn=3,
        text="receipt-atomic-extra",
    ).items[0]
    mutations: tuple[tuple[str, str, object], ...] = (
        ("substitute", "completed_items", (substitute, second)),
        ("reorder", "completed_items", (second, first)),
        ("omit", "completed_items", (first,)),
        ("extra", "completed_items", (first, second, third)),
        (
            "identifier",
            "completed_items",
            (
                replace(
                    first,
                    item_id=changed_id,
                    canonical_input=changed_input,
                ),
                second,
            ),
        ),
        (
            "order",
            "completed_items",
            (
                replace(
                    first,
                    order=conversation.ProviderItemOrder(8),
                ),
                second,
            ),
        ),
        ("type", "completed_items", (changed_type, second)),
        (
            "lane",
            "completed_items",
            (
                replace(
                    first,
                    lane_id=conversation.ProviderLaneId(
                        "receipt-atomic-other-lane"
                    ),
                ),
                second,
            ),
        ),
        (
            "usage",
            "usage",
            conversation.ProviderUsage(
                input_tokens=output.usage.input_tokens,
                output_tokens=output.usage.output_tokens + 1,
            ),
        ),
    )

    async def reject(
        suffix: str,
        staged_commit: conversation.AtomicConversationCommit,
        tampered_commit: conversation.AtomicConversationCommit,
    ) -> None:
        store = conversation.InMemoryConversationStore()
        prepared = await _prepare_atomic(store, staged_commit)
        tampered = copy(tampered_commit)
        object.__setattr__(tampered, "owner_token", prepared.owner_token)
        object.__setattr__(
            tampered,
            "execution_attestations",
            prepared.execution_attestations,
        )
        before = _atomic_store_snapshot(store)
        with pytest.raises(conversation.ConversationValidationError):
            await store.commit_atomic(tampered)
        assert _atomic_store_snapshot(store) == before
        settlement = await store.inspect_idempotency_settlement(
            prepared.idempotency,
            prepared.owner_token,
        )
        assert settlement.disposition is (
            conversation.IdempotencySettlementDisposition.LEASED
        ), suffix
        await store.abandon_idempotency(
            prepared.idempotency,
            prepared.owner_token,
            ambiguous=False,
        )

    for suffix, field, value in mutations:
        mutated_output = copy(output)
        object.__setattr__(mutated_output, field, value)
        tampered = copy(base)
        object.__setattr__(tampered, "output_candidates", (mutated_output,))
        await reject(suffix, base, tampered)

    lane = base.candidate.checkpoint.content.lanes[0]
    assert isinstance(lane, conversation.StatelessProviderLaneSnapshot)
    drifted_binding = replace(
        lane.binding,
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("receipt-atomic-drift")
        ),
    )
    drifted_lane = replace(lane, binding=drifted_binding)
    drifted_checkpoint = conversation.with_checkpoint_integrity(
        replace(
            base.candidate.checkpoint,
            content=replace(
                base.candidate.checkpoint.content,
                lanes=(drifted_lane,),
            ),
        )
    )
    drifted_output = copy(output)
    object.__setattr__(drifted_output, "binding", drifted_binding)
    drifted_commit = replace(
        base,
        candidate=replace(base.candidate, checkpoint=drifted_checkpoint),
        output_candidates=(drifted_output,),
    )
    await reject("binding", base, drifted_commit)

    changed_turn_checkpoint = conversation.with_checkpoint_integrity(
        replace(
            base.candidate.checkpoint,
            identity=replace(
                base.candidate.checkpoint.identity,
                logical_turn_id=conversation.LogicalTurnId(
                    "receipt-atomic-other-turn"
                ),
            ),
        )
    )
    changed_turn = replace(
        base,
        candidate=replace(
            base.candidate,
            checkpoint=changed_turn_checkpoint,
        ),
    )
    await reject("turn", base, changed_turn)

    stored = _stored_atomic_commit("receipt-atomic-upstream")
    stored_output = copy(stored.output_candidates[0])
    object.__setattr__(
        stored_output,
        "upstream_response_id",
        conversation.UpstreamResponseId("receipt-atomic-wrong-upstream"),
    )
    tampered_stored = copy(stored)
    object.__setattr__(
        tampered_stored,
        "output_candidates",
        (stored_output,),
    )
    await reject("upstream", stored, tampered_stored)


async def test_execution_staging_is_owner_bound_and_one_time() -> None:
    """Reject forged, stale, recomputed, and owner-drifted attestations."""
    store = conversation.InMemoryConversationStore()
    prepared = await _prepare_atomic(
        store,
        _two_item_atomic_commit("execution-staging"),
    )
    output = prepared.output_candidates[0]
    stage = _execution_stage(prepared, output, prepared.owner_token)
    with pytest.raises(conversation.ConversationValidationError):
        await store.stage_execution(
            cast(conversation.ProviderLaneExecutionStage, object())
        )
    tampered_stage = copy(stage)
    object.__setattr__(
        tampered_stage,
        "execution_receipt",
        replace(
            stage.execution_receipt,
            digest=conversation.IntegrityDigest("f" * 64),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await store.stage_execution(tampered_stage)
    with pytest.raises(conversation.ConversationConflictError):
        await store.stage_execution(stage)
    with pytest.raises(conversation.ConversationConflictError):
        await store.stage_execution(
            replace(stage, owner_token="different-execution-owner")
        )

    attestation = prepared.execution_attestations[0]
    forged = replace(
        attestation,
        staging_id="execution-stage-forged",
    )
    wrong_lane = replace(
        attestation,
        lane_id=conversation.ProviderLaneId("execution-stage-wrong-lane"),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(prepared, execution_attestations=(wrong_lane,))
    snapshot = _atomic_store_snapshot(store)
    drifted_identity = replace(
        prepared.candidate.checkpoint.identity,
        logical_turn_id=conversation.LogicalTurnId(
            "execution-staging-drifted-turn"
        ),
    )
    assert (
        await store.reserve_idempotency(
            prepared.idempotency,
            execution=replace(
                _execution_reservation(prepared),
                identity=drifted_identity,
            ),
        )
    ).disposition is conversation.IdempotencyDisposition.CONFLICT
    assert _atomic_store_snapshot(store) == snapshot
    for invalid in (
        replace(prepared, execution_attestations=()),
        replace(prepared, execution_attestations=(forged,)),
        replace(prepared, owner_token="different-execution-owner"),
    ):
        with pytest.raises(conversation.ConversationConflictError):
            await store.commit_atomic(invalid)
        assert _atomic_store_snapshot(store) == snapshot

    changed_usage = conversation.ProviderUsage(
        input_tokens=output.usage.input_tokens + 1,
        output_tokens=output.usage.output_tokens,
    )
    changed_receipt = conversation.provider_lane_execution_receipt(
        authority=prepared.candidate.checkpoint.authority,
        identity=prepared.candidate.checkpoint.identity,
        binding=output.binding,
        mode=output.mode,
        scope=output.scope,
        completed_items=output.completed_items,
        reasoning=output.reasoning,
        usage=changed_usage,
        upstream_response_id=output.upstream_response_id,
    )
    changed_output = replace(
        output,
        usage=changed_usage,
        execution_receipt=changed_receipt,
    )
    lane = prepared.candidate.checkpoint.content.lanes[0]
    assert isinstance(lane, conversation.StatelessProviderLaneSnapshot)
    changed_checkpoint = conversation.with_checkpoint_integrity(
        replace(
            prepared.candidate.checkpoint,
            content=replace(
                prepared.candidate.checkpoint.content,
                lanes=(replace(lane, execution_receipt=changed_receipt),),
            ),
        )
    )
    recomputed = replace(
        prepared,
        candidate=replace(
            prepared.candidate,
            checkpoint=changed_checkpoint,
        ),
        output_candidates=(changed_output,),
    )
    with pytest.raises(conversation.ConversationConflictError):
        await store.commit_atomic(recomputed)
    assert _atomic_store_snapshot(store) == snapshot

    receipt = await store.commit_atomic(prepared)
    assert (
        receipt.checkpoint.identity == prepared.candidate.checkpoint.identity
    )
    assert store.diagnostics.staged_executions == 0
    assert not store._execution_stage_keys
    with pytest.raises(conversation.ConversationConflictError):
        await store.commit_atomic(prepared)
    assert store.diagnostics.staged_executions == 0


async def test_execution_staging_requires_reservation_and_is_reclaimed() -> (
    None
):
    """Bound uncommitted staging and reclaim it at every owner terminal."""
    unstaged_store = conversation.InMemoryConversationStore()
    raw = _atomic_commit("execution-unstaged")
    resolution = await unstaged_store.reserve_idempotency(
        raw.idempotency,
        execution=_execution_reservation(raw),
    )
    assert resolution.owner_token is not None
    assert raw.provisional_response_id is not None
    assert raw.public_response_id is not None
    await unstaged_store.allocate_public_response(
        conversation.ProvisionalPublicResponse(
            provisional_response_id=raw.provisional_response_id,
            public_response_id=raw.public_response_id,
            owner_token=resolution.owner_token,
            authority_digest=str(
                conversation.authority_digest(raw.idempotency.authority)
            ),
        )
    )
    unstaged = replace(raw, owner_token=resolution.owner_token)
    snapshot = _atomic_store_snapshot(unstaged_store)
    with pytest.raises(conversation.ConversationConflictError):
        await unstaged_store.commit_atomic(unstaged)
    assert _atomic_store_snapshot(unstaged_store) == snapshot
    await unstaged_store.abandon_idempotency(
        raw.idempotency,
        resolution.owner_token,
        ambiguous=False,
    )

    limited = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(max_staged_execution_records=1)
    )
    first = await _prepare_atomic(
        limited,
        _atomic_commit("execution-stage-limit-first"),
    )
    assert limited.diagnostics.staged_executions == 1
    with pytest.raises(conversation.ConversationLimitError):
        await _prepare_atomic(
            limited,
            _atomic_commit("execution-stage-limit-second"),
        )
    assert limited.diagnostics.staged_executions == 1
    await limited.abandon_idempotency(
        first.idempotency,
        first.owner_token,
        ambiguous=False,
    )
    assert limited.diagnostics.staged_executions == 0

    clock = conversation.DeterministicFakeClock(NOW)
    expiring = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(idempotency_lease_seconds=1),
        clock=clock,
    )
    expiring_commit = await _prepare_atomic(
        expiring,
        _atomic_commit("execution-stage-expiry"),
    )
    assert expiring.diagnostics.staged_executions == 1
    clock.set(NOW + timedelta(seconds=2))
    assert (
        await expiring.reserve_idempotency(
            expiring_commit.idempotency,
            execution=_execution_reservation(expiring_commit),
        )
    ).disposition is conversation.IdempotencyDisposition.FENCED
    assert expiring.diagnostics.staged_executions == 0
    assert expiring.diagnostics.provisional_responses == 0

    closing = conversation.InMemoryConversationStore()
    await _prepare_atomic(
        closing,
        _atomic_commit("execution-stage-close"),
    )
    assert closing.diagnostics.staged_executions == 1
    await closing.close()
    assert closing.diagnostics.staged_executions == 0
    assert not closing._execution_stage_keys


async def test_execution_staging_internal_consistency_is_defensive() -> None:
    """Reject corrupt private staging indexes before authoritative writes."""
    raw = _atomic_commit("execution-internal-validation")
    validation_store = conversation.InMemoryConversationStore()
    with pytest.raises(conversation.ConversationValidationError):
        await validation_store.reserve_idempotency(
            raw.idempotency,
            execution=cast(
                conversation.ConversationExecutionReservation,
                object(),
            ),
        )
    reservation = _execution_reservation(raw)
    with pytest.raises(conversation.ConversationValidationError):
        await validation_store.reserve_idempotency(
            raw.idempotency,
            execution=replace(
                reservation,
                idempotency=replace(
                    raw.idempotency,
                    key=conversation.RequestIdempotencyKey(
                        "different-execution-reservation-key"
                    ),
                ),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        validation_store._checkpoint_identity_key(
            cast(conversation.CheckpointIdentity, object())
        )

    store = conversation.InMemoryConversationStore()
    prepared = await _prepare_atomic(
        store,
        _atomic_commit("execution-internal-corruption"),
    )
    key = store._idempotency_key(prepared.idempotency)
    entry = store._idempotency[key]
    assert entry.execution is not None
    original_identity = entry.execution.checkpoint_identity
    store._idempotency[key] = replace(
        entry,
        execution=replace(
            entry.execution,
            checkpoint_identity=(
                original_identity[0],
                original_identity[1],
                original_identity[2],
                "different-execution-checkpoint",
                original_identity[4],
                original_identity[5],
                original_identity[6],
                original_identity[7],
            ),
        ),
    )
    with pytest.raises(conversation.ConversationConflictError):
        await store.commit_atomic(prepared)
    store._idempotency[key] = entry

    staging_id, staged = next(iter(store._execution_staging.items()))
    extra_id = "execution-stage-extra-owned"
    store._execution_staging[extra_id] = replace(
        staged,
        staging_id=extra_id,
        lane_id="execution-stage-extra-lane",
    )
    with pytest.raises(conversation.ConversationConflictError):
        await store.commit_atomic(prepared)
    del store._execution_staging[extra_id]

    with pytest.raises(conversation.ConversationConflictError):
        store._validate_provisional_locked(
            prepared,
            "different-authority-digest",
        )
    missing_head = copy(prepared)
    object.__setattr__(
        missing_head,
        "head_id",
        conversation.NamedHeadId("missing-execution-head"),
    )
    object.__setattr__(
        missing_head,
        "expected_head_revision",
        conversation.NamedHeadRevision(0),
    )
    with pytest.raises(conversation.ConversationConflictError):
        store._validate_head_locked(
            missing_head,
            prepared.candidate.checkpoint,
        )

    stage_key = (
        staged.owner_token,
        staged.checkpoint_identity[3],
        staged.lane_id,
    )
    del store._execution_stage_keys[stage_key]
    with pytest.raises(conversation.ConversationStorageError):
        store._consume_staged_executions_locked((staging_id,))

    cleanup_store = conversation.InMemoryConversationStore()
    cleanup = await _prepare_atomic(
        cleanup_store,
        _atomic_commit("execution-cleanup-corruption"),
    )
    cleanup_id, cleanup_record = next(
        iter(cleanup_store._execution_staging.items())
    )
    cleanup_key = (
        cleanup_record.owner_token,
        cleanup_record.checkpoint_identity[3],
        cleanup_record.lane_id,
    )
    cleanup_store._execution_stage_keys[cleanup_key] = (
        "different-execution-stage"
    )
    cleanup_store._remove_staged_execution_owner_locked(cleanup.owner_token)
    assert cleanup_id not in cleanup_store._execution_staging
    assert (
        cleanup_store._execution_stage_keys[cleanup_key]
        == "different-execution-stage"
    )
    cleanup_store._execution_stage_keys.clear()


async def test_atomic_commit_cross_field_mismatches_preserve_snapshot() -> (
    None
):
    """Reject every cross-field drift before authoritative mutation."""
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    base = await _prepare_atomic(store, _atomic_commit("cross-field"))
    assert isinstance(
        base.candidate,
        conversation.OutwardTurnCheckpointCandidate,
    )
    checkpoint = base.candidate.checkpoint
    original_lane = checkpoint.content.lanes[0]

    extra_binding = binding("lane-cross-field-extra")
    extra_result = conversation.fake_provider_result(
        empty_stateless_plan(extra_binding),
        turn=2,
    )
    extra_receipt = conversation.provider_lane_execution_receipt(
        authority=checkpoint.authority,
        identity=checkpoint.identity,
        binding=extra_binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=extra_result.items,
        reasoning=extra_result.reasoning,
        usage=extra_result.usage,
        upstream_response_id=None,
    )
    extra_lane = conversation.StatelessProviderLaneSnapshot(
        binding=extra_binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=extra_binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=extra_result.items,
        ),
        reasoning=extra_result.reasoning,
        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
        execution_receipt=extra_receipt,
    )
    expanded_checkpoint = conversation.with_checkpoint_integrity(
        replace(
            checkpoint,
            content=replace(
                checkpoint.content,
                lanes=(original_lane, extra_lane),
            ),
        )
    )
    missing_output = replace(
        base,
        candidate=replace(
            base.candidate,
            checkpoint=expanded_checkpoint,
        ),
    )

    extra_output = conversation.ProviderLaneOutputCandidate(
        lane_id=extra_binding.lane_id,
        binding=extra_binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=extra_result.items,
        reasoning=extra_result.reasoning,
        usage=extra_result.usage,
        execution_receipt=extra_receipt,
    )
    extra_output_commit = copy(base)
    object.__setattr__(
        extra_output_commit,
        "output_candidates",
        base.output_candidates + (extra_output,),
    )

    assert base.public_response_id is not None
    mismatched_candidate = replace(
        base.candidate,
        public_response_id=conversation.PublicResponseId(
            "different-public-response"
        ),
    )
    candidate_public_mismatch = copy(base)
    object.__setattr__(
        candidate_public_mismatch,
        "candidate",
        mismatched_candidate,
    )

    authority_mismatch = copy(base)
    object.__setattr__(
        authority_mismatch,
        "idempotency",
        replace(
            base.idempotency,
            authority=authority(
                "cross-field-other-principal",
                tenant="cross-field-other-tenant",
            ),
        ),
    )

    result_mode_mismatch = copy(base)
    object.__setattr__(
        result_mode_mismatch,
        "result_mode",
        conversation.ConversationMode.STORED,
    )

    drifted_binding = replace(
        base.output_candidates[0].binding,
        model_or_deployment="different-model",
    )
    binding_mismatch = copy(base)
    object.__setattr__(
        binding_mismatch,
        "output_candidates",
        (
            replace(
                base.output_candidates[0],
                binding=drifted_binding,
            ),
        ),
    )

    snapshot = _atomic_store_snapshot(store)
    for invalid in (
        missing_output,
        extra_output_commit,
        candidate_public_mismatch,
        authority_mismatch,
        result_mode_mismatch,
        binding_mismatch,
    ):
        with pytest.raises(conversation.ConversationValidationError):
            await store.commit_atomic(invalid)
        assert _atomic_store_snapshot(store) == snapshot

    receipt = await store.commit_atomic(base)
    assert receipt.result is not None
    assert receipt.outbox is not None
    assert receipt.result.public_response_id == base.public_response_id
    assert receipt.outbox.intent.public_response_id == base.public_response_id
    assert receipt.result.lane_outputs == tuple(
        output.public_output for output in base.output_candidates
    )
    wrong_handle = replace(
        receipt.result.handle,
        checkpoint_id=conversation.CheckpointId("wrong-result-checkpoint"),
    )
    wrong_result = replace(receipt.result, handle=wrong_handle)
    with pytest.raises(conversation.ConversationValidationError):
        replace(receipt, result=wrong_result)


async def test_atomic_artifact_conflicts_are_exact() -> None:
    """Reject missing allocation, duplicate mapping/outbox, and stale head."""
    missing = _atomic_commit("missing-allocation")
    store = conversation.InMemoryConversationStore()
    resolution = await store.reserve_idempotency(missing.idempotency)
    assert resolution.owner_token is not None
    missing = replace(missing, owner_token=resolution.owner_token)
    with pytest.raises(conversation.ConversationConflictError):
        await store.commit_atomic(missing)

    first = _atomic_commit("public-first")
    second = _atomic_commit("public-second")
    assert first.public_response_id is not None
    assert isinstance(
        second.candidate, conversation.OutwardTurnCheckpointCandidate
    )
    second = replace(
        second,
        candidate=replace(
            second.candidate,
            public_response_id=first.public_response_id,
        ),
        public_response_id=first.public_response_id,
    )
    store = conversation.InMemoryConversationStore()
    first = await _prepare_atomic(store, first)
    with pytest.raises(conversation.ConversationConflictError):
        await _prepare_atomic(store, second)
    await store.commit_atomic(first)

    first = _atomic_commit("outbox-limit-first")
    second = _atomic_commit("outbox-limit-second")
    store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(max_outbox_records=1)
    )
    first = await _prepare_atomic(store, first)
    await store.commit_atomic(first)
    with pytest.raises(conversation.ConversationLimitError):
        await _prepare_atomic(store, second)
    assert store.diagnostics.provisional_responses == 0
    assert store.diagnostics.idempotency_records == 1

    first = _atomic_commit("outbox-duplicate-first")
    second = replace(
        _atomic_commit("outbox-duplicate-second"),
        outbox_intent_id=first.outbox_intent_id,
    )
    store = conversation.InMemoryConversationStore()
    first = await _prepare_atomic(store, first)
    second = await _prepare_atomic(store, second)
    await store.commit_atomic(first)
    with pytest.raises(conversation.ConversationConflictError):
        await store.commit_atomic(second)

    store = conversation.InMemoryConversationStore()
    root_commit = _atomic_commit("head-root")
    root_commit = await _prepare_atomic(store, root_commit)
    root = await store.commit_atomic(root_commit)
    head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("stale-head"),
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=root.checkpoint.identity.checkpoint_id,
    )
    await store.create_head(head, authority())
    root_lane = root.checkpoint.content.lanes[0]
    assert isinstance(root_lane, conversation.StatelessProviderLaneSnapshot)
    root_result = conversation.ProviderResult(
        items=root_lane.ledger.items,
        reasoning=root_lane.reasoning,
    )
    child_candidate = _child_candidate(
        root.checkpoint, root_result, suffix="stale-head-child"
    )
    child_lane = child_candidate.checkpoint.content.lanes[0]
    assert isinstance(child_lane, conversation.StatelessProviderLaneSnapshot)
    assert child_lane.execution_receipt is not None
    stale = conversation.AtomicConversationCommit(
        candidate=child_candidate,
        idempotency=conversation.RequestIdempotencyIdentity(
            authority=authority(),
            operation=conversation.ConversationOperation.CONTINUE,
            key=conversation.RequestIdempotencyKey("stale-head-key"),
            request_digest=conversation.CanonicalRequestDigest(
                "stale-head-digest"
            ),
        ),
        owner_token="unreserved-stale-head-owner",
        output_candidates=(
            conversation.ProviderLaneOutputCandidate(
                lane_id=child_lane.lane_id,
                binding=child_lane.binding,
                mode=conversation.ConversationMode.STATELESS,
                scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
                completed_items=(child_lane.ledger.items[-1],),
                reasoning=child_lane.reasoning,
                usage=conversation.ProviderUsage(
                    input_tokens=20,
                    output_tokens=10,
                ),
                execution_receipt=child_lane.execution_receipt,
            ),
        ),
        committed_at=NOW + timedelta(seconds=1),
        result_mode=conversation.ConversationMode.STATELESS,
        head_id=head.head_id,
        expected_head_revision=conversation.NamedHeadRevision(1),
    )
    resolution = await store.reserve_idempotency(stale.idempotency)
    assert resolution.owner_token is not None
    stale = replace(stale, owner_token=resolution.owner_token)
    with pytest.raises(conversation.ConversationConflictError):
        await store.commit_atomic(stale)


async def test_atomic_commit_rechecks_reserved_capacity_state_drift() -> None:
    """Recheck public and outbox bounds at the atomic commit lock."""

    async def inject_provisional(
        store: conversation.InMemoryConversationStore,
        commit: conversation.AtomicConversationCommit,
    ) -> conversation.AtomicConversationCommit:
        resolution = await store.reserve_idempotency(
            commit.idempotency,
            execution=_execution_reservation(commit),
        )
        assert resolution.owner_token is not None
        assert commit.provisional_response_id is not None
        assert commit.public_response_id is not None
        store._provisional[commit.provisional_response_id] = (
            conversation.ProvisionalPublicResponse(
                provisional_response_id=commit.provisional_response_id,
                public_response_id=commit.public_response_id,
                owner_token=resolution.owner_token,
                authority_digest=str(
                    conversation.authority_digest(commit.idempotency.authority)
                ),
            )
        )
        attestations = await _stage_atomic(
            store,
            commit,
            resolution.owner_token,
        )
        return replace(
            commit,
            owner_token=resolution.owner_token,
            execution_attestations=attestations,
        )

    duplicate_store = conversation.InMemoryConversationStore()
    duplicate_first = await _prepare_atomic(
        duplicate_store,
        _atomic_commit("recheck-duplicate-first"),
    )
    await duplicate_store.commit_atomic(duplicate_first)
    assert duplicate_first.public_response_id is not None
    duplicate_second = _atomic_commit("recheck-duplicate-second")
    assert isinstance(
        duplicate_second.candidate,
        conversation.OutwardTurnCheckpointCandidate,
    )
    duplicate_second = replace(
        duplicate_second,
        candidate=replace(
            duplicate_second.candidate,
            public_response_id=duplicate_first.public_response_id,
        ),
        public_response_id=duplicate_first.public_response_id,
    )
    duplicate_second = await inject_provisional(
        duplicate_store,
        duplicate_second,
    )
    with pytest.raises(conversation.ConversationConflictError):
        await duplicate_store.commit_atomic(duplicate_second)

    public_store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(max_public_responses=1)
    )
    public_first = await _prepare_atomic(
        public_store,
        _atomic_commit("recheck-public-first"),
    )
    await public_store.commit_atomic(public_first)
    public_second = await inject_provisional(
        public_store,
        _atomic_commit("recheck-public-second"),
    )
    with pytest.raises(conversation.ConversationLimitError):
        await public_store.commit_atomic(public_second)

    outbox_store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(max_outbox_records=1)
    )
    outbox_first = await _prepare_atomic(
        outbox_store,
        _atomic_commit("recheck-outbox-first"),
    )
    await outbox_store.commit_atomic(outbox_first)
    outbox_second = await inject_provisional(
        outbox_store,
        _atomic_commit("recheck-outbox-second"),
    )
    with pytest.raises(conversation.ConversationLimitError):
        await outbox_store.commit_atomic(outbox_second)


async def test_terminal_metadata_bound_conceals_inactive_head() -> None:
    """Bound terminal metadata and conceal a non-active named head."""
    store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(max_terminal_metadata=1)
    )
    first = _atomic_commit("terminal-first")
    second = _atomic_commit("terminal-second")
    first = await _prepare_atomic(store, first)
    second = await _prepare_atomic(store, second)
    first_receipt = await store.commit_atomic(first)
    second_receipt = await store.commit_atomic(second)
    head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("inactive-head"),
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=second_receipt.checkpoint.identity.checkpoint_id,
    )
    await store.create_head(head, authority())
    store._heads[
        (str(conversation.authority_digest(authority())), head.head_id)
    ] = replace(head, lifecycle=conversation.NamedHeadLifecycle.TOMBSTONED)
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load_head(head.head_id, authority())
    assert first.public_response_id is not None
    assert second.public_response_id is not None
    await store.tombstone(
        first.public_response_id,
        authority(),
        NOW + timedelta(seconds=1),
    )
    await store.tombstone(
        second.public_response_id,
        authority(),
        NOW + timedelta(seconds=2),
    )
    assert store.diagnostics.terminal_metadata == 1

    assert (
        first_receipt.checkpoint.identity.checkpoint_id not in store._terminal
    )
