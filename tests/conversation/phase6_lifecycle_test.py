"""Verify Phase 6 quarantine and lifecycle reconciliation invariants."""

from asyncio import CancelledError, Event, create_task, gather, sleep
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import cast

import pytest
from phase2_fixtures import (
    NOW,
    authority,
    binding,
    first_stored_plan,
    request,
    root_identity,
)

import avalan.conversation as conversation
from avalan.conversation import coordinator as coordinator_module

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run lifecycle tests on asyncio only."""
    return "asyncio"


class _CommitFailureHook:
    def __init__(self) -> None:
        self.failed = False

    async def reach(self, boundary: conversation.StoreAwaitBoundary) -> None:
        if (
            boundary is conversation.StoreAwaitBoundary.COMMIT_ATOMIC
            and not self.failed
        ):
            self.failed = True
            raise conversation.ConversationStorageError()


class _StageFailureHook:
    async def reach(
        self,
        boundary: conversation.CoordinatorAwaitBoundary,
    ) -> None:
        if boundary is conversation.CoordinatorAwaitBoundary.STAGE_EXECUTION:
            raise conversation.ConversationStorageError()


@dataclass(slots=True)
class _LifecycleAdapter:
    binding: conversation.ProviderLaneBinding
    fail_delete: bool = False
    deleted: list[conversation.UpstreamResponseId] | None = None

    def __post_init__(self) -> None:
        self.deleted = []

    async def retrieve(
        self,
        upstream_response_id: conversation.UpstreamResponseId,
    ) -> conversation.RetrievedUpstreamResponse:
        return conversation.RetrievedUpstreamResponse(
            upstream_response_id=upstream_response_id,
            availability=conversation.UpstreamAvailability.AVAILABLE,
            retention=conversation.UpstreamRetentionMetadata.unknown(),
        )

    async def delete(
        self,
        upstream_response_id: conversation.UpstreamResponseId,
    ) -> conversation.UpstreamDeleteResult:
        if self.fail_delete:
            raise conversation.ConversationProviderResponseError()
        assert self.deleted is not None
        self.deleted.append(upstream_response_id)
        return conversation.UpstreamDeleteResult(
            disposition=conversation.UpstreamDeleteDisposition.DELETED
        )


async def _clock() -> datetime:
    return NOW


def _coordinator(
    store: conversation.InMemoryConversationStore,
    lane_binding: conversation.ProviderLaneBinding,
    result: conversation.ProviderResult,
) -> conversation.RunScopedConversationCoordinator:
    return conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            authority()
        ),
        clock=conversation.DeterministicFakeClock(NOW),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(
            conversation.ConversationLaneRuntime(
                binding=lane_binding,
                capability_profile=conversation.fake_capability_profile(
                    lane_binding
                ),
                provider_script=conversation.DeterministicFakeProviderScript(
                    results=(result,)
                ),
            ),
        ),
    )


async def test_quarantine_internal_boundaries_reject_forged_state() -> None:
    """Reject forged completed state and missing execution runtimes."""
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._CompletedStoredProviderResponse(
            binding=cast(conversation.ProviderLaneBinding, object()),
            upstream_response_id=conversation.UpstreamResponseId(
                "private-forged"
            ),
        )
    lane_binding = binding("lane-phase6-internal-boundaries")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="internal boundaries",
    )
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    coordinator = _coordinator(store, lane_binding, result)
    run = request(
        scope=authority(),
        identity=root_identity("phase6-internal-boundaries"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-internal-boundaries",
        key="phase6-internal-boundaries",
    )
    with pytest.raises(conversation.ConversationValidationError):
        await coordinator.execute(
            run,
            stored_provider_resolver=cast(
                conversation.StoredProviderResolver,
                object(),
            ),
        )
    await coordinator._quarantine_completed_upstream(run, (), NOW)
    with pytest.raises(conversation.ConversationCapabilityError):
        coordinator._execution_reservation(
            run,
            coordinator._idempotency(run),
            {},
        )
    await coordinator.close()


async def test_quarantine_persistence_outlives_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finish durable quarantine after cancellation reaches its owner task."""
    lane_binding = binding("lane-phase6-quarantine-cancellation-owner")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="completed",
    )
    assert result.upstream_response_id is not None
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    coordinator = _coordinator(store, lane_binding, result)
    run = request(
        scope=authority(),
        identity=root_identity("phase6-quarantine-cancellation-owner"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-quarantine-cancellation-owner",
        key="phase6-quarantine-cancellation-owner",
    )
    completed = (
        coordinator_module._CompletedStoredProviderResponse(
            binding=lane_binding,
            upstream_response_id=result.upstream_response_id,
        ),
    )
    started = Event()
    release = Event()
    persisted = False

    async def delayed_quarantine(
        request_value: conversation.ConversationRunRequest,
        completed_value: tuple[
            coordinator_module._CompletedStoredProviderResponse,
            ...,
        ],
        at: datetime,
    ) -> None:
        nonlocal persisted
        assert request_value is run
        assert completed_value == completed
        assert at == NOW
        started.set()
        await release.wait()
        persisted = True

    monkeypatch.setattr(
        coordinator,
        "_quarantine_completed_upstream",
        delayed_quarantine,
    )
    owner = create_task(
        coordinator._persist_completed_upstream_quarantine(
            run,
            completed,
            NOW,
        )
    )
    await started.wait()
    owner.cancel()
    await sleep(0)
    assert not owner.done()
    release.set()
    await owner
    assert persisted
    await coordinator.close()


async def test_commit_failure_quarantines_only_completed_upstream() -> None:
    """Preserve caller state and durably quarantine completed provider work."""
    lane_binding = binding("lane-phase6-quarantine")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="quarantine",
    )
    assert result.upstream_response_id is not None
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW),
        boundary_hook=_CommitFailureHook(),
    )
    coordinator = _coordinator(store, lane_binding, result)
    run = request(
        scope=authority(),
        identity=root_identity("phase6-quarantine"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-quarantine",
        key="phase6-quarantine",
    )

    with pytest.raises(conversation.ConversationStorageError):
        await coordinator.execute(run)

    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.retrieve(
            conversation.PublicResponseId("response-phase6-quarantine"),
            authority(),
        )
    page = await store.list_checkpoints(authority(), cursor=None, limit=10)
    assert len(page.checkpoints) == 1
    quarantine = page.checkpoints[0]
    assert str(quarantine.identity.checkpoint_id).startswith("quarantine-")
    assert quarantine.identity.parent_checkpoint_id is None
    assert quarantine.kind is (
        conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
    )

    work = await store.claim_provider_lifecycle(authority(), limit=10)
    assert len(work) == 1
    assert work[0].origin is (
        conversation.ProviderLifecycleOrigin.COMMIT_QUARANTINE
    )
    assert work[0].upstream_response_id == result.upstream_response_id
    assert str(result.upstream_response_id) not in repr(work[0])

    adapter = _LifecycleAdapter(binding=lane_binding)
    resolver = conversation.StoredProviderResolver(
        (
            conversation.StoredProviderResolverEntry(
                adapter=adapter,
                revision="phase6-quarantine-resolver",
                valid_from=NOW - timedelta(minutes=1),
                valid_until=NOW + timedelta(days=1),
            ),
        ),
        clock=_clock,
    )
    await store.acknowledge_provider_lifecycle(work[0], succeeded=False)
    reconciler = conversation.ProviderLifecycleReconciler(
        store=store,
        resolver=resolver,
        authority=authority(),
    )
    assert await reconciler.run_once(limit=10) == 1
    assert adapter.deleted == [result.upstream_response_id]


async def test_quarantine_survives_ordinary_checkpoint_capacity() -> None:
    """Reserve lifecycle capacity after a completed provider response."""
    lane_binding = binding("lane-phase6-quarantine-capacity")
    retained = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="retained",
    )
    assert retained.upstream_response_id is not None
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW),
        limits=conversation.StoreLimits(max_checkpoints=1),
    )
    first_run = request(
        scope=authority(),
        identity=root_identity("phase6-capacity-existing"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-capacity-existing",
        key="phase6-capacity-existing",
    )
    existing_candidate = conversation.build_checkpoint_candidate(
        first_run,
        parent=None,
        completed_lanes=(
            conversation.StoredProviderLaneSnapshot(
                binding=lane_binding,
                upstream_response_id=retained.upstream_response_id,
                reasoning=retained.reasoning,
                lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
                retention_policy=(
                    conversation.ChildLaneRetentionPolicy.RETAIN
                ),
            ),
        ),
        created_at=NOW,
    )
    await store.commit(existing_candidate)
    assert (
        type(existing_candidate) is conversation.OutwardTurnCheckpointCandidate
    )
    spoofed_checkpoint = conversation.with_checkpoint_integrity(
        replace(
            existing_candidate.checkpoint,
            identity=replace(
                existing_candidate.checkpoint.identity,
                checkpoint_id=conversation.CheckpointId(
                    "quarantine-reserved-prefix-spoof"
                ),
            ),
            integrity=None,
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        await store.commit(
            conversation.OutwardTurnCheckpointCandidate(
                checkpoint=spoofed_checkpoint,
                public_response_id=existing_candidate.public_response_id,
            )
        )
    completed = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=2,
        text="completed before capacity failure",
    )
    coordinator = _coordinator(store, lane_binding, completed)
    second_run = request(
        scope=authority(),
        identity=root_identity("phase6-capacity-failure"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-capacity-failure",
        key="phase6-capacity-failure",
    )
    with pytest.raises(conversation.ConversationLimitError):
        await coordinator.execute(second_run)
    page = await store.list_checkpoints(authority(), cursor=None, limit=10)
    assert len(page.checkpoints) == 2
    assert (
        sum(
            str(item.identity.checkpoint_id).startswith("quarantine-")
            for item in page.checkpoints
        )
        == 1
    )
    quarantine = next(
        item
        for item in page.checkpoints
        if str(item.identity.checkpoint_id).startswith("quarantine-")
    )
    staged_quarantine = conversation.with_checkpoint_integrity(
        replace(
            quarantine,
            lifecycle=conversation.CheckpointLifecycle.STAGED,
            timestamps=replace(
                quarantine.timestamps,
                committed_at=None,
            ),
            integrity=None,
        )
    )
    replay_request = conversation.ProviderQuarantineRequest(
        candidate=conversation.ExecutionSegmentCheckpointCandidate(
            checkpoint=staged_quarantine
        ),
        created_at=NOW,
    )
    replay = await store.quarantine_provider_checkpoint(replay_request)
    assert replay.checkpoint_id == quarantine.identity.checkpoint_id
    with pytest.raises(conversation.ConversationConflictError):
        await store.quarantine_provider_checkpoint(
            replace(
                replay_request,
                created_at=NOW + timedelta(seconds=1),
            )
        )
    assert (
        len(await store.claim_provider_lifecycle(authority(), limit=10)) == 1
    )


async def test_stage_failure_after_provider_completion_is_quarantined() -> (
    None
):
    """Persist cleanup before rollback when execution staging fails."""
    lane_binding = binding("lane-phase6-stage-failure")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="completed",
    )
    assert result.upstream_response_id is not None
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    coordinator = _coordinator(store, lane_binding, result)
    coordinator._hook = _StageFailureHook()
    run = request(
        scope=authority(),
        identity=root_identity("phase6-stage-failure"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-stage-failure",
        key="phase6-stage-failure",
    )
    with pytest.raises(conversation.ConversationStorageError):
        await coordinator.execute(run)
    work = await store.claim_provider_lifecycle(authority(), limit=1)
    assert len(work) == 1
    assert work[0].upstream_response_id == result.upstream_response_id


async def test_ambiguity_reconciliation_is_authorized_and_race_safe() -> None:
    """Persist one exact fence decision without mutating unrelated state."""
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    identity = conversation.RequestIdempotencyIdentity(
        authority=authority(),
        operation=conversation.ConversationOperation.CREATE,
        key=conversation.RequestIdempotencyKey("phase6-ambiguous-key"),
        request_digest=conversation.CanonicalRequestDigest(
            "phase6-ambiguous-digest"
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await store.reconcile_ambiguous_dispatch(
            cast(
                conversation.AmbiguousDispatchReconciliationRequest,
                object(),
            )
        )
    reserved = await store.reserve_idempotency(identity)
    assert reserved.owner_token is not None
    await store.abandon_idempotency(
        identity,
        reserved.owner_token,
        ambiguous=True,
    )
    assert (await store.reserve_idempotency(identity)).disposition is (
        conversation.IdempotencyDisposition.FENCED
    )
    request_value = conversation.AmbiguousDispatchReconciliationRequest(
        authority=authority(),
        operation=conversation.ConversationOperation.CREATE,
        idempotency_key=identity.key,
        resolution=(
            conversation.AmbiguousDispatchResolution.CONFIRMED_NO_DISPATCH
        ),
    )
    raced = await gather(
        store.reconcile_ambiguous_dispatch(request_value),
        store.reconcile_ambiguous_dispatch(request_value),
    )
    assert {item.disposition for item in raced} == {
        conversation.AmbiguousDispatchReconciliationDisposition.RESOLVED_NO_DISPATCH,
        conversation.AmbiguousDispatchReconciliationDisposition.ALREADY_RESOLVED_NO_DISPATCH,
    }
    retried = await store.reserve_idempotency(identity)
    assert retried.disposition is conversation.IdempotencyDisposition.EXECUTE
    with pytest.raises(conversation.ConversationConflictError):
        await store.reconcile_ambiguous_dispatch(request_value)
    wrong = replace(
        request_value,
        authority=replace(authority(), principal_id="wrong-principal"),
    )
    concealed = await store.reconcile_ambiguous_dispatch(wrong)
    assert concealed.disposition is (
        conversation.AmbiguousDispatchReconciliationDisposition.NOT_FOUND_OR_UNAUTHORIZED
    )


async def test_post_commit_failure_recovers_without_quarantine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recognize an acknowledged-late commit without deleting live state."""
    lane_binding = binding("lane-phase6-post-commit")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="committed",
    )
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    commit_atomic = store.commit_atomic

    async def fail_after_commit(
        commit: conversation.AtomicConversationCommit,
    ) -> conversation.AtomicCommitReceipt:
        await commit_atomic(commit)
        raise conversation.ConversationStorageError()

    monkeypatch.setattr(store, "commit_atomic", fail_after_commit)
    coordinator = _coordinator(store, lane_binding, result)
    run = request(
        scope=authority(),
        identity=root_identity("phase6-post-commit"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-post-commit",
        key="phase6-post-commit",
    )

    receipt = await coordinator.execute(run)

    assert receipt.checkpoint.identity == run.identity
    assert receipt.result is not None
    assert await store.claim_provider_lifecycle(authority(), limit=10) == ()
    assert not str(receipt.checkpoint.identity.checkpoint_id).startswith(
        "quarantine-"
    )


async def test_post_commit_cancellation_recovers_then_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve a committed result while propagating caller cancellation."""
    lane_binding = binding("lane-phase6-post-commit-cancel")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="committed cancellation",
    )
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    commit_atomic = store.commit_atomic

    async def cancel_after_commit(
        commit: conversation.AtomicConversationCommit,
    ) -> conversation.AtomicCommitReceipt:
        await commit_atomic(commit)
        raise CancelledError()

    monkeypatch.setattr(store, "commit_atomic", cancel_after_commit)
    coordinator = _coordinator(store, lane_binding, result)
    run = request(
        scope=authority(),
        identity=root_identity("phase6-post-commit-cancel"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-post-commit-cancel",
        key="phase6-post-commit-cancel",
    )

    with pytest.raises(CancelledError):
        await coordinator.execute(run)
    committed = await store.load(run.identity.checkpoint_id, authority())
    assert committed.identity == run.identity
    assert await store.claim_provider_lifecycle(authority(), limit=10) == ()


async def test_post_commit_recovery_rejects_output_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an acknowledged commit whose durable outputs do not match."""
    lane_binding = binding("lane-phase6-recovery-mismatch")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="mismatched recovery",
    )
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    commit_atomic = store.commit_atomic

    async def fail_after_commit(
        commit: conversation.AtomicConversationCommit,
    ) -> conversation.AtomicCommitReceipt:
        await commit_atomic(commit)
        raise conversation.ConversationStorageError()

    async def mismatched_outputs(
        checkpoint_id: conversation.CheckpointId,
        scope: conversation.AuthorityScope,
    ) -> tuple[conversation.ProviderLaneOutputCandidate, ...]:
        assert checkpoint_id == run.identity.checkpoint_id
        assert scope == authority()
        return ()

    monkeypatch.setattr(store, "commit_atomic", fail_after_commit)
    monkeypatch.setattr(
        store,
        "retrieve_output_candidates",
        mismatched_outputs,
    )
    coordinator = _coordinator(store, lane_binding, result)
    run = request(
        scope=authority(),
        identity=root_identity("phase6-recovery-mismatch"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-recovery-mismatch",
        key="phase6-recovery-mismatch",
    )

    with pytest.raises(conversation.ConversationConflictError):
        await coordinator.execute(run)


async def test_commit_and_quarantine_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return one typed commit failure if private quarantine cannot persist."""
    lane_binding = binding("lane-phase6-quarantine-failure")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="quarantine failure",
    )
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW),
        boundary_hook=_CommitFailureHook(),
    )

    async def fail_quarantine(
        request_value: conversation.ProviderQuarantineRequest,
    ) -> conversation.ProviderQuarantineReceipt:
        assert type(request_value) is conversation.ProviderQuarantineRequest
        raise RuntimeError("private quarantine storage failure")

    monkeypatch.setattr(
        store,
        "quarantine_provider_checkpoint",
        fail_quarantine,
    )
    coordinator = _coordinator(store, lane_binding, result)
    run = request(
        scope=authority(),
        identity=root_identity("phase6-quarantine-failure"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-quarantine-failure",
        key="phase6-quarantine-failure",
    )

    with pytest.raises(conversation.ConversationCommitError) as failure:
        await coordinator.execute(run)
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert failure.value.__notes__ == [
        "provider cleanup quarantine could not be persisted"
    ]
    assert "private quarantine" not in repr(failure.value)


async def test_commit_cancellation_preserves_quarantine_failure_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate caller cancellation after attempted durable quarantine."""
    lane_binding = binding("lane-phase6-cancel-quarantine-failure")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="cancel quarantine failure",
    )
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )

    async def cancel_commit(
        commit: conversation.AtomicConversationCommit,
    ) -> conversation.AtomicCommitReceipt:
        assert type(commit) is conversation.AtomicConversationCommit
        raise CancelledError()

    async def fail_quarantine(
        request_value: conversation.ProviderQuarantineRequest,
    ) -> conversation.ProviderQuarantineReceipt:
        assert type(request_value) is conversation.ProviderQuarantineRequest
        raise RuntimeError("private quarantine cancellation failure")

    monkeypatch.setattr(store, "commit_atomic", cancel_commit)
    monkeypatch.setattr(
        store,
        "quarantine_provider_checkpoint",
        fail_quarantine,
    )
    coordinator = _coordinator(store, lane_binding, result)
    run = request(
        scope=authority(),
        identity=root_identity("phase6-cancel-quarantine-failure"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-cancel-quarantine-failure",
        key="phase6-cancel-quarantine-failure",
    )

    with pytest.raises(CancelledError) as cancellation:
        await coordinator.execute(run)
    assert isinstance(cancellation.value.__cause__, RuntimeError)
    assert "private quarantine" not in repr(cancellation.value)


async def test_quarantine_accepts_every_completed_stored_boundary() -> None:
    """Persist cleanup targets even before a final candidate exists."""
    retained_binding = binding("lane-phase6-quarantine-retained")
    foreign_binding = binding("lane-phase6-quarantine-foreign")
    retained_result = conversation.fake_provider_result(
        first_stored_plan(retained_binding),
        turn=1,
        text="retained",
    )
    foreign_result = conversation.fake_provider_result(
        first_stored_plan(foreign_binding),
        turn=1,
        text="foreign",
    )
    assert retained_result.upstream_response_id is not None
    assert foreign_result.upstream_response_id is not None
    run = request(
        scope=authority(),
        identity=root_identity("phase6-quarantine-lane-mismatch"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(retained_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-quarantine-lane-mismatch",
        key="phase6-quarantine-lane-mismatch",
    )
    foreign_receipt = conversation.provider_lane_execution_receipt(
        authority=authority(),
        identity=run.identity,
        binding=foreign_binding,
        mode=conversation.ConversationMode.STORED,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=foreign_result.items,
        reasoning=foreign_result.reasoning,
        usage=foreign_result.usage,
        upstream_response_id=foreign_result.upstream_response_id,
    )
    foreign_output = conversation.ProviderLaneOutputCandidate(
        lane_id=foreign_binding.lane_id,
        binding=foreign_binding,
        mode=conversation.ConversationMode.STORED,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=foreign_result.items,
        reasoning=foreign_result.reasoning,
        usage=foreign_result.usage,
        execution_receipt=foreign_receipt,
        upstream_response_id=foreign_result.upstream_response_id,
    )
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    coordinator = _coordinator(store, retained_binding, retained_result)

    await coordinator._quarantine_completed_upstream(
        run,
        (
            coordinator_module._CompletedStoredProviderResponse(
                binding=foreign_output.binding,
                upstream_response_id=foreign_result.upstream_response_id,
            ),
        ),
        NOW,
    )
    work = await store.claim_provider_lifecycle(authority(), limit=1)
    assert len(work) == 1
    assert work[0].binding_digest == foreign_binding.integrity_digest


def test_codec_restoration_cannot_bypass_public_id_separation() -> None:
    """Reapply ID separation after strict checkpoint codec restoration."""
    lane_binding = binding("lane-phase6-codec-alias")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="private",
    )
    assert result.upstream_response_id is not None
    run = request(
        scope=authority(),
        identity=root_identity("phase6-codec-alias"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-codec-alias",
        key="phase6-codec-alias",
    )
    candidate = conversation.build_checkpoint_candidate(
        run,
        parent=None,
        completed_lanes=(
            conversation.StoredProviderLaneSnapshot(
                binding=lane_binding,
                upstream_response_id=result.upstream_response_id,
                reasoning=result.reasoning,
                lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
                retention_policy=(
                    conversation.ChildLaneRetentionPolicy.RETAIN
                ),
            ),
        ),
        created_at=NOW,
    )
    restored = conversation.ConversationCheckpointCodec().decode(
        conversation.ConversationCheckpointCodec().encode(candidate.checkpoint)
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.OutwardTurnCheckpointCandidate(
            checkpoint=restored,
            public_response_id=conversation.PublicResponseId(
                str(result.upstream_response_id)
            ),
        )


async def test_reconciler_retries_failure_without_restoring_local_access() -> (
    None
):
    """Retry provider cleanup while a local tombstone remains authoritative."""
    lane_binding = binding("lane-phase6-retry")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="retry",
    )
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    coordinator = _coordinator(store, lane_binding, result)
    run = request(
        scope=authority(),
        identity=root_identity("phase6-retry"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-retry",
        key="phase6-retry",
    )
    receipt = await coordinator.execute(run)
    public_id = conversation.PublicResponseId("response-phase6-retry")
    active = await store.prepare_deletion(public_id, authority())
    assert active.state is conversation.LocalDeletionState.ACTIVE
    assert "checkpoint_available=True" in repr(active)
    tombstone = await store.tombstone(
        public_id, authority(), NOW + timedelta(seconds=1)
    )
    tombstoned = await store.prepare_deletion(public_id, authority())
    assert tombstoned.state is conversation.LocalDeletionState.TOMBSTONED

    adapter = _LifecycleAdapter(binding=lane_binding, fail_delete=True)
    resolver = conversation.StoredProviderResolver(
        (
            conversation.StoredProviderResolverEntry(
                adapter=adapter,
                revision="phase6-retry-resolver",
                valid_from=NOW - timedelta(minutes=1),
            ),
        ),
        clock=_clock,
    )
    reconciler = conversation.ProviderLifecycleReconciler(
        store=store,
        resolver=resolver,
        authority=authority(),
    )
    assert await reconciler.run_once(limit=1) == 0
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.retrieve(public_id, authority())
    with pytest.raises(conversation.ConversationTransitionError):
        await store.delete(public_id, authority(), NOW + timedelta(seconds=2))

    adapter.fail_delete = False
    assert await reconciler.run_once(limit=1) == 1
    await store.delete(public_id, authority(), NOW + timedelta(seconds=3))
    deleted = await store.prepare_deletion(public_id, authority())
    assert deleted.state is conversation.LocalDeletionState.DELETED
    assert "checkpoint_available=False" in repr(deleted)
    assert receipt.result is not None
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(
            receipt.checkpoint.identity.checkpoint_id, authority()
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.LocalDeletionPreparation(
            state=cast(conversation.LocalDeletionState, "invalid"),
            checkpoint=None,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.LocalDeletionPreparation(
            state=conversation.LocalDeletionState.DELETED,
            checkpoint=receipt.checkpoint,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.LocalDeletionPreparation(
            state=conversation.LocalDeletionState.ACTIVE,
            checkpoint=None,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.LocalDeletionPreparation(
            state=conversation.LocalDeletionState.ACTIVE,
            checkpoint=tombstone,
        )


async def test_lifecycle_store_validation_leases_and_deduplicates() -> None:
    """Close lifecycle authority, lease, acknowledgement, and dedupe edges."""
    lane_binding = binding("lane-phase6-store-validation")
    result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="store validation",
    )
    clock = conversation.DeterministicFakeClock(NOW)
    store = conversation.InMemoryConversationStore(clock=clock)
    coordinator = _coordinator(store, lane_binding, result)
    run = request(
        scope=authority(),
        identity=root_identity("phase6-store-validation"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-store-validation",
        key="phase6-store-validation",
    )
    receipt = await coordinator.execute(run)
    public_id = conversation.PublicResponseId(
        "response-phase6-store-validation"
    )

    with pytest.raises(conversation.ConversationValidationError):
        await store.prepare_deletion(
            public_id,
            cast(conversation.AuthorityScope, object()),
        )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.prepare_deletion(
            conversation.PublicResponseId("response-phase6-unknown"),
            authority(),
        )
    checkpoint_id = receipt.checkpoint.identity.checkpoint_id
    stored = store._checkpoints.pop(checkpoint_id)
    try:
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.prepare_deletion(public_id, authority())
    finally:
        store._checkpoints[checkpoint_id] = stored

    tombstone = await store.tombstone(
        public_id,
        authority(),
        NOW + timedelta(seconds=1),
    )
    store._enqueue_provider_lifecycle_locked(
        tombstone,
        conversation.ProviderLifecycleOrigin.LOCAL_TOMBSTONE,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await store.claim_provider_lifecycle(
            cast(conversation.AuthorityScope, object()), limit=1
        )
    with pytest.raises(conversation.ConversationLimitError):
        await store.claim_provider_lifecycle(authority(), limit=0)
    assert (
        await store.claim_provider_lifecycle(authority("wrong"), limit=1) == ()
    )
    claimed = await store.claim_provider_lifecycle(authority(), limit=1)
    assert len(claimed) == 1
    assert await store.claim_provider_lifecycle(authority(), limit=1) == ()
    with pytest.raises(conversation.ConversationValidationError):
        await store.acknowledge_provider_lifecycle(
            cast(conversation.ProviderLifecycleWorkRecord, object()),
            succeeded=True,
        )
    with pytest.raises(conversation.ConversationConflictError):
        await store.acknowledge_provider_lifecycle(
            replace(claimed[0], attempts=claimed[0].attempts + 1),
            succeeded=True,
        )
    clock.set(NOW + timedelta(minutes=10))
    reclaimed = await store.claim_provider_lifecycle(authority(), limit=1)
    assert len(reclaimed) == 1
    assert reclaimed[0].attempts == claimed[0].attempts + 1
    await store.acknowledge_provider_lifecycle(reclaimed[0], succeeded=True)
    await store.delete(public_id, authority(), NOW + timedelta(minutes=11))
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.prepare_deletion(public_id, authority("wrong"))
    with pytest.raises(conversation.ConversationValidationError):
        await store.quarantine_provider_checkpoint(
            cast(conversation.ProviderQuarantineRequest, object())
        )


async def test_pgsql_lifecycle_adapter_rejects_malformed_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed Pgsql lifecycle records before private dispatch."""
    store = object.__new__(conversation.PgsqlConversationStore)
    malformed = conversation.ReconciliationWorkRecord(
        reconciliation_id="phase6-malformed-pgsql-work",
        checkpoint_id=conversation.CheckpointId("phase6-malformed-checkpoint"),
        lane_id=conversation.ProviderLaneId("phase6-malformed-lane"),
        work_kind="delete_upstream",
        state=conversation.ReconciliationWorkState.CLAIMED,
        attempts=1,
        upstream_response_id=conversation.UpstreamResponseId(
            "phase6-private-malformed"
        ),
        lease_owner="phase6-malformed-owner",
        lease_expires_at=NOW + timedelta(minutes=1),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            malformed,
            checkpoint_lifecycle=cast(
                conversation.CheckpointLifecycle,
                object(),
            ),
        )

    async def claim_reconciliation(
        scope: conversation.AuthorityScope,
        *,
        limit: int,
        provider_lifecycle_only: bool,
    ) -> tuple[conversation.ReconciliationWorkRecord, ...]:
        assert scope == authority()
        assert limit == 1
        assert provider_lifecycle_only
        return (malformed,)

    monkeypatch.setattr(store, "_claim_reconciliation", claim_reconciliation)
    with pytest.raises(conversation.ConversationValidationError):
        await store._load_checkpoint_lifecycle(
            conversation.CheckpointId("phase6-malformed-checkpoint"),
            authority(),
            cast(conversation.CheckpointLifecycle, object()),
        )
    with pytest.raises(conversation.ConversationStorageError):
        await store.claim_provider_lifecycle(authority(), limit=1)
    with pytest.raises(conversation.ConversationValidationError):
        await store.quarantine_provider_checkpoint(
            cast(conversation.ProviderQuarantineRequest, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        await store.acknowledge_provider_lifecycle(
            cast(conversation.ProviderLifecycleWorkRecord, object()),
            succeeded=True,
        )


@pytest.mark.parametrize(
    "corruption,expected",
    (
        ("identity", conversation.ConversationConflictError),
        ("payload_missing", conversation.ConversationStorageError),
        ("payload_mismatch", conversation.ConversationConflictError),
    ),
)
async def test_pgsql_quarantine_replay_rejects_corrupt_existing_state(
    corruption: str,
    expected: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject identity, payload, and decoded-byte drift on PG replay."""
    lane_binding = binding("lane-phase6-pgsql-quarantine-corruption")
    provider_result = conversation.fake_provider_result(
        first_stored_plan(lane_binding),
        turn=1,
        text="pgsql quarantine corruption",
    )
    assert provider_result.upstream_response_id is not None
    source_store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW)
    )
    source_coordinator = _coordinator(
        source_store,
        lane_binding,
        provider_result,
    )
    run = request(
        scope=authority(),
        identity=root_identity("phase6-pgsql-quarantine-corruption"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(lane_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="phase6-pgsql-quarantine-corruption",
        key="phase6-pgsql-quarantine-corruption",
    )
    captured: list[conversation.ProviderQuarantineRequest] = []

    async def capture_quarantine(
        request_value: conversation.ProviderQuarantineRequest,
    ) -> conversation.ProviderQuarantineReceipt:
        captured.append(request_value)
        return conversation.ProviderQuarantineReceipt(
            checkpoint_id=request_value.candidate.checkpoint.identity.checkpoint_id,
            target_count=1,
        )

    monkeypatch.setattr(
        source_store,
        "quarantine_provider_checkpoint",
        capture_quarantine,
    )
    await source_coordinator._quarantine_completed_upstream(
        run,
        (
            coordinator_module._CompletedStoredProviderResponse(
                binding=lane_binding,
                upstream_response_id=provider_result.upstream_response_id,
            ),
        ),
        NOW,
    )
    assert len(captured) == 1
    request_value = captured[0]
    staged = request_value.candidate.checkpoint
    committed = conversation.with_checkpoint_integrity(
        replace(
            staged,
            lifecycle=conversation.CheckpointLifecycle.COMMITTED,
            timestamps=replace(staged.timestamps, committed_at=NOW),
            integrity=None,
        )
    )

    @dataclass(frozen=True, slots=True)
    class Prepared:
        checkpoint: conversation.ConversationCheckpoint

    store = object.__new__(conversation.PgsqlConversationStore)

    async def prepare_checkpoint(
        candidate: conversation.CheckpointCandidate,
        *,
        committed_at: datetime,
        output_candidates: tuple[
            conversation.ProviderLaneOutputCandidate,
            ...,
        ],
    ) -> Prepared:
        assert candidate is request_value.candidate
        assert committed_at == NOW
        assert output_candidates == ()
        return Prepared(checkpoint=committed)

    authority_key = str(conversation.authority_digest(committed.authority))

    async def fetchone(
        cursor: object,
        operation_name: str,
        sql: str,
        parameters: tuple[object, ...],
    ) -> Mapping[str, object] | None:
        assert cursor is not None
        assert sql
        assert parameters
        if operation_name == "provider_quarantine_existing":
            return {
                "authority_digest": (
                    "wrong-authority"
                    if corruption == "identity"
                    else authority_key
                ),
                "conversation_id": str(committed.identity.conversation_id),
            }
        assert operation_name == "provider_quarantine_existing_payload"
        if corruption == "payload_missing":
            return None
        return {
            "lifecycle_state": conversation.CheckpointLifecycle.COMMITTED.value
        }

    async def transaction(
        name: str,
        operation: Callable[[object], Awaitable[None]],
    ) -> None:
        assert name == "provider_quarantine"
        await operation(object())

    def validate_payload(*args: object, **kwargs: object) -> None:
        assert args or kwargs

    async def decrypt_payload(
        payload: Mapping[str, object],
    ) -> bytes:
        assert payload["lifecycle_state"] == "committed"
        return b"mismatched-payload"

    class MismatchedCodec:
        def decode(
            self, payload: bytes
        ) -> conversation.ConversationCheckpoint:
            assert payload == b"mismatched-payload"
            return staged

    monkeypatch.setattr(store, "_prepare_checkpoint", prepare_checkpoint)
    monkeypatch.setattr(store, "_fetchone", fetchone)
    monkeypatch.setattr(store, "_transaction", transaction)
    monkeypatch.setattr(
        store,
        "_validate_payload_reference_row",
        validate_payload,
    )
    monkeypatch.setattr(store, "_decrypt_payload_row", decrypt_payload)
    monkeypatch.setattr(
        store,
        "_checkpoint_codec",
        MismatchedCodec(),
        raising=False,
    )
    with pytest.raises(expected):
        await store.quarantine_provider_checkpoint(request_value)
    await source_coordinator.close()


def test_lifecycle_values_reject_invalid_shapes_and_hide_ids() -> None:
    """Reject invalid lifecycle metadata and keep private IDs out of reprs."""
    unknown = conversation.UpstreamRetentionMetadata.unknown()
    assert unknown.status is conversation.UpstreamLifetimeStatus.UNKNOWN
    invalid_metadata = (
        {"status": conversation.UpstreamLifetimeStatus.NOT_APPLICABLE},
        {
            "status": conversation.UpstreamLifetimeStatus.UNKNOWN,
            "ttl_seconds": 1,
        },
        {"status": conversation.UpstreamLifetimeStatus.KNOWN},
    )
    for values in invalid_metadata:
        with pytest.raises(conversation.ConversationValidationError):
            conversation.UpstreamRetentionMetadata(**values)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DirectDeletionResult(
            public_response_id=conversation.PublicResponseId("response"),
            local_tombstoned=False,
            upstream_pending=False,
        )
