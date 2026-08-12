"""Exercise the strict durable semantic-store checkpoint in memory."""

from asyncio import create_task, gather, run, sleep
from collections.abc import Awaitable, Callable
from dataclasses import replace

import pytest
from phase_6_contract_test import _approved_with_service

from avalan.patch import durable_approval as durable_approval
from avalan.patch import durable_retention as retention
from avalan.patch import durable_store as durable
from avalan.patch import pgsql_store as pgsql_durable
from avalan.patch.coordinator import RetransmissionKey, _sealed_journal_steps
from avalan.patch.domain import (
    AlgorithmDigest,
    ArtifactState,
    Audience,
    ByteSize,
    CommitStepState,
    CommitTruth,
    DurationTicks,
    ErrorStage,
    ExpiryTick,
    LifecyclePhase,
    LineageState,
    MutationState,
    PatchApprovalId,
    PatchArtifactId,
    PatchCommitOwnerId,
    PatchContextId,
    PatchDiagnostic,
    PatchDomainId,
    PatchErrorCode,
    PatchEventId,
    PatchExecutionId,
    PatchGrantId,
    PatchLineageId,
    PatchObserverCorrelationId,
    PatchPendingOperationId,
    PatchPlanId,
    PatchRequestId,
    PatchResult,
    PatchRetentionKeyId,
    PatchRetentionRecordId,
    PatchStatus,
    PatchStepId,
    PatchWorkspaceId,
    PostconditionState,
    RequestedEffectOccurrence,
    Retryability,
    SequenceNumber,
    WorkspaceChange,
)
from avalan.patch.durable_approval import (
    DurableApprovalSigningKey,
    HmacDurableApprovalAuthority,
    PhaseFiveDurableApprovalIssuer,
)
from avalan.patch.durable_retention import (
    AesGcmDurableRetentionCipher,
    AesGcmDurableRetentionEnvelopeValidator,
    DurableEncryptedRetention,
    DurableRetentionBinding,
    DurableRetentionKey,
    InMemoryDurableRetentionKeyResolver,
    StaticDurableRetentionAuthorizer,
)
from avalan.patch.durable_store import (
    DenyDurableApprovalVerifier,
    DenyDurableRetentionAuthorizer,
    DenyDurableRetentionEnvelopeValidator,
    DurableApproval,
    DurableArtifactJournalEntry,
    DurableArtifactState,
    DurableCommitClaim,
    DurableCommitClaimState,
    DurableCommitLease,
    DurableJournal,
    DurableJournalCursor,
    DurablePatchStore,
    DurablePendingAccess,
    DurablePendingRecord,
    DurablePendingRequest,
    DurablePlanReference,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableReservation,
    DurableRetentionAccess,
    DurableRetentionCleanup,
    DurableRetentionKind,
    DurableRetentionPolicy,
    DurableRetentionRecord,
    DurableStepBinding,
    DurableStepJournalEntry,
    DurableStoreError,
    DurableStoreErrorCode,
    DurableStoreLimits,
    DurableWorkerBinding,
    EncryptedRetentionValue,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.pgsql_store import PgsqlDurablePatchStore
from avalan.patch.policy import (
    PatchPrincipalId,
    PatchTenantId,
    PolicyBrokerId,
    PolicyReviewerRole,
    PolicyRouteId,
)

_APPROVAL_AUTHORITY = HmacDurableApprovalAuthority(
    DurableApprovalSigningKey(b"p" * 32)
)


def _digest(token: str) -> AlgorithmDigest:
    """Return deterministic opaque digest evidence for a test value."""
    return AlgorithmDigest("sha256", token * 64)


def _identity(token: str = "a") -> DurableRequestIdentity:
    """Return one authenticated retransmission tuple."""
    return DurableRequestIdentity(
        PatchTenantId("tenant-" + token),
        PatchPrincipalId("principal-" + token),
        PatchExecutionId("execution_" + token * 16),
        PolicyRouteId("route-" + token),
        RetransmissionKey("retransmission-" + token),
    )


def _plan(
    digest: AlgorithmDigest,
    token: str = "a",
    step_count: int = 2,
) -> DurablePlanReference:
    """Return a fixed sealed non-content plan reference."""
    return DurablePlanReference(
        PatchPlanId("plan_" + token * 16),
        digest,
        _digest("b"),
        _digest("c"),
        PatchContextId("context_" + token * 16),
        PatchWorkspaceId("workspace_" + token * 16),
        PatchDomainId("domain_" + token * 16),
        tuple(
            DurableStepBinding(
                PatchStepId("step_" + f"{index:x}" * 16),
                PatchLineageId("lineage_" + f"{index:x}" * 16),
            )
            for index in range(1, step_count + 1)
        ),
    )


def _approval(
    identity: DurableRequestIdentity,
    digest: AlgorithmDigest,
    plan: DurablePlanReference,
    token: str = "a",
    expires_at: int = 100,
) -> DurableApproval:
    """Return one exact unconsumed durable approval binding."""
    return _APPROVAL_AUTHORITY.seal(
        DurableApproval(
            PatchGrantId("grant_" + token * 16),
            PatchApprovalId("approval_" + token * 16),
            identity,
            digest,
            plan.plan_id,
            plan.fingerprint_digest,
            plan.review_digest,
            plan.context_id,
            plan.workspace_id,
            plan.domain_id,
            "policy-" + token,
            PolicyBrokerId("broker-" + token),
            PolicyReviewerRole("reviewer-" + token),
            (PatchPrincipalId("reviewer-" + token),),
            ExpiryTick(expires_at),
            b"\x00" * 32,
        )
    )


def _backend(
    limits: DurableStoreLimits = DurableStoreLimits(),
) -> InMemoryDurablePatchBackend:
    """Return a store backend configured with one broker signing authority."""
    return InMemoryDurablePatchBackend(
        limits,
        approval_verifier=_APPROVAL_AUTHORITY,
    )


def _retention_backend(
    key: DurableRetentionKey,
    limits: DurableStoreLimits = DurableStoreLimits(),
) -> InMemoryDurablePatchBackend:
    """Return a broker and AEAD configured test-host durable backend."""
    cipher = AesGcmDurableRetentionCipher(
        InMemoryDurableRetentionKeyResolver(key.key_id, {key.key_id: key})
    )
    return InMemoryDurablePatchBackend(
        limits,
        approval_verifier=_APPROVAL_AUTHORITY,
        retention_authorizer=StaticDurableRetentionAuthorizer(
            frozenset((Audience.APPROVER,))
        ),
        retention_validator=AesGcmDurableRetentionEnvelopeValidator(cipher),
    )


def _retention_key(token: str) -> DurableRetentionKey:
    """Return a deterministic AES-256-GCM retention key for one test."""
    return DurableRetentionKey(
        PatchRetentionKeyId("retention_" + token * 16),
        token.encode("ascii") * 32,
    )


async def _phase_five_approval() -> tuple[
    DurableRequestIdentity,
    AlgorithmDigest,
    DurablePlanReference,
    DurableApproval,
]:
    """Issue one durable claim through the Phase 5 broker grant boundary."""
    sealed, grant, approvals = await _approved_with_service(step_count=1)
    identity = DurableRequestIdentity(
        sealed.binding.subject.tenant,
        sealed.binding.subject.principal,
        sealed.binding.request.execution_id,
        sealed.binding.final.approval.route,
        RetransmissionKey("phase-eight-broker"),
    )
    plan = DurablePlanReference(
        sealed.plan_id,
        sealed.binding.request_digest,
        sealed.fingerprint.digest(),
        sealed.review.diff.digest,
        sealed.binding.target.context_id,
        sealed.binding.target.workspace_id,
        sealed.binding.target.domain_id,
        tuple(
            DurableStepBinding(step_id, lineage_id)
            for step_id, lineage_id in _sealed_journal_steps(sealed)
        ),
    )
    approval = await PhaseFiveDurableApprovalIssuer(
        approvals, _APPROVAL_AUTHORITY
    ).issue(identity, plan, grant, sealed, sealed.binding.subject)
    return identity, plan.canonical_digest, plan, approval


def _owner(token: str) -> PatchCommitOwnerId:
    """Return one deterministic owner identity."""
    return PatchCommitOwnerId("owner_" + token * 16)


def _artifact(token: str = "a") -> PatchArtifactId:
    """Return one target-owned staging artifact identity."""
    return PatchArtifactId("artifact_" + token * 16)


def _correlation(token: str = "a") -> PatchObserverCorrelationId:
    """Return one original branch correlation identity."""
    return PatchObserverCorrelationId("correlation_" + token * 16)


def _pending(token: str = "a") -> DurablePendingRequest:
    """Return one nonterminal host-pending request."""
    return DurablePendingRequest(
        PatchPendingOperationId("pending_" + token * 16),
        _correlation(token),
        DurationTicks(10),
    )


async def _exclusive_recovery_contract(
    owner_store: DurablePatchStore,
    competing_store: DurablePatchStore,
) -> None:
    """Keep one expired reaped owner exclusive until recovery settles."""
    owner_identity = _identity("d")
    owner_digest = _digest("d")
    owner_reservation = await owner_store.reserve(owner_identity, owner_digest)
    owner_plan = _plan(owner_digest, "d", step_count=1)
    owner_approval = _approval(owner_identity, owner_digest, owner_plan, "d")
    await owner_store.persist_plan(owner_reservation, owner_plan)
    owner_claim = await owner_store.claim_commit(
        owner_reservation,
        owner_plan,
        owner_approval,
        _owner("d"),
        ExpiryTick(10),
        DurationTicks(10),
        (),
    )
    assert owner_claim.lease is not None
    worker = DurableWorkerBinding(
        "session-d",
        "channel-d",
        "implementation-d",
        _digest("a"),
        _digest("b"),
    )
    await owner_store.bind_worker(owner_claim.lease, worker, ExpiryTick(11))
    with pytest.raises(DurableStoreError) as absent_bound_worker:
        await owner_store.mark_worker_absent(owner_claim.lease)
    assert absent_bound_worker.value.code is DurableStoreErrorCode.FENCED
    with pytest.raises(DurableStoreError) as unreaped_expiry:
        await owner_store.replace_expired_owner(
            owner_reservation,
            owner_claim.lease,
            _owner("f"),
            ExpiryTick(21),
            DurationTicks(10),
        )
    assert unreaped_expiry.value.code is DurableStoreErrorCode.FENCED
    await owner_store.mark_worker_reaped(owner_claim.lease, worker)

    competing_identity = _identity("e")
    competing_digest = _digest("e")
    competing_reservation = await competing_store.reserve(
        competing_identity, competing_digest
    )
    competing_plan = replace(
        _plan(competing_digest, "e", step_count=1),
        domain_id=owner_plan.domain_id,
    )
    competing_approval = _approval(
        competing_identity,
        competing_digest,
        competing_plan,
        "e",
    )
    await competing_store.persist_plan(competing_reservation, competing_plan)
    with pytest.raises(DurableStoreError) as exclusive:
        await competing_store.claim_commit(
            competing_reservation,
            competing_plan,
            competing_approval,
            _owner("e"),
            ExpiryTick(21),
            DurationTicks(10),
            (),
        )
    assert exclusive.value.code is DurableStoreErrorCode.FENCED

    attached = await owner_store.claim_commit(
        owner_reservation,
        owner_plan,
        owner_approval,
        _owner("f"),
        ExpiryTick(21),
        DurationTicks(10),
        (),
    )
    assert attached.state is DurableCommitClaimState.ATTACHED
    recovery = await owner_store.replace_expired_owner(
        owner_reservation,
        owner_claim.lease,
        _owner("f"),
        ExpiryTick(21),
        DurationTicks(10),
    )
    await owner_store.mark_worker_absent(recovery)
    journal = await owner_store.append_step(
        recovery,
        DurableJournalCursor(owner_reservation.request_id, SequenceNumber(0)),
        owner_plan.steps[0].step_id,
        CommitStepState.PLANNED,
        ExpiryTick(22),
    )
    journal = await owner_store.append_step(
        recovery,
        journal.cursor,
        owner_plan.steps[0].step_id,
        CommitStepState.UNKNOWN,
        ExpiryTick(22),
    )
    terminal = await owner_store.settle(
        recovery,
        journal.cursor,
        _result(
            owner_reservation.request_id,
            owner_plan,
            MutationState.INDETERMINATE,
        ),
        _correlation("d"),
        ExpiryTick(22),
    )
    assert terminal.result.status is PatchStatus.INDETERMINATE

    later_claim = await competing_store.claim_commit(
        competing_reservation,
        competing_plan,
        competing_approval,
        _owner("e"),
        ExpiryTick(23),
        DurationTicks(10),
        (),
    )
    assert later_claim.state is DurableCommitClaimState.OWNER
    assert later_claim.lease is not None
    assert later_claim.lease.fence.value == recovery.fence.value + 1


async def _worker_transition_lease_parity_contract(
    store: DurablePatchStore,
) -> None:
    """Require the exact renewed lease for every worker transition."""
    bound_identity = _identity("c")
    bound_digest = _digest("c")
    bound_reservation = await store.reserve(bound_identity, bound_digest)
    bound_plan = _plan(bound_digest, "c", step_count=1)
    await store.persist_plan(bound_reservation, bound_plan)
    bound_claim = await store.claim_commit(
        bound_reservation,
        bound_plan,
        _approval(bound_identity, bound_digest, bound_plan, "c"),
        _owner("c"),
        ExpiryTick(10),
        DurationTicks(20),
        (),
    )
    assert bound_claim.lease is not None
    binding = DurableWorkerBinding(
        "session-c",
        "channel-c",
        "implementation-c",
        _digest("a"),
        _digest("b"),
    )
    await store.bind_worker(bound_claim.lease, binding, ExpiryTick(11))
    bound_renewed = await store.renew_lease(
        bound_claim.lease, ExpiryTick(12), DurationTicks(20)
    )
    with pytest.raises(DurableStoreError) as stale_reaped:
        await store.mark_worker_reaped(bound_claim.lease, binding)
    assert stale_reaped.value.code is DurableStoreErrorCode.FENCED
    with pytest.raises(DurableStoreError) as bound_absent:
        await store.mark_worker_absent(bound_renewed)
    assert bound_absent.value.code is DurableStoreErrorCode.FENCED
    await store.mark_worker_reaped(bound_renewed, binding)

    unbound_identity = _identity("d")
    unbound_digest = _digest("d")
    unbound_reservation = await store.reserve(unbound_identity, unbound_digest)
    unbound_plan = _plan(unbound_digest, "d", step_count=1)
    await store.persist_plan(unbound_reservation, unbound_plan)
    unbound_claim = await store.claim_commit(
        unbound_reservation,
        unbound_plan,
        _approval(unbound_identity, unbound_digest, unbound_plan, "d"),
        _owner("d"),
        ExpiryTick(10),
        DurationTicks(20),
        (),
    )
    assert unbound_claim.lease is not None
    unbound_renewed = await store.renew_lease(
        unbound_claim.lease, ExpiryTick(12), DurationTicks(20)
    )
    with pytest.raises(DurableStoreError) as stale_absent:
        await store.mark_worker_absent(unbound_claim.lease)
    assert stale_absent.value.code is DurableStoreErrorCode.FENCED
    with pytest.raises(DurableStoreError) as unbound_reaped:
        await store.mark_worker_reaped(unbound_renewed, binding)
    assert unbound_reaped.value.code is DurableStoreErrorCode.FENCED
    await store.mark_worker_absent(unbound_renewed)


def _result(
    request_id: PatchRequestId,
    plan: DurablePlanReference,
    mutation: MutationState,
    artifact: ArtifactState = ArtifactState.ABSENT,
) -> PatchResult:
    """Return terminal domain truth that matches a durable journal state."""
    match mutation:
        case MutationState.COMMITTED:
            truth = CommitTruth(
                MutationState.COMMITTED,
                LineageState.COMMITTED,
                RequestedEffectOccurrence.TRUE,
                artifact,
                WorkspaceChange.CHANGED,
                True,
                PostconditionState.ESTABLISHED,
            )
            status = PatchStatus.COMMITTED
            diagnostic = None
        case MutationState.PARTIALLY_COMMITTED:
            truth = CommitTruth(
                MutationState.PARTIALLY_COMMITTED,
                LineageState.PARTIALLY_COMMITTED,
                RequestedEffectOccurrence.TRUE,
                artifact,
                WorkspaceChange.CHANGED,
                True,
                PostconditionState.ESTABLISHED,
            )
            status = PatchStatus.PARTIAL
            diagnostic = PatchDiagnostic(
                ErrorStage.COMMIT,
                PatchErrorCode.PARTIAL_COMMIT,
                Retryability.NOT_RETRYABLE,
            )
        case MutationState.INDETERMINATE:
            truth = CommitTruth(
                MutationState.INDETERMINATE,
                LineageState.INDETERMINATE,
                RequestedEffectOccurrence.UNKNOWN,
                artifact,
                WorkspaceChange.UNKNOWN,
                False,
                PostconditionState.UNKNOWN,
            )
            status = PatchStatus.INDETERMINATE
            diagnostic = PatchDiagnostic(
                ErrorStage.SETTLEMENT,
                PatchErrorCode.INDETERMINATE,
                Retryability.NOT_RETRYABLE,
            )
        case MutationState.NOT_COMMITTED:
            truth = CommitTruth(
                MutationState.NOT_COMMITTED,
                LineageState.NOT_COMMITTED,
                RequestedEffectOccurrence.FALSE,
                artifact,
                WorkspaceChange.UNCHANGED,
                True,
                PostconditionState.UNKNOWN,
            )
            status = PatchStatus.COMMIT_FAILED
            diagnostic = PatchDiagnostic(
                ErrorStage.COMMIT,
                PatchErrorCode.COMMIT_FAILED,
                Retryability.NOT_RETRYABLE,
            )
    return PatchResult(
        1,
        request_id,
        plan.plan_id,
        LifecyclePhase.REQUEST_COMPLETED,
        status,
        truth,
        diagnostic,
    )


async def _claimed(
    *,
    token: str = "a",
    step_count: int = 2,
    artifact_ids: tuple[PatchArtifactId, ...] = (),
    backend: InMemoryDurablePatchBackend | None = None,
) -> tuple[
    InMemoryDurablePatchStore,
    InMemoryDurablePatchBackend,
    DurableRequestIdentity,
    DurableReservation,
    DurablePlanReference,
    DurableCommitLease,
]:
    """Create one claimed durable record without a target worker."""
    backend_value = backend or _backend()
    store = InMemoryDurablePatchStore(backend_value)
    identity = _identity(token)
    digest = _digest(token)
    reservation = await store.reserve(identity, digest)
    plan = _plan(digest, token, step_count)
    await store.persist_plan(reservation, plan)
    claim = await store.claim_commit(
        reservation,
        plan,
        _approval(identity, digest, plan, token),
        _owner(token),
        ExpiryTick(10),
        DurationTicks(20),
        artifact_ids,
    )
    assert claim.state is DurableCommitClaimState.OWNER
    assert claim.lease is not None
    return store, backend_value, identity, reservation, plan, claim.lease


def test_reservation_is_linearizable_and_conflicts_before_plan() -> None:
    """Reserve exact duplicates and reject conflicting canonical input."""

    async def scenario() -> None:
        backend = _backend()
        first = InMemoryDurablePatchStore(backend)
        fresh_client = InMemoryDurablePatchStore(backend)
        identity = _identity()
        digest = _digest("a")
        reservation = await first.reserve(identity, digest)
        duplicate = await fresh_client.reserve(identity, digest)

        assert not reservation.replayed
        assert duplicate.replayed
        assert duplicate.request_id == reservation.request_id
        with pytest.raises(DurableStoreError) as raised:
            await fresh_client.reserve(identity, _digest("d"))
        assert raised.value.code is DurableStoreErrorCode.IDEMPOTENCY_CONFLICT
        assert await first.inspect(
            DurableRequestAccess(reservation.request_id, identity)
        ) == await fresh_client.inspect(
            DurableRequestAccess(reservation.request_id, identity)
        )

    run(scenario())


def test_atomic_claim_consumes_one_approval_and_attaches_duplicates() -> None:
    """Claim one owner atomically while a concurrent caller only attaches."""

    async def scenario() -> None:
        backend = _backend()
        first = InMemoryDurablePatchStore(backend)
        second = InMemoryDurablePatchStore(backend)
        identity = _identity()
        digest = _digest("a")
        reservation = await first.reserve(identity, digest)
        plan = _plan(digest)
        await first.persist_plan(reservation, plan)
        approval = _approval(identity, digest, plan)
        first_claim, second_claim = await gather(
            first.claim_commit(
                reservation,
                plan,
                approval,
                _owner("a"),
                ExpiryTick(10),
                DurationTicks(10),
                (),
            ),
            second.claim_commit(
                reservation,
                plan,
                approval,
                _owner("b"),
                ExpiryTick(10),
                DurationTicks(10),
                (),
            ),
        )

        assert {
            first_claim.state,
            second_claim.state,
        } == {
            DurableCommitClaimState.OWNER,
            DurableCommitClaimState.ATTACHED,
        }
        owner_claim = next(
            item
            for item in (first_claim, second_claim)
            if item.state is DurableCommitClaimState.OWNER
        )
        assert owner_claim.lease is not None
        assert (
            await first.inspect(
                DurableRequestAccess(reservation.request_id, identity)
            )
        ).lifecycle is LifecyclePhase.COMMIT_STARTED

    run(scenario())


def test_claim_accepts_only_complete_phase_five_broker_attestations() -> None:
    """Reject altered Phase 5 claims before one valid durable consume."""

    async def scenario() -> None:
        sealed, grant, approvals = await _approved_with_service(step_count=1)
        wrong_route_identity = DurableRequestIdentity(
            sealed.binding.subject.tenant,
            sealed.binding.subject.principal,
            sealed.binding.request.execution_id,
            PolicyRouteId("route-forged"),
            RetransmissionKey("phase-eight-forged-route"),
        )
        plan_from_sealed = DurablePlanReference(
            sealed.plan_id,
            sealed.binding.request_digest,
            sealed.fingerprint.digest(),
            sealed.review.diff.digest,
            sealed.binding.target.context_id,
            sealed.binding.target.workspace_id,
            sealed.binding.target.domain_id,
            tuple(
                DurableStepBinding(step_id, lineage_id)
                for step_id, lineage_id in _sealed_journal_steps(sealed)
            ),
        )
        with pytest.raises(DurableStoreError) as route_denied:
            await PhaseFiveDurableApprovalIssuer(
                approvals, _APPROVAL_AUTHORITY
            ).issue(
                wrong_route_identity,
                plan_from_sealed,
                grant,
                sealed,
                sealed.binding.subject,
            )
        assert (
            route_denied.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH
        )
        identity, digest, plan, approval = await _phase_five_approval()
        store = InMemoryDurablePatchStore(_backend())
        reservation = await store.reserve(identity, digest)
        await store.persist_plan(reservation, plan)
        altered = (
            replace(
                approval,
                grant_id=PatchGrantId("grant_" + "f" * 16),
            ),
            replace(
                approval,
                fingerprint_digest=_digest("e"),
            ),
            replace(
                approval,
                context_id=PatchContextId("context_" + "f" * 16),
            ),
            replace(
                approval,
                reviewers=(PatchPrincipalId("reviewer-forged"),),
            ),
        )
        for forged in altered:
            with pytest.raises(DurableStoreError) as denied:
                await store.claim_commit(
                    reservation,
                    plan,
                    forged,
                    _owner("a"),
                    ExpiryTick(10),
                    DurationTicks(20),
                    (),
                )
            assert denied.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH
        claimed = await store.claim_commit(
            reservation,
            plan,
            approval,
            _owner("a"),
            ExpiryTick(10),
            DurationTicks(20),
            (),
        )
        assert claimed.state is DurableCommitClaimState.OWNER

    run(scenario())


def test_lease_renewal_expiry_and_replacement_fence_old_owner() -> None:
    """Renew one lease, then fence it after expiry before replacement work."""

    async def scenario() -> None:
        store, backend, identity, reservation, plan, lease = await _claimed(
            step_count=1
        )
        assert await store.is_current_fence(lease, ExpiryTick(11))
        renewed = await store.renew_lease(
            lease, ExpiryTick(20), DurationTicks(11)
        )
        assert renewed.fence == lease.fence
        assert not await store.is_current_fence(lease, ExpiryTick(21))
        with pytest.raises(DurableStoreError) as raised:
            await store.replace_expired_owner(
                reservation,
                renewed,
                _owner("b"),
                ExpiryTick(30),
                DurationTicks(10),
            )
        assert raised.value.code is DurableStoreErrorCode.LEASE_EXPIRED

        replacement = await InMemoryDurablePatchStore(
            backend
        ).replace_expired_owner(
            reservation,
            renewed,
            _owner("b"),
            ExpiryTick(31),
            DurationTicks(10),
        )
        assert replacement.fence == SequenceNumber(lease.fence.value + 1)
        assert not await store.is_current_fence(renewed, ExpiryTick(32))
        with pytest.raises(DurableStoreError) as raised:
            await store.append_step(
                renewed,
                DurableJournalCursor(
                    reservation.request_id, SequenceNumber(0)
                ),
                plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(32),
            )
        assert raised.value.code is DurableStoreErrorCode.FENCED
        assert await store.is_current_fence(replacement, ExpiryTick(32))

    run(scenario())


def test_expired_reaped_owner_remains_exclusive_until_settlement() -> None:
    """Reject a competing claim until original recovery becomes terminal."""
    backend = _backend()
    run(
        _exclusive_recovery_contract(
            InMemoryDurablePatchStore(backend),
            InMemoryDurablePatchStore(backend),
        )
    )


def test_worker_transitions_require_the_exact_renewed_lease() -> None:
    """Fence stale worker transitions while permitting the renewed lease."""
    run(
        _worker_transition_lease_parity_contract(
            InMemoryDurablePatchStore(_backend())
        )
    )


def test_expired_owner_recovery_never_advances_a_newer_domain_fence() -> None:
    """Fail recovery CAS without changing a newer domain owner epoch."""

    async def scenario() -> None:
        store, backend, _, reservation, _, lease = await _claimed(step_count=1)
        newer = replace(
            lease,
            owner_id=_owner("d"),
            fence=SequenceNumber(lease.fence.value + 1),
            expires_at=ExpiryTick(100),
        )
        backend.fences[lease.domain_id] = newer.fence.value
        backend.active_leases[lease.domain_id] = newer
        with pytest.raises(DurableStoreError) as fenced:
            await store.replace_expired_owner(
                reservation,
                lease,
                _owner("e"),
                ExpiryTick(lease.expires_at.value),
                DurationTicks(10),
            )
        assert fenced.value.code is DurableStoreErrorCode.FENCED
        assert backend.fences[lease.domain_id] == newer.fence.value
        assert backend.active_leases[lease.domain_id] == newer

    run(scenario())


def test_pgsql_claim_and_recovery_lock_domain_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquire one domain lock before either request-row lock path."""

    class Selected(Exception):
        """Stop the simulated transaction after recording lock order."""

    calls: list[str] = []

    async def lock_domain(cursor: object, domain_id: PatchDomainId) -> None:
        """Record the domain advisory lock boundary."""
        del cursor, domain_id
        calls.append("domain")

    async def select_reservation(
        cursor: object, reservation: DurableReservation
    ) -> dict[str, object]:
        """Record the request-row lock and stop the transaction."""
        del cursor, reservation
        calls.append("request")
        raise Selected

    async def transaction(
        operation: str,
        callback: Callable[[object], Awaitable[object]],
    ) -> object:
        """Execute one operation against a deterministic fake cursor."""
        del operation
        return await callback(object())

    identity = _identity("a")
    digest = _digest("a")
    reservation = DurableReservation(
        PatchRequestId("request_" + "a" * 16), identity, digest, False
    )
    plan = _plan(digest, "a", step_count=1)
    approval = _approval(identity, digest, plan, "a")
    lease = DurableCommitLease(
        reservation.request_id,
        plan.domain_id,
        _owner("a"),
        SequenceNumber(1),
        ExpiryTick(20),
    )
    store = PgsqlDurablePatchStore(
        type("Pool", (), {"connection": lambda self: None})(),
        approval_verifier=_APPROVAL_AUTHORITY,
    )
    monkeypatch.setattr(store, "_transaction", transaction)
    monkeypatch.setattr(pgsql_durable, "_lock_domain", lock_domain)
    monkeypatch.setattr(
        pgsql_durable, "_select_reservation_for_update", select_reservation
    )

    with pytest.raises(Selected):
        run(
            store.claim_commit(
                reservation,
                plan,
                approval,
                _owner("b"),
                ExpiryTick(10),
                DurationTicks(10),
                (),
            )
        )
    assert calls == ["domain", "request"]
    calls.clear()
    with pytest.raises(Selected):
        run(
            store.replace_expired_owner(
                reservation,
                lease,
                _owner("b"),
                ExpiryTick(20),
                DurationTicks(10),
            )
        )
    assert calls == ["domain", "request"]


def test_pgsql_worker_transitions_bind_exact_lease_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind both worker transition CAS operations to the complete lease."""

    class Cursor:
        """Record one worker transition CAS statement and its parameters."""

        def __init__(self, row: dict[str, object] | None) -> None:
            """Return one configured CAS result for the transition."""
            self.row = row
            self.calls: list[tuple[str, object]] = []

        async def execute(self, statement: str, parameters: object) -> None:
            """Record one parameterized CAS statement."""
            self.calls.append((statement, parameters))

        async def fetchone(self) -> dict[str, object] | None:
            """Return the configured update result."""
            return self.row

    cursor = Cursor(None)

    async def transaction(
        operation: str,
        callback: Callable[[object], Awaitable[object]],
    ) -> object:
        """Execute one public transition against the current fake cursor."""
        del operation
        return await callback(cursor)

    identity = _identity("c")
    digest = _digest("c")
    reservation = DurableReservation(
        PatchRequestId("request_" + "c" * 16), identity, digest, False
    )
    plan = _plan(digest, "c", step_count=1)
    stale = DurableCommitLease(
        reservation.request_id,
        plan.domain_id,
        _owner("c"),
        SequenceNumber(1),
        ExpiryTick(20),
    )
    renewed = replace(stale, expires_at=ExpiryTick(32))
    binding = DurableWorkerBinding(
        "session-c",
        "channel-c",
        "implementation-c",
        _digest("a"),
        _digest("b"),
    )
    store = PgsqlDurablePatchStore(
        type("Pool", (), {"connection": lambda self: None})(),
        approval_verifier=_APPROVAL_AUTHORITY,
    )
    monkeypatch.setattr(store, "_transaction", transaction)

    with pytest.raises(DurableStoreError) as stale_reaped:
        run(store.mark_worker_reaped(stale, binding))
    assert stale_reaped.value.code is DurableStoreErrorCode.FENCED
    assert cursor.calls == [
        (
            pgsql_durable._MARK_WORKER_REAPED_SQL,
            (
                stale.request_id.value,
                stale.domain_id.value,
                stale.owner_id.value,
                stale.fence.value,
                stale.expires_at.value,
                binding.fingerprint(),
            ),
        )
    ]

    cursor = Cursor({"request_id": renewed.request_id.value})
    run(store.mark_worker_reaped(renewed, binding))
    assert cursor.calls == [
        (
            pgsql_durable._MARK_WORKER_REAPED_SQL,
            (
                renewed.request_id.value,
                renewed.domain_id.value,
                renewed.owner_id.value,
                renewed.fence.value,
                renewed.expires_at.value,
                binding.fingerprint(),
            ),
        )
    ]

    cursor = Cursor(None)
    with pytest.raises(DurableStoreError) as stale_absent:
        run(store.mark_worker_absent(stale))
    assert stale_absent.value.code is DurableStoreErrorCode.FENCED
    assert cursor.calls == [
        (
            pgsql_durable._MARK_WORKER_ABSENT_SQL,
            (
                stale.request_id.value,
                stale.domain_id.value,
                stale.owner_id.value,
                stale.fence.value,
                stale.expires_at.value,
            ),
        )
    ]

    cursor = Cursor({"request_id": renewed.request_id.value})
    run(store.mark_worker_absent(renewed))
    assert cursor.calls == [
        (
            pgsql_durable._MARK_WORKER_ABSENT_SQL,
            (
                renewed.request_id.value,
                renewed.domain_id.value,
                renewed.owner_id.value,
                renewed.fence.value,
                renewed.expires_at.value,
            ),
        )
    ]
    assert (
        'AND "lease_expires_at" = %s' in pgsql_durable._MARK_WORKER_REAPED_SQL
    )
    assert (
        'AND "lease_expires_at" = %s' in pgsql_durable._MARK_WORKER_ABSENT_SQL
    )


def test_journal_requires_exact_cas_terminal_steps_and_artifact_cleanup() -> (
    None
):
    """Persist monotonic journals and one terminal outbox record atomically."""

    async def scenario() -> None:
        artifact = _artifact()
        store, _, identity, reservation, plan, lease = await _claimed(
            artifact_ids=(artifact,)
        )
        cursor = (
            await store.inspect(
                DurableRequestAccess(reservation.request_id, identity)
            )
        ).journal.cursor
        with pytest.raises(DurableStoreError) as raised:
            await store.append_step(
                lease,
                cursor,
                plan.steps[0].step_id,
                CommitStepState.COMMITTED,
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT

        journal = await store.append_step(
            lease,
            cursor,
            plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(11),
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.append_step(
                lease,
                cursor,
                plan.steps[0].step_id,
                CommitStepState.COMMITTED,
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT

        journal = await store.append_step(
            lease,
            journal.cursor,
            plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(11),
        )
        journal = await store.append_step(
            lease,
            journal.cursor,
            plan.steps[1].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(11),
        )
        journal = await store.append_step(
            lease,
            journal.cursor,
            plan.steps[1].step_id,
            CommitStepState.NOT_COMMITTED,
            ExpiryTick(11),
        )
        journal = await store.append_artifact(
            lease,
            journal.cursor,
            artifact,
            DurableArtifactState.PRESENT,
            ExpiryTick(11),
        )
        journal = await store.append_artifact(
            lease,
            journal.cursor,
            artifact,
            DurableArtifactState.REMOVED,
            ExpiryTick(11),
        )
        result = _result(
            reservation.request_id,
            plan,
            MutationState.PARTIALLY_COMMITTED,
            ArtifactState.CLEANED,
        )
        terminal = await store.settle(
            lease,
            journal.cursor,
            result,
            _correlation(),
            ExpiryTick(12),
        )
        assert terminal.result is result
        replay = await store.settle(
            lease,
            journal.cursor,
            result,
            _correlation(),
            ExpiryTick(30),
        )
        assert replay is terminal
        assert await store.outbox(
            DurableRequestAccess(reservation.request_id, identity),
            SequenceNumber(0),
            10,
        ) == (terminal.outbox,)
        with pytest.raises(DurableStoreError) as raised:
            await store.settle(
                lease,
                journal.cursor,
                _result(
                    reservation.request_id,
                    plan,
                    MutationState.INDETERMINATE,
                    ArtifactState.CLEANED,
                ),
                _correlation(),
                ExpiryTick(30),
            )
        assert raised.value.code is DurableStoreErrorCode.TERMINAL_CONFLICT

    run(scenario())


def test_terminal_truth_cannot_relabel_leaked_or_unknown_artifacts() -> None:
    """Reject cleaned terminal truth that differs from durable artifacts."""

    async def scenario() -> None:
        for token, state in (
            ("c", DurableArtifactState.LEAKED),
            ("d", DurableArtifactState.UNKNOWN),
        ):
            artifact = _artifact(token)
            store, _, identity, reservation, plan, lease = await _claimed(
                token=token,
                step_count=1,
                artifact_ids=(artifact,),
            )
            journal = await store.append_step(
                lease,
                (
                    await store.inspect(
                        DurableRequestAccess(reservation.request_id, identity)
                    )
                ).journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(11),
            )
            journal = await store.append_step(
                lease,
                journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.COMMITTED,
                ExpiryTick(11),
            )
            journal = await store.append_artifact(
                lease,
                journal.cursor,
                artifact,
                DurableArtifactState.PRESENT,
                ExpiryTick(11),
            )
            journal = await store.append_artifact(
                lease,
                journal.cursor,
                artifact,
                state,
                ExpiryTick(11),
            )
            with pytest.raises(DurableStoreError) as denied:
                await store.settle(
                    lease,
                    journal.cursor,
                    _result(
                        reservation.request_id,
                        plan,
                        MutationState.COMMITTED,
                        ArtifactState.CLEANED,
                    ),
                    _correlation(token),
                    ExpiryTick(12),
                )
            assert denied.value.code is DurableStoreErrorCode.TERMINAL_CONFLICT

    run(scenario())


def test_pending_access_await_cancellation_and_outbox_preserve_branch() -> (
    None
):
    """Suspend and resume only the original authorized pending branch."""

    async def scenario() -> None:
        store, backend, identity, reservation, plan, lease = await _claimed(
            step_count=1
        )
        journal = await store.append_step(
            lease,
            DurableJournalCursor(reservation.request_id, SequenceNumber(0)),
            plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(11),
        )
        journal = await store.append_step(
            lease,
            journal.cursor,
            plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(11),
        )
        pending_request = _pending()
        pending = await store.suspend(lease, pending_request, ExpiryTick(12))
        access = DurablePendingAccess(
            DurableRequestAccess(reservation.request_id, identity),
            pending_request.pending_operation_id,
            pending_request.correlation_id,
        )
        assert await store.inspect_pending(access) == pending
        cancellation = await InMemoryDurablePatchStore(
            backend
        ).request_cancellation(access.request)
        assert cancellation.cancellation_requested
        pending_after_cancellation = await store.inspect_pending(access)
        assert isinstance(pending_after_cancellation, DurablePendingRecord)
        assert pending_after_cancellation.cancellation_requested

        wrong = DurablePendingAccess(
            DurableRequestAccess(reservation.request_id, _identity("b")),
            pending_request.pending_operation_id,
            pending_request.correlation_id,
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.inspect_pending(wrong)
        assert raised.value.code is DurableStoreErrorCode.ACCESS_DENIED
        wrong_correlation = DurablePendingAccess(
            access.request,
            access.pending_operation_id,
            _correlation("b"),
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.inspect_pending(wrong_correlation)
        assert raised.value.code is DurableStoreErrorCode.ACCESS_DENIED

        waiter = create_task(store.await_terminal(access))
        await sleep(0)
        result = _result(reservation.request_id, plan, MutationState.COMMITTED)
        terminal = await store.settle(
            lease,
            journal.cursor,
            result,
            pending_request.correlation_id,
            ExpiryTick(13),
        )
        assert await waiter is terminal
        assert await store.inspect_pending(access) is terminal
        assert (
            terminal.pending_operation_id
            == pending_request.pending_operation_id
        )
        wrong_handle = DurablePendingAccess(
            access.request,
            PatchPendingOperationId("pending_" + "f" * 16),
            pending_request.correlation_id,
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.inspect_pending(wrong_handle)
        assert raised.value.code is DurableStoreErrorCode.ACCESS_DENIED
        events = await store.outbox(access.request, SequenceNumber(0), 10)
        assert tuple(item.lifecycle for item in events) == (
            LifecyclePhase.SETTLEMENT_PENDING,
            LifecyclePhase.REQUEST_COMPLETED,
        )
        assert tuple(item.sequence for item in events) == (
            SequenceNumber(1),
            SequenceNumber(2),
        )

    run(scenario())


def test_incomplete_journal_fails_closed_to_exact_terminal_truth() -> None:
    """Reject missing planned truth and accept journal-derived uncertainty."""

    async def scenario() -> None:
        store, _, _, reservation, plan, lease = await _claimed(step_count=1)
        cursor = DurableJournalCursor(
            reservation.request_id, SequenceNumber(0)
        )
        journal = await store.append_step(
            lease,
            cursor,
            plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(11),
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.settle(
                lease,
                journal.cursor,
                _result(
                    reservation.request_id,
                    plan,
                    MutationState.INDETERMINATE,
                ),
                _correlation(),
                ExpiryTick(12),
            )
        assert raised.value.code is DurableStoreErrorCode.JOURNAL_INCOMPLETE
        journal = await store.append_step(
            lease,
            journal.cursor,
            plan.steps[0].step_id,
            CommitStepState.UNKNOWN,
            ExpiryTick(12),
        )
        terminal = await store.settle(
            lease,
            journal.cursor,
            _result(
                reservation.request_id,
                plan,
                MutationState.INDETERMINATE,
            ),
            _correlation(),
            ExpiryTick(12),
        )
        assert (
            terminal.result.truth.mutation_state is MutationState.INDETERMINATE
        )

    run(scenario())


def test_retention_is_encrypted_audience_limited_bounded_and_expired() -> None:
    """Retain ciphertext only through audience, expiry, and capacity policy."""

    async def scenario() -> None:
        limits = DurableStoreLimits(
            max_journal_entries=8,
            max_artifacts=2,
            max_retention_records=1,
            max_retention_bytes=ByteSize(4096),
        )
        key = _retention_key("a")
        cipher = AesGcmDurableRetentionCipher(
            InMemoryDurableRetentionKeyResolver(key.key_id, {key.key_id: key})
        )
        backend = _retention_backend(key, limits)
        store = InMemoryDurablePatchStore(backend)
        identity = _identity()
        reservation = await store.reserve(identity, _digest("a"))
        retention_id = PatchRetentionRecordId("retained_" + "a" * 16)
        sealed = await cipher.seal(
            b"cipher",
            DurableRetentionBinding(
                reservation.request_id,
                retention_id,
                DurableRetentionKind.REVIEW_ARTIFACT,
            ),
        )
        retained = DurableRetentionRecord(
            retention_id,
            DurableRetentionKind.REVIEW_ARTIFACT,
            sealed.key_id,
            sealed.value,
            DurableRetentionPolicy(ExpiryTick(20), False),
        )
        await store.put_retention(reservation, retained)
        assert "cipher" not in repr(retained.value)
        approver = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, identity),
        )
        assert (
            await store.get_retention(
                approver, retained.retention_id, ExpiryTick(19)
            )
            is retained
        )
        with pytest.raises(TypeError):
            getattr(DurableRetentionAccess, "__init__")(
                approver, approver.request, Audience.PUBLIC
            )
        wrong_binding_id = PatchRetentionRecordId("retained_" + "c" * 16)
        wrong_binding_value = await cipher.seal(
            b"wrong-binding",
            DurableRetentionBinding(
                reservation.request_id,
                retained.retention_id,
                DurableRetentionKind.REVIEW_ARTIFACT,
            ),
        )
        invalid_records = (
            DurableRetentionRecord(
                wrong_binding_id,
                DurableRetentionKind.REVIEW_ARTIFACT,
                wrong_binding_value.key_id,
                wrong_binding_value.value,
                retained.policy,
            ),
            DurableRetentionRecord(
                wrong_binding_id,
                DurableRetentionKind.REVIEW_ARTIFACT,
                PatchRetentionKeyId("retention_" + "b" * 16),
                retained.value,
                retained.policy,
            ),
            DurableRetentionRecord(
                wrong_binding_id,
                DurableRetentionKind.REVIEW_ARTIFACT,
                key.key_id,
                EncryptedRetentionValue(b"plaintext-is-not-an-envelope"),
                retained.policy,
            ),
        )
        for invalid in invalid_records:
            with pytest.raises(DurableStoreError) as denied:
                await store.put_retention(reservation, invalid)
            assert denied.value.code is DurableStoreErrorCode.RETENTION_DENIED
        second_id = PatchRetentionRecordId("retained_" + "b" * 16)
        second = await cipher.seal(
            b"second",
            DurableRetentionBinding(
                reservation.request_id,
                second_id,
                DurableRetentionKind.SEALED_PLAN,
            ),
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.put_retention(
                reservation,
                DurableRetentionRecord(
                    second_id,
                    DurableRetentionKind.SEALED_PLAN,
                    second.key_id,
                    second.value,
                    retained.policy,
                ),
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_LIMIT
        cleanup = await store.cleanup_retention(ExpiryTick(20))
        assert cleanup.records_deleted == 1
        assert cleanup.bytes_deleted == retained.value.size()
        with pytest.raises(DurableStoreError) as raised:
            await store.get_retention(
                approver, retained.retention_id, ExpiryTick(20)
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED

    run(scenario())


def test_retention_key_rotation_and_terminal_cleanup_survive_restart() -> None:
    """Retain versioned ciphertext and delete terminal-selected records."""

    async def scenario() -> None:
        first_key = _retention_key("a")
        rotated_key = _retention_key("b")
        resolver = InMemoryDurableRetentionKeyResolver(
            rotated_key.key_id,
            {
                first_key.key_id: first_key,
                rotated_key.key_id: rotated_key,
            },
        )
        cipher = AesGcmDurableRetentionCipher(resolver)
        backend = InMemoryDurablePatchBackend(
            approval_verifier=_APPROVAL_AUTHORITY,
            retention_authorizer=StaticDurableRetentionAuthorizer(
                frozenset((Audience.APPROVER,))
            ),
            retention_validator=AesGcmDurableRetentionEnvelopeValidator(
                cipher
            ),
        )
        store, backend, identity, reservation, plan, lease = await _claimed(
            step_count=1, backend=backend
        )
        policy = DurableRetentionPolicy(ExpiryTick(100), True)
        first_id = PatchRetentionRecordId("retained_" + "a" * 16)
        rotated_id = PatchRetentionRecordId("retained_" + "b" * 16)
        old_cipher = AesGcmDurableRetentionCipher(
            InMemoryDurableRetentionKeyResolver(
                first_key.key_id,
                {
                    first_key.key_id: first_key,
                    rotated_key.key_id: rotated_key,
                },
            )
        )
        first_value = await old_cipher.seal(
            b"old-key-ciphertext",
            DurableRetentionBinding(
                reservation.request_id,
                first_id,
                DurableRetentionKind.SEALED_PLAN,
            ),
        )
        rotated_value = await cipher.seal(
            b"new-key-ciphertext",
            DurableRetentionBinding(
                reservation.request_id,
                rotated_id,
                DurableRetentionKind.REVIEW_ARTIFACT,
            ),
        )
        first = DurableRetentionRecord(
            first_id,
            DurableRetentionKind.SEALED_PLAN,
            first_value.key_id,
            first_value.value,
            policy,
        )
        rotated = DurableRetentionRecord(
            rotated_id,
            DurableRetentionKind.REVIEW_ARTIFACT,
            rotated_value.key_id,
            rotated_value.value,
            policy,
        )
        await store.put_retention(reservation, first)
        await store.put_retention(reservation, rotated)
        access = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, identity),
        )
        assert (
            await store.get_retention(
                access, first.retention_id, ExpiryTick(11)
            )
            == first
        )
        assert (
            await InMemoryDurablePatchStore(backend).get_retention(
                access, rotated.retention_id, ExpiryTick(11)
            )
            == rotated
        )

        journal = await store.append_step(
            lease,
            DurableJournalCursor(reservation.request_id, SequenceNumber(0)),
            plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(12),
        )
        journal = await store.append_step(
            lease,
            journal.cursor,
            plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(12),
        )
        await store.settle(
            lease,
            journal.cursor,
            _result(reservation.request_id, plan, MutationState.COMMITTED),
            _correlation(),
            ExpiryTick(13),
        )
        fresh = InMemoryDurablePatchStore(backend)
        for record in (first, rotated):
            with pytest.raises(DurableStoreError) as raised:
                await fresh.get_retention(
                    access, record.retention_id, ExpiryTick(14)
                )
            assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED

    run(scenario())


def test_concurrent_journal_and_terminal_outbox_have_one_winner() -> None:
    """Race one journal CAS and terminal replay without duplicate outbox."""

    async def scenario() -> None:
        store, backend, identity, reservation, plan, lease = await _claimed(
            step_count=1
        )
        cursor = DurableJournalCursor(
            reservation.request_id, SequenceNumber(0)
        )
        first, second = await gather(
            store.append_step(
                lease,
                cursor,
                plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(11),
            ),
            InMemoryDurablePatchStore(backend).append_step(
                lease,
                cursor,
                plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(11),
            ),
            return_exceptions=True,
        )
        journals = tuple(
            item
            for item in (first, second)
            if isinstance(item, DurableJournal)
        )
        assert len(journals) == 1
        journal = journals[0]
        journal = await store.append_step(
            lease,
            journal.cursor,
            plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(12),
        )
        result = _result(reservation.request_id, plan, MutationState.COMMITTED)
        settled = await gather(
            store.settle(
                lease, journal.cursor, result, _correlation(), ExpiryTick(13)
            ),
            InMemoryDurablePatchStore(backend).settle(
                lease, journal.cursor, result, _correlation(), ExpiryTick(13)
            ),
        )
        assert settled[0] is settled[1]
        events = await store.outbox(
            DurableRequestAccess(reservation.request_id, identity),
            SequenceNumber(0),
            10,
        )
        assert len(events) == 1
        assert events[0] == settled[0].outbox

    run(scenario())


def test_durable_crypto_and_attestation_invariants_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed durable keys, envelopes, and broker attestations."""
    with pytest.raises(DurableStoreError) as raised:
        DurableApprovalSigningKey(b"short")
    assert raised.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH
    key = DurableApprovalSigningKey(b"a" * 32)
    assert repr(key) == "DurableApprovalSigningKey(<redacted>)"
    authority = HmacDurableApprovalAuthority(key)
    with pytest.raises(DurableStoreError) as raised:
        getattr(HmacDurableApprovalAuthority, "__init__")(authority, object())
    assert raised.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH
    assert isinstance(
        HmacDurableApprovalAuthority.random(), HmacDurableApprovalAuthority
    )
    with pytest.raises(DurableStoreError) as raised:
        getattr(authority, "seal")(object())
    assert raised.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH
    with pytest.raises(DurableStoreError) as raised:
        getattr(PhaseFiveDurableApprovalIssuer, "__init__")(
            object(), object(), authority
        )
    assert raised.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH

    async def scenario() -> None:
        retention_key = _retention_key("a")
        binding = DurableRetentionBinding(
            PatchRequestId("request_" + "a" * 16),
            PatchRetentionRecordId("retained_" + "a" * 16),
            DurableRetentionKind.REVIEW_ARTIFACT,
        )
        record = DurableRetentionRecord(
            binding.retention_id,
            binding.kind,
            retention_key.key_id,
            EncryptedRetentionValue(b"envelope"),
            DurableRetentionPolicy(ExpiryTick(20), False),
        )
        with pytest.raises(DurableStoreError) as raised:
            DurableRetentionKey(retention_key.key_id, b"short")
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        assert "<redacted>" in repr(retention_key)
        with pytest.raises(DurableStoreError) as raised:
            DurableRetentionBinding(
                getattr(binding, "retention_id"),
                binding.retention_id,
                binding.kind,
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as raised:
            getattr(binding, "associated_data")(object())
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as raised:
            DurableEncryptedRetention(
                retention_key.key_id, getattr(binding, "retention_id")
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as raised:
            getattr(AesGcmDurableRetentionCipher, "__init__")(
                object(), object()
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as raised:
            getattr(AesGcmDurableRetentionEnvelopeValidator, "__init__")(
                object(), object()
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as raised:
            StaticDurableRetentionAuthorizer(frozenset())
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as raised:
            InMemoryDurableRetentionKeyResolver(
                PatchRetentionKeyId("retention_" + "b" * 16),
                {retention_key.key_id: retention_key},
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED

        class FailingResolver:
            """Fail every key lookup without exposing key material."""

            async def active_key(self) -> DurableRetentionKey:
                """Fail the active-key lookup."""
                raise RuntimeError("unavailable")

            async def read_key(
                self, key_id: PatchRetentionKeyId
            ) -> DurableRetentionKey:
                """Fail the historical-key lookup."""
                del key_id
                raise RuntimeError("unavailable")

        failing_cipher = AesGcmDurableRetentionCipher(FailingResolver())
        with pytest.raises(DurableStoreError) as raised:
            await failing_cipher.seal(b"value", binding)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        valid_cipher = AesGcmDurableRetentionCipher(
            InMemoryDurableRetentionKeyResolver(
                retention_key.key_id, {retention_key.key_id: retention_key}
            )
        )
        sealed = await valid_cipher.seal(b"value", binding)
        with pytest.raises(DurableStoreError) as raised:
            await failing_cipher.open(sealed, binding)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        tampered = DurableEncryptedRetention(
            sealed.key_id,
            EncryptedRetentionValue(
                sealed.value._ciphertext[:-1]
                + bytes((sealed.value._ciphertext[-1] ^ 1,))
            ),
        )
        with pytest.raises(DurableStoreError) as raised:
            await valid_cipher.open(tampered, binding)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        monkeypatch.setattr(retention, "_MAX_PLAINTEXT_BYTES", 4)
        with pytest.raises(DurableStoreError) as raised:
            await valid_cipher.open(sealed, binding)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_LIMIT
        monkeypatch.setattr(retention, "_MAX_PLAINTEXT_BYTES", 1_048_548)
        resolver = InMemoryDurableRetentionKeyResolver(
            retention_key.key_id, {retention_key.key_id: retention_key}
        )
        with pytest.raises(DurableStoreError) as raised:
            await resolver.read_key(
                PatchRetentionKeyId("retention_" + "b" * 16)
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as raised:
            await getattr(valid_cipher, "seal")(object(), binding)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_LIMIT
        with pytest.raises(DurableStoreError) as raised:
            await getattr(valid_cipher, "open")(object(), binding)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as raised:
            await getattr(
                AesGcmDurableRetentionEnvelopeValidator(valid_cipher),
                "validate",
            )(object(), record)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as raised:
            await getattr(
                StaticDurableRetentionAuthorizer(
                    frozenset((Audience.APPROVER,))
                ),
                "audiences_for",
            )(object(), record.kind)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED

        class InvalidResolver:
            """Return untrusted key shapes for closed-boundary testing."""

            async def active_key(self) -> object:
                """Return an invalid active-key value."""
                return object()

            async def read_key(
                self, key_id: PatchRetentionKeyId
            ) -> DurableRetentionKey:
                """Return a mismatched historical key identifier."""
                del key_id
                return retention_key

        invalid_cipher = getattr(retention, "AesGcmDurableRetentionCipher")(
            InvalidResolver()
        )
        with pytest.raises(DurableStoreError) as raised:
            await invalid_cipher.seal(b"value", binding)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        mismatched = DurableEncryptedRetention(
            PatchRetentionKeyId("retention_" + "b" * 16), sealed.value
        )
        with pytest.raises(DurableStoreError) as raised:
            await invalid_cipher.open(mismatched, binding)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED

        class InterruptingResolver:
            """Propagate explicit process interrupts from key resolution."""

            async def active_key(self) -> DurableRetentionKey:
                """Interrupt the active-key resolution boundary."""
                raise RuntimeError("interrupted")

            async def read_key(
                self, key_id: PatchRetentionKeyId
            ) -> DurableRetentionKey:
                """Interrupt the historical-key resolution boundary."""
                del key_id
                raise RuntimeError("interrupted")

        interrupted_cipher = AesGcmDurableRetentionCipher(
            InterruptingResolver()
        )
        monkeypatch.setattr(
            retention, "KeyboardInterrupt", RuntimeError, raising=False
        )
        with pytest.raises(RuntimeError):
            await interrupted_cipher.seal(b"value", binding)
        with pytest.raises(RuntimeError):
            await interrupted_cipher.open(sealed, binding)

    run(scenario())


def test_phase_five_durable_issuer_rejects_subject_and_broker_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate subject mismatch and broker validation failures to denial."""

    async def scenario() -> None:
        sealed, grant, approvals = await _approved_with_service(step_count=1)
        identity = DurableRequestIdentity(
            sealed.binding.subject.tenant,
            sealed.binding.subject.principal,
            sealed.binding.request.execution_id,
            sealed.binding.final.approval.route,
            RetransmissionKey("phase-eight-issuer-failure"),
        )
        plan = DurablePlanReference(
            sealed.plan_id,
            sealed.binding.request_digest,
            sealed.fingerprint.digest(),
            sealed.review.diff.digest,
            sealed.binding.target.context_id,
            sealed.binding.target.workspace_id,
            sealed.binding.target.domain_id,
            tuple(
                DurableStepBinding(step_id, lineage_id)
                for step_id, lineage_id in _sealed_journal_steps(sealed)
            ),
        )
        issuer = PhaseFiveDurableApprovalIssuer(approvals, _APPROVAL_AUTHORITY)
        with pytest.raises(DurableStoreError) as raised:
            await getattr(issuer, "issue")(
                identity, plan, grant, sealed, object()
            )
        assert raised.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH

        async def unavailable(*args: object) -> None:
            """Model a broker validation outage without a grant result."""
            del args
            raise RuntimeError("broker unavailable")

        monkeypatch.setattr(approvals, "validate_grant", unavailable)
        with pytest.raises(DurableStoreError) as raised:
            await issuer.issue(
                identity,
                plan,
                grant,
                sealed,
                sealed.binding.subject,
            )
        assert raised.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH

        async def interrupted(*args: object) -> None:
            """Preserve interruption instead of converting it to denial."""
            del args
            raise RuntimeError("interrupted")

        monkeypatch.setattr(approvals, "validate_grant", interrupted)
        monkeypatch.setattr(
            durable_approval, "KeyboardInterrupt", RuntimeError, raising=False
        )
        with pytest.raises(RuntimeError):
            await issuer.issue(
                identity,
                plan,
                grant,
                sealed,
                sealed.binding.subject,
            )

    run(scenario())


def test_durable_store_denials_and_opaque_values_are_closed() -> None:
    """Reject unconfigured durable boundaries without exposing values."""

    async def scenario() -> None:
        identity = _identity("a")
        digest = _digest("a")
        plan = _plan(digest)
        approval = _approval(identity, digest, plan)
        with pytest.raises(DurableStoreError) as raised:
            DenyDurableApprovalVerifier().verify(approval)
        assert raised.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH
        with pytest.raises(DurableStoreError) as raised:
            getattr(DenyDurableApprovalVerifier(), "verify")(object())
        assert raised.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH
        encrypted = EncryptedRetentionValue(b"opaque")
        assert repr(encrypted) == "EncryptedRetentionValue(<redacted>)"
        assert str(encrypted) == "<redacted>"
        assert encrypted.size() == ByteSize(6)
        assert encrypted.digest().algorithm == "sha256"
        assert encrypted.digest().value != "opaque"
        record = DurableRetentionRecord(
            PatchRetentionRecordId("retained_" + "a" * 16),
            DurableRetentionKind.REVIEW_ARTIFACT,
            PatchRetentionKeyId("retention_" + "a" * 16),
            encrypted,
            DurableRetentionPolicy(ExpiryTick(10), False),
        )
        authorizer = DenyDurableRetentionAuthorizer()
        assert (
            await authorizer.audiences_for(
                identity, DurableRetentionKind.REVIEW_ARTIFACT
            )
            == frozenset()
        )
        with pytest.raises(DurableStoreError) as raised:
            await getattr(authorizer, "audiences_for")(object(), record.kind)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        validator = DenyDurableRetentionEnvelopeValidator()
        with pytest.raises(DurableStoreError) as raised:
            await validator.validate(
                PatchRequestId("request_" + "a" * 16), record
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as raised:
            await getattr(validator, "validate")(object(), record)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as raised:
            EncryptedRetentionValue(getattr(record, "policy"))
        assert raised.value.code is DurableStoreErrorCode.RETENTION_LIMIT
        with pytest.raises(DurableStoreError) as raised:
            DurableRetentionCleanup(-1, ByteSize(0))
        assert raised.value.code is DurableStoreErrorCode.RETENTION_CONFLICT
        with pytest.raises(DurableStoreError) as raised:
            DurableStoreLimits(max_journal_entries=0)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_LIMIT

    run(scenario())


def test_durable_store_value_contracts_reject_invalid_state() -> None:
    """Reject malformed durable records before they become recovery state."""
    identity = _identity("b")
    digest = _digest("b")
    reservation = DurableReservation(
        PatchRequestId("request_" + "b" * 16), identity, digest, False
    )
    plan = _plan(digest, "b", step_count=1)
    lease = DurableCommitLease(
        reservation.request_id,
        plan.domain_id,
        _owner("b"),
        SequenceNumber(1),
        ExpiryTick(20),
    )
    cursor = DurableJournalCursor(reservation.request_id, SequenceNumber(1))
    artifact = _artifact("b")
    step = DurableStepJournalEntry(
        cursor,
        plan.steps[0].step_id,
        plan.steps[0].lineage_id,
        CommitStepState.PLANNED,
    )
    artifact_entry = DurableArtifactJournalEntry(
        cursor, artifact, DurableArtifactState.INTENDED
    )
    encrypted = EncryptedRetentionValue(b"opaque")
    retention = DurableRetentionRecord(
        PatchRetentionRecordId("retained_" + "b" * 16),
        DurableRetentionKind.SEALED_PLAN,
        PatchRetentionKeyId("retention_" + "b" * 16),
        encrypted,
        DurableRetentionPolicy(ExpiryTick(20), True),
    )
    invalid_values = (
        lambda: getattr(durable, "DurableRequestIdentity")(
            object(), object(), object(), object(), object()
        ),
        lambda: getattr(durable, "DurableReservation")(
            object(), object(), object(), object()
        ),
        lambda: getattr(durable, "DurableStepBinding")(object(), object()),
        lambda: DurablePlanReference(
            plan.plan_id,
            plan.canonical_digest,
            plan.fingerprint_digest,
            plan.review_digest,
            plan.context_id,
            plan.workspace_id,
            plan.domain_id,
            (plan.steps[0], plan.steps[0]),
        ),
        lambda: replace(
            _approval(identity, digest, plan, "b"), policy_revision=""
        ),
        lambda: replace(lease, fence=SequenceNumber(0)),
        lambda: getattr(durable, "DurableCommitClaim")(object(), None, None),
        lambda: getattr(durable, "DurableCommitClaim")(
            DurableCommitClaimState.OWNER, None, None
        ),
        lambda: getattr(durable, "DurableCommitClaim")(
            DurableCommitClaimState.ATTACHED, lease, None
        ),
        lambda: getattr(durable, "DurableCommitClaim")(
            DurableCommitClaimState.TERMINAL, None, None
        ),
        lambda: getattr(durable, "DurableJournalCursor")(object(), object()),
        lambda: getattr(durable, "DurableStepJournalEntry")(
            object(), object(), object(), object()
        ),
        lambda: getattr(durable, "DurableArtifactJournalEntry")(
            object(), object(), object()
        ),
        lambda: DurableJournal(cursor, (), ()),
        lambda: getattr(durable, "DurablePendingRequest")(
            object(), object(), object()
        ),
        lambda: getattr(durable, "DurablePendingRecord")(
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        ),
        lambda: getattr(durable, "DurableRequestAccess")(object(), object()),
        lambda: getattr(durable, "DurablePendingAccess")(
            object(), object(), object()
        ),
        lambda: getattr(durable, "DurableOutboxRecord")(
            object(), object(), object(), object(), object()
        ),
        lambda: getattr(durable, "DurableTerminalRecord")(
            object(), object(), None
        ),
        lambda: getattr(durable, "DurableRequestSnapshot")(
            object(),
            None,
            object(),
            None,
            object(),
            None,
            None,
            object(),
            object(),
            object(),
            object(),
        ),
        lambda: getattr(durable, "DurableRetentionPolicy")(object(), object()),
        lambda: getattr(durable, "DurableRetentionRecord")(
            object(), object(), object(), object(), object()
        ),
        lambda: getattr(durable, "DurableRetentionAccess")(object()),
    )
    for invalid in invalid_values:
        with pytest.raises(DurableStoreError):
            invalid()
    assert DurableJournal(cursor, (step,), ()) == DurableJournal(
        cursor, (step,), ()
    )
    assert DurableJournal(cursor, (), (artifact_entry,)) == DurableJournal(
        cursor, (), (artifact_entry,)
    )
    assert retention.value == encrypted


def test_durable_store_rejects_stale_lifecycle_and_branch_conflicts() -> None:
    """Reject stale ownership, terminal relabeling, and bad branch requests."""

    async def scenario() -> None:
        backend = _backend()
        store = InMemoryDurablePatchStore(backend)
        identity = _identity("c")
        digest = _digest("c")
        reservation = await store.reserve(identity, digest)
        plan = _plan(digest, "c", step_count=1)
        with pytest.raises(DurableStoreError) as raised:
            await store.persist_plan(reservation, _plan(_digest("e"), "e"))
        assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH
        await store.persist_plan(reservation, plan)
        with pytest.raises(DurableStoreError) as raised:
            await store.persist_plan(reservation, _plan(digest, "e"))
        assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH

        with pytest.raises(DurableStoreError) as raised:
            await store.claim_commit(
                reservation,
                _plan(digest, "e"),
                _approval(identity, digest, plan, "c"),
                _owner("c"),
                ExpiryTick(10),
                DurationTicks(10),
                (),
            )
        assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH
        record = backend.records[identity]
        record.lifecycle = LifecyclePhase.RECEIVED
        with pytest.raises(DurableStoreError) as raised:
            await store.claim_commit(
                reservation,
                plan,
                _approval(identity, digest, plan, "c"),
                _owner("c"),
                ExpiryTick(10),
                DurationTicks(10),
                (),
            )
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        record.lifecycle = LifecyclePhase.PLANNED
        approval = _approval(identity, digest, plan, "c")
        backend.consumed_grants[approval.grant_id] = PatchRequestId(
            "request_" + "f" * 16
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.claim_commit(
                reservation,
                plan,
                approval,
                _owner("c"),
                ExpiryTick(10),
                DurationTicks(10),
                (),
            )
        assert raised.value.code is DurableStoreErrorCode.APPROVAL_CONSUMED
        backend.consumed_grants.clear()
        claim = await store.claim_commit(
            reservation,
            plan,
            approval,
            _owner("c"),
            ExpiryTick(10),
            DurationTicks(10),
            (),
        )
        assert claim.lease is not None
        with pytest.raises(DurableStoreError) as raised:
            await store.renew_lease(
                claim.lease, ExpiryTick(10), DurationTicks(10)
            )
        assert raised.value.code is DurableStoreErrorCode.LEASE_EXPIRED
        pending = DurablePendingRequest(
            PatchPendingOperationId("pending_" + "c" * 16),
            _correlation("c"),
            DurationTicks(5),
        )
        first_pending = await store.suspend(
            claim.lease, pending, ExpiryTick(11)
        )
        assert (
            await store.suspend(claim.lease, pending, ExpiryTick(11))
            == first_pending
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.suspend(
                claim.lease,
                replace(
                    pending,
                    pending_operation_id=PatchPendingOperationId(
                        "pending_" + "e" * 16
                    ),
                ),
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        renewed = await store.renew_lease(
            claim.lease, ExpiryTick(12), DurationTicks(20)
        )
        assert (
            await store.inspect(
                DurableRequestAccess(reservation.request_id, identity)
            )
        ).pending is not None
        with pytest.raises(DurableStoreError) as raised:
            await store.replace_expired_owner(
                reservation,
                renewed,
                _owner("d"),
                ExpiryTick(31),
                DurationTicks(10),
            )
        assert raised.value.code is DurableStoreErrorCode.LEASE_EXPIRED
        with pytest.raises(DurableStoreError) as raised:
            await store.replace_expired_owner(
                reservation,
                renewed,
                _owner("c"),
                ExpiryTick(32),
                DurationTicks(10),
            )
        assert raised.value.code is DurableStoreErrorCode.FENCED
        replacement = await store.replace_expired_owner(
            reservation,
            renewed,
            _owner("d"),
            ExpiryTick(32),
            DurationTicks(10),
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.append_artifact(
                replacement,
                DurableJournalCursor(
                    reservation.request_id, SequenceNumber(0)
                ),
                _artifact("c"),
                DurableArtifactState.PRESENT,
                ExpiryTick(33),
            )
        assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
        access = DurableRequestAccess(reservation.request_id, identity)
        assert (
            await store.request_cancellation(access)
        ).cancellation_requested
        with pytest.raises(DurableStoreError) as raised:
            await store.outbox(access, SequenceNumber(0), 0)
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT

        terminal_backend = _backend()
        terminal_store = InMemoryDurablePatchStore(terminal_backend)
        terminal_identity = _identity("d")
        terminal_digest = _digest("d")
        terminal_reservation = await terminal_store.reserve(
            terminal_identity, terminal_digest
        )
        terminal_plan = _plan(terminal_digest, "d", step_count=1)
        await terminal_store.persist_plan(terminal_reservation, terminal_plan)
        terminal_claim = await terminal_store.claim_commit(
            terminal_reservation,
            terminal_plan,
            _approval(terminal_identity, terminal_digest, terminal_plan, "d"),
            _owner("d"),
            ExpiryTick(10),
            DurationTicks(20),
            (),
        )
        assert terminal_claim.lease is not None
        journal = await terminal_store.append_step(
            terminal_claim.lease,
            DurableJournalCursor(
                terminal_reservation.request_id, SequenceNumber(0)
            ),
            terminal_plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(11),
        )
        journal = await terminal_store.append_step(
            terminal_claim.lease,
            journal.cursor,
            terminal_plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(11),
        )
        result = _result(
            terminal_reservation.request_id,
            terminal_plan,
            MutationState.COMMITTED,
        )
        terminal = await terminal_store.settle(
            terminal_claim.lease,
            journal.cursor,
            result,
            _correlation("d"),
            ExpiryTick(12),
        )
        assert (
            await terminal_store.settle(
                terminal_claim.lease,
                journal.cursor,
                result,
                _correlation("d"),
                ExpiryTick(13),
            )
            == terminal
        )
        with pytest.raises(DurableStoreError) as raised:
            await terminal_store.settle(
                terminal_claim.lease,
                journal.cursor,
                _result(
                    terminal_reservation.request_id,
                    terminal_plan,
                    MutationState.INDETERMINATE,
                ),
                _correlation("d"),
                ExpiryTick(13),
            )
        assert raised.value.code is DurableStoreErrorCode.TERMINAL_CONFLICT
        assert (
            await terminal_store.persist_plan(
                terminal_reservation, terminal_plan
            )
        ).terminal == terminal
        with pytest.raises(DurableStoreError) as raised:
            await terminal_store.replace_expired_owner(
                terminal_reservation,
                terminal_claim.lease,
                _owner("e"),
                ExpiryTick(14),
                DurationTicks(10),
            )
        assert raised.value.code is DurableStoreErrorCode.FENCED
        with pytest.raises(DurableStoreError) as raised:
            await terminal_store.persist_plan(
                terminal_reservation, _plan(terminal_digest, "e")
            )
        assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH
        assert await terminal_store.claim_commit(
            terminal_reservation,
            terminal_plan,
            _approval(terminal_identity, terminal_digest, terminal_plan, "d"),
            _owner("e"),
            ExpiryTick(14),
            DurationTicks(10),
            (),
        ) == DurableCommitClaim(
            DurableCommitClaimState.TERMINAL, None, terminal
        )

    run(scenario())


def test_durable_store_private_recovery_guards_fail_closed() -> None:
    """Reject malformed private recovery state before it can be persisted."""
    with pytest.raises(DurableStoreError) as raised:
        getattr(durable, "InMemoryDurablePatchBackend")(
            approval_verifier=object()
        )
    assert raised.value.code is DurableStoreErrorCode.RETENTION_LIMIT
    with pytest.raises(DurableStoreError) as raised:
        getattr(durable, "InMemoryDurablePatchStore")(object())
    assert raised.value.code is DurableStoreErrorCode.INVALID_RESERVATION
    identity = _identity("e")
    digest = _digest("e")
    plan = _plan(digest, "e", step_count=1)
    reservation = DurableReservation(
        PatchRequestId("request_" + "e" * 16), identity, digest, False
    )
    backend = _backend(DurableStoreLimits(max_journal_entries=1))
    store = InMemoryDurablePatchStore(backend)
    record = durable._DurableRecord(reservation, plan=plan)
    backend.records[identity] = record
    backend.by_request[reservation.request_id] = record
    lease = DurableCommitLease(
        reservation.request_id,
        plan.domain_id,
        _owner("e"),
        SequenceNumber(1),
        ExpiryTick(10),
    )
    record.lease = lease
    record.lifecycle = LifecyclePhase.COMMIT_STARTED
    backend.fences[plan.domain_id] = 1
    with pytest.raises(DurableStoreError) as raised:
        store._record_for_reservation(
            replace(
                reservation, request_id=PatchRequestId("request_" + "f" * 16)
            )
        )
    assert raised.value.code is DurableStoreErrorCode.INVALID_RESERVATION
    with pytest.raises(DurableStoreError) as raised:
        store._record_for_lease(
            replace(lease, request_id=PatchRequestId("request_" + "f" * 16))
        )
    assert raised.value.code is DurableStoreErrorCode.FENCED
    with pytest.raises(DurableStoreError) as raised:
        store._current_record(lease, ExpiryTick(10))
    assert raised.value.code is DurableStoreErrorCode.LEASE_EXPIRED
    with pytest.raises(DurableStoreError) as raised:
        store._validate_approval(
            reservation,
            plan,
            replace(
                _approval(identity, digest, plan, "e"),
                domain_id=PatchDomainId("domain_" + "f" * 16),
            ),
            ExpiryTick(1),
        )
    assert raised.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH
    with pytest.raises(DurableStoreError) as raised:
        store._validate_approval(
            reservation,
            plan,
            _approval(identity, digest, plan, "e", expires_at=1),
            ExpiryTick(1),
        )
    assert raised.value.code is DurableStoreErrorCode.APPROVAL_EXPIRED
    record.step_history.append(
        DurableStepJournalEntry(
            DurableJournalCursor(reservation.request_id, SequenceNumber(1)),
            plan.steps[0].step_id,
            plan.steps[0].lineage_id,
            CommitStepState.PLANNED,
        )
    )
    with pytest.raises(DurableStoreError) as raised:
        store._next_cursor(record)
    assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        durable._require_exact(object(), DurableRequestIdentity)
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        durable._require_artifact_ids(
            (_artifact("e"), _artifact("e")), backend.limits
        )
    assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        durable._lease_expiry(ExpiryTick(2**63 - 1), DurationTicks(1))
    assert raised.value.code is DurableStoreErrorCode.LEASE_EXPIRED
    with pytest.raises(DurableStoreError) as raised:
        durable._require_plan(durable._DurableRecord(reservation))
    assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH
    incomplete = durable._DurableRecord(reservation, plan=plan)
    incomplete.steps[plan.steps[0].step_id] = CommitStepState.NOT_COMMITTED
    assert (
        durable._journal_mutation_state(incomplete, plan)
        is MutationState.NOT_COMMITTED
    )
    with pytest.raises(DurableStoreError) as raised:
        getattr(durable, "derive_artifact_state")(object())
    assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
    artifact = _artifact("e")
    first = DurableArtifactJournalEntry(
        DurableJournalCursor(reservation.request_id, SequenceNumber(1)),
        artifact,
        DurableArtifactState.PRESENT,
    )
    with pytest.raises(DurableStoreError) as raised:
        durable.derive_artifact_state((first,))
    assert raised.value.code is DurableStoreErrorCode.JOURNAL_INCOMPLETE
    intended = DurableArtifactJournalEntry(
        DurableJournalCursor(reservation.request_id, SequenceNumber(1)),
        artifact,
        DurableArtifactState.INTENDED,
    )
    repeated = DurableArtifactJournalEntry(
        DurableJournalCursor(reservation.request_id, SequenceNumber(2)),
        artifact,
        DurableArtifactState.INTENDED,
    )
    with pytest.raises(DurableStoreError) as raised:
        durable.derive_artifact_state((intended, repeated))
    assert raised.value.code is DurableStoreErrorCode.JOURNAL_INCOMPLETE
    with pytest.raises(DurableStoreError) as raised:
        durable.derive_artifact_state((intended,))
    assert raised.value.code is DurableStoreErrorCode.JOURNAL_INCOMPLETE
    other = _artifact("f")
    entries = (
        intended,
        DurableArtifactJournalEntry(
            DurableJournalCursor(reservation.request_id, SequenceNumber(2)),
            artifact,
            DurableArtifactState.NOT_CREATED,
        ),
        DurableArtifactJournalEntry(
            DurableJournalCursor(reservation.request_id, SequenceNumber(3)),
            other,
            DurableArtifactState.INTENDED,
        ),
        DurableArtifactJournalEntry(
            DurableJournalCursor(reservation.request_id, SequenceNumber(4)),
            other,
            DurableArtifactState.PRESENT,
        ),
        DurableArtifactJournalEntry(
            DurableJournalCursor(reservation.request_id, SequenceNumber(5)),
            other,
            DurableArtifactState.REMOVED,
        ),
    )
    assert durable.derive_artifact_state(entries) is ArtifactState.CLEANED


def test_durable_store_rechecks_retention_and_pending_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid pending, terminal, and retention transitions."""

    class AcceptValidator:
        """Accept test ciphertext after the store has bound its request."""

        async def validate(
            self,
            request_id: PatchRequestId,
            record: DurableRetentionRecord,
        ) -> None:
            """Consume exact retention arguments without a side effect."""
            assert type(request_id) is PatchRequestId
            assert type(record) is DurableRetentionRecord

    class Audiences:
        """Return configurable retention audiences for one test request."""

        def __init__(self, value: frozenset[Audience]) -> None:
            """Store the exact configured authorization response."""
            self.value = value

        async def audiences_for(
            self,
            identity: DurableRequestIdentity,
            kind: DurableRetentionKind,
        ) -> frozenset[Audience]:
            """Return the configured response after receiving typed facts."""
            assert type(identity) is DurableRequestIdentity
            assert type(kind) is DurableRetentionKind
            return self.value

    async def scenario() -> None:
        authorizer = Audiences(frozenset((Audience.APPROVER,)))
        backend = InMemoryDurablePatchBackend(
            DurableStoreLimits(max_retention_bytes=ByteSize(11)),
            approval_verifier=_APPROVAL_AUTHORITY,
            retention_authorizer=authorizer,
            retention_validator=AcceptValidator(),
        )
        store = InMemoryDurablePatchStore(backend)
        identity = _identity("d")
        digest = _digest("d")
        reservation = await store.reserve(identity, digest)
        plan = _plan(digest, "d", step_count=1)
        backend.records[identity].lifecycle = LifecyclePhase.PLANNED
        with pytest.raises(DurableStoreError) as raised:
            await store.persist_plan(reservation, plan)
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        backend.records[identity].lifecycle = LifecyclePhase.RECEIVED
        await store.persist_plan(reservation, plan)
        with pytest.raises(DurableStoreError) as raised:
            await store.request_cancellation(
                DurableRequestAccess(reservation.request_id, identity)
            )
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        claim = await store.claim_commit(
            reservation,
            plan,
            _approval(identity, digest, plan, "d"),
            _owner("d"),
            ExpiryTick(10),
            DurationTicks(20),
            (),
        )
        assert claim.lease is not None
        backend.records[identity].lifecycle = LifecyclePhase.SETTLEMENT_PENDING
        with pytest.raises(DurableStoreError) as raised:
            await store.suspend(
                claim.lease,
                _pending("d"),
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        backend.records[identity].lifecycle = LifecyclePhase.COMMIT_STARTED
        retained = DurableRetentionRecord(
            PatchRetentionRecordId("retained_" + "d" * 16),
            DurableRetentionKind.PRIVATE_STAGING,
            PatchRetentionKeyId("retention_" + "d" * 16),
            EncryptedRetentionValue(b"opaque"),
            DurableRetentionPolicy(ExpiryTick(12), False),
        )
        await store.put_retention(reservation, retained)
        await store.put_retention(reservation, retained)
        with pytest.raises(DurableStoreError) as raised:
            await store.put_retention(
                reservation,
                replace(
                    retained,
                    policy=DurableRetentionPolicy(ExpiryTick(13), False),
                ),
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_CONFLICT
        with pytest.raises(DurableStoreError) as raised:
            await store.put_retention(
                reservation,
                replace(
                    retained,
                    retention_id=PatchRetentionRecordId(
                        "retained_" + "e" * 16
                    ),
                ),
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_LIMIT
        access = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, identity)
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.get_retention(
                access, retained.retention_id, ExpiryTick(12)
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        assert retained.retention_id not in backend.records[identity].retention
        await store.put_retention(
            reservation,
            replace(
                retained, policy=DurableRetentionPolicy(ExpiryTick(20), False)
            ),
        )
        setattr(authorizer, "value", object())
        with pytest.raises(DurableStoreError) as raised:
            await store.get_retention(
                access, retained.retention_id, ExpiryTick(13)
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        authorizer.value = frozenset()
        with pytest.raises(DurableStoreError) as raised:
            await store.get_retention(
                access, retained.retention_id, ExpiryTick(13)
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        authorizer.value = frozenset((Audience.APPROVER,))
        pending = _pending("d")
        await store.suspend(claim.lease, pending, ExpiryTick(14))
        wrong_access = DurablePendingAccess(
            access.request,
            PatchPendingOperationId("pending_" + "e" * 16),
            pending.correlation_id,
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.await_terminal(wrong_access)
        assert raised.value.code is DurableStoreErrorCode.ACCESS_DENIED
        journal = await store.append_step(
            claim.lease,
            DurableJournalCursor(reservation.request_id, SequenceNumber(0)),
            plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(14),
        )
        journal = await store.append_step(
            claim.lease,
            journal.cursor,
            plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(14),
        )
        result = _result(reservation.request_id, plan, MutationState.COMMITTED)
        with pytest.raises(DurableStoreError) as raised:
            await store.settle(
                claim.lease,
                journal.cursor,
                replace(
                    result,
                    plan_id=PatchPlanId("plan_" + "e" * 16),
                ),
                pending.correlation_id,
                ExpiryTick(15),
            )
        assert raised.value.code is DurableStoreErrorCode.TERMINAL_CONFLICT
        with pytest.raises(DurableStoreError) as raised:
            await store.settle(
                claim.lease,
                journal.cursor,
                result,
                _correlation("e"),
                ExpiryTick(15),
            )
        assert raised.value.code is DurableStoreErrorCode.ACCESS_DENIED
        terminal = await store.settle(
            claim.lease,
            journal.cursor,
            result,
            pending.correlation_id,
            ExpiryTick(15),
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.await_terminal(wrong_access)
        assert raised.value.code is DurableStoreErrorCode.ACCESS_DENIED
        with pytest.raises(DurableStoreError) as raised:
            await store.put_retention(reservation, retained)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED

        duplicate_backend = _backend()
        duplicate_store = InMemoryDurablePatchStore(duplicate_backend)
        duplicate_identity = _identity("e")
        duplicate_digest = _digest("e")
        duplicate_reservation = await duplicate_store.reserve(
            duplicate_identity, duplicate_digest
        )
        duplicate_plan = _plan(duplicate_digest, "e", step_count=1)
        await duplicate_store.persist_plan(
            duplicate_reservation, duplicate_plan
        )
        duplicate_claim = await duplicate_store.claim_commit(
            duplicate_reservation,
            duplicate_plan,
            _approval(
                duplicate_identity, duplicate_digest, duplicate_plan, "e"
            ),
            _owner("e"),
            ExpiryTick(10),
            DurationTicks(20),
            (),
        )
        assert duplicate_claim.lease is not None
        duplicate_journal = await duplicate_store.append_step(
            duplicate_claim.lease,
            DurableJournalCursor(
                duplicate_reservation.request_id, SequenceNumber(0)
            ),
            duplicate_plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(11),
        )
        duplicate_journal = await duplicate_store.append_step(
            duplicate_claim.lease,
            duplicate_journal.cursor,
            duplicate_plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(11),
        )
        event_id = PatchEventId("event_" + "f" * 16)
        duplicate_backend.event_ids.add(event_id)
        monkeypatch.setattr(
            PatchEventId, "new", staticmethod(lambda: event_id)
        )
        with pytest.raises(DurableStoreError) as raised:
            await duplicate_store.settle(
                duplicate_claim.lease,
                duplicate_journal.cursor,
                _result(
                    duplicate_reservation.request_id,
                    duplicate_plan,
                    MutationState.COMMITTED,
                ),
                _correlation("e"),
                ExpiryTick(12),
            )
        assert raised.value.code is DurableStoreErrorCode.TERMINAL_CONFLICT
        backend.event_ids.add(event_id)
        with pytest.raises(DurableStoreError) as raised:
            duplicate_store._append_outbox(
                backend.records[identity],
                LifecyclePhase.SETTLEMENT_PENDING,
                pending.correlation_id,
            )
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        assert terminal.result is result

        class TerminalFlipRecord:
            """Model a terminal transition during a retention recheck."""

            def __init__(self, value: DurableReservation) -> None:
                """Initialize a reserved record with no retained values."""
                self.reservation = value
                self.retention: dict[
                    PatchRetentionRecordId, DurableRetentionRecord
                ] = {}
                self._reads = 0

            @property
            def terminal(self) -> object | None:
                """Return terminal truth only on the second locked recheck."""
                self._reads += 1
                return None if self._reads == 1 else object()

        flip_backend = InMemoryDurablePatchBackend(
            approval_verifier=_APPROVAL_AUTHORITY,
            retention_authorizer=authorizer,
            retention_validator=AcceptValidator(),
        )
        flip_store = InMemoryDurablePatchStore(flip_backend)
        flip_identity = _identity("f")
        flip_reservation = DurableReservation(
            PatchRequestId("request_" + "f" * 16),
            flip_identity,
            _digest("f"),
            False,
        )
        getattr(flip_backend.records, "__setitem__")(
            flip_identity, TerminalFlipRecord(flip_reservation)
        )
        with pytest.raises(DurableStoreError) as raised:
            await flip_store.put_retention(
                flip_reservation,
                replace(
                    retained,
                    retention_id=PatchRetentionRecordId(
                        "retained_" + "f" * 16
                    ),
                ),
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED

    run(scenario())
