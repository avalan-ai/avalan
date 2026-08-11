"""Exercise Phase 8 durable reconciliation and test-host continuation E2Es."""

from asyncio import create_task, run, sleep
from dataclasses import replace

import pytest
from phase_8_store_test import _approval as _stored_approval
from phase_8_store_test import _artifact, _backend

from avalan.patch.coordinator import (
    ArtifactJournal,
    JournalStep,
    RetransmissionKey,
    SettlementJournal,
    WorkerReport,
    WorkerState,
)
from avalan.patch.domain import (
    AlgorithmDigest,
    ArtifactState,
    CommitStepState,
    CommitTruth,
    DurationTicks,
    ErrorStage,
    ExpiryTick,
    LifecyclePhase,
    LineageState,
    MutationState,
    PatchCommitOwnerId,
    PatchContextId,
    PatchDiagnostic,
    PatchDomainId,
    PatchErrorCode,
    PatchExecutionId,
    PatchLineageId,
    PatchObserverCorrelationId,
    PatchPending,
    PatchPendingOperationId,
    PatchPlanId,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PatchStepId,
    PatchWorkspaceId,
    PostconditionState,
    RequestedEffectOccurrence,
    Retryability,
    SequenceNumber,
    WorkspaceChange,
)
from avalan.patch.durable_coordinator import (
    DurableArtifactObservation,
    DurablePatchReconciler,
    DurablePatchTestHost,
    DurablePatchTestHostProfile,
)
from avalan.patch.durable_store import (
    DurableApproval,
    DurableArtifactState,
    DurableCommitClaimState,
    DurableCommitLease,
    DurableJournalCursor,
    DurablePendingAccess,
    DurablePendingRequest,
    DurablePlanReference,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableReservation,
    DurableStepBinding,
    DurableStoreError,
    DurableStoreErrorCode,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.policy import (
    PatchPrincipalId,
    PatchTenantId,
    PolicyRouteId,
)


def _digest(token: str) -> AlgorithmDigest:
    """Return deterministic opaque digest evidence for a test request."""
    return AlgorithmDigest("sha256", token * 64)


def _identity(token: str) -> DurableRequestIdentity:
    """Return one authenticated durable retransmission identity."""
    return DurableRequestIdentity(
        PatchTenantId("tenant-" + token),
        PatchPrincipalId("principal-" + token),
        PatchExecutionId("execution_" + token * 16),
        PolicyRouteId("route-" + token),
        RetransmissionKey("retransmission-" + token),
    )


def _plan(
    digest: AlgorithmDigest, token: str, step_count: int
) -> DurablePlanReference:
    """Return a sealed durable plan reference with exact ordered steps."""
    return DurablePlanReference(
        PatchPlanId("plan_" + token * 16),
        digest,
        _digest("f"),
        _digest("e"),
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
    token: str,
) -> DurableApproval:
    """Return one unconsumed exact approval record."""
    return _stored_approval(identity, digest, plan, token)


def _owner(token: str) -> PatchCommitOwnerId:
    """Return one deterministic reconciliation-owner identity."""
    return PatchCommitOwnerId("owner_" + token * 16)


def _correlation(token: str) -> PatchObserverCorrelationId:
    """Return one original suspended-branch correlation identity."""
    return PatchObserverCorrelationId("correlation_" + token * 16)


def _result(
    request_id: PatchRequestId,
    plan: DurablePlanReference,
    mutation: MutationState,
) -> PatchResult:
    """Return terminal truth matching the supplied durable step journal."""
    match mutation:
        case MutationState.COMMITTED:
            truth = CommitTruth(
                mutation,
                LineageState.COMMITTED,
                RequestedEffectOccurrence.TRUE,
                ArtifactState.ABSENT,
                WorkspaceChange.CHANGED,
                True,
                PostconditionState.ESTABLISHED,
            )
            status = PatchStatus.COMMITTED
            diagnostic = None
        case MutationState.PARTIALLY_COMMITTED:
            truth = CommitTruth(
                mutation,
                LineageState.PARTIALLY_COMMITTED,
                RequestedEffectOccurrence.TRUE,
                ArtifactState.ABSENT,
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
                mutation,
                LineageState.INDETERMINATE,
                RequestedEffectOccurrence.UNKNOWN,
                ArtifactState.ABSENT,
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
        case _:
            raise AssertionError("test result requires a commit-stage truth")
    return PatchResult(
        1,
        request_id,
        plan.plan_id,
        LifecyclePhase.REQUEST_COMPLETED,
        status,
        truth,
        diagnostic,
    )


def _report(
    plan: DurablePlanReference, states: tuple[CommitStepState, ...]
) -> WorkerReport:
    """Return Phase 6-shaped final worker evidence for the durable adapter."""
    return WorkerReport(
        WorkerState.SETTLED,
        SettlementJournal(
            tuple(
                JournalStep(binding.step_id, binding.lineage_id, state)
                for binding, state in zip(plan.steps, states, strict=True)
            ),
            (),
            (
                PostconditionState.UNKNOWN
                if CommitStepState.UNKNOWN in states
                else PostconditionState.ESTABLISHED
            ),
        ),
    )


async def _claimed(token: str, step_count: int = 1) -> tuple[
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
    DurableRequestIdentity,
    DurableReservation,
    DurablePlanReference,
    DurableCommitLease,
]:
    """Persist the pre-effect commit fence for a controlled crash boundary."""
    backend = _backend()
    store = InMemoryDurablePatchStore(backend)
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
        DurationTicks(30),
        (),
    )
    assert claim.state is DurableCommitClaimState.OWNER
    assert claim.lease is not None
    return backend, store, identity, reservation, plan, claim.lease


def test_patch_e2e_006_crash_after_commit_started_reconciles_once() -> None:
    """Reconcile a fresh durable client without reapplying the one effect."""

    async def scenario() -> None:
        backend, _, identity, reservation, plan, lease = await _claimed("a")
        effects = 1
        fresh_store = InMemoryDurablePatchStore(backend)
        reconciler = DurablePatchReconciler(fresh_store)
        result = _result(reservation.request_id, plan, MutationState.COMMITTED)

        completed = await reconciler.reconcile(
            DurableRequestAccess(reservation.request_id, identity),
            lease,
            _report(plan, (CommitStepState.COMMITTED,)),
            result,
            _correlation("a"),
            ExpiryTick(20),
        )

        assert completed is result
        assert effects == 1
        assert (
            await fresh_store.inspect(
                DurableRequestAccess(reservation.request_id, identity)
            )
        ).terminal is not None

    run(scenario())


def test_patch_e2e_007_retransmission_replays_terminal_without_effect() -> (
    None
):
    """Attach a same-key retransmission to the recorded terminal result."""

    async def scenario() -> None:
        backend, store, identity, reservation, plan, lease = await _claimed(
            "b"
        )
        effects = 1
        result = _result(reservation.request_id, plan, MutationState.COMMITTED)
        reconciler = DurablePatchReconciler(store)
        await reconciler.reconcile(
            DurableRequestAccess(reservation.request_id, identity),
            lease,
            _report(plan, (CommitStepState.COMMITTED,)),
            result,
            _correlation("b"),
            ExpiryTick(20),
        )

        fresh_store = InMemoryDurablePatchStore(backend)
        replay = await fresh_store.reserve(identity, _digest("b"))
        replayed = await DurablePatchReconciler(fresh_store).reconcile(
            DurableRequestAccess(replay.request_id, identity),
            lease,
            _report(plan, (CommitStepState.COMMITTED,)),
            result,
            _correlation("b"),
            ExpiryTick(21),
        )

        assert replay.replayed
        assert replayed is result
        assert effects == 1

    run(scenario())


def test_patch_e2e_008_pending_restart_authenticates_original_branch() -> None:
    """Suspend, restart, inspect, await, and resume one original branch."""

    async def scenario() -> None:
        backend, store, identity, reservation, plan, lease = await _claimed(
            "c"
        )
        reconciler = DurablePatchReconciler(store)
        correlation = _correlation("c")
        pending = DurablePendingRequest(
            PatchPendingOperationId("pending_" + "c" * 16),
            correlation,
            DurationTicks(5),
        )
        projected = await reconciler.reconcile(
            DurableRequestAccess(reservation.request_id, identity),
            lease,
            WorkerReport(WorkerState.LIVE, None),
            _result(reservation.request_id, plan, MutationState.COMMITTED),
            correlation,
            ExpiryTick(20),
            pending=pending,
        )
        assert isinstance(projected, PatchPending)

        fresh_store = InMemoryDurablePatchStore(backend)
        host = DurablePatchTestHost(
            fresh_store, DurablePatchTestHostProfile(True, True)
        )
        access = DurablePendingAccess(
            DurableRequestAccess(reservation.request_id, identity),
            pending.pending_operation_id,
            correlation,
        )
        assert await host.inspect(access) == projected
        awaiting = create_task(host.await_resume(access))
        await sleep(0)
        result = _result(reservation.request_id, plan, MutationState.COMMITTED)
        completed = await DurablePatchReconciler(fresh_store).reconcile(
            DurableRequestAccess(reservation.request_id, identity),
            lease,
            _report(plan, (CommitStepState.COMMITTED,)),
            result,
            correlation,
            ExpiryTick(21),
        )
        assert completed is result
        assert await awaiting is result
        with pytest.raises(DurableStoreError) as raised:
            await host.inspect(
                DurablePendingAccess(
                    DurableRequestAccess(
                        reservation.request_id, _identity("d")
                    ),
                    pending.pending_operation_id,
                    correlation,
                )
            )
        assert raised.value.code is DurableStoreErrorCode.ACCESS_DENIED
        with pytest.raises(DurableStoreError):
            DurablePatchTestHost(
                fresh_store, DurablePatchTestHostProfile(False, True)
            )

    assert run(scenario()) is None


@pytest.mark.parametrize(
    ("token", "states", "mutation"),
    (
        (
            "d",
            (CommitStepState.COMMITTED, CommitStepState.NOT_COMMITTED),
            MutationState.PARTIALLY_COMMITTED,
        ),
        (
            "e",
            (CommitStepState.UNKNOWN,),
            MutationState.INDETERMINATE,
        ),
    ),
)
def test_patch_e2e_009_non_success_truth_survives_restart_without_retry(
    token: str,
    states: tuple[CommitStepState, ...],
    mutation: MutationState,
) -> None:
    """Persist partial or unknown truth without an automatic target retry."""

    async def scenario() -> None:
        backend, _, identity, reservation, plan, lease = await _claimed(
            token, len(states)
        )
        effects = 1
        fresh_store = InMemoryDurablePatchStore(backend)
        result = _result(reservation.request_id, plan, mutation)
        completed = await DurablePatchReconciler(fresh_store).reconcile(
            DurableRequestAccess(reservation.request_id, identity),
            lease,
            _report(plan, states),
            result,
            _correlation(token),
            ExpiryTick(20),
        )
        replayed = await DurablePatchReconciler(
            InMemoryDurablePatchStore(backend)
        ).reconcile(
            DurableRequestAccess(reservation.request_id, identity),
            lease,
            _report(plan, states),
            result,
            _correlation(token),
            ExpiryTick(21),
        )

        assert completed is result
        assert replayed is result
        assert effects == 1

    run(scenario())


def test_durable_reconciler_rejects_malformed_recovery_evidence() -> None:
    """Fail closed for invalid host activation and recovery journal facts."""
    with pytest.raises(DurableStoreError) as raised:
        getattr(DurablePatchTestHostProfile, "__call__")(1, True)
    assert raised.value.code is DurableStoreErrorCode.ACCESS_DENIED
    with pytest.raises(DurableStoreError) as raised:
        getattr(DurableArtifactObservation, "__call__")("", object(), object())
    assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT

    async def scenario() -> None:
        backend = _backend()
        store = InMemoryDurablePatchStore(backend)
        identity = _identity("f")
        digest = _digest("f")
        reservation = await store.reserve(identity, digest)
        access = DurableRequestAccess(reservation.request_id, identity)
        reconciler = DurablePatchReconciler(store)
        with pytest.raises(DurableStoreError) as raised:
            await reconciler.replace_expired_owner(
                access,
                _owner("f"),
                ExpiryTick(10),
                DurationTicks(10),
            )
        assert raised.value.code is DurableStoreErrorCode.FENCED

        plan = _plan(digest, "f", 1)
        await store.persist_plan(reservation, plan)
        claim = await store.claim_commit(
            reservation,
            plan,
            _approval(identity, digest, plan, "f"),
            _owner("f"),
            ExpiryTick(10),
            DurationTicks(20),
            (_artifact("f"),),
        )
        assert claim.lease is not None
        result = _result(reservation.request_id, plan, MutationState.COMMITTED)
        settled = _report(plan, (CommitStepState.COMMITTED,))
        assert settled.journal is not None
        with pytest.raises(DurableStoreError) as raised:
            await getattr(reconciler, "reconcile")(
                access,
                claim.lease,
                settled,
                result,
                _correlation("f"),
                ExpiryTick(11),
                artifacts=(object(),),
            )
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        with pytest.raises(DurableStoreError) as raised:
            await reconciler.reconcile(
                access,
                replace(claim.lease, fence=SequenceNumber(2)),
                settled,
                result,
                _correlation("f"),
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.FENCED
        invalid_report = object.__new__(WorkerReport)
        object.__setattr__(invalid_report, "state", object())
        object.__setattr__(invalid_report, "journal", None)
        with pytest.raises(DurableStoreError) as raised:
            await reconciler.reconcile(
                access,
                claim.lease,
                invalid_report,
                result,
                _correlation("f"),
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        missing_journal = object.__new__(WorkerReport)
        object.__setattr__(missing_journal, "state", WorkerState.SETTLED)
        object.__setattr__(missing_journal, "journal", None)
        with pytest.raises(DurableStoreError) as raised:
            await reconciler.reconcile(
                access,
                claim.lease,
                missing_journal,
                result,
                _correlation("f"),
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.JOURNAL_INCOMPLETE
        with pytest.raises(DurableStoreError) as raised:
            await reconciler.reconcile(
                access,
                claim.lease,
                WorkerReport(WorkerState.LIVE, None),
                result,
                _correlation("f"),
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT

        snapshot = await store.inspect(access)
        with pytest.raises(DurableStoreError) as raised:
            await reconciler._append_journal(
                replace(snapshot, plan=None),
                claim.lease,
                settled.journal,
                (),
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH
        with pytest.raises(DurableStoreError) as raised:
            planned_report = _report(plan, (CommitStepState.PLANNED,))
            assert planned_report.journal is not None
            await reconciler._append_journal(
                snapshot,
                claim.lease,
                planned_report.journal,
                (),
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
        mismatch = SettlementJournal(
            settled.journal.steps,
            (ArtifactJournal("worker", ArtifactState.STAGED),),
            settled.journal.postcondition,
        )
        with pytest.raises(DurableStoreError) as raised:
            await reconciler._append_journal(
                snapshot,
                claim.lease,
                mismatch,
                (),
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
        unknown_artifact = DurableArtifactObservation(
            "worker", _artifact("e"), DurableArtifactState.PRESENT
        )
        with pytest.raises(DurableStoreError) as raised:
            await reconciler._append_journal(
                snapshot,
                claim.lease,
                mismatch,
                (unknown_artifact,),
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
        intended = DurableArtifactObservation(
            "worker", _artifact("f"), DurableArtifactState.INTENDED
        )
        current_snapshot = await store.inspect(access)
        assert (
            await reconciler._append_journal(
                current_snapshot,
                claim.lease,
                mismatch,
                (intended,),
                ExpiryTick(11),
            )
            == current_snapshot.journal
        )
        changed = DurableArtifactObservation(
            "worker", _artifact("f"), DurableArtifactState.PRESENT
        )
        assert (
            await reconciler._append_journal(
                await store.inspect(access),
                claim.lease,
                mismatch,
                (changed,),
                ExpiryTick(11),
            )
        ).artifacts[-1].state is DurableArtifactState.PRESENT
        with pytest.raises(DurableStoreError) as raised:
            not_committed_report = _report(
                plan, (CommitStepState.NOT_COMMITTED,)
            )
            assert not_committed_report.journal is not None
            await reconciler._append_journal(
                await store.inspect(access),
                claim.lease,
                not_committed_report.journal,
                (),
                ExpiryTick(11),
            )
        assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT

    run(scenario())


def test_reconciler_projects_terminal_state_and_replaces_owner() -> None:
    """Expose only terminal branch state and a newly fenced replacement."""

    async def scenario() -> None:
        backend, store, identity, reservation, plan, lease = await _claimed(
            "a"
        )
        reconciler = DurablePatchReconciler(store)
        replacement = await reconciler.replace_expired_owner(
            DurableRequestAccess(reservation.request_id, identity),
            _owner("b"),
            ExpiryTick(40),
            DurationTicks(10),
        )
        assert replacement.owner_id == _owner("b")
        pending = DurablePendingRequest(
            PatchPendingOperationId("pending_" + "a" * 16),
            _correlation("a"),
            DurationTicks(5),
        )
        await store.suspend(replacement, pending, ExpiryTick(41))
        journal = await store.append_step(
            replacement,
            DurableJournalCursor(reservation.request_id, SequenceNumber(0)),
            plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(41),
        )
        journal = await store.append_step(
            replacement,
            journal.cursor,
            plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(41),
        )
        result = await reconciler.reconcile(
            DurableRequestAccess(reservation.request_id, identity),
            replacement,
            _report(plan, (CommitStepState.COMMITTED,)),
            _result(reservation.request_id, plan, MutationState.COMMITTED),
            _correlation("a"),
            ExpiryTick(42),
        )
        host = DurablePatchTestHost(
            InMemoryDurablePatchStore(backend),
            DurablePatchTestHostProfile(True, True),
        )
        access = DurablePendingAccess(
            DurableRequestAccess(reservation.request_id, identity),
            pending.pending_operation_id,
            _correlation("a"),
        )
        terminal = await store.inspect(access.request)
        assert terminal.terminal is not None
        assert terminal.terminal.pending_operation_id is not None
        terminal_access = DurablePendingAccess(
            access.request,
            terminal.terminal.pending_operation_id,
            _correlation("a"),
        )
        assert await host.inspect(terminal_access) == result
        assert await host.resume(terminal_access) == result
        assert journal.cursor.revision.value == 2

    run(scenario())
