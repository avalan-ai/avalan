"""Reconcile durable patch settlement only for authenticated test hosts.

This module deliberately has no tool, server, CLI, agent, or protocol
registration.  It adapts the Phase 6 worker journal algebra to the strict
durable store contract while keeping a pending invocation suspended until its
original branch receives the one recorded terminal result.
"""

from dataclasses import dataclass

from avalan.patch.coordinator import (
    SettlementJournal,
    WorkerReport,
    WorkerState,
)
from avalan.patch.domain import (
    CommitStepState,
    DomainFacade,
    DurationTicks,
    ExpiryTick,
    PatchArtifactId,
    PatchCommitOwnerId,
    PatchObserverCorrelationId,
    PatchPending,
    PatchResult,
)
from avalan.patch.durable_store import (
    DurableArtifactState,
    DurableCommitLease,
    DurableJournal,
    DurablePatchStore,
    DurablePendingAccess,
    DurablePendingRecord,
    DurablePendingRequest,
    DurableRequestAccess,
    DurableRequestSnapshot,
    DurableStoreError,
    DurableStoreErrorCode,
)


@dataclass(frozen=True, slots=True)
class DurablePatchTestHostProfile:
    """Select the isolated authenticated host profile for durable tests."""

    enabled: bool = False
    authenticated: bool = False

    def __post_init__(self) -> None:
        """Require explicit boolean test-host activation witnesses."""
        if (
            type(self.enabled) is not bool
            or type(self.authenticated) is not bool
        ):
            raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)


@dataclass(frozen=True, slots=True)
class DurableArtifactObservation:
    """Carry one reconciled target-owned artifact state transition."""

    worker_identifier: str
    artifact_id: PatchArtifactId
    state: DurableArtifactState

    def __post_init__(self) -> None:
        """Require exact typed durable artifact reconciliation evidence."""
        if (
            not self.worker_identifier
            or type(self.artifact_id) is not PatchArtifactId
            or type(self.state) is not DurableArtifactState
        ):
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)


class DurablePatchTestHost:
    """Project durable pending state only to an authenticated test host."""

    def __init__(
        self,
        store: DurablePatchStore,
        profile: DurablePatchTestHostProfile,
    ) -> None:
        """Bind one disabled-by-default test-host continuation projection."""
        if (
            type(profile) is not DurablePatchTestHostProfile
            or not profile.enabled
            or not profile.authenticated
        ):
            raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)
        self._store = store
        self._domain = DomainFacade()

    async def inspect(
        self, access: DurablePendingAccess
    ) -> PatchPending | PatchResult:
        """Return pending or terminal state on the original branch."""
        value = await self._store.inspect_pending(access)
        if isinstance(value, DurablePendingRecord):
            return self._pending(value)
        return value.result

    async def await_resume(self, access: DurablePendingAccess) -> PatchResult:
        """Await and return the terminal result on the original branch only."""
        return (await self._store.await_terminal(access)).result

    async def resume(self, access: DurablePendingAccess) -> PatchResult:
        """Resume the suspended original branch with its terminal result."""
        return await self.await_resume(access)

    def _pending(self, value: DurablePendingRecord) -> PatchPending:
        """Project durable pending facts into the closed domain envelope."""
        return self._domain.pending(
            value.request_id,
            value.pending_operation_id,
            value.correlation_id,
        )


class DurablePatchReconciler:
    """Journal worker recovery evidence without reissuing target effects."""

    def __init__(self, store: DurablePatchStore) -> None:
        """Bind the strict store used for recovery compare-and-set calls."""
        self._store = store
        self._domain = DomainFacade()

    async def replace_expired_owner(
        self,
        access: DurableRequestAccess,
        owner_id: PatchCommitOwnerId,
        now: ExpiryTick,
        lease_duration: DurationTicks,
    ) -> DurableCommitLease:
        """Fence an expired worker before a fresh reconciler settles it."""
        snapshot = await self._store.inspect(access)
        if snapshot.terminal is not None or snapshot.lease is None:
            raise DurableStoreError(DurableStoreErrorCode.FENCED)
        return await self._store.replace_expired_owner(
            snapshot.reservation,
            snapshot.lease,
            owner_id,
            now,
            lease_duration,
        )

    async def reconcile(
        self,
        access: DurableRequestAccess,
        lease: DurableCommitLease,
        report: WorkerReport,
        result: PatchResult,
        correlation_id: PatchObserverCorrelationId,
        now: ExpiryTick,
        *,
        pending: DurablePendingRequest | None = None,
        artifacts: tuple[DurableArtifactObservation, ...] = (),
    ) -> PatchPending | PatchResult:
        """Reconcile one owned worker report without retrying target action."""
        if (
            type(access) is not DurableRequestAccess
            or type(lease) is not DurableCommitLease
            or type(report) is not WorkerReport
            or type(result) is not PatchResult
            or type(correlation_id) is not PatchObserverCorrelationId
            or type(now) is not ExpiryTick
            or type(artifacts) is not tuple
            or any(
                type(item) is not DurableArtifactObservation
                for item in artifacts
            )
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        snapshot = await self._store.inspect(access)
        if snapshot.terminal is not None:
            return snapshot.terminal.result
        if snapshot.lease != lease or lease.request_id != access.request_id:
            raise DurableStoreError(DurableStoreErrorCode.FENCED)
        if report.state is WorkerState.LIVE:
            return await self._suspend(snapshot, lease, pending, now)
        if report.state not in {WorkerState.SETTLED, WorkerState.FENCED}:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        if report.journal is None:
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_INCOMPLETE)
        journal = await self._append_journal(
            snapshot,
            lease,
            report.journal,
            artifacts,
            now,
        )
        try:
            terminal = await self._store.settle(
                lease,
                journal.cursor,
                result,
                correlation_id,
                now,
            )
        except DurableStoreError as error:
            if (
                error.code is DurableStoreErrorCode.JOURNAL_INCOMPLETE
                and pending is not None
            ):
                refreshed = await self._store.inspect(access)
                return await self._suspend(refreshed, lease, pending, now)
            raise
        await self._store.release_terminal_coordination(access)
        return terminal.result

    async def _suspend(
        self,
        snapshot: DurableRequestSnapshot,
        lease: DurableCommitLease,
        pending: DurablePendingRequest | None,
        now: ExpiryTick,
    ) -> PatchPending:
        """Persist or project only the original pending branch state."""
        current = snapshot.pending
        if current is None:
            if type(pending) is not DurablePendingRequest:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            current = await self._store.suspend(lease, pending, now)
        return self._domain.pending(
            current.request_id,
            current.pending_operation_id,
            current.correlation_id,
        )

    async def _append_journal(
        self,
        snapshot: DurableRequestSnapshot,
        lease: DurableCommitLease,
        journal: SettlementJournal,
        artifacts: tuple[DurableArtifactObservation, ...],
        now: ExpiryTick,
    ) -> DurableJournal:
        """Write only monotonic recovery facts through durable journal CAS."""
        plan = snapshot.plan
        if plan is None:
            raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)
        expected_steps = tuple(
            (item.step_id, item.lineage_id) for item in plan.steps
        )
        observed_steps = tuple(
            (item.identifier, item.lineage) for item in journal.steps
        )
        if observed_steps != expected_steps or any(
            item.state is CommitStepState.PLANNED for item in journal.steps
        ):
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
        if tuple(item.identifier for item in journal.artifacts) != tuple(
            item.worker_identifier for item in artifacts
        ):
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
        current = snapshot.journal
        states = {item.step_id: item.state for item in current.steps}
        for step in journal.steps:
            step_state = states.get(step.identifier)
            if step_state is None:
                current = await self._store.append_step(
                    lease,
                    current.cursor,
                    step.identifier,
                    CommitStepState.PLANNED,
                    now,
                )
                step_state = CommitStepState.PLANNED
            if step_state is CommitStepState.PLANNED:
                current = await self._store.append_step(
                    lease,
                    current.cursor,
                    step.identifier,
                    step.state,
                    now,
                )
            elif step_state is not step.state:
                raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
        artifact_states = {
            item.artifact_id: item.state for item in current.artifacts
        }
        for artifact in artifacts:
            artifact_state = artifact_states.get(artifact.artifact_id)
            if artifact_state is None or artifact_state is artifact.state:
                if artifact_state is None:
                    raise DurableStoreError(
                        DurableStoreErrorCode.JOURNAL_CONFLICT
                    )
                continue
            if (
                artifact_state is DurableArtifactState.INTENDED
                and artifact.state
                in {
                    DurableArtifactState.REMOVED,
                    DurableArtifactState.LEAKED,
                }
            ):
                current = await self._store.append_artifact(
                    lease,
                    current.cursor,
                    artifact.artifact_id,
                    DurableArtifactState.PRESENT,
                    now,
                )
            current = await self._store.append_artifact(
                lease,
                current.cursor,
                artifact.artifact_id,
                artifact.state,
                now,
            )
        return current
