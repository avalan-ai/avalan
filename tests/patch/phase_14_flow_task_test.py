"""Exercise Phase 14 flow and queued-task durable orchestration adapters."""

from asyncio import CancelledError, create_task, get_running_loop, run, sleep
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from runpy import run_path
from sys import path as sys_path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from patch_activation_support import activated_patch_test_profile
from phase_8_store_test import _approval, _backend

from avalan.flow.flow import Flow
from avalan.flow.node import Node
from avalan.patch.coordinator import (
    JournalStep,
    RetransmissionKey,
    SettlementJournal,
    WorkerReport,
    WorkerState,
)
from avalan.patch.domain import (
    AlgorithmDigest,
    ApprovalMode,
    ArtifactState,
    CommitStepState,
    CommitTruth,
    DurationTicks,
    ExpiryTick,
    LifecyclePhase,
    LineageState,
    MutationState,
    OperationType,
    PatchContextId,
    PatchDomainId,
    PatchExecutionId,
    PatchInvocationOutcome,
    PatchLineageId,
    PatchObserverCorrelationId,
    PatchPendingOperationId,
    PatchPlanId,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PatchStepId,
    PatchWorkspaceId,
    PostconditionState,
    RequestedEffectOccurrence,
    WorkspaceChange,
)
from avalan.patch.durable_approval import (
    HmacDurableApprovalAuthority,
    PhaseFiveDurableApprovalIssuer,
)
from avalan.patch.durable_store import (
    DurableApproval,
    DurableCommitClaim,
    DurableCommitClaimState,
    DurableCommitLease,
    DurablePendingAccess,
    DurablePendingRequest,
    DurablePlanReference,
    DurableProtocolOrigin,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableReservation,
    DurableStepBinding,
    DurableStoreError,
    DurableStoreErrorCode,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.parser import (
    PatchInputLimits,
    PatchRequestParser,
    RawProviderProfile,
    RawToolCallId,
)
from avalan.patch.policy import (
    ApprovalService,
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PlanReviewRequest,
    PolicyRouteId,
    RuntimeGrantStore,
)
from avalan.patch.protocols import (
    PatchOrchestrationChecklist,
    PatchProtocolApprovalGate,
    PatchProtocolApprovalPort,
    PatchProtocolChecklist,
    PatchProtocolDurableCoordinator,
    PatchProtocolEffectPort,
    PatchProtocolEffectReceipt,
    PatchProtocolError,
    PatchProtocolFlowAdapter,
    PatchProtocolFlowRequest,
    PatchProtocolFlowSuspension,
    PatchProtocolIdentity,
    PatchProtocolOrchestrationAdapter,
    PatchProtocolPlanPort,
    PatchProtocolProfile,
    PatchProtocolProviderAdapter,
    PatchProtocolProviderCall,
    PatchProtocolQueuedTaskAdapter,
    PatchProtocolReservation,
    PatchProtocols,
    PatchProtocolSelectedRuntime,
    PatchProtocolSurface,
    PatchProviderCodecChecklist,
)
from avalan.patch.sandbox_commit import (
    SandboxPatchRuntimeBinder,
    SandboxPatchSdkService,
    SandboxPatchServiceConfiguration,
    _durable_plan,
)
from avalan.patch.target import TargetInspectionError
from avalan.patch.toolset import (
    PatchApprovalBinding,
    PatchCoordinatorBinding,
    PatchPersistenceBinding,
    PatchSdkHost,
    PatchToolLoader,
    PatchToolSet,
)
from avalan.server.patch_protocols import (
    PatchProtocolAdapterConfiguration,
    PatchProtocolIdentityResolver,
    install_patch_protocol_test_routes,
)


def _profile(surface: PatchProtocolSurface) -> PatchProtocolProfile:
    """Return one complete exact orchestration test profile."""
    return PatchProtocolProfile(
        surface=surface,
        enabled=True,
        authenticated=True,
        loopback_only=True,
        protocol=PatchProtocolChecklist(
            canonical_input=True,
            trusted_authority=True,
            plan_approval=True,
            detached_resume=True,
            retransmission_reservation=True,
            owner_fence_before_effect=True,
            structured_terminal_result=True,
            branch_suspension=True,
            privacy_safe_events=True,
        ),
        orchestration=PatchOrchestrationChecklist(
            shared_coordinator=True,
            originating_identity=True,
            approval_or_denial=True,
            retry_blocked_after_commit=True,
            durable_resume=True,
            dependent_suspension=True,
            coordinated_parallelism=True,
            committed_state_visibility=True,
        ),
        provider_codec=PatchProviderCodecChecklist(),
    )


def _identity(suffix: str) -> PatchProtocolIdentity:
    """Return one complete server-derived flow or task identity."""
    return PatchProtocolIdentity(
        tenant=PatchTenantId("tenant-flow-task-" + suffix),
        principal=PatchPrincipalId("principal-flow-task-" + suffix),
        execution=PatchExecutionId("execution_" + suffix * 16),
        run=PatchRunId("run-flow-task-" + suffix),
        session=PatchSessionId("session-flow-task-" + suffix),
        task=PatchTaskId("task-flow-task-" + suffix),
        agent=PatchAgentId("agent-flow-task-" + suffix),
        route=PolicyRouteId("route-flow-task-" + suffix),
        context=PatchContextId("context_" + suffix * 16),
        workspace=PatchWorkspaceId("workspace_" + suffix * 16),
    )


def _request(token: str) -> PatchProtocolFlowRequest:
    """Return a stable canonical edit request for one flow/task node."""
    return PatchProtocolFlowRequest(
        OperationType.EDIT,
        b'{"path":"note.txt","edits":['
        b'{"old_text":"before","new_text":"after"}]}',
        RetransmissionKey("phase14-flow-task-" + token),
        PatchObserverCorrelationId("correlation_" + token * 16),
        "patch_node",
    )


def _digest(value: str) -> AlgorithmDigest:
    """Return deterministic opaque digest evidence."""
    return AlgorithmDigest("sha256", value * 64)


@dataclass
class _Planner(PatchProtocolPlanPort):
    """Build one plan after reservation without a target effect."""

    calls: int = 0

    async def plan(
        self,
        reservation: PatchProtocolReservation,
        operation: OperationType,
        raw_arguments: bytes,
    ) -> DurablePlanReference:
        """Return a complete target-bound plan from the canonical request."""
        assert operation is OperationType.EDIT
        assert raw_arguments.startswith(b'{"path"')
        self.calls += 1
        return DurablePlanReference(
            PatchPlanId(
                "plan_"
                + reservation.request_id.value.removeprefix("request_")[:16]
            ),
            reservation.digest,
            _digest("f"),
            _digest("d"),
            reservation.identity.context,
            reservation.identity.workspace,
            PatchDomainId(
                "domain_"
                + reservation.identity.workspace.value.removeprefix(
                    "workspace_"
                )
            ),
            (
                DurableStepBinding(
                    PatchStepId("step_" + "a" * 16),
                    PatchLineageId("lineage_" + "a" * 16),
                ),
            ),
        )


@dataclass
class _Approvals(PatchProtocolApprovalPort):
    """Issue only the exact broker-attested durable approval for tests."""

    calls: int = 0

    async def approve(
        self,
        reservation: PatchProtocolReservation,
        plan: DurablePlanReference,
    ) -> DurableApproval:
        """Return one approval scoped to the durable request and plan."""
        self.calls += 1
        return _approval(
            reservation.durable.identity,
            reservation.digest,
            plan,
            "a",
        )


@dataclass
class _Clock:
    """Return monotonically selected test time for durable claims."""

    value: int = 1

    async def now(self) -> ExpiryTick:
        """Return the current durable test tick."""
        return ExpiryTick(self.value)


@dataclass
class _Effect(PatchProtocolEffectPort):
    """Record an effect only after durable owner/fence claim ordering."""

    store: InMemoryDurablePatchStore
    live: bool = True
    commits: int = 0
    reconciles: int = 0
    visible: str = "before"

    def _receipt(
        self,
        reservation: PatchProtocolReservation,
        plan: DurablePlanReference,
    ) -> PatchProtocolEffectReceipt:
        """Return pending or terminal target-owned journal evidence."""
        if self.live:
            return PatchProtocolEffectReceipt(
                WorkerReport(WorkerState.LIVE, None),
                _result(reservation, plan),
                ExpiryTick(2),
                DurablePendingRequest(
                    PatchPendingOperationId("pending_" + "a" * 16),
                    reservation.correlation,
                    DurationTicks(1),
                ),
            )
        return PatchProtocolEffectReceipt(
            WorkerReport(
                WorkerState.SETTLED,
                SettlementJournal(
                    (
                        JournalStep(
                            plan.steps[0].step_id,
                            plan.steps[0].lineage_id,
                            CommitStepState.COMMITTED,
                        ),
                    ),
                    (),
                    PostconditionState.ESTABLISHED,
                ),
            ),
            _result(reservation, plan),
            ExpiryTick(3),
        )

    async def commit(
        self,
        reservation: PatchProtocolReservation,
        plan: DurablePlanReference,
        lease: DurableCommitLease,
    ) -> PatchProtocolEffectReceipt:
        """Perform the one visible effect after durable commit ownership."""
        snapshot = await self.store.inspect(
            DurableRequestAccess(
                reservation.request_id, reservation.durable.identity
            )
        )
        assert snapshot.lifecycle is LifecyclePhase.COMMIT_STARTED
        assert snapshot.lease == lease
        self.commits += 1
        self.visible = "after"
        return self._receipt(reservation, plan)

    async def reconcile(
        self,
        reservation: PatchProtocolReservation,
        plan: DurablePlanReference,
        lease: DurableCommitLease,
    ) -> PatchProtocolEffectReceipt:
        """Read recovery truth without another visible effect."""
        del lease
        self.reconciles += 1
        return self._receipt(reservation, plan)


def _result(
    reservation: PatchProtocolReservation, plan: DurablePlanReference
) -> PatchResult:
    """Return one committed terminal result matching the sealed plan."""
    return PatchResult(
        1,
        reservation.request_id,
        plan.plan_id,
        LifecyclePhase.REQUEST_COMPLETED,
        PatchStatus.COMMITTED,
        CommitTruth(
            MutationState.COMMITTED,
            LineageState.COMMITTED,
            RequestedEffectOccurrence.TRUE,
            ArtifactState.ABSENT,
            WorkspaceChange.CHANGED,
            True,
            PostconditionState.ESTABLISHED,
        ),
        None,
    )


def _adapter(
    surface: PatchProtocolSurface,
    identity: PatchProtocolIdentity,
    store: InMemoryDurablePatchStore,
    effect: _Effect,
) -> PatchProtocolOrchestrationAdapter:
    """Bind durable coordinator ports for one orchestration surface."""
    return PatchProtocolOrchestrationAdapter(
        _profile(surface),
        identity,
        store,
        PatchRequestParser(PatchInputLimits()),
        PatchProtocolDurableCoordinator(
            store,
            _Planner(),
            _Approvals(),
            effect,
            _Clock(),
            DurationTicks(10),
        ),
    )


def test_patch_phase_14_approval_gate_rejects_replays_and_forgery() -> None:
    """Fence one real sealed review against replay, failure, and tampering."""

    async def scenario() -> None:
        """Drive the detached gate with a valid Phase 5 broker artifact."""
        phase_five = run_path("tests/patch/phase_5_contract_test.py")
        sealed_factory = phase_five["_sealed_plan"]
        subject_factory = phase_five["_subject"]
        requirements_factory = phase_five["_requirements"]
        assert callable(sealed_factory)
        assert callable(subject_factory)
        assert callable(requirements_factory)
        sealed = await sealed_factory()
        subject = subject_factory()
        requirements = requirements_factory(ApprovalMode.REQUIRE_REVIEW)
        assert type(subject) is not object
        review = PlanReviewRequest(sealed, subject, requirements)
        identity = PatchProtocolIdentity(
            subject.tenant,
            subject.principal,
            review.plan.binding.request.execution_id,
            subject.run,
            subject.session,
            subject.task,
            subject.agent,
            requirements.route,
            review.plan.binding.target.context_id,
            review.plan.binding.target.workspace_id,
        )
        durable_identity = DurableRequestIdentity(
            identity.tenant,
            identity.principal,
            identity.execution,
            identity.route,
            RetransmissionKey("phase14-gate-review"),
        )
        reservation = PatchProtocolReservation(
            PatchProtocolSurface.FLOW,
            identity,
            OperationType.EDIT,
            PatchObserverCorrelationId("correlation_" + "a" * 32),
            DurableReservation(
                review.plan.binding.request.request_id,
                durable_identity,
                review.plan.binding.request_digest,
                False,
            ),
        )
        gate = PatchProtocolApprovalGate()
        with pytest.raises(PatchProtocolError):
            gate.review_future(object())
        with pytest.raises(PatchProtocolError):
            await gate.decide(object())
        with pytest.raises(PatchProtocolError):
            await gate.approve(object())
        with pytest.raises(PatchProtocolError):
            await gate.approve(reservation)
        decision = create_task(gate.decide(review))
        await sleep(0)
        await gate.approve(reservation)
        assert (await decision).decisions[0].state.value == "approved"
        with pytest.raises(PatchProtocolError):
            await gate.decide(review)
        with pytest.raises(PatchProtocolError):
            await gate.approve(reservation)

        failed = PatchProtocolApprovalGate()
        failed.review_future(reservation.request_id)
        failed.fail(reservation.request_id, RuntimeError("unavailable"))
        with pytest.raises(PatchProtocolError):
            await failed.approve(reservation)
        with pytest.raises(PatchProtocolError):
            failed.fail(object(), RuntimeError("unavailable"))

        interrupted = PatchProtocolApprovalGate()
        waiting = create_task(interrupted.decide(review))
        await sleep(0)
        interrupted.fail(reservation.request_id, RuntimeError("unavailable"))
        with pytest.raises(RuntimeError):
            await waiting

        replay_race = PatchProtocolApprovalGate()
        replay_race._reviews[reservation.request_id] = (
            get_running_loop().create_future()
        )
        replay_race._decisions[reservation.request_id] = (
            get_running_loop().create_future()
        )
        with pytest.raises(PatchProtocolError):
            await replay_race.decide(review)

        forged = PatchProtocolApprovalGate()
        forged_review = deepcopy(review)
        object.__setattr__(
            forged_review,
            "subject",
            replace(
                forged_review.subject,
                principal=PatchPrincipalId("forged-principal"),
            ),
        )
        forged.review_future(reservation.request_id).set_result(forged_review)
        forged._decisions[reservation.request_id] = (
            get_running_loop().create_future()
        )
        with pytest.raises(PatchProtocolError):
            await forged.approve(reservation)

    run(scenario())


def test_patch_phase_14_durable_coordinator_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed coordinator ports without creating an extra effect."""

    class _BadClock:
        """Return an untyped clock value from a faulty durable clock port."""

        async def now(self) -> object:
            """Return an invalid selected clock value."""
            return object()

    class _BadPlanner:
        """Return a non-plan value from a faulty planner port."""

        async def plan(
            self,
            reservation: PatchProtocolReservation,
            operation: OperationType,
            raw_arguments: bytes,
        ) -> object:
            """Return a malformed planner response after typed input checks."""
            del reservation, operation, raw_arguments
            return object()

    class _BadApprovals:
        """Return a non-approval value from a faulty broker port."""

        async def approve(
            self,
            reservation: PatchProtocolReservation,
            plan: DurablePlanReference,
        ) -> object:
            """Return malformed evidence after valid plan selection."""
            del reservation, plan
            return object()

    async def reservation_for(
        store: InMemoryDurablePatchStore, token: str
    ) -> PatchProtocolReservation:
        """Reserve one exact flow request for a coordinator branch."""
        identity = _identity(token)
        request = _request(token)
        return await PatchProtocols(
            _profile(PatchProtocolSurface.FLOW), identity
        ).reserve(
            store,
            request.operation,
            request.raw_arguments,
            request.retransmission_key,
            request.correlation,
            PatchRequestParser(PatchInputLimits()),
        )

    def coordinator(
        store: InMemoryDurablePatchStore,
        planner: object = None,
        approvals: object = None,
        clock: object = None,
        effect: _Effect | None = None,
    ) -> PatchProtocolDurableCoordinator:
        """Bind one selected coordinator around a real in-memory store."""
        selected_effect = effect or _Effect(store)
        return PatchProtocolDurableCoordinator(
            store,
            _Planner() if planner is None else planner,
            _Approvals() if approvals is None else approvals,
            selected_effect,
            _Clock() if clock is None else clock,
            DurationTicks(10),
        )

    async def scenario() -> None:
        """Exercise invalid planning, approval, recovery, and receipt truth."""
        backend = _backend()
        store = InMemoryDurablePatchStore(backend)
        effect = _Effect(store)
        with pytest.raises(PatchProtocolError):
            PatchProtocolDurableCoordinator(
                object(),
                _Planner(),
                _Approvals(),
                effect,
                _Clock(),
                DurationTicks(1),
            )
        subject = coordinator(store, effect=effect)
        with pytest.raises(PatchProtocolError):
            await subject.plan(object(), OperationType.EDIT, b"{}")
        missing_plan = await reservation_for(store, "a")
        with pytest.raises(PatchProtocolError):
            await subject.approve(missing_plan)
        with pytest.raises(PatchProtocolError):
            await subject.await_result(missing_plan)

        bad_plan = await reservation_for(store, "b")
        with pytest.raises(PatchProtocolError):
            await coordinator(store, planner=_BadPlanner()).plan(
                bad_plan, OperationType.EDIT, _request("b").raw_arguments
            )

        bad_clock = await reservation_for(store, "c")
        bad_clock_coordinator = coordinator(store, clock=_BadClock())
        await bad_clock_coordinator.plan(
            bad_clock, OperationType.EDIT, _request("c").raw_arguments
        )
        with pytest.raises(PatchProtocolError):
            await bad_clock_coordinator.approve(bad_clock)

        bad_approval = await reservation_for(store, "d")
        bad_approval_coordinator = coordinator(
            store, approvals=_BadApprovals()
        )
        await bad_approval_coordinator.plan(
            bad_approval, OperationType.EDIT, _request("d").raw_arguments
        )
        with pytest.raises(PatchProtocolError):
            await bad_approval_coordinator.approve(bad_approval)

        attached = await reservation_for(store, "e")
        attached_coordinator = coordinator(store)
        await attached_coordinator.plan(
            attached, OperationType.EDIT, _request("e").raw_arguments
        )
        with monkeypatch.context() as patched:

            async def attached_claim(*_: object) -> DurableCommitClaim:
                """Model an already-owned durable request."""
                return DurableCommitClaim(
                    DurableCommitClaimState.ATTACHED, None, None
                )

            patched.setattr(store, "claim_commit", attached_claim)
            await attached_coordinator.approve(attached)

        absent_lease = await reservation_for(store, "f")
        absent_lease_coordinator = coordinator(store)
        await absent_lease_coordinator.plan(
            absent_lease, OperationType.EDIT, _request("f").raw_arguments
        )
        malformed_claim = object.__new__(DurableCommitClaim)
        object.__setattr__(
            malformed_claim, "state", DurableCommitClaimState.OWNER
        )
        object.__setattr__(malformed_claim, "lease", None)
        object.__setattr__(malformed_claim, "terminal", None)
        with monkeypatch.context() as patched:

            async def owner_without_lease(*_: object) -> DurableCommitClaim:
                """Model corrupted owner evidence without a fence lease."""
                return malformed_claim

            patched.setattr(store, "claim_commit", owner_without_lease)
            with pytest.raises(PatchProtocolError):
                await absent_lease_coordinator.approve(absent_lease)

        terminal_store = InMemoryDurablePatchStore(_backend())
        terminal_effect = _Effect(terminal_store, live=False)
        terminal_coordinator = coordinator(
            terminal_store, effect=terminal_effect
        )
        terminal = await reservation_for(terminal_store, "a")
        await terminal_coordinator.plan(
            terminal, OperationType.EDIT, _request("a").raw_arguments
        )
        await terminal_coordinator.approve(terminal)
        await terminal_coordinator.approve(terminal)
        await terminal_coordinator.await_result(terminal)

        recovery_store = InMemoryDurablePatchStore(_backend())
        recovery_clock = _Clock()
        recovery_effect = _Effect(recovery_store)
        recovery = coordinator(
            recovery_store, clock=recovery_clock, effect=recovery_effect
        )
        pending = await reservation_for(recovery_store, "b")
        await recovery.plan(
            pending, OperationType.EDIT, _request("b").raw_arguments
        )
        await recovery.approve(pending)
        snapshot = await recovery_store.inspect(
            DurableRequestAccess(pending.request_id, pending.durable.identity)
        )
        assert snapshot.plan is not None and snapshot.lease is not None
        with pytest.raises(PatchProtocolError):
            await recovery._reconcile_commit(
                pending, snapshot.plan, snapshot.lease, object()
            )
        recovery_clock.value = snapshot.lease.expires_at.value
        await recovery.await_result(pending)

        bad_recovery_clock = coordinator(
            recovery_store, clock=_BadClock(), effect=recovery_effect
        )
        with pytest.raises(PatchProtocolError):
            await bad_recovery_clock.await_result(pending)

        with pytest.raises(PatchProtocolError):
            PatchProtocolEffectReceipt(object(), object(), object())

        invalid_snapshot = await reservation_for(recovery_store, "c")
        with monkeypatch.context() as patched:

            async def corrupt_inspect(*_: object) -> object:
                """Model an invalid durable read before continuation I/O."""
                return object()

            patched.setattr(recovery_store, "inspect", corrupt_inspect)
            with pytest.raises(PatchProtocolError):
                await recovery.inspect(invalid_snapshot)

    run(scenario())


def test_patch_phase_14_selected_runtime_rehydration_fails_closed() -> None:
    """Reject rehydrated selected-runtime faults without target effects."""

    async def scenario() -> None:
        """Exercise real durable requests through effect-free port failures."""
        backend = _backend()
        store = InMemoryDurablePatchStore(backend)
        approvals = PatchProtocolApprovalGate()
        service = object.__new__(SandboxPatchSdkService)
        service._protocol_claimed = set()
        service._protocol_claim_waiters = {}
        with pytest.raises(TargetInspectionError):
            await service._await_protocol_claim(object())
        claimed = PatchRequestId("request_" + "a" * 32)
        service._signal_protocol_claim(claimed)
        await service._await_protocol_claim(claimed)
        unavailable = PatchRequestId("request_" + "b" * 32)
        failed_claim = create_task(service._await_protocol_claim(unavailable))
        await sleep(0)
        service._fail_protocol_claim(unavailable)
        with pytest.raises(TargetInspectionError):
            await failed_claim
        identity = _identity("c")
        service.configuration = SimpleNamespace(
            input_limits=PatchInputLimits(),
            subject=SimpleNamespace(
                tenant=identity.tenant,
                principal=identity.principal,
                run=identity.run,
                session=identity.session,
                task=identity.task,
                agent=identity.agent,
            ),
            execution_id=identity.execution,
        )
        service.policy = SimpleNamespace(
            approval=SimpleNamespace(route=identity.route)
        )
        service.runtime = SimpleNamespace(
            profile=SimpleNamespace(
                identity=SimpleNamespace(
                    context_id=identity.context,
                    workspace_id=identity.workspace,
                )
            )
        )
        with pytest.raises(TargetInspectionError):
            await service.invoke(
                OperationType.EDIT,
                _request("c").raw_arguments,
                object(),
                PatchRequestId("request_" + "c" * 32),
                _request("c").correlation,
                identity=object(),
            )
        with pytest.raises(TargetInspectionError):
            await service.invoke_remote(
                OperationType.EDIT,
                _request("c").raw_arguments,
                object(),
                PatchRequestId("request_" + "d" * 32),
                _request("c").correlation,
                object(),
            )
        host = object.__new__(PatchSdkHost)
        host._service = service
        toolset = object.__new__(PatchToolSet)
        toolset.sdk_host = lambda: host

        async def reserve(token: str) -> PatchProtocolReservation:
            """Reserve one trusted request before each rejection branch."""
            request = _request(token)
            return await PatchProtocols(
                _profile(PatchProtocolSurface.FLOW), identity
            ).reserve(
                store,
                request.operation,
                request.raw_arguments,
                request.retransmission_key,
                request.correlation,
                PatchRequestParser(PatchInputLimits()),
            )

        with pytest.raises(PatchProtocolError):
            PatchProtocolSelectedRuntime(toolset, service, object(), approvals)
        toolset.sdk_host = lambda: object()
        with pytest.raises(PatchProtocolError):
            PatchProtocolSelectedRuntime(toolset, service, store, approvals)
        toolset.sdk_host = lambda: host
        runtime = PatchProtocolSelectedRuntime(
            toolset, service, store, approvals
        )

        host_mismatch = await reserve("a")
        toolset.sdk_host = lambda: object()
        with pytest.raises(PatchProtocolError):
            await runtime.plan(
                host_mismatch,
                OperationType.EDIT,
                _request("a").raw_arguments,
            )
        toolset.sdk_host = lambda: host

        review_fault = await reserve("b")
        with pytest.raises(PatchProtocolError):
            await runtime.plan(
                review_fault,
                OperationType.EDIT,
                _request("b").raw_arguments,
            )

        drift = await reserve("c")

        async def pending_task() -> PatchInvocationOutcome:
            """Remain pending until the rejected resume branch cancels it."""
            await sleep(60)
            raise AssertionError("rehydrated selected task remained live")

        waiting = create_task(pending_task())
        runtime._tasks[drift.request_id] = waiting
        review = approvals.review_future(drift.request_id)
        review.set_result(object())
        with pytest.raises(PatchProtocolError):
            await runtime.plan(
                drift, OperationType.EDIT, _request("c").raw_arguments
            )
        waiting.cancel()
        with pytest.raises(CancelledError):
            await waiting

        no_record = await reserve("d")
        with pytest.raises(PatchProtocolError):
            await runtime.approve(no_record)
        with pytest.raises(PatchProtocolError):
            await runtime.await_result(no_record)

        cancellation = await reserve("e")

        async def cancellation_fault() -> PatchInvocationOutcome:
            """Translate cancellation into a selected runtime failure."""
            try:
                await sleep(60)
            except CancelledError as error:
                raise RuntimeError("rehydrated runtime failed") from error
            raise AssertionError("rehydrated selected task did not cancel")

        runtime._requests[cancellation.request_id] = (
            cancellation,
            OperationType.EDIT,
            _request("e").raw_arguments,
        )
        runtime._tasks[cancellation.request_id] = create_task(
            cancellation_fault()
        )
        await sleep(0)
        service._signal_protocol_claim(cancellation.request_id)

        async def bypass_approval(*_: object) -> None:
            """Release the selected cancellation branch only."""

        original_approve = approvals.approve
        approvals.approve = bypass_approval
        try:
            with pytest.raises(PatchProtocolError):
                await runtime.approve(cancellation)
        finally:
            approvals.approve = original_approve

        no_host = await reserve("f")
        runtime._requests[no_host.request_id] = (
            no_host,
            OperationType.EDIT,
            _request("f").raw_arguments,
        )
        toolset.sdk_host = lambda: object()
        with pytest.raises(PatchProtocolError):
            await runtime.await_result(no_host)
        toolset.sdk_host = lambda: host

        inspected = await reserve("aa")
        inspection_task = create_task(pending_task())
        runtime._requests[inspected.request_id] = (
            inspected,
            OperationType.EDIT,
            _request("aa").raw_arguments,
        )
        runtime._hosts[inspected.request_id] = host
        runtime._tasks[inspected.request_id] = inspection_task
        with pytest.raises(PatchProtocolError):
            await runtime.await_result(inspected)
        inspection_task.cancel()
        with pytest.raises(CancelledError):
            await inspection_task

        dispatch_fault = await reserve("bb")
        runtime._requests[dispatch_fault.request_id] = (
            dispatch_fault,
            OperationType.EDIT,
            _request("bb").raw_arguments,
        )
        with pytest.raises(PatchProtocolError):
            await runtime.await_result(dispatch_fault)

        with pytest.raises(PatchProtocolError):
            await runtime.inspect(object())
        invalid_snapshot = await reserve("cc")
        inspect = store.inspect

        async def corrupt_inspect(
            access: DurableRequestAccess,
        ) -> object:
            """Return corrupt durable evidence for one selected request."""
            if access.request_id == invalid_snapshot.request_id:
                return object()
            return await inspect(access)

        store.inspect = corrupt_inspect
        try:
            with pytest.raises(PatchProtocolError):
                await runtime.inspect(invalid_snapshot)
        finally:
            store.inspect = inspect

        failed_request = PatchRequestId("request_" + "f" * 32)

        async def failed_task() -> PatchInvocationOutcome:
            """Raise one rehydrated selected-runtime task failure."""
            raise RuntimeError("rehydrated selected task failed")

        selected_task = create_task(failed_task())
        await sleep(0)
        with pytest.raises(RuntimeError):
            await selected_task
        runtime._complete_task(failed_request, selected_task)
        failed_review = approvals.review_future(failed_request)
        with pytest.raises(PatchProtocolError):
            failed_review.result()

        wrong_snapshot = await reserve("ee")
        with pytest.raises(PatchProtocolError):
            runtime._validate_snapshot(wrong_snapshot, object())
        wrong_origin = DurablePlanReference(
            PatchPlanId("plan_" + "e" * 16),
            wrong_snapshot.digest,
            _digest("f"),
            _digest("a"),
            wrong_snapshot.identity.context,
            wrong_snapshot.identity.workspace,
            PatchDomainId("domain_" + "e" * 32),
            (
                DurableStepBinding(
                    PatchStepId("step_" + "e" * 32),
                    PatchLineageId("lineage_" + "e" * 32),
                ),
            ),
            replace(
                wrong_snapshot.identity.durable_origin(),
                agent_id=PatchAgentId("agent-flow-task-other"),
            ),
        )
        backend.records[wrong_snapshot.durable.identity].plan = wrong_origin
        with pytest.raises(PatchProtocolError):
            await runtime.inspect(wrong_snapshot)

        impossible = await reserve("ff")
        backend.records[impossible.durable.identity].lifecycle = (
            LifecyclePhase.PARSED
        )
        with pytest.raises(PatchProtocolError):
            await PatchProtocols(
                _profile(PatchProtocolSurface.FLOW), identity
            ).inspect(store, impossible)

        service.store = store
        service._requests = {}
        service.configuration.approval_issuer = SimpleNamespace()
        service_request = PatchRequestId("request_" + "d" * 32)
        service_identity = DurableRequestIdentity(
            identity.tenant,
            identity.principal,
            identity.execution,
            identity.route,
            RetransmissionKey("sandbox-" + service_request.value),
        )
        service_origin = identity.durable_origin()

        def plan_for(
            reservation: DurableReservation,
            origin: DurableProtocolOrigin,
        ) -> DurablePlanReference:
            """Return one persisted plan shape for service recovery faults."""
            return DurablePlanReference(
                PatchPlanId("plan_" + "d" * 16),
                reservation.canonical_digest,
                _digest("f"),
                _digest("a"),
                identity.context,
                identity.workspace,
                PatchDomainId("domain_" + "d" * 32),
                (
                    DurableStepBinding(
                        PatchStepId("step_" + "d" * 32),
                        PatchLineageId("lineage_" + "d" * 32),
                    ),
                ),
                origin,
                b"x",
            )

        async def invoke_with_snapshot(
            request_id: PatchRequestId,
            prepare: object,
        ) -> None:
            """Inject persisted recovery evidence after its exact reserve."""
            inspect = store.inspect

            async def prepared(access: DurableRequestAccess) -> object:
                """Prepare one deterministic durable service snapshot."""
                callback = prepare
                assert callable(callback)
                callback(backend.records[access.identity])
                return await inspect(access)

            store.inspect = prepared
            try:
                await service.invoke(
                    OperationType.EDIT,
                    _request("d").raw_arguments,
                    object(),
                    request_id,
                    _request("d").correlation,
                    identity=DurableRequestIdentity(
                        service_identity.tenant_id,
                        service_identity.principal_id,
                        service_identity.execution_id,
                        service_identity.route_id,
                        RetransmissionKey("sandbox-" + request_id.value),
                    ),
                    origin=service_origin,
                )
            finally:
                store.inspect = inspect

        def mismatch_origin(record: object) -> None:
            """Inject a prior plan sealed for a different originating agent."""
            reservation = getattr(record, "reservation")
            assert type(reservation) is DurableReservation
            record.plan = plan_for(
                reservation,
                replace(
                    service_origin,
                    agent_id=PatchAgentId("agent-flow-task-stale"),
                ),
            )

        with pytest.raises(TargetInspectionError):
            await invoke_with_snapshot(service_request, mismatch_origin)

        def missing_plan(record: object) -> None:
            """Inject an impossible planned record with no sealed plan."""
            record.lifecycle = LifecyclePhase.PLANNED
            record.plan = None

        with pytest.raises(TargetInspectionError):
            await invoke_with_snapshot(
                PatchRequestId("request_" + "e" * 32), missing_plan
            )

        def durable_plan(record: object) -> None:
            """Inject an authentic-shaped plan before capsule reopening."""
            reservation = getattr(record, "reservation")
            assert type(reservation) is DurableReservation
            record.lifecycle = LifecyclePhase.PLANNED
            record.plan = plan_for(reservation, service_origin)

        def capsule_fault(*_: object) -> object:
            """Fail opening a persisted capsule before approval or effect."""
            raise RuntimeError("capsule fault")

        service.configuration.approval_issuer.open_plan_material = (
            capsule_fault
        )
        with pytest.raises(TargetInspectionError):
            await invoke_with_snapshot(
                PatchRequestId("request_" + "f" * 32), durable_plan
            )

        def capsule_interrupt(*_: object) -> object:
            """Preserve process interruption during capsule reopening."""
            raise KeyboardInterrupt

        service.configuration.approval_issuer.open_plan_material = (
            capsule_interrupt
        )
        with pytest.raises(KeyboardInterrupt):
            await invoke_with_snapshot(
                PatchRequestId("request_" + "a" * 32), durable_plan
            )

    run(scenario())


def test_patch_phase_14_sealed_restart_capsule_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind every decrypted plan fact to its original protocol authority."""
    phase_six = run_path("tests/patch/phase_6_contract_test.py")
    sealed_plan = phase_six["_sealed_plan"]
    assert callable(sealed_plan)
    plan = run(sealed_plan())
    authority = HmacDurableApprovalAuthority.random()
    issuer = object.__new__(PhaseFiveDurableApprovalIssuer)
    issuer._authority = authority
    identity = DurableRequestIdentity(
        PatchTenantId("tenant-a"),
        PatchPrincipalId("principal-a"),
        PatchExecutionId("execution_" + "a" * 16),
        PolicyRouteId("route-six"),
        RetransmissionKey("phase14-capsule"),
    )
    origin = DurableProtocolOrigin(
        identity.tenant_id,
        identity.principal_id,
        identity.execution_id,
        PatchRunId("run-a"),
        PatchSessionId("session-a"),
        PatchTaskId("task-a"),
        PatchAgentId("agent-a"),
        identity.route_id,
        PatchContextId("context_" + "a" * 16),
        PatchWorkspaceId("workspace_" + "a" * 16),
    )
    reference = _durable_plan(
        plan,
        origin,
        issuer.seal_plan_material(identity, origin, plan),
    )
    assert (
        issuer.open_plan_material(
            identity, origin, plan.binding.request.request_id, reference
        )
        == plan
    )

    def assert_mismatch(candidate: object) -> None:
        """Reject untrusted rehydration before review or effect."""
        with pytest.raises(
            DurableStoreError,
            match=DurableStoreErrorCode.PLAN_MISMATCH.value,
        ):
            issuer.open_plan_material(
                identity,
                origin,
                plan.binding.request.request_id,
                candidate,
            )

    assert_mismatch(object())
    assert_mismatch(replace(reference, rehydration=b"x"))
    assert_mismatch(replace(reference, rehydration=b"x" * 13))
    assert_mismatch(replace(reference, fingerprint_digest=_digest("f")))
    with pytest.raises(
        DurableStoreError, match=DurableStoreErrorCode.PLAN_MISMATCH.value
    ):
        issuer.seal_plan_material(
            identity,
            replace(origin, agent_id=PatchAgentId("agent-b")),
            plan,
        )

    monkeypatch.setattr(
        "avalan.patch.durable_approval.pickle_loads", lambda _: object()
    )
    assert_mismatch(reference)
    monkeypatch.undo()

    def fail_seal(*_: object, **__: object) -> object:
        """Model a trusted seal failure while reopening persisted material."""
        raise RuntimeError("capsule seal fault")

    monkeypatch.setattr("avalan.patch.durable_approval.seal_plan", fail_seal)
    assert_mismatch(reference)

    def interrupt_seal(*_: object, **__: object) -> object:
        """Preserve process interrupts from the trusted recovery boundary."""
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "avalan.patch.durable_approval.seal_plan", interrupt_seal
    )
    with pytest.raises(KeyboardInterrupt):
        issuer.open_plan_material(
            identity, origin, plan.binding.request.request_id, reference
        )


def test_patch_e2e_033_flow_suspends_dependents_then_reads_once() -> None:
    """Suspend a real flow until one restarted patch request settles."""

    async def scenario() -> tuple[str, int, int, int]:
        store = InMemoryDurablePatchStore(_backend())
        identity = _identity("a")
        effect = _Effect(store)
        patch = PatchProtocolFlowAdapter(
            _adapter(PatchProtocolSurface.FLOW, identity, store, effect)
        )
        request = _request("a")
        reads = 0

        async def patch_node(_: Mapping[str, object]) -> object:
            return await patch.execute(request, approve=True)

        async def read_node(inputs: Mapping[str, object]) -> str:
            nonlocal reads
            reads += 1
            assert effect.visible == "after"
            value = next(iter(inputs.values()))
            assert isinstance(value, PatchResult)
            assert value.status is PatchStatus.COMMITTED
            return effect.visible

        first = Flow()
        first.add_node(Node("patch", func=patch_node))
        first.add_node(Node("read", func=read_node))
        first.add_connection("patch", "read")
        suspended = await first.execute_async()
        assert isinstance(suspended, PatchProtocolFlowSuspension)
        assert reads == 0
        assert effect.commits == 1

        effect.live = False
        restarted = PatchProtocolFlowAdapter(
            _adapter(PatchProtocolSurface.FLOW, identity, store, effect)
        )

        async def resumed_node(_: Mapping[str, object]) -> object:
            return await restarted.resume(request)

        resumed = Flow()
        resumed.add_node(Node("patch", func=resumed_node))
        resumed.add_node(Node("read", func=read_node))
        resumed.add_connection("patch", "read")
        assert await resumed.execute_async() == "after"
        assert (effect.commits, effect.reconciles, reads) == (1, 1, 1)

        with pytest.raises(Exception):
            await restarted.execute(
                PatchProtocolFlowRequest(
                    request.operation,
                    request.raw_arguments.replace(b"after", b"later"),
                    request.retransmission_key,
                    request.correlation,
                    request.mutation_slot,
                )
            )
        assert effect.commits == 1
        with pytest.raises(Exception):
            await restarted.execute(
                PatchProtocolFlowRequest(
                    request.operation,
                    request.raw_arguments,
                    RetransmissionKey("phase14-flow-task-fresh-retry"),
                    request.correlation,
                    request.mutation_slot,
                )
            )
        assert effect.commits == 1
        return effect.visible, effect.commits, effect.reconciles, reads

    observed = run(scenario())
    assert observed[0] == "after"
    assert observed[1:] == (1, 1, 1)


def test_patch_e2e_034_task_adapter_reconciles_without_effect_retry() -> None:
    """Recover one task-owned pending effect without issuing another commit."""

    async def scenario() -> None:
        store = InMemoryDurablePatchStore(_backend())
        identity = _identity("b")
        effect = _Effect(store)
        task = PatchProtocolQueuedTaskAdapter(
            _adapter(PatchProtocolSurface.TASK, identity, store, effect)
        )
        request = _request("b")
        pending = await task.execute(request, approve=True)
        assert pending.pending is not None
        assert effect.commits == 1
        effect.live = False
        terminal = await PatchProtocolQueuedTaskAdapter(
            _adapter(PatchProtocolSurface.TASK, identity, store, effect)
        ).recover(request)
        assert terminal.result is not None
        assert effect.commits == 1
        assert effect.reconciles == 1

    run(scenario())


def test_patch_e2e_033_selected_runtime_suspends_then_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run flow stages through the selected sandbox SDK effect runtime."""

    async def scenario() -> None:
        sys_path.insert(0, "tests/patch")
        try:
            phase_ten = run_path("tests/patch/phase_10_contract_test.py")
        finally:
            sys_path.remove("tests/patch")
        settings_factory = phase_ten["_settings"]
        subject_factory = phase_ten["_runtime_subject"]
        policy_factory = phase_ten["_sandbox_corpus_policy"]
        clock_type = phase_ten["_RuntimeClock"]
        blocking_store_type = phase_ten["_BlockingFenceStore"]
        native_probe = phase_ten["_native_probe"]
        assert callable(settings_factory)
        assert callable(subject_factory)
        assert callable(policy_factory)
        assert callable(clock_type)
        assert callable(blocking_store_type)
        assert callable(native_probe)
        if not await native_probe():
            pytest.skip(
                "selected runtime E2E requires an available native sandbox"
            )
        root = tmp_path / "sandbox-view"
        namespace = tmp_path / "sandbox-private"
        root.mkdir()
        namespace.mkdir()
        note = root / "note.txt"
        note.write_text("before\n", encoding="utf-8")
        settings = settings_factory(root, namespace)
        subject = subject_factory()
        policy = policy_factory()
        clock = clock_type()
        approvals = PatchProtocolApprovalGate()
        authority = HmacDurableApprovalAuthority.random()
        store = blocking_store_type(
            InMemoryDurablePatchBackend(approval_verifier=authority)
        )
        approval_service = ApprovalService(
            approvals, clock, RuntimeGrantStore()
        )
        configuration = SandboxPatchServiceConfiguration(
            subject,
            phase_ten["PlannerFacade"](
                phase_ten["BoundedPlannerWorker"](1),
                phase_ten["PlannerLimits"](),
            ),
            approval_service,
            phase_ten["PhaseFiveDurableApprovalIssuer"](
                approval_service,
                authority,
            ),
            clock,
            DurationTicks(10),
            DurationTicks(10),
            execution_id=PatchExecutionId("execution_" + "b" * 32),
        )
        binder = SandboxPatchRuntimeBinder.from_settings(
            settings,
            configuration,
            policy,
            PatchApprovalBinding(True),
            PatchCoordinatorBinding(True, store),
            PatchPersistenceBinding(True, store),
        )
        bundle = await PatchToolLoader(
            binder,
            activated_patch_test_profile(),
        ).load(enable_tools=["patch.edit", "patch.apply"])
        toolset = bundle.toolset
        binding = bundle.runtime_binding
        assert binding is not None
        assert isinstance(binding.service, SandboxPatchSdkService)
        await toolset.__aenter__()
        try:
            claimed_request = PatchRequestId("request_" + "d" * 32)
            with pytest.raises(TargetInspectionError):
                await binding.service._await_protocol_claim(object())
            binding.service._signal_protocol_claim(claimed_request)
            await binding.service._await_protocol_claim(claimed_request)
            failed_request = PatchRequestId("request_" + "e" * 32)
            failed_claim = create_task(
                binding.service._await_protocol_claim(failed_request)
            )
            await sleep(0)
            binding.service._fail_protocol_claim(failed_request)
            with pytest.raises(TargetInspectionError):
                await failed_claim
            with pytest.raises(TargetInspectionError):
                await binding.service.invoke(
                    OperationType.EDIT,
                    b'{"path":"note.txt","edits":['
                    b'{"old_text":"before\\n","new_text":"after\\n"}]}',
                    toolset.capability,
                    PatchRequestId("request_" + "f" * 32),
                    PatchObserverCorrelationId("correlation_" + "f" * 32),
                    identity=object(),
                )
            with pytest.raises(TargetInspectionError):
                await binding.service.invoke_remote(
                    OperationType.EDIT,
                    b'{"path":"note.txt","edits":['
                    b'{"old_text":"before\\n","new_text":"after\\n"}]}',
                    toolset.capability,
                    PatchRequestId("request_" + "a" * 32),
                    PatchObserverCorrelationId("correlation_" + "a" * 32),
                    object(),
                )
            runtime_identity = settings.context.identity
            identity = PatchProtocolIdentity(
                subject.tenant,
                subject.principal,
                PatchExecutionId("execution_" + "b" * 32),
                subject.run,
                subject.session,
                subject.task,
                subject.agent,
                policy.approval.route,
                runtime_identity.context_id,
                runtime_identity.workspace_id,
            )
            runtime = PatchProtocolSelectedRuntime(
                toolset, binding.service, store, approvals
            )

            adapter = PatchProtocolFlowAdapter(
                PatchProtocolOrchestrationAdapter(
                    _profile(PatchProtocolSurface.FLOW),
                    identity,
                    store,
                    PatchRequestParser(PatchInputLimits()),
                    runtime,
                )
            )
            request = PatchProtocolFlowRequest(
                OperationType.EDIT,
                b'{"path":"note.txt","edits":['
                b'{"old_text":"before\\n","new_text":"after\\n"}]}',
                RetransmissionKey("phase14-selected-runtime"),
                PatchObserverCorrelationId("correlation_" + "b" * 32),
                "selected_runtime",
            )
            rejected_origins = (
                replace(
                    identity,
                    tenant=PatchTenantId("tenant-phase14-other"),
                ),
                replace(
                    identity,
                    principal=PatchPrincipalId("principal-phase14-other"),
                ),
                replace(
                    identity,
                    execution=PatchExecutionId("execution_" + "c" * 32),
                ),
                replace(identity, run=PatchRunId("run-phase14-other")),
                replace(
                    identity,
                    session=PatchSessionId("session-phase14-other"),
                ),
                replace(identity, task=PatchTaskId("task-phase14-other")),
                replace(identity, agent=PatchAgentId("agent-phase14-other")),
                replace(identity, route=PolicyRouteId("route-phase14-other")),
                replace(
                    identity,
                    context=PatchContextId("context_" + "c" * 32),
                ),
                replace(
                    identity,
                    workspace=PatchWorkspaceId("workspace_" + "c" * 32),
                ),
            )
            for rejected_identity in rejected_origins:
                rejected_reservation = PatchProtocolReservation(
                    PatchProtocolSurface.FLOW,
                    rejected_identity,
                    request.operation,
                    request.correlation,
                    DurableReservation(
                        PatchRequestId("request_" + "c" * 32),
                        rejected_identity.durable_identity(
                            request.retransmission_key
                        ),
                        AlgorithmDigest("sha256", "c" * 64),
                        False,
                    ),
                )
                with pytest.raises(PatchProtocolError):
                    await runtime.plan(
                        rejected_reservation,
                        request.operation,
                        request.raw_arguments,
                    )
            assert not runtime._tasks
            first = await adapter.execute(request)
            assert isinstance(first, PatchProtocolFlowSuspension)
            assert note.read_text(encoding="utf-8") == "before\n"
            pending = await adapter.execute(request, approve=True)
            assert isinstance(pending, PatchProtocolFlowSuspension)
            assert pending.continuation.pending is not None
            await store.effect_reached.wait()
            still_pending = await adapter.resume(request)
            assert isinstance(still_pending, PatchProtocolFlowSuspension)
            store.release_effect.set()
            terminal = await store.await_terminal(
                DurablePendingAccess(
                    DurableRequestAccess(
                        pending.continuation.reservation.request_id,
                        pending.continuation.reservation.durable.identity,
                    ),
                    pending.continuation.pending.pending_operation_id,
                    pending.continuation.pending.correlation_id,
                )
            )
            resumed = await adapter.resume(request)
            assert isinstance(resumed, PatchResult)
            assert resumed == terminal.result
            assert note.read_text(encoding="utf-8") == "after\n"
            task = PatchProtocolQueuedTaskAdapter(
                PatchProtocolOrchestrationAdapter(
                    _profile(PatchProtocolSurface.TASK),
                    identity,
                    store,
                    PatchRequestParser(PatchInputLimits()),
                    runtime,
                )
            )
            task_request = PatchProtocolFlowRequest(
                OperationType.EDIT,
                b'{"path":"note.txt","edits":['
                b'{"old_text":"after\\n","new_text":"final\\n"}]}',
                RetransmissionKey("phase14-selected-runtime-task"),
                PatchObserverCorrelationId("correlation_" + "c" * 32),
                "selected_runtime_task",
            )
            task_review = await task.execute(task_request)
            assert task_review.pending is None
            task_pending = await task.execute(task_request, approve=True)
            assert task_pending.pending is not None
            task_terminal = await store.await_terminal(
                DurablePendingAccess(
                    DurableRequestAccess(
                        task_pending.reservation.request_id,
                        task_pending.reservation.durable.identity,
                    ),
                    task_pending.pending.pending_operation_id,
                    task_pending.pending.correlation_id,
                )
            )
            recovered = await task.recover(task_request)
            assert recovered.result == task_terminal.result
            assert note.read_text(encoding="utf-8") == "final\n"

            @dataclass(frozen=True)
            class _SelectedResolver:
                """Return the one server-derived selected runtime identity."""

                async def __call__(
                    self, request: Request
                ) -> PatchProtocolIdentity | None:
                    """Resolve the exact authenticated protocol identity."""
                    assert isinstance(request, Request)
                    return identity

            resolver: PatchProtocolIdentityResolver = _SelectedResolver()

            protocol_app = FastAPI()
            install_patch_protocol_test_routes(
                protocol_app,
                PatchProtocolAdapterConfiguration(
                    _profile(PatchProtocolSurface.MCP),
                    _profile(PatchProtocolSurface.A2A),
                    store,
                    resolver,
                    runtime,
                    b"r" * 32,
                ),
            )
            prefix = "/__avalan_test__/patch-protocol/v1"
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=protocol_app, client=("127.0.0.1", 1)
                ),
                base_url="http://testserver",
            ) as client:
                mcp = await client.post(
                    prefix + "/mcp",
                    json={
                        "id": "selected-mcp",
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "arguments": {
                                "edits": [
                                    {
                                        "new_text": "mcp\n",
                                        "old_text": "final\n",
                                    }
                                ],
                                "path": "note.txt",
                            },
                            "name": "patch.edit",
                            "retransmission_key": "phase14-selected-mcp",
                        },
                    },
                )
                assert mcp.status_code == 200
                mcp_value = mcp.json()["result"]["structuredContent"]
                mcp_handle = mcp_value["operation_handle"]
                assert mcp_value["state"] == "approval_required"
                approved = await client.post(
                    prefix + "/mcp/operations/" + mcp_handle + "/approval"
                )
                assert approved.status_code == 200
                assert (
                    approved.json()["result"]["structuredContent"]["state"]
                    == "settlement_pending"
                )
                mcp_reservation = next(
                    reservation
                    for reservation, _, _ in runtime._requests.values()
                    if (
                        reservation.surface is PatchProtocolSurface.MCP
                        and (
                            reservation.durable.identity.retransmission_key.value
                            == "phase14-selected-mcp"
                        )
                    )
                )
                mcp_snapshot = await store.inspect(
                    DurableRequestAccess(
                        mcp_reservation.request_id,
                        mcp_reservation.durable.identity,
                    )
                )
                assert mcp_snapshot.pending is not None
                await store.await_terminal(
                    DurablePendingAccess(
                        DurableRequestAccess(
                            mcp_snapshot.reservation.request_id,
                            mcp_snapshot.reservation.identity,
                        ),
                        mcp_snapshot.pending.pending_operation_id,
                        mcp_snapshot.pending.correlation_id,
                    )
                )
                await runtime.await_result(mcp_reservation)
                terminal_mcp = await client.post(
                    prefix + "/mcp/operations/" + mcp_handle + "/status"
                )
                assert terminal_mcp.status_code == 200
                assert (
                    terminal_mcp.json()["result"]["structuredContent"]["state"]
                    == "terminal"
                )
                assert note.read_text(encoding="utf-8") == "mcp\n"
                a2a = await client.post(
                    prefix + "/a2a",
                    json={
                        "id": "selected-a2a",
                        "jsonrpc": "2.0",
                        "method": "message/send",
                        "params": {
                            "message": {
                                "arguments": {
                                    "edits": [
                                        {
                                            "new_text": "a2a\n",
                                            "old_text": "mcp\n",
                                        }
                                    ],
                                    "path": "note.txt",
                                },
                                "kind": "patch.call",
                                "name": "patch.edit",
                                "retransmission_key": "phase14-selected-a2a",
                            },
                            "task_id": identity.task.value,
                        },
                    },
                )
                assert a2a.status_code == 200
                a2a_value = a2a.json()["result"]["status"]["message"]
                a2a_handle = a2a_value["operation_handle"]
                assert a2a_value["state"] == "approval_required"
                approved_a2a = await client.post(
                    prefix + "/a2a",
                    json={
                        "id": "selected-a2a-approval",
                        "jsonrpc": "2.0",
                        "method": "message/send",
                        "params": {
                            "message": {
                                "kind": "patch.approval",
                                "operation_handle": a2a_handle,
                            },
                            "task_id": identity.task.value,
                        },
                    },
                )
                assert approved_a2a.status_code == 200
                assert (
                    approved_a2a.json()["result"]["status"]["message"]["state"]
                    == "settlement_pending"
                )
                a2a_reservation = next(
                    reservation
                    for reservation, _, _ in runtime._requests.values()
                    if (
                        reservation.surface is PatchProtocolSurface.A2A
                        and (
                            reservation.durable.identity.retransmission_key.value
                            == "phase14-selected-a2a"
                        )
                    )
                )
                a2a_snapshot = await store.inspect(
                    DurableRequestAccess(
                        a2a_reservation.request_id,
                        a2a_reservation.durable.identity,
                    )
                )
                assert a2a_snapshot.pending is not None
                await store.await_terminal(
                    DurablePendingAccess(
                        DurableRequestAccess(
                            a2a_snapshot.reservation.request_id,
                            a2a_snapshot.reservation.identity,
                        ),
                        a2a_snapshot.pending.pending_operation_id,
                        a2a_snapshot.pending.correlation_id,
                    )
                )
                terminal_a2a = await client.post(
                    prefix + "/a2a",
                    json={
                        "id": "selected-a2a-terminal",
                        "jsonrpc": "2.0",
                        "method": "tasks/get",
                        "params": {
                            "operation_handle": a2a_handle,
                            "task_id": identity.task.value,
                        },
                    },
                )
                assert terminal_a2a.status_code == 200
                assert (
                    terminal_a2a.json()["result"]["status"]["state"]
                    == "completed"
                )
                assert note.read_text(encoding="utf-8") == "a2a\n"
                provider = PatchProtocolProviderAdapter(
                    replace(
                        _profile(PatchProtocolSurface.PROVIDER_FREEFORM),
                        provider_codec=PatchProviderCodecChecklist(
                            advertised=True,
                            complete_buffering=True,
                            grammar_and_limits=True,
                            stable_correlation=True,
                            replay_fencing=True,
                            result_injection=True,
                            approval_suspension=True,
                            idempotency_and_resume=True,
                            authority_and_disclosure=True,
                        ),
                    ),
                    identity,
                    RawProviderProfile("phase14-selected-provider"),
                    store,
                    PatchRequestParser(PatchInputLimits()),
                    runtime,
                )
                provider_key = RetransmissionKey("phase14-selected-provider")
                provider_call_id = RawToolCallId("phase14-selected-call")
                provider_correlation = provider.correlation_for(
                    provider_call_id, provider_key
                )
                provider_review = await provider.apply_freeform(
                    PatchProtocolProviderCall(
                        RawProviderProfile("phase14-selected-provider"),
                        provider_call_id,
                        provider_correlation,
                        provider_key,
                        "grammar-v1",
                        (
                            (
                                b"*** Begin Patch v1\n"
                                b"*** Update File: note.txt\n@@\n"
                            ),
                            b"-a2a\n+provider\n*** End Patch\n",
                        ),
                        True,
                    )
                )
                assert provider_review.pending is None
                provider_pending = await provider.approve(
                    provider_review.reservation
                )
                if provider_pending.result is None:
                    provider_snapshot = await store.inspect(
                        DurableRequestAccess(
                            provider_pending.reservation.request_id,
                            provider_pending.reservation.durable.identity,
                        )
                    )
                    if provider_snapshot.pending is not None:
                        await store.await_terminal(
                            DurablePendingAccess(
                                DurableRequestAccess(
                                    provider_snapshot.reservation.request_id,
                                    provider_snapshot.reservation.identity,
                                ),
                                provider_snapshot.pending.pending_operation_id,
                                provider_snapshot.pending.correlation_id,
                            )
                        )
                    provider_pending = await provider.resume(
                        provider_review.reservation
                    )
                provider_result = provider.reinject(provider_pending)
                assert provider_result.correlation == provider_correlation
                assert note.read_text(encoding="utf-8") == "provider\n"
        finally:
            await toolset.__aexit__(None, None, None)

    run(scenario())
