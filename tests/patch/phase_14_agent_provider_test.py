"""Exercise Phase 14 multi-agent and selected provider projections."""

from asyncio import run
from dataclasses import dataclass, replace

import pytest
from phase_8_store_test import _approval, _backend

from avalan.patch.coordinator import (
    RetransmissionKey,
)
from avalan.patch.domain import (
    AlgorithmDigest,
    ArtifactState,
    Audience,
    CommitStepState,
    CommitTruth,
    DurationTicks,
    ExpiryTick,
    LifecyclePhase,
    LineageState,
    LogicalPath,
    MutationState,
    OperationType,
    PatchCommitOwnerId,
    PatchContextId,
    PatchDomainId,
    PatchEventId,
    PatchExecutionId,
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
    SequenceNumber,
    WorkspaceChange,
)
from avalan.patch.durable_retention import StaticDurableRetentionAuthorizer
from avalan.patch.durable_store import (
    DurableCoordinationAccess,
    DurableCoordinationAdmission,
    DurableOutboxRecord,
    DurablePendingRecord,
    DurablePlanReference,
    DurableRequestAccess,
    DurableRequestSnapshot,
    DurableReservation,
    DurableRetentionAccess,
    DurableRetentionKind,
    DurableRetentionPolicy,
    DurableRetentionRecord,
    DurableStepBinding,
    DurableStoreError,
    DurableStoreErrorCode,
    DurableTerminalRecord,
    EncryptedRetentionValue,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.parser import (
    CanonicalPatchRequest,
    PatchDocumentSyntax,
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderProfile,
    RawToolCallId,
    UpdateDeclarationSyntax,
)
from avalan.patch.policy import (
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PolicyRouteId,
)
from avalan.patch.protocols import (
    PatchOrchestrationChecklist,
    PatchProtocolChecklist,
    PatchProtocolContinuation,
    PatchProtocolContinuationKind,
    PatchProtocolCoordinationDomain,
    PatchProtocolError,
    PatchProtocolFlowAdapter,
    PatchProtocolFlowRequest,
    PatchProtocolFlowSuspension,
    PatchProtocolIdentity,
    PatchProtocolMultiAgentAdapter,
    PatchProtocolOrchestrationAdapter,
    PatchProtocolProfile,
    PatchProtocolProviderAdapter,
    PatchProtocolProviderCall,
    PatchProtocolProviderItemOrigin,
    PatchProtocolQueuedTaskAdapter,
    PatchProtocolReservation,
    PatchProtocolResultInjection,
    PatchProtocols,
    PatchProtocolSurface,
    PatchProviderCodecChecklist,
    _apply_json_arguments,
    _canonical_json_request,
    _canonical_paths,
    _json_string,
)


def _protocol() -> PatchProtocolChecklist:
    """Return the complete exact protocol activation checklist."""
    return PatchProtocolChecklist(
        canonical_input=True,
        trusted_authority=True,
        plan_approval=True,
        detached_resume=True,
        retransmission_reservation=True,
        owner_fence_before_effect=True,
        structured_terminal_result=True,
        branch_suspension=True,
        privacy_safe_events=True,
    )


def _orchestration() -> PatchOrchestrationChecklist:
    """Return the complete exact orchestration checklist."""
    return PatchOrchestrationChecklist(
        shared_coordinator=True,
        originating_identity=True,
        approval_or_denial=True,
        retry_blocked_after_commit=True,
        durable_resume=True,
        dependent_suspension=True,
        coordinated_parallelism=True,
        committed_state_visibility=True,
    )


def _codec() -> PatchProviderCodecChecklist:
    """Return the complete exact selected-codec checklist."""
    return PatchProviderCodecChecklist(
        advertised=True,
        complete_buffering=True,
        grammar_and_limits=True,
        stable_correlation=True,
        replay_fencing=True,
        result_injection=True,
        approval_suspension=True,
        idempotency_and_resume=True,
        authority_and_disclosure=True,
    )


def _profile(surface: PatchProtocolSurface) -> PatchProtocolProfile:
    """Return one exact active test-only profile."""
    return PatchProtocolProfile(
        surface=surface,
        enabled=True,
        authenticated=True,
        loopback_only=True,
        protocol=_protocol(),
        orchestration=_orchestration(),
        provider_codec=_codec(),
    )


def _identity(
    suffix: str, workspace: PatchWorkspaceId
) -> PatchProtocolIdentity:
    """Return one different agent/context for the same backing workspace."""
    return PatchProtocolIdentity(
        tenant=PatchTenantId("tenant-phase14-agents"),
        principal=PatchPrincipalId("principal-phase14-agents"),
        execution=PatchExecutionId("execution_" + suffix * 32),
        run=PatchRunId("run-phase14-" + suffix),
        session=PatchSessionId("session-phase14-" + suffix),
        task=PatchTaskId("task-phase14-" + suffix),
        agent=PatchAgentId("agent-phase14-" + suffix),
        route=PolicyRouteId("route-phase14-agents"),
        context=PatchContextId("context_" + suffix * 32),
        workspace=workspace,
    )


def _request(path: str, token: str) -> PatchProtocolFlowRequest:
    """Return one stable agent-owned canonical JSON edit request."""
    return PatchProtocolFlowRequest(
        OperationType.EDIT,
        (
            b'{"path":"'
            + path.encode()
            + b'","edits":[{"old_text":"before","new_text":"after"}]}'
        ),
        RetransmissionKey("phase14-agent-" + token),
        PatchObserverCorrelationId("correlation_" + token * 32),
        "mutation_" + token,
    )


def _digest(token: str) -> AlgorithmDigest:
    """Return a deterministic opaque digest for sealed test plan fields."""
    return AlgorithmDigest("sha256", token * 64)


def _result(
    reservation: PatchProtocolReservation, plan: DurablePlanReference
) -> PatchResult:
    """Return one content-free committed terminal result."""
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


@dataclass
class _Runtime:
    """Implement one target-free durable runtime with pending settlement."""

    backend: InMemoryDurablePatchBackend
    domain: PatchDomainId
    plans: int = 0
    approvals: int = 0
    resumes: int = 0

    async def plan(
        self,
        reservation: PatchProtocolReservation,
        operation: OperationType,
        raw_arguments: bytes,
    ) -> None:
        """Persist one exact domain-owned plan after reservation."""
        assert operation in {OperationType.EDIT, OperationType.APPLY}
        assert raw_arguments
        record = self.backend.records[reservation.durable.identity]
        if record.plan is not None:
            return
        self.plans += 1
        plan = DurablePlanReference(
            PatchPlanId.new(),
            reservation.digest,
            _digest("f"),
            _digest("a"),
            reservation.identity.context,
            reservation.identity.workspace,
            self.domain,
            (
                DurableStepBinding(
                    PatchStepId("step_" + "a" * 32),
                    PatchLineageId("lineage_" + "b" * 32),
                ),
            ),
        )
        record.plan = plan
        record.lifecycle = LifecyclePhase.PLANNED

    async def approve(self, reservation: PatchProtocolReservation) -> None:
        """Enter durable settlement pending without a second planning path."""
        self.approvals += 1
        record = self.backend.records[reservation.durable.identity]
        if record.terminal is not None:
            return
        assert record.lifecycle is LifecyclePhase.PLANNED
        record.lifecycle = LifecyclePhase.SETTLEMENT_PENDING
        record.pending = DurablePendingRecord(
            reservation.request_id,
            reservation.identity.execution,
            PatchPendingOperationId.new(),
            reservation.correlation,
            SequenceNumber(1),
            SequenceNumber(1),
            False,
            DurationTicks(1),
        )

    async def await_result(
        self, reservation: PatchProtocolReservation
    ) -> None:
        """Settle only the already planned pending request."""
        self.resumes += 1
        record = self.backend.records[reservation.durable.identity]
        pending = record.pending
        assert record.plan is not None
        assert record.lifecycle is LifecyclePhase.SETTLEMENT_PENDING
        assert pending is not None
        record.pending = None
        record.lifecycle = LifecyclePhase.REQUEST_COMPLETED
        record.terminal = DurableTerminalRecord(
            _result(reservation, record.plan),
            DurableOutboxRecord(
                PatchEventId.new(),
                reservation.request_id,
                SequenceNumber(2),
                LifecyclePhase.REQUEST_COMPLETED,
                reservation.correlation,
            ),
            pending.pending_operation_id,
        )

    async def inspect(
        self, reservation: PatchProtocolReservation
    ) -> DurableRequestSnapshot:
        """Return exact durable truth for the selected request only."""
        return await InMemoryDurablePatchStore(self.backend).inspect(
            DurableRequestAccess(
                reservation.request_id, reservation.durable.identity
            )
        )


@dataclass
class _PlanFaultRuntime(_Runtime):
    """Fail after admission and before durable plan persistence."""

    async def plan(
        self,
        reservation: PatchProtocolReservation,
        operation: OperationType,
        raw_arguments: bytes,
    ) -> None:
        """Raise the injected planning fault without recording a plan."""
        del reservation, operation, raw_arguments
        self.plans += 1
        raise RuntimeError("injected multi-agent planning fault")


@dataclass
class _TerminalAfterAdmissionRuntime(_Runtime):
    """Settle only after durable shared-workspace admission is recorded."""

    inspections: int = 0

    async def inspect(
        self, reservation: PatchProtocolReservation
    ) -> DurableRequestSnapshot:
        """Return a terminal snapshot at the post-admission race boundary."""
        self.inspections += 1
        if self.inspections == 2:
            record = self.backend.records[reservation.durable.identity]
            plan = DurablePlanReference(
                PatchPlanId("plan_" + "c" * 16),
                reservation.digest,
                _digest("f"),
                _digest("a"),
                reservation.identity.context,
                reservation.identity.workspace,
                self.domain,
                (
                    DurableStepBinding(
                        PatchStepId("step_" + "c" * 32),
                        PatchLineageId("lineage_" + "c" * 32),
                    ),
                ),
            )
            record.plan = plan
            record.lifecycle = LifecyclePhase.REQUEST_COMPLETED
            record.terminal = DurableTerminalRecord(
                _result(reservation, plan),
                DurableOutboxRecord(
                    PatchEventId("event_" + "c" * 32),
                    reservation.request_id,
                    SequenceNumber(1),
                    LifecyclePhase.REQUEST_COMPLETED,
                    reservation.correlation,
                ),
                None,
            )
        return await super().inspect(reservation)


def test_patch_phase_14_releases_unplanned_agent_admission_on_plan_fault() -> (
    None
):
    """Release durable workspace ownership when planning raises after admit."""

    async def scenario() -> None:
        workspace = PatchWorkspaceId("workspace_" + "f" * 32)
        domain = PatchProtocolCoordinationDomain.for_workspace(workspace)
        backend = InMemoryDurablePatchBackend()
        store = InMemoryDurablePatchStore(backend)
        adapter = PatchProtocolMultiAgentAdapter(
            _profile(PatchProtocolSurface.MULTI_AGENT),
            _identity("f", workspace),
            domain,
            store,
            PatchRequestParser(PatchInputLimits()),
            _PlanFaultRuntime(backend, domain.owner),
        )
        with pytest.raises(RuntimeError, match="planning fault"):
            await adapter.execute(_request("note.txt", "f"))
        assert not backend.coordination

    run(scenario())


def test_phase_14_agent_admission_terminal_race_rejects_corruption() -> None:
    """Release a terminal race while refusing corrupt unplanned evidence."""

    async def scenario() -> None:
        workspace = PatchWorkspaceId("workspace_" + "c" * 32)
        identity = _identity("c", workspace)
        domain = PatchProtocolCoordinationDomain.for_workspace(workspace)
        backend = InMemoryDurablePatchBackend()
        store = InMemoryDurablePatchStore(backend)
        adapter = PatchProtocolMultiAgentAdapter(
            _profile(PatchProtocolSurface.MULTI_AGENT),
            identity,
            domain,
            store,
            PatchRequestParser(PatchInputLimits()),
            _TerminalAfterAdmissionRuntime(backend, domain.owner),
        )
        terminal = await adapter.execute(_request("note.txt", "c"))
        assert terminal.result is not None
        assert not backend.coordination

        reservation = await PatchProtocols(
            _profile(PatchProtocolSurface.MULTI_AGENT), identity
        ).reserve(
            store,
            OperationType.EDIT,
            _request("note.txt", "d").raw_arguments,
            RetransmissionKey("phase14-corrupt-release"),
            PatchObserverCorrelationId("correlation_" + "d" * 32),
            PatchRequestParser(PatchInputLimits()),
        )
        inspect = store.inspect

        async def corrupt(_: DurableRequestAccess) -> object:
            """Model an unavailable durable read before admission cleanup."""
            return object()

        store.inspect = corrupt
        try:
            with pytest.raises(PatchProtocolError):
                await adapter._release_unplanned(reservation)
        finally:
            store.inspect = inspect

    run(scenario())


def test_patch_phase_14_terminal_settlement_releases_detached_admission() -> (
    None
):
    """Delete terminal ownership during settlement."""

    async def scenario() -> None:
        workspace = PatchWorkspaceId("workspace_" + "e" * 32)
        identity = _identity("e", workspace)
        domain = PatchProtocolCoordinationDomain.for_workspace(workspace)
        backend = _backend()
        store = InMemoryDurablePatchStore(backend)
        reservation = await store.reserve(
            identity.durable_identity(RetransmissionKey("phase14-detached-e")),
            _digest("e"),
            PatchRequestId("request_" + "e" * 32),
        )
        admission = DurableCoordinationAdmission(
            DurableCoordinationAccess(
                reservation,
                identity.run,
                identity.session,
                identity.task,
                identity.agent,
                identity.context,
                identity.workspace,
                domain.owner,
            ),
            frozenset((LogicalPath("note.txt"),)),
        )
        await store.admit_coordination(admission)
        with pytest.raises(DurableStoreError):
            await store.release_terminal_coordination(
                DurableRequestAccess(
                    reservation.request_id, reservation.identity
                )
            )
        origin = identity.durable_origin()
        assert not origin.matches(object())
        with pytest.raises(DurableStoreError):
            replace(origin, tenant_id=object())
        plan = DurablePlanReference(
            PatchPlanId("plan_" + "e" * 16),
            reservation.canonical_digest,
            _digest("f"),
            _digest("a"),
            identity.context,
            identity.workspace,
            domain.owner,
            (
                DurableStepBinding(
                    PatchStepId("step_" + "e" * 32),
                    PatchLineageId("lineage_" + "e" * 32),
                ),
            ),
            identity.durable_origin(),
        )
        await store.persist_plan(reservation, plan)
        claim = await store.claim_commit(
            reservation,
            plan,
            _approval(
                reservation.identity,
                reservation.canonical_digest,
                plan,
                "e",
            ),
            PatchCommitOwnerId("owner_" + "e" * 16),
            ExpiryTick(1),
            DurationTicks(10),
            (),
        )
        assert claim.lease is not None
        journal = (
            await store.inspect(
                DurableRequestAccess(
                    reservation.request_id, reservation.identity
                )
            )
        ).journal
        step = plan.steps[0]
        journal = await store.append_step(
            claim.lease,
            journal.cursor,
            step.step_id,
            CommitStepState.PLANNED,
            ExpiryTick(2),
        )
        journal = await store.append_step(
            claim.lease,
            journal.cursor,
            step.step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(3),
        )
        await store.settle(
            claim.lease,
            journal.cursor,
            PatchResult(
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
            ),
            PatchObserverCorrelationId("correlation_" + "e" * 32),
            ExpiryTick(4),
        )
        assert not await store.is_coordination_admitted(admission.access)
        backend.coordination[admission.access.workspace_id] = admission
        assert await store.is_coordination_admitted(admission.access)
        await store.release_terminal_coordination(
            DurableRequestAccess(reservation.request_id, reservation.identity)
        )
        assert not await store.is_coordination_admitted(admission.access)

    run(scenario())


def test_patch_phase_14_duplicate_active_commit_returns_public_pending() -> (
    None
):
    """Expose a stable suspension instead of an unavailable active commit."""

    async def scenario() -> None:
        workspace = PatchWorkspaceId("workspace_" + "d" * 32)
        identity = _identity("d", workspace)
        domain = PatchProtocolCoordinationDomain.for_workspace(workspace)
        backend = InMemoryDurablePatchBackend()
        runtime = _Runtime(backend, domain.owner)
        adapter = PatchProtocolFlowAdapter(
            PatchProtocolOrchestrationAdapter(
                _profile(PatchProtocolSurface.FLOW),
                identity,
                InMemoryDurablePatchStore(backend),
                PatchRequestParser(PatchInputLimits()),
                runtime,
            )
        )
        request = _request("note.txt", "d")
        initial = await adapter.execute(request)
        assert type(initial) is PatchProtocolFlowSuspension
        reservation = initial.continuation.reservation
        record = backend.records[reservation.durable.identity]
        record.lifecycle = LifecyclePhase.COMMIT_STARTED
        duplicate = await adapter.execute(request)
        assert type(duplicate) is PatchProtocolFlowSuspension
        pending = duplicate.continuation.pending
        assert pending is not None
        assert pending.request_id == reservation.request_id
        assert pending.correlation_id == reservation.correlation
        assert runtime.plans == 1

    run(scenario())


def test_patch_e2e_035_coordinates_three_agent_contexts() -> None:
    """Serialize same-workspace agents and deny cross-agent opaque access."""

    async def scenario() -> tuple[int, int, int]:
        workspace = PatchWorkspaceId("workspace_" + "c" * 32)
        domain = PatchProtocolCoordinationDomain.for_workspace(workspace)
        backend = InMemoryDurablePatchBackend()
        store = InMemoryDurablePatchStore(backend)
        runtime = _Runtime(backend, domain.owner)
        local = PatchProtocolMultiAgentAdapter(
            _profile(PatchProtocolSurface.MULTI_AGENT),
            _identity("a", workspace),
            domain,
            store,
            PatchRequestParser(PatchInputLimits()),
            runtime,
        )
        sandbox = PatchProtocolMultiAgentAdapter(
            _profile(PatchProtocolSurface.MULTI_AGENT),
            _identity("b", workspace),
            domain,
            store,
            PatchRequestParser(PatchInputLimits()),
            runtime,
        )
        container = PatchProtocolMultiAgentAdapter(
            _profile(PatchProtocolSurface.MULTI_AGENT),
            _identity("c", workspace),
            domain,
            store,
            PatchRequestParser(PatchInputLimits()),
            runtime,
        )
        assert (
            local.domain_owner
            == sandbox.domain_owner
            == container.domain_owner
        )
        local_request = _request("local.txt", "a")
        local_pending = await local.execute(local_request, approve=True)
        assert local_pending.pending is not None
        assert await domain.is_admitted(
            store, _identity("a", workspace), local_pending.reservation
        )
        with pytest.raises(PatchProtocolError):
            await sandbox.inspect(local_pending.reservation)
        with pytest.raises(PatchProtocolError):
            await sandbox.execute(_request("local.txt", "b"), approve=True)
        with pytest.raises(PatchProtocolError):
            await container.execute(_request("other.txt", "c"), approve=True)
        assert runtime.plans == runtime.approvals == 1
        local_terminal = await local.resume(local_request)
        assert local_terminal.result is not None
        assert not await domain.is_admitted(
            store, _identity("a", workspace), local_terminal.reservation
        )
        container_pending = await container.execute(
            _request("other.txt", "c"), approve=True
        )
        assert container_pending.pending is not None
        container_terminal = await container.resume(_request("other.txt", "c"))
        assert container_terminal.result is not None
        return runtime.plans, runtime.approvals, runtime.resumes

    assert run(scenario()) == (2, 2, 2)


def test_patch_e2e_036_provider_json_freeform_parity_and_inert_replay() -> (
    None
):
    """Project only complete current freeform input onto canonical JSON."""

    async def scenario() -> tuple[PatchProtocolResultInjection, int]:
        workspace = PatchWorkspaceId("workspace_" + "d" * 32)
        identity = _identity("d", workspace)
        domain = PatchProtocolCoordinationDomain.for_workspace(workspace)
        backend = InMemoryDurablePatchBackend()
        store = InMemoryDurablePatchStore(backend)
        runtime = _Runtime(backend, domain.owner)
        adapter = PatchProtocolProviderAdapter(
            _profile(PatchProtocolSurface.PROVIDER_FREEFORM),
            identity,
            RawProviderProfile("phase14-provider"),
            store,
            PatchRequestParser(PatchInputLimits()),
            runtime,
        )
        assert adapter.advertised_tools == ("patch.apply",)
        key = RetransmissionKey("phase14-provider-key")
        call_id = RawToolCallId("phase14-provider-call")
        correlation = adapter.correlation_for(call_id, key)
        document = (
            b"*** Begin Patch v1\n*** Add File: note.txt\n+after\n"
            b"*** End Patch\n"
        )
        call = PatchProtocolProviderCall(
            RawProviderProfile("phase14-provider"),
            call_id,
            correlation,
            key,
            "grammar-v1",
            (document[:11], document[11:37], document[37:]),
            True,
        )
        freeform = await adapter.apply_freeform(call)
        portable = await adapter.apply_json(
            b'{"patch":"*** Begin Patch v1\\n*** Add File: note.txt\\n'
            b'+after\\n*** End Patch\\n"}',
            key,
            correlation,
        )
        assert (
            freeform.reservation.request_id == portable.reservation.request_id
        )
        assert not freeform.reservation.durable.replayed
        assert portable.reservation.durable.replayed
        assert freeform.reservation.digest == portable.reservation.digest
        assert runtime.plans == 1
        with pytest.raises(PatchProtocolError):
            adapter.reinject(freeform)
        for origin in PatchProtocolProviderItemOrigin:
            if origin is PatchProtocolProviderItemOrigin.CURRENT:
                continue
            with pytest.raises(PatchProtocolError):
                await adapter.apply_freeform(call, origin=origin)
        with pytest.raises(PatchProtocolError):
            await adapter.apply_freeform(
                replace(
                    call,
                    correlation=PatchObserverCorrelationId(
                        "correlation_" + "e" * 32
                    ),
                )
            )
        changed_key = RetransmissionKey("phase14-provider-other-key")
        with pytest.raises(PatchProtocolError):
            await adapter.apply_freeform(
                replace(
                    call,
                    retransmission_key=changed_key,
                    correlation=adapter.correlation_for(call_id, changed_key),
                )
            )
        with pytest.raises(PatchProtocolError):
            await adapter.apply_json(
                b'{"patch":"*** Begin Patch v1\\n*** Add File: other.txt\\n'
                b'+after\\n*** End Patch\\n"}',
                key,
                correlation,
            )
        assert runtime.plans == 1
        pending = await adapter.approve(freeform.reservation)
        assert pending.pending is not None
        terminal = await adapter.resume(freeform.reservation)
        injection = adapter.reinject(terminal)
        assert injection.request_id == freeform.reservation.request_id
        assert injection.correlation == correlation
        assert injection.status == PatchStatus.COMMITTED.value
        with pytest.raises(PatchProtocolError):
            PatchProtocolProviderCall(
                call.provider_profile,
                RawToolCallId("incomplete"),
                adapter.correlation_for(RawToolCallId("incomplete"), key),
                key,
                "grammar-v1",
                (),
                False,
            )
        return injection, len(backend.records)

    injection, records = run(scenario())
    assert injection.lifecycle is LifecyclePhase.REQUEST_COMPLETED
    assert records == 1


def test_patch_phase_14_provider_rejects_foreign_and_noncurrent_calls() -> (
    None
):
    """Exercise provider rejection branches without bypassing its parser."""

    async def scenario() -> PatchProtocolContinuation:
        """Dispatch selected calls, then reject every unowned projection."""
        workspace = PatchWorkspaceId("workspace_" + "e" * 32)
        identity = _identity("e", workspace)
        domain = PatchProtocolCoordinationDomain.for_workspace(workspace)
        backend = InMemoryDurablePatchBackend()
        store = InMemoryDurablePatchStore(backend)
        runtime = _Runtime(backend, domain.owner)
        parser = PatchRequestParser(PatchInputLimits())
        profile = RawProviderProfile("phase14-negative-provider")
        adapter = PatchProtocolProviderAdapter(
            _profile(PatchProtocolSurface.PROVIDER_NATIVE),
            identity,
            profile,
            store,
            parser,
            runtime,
        )
        inactive = PatchProtocolProviderAdapter(
            replace(
                _profile(PatchProtocolSurface.PROVIDER_NATIVE), enabled=False
            ),
            identity,
            profile,
            store,
            parser,
            runtime,
        )
        assert inactive.advertised_tools == ()
        key = RetransmissionKey("phase14-provider-negative-key")
        call_id = RawToolCallId("phase14-provider-negative-call")
        correlation = adapter.correlation_for(call_id, key)
        with pytest.raises(PatchProtocolError):
            adapter.correlation_for(object(), key)
        with pytest.raises(PatchProtocolError):
            adapter.correlation_for(call_id, object())
        with pytest.raises(PatchProtocolError):
            PatchProtocolProviderCall(
                profile,
                call_id,
                correlation,
                key,
                "grammar-v2",
                (b"value",),
                True,
            )
        with pytest.raises(PatchProtocolError):
            PatchProtocolProviderCall(
                profile,
                call_id,
                correlation,
                key,
                "grammar-v1",
                (b"value",),
                False,
            )
        with pytest.raises(PatchProtocolError):
            await adapter.apply_json(
                b"{}",
                key,
                correlation,
                origin=PatchProtocolProviderItemOrigin.HISTORY,
            )
        with pytest.raises(PatchProtocolError):
            await inactive.apply_json(b"{}", key, correlation)
        with pytest.raises(PatchProtocolError):
            await adapter.apply_json(b"not-json", key, correlation)
        invalid = PatchProtocolProviderCall(
            profile,
            call_id,
            correlation,
            key,
            "grammar-v1",
            (b"not-a-patch",),
            True,
        )
        with pytest.raises(PatchProtocolError):
            await adapter.apply_freeform(invalid)
        with pytest.raises(PatchProtocolError):
            await adapter.apply_freeform(
                replace(invalid, provider_profile=RawProviderProfile("other"))
            )
        first = await adapter.apply_json(
            b'{"patch":"*** Begin Patch v1\\n*** Add File: note.txt\\n'
            b'+after\\n*** End Patch\\n"}',
            key,
            correlation,
        )
        assert first.kind is PatchProtocolContinuationKind.APPROVAL_REQUIRED
        assert await adapter.resume(first.reservation) == first
        foreign = PatchProtocolProviderAdapter(
            _profile(PatchProtocolSurface.PROVIDER_NATIVE),
            identity,
            profile,
            store,
            parser,
            runtime,
        )
        with pytest.raises(PatchProtocolError):
            await foreign.approve(first.reservation)
        with pytest.raises(PatchProtocolError):
            foreign.reinject(
                PatchProtocolContinuation(
                    PatchProtocolContinuationKind.APPROVAL_REQUIRED,
                    first.reservation,
                )
            )
        pending = await adapter.approve(first.reservation)
        assert pending.kind is PatchProtocolContinuationKind.SETTLEMENT_PENDING
        terminal = await adapter.resume(first.reservation)
        assert terminal.completed
        assert await adapter.approve(first.reservation) == terminal
        with pytest.raises(PatchProtocolError):
            adapter.reinject(object())
        with pytest.raises(PatchProtocolError):
            PatchProtocolResultInjection(
                first.reservation.request_id,
                correlation,
                LifecyclePhase.PLANNED,
                PatchStatus.COMMITTED.value,
            )
        return terminal

    observed = run(scenario())
    assert observed.kind is PatchProtocolContinuationKind.TERMINAL
    assert observed.completed


def test_patch_phase_14_multi_agent_rejects_bad_durable_owner_bindings() -> (
    None
):
    """Reject malformed or mismatched durable coordination ownership."""

    async def scenario() -> None:
        """Exercise owner access, invalid plans, and terminal release truth."""
        workspace = PatchWorkspaceId("workspace_" + "f" * 32)
        identity = _identity("f", workspace)
        domain = PatchProtocolCoordinationDomain.for_workspace(workspace)
        invalid_backend = InMemoryDurablePatchBackend()
        invalid_store = InMemoryDurablePatchStore(invalid_backend)
        runtime = _Runtime(
            invalid_backend, PatchDomainId("domain_" + "a" * 32)
        )
        adapter = PatchProtocolMultiAgentAdapter(
            _profile(PatchProtocolSurface.MULTI_AGENT),
            identity,
            domain,
            invalid_store,
            PatchRequestParser(PatchInputLimits()),
            runtime,
        )
        with pytest.raises(PatchProtocolError):
            PatchProtocolCoordinationDomain(workspace, domain.owner, object())
        with pytest.raises(PatchProtocolError):
            PatchProtocolCoordinationDomain.for_workspace(object())
        with pytest.raises(PatchProtocolError):
            await adapter.execute(_request("bad.txt", "f"))
        assert runtime.plans == 1
        backend = InMemoryDurablePatchBackend()
        store = InMemoryDurablePatchStore(backend)
        protocols = PatchProtocolMultiAgentAdapter(
            _profile(PatchProtocolSurface.MULTI_AGENT),
            identity,
            domain,
            store,
            PatchRequestParser(PatchInputLimits()),
            _Runtime(backend, domain.owner),
        )
        request = _request("released.txt", "d")
        reservation = await protocols._protocol.reserve(
            store,
            request.operation,
            request.raw_arguments,
            request.retransmission_key,
            request.correlation,
            PatchRequestParser(PatchInputLimits()),
            request_id=protocols._request_id(request),
        )
        await domain.admit(
            store,
            identity,
            reservation,
            frozenset((LogicalPath("released.txt"),)),
        )
        assert await domain.is_admitted(store, identity, reservation)
        with pytest.raises(PatchProtocolError):
            await domain.is_admitted(
                store, _identity("a", workspace), reservation
            )
        await domain.release(store, identity, reservation)
        assert not await domain.is_admitted(store, identity, reservation)
        with pytest.raises(PatchProtocolError):
            await protocols._release_terminal(object())

    run(scenario())


def test_patch_phase_14_protocol_values_and_coordination_fail_closed() -> None:
    """Reject malformed protocol values and preserve serial ownership truth."""

    async def scenario() -> None:
        """Exercise value validation around real durable admissions."""
        workspace = PatchWorkspaceId("workspace_" + "a" * 32)
        identity = _identity("a", workspace)
        backend = InMemoryDurablePatchBackend()
        store = InMemoryDurablePatchStore(backend)
        protocol = PatchProtocols(
            _profile(PatchProtocolSurface.MULTI_AGENT), identity
        )
        request = _request("owned.txt", "a")
        reservation = await protocol.reserve(
            store,
            request.operation,
            request.raw_arguments,
            request.retransmission_key,
            request.correlation,
            PatchRequestParser(PatchInputLimits()),
        )
        assert reservation.request_id == reservation.durable.request_id
        assert reservation.digest == reservation.durable.canonical_digest
        with pytest.raises(PatchProtocolError):
            await protocol.reserve(
                store,
                object(),
                request.raw_arguments,
                request.retransmission_key,
                request.correlation,
                PatchRequestParser(PatchInputLimits()),
            )
        with pytest.raises(PatchProtocolError):
            await protocol.inspect(store, object())
        with pytest.raises(PatchProtocolError):
            PatchProtocolFlowRequest(
                request.operation,
                request.raw_arguments,
                request.retransmission_key,
                request.correlation,
                "not-a-slot",
            )
        with pytest.raises(PatchProtocolError):
            PatchProtocolReservation(
                PatchProtocolSurface.MULTI_AGENT,
                identity,
                request.operation,
                request.correlation,
                object(),
            )
        with pytest.raises(PatchProtocolError):
            PatchProtocolContinuation(
                PatchProtocolContinuationKind.TERMINAL, reservation
            )
        with pytest.raises(PatchProtocolError):
            _json_string(object())

        domain = PatchProtocolCoordinationDomain.for_workspace(workspace)
        paths = frozenset((LogicalPath("owned.txt"),))
        await domain.admit(store, identity, reservation, paths)
        assert await domain.is_admitted(store, identity, reservation)
        other_identity = _identity("b", workspace)
        other_reservation = await PatchProtocols(
            _profile(PatchProtocolSurface.MULTI_AGENT), other_identity
        ).reserve(
            store,
            request.operation,
            request.raw_arguments,
            RetransmissionKey("phase14-agent-foreign-release"),
            PatchObserverCorrelationId("correlation_" + "b" * 32),
            PatchRequestParser(PatchInputLimits()),
        )
        foreign_access = DurableCoordinationAccess(
            other_reservation.durable,
            other_identity.run,
            other_identity.session,
            other_identity.task,
            other_identity.agent,
            other_identity.context,
            other_identity.workspace,
            domain.owner,
        )
        with pytest.raises(DurableStoreError):
            await store.release_coordination(foreign_access)
        with pytest.raises(PatchProtocolError):
            await domain.is_admitted(
                store, _identity("b", workspace), reservation
            )
        with pytest.raises(DurableStoreError):
            await domain.admit(store, identity, reservation, frozenset())
        await domain.release(store, identity, reservation)
        assert not await domain.is_admitted(store, identity, reservation)
        await domain.release(store, identity, reservation)

        await domain.admit(store, identity, reservation, paths)
        runtime = _Runtime(backend, domain.owner)
        await runtime.plan(
            reservation, request.operation, request.raw_arguments
        )
        with pytest.raises(PatchProtocolError):
            await domain.release(store, identity, reservation)
        await runtime.approve(reservation)
        await runtime.await_result(reservation)
        with pytest.raises(PatchProtocolError):
            await domain.admit(store, identity, reservation, paths)
        await domain.release(store, identity, reservation)

        access = DurableCoordinationAccess(
            reservation.durable,
            identity.run,
            identity.session,
            identity.task,
            identity.agent,
            identity.context,
            identity.workspace,
            domain.owner,
        )
        with pytest.raises(DurableStoreError):
            DurableCoordinationAccess(
                object(),
                identity.run,
                identity.session,
                identity.task,
                identity.agent,
                identity.context,
                identity.workspace,
                domain.owner,
            )
        with pytest.raises(DurableStoreError):
            DurableCoordinationAdmission(access, frozenset())

    run(scenario())


def test_patch_phase_14_orchestration_boundaries_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject inert orchestration bindings and unavailable serial storage."""

    class _UnavailableStore:
        """Model one durable coordination backend that rejects every call."""

        async def admit_coordination(
            self, _: DurableCoordinationAdmission
        ) -> None:
            """Reject an admission after authenticating its full authority."""
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

        async def release_coordination(
            self, _: DurableCoordinationAccess
        ) -> None:
            """Reject a release after authenticating its full authority."""
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

        async def is_coordination_admitted(
            self, _: DurableCoordinationAccess
        ) -> bool:
            """Reject an ownership read after authenticating its authority."""
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

    async def scenario() -> None:
        """Drive every adapter error through a real durable reservation."""
        workspace = PatchWorkspaceId("workspace_" + "b" * 32)
        identity = _identity("b", workspace)
        domain = PatchProtocolCoordinationDomain.for_workspace(workspace)
        backend = InMemoryDurablePatchBackend()
        store = InMemoryDurablePatchStore(backend)
        request = _request("orchestration.txt", "b")
        reservation = await PatchProtocols(
            _profile(PatchProtocolSurface.MULTI_AGENT), identity
        ).reserve(
            store,
            request.operation,
            request.raw_arguments,
            request.retransmission_key,
            request.correlation,
            PatchRequestParser(PatchInputLimits()),
        )
        paths = frozenset((LogicalPath("orchestration.txt"),))
        assert domain.policy.value == "serial"
        with pytest.raises(PatchProtocolError):
            await domain.admit(object(), identity, reservation, paths)
        with pytest.raises(PatchProtocolError):
            await domain.release(object(), identity, reservation)
        with pytest.raises(PatchProtocolError):
            await domain.is_admitted(object(), identity, reservation)
        unavailable = _UnavailableStore()
        with pytest.raises(PatchProtocolError):
            await domain.admit(unavailable, identity, reservation, paths)
        with pytest.raises(PatchProtocolError):
            await domain.release(unavailable, identity, reservation)
        with pytest.raises(PatchProtocolError):
            await domain.is_admitted(unavailable, identity, reservation)
        with monkeypatch.context() as patched:

            def unavailable_access(*_: object) -> DurableCoordinationAccess:
                """Model a rejected durable coordination access constructor."""
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )

            patched.setattr(
                "avalan.patch.protocols.DurableCoordinationAccess",
                unavailable_access,
            )
            with pytest.raises(PatchProtocolError):
                await domain.is_admitted(store, identity, reservation)

        with pytest.raises(PatchProtocolError):
            PatchProtocolFlowSuspension(object())
        with pytest.raises(PatchProtocolError):
            PatchProtocolOrchestrationAdapter(
                object(),
                identity,
                store,
                PatchRequestParser(PatchInputLimits()),
                object(),
            )
        with pytest.raises(PatchProtocolError):
            PatchProtocolFlowAdapter(object())
        with pytest.raises(PatchProtocolError):
            PatchProtocolQueuedTaskAdapter(object())
        with pytest.raises(PatchProtocolError):
            PatchProtocolMultiAgentAdapter(
                object(),
                identity,
                domain,
                store,
                PatchRequestParser(PatchInputLimits()),
                object(),
            )

        runtime = _Runtime(backend, domain.owner)
        adapter = PatchProtocolMultiAgentAdapter(
            _profile(PatchProtocolSurface.MULTI_AGENT),
            identity,
            domain,
            store,
            PatchRequestParser(PatchInputLimits()),
            runtime,
        )
        with pytest.raises(PatchProtocolError):
            await adapter.execute(request, approve=object())
        with pytest.raises(PatchProtocolError):
            await adapter._reserve(object())
        pending = await adapter.execute(request, approve=True)
        assert pending.pending is not None
        terminal = await adapter.resume(request)
        assert terminal.completed
        replay = await adapter.execute(request)
        assert replay.completed

        flow_backend = InMemoryDurablePatchBackend()
        flow_store = InMemoryDurablePatchStore(flow_backend)
        flow_runtime = _Runtime(flow_backend, domain.owner)
        flow_orchestration = PatchProtocolOrchestrationAdapter(
            _profile(PatchProtocolSurface.FLOW),
            identity,
            flow_store,
            PatchRequestParser(PatchInputLimits()),
            flow_runtime,
        )
        with pytest.raises(PatchProtocolError):
            await flow_orchestration.advance(request, approve=object())
        with pytest.raises(PatchProtocolError):
            await flow_orchestration._reserve(object())
        flow = PatchProtocolFlowAdapter(flow_orchestration)
        flow_pending = await flow.execute(request, approve=True)
        assert isinstance(flow_pending, PatchProtocolFlowSuspension)
        flow_terminal = await flow.resume(request)
        assert isinstance(flow_terminal, PatchResult)
        assert isinstance(await flow.execute(request), PatchResult)

    run(scenario())


def test_patch_phase_14_provider_helpers_reject_corrupt_projection() -> None:
    """Conserve canonical paths and deny malformed provider projections."""
    parser = PatchRequestParser(PatchInputLimits())
    correlation = PatchObserverCorrelationId("correlation_" + "c" * 32)
    with pytest.raises(PatchProtocolError):
        _canonical_json_request(
            object(), OperationType.APPLY, b"{}", correlation
        )
    with pytest.raises(PatchProtocolError):
        _canonical_paths(object())
    with pytest.raises(PatchProtocolError):
        _apply_json_arguments(object())
    with pytest.raises(PatchProtocolError):
        PatchProtocolProviderAdapter(
            object(), object(), object(), object(), object(), object()
        )

    document = PatchDocumentSyntax(
        (
            UpdateDeclarationSyntax(
                LogicalPath("before.txt"), LogicalPath("after.txt"), ()
            ),
        ),
        b"*** Begin Patch v1\n*** End Patch\n",
    )
    canonical_bytes = document.canonical_bytes
    canonical = CanonicalPatchRequest(
        OperationType.APPLY,
        document,
        canonical_bytes,
        AlgorithmDigest.from_bytes(canonical_bytes),
    )
    assert _canonical_paths(canonical) == frozenset(
        (
            LogicalPath("before.txt"),
            LogicalPath("after.txt"),
        )
    )
    assert _apply_json_arguments(canonical).startswith(b'{"patch":"')

    empty_document = PatchDocumentSyntax((), b"")
    empty = CanonicalPatchRequest(
        OperationType.APPLY,
        empty_document,
        b"",
        AlgorithmDigest.from_bytes(b""),
    )
    with pytest.raises(PatchProtocolError):
        _canonical_paths(empty)
    corrupted = CanonicalPatchRequest(
        OperationType.APPLY,
        object(),
        b"",
        AlgorithmDigest.from_bytes(b""),
    )
    with pytest.raises(PatchProtocolError):
        _canonical_paths(corrupted)
    with pytest.raises(PatchProtocolError):
        _apply_json_arguments(corrupted)

    ingress = RawPatchIngress(
        RawProviderProfile("phase14-provider-helper"),
        RawToolCallId(correlation.value),
        RawPatchInputKind.APPLY_JSON,
        RawPatchInputState.COMPLETE,
        b'{"patch":"*** Begin Patch v1\\n*** Add File: note.txt\\n'
        b'+after\\n*** End Patch\\n"}',
    )
    parsed = parser.parse(ingress)
    assert parsed.operation is OperationType.APPLY
    assert (
        _canonical_json_request(
            parser,
            OperationType.APPLY,
            ingress.raw_bytes,
            correlation,
        ).digest
        == parsed.digest
    )


def test_patch_phase_14_provider_dispatch_fails_on_codec_witness_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a provider codec that projects incompatible canonical truth."""

    async def scenario() -> None:
        """Drive freeform parsing and dispatch with real durable identity."""
        workspace = PatchWorkspaceId("workspace_" + "d" * 32)
        identity = _identity("d", workspace)
        backend = InMemoryDurablePatchBackend()
        store = InMemoryDurablePatchStore(backend)
        domain = PatchProtocolCoordinationDomain.for_workspace(workspace)
        runtime = _Runtime(backend, domain.owner)
        parser = PatchRequestParser(PatchInputLimits())
        provider_profile = RawProviderProfile("phase14-codec-drift")
        adapter = PatchProtocolProviderAdapter(
            _profile(PatchProtocolSurface.PROVIDER_FREEFORM),
            identity,
            provider_profile,
            store,
            parser,
            runtime,
        )
        key = RetransmissionKey("phase14-codec-drift")
        call_id = RawToolCallId("phase14-codec-drift-call")
        correlation = adapter.correlation_for(call_id, key)
        document = (
            b"*** Begin Patch v1\n*** Add File: note.txt\n+after\n"
            b"*** End Patch\n"
        )
        call = PatchProtocolProviderCall(
            provider_profile,
            call_id,
            correlation,
            key,
            "grammar-v1",
            (document,),
            True,
        )
        edit = parser.parse(
            RawPatchIngress(
                provider_profile,
                call_id,
                RawPatchInputKind.EDIT_JSON,
                RawPatchInputState.COMPLETE,
                b'{"path":"note.txt","edits":['
                b'{"old_text":"before","new_text":"after"}]}',
            )
        )
        with monkeypatch.context() as patched:
            patched.setattr(
                PatchRequestParser,
                "parse",
                lambda self, ingress: edit,
            )
            with pytest.raises(PatchProtocolError):
                await adapter.apply_freeform(call)

        with monkeypatch.context() as patched:
            patched.setattr(
                PatchProtocolProviderAdapter,
                "_current_json",
                lambda self, *args: edit,
            )
            with pytest.raises(PatchProtocolError):
                await adapter.apply_freeform(call)

        canonical = parser.parse(
            RawPatchIngress(
                provider_profile,
                call_id,
                RawPatchInputKind.VERIFIED_FREEFORM,
                RawPatchInputState.COMPLETE,
                document,
            )
        )
        raw_arguments = _apply_json_arguments(canonical)
        base = await adapter._protocol.reserve(
            store,
            OperationType.APPLY,
            raw_arguments,
            key,
            correlation,
            parser,
        )
        mismatched = replace(
            base,
            durable=DurableReservation(
                base.request_id,
                base.durable.identity,
                AlgorithmDigest("sha256", "e" * 64),
                False,
            ),
        )

        async def reserve_mismatched(
            self: PatchProtocols, *_: object, **__: object
        ) -> PatchProtocolReservation:
            """Model a store response whose digest disagrees with parsing."""
            del self
            return mismatched

        with monkeypatch.context() as patched:
            patched.setattr(PatchProtocols, "reserve", reserve_mismatched)
            with pytest.raises(PatchProtocolError):
                await adapter._dispatch(
                    canonical, raw_arguments, key, correlation
                )

    run(scenario())


def test_patch_phase_14_retention_audience_expiry_and_validation() -> None:
    """Deny retention reads outside exact kind, audience, and expiry truth."""

    class _Validator:
        """Accept the test ciphertext after durable reader authorization."""

        async def validate(
            self,
            request_id: PatchRequestId,
            record: DurableRetentionRecord,
        ) -> None:
            """Validate the exact retained record without plaintext access."""
            del request_id, record

    async def scenario() -> None:
        """Use one in-memory store for absence, expiry, and audience denial."""
        workspace = PatchWorkspaceId("workspace_" + "f" * 32)
        identity = _identity("f", workspace).durable_identity(
            RetransmissionKey("phase14-retention")
        )
        backend = InMemoryDurablePatchBackend(
            retention_authorizer=StaticDurableRetentionAuthorizer(
                frozenset((Audience.AUDIT,))
            ),
            retention_validator=_Validator(),
        )
        store = InMemoryDurablePatchStore(backend)
        reservation = await store.reserve(identity, _digest("f"))
        access = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, identity)
        )
        retained = DurableRetentionRecord(
            PatchRetentionRecordId("retained_" + "a" * 16),
            DurableRetentionKind.AUDIT_PROJECTION,
            PatchRetentionKeyId("retention_" + "a" * 16),
            EncryptedRetentionValue(b"phase14-retention"),
            DurableRetentionPolicy(ExpiryTick(10), False),
        )
        with pytest.raises(DurableStoreError) as absent:
            await store.get_retention_for_audience(
                access,
                retained.retention_id,
                retained.kind,
                Audience.AUDIT,
                ExpiryTick(1),
            )
        assert absent.value.code is DurableStoreErrorCode.RETENTION_DENIED
        await store.put_retention(reservation, retained)
        with pytest.raises(DurableStoreError) as wrong_kind:
            await store.get_retention_for_audience(
                access,
                retained.retention_id,
                DurableRetentionKind.METRICS_PROJECTION,
                Audience.AUDIT,
                ExpiryTick(1),
            )
        assert wrong_kind.value.code is DurableStoreErrorCode.RETENTION_DENIED

        expired = DurableRetentionRecord(
            PatchRetentionRecordId("retained_" + "b" * 16),
            DurableRetentionKind.AUDIT_PROJECTION,
            PatchRetentionKeyId("retention_" + "b" * 16),
            EncryptedRetentionValue(b"phase14-expired"),
            DurableRetentionPolicy(ExpiryTick(2), False),
        )
        await store.put_retention(reservation, expired)
        with pytest.raises(DurableStoreError) as expiry:
            await store.get_retention_for_audience(
                access,
                expired.retention_id,
                expired.kind,
                Audience.AUDIT,
                ExpiryTick(2),
            )
        assert expiry.value.code is DurableStoreErrorCode.RETENTION_DENIED

        backend.retention_authorizer = StaticDurableRetentionAuthorizer(
            frozenset((Audience.PUBLIC,))
        )
        with pytest.raises(DurableStoreError) as audience:
            await store.get_retention_for_audience(
                access,
                retained.retention_id,
                retained.kind,
                Audience.AUDIT,
                ExpiryTick(1),
            )
        assert audience.value.code is DurableStoreErrorCode.RETENTION_DENIED
        backend.retention_authorizer = StaticDurableRetentionAuthorizer(
            frozenset((Audience.AUDIT,))
        )
        assert (
            await store.get_retention_for_audience(
                access,
                retained.retention_id,
                retained.kind,
                Audience.AUDIT,
                ExpiryTick(1),
            )
            == retained
        )

    run(scenario())
