"""Exercise the fail-closed Phase 14 protocol activation foundation."""

from asyncio import run
from dataclasses import replace

import pytest

from avalan.patch.coordinator import RetransmissionKey
from avalan.patch.domain import (
    ArtifactState,
    CommitTruth,
    DurationTicks,
    LifecyclePhase,
    LineageState,
    MutationState,
    OperationType,
    PatchContextId,
    PatchEventId,
    PatchExecutionId,
    PatchObserverCorrelationId,
    PatchPending,
    PatchPendingOperationId,
    PatchPlanId,
    PatchResult,
    PatchStatus,
    PatchWorkspaceId,
    PostconditionState,
    RequestedEffectOccurrence,
    SequenceNumber,
    WorkspaceChange,
)
from avalan.patch.durable_store import (
    DurableOutboxRecord,
    DurablePendingRecord,
    DurableRequestAccess,
    DurableTerminalRecord,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.parser import PatchInputLimits, PatchRequestParser
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
    PatchProtocolError,
    PatchProtocolIdentity,
    PatchProtocolProfile,
    PatchProtocolReservation,
    PatchProtocols,
    PatchProtocolSurface,
    PatchProviderCodecChecklist,
)


def _complete_protocol() -> PatchProtocolChecklist:
    """Return an exact complete protocol checklist for test-only use."""
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


def _complete_orchestration() -> PatchOrchestrationChecklist:
    """Return an exact complete orchestration checklist for test-only use."""
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


def _complete_codec() -> PatchProviderCodecChecklist:
    """Return an exact complete provider-codec checklist for test-only use."""
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


def _identity(suffix: str = "one") -> PatchProtocolIdentity:
    """Return one complete authenticated protocol identity."""
    return PatchProtocolIdentity(
        tenant=PatchTenantId(f"tenant-phase14-{suffix}"),
        principal=PatchPrincipalId(f"principal-phase14-{suffix}"),
        execution=PatchExecutionId.new(),
        run=PatchRunId(f"run-phase14-{suffix}"),
        session=PatchSessionId(f"session-phase14-{suffix}"),
        task=PatchTaskId(f"task-phase14-{suffix}"),
        agent=PatchAgentId(f"agent-phase14-{suffix}"),
        route=PolicyRouteId(f"route-phase14-{suffix}"),
        context=PatchContextId.new(),
        workspace=PatchWorkspaceId.new(),
    )


def _profile(surface: PatchProtocolSurface) -> PatchProtocolProfile:
    """Return a complete exact test-only profile for one surface."""
    return PatchProtocolProfile(
        surface=surface,
        enabled=True,
        authenticated=True,
        loopback_only=True,
        protocol=_complete_protocol(),
        orchestration=_complete_orchestration(),
        provider_codec=_complete_codec(),
    )


def _edit() -> bytes:
    """Return one closed portable JSON edit request."""
    return (
        b'{"path":"note.txt","edits":['
        b'{"old_text":"before","new_text":"after"}]}'
    )


def _apply() -> bytes:
    """Return one closed portable JSON apply request."""
    return (
        b'{"patch":"*** Begin Patch v1\\n*** Update File: note.txt\\n@@\\n'
        b'-before\\n+after\\n*** End Patch"}'
    )


def test_patch_phase_14_requirements() -> None:
    """Gate every protocol surface on an exact complete test-only profile."""
    identity = _identity()
    advertised = {
        surface: PatchProtocols(_profile(surface), identity).advertised_tools()
        for surface in PatchProtocolSurface
    }
    assert advertised == {
        PatchProtocolSurface.MCP: ("patch.edit", "patch.apply"),
        PatchProtocolSurface.A2A: ("patch.edit", "patch.apply"),
        PatchProtocolSurface.FLOW: (),
        PatchProtocolSurface.TASK: (),
        PatchProtocolSurface.MULTI_AGENT: (),
        PatchProtocolSurface.PROVIDER_FREEFORM: ("patch.apply",),
        PatchProtocolSurface.PROVIDER_NATIVE: ("patch.apply",),
    }
    profile = _profile(PatchProtocolSurface.MCP)
    for incomplete in (
        replace(profile, enabled=False),
        replace(profile, authenticated=False),
        replace(profile, loopback_only=False),
        replace(profile, name="production"),
        replace(profile, protocol=PatchProtocolChecklist()),
    ):
        assert PatchProtocols(incomplete, identity).advertised_tools() == ()
    for surface in (
        PatchProtocolSurface.FLOW,
        PatchProtocolSurface.TASK,
        PatchProtocolSurface.MULTI_AGENT,
    ):
        incomplete = replace(
            _profile(surface),
            orchestration=PatchOrchestrationChecklist(),
        )
        assert not PatchProtocols(incomplete, identity).active
    for surface in (
        PatchProtocolSurface.PROVIDER_FREEFORM,
        PatchProtocolSurface.PROVIDER_NATIVE,
    ):
        incomplete = replace(
            _profile(surface),
            provider_codec=PatchProviderCodecChecklist(),
        )
        assert not PatchProtocols(incomplete, identity).active


def test_patch_protocol_reserves_before_planning_and_replays_once() -> None:
    """Reserve the durable key and digest before the sole planning callback."""

    async def scenario() -> tuple[bool, str, tuple[str, ...]]:
        profile = _profile(PatchProtocolSurface.MCP)
        protocols = PatchProtocols(profile, _identity())
        store = InMemoryDurablePatchStore(InMemoryDurablePatchBackend())
        order: list[str] = []

        async def planner(value: PatchProtocolReservation) -> str:
            assert value.request_id.value.startswith("request_")
            order.append("planned")
            return "sealed-plan"

        first, first_plan = await protocols.reserve_before_planning(
            store,
            OperationType.EDIT,
            _edit(),
            RetransmissionKey("phase14-reserve-key"),
            PatchObserverCorrelationId.new(),
            PatchRequestParser(PatchInputLimits()),
            planner,
        )
        assert first_plan is not None
        order.insert(0, "reserved")
        replay, replay_plan = await protocols.reserve_before_planning(
            store,
            OperationType.EDIT,
            _edit(),
            RetransmissionKey("phase14-reserve-key"),
            PatchObserverCorrelationId.new(),
            PatchRequestParser(PatchInputLimits()),
            planner,
        )
        return (
            replay.durable.replayed,
            first_plan + ":" + str(replay_plan),
            tuple(order),
        )

    replayed, plans, order = run(scenario())
    assert replayed
    assert plans == "sealed-plan:None"
    assert order == ("reserved", "planned")


def test_patch_protocol_continuations_are_typed_and_identity_bound() -> None:
    """Return approval state without treating it as pending or completed."""

    async def scenario() -> PatchProtocolContinuation:
        protocols = PatchProtocols(
            _profile(PatchProtocolSurface.A2A), _identity()
        )
        store = InMemoryDurablePatchStore(InMemoryDurablePatchBackend())

        async def planner(_: PatchProtocolReservation) -> None:
            return None

        reservation, value = await protocols.reserve_before_planning(
            store,
            OperationType.EDIT,
            _edit(),
            RetransmissionKey("phase14-continuation-key"),
            PatchObserverCorrelationId.new(),
            PatchRequestParser(PatchInputLimits()),
            planner,
        )
        assert value is None
        return await protocols.inspect(store, reservation)

    continuation = run(scenario())
    assert continuation.kind is PatchProtocolContinuationKind.APPROVAL_REQUIRED
    assert continuation.pending is None
    assert continuation.result is None
    assert not continuation.completed
    with pytest.raises(PatchProtocolError):
        PatchProtocolContinuation(
            PatchProtocolContinuationKind.SETTLEMENT_PENDING,
            continuation.reservation,
        )


def test_patch_protocol_rejects_unadvertised_and_cross_agent_reads() -> None:
    """Fail closed before parsing and deny another agent's opaque handle."""

    async def scenario() -> None:
        owner_identity = _identity("owner")
        owner = PatchProtocols(
            _profile(PatchProtocolSurface.MCP), owner_identity
        )
        store = InMemoryDurablePatchStore(InMemoryDurablePatchBackend())
        reservation = await owner.reserve(
            store,
            OperationType.EDIT,
            _edit(),
            RetransmissionKey("phase14-owner-key"),
            PatchObserverCorrelationId.new(),
            PatchRequestParser(PatchInputLimits()),
        )
        unadvertised = PatchProtocols(
            PatchProtocolProfile(surface=PatchProtocolSurface.MCP),
            owner_identity,
        )
        with pytest.raises(PatchProtocolError):
            await unadvertised.reserve(
                store,
                OperationType.EDIT,
                _edit(),
                RetransmissionKey("phase14-inert-key"),
                PatchObserverCorrelationId.new(),
                PatchRequestParser(PatchInputLimits()),
            )
        other = PatchProtocols(
            _profile(PatchProtocolSurface.MCP), _identity("other")
        )
        with pytest.raises(PatchProtocolError):
            await other.inspect(store, reservation)

    run(scenario())


def test_patch_protocol_checks_invalid_values_and_all_continuation_truth() -> (
    None
):
    """Reject malformed bindings and separate pending from terminal truth."""

    class _OtherKey(RetransmissionKey):
        """Represent an invalid subtype at the exact durable boundary."""

    class _OtherParser(PatchRequestParser):
        """Represent an invalid parser subtype at the exact boundary."""

    class _OtherPending(PatchPending):
        """Represent an invalid pending subtype at the exact boundary."""

    class _OtherProfile(PatchProtocolProfile):
        """Represent an invalid profile subtype at the exact boundary."""

    protocol = _complete_protocol()
    object.__setattr__(protocol, "canonical_input", "invalid")
    with pytest.raises(PatchProtocolError):
        replace(protocol)
    orchestration = _complete_orchestration()
    object.__setattr__(orchestration, "shared_coordinator", "invalid")
    with pytest.raises(PatchProtocolError):
        replace(orchestration)
    codec = _complete_codec()
    object.__setattr__(codec, "advertised", "invalid")
    with pytest.raises(PatchProtocolError):
        replace(codec)
    profile = _profile(PatchProtocolSurface.MCP)
    object.__setattr__(profile, "protocol", object())
    with pytest.raises(PatchProtocolError):
        replace(profile)
    identity = _identity()
    object.__setattr__(identity, "context", object())
    with pytest.raises(PatchProtocolError):
        replace(identity)
    with pytest.raises(PatchProtocolError):
        _identity().durable_identity(_OtherKey("phase14-subtype-key"))
    with pytest.raises(PatchProtocolError):
        PatchProtocols(
            _OtherProfile(surface=PatchProtocolSurface.MCP), _identity()
        )

    async def scenario() -> tuple[
        PatchProtocolContinuation,
        PatchProtocolContinuation,
    ]:
        active_identity = _identity("continuation")
        protocols = PatchProtocols(
            _profile(PatchProtocolSurface.A2A), active_identity
        )
        backend = InMemoryDurablePatchBackend()
        store = InMemoryDurablePatchStore(backend)
        reservation = await protocols.reserve(
            store,
            OperationType.EDIT,
            _edit(),
            RetransmissionKey("phase14-pending-key"),
            PatchObserverCorrelationId.new(),
            PatchRequestParser(PatchInputLimits()),
        )
        assert reservation.digest.algorithm == "sha256"
        invalid_reservation = reservation
        object.__setattr__(invalid_reservation, "operation", "invalid")
        with pytest.raises(PatchProtocolError):
            replace(invalid_reservation)
        with pytest.raises(PatchProtocolError):
            await protocols.reserve(
                store,
                OperationType.EDIT,
                _edit(),
                RetransmissionKey("phase14-invalid-parser-key"),
                PatchObserverCorrelationId.new(),
                _OtherParser(PatchInputLimits()),
            )
        with pytest.raises(PatchProtocolError):
            await protocols.reserve(
                store,
                OperationType.EDIT,
                b"not-json",
                RetransmissionKey("phase14-malformed-key"),
                PatchObserverCorrelationId.new(),
                PatchRequestParser(PatchInputLimits()),
            )
        applied = await protocols.reserve(
            store,
            OperationType.APPLY,
            _apply(),
            RetransmissionKey("phase14-apply-key"),
            PatchObserverCorrelationId.new(),
            PatchRequestParser(PatchInputLimits()),
        )
        assert applied.operation is OperationType.APPLY
        record = backend.records[reservation.durable.identity]
        pending_id = PatchPendingOperationId.new()
        record.lifecycle = LifecyclePhase.SETTLEMENT_PENDING
        record.pending = DurablePendingRecord(
            reservation.request_id,
            active_identity.execution,
            pending_id,
            reservation.correlation,
            SequenceNumber(1),
            SequenceNumber(1),
            False,
            DurationTicks(1),
        )
        pending = await protocols.inspect(store, reservation)
        assert pending.pending is not None
        invalid_pending = _OtherPending(
            pending.pending.schema_version,
            pending.pending.pending_operation_id,
            pending.pending.request_id,
            pending.pending.correlation_id,
            pending.pending.lifecycle,
        )
        with pytest.raises(PatchProtocolError):
            PatchProtocolContinuation(
                PatchProtocolContinuationKind.SETTLEMENT_PENDING,
                reservation,
                pending=invalid_pending,
            )
        record.pending = None
        record.lifecycle = LifecyclePhase.COMMIT_STARTED
        attached = await protocols.inspect(store, reservation)
        assert (
            attached.kind is PatchProtocolContinuationKind.SETTLEMENT_PENDING
        )
        assert attached.pending is not None
        assert attached.pending.request_id == reservation.request_id
        assert attached.pending.correlation_id == reservation.correlation
        result = PatchResult(
            schema_version=1,
            request_id=reservation.request_id,
            plan_id=PatchPlanId.new(),
            lifecycle=LifecyclePhase.REQUEST_COMPLETED,
            status=PatchStatus.COMMITTED,
            truth=CommitTruth(
                MutationState.COMMITTED,
                LineageState.COMMITTED,
                RequestedEffectOccurrence.TRUE,
                ArtifactState.CLEANED,
                WorkspaceChange.CHANGED,
                True,
                PostconditionState.ESTABLISHED,
            ),
            diagnostic=None,
        )
        record.lifecycle = LifecyclePhase.REQUEST_COMPLETED
        record.terminal = DurableTerminalRecord(
            result,
            DurableOutboxRecord(
                PatchEventId.new(),
                reservation.request_id,
                SequenceNumber(1),
                LifecyclePhase.REQUEST_COMPLETED,
                reservation.correlation,
            ),
            pending_id,
        )
        terminal = await protocols.inspect(store, reservation)
        other = await protocols.reserve(
            store,
            OperationType.EDIT,
            _edit(),
            RetransmissionKey("phase14-other-key"),
            PatchObserverCorrelationId.new(),
            PatchRequestParser(PatchInputLimits()),
        )
        snapshot = await store.inspect(
            DurableRequestAccess(
                reservation.request_id,
                reservation.durable.identity,
            )
        )
        with pytest.raises(PatchProtocolError):
            protocols._continuation(other, snapshot)
        object.__setattr__(store, "inspect", None)
        with pytest.raises(PatchProtocolError):
            await protocols.inspect(store, reservation)
        return pending, terminal

    pending, terminal = run(scenario())
    assert pending.kind is PatchProtocolContinuationKind.SETTLEMENT_PENDING
    assert pending.pending is not None
    assert not pending.completed
    assert terminal.kind is PatchProtocolContinuationKind.TERMINAL
    assert terminal.result is not None
    assert terminal.completed
