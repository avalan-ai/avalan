"""Exercise the attached-lifetime scripted patch commit coordinator."""

from asyncio import Event, create_task, gather, run, sleep
from dataclasses import replace
from json import dumps

import pytest

import avalan.patch.coordinator as coordinator_module
from avalan.patch.coordinator import (
    ArtifactJournal,
    CommitLease,
    CoordinatorBoundary,
    CoordinatorError,
    CoordinatorErrorCode,
    InMemoryCoordinatorStore,
    InMemoryLeaseManager,
    InMemoryPatchCoordinator,
    JournalStep,
    LockFootprint,
    Reservation,
    RetransmissionKey,
    RevalidationFact,
    RevalidationField,
    RevalidationResult,
    RevalidationSnapshot,
    RuntimeIdentity,
    RuntimeResources,
    ScriptedCommitWorker,
    ScriptedFaultController,
    ScriptedReconciler,
    SealedCommitCommand,
    SettlementJournal,
    WorkerReport,
    WorkerState,
    footprint_for,
)
from avalan.patch.domain import (
    AlgorithmDigest,
    ApprovalMode,
    ArtifactState,
    ByteSize,
    Capability,
    CommitStepState,
    ContextKind,
    DurationTicks,
    ExpiryTick,
    FileMode,
    LifecyclePhase,
    LogicalPath,
    MetadataProfile,
    MutationState,
    OperationType,
    PatchApprovalId,
    PatchContextId,
    PatchDomainId,
    PatchErrorCode,
    PatchExecutionId,
    PatchGrantId,
    PatchInput,
    PatchLimits,
    PatchLineageId,
    PatchPlanId,
    PatchProtocolId,
    PatchRequest,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PatchStepId,
    PatchTargetId,
    PatchValidationError,
    PatchWorkspaceId,
    PostconditionState,
    SourceBytes,
)
from avalan.patch.parser import (
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderProfile,
    RawToolCallId,
)
from avalan.patch.planner import (
    LogicalText,
    PlannerFile,
    PlannerWorkspace,
)
from avalan.patch.planner import (
    plan as plan_candidate,
)
from avalan.patch.policy import (
    ApprovalClock,
    ApprovalDecisionState,
    ApprovalRequirements,
    ApprovalService,
    BrokerDecision,
    CapabilityMode,
    ExecutionSubject,
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PlanBinding,
    PlanBoundGrant,
    PlanReviewRequest,
    PolicyAuthorizer,
    PolicyBrokerId,
    PolicyPathSelector,
    PolicyReviewerRole,
    PolicyRevision,
    PolicyRouteId,
    PolicyRule,
    PreauthorizationClass,
    PreflightRequest,
    ReviewerDecision,
    RuntimeGrantStore,
    SealedPlan,
    TrustedPatchPolicy,
    compose_limits,
    seal_plan,
)
from avalan.patch.target import (
    PrimitiveProbe,
    ProbeState,
    TargetHandshake,
    TargetIdentity,
    TargetPrimitive,
)


def _limits() -> PatchLimits:
    """Return finite limits for a self-contained sealed-plan fixture."""
    return PatchLimits(
        ByteSize(10_000),
        ByteSize(20),
        ByteSize(512),
        ByteSize(20),
        ByteSize(20),
        ByteSize(10_000),
        ByteSize(10_000),
        ByteSize(10_000),
        DurationTicks(100),
        DurationTicks(100),
        DurationTicks(100),
    )


def _subject() -> ExecutionSubject:
    """Return fixed identities for one sealed coordinator plan."""
    return ExecutionSubject(
        PatchPrincipalId("principal-a"),
        PatchTenantId("tenant-a"),
        PatchRunId("run-a"),
        PatchSessionId("session-a"),
        PatchTaskId("task-a"),
        PatchAgentId("agent-a"),
    )


def _target(
    *,
    suffix: str = "a",
    filesystem_id: str = "filesystem-a",
    persistent_lease_id: str = "persistent-lease-a",
) -> TargetIdentity:
    """Return a no-handle target identity for the scripted phase only."""
    return TargetIdentity(
        PatchContextId("context_" + suffix * 16),
        PatchWorkspaceId("workspace_" + "a" * 16),
        PatchDomainId("domain_" + "a" * 16),
        PatchTargetId("target_" + "a" * 16),
        PatchProtocolId("protocol_" + "a" * 16),
        filesystem_id,
        "mount-a",
        "policy-six",
        persistent_lease_id,
        PatchApprovalId("approval_" + "a" * 16),
    )


def _requirements() -> ApprovalRequirements:
    """Return a one-reviewer approval requirement for a sealed fixture."""
    return ApprovalRequirements(
        ApprovalMode.REQUIRE_REVIEW,
        PolicyRouteId("route-six"),
        PolicyBrokerId("broker-six"),
        PolicyReviewerRole("reviewer-six"),
        1,
    )


def _policy() -> TrustedPatchPolicy:
    """Return bounded update and read authority for virtual planning only."""
    reader = PreauthorizationClass("reader-six")
    return TrustedPatchPolicy(
        PolicyRevision("policy-six"),
        frozenset((OperationType.EDIT,)),
        (
            PolicyRule(
                PolicyPathSelector(None),
                (
                    CapabilityMode(
                        Capability.UPDATE, ApprovalMode.REQUIRE_REVIEW
                    ),
                    CapabilityMode(
                        Capability.READ_FOR_MUTATION,
                        ApprovalMode.PREAUTHORIZED,
                        reader,
                    ),
                    CapabilityMode(
                        Capability.OBSERVE_MUTATION_PRECONDITIONS,
                        ApprovalMode.PREAUTHORIZED,
                        reader,
                    ),
                ),
            ),
        ),
        _limits(),
        _requirements(),
    )


def _handshake(target: TargetIdentity) -> TargetHandshake:
    """Return a frozen capability witness without a writable target."""
    available = (
        TargetPrimitive.ROOTED_CONTAINMENT,
        TargetPrimitive.NOFOLLOW_INSPECTION,
        TargetPrimitive.REGULAR_FILE_IDENTITY,
        TargetPrimitive.BOUNDED_READ,
    )
    future = (
        TargetPrimitive.METADATA_PRESERVATION,
        TargetPrimitive.BOUNDED_WRITE,
        TargetPrimitive.REPLACE_PUBLICATION,
        TargetPrimitive.NOREPLACE_CREATE_MOVE,
        TargetPrimitive.DIRECTORY_ENTRY_DELETE,
        TargetPrimitive.SAME_FILESYSTEM_MOVE,
        TargetPrimitive.STAGING,
        TargetPrimitive.STRUCTURAL_VERIFICATION,
    )
    return TargetHandshake(
        target,
        frozenset(available),
        (),
        tuple(PrimitiveProbe(item, ProbeState.AVAILABLE) for item in future),
    )


async def _sealed_plan(
    *,
    target: TargetIdentity | None = None,
    context_kind: ContextKind = ContextKind.LOCAL,
    step_count: int = 1,
) -> SealedPlan:
    """Run Phase 5 planning and sealing with no commit target attached."""
    assert step_count > 0
    target_value = target or _target()
    paths = tuple(
        LogicalPath("note" + str(index) + ".txt")
        for index in range(step_count)
    )
    document = "\n".join(
        (
            "*** Begin Patch v1",
            *(
                item
                for path in paths
                for item in (
                    "*** Update File: " + path.value,
                    "@@",
                    "-before",
                    "+after",
                )
            ),
            "*** End Patch",
            "",
        )
    )
    request = PatchRequestParser(PatchInputLimits()).parse(
        RawPatchIngress(
            RawProviderProfile("phase-six"),
            RawToolCallId("phase-six"),
            RawPatchInputKind.APPLY_JSON,
            RawPatchInputState.COMPLETE,
            dumps({"patch": document}, separators=(",", ":")).encode(),
        )
    )
    text = LogicalText.from_bytes(b"before\n")
    candidate = plan_candidate(
        request,
        PlannerWorkspace(
            (
                *(
                    PlannerFile(
                        path,
                        SourceBytes(b"before\n"),
                        MetadataProfile(FileMode(0o644), text.has_bom, "lf"),
                        None,
                        "mount-a",
                        "identity-note-" + str(index),
                    )
                    for index, path in enumerate(paths, start=1)
                ),
            ),
            frozenset(),
        ),
    )
    authorizer = PolicyAuthorizer(_policy())
    limits = _limits()
    preflight = await authorizer.authorize_preinspection(
        PreflightRequest(
            OperationType.EDIT,
            paths,
            frozenset((Capability.UPDATE,)),
            frozenset(paths),
            compose_limits(limits, limits, limits, limits, limits),
        )
    )
    final = await authorizer.authorize_final(
        preflight, candidate, _handshake(target_value)
    )
    return seal_plan(
        PatchPlanId("plan_" + "a" * 16),
        PlanBinding(
            PatchRequest(
                1,
                PatchRequestId("request_" + "a" * 16),
                PatchExecutionId("execution_" + "a" * 16),
                OperationType.EDIT,
                PatchInput(b"phase-six-request"),
                paths,
            ),
            candidate.request_digest,
            _subject(),
            context_kind,
            target_value,
            None,
            preflight,
            final,
        ),
        candidate,
        ExpiryTick(100),
    )


class _Clock(ApprovalClock):
    """Read one fixed approval tick through the typed async boundary."""

    def __init__(self, tick: int = 1) -> None:
        """Initialize the deterministic trusted approval tick."""
        self.tick = tick

    async def now(self) -> ExpiryTick:
        """Return the fixed nonexpired tick for the test grant."""
        return ExpiryTick(self.tick)


class _Broker:
    """Return one correct typed reviewer decision for the sealed plan."""

    async def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Approve the exact review through the policy broker protocol."""
        return BrokerDecision(
            request.requirements.broker,
            (
                ReviewerDecision(
                    PatchPrincipalId("reviewer-six"),
                    request.subject.tenant,
                    request.requirements.reviewer_role,
                    ApprovalDecisionState.APPROVED,
                ),
            ),
        )


class _BlockedReconciler(ScriptedReconciler):
    """Expose one controlled reconciliation wait for race assertions."""

    def __init__(self, current: RevalidationSnapshot) -> None:
        """Initialize a reconciler with an explicit settled report."""
        super().__init__(current)
        self.calls = 0
        self.started = Event()
        self.release = Event()
        self.report: WorkerReport | None = None

    def set_terminal_report(self, report: WorkerReport) -> None:
        """Set the report returned once the wait is released."""
        self.report = report

    async def reconcile(self, request_id: PatchRequestId) -> WorkerReport:
        """Wait once and return the configured terminal target evidence."""
        del request_id
        self.calls += 1
        self.started.set()
        await self.release.wait()
        assert self.report is not None
        return self.report


class _FaultingReconciler(ScriptedReconciler):
    """Allow a test to model one post-start reconciliation contact loss."""

    def __init__(self, current: RevalidationSnapshot) -> None:
        """Initialize a reconciler that starts in contact."""
        super().__init__(current)
        self.unavailable = False

    async def reconcile(self, request_id: PatchRequestId) -> WorkerReport:
        """Raise only while the deterministic contact-loss switch is set."""
        if self.unavailable:
            raise RuntimeError("scripted reconciliation loss")
        return await super().reconcile(request_id)


class _UnavailableLeaseManager(InMemoryLeaseManager):
    """Contain one deliberate release failure after settlement."""

    async def release(self, lease: CommitLease) -> None:
        """Reject the release without exposing any target handle."""
        del lease
        raise CoordinatorError(CoordinatorErrorCode.FENCED)


def _snapshot(
    changed: RevalidationField | None = None,
) -> RevalidationSnapshot:
    """Return a complete deterministic revalidation witness matrix."""
    facts = tuple(
        RevalidationFact(
            field,
            "key-" + field.value,
            (
                "changed-" + field.value
                if field is changed
                else "value-" + field.value
            ),
        )
        for field in RevalidationField
    )
    return RevalidationSnapshot(
        tuple(sorted(facts, key=lambda item: (item.field.value, item.key)))
    )


def _journal(
    plan: SealedPlan,
    *states: CommitStepState,
    artifact: ArtifactState = ArtifactState.CLEANED,
    postcondition: PostconditionState = PostconditionState.ESTABLISHED,
) -> SettlementJournal:
    """Return complete requested-effect and separate artifact truth records."""
    expected_steps = coordinator_module._sealed_journal_steps(plan)
    assert len(states) == len(expected_steps)
    return SettlementJournal(
        tuple(
            JournalStep(identifier, lineage, state)
            for (identifier, lineage), state in zip(
                expected_steps, states, strict=True
            )
        ),
        tuple(
            ArtifactJournal(identifier, artifact)
            for identifier in coordinator_module._sealed_artifact_identifiers(
                plan
            )
        ),
        postcondition,
    )


async def _approved_with_service(
    *,
    target: TargetIdentity | None = None,
    context_kind: ContextKind = ContextKind.LOCAL,
    step_count: int = 1,
) -> tuple[SealedPlan, PlanBoundGrant, ApprovalService]:
    """Produce an exact Phase 5-approved plan without target mutation."""
    plan = await _sealed_plan(
        target=target,
        context_kind=context_kind,
        step_count=step_count,
    )
    service = ApprovalService(
        _Broker(),
        _Clock(),
        RuntimeGrantStore(),
    )
    grant = await _issue_grant(plan, service)
    return plan, grant, service


async def _issue_grant(
    plan: SealedPlan, service: ApprovalService
) -> PlanBoundGrant:
    """Issue one exact in-store grant through the shared approval service."""
    decision = await service.await_review(
        PlanReviewRequest(
            plan,
            plan.binding.subject,
            plan.binding.final.approval,
        )
    )
    assert decision.grant is not None
    return decision.grant


async def _approved(
    *,
    target: TargetIdentity | None = None,
    context_kind: ContextKind = ContextKind.LOCAL,
    step_count: int = 1,
) -> tuple[SealedPlan, PlanBoundGrant]:
    """Return a plan and issued grant for non-commit fixture setup."""
    plan, grant, _ = await _approved_with_service(
        target=target,
        context_kind=context_kind,
        step_count=step_count,
    )
    return plan, grant


async def _coordinator(
    *,
    current: RevalidationSnapshot | None = None,
) -> tuple[
    InMemoryPatchCoordinator,
    ScriptedReconciler,
    RuntimeIdentity,
    AlgorithmDigest,
]:
    """Return isolated coordinator components for one deterministic flow."""
    plan, _ = await _approved()
    store = InMemoryCoordinatorStore()
    return (
        InMemoryPatchCoordinator(
            store,
            InMemoryLeaseManager(store),
            ScriptedReconciler(current or _snapshot()),
        ),
        ScriptedReconciler(current or _snapshot()),
        RuntimeIdentity(
            plan.binding.subject,
            plan.binding.final.approval.route,
            RetransmissionKey("phase-six-key"),
        ),
        plan.binding.request_digest,
    )


async def _runtime(
    *,
    key: str = "phase-six-key",
    current: RevalidationSnapshot | None = None,
    faults: ScriptedFaultController | None = None,
    reconciler: ScriptedReconciler | None = None,
    step_count: int = 1,
) -> tuple[
    InMemoryPatchCoordinator,
    ScriptedReconciler,
    RuntimeIdentity,
    AlgorithmDigest,
    SealedPlan,
    PlanBoundGrant,
]:
    """Build one coordinator and one separately sealed approved request."""
    plan, grant, approvals = await _approved_with_service(
        step_count=step_count
    )
    store = InMemoryCoordinatorStore(approvals)
    current_value = current or _snapshot()
    current_reconciler = reconciler or ScriptedReconciler(current_value)
    return (
        InMemoryPatchCoordinator(
            store,
            InMemoryLeaseManager(store),
            current_reconciler,
            faults,
        ),
        current_reconciler,
        RuntimeIdentity(
            plan.binding.subject,
            plan.binding.final.approval.route,
            RetransmissionKey(key),
        ),
        plan.binding.request_digest,
        plan,
        grant,
    )


async def _continue(
    coordinator: InMemoryPatchCoordinator,
    reservation: Reservation,
    plan: SealedPlan,
    grant: PlanBoundGrant,
    controller: str = "controller-a",
) -> PatchResult | object:
    """Continue only through the same attached coordinator invocation."""
    return await coordinator.execute(
        reservation,
        plan,
        grant,
        _snapshot(),
        ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
        controller,
    )


def test_patch_phase_6_lifecycle_and_scripted_commit_e2e() -> None:
    """Commit one fully revalidated multi-step scripted plan exactly once."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        coordinator, _, identity, digest, plan, grant = await _runtime(
            step_count=2
        )
        reservation = await coordinator.reserve(identity, digest)
        result = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(
                WorkerReport(
                    WorkerState.SETTLED,
                    _journal(
                        plan,
                        CommitStepState.COMMITTED,
                        CommitStepState.COMMITTED,
                    ),
                )
            ),
            "controller-a",
        )
        assert isinstance(result, PatchResult)
        assert result.status is PatchStatus.COMMITTED
        assert result.truth.mutation_state is MutationState.COMMITTED
        assert result.diagnostic is None
        assert result.lifecycle is LifecyclePhase.REQUEST_COMPLETED
        assert coordinator.resources.lease_depth == 0
        assert coordinator.scheduler_parallel_safe is False
        assert footprint_for(plan).keys[0] == "workspace"

    run(execute())


def test_patch_phase_6_idempotency_and_owner_races_are_linearizable() -> None:
    """Attach exact retries and reject conflicts before target work."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        coordinator, _, identity, digest, plan, grant = await _runtime()
        first = await coordinator.reserve(identity, digest)
        attached = await coordinator.reserve(identity, digest)
        assert attached.replayed and attached.request_id == first.request_id
        with pytest.raises(CoordinatorError) as conflict:
            await coordinator.reserve(
                identity,
                AlgorithmDigest.from_bytes(b"different-canonical-request"),
            )
        assert conflict.value.code is CoordinatorErrorCode.IDEMPOTENCY_CONFLICT
        started = Event()
        release = Event()
        worker = ScriptedCommitWorker(
            WorkerReport(
                WorkerState.SETTLED, _journal(plan, CommitStepState.COMMITTED)
            ),
            started,
            release,
        )
        first_task = create_task(
            coordinator.execute(
                first, plan, grant, _snapshot(), worker, "controller-a"
            )
        )
        await started.wait()
        retry_task = create_task(
            coordinator.execute(
                attached, plan, grant, _snapshot(), worker, "controller-a"
            )
        )
        await sleep(0)
        assert not retry_task.done()
        release.set()
        first_result, retry_result = await gather(first_task, retry_task)
        assert isinstance(first_result, PatchResult)
        assert retry_result == first_result
        assert len(worker.commands) == 1

    run(execute())


def test_patch_phase_6_revalidation_matrix_stales_before_effects() -> None:
    """Fail every changed fact before grant consumption or commands."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        for field in RevalidationField:
            (
                coordinator,
                reconciler,
                identity,
                digest,
                plan,
                grant,
            ) = await _runtime(
                key="phase-six-" + field.value,
            )
            reservation = await coordinator.reserve(identity, digest)
            worker = ScriptedCommitWorker(
                WorkerReport(
                    WorkerState.SETTLED,
                    _journal(plan, CommitStepState.COMMITTED),
                )
            )
            reconciler.replace_current(_snapshot(field))
            result = await coordinator.execute(
                reservation,
                plan,
                grant,
                _snapshot(),
                worker,
                "controller-a",
            )
            assert isinstance(result, PatchResult)
            assert result.status is PatchStatus.STALE
            assert result.truth.mutation_state is MutationState.NOT_COMMITTED
            assert worker.commands == []
            assert coordinator.resources.lease_depth == 0

    run(execute())


@pytest.mark.parametrize(
    ("states", "artifact", "postcondition", "status", "mutation"),
    (
        (
            (CommitStepState.NOT_COMMITTED,),
            ArtifactState.ABSENT,
            PostconditionState.UNKNOWN,
            PatchStatus.COMMIT_FAILED,
            MutationState.NOT_COMMITTED,
        ),
        (
            (CommitStepState.COMMITTED,),
            ArtifactState.CLEANED,
            PostconditionState.ESTABLISHED,
            PatchStatus.COMMITTED,
            MutationState.COMMITTED,
        ),
        (
            (CommitStepState.COMMITTED,),
            ArtifactState.STAGED,
            PostconditionState.ESTABLISHED,
            PatchStatus.COMMITTED,
            MutationState.COMMITTED,
        ),
        (
            (CommitStepState.COMMITTED, CommitStepState.NOT_COMMITTED),
            ArtifactState.LEAKED,
            PostconditionState.SUPERSEDED,
            PatchStatus.PARTIAL,
            MutationState.PARTIALLY_COMMITTED,
        ),
        (
            (CommitStepState.UNKNOWN,),
            ArtifactState.UNKNOWN,
            PostconditionState.UNKNOWN,
            PatchStatus.INDETERMINATE,
            MutationState.INDETERMINATE,
        ),
    ),
)
def test_patch_phase_6_journal_truth_never_guesses_effects(
    states: tuple[CommitStepState, ...],
    artifact: ArtifactState,
    postcondition: PostconditionState,
    status: PatchStatus,
    mutation: MutationState,
) -> None:
    """Derive every terminal outcome from exact steps, artifacts, and facts."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        coordinator, _, identity, digest, plan, grant = await _runtime(
            key="phase-six-truth-" + status.value,
            step_count=len(states),
        )
        result = await coordinator.execute(
            await coordinator.reserve(identity, digest),
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(
                WorkerReport(
                    WorkerState.SETTLED,
                    _journal(
                        plan,
                        *states,
                        artifact=artifact,
                        postcondition=postcondition,
                    ),
                )
            ),
            "controller-a",
        )
        assert isinstance(result, PatchResult)
        assert result.status is status
        assert result.truth.mutation_state is mutation
        assert result.truth.artifact_state is artifact
        assert result.truth.postcondition is postcondition

    run(execute())


@pytest.mark.parametrize(
    ("mutation", "status", "error"),
    (
        (
            MutationState.NOT_COMMITTED,
            PatchStatus.COMMIT_FAILED,
            PatchErrorCode.COMMIT_FAILED,
        ),
        (MutationState.COMMITTED, PatchStatus.COMMITTED, None),
        (
            MutationState.PARTIALLY_COMMITTED,
            PatchStatus.PARTIAL,
            PatchErrorCode.PARTIAL_COMMIT,
        ),
        (
            MutationState.INDETERMINATE,
            PatchStatus.INDETERMINATE,
            PatchErrorCode.INDETERMINATE,
        ),
    ),
)
def test_patch_phase_6_status_error_pairs_are_exhaustive(
    mutation: MutationState,
    status: PatchStatus,
    error: PatchErrorCode | None,
) -> None:
    """Map every mutation truth to one exact status and error absence."""
    assert coordinator_module._status(mutation) == (status, error)


@pytest.mark.parametrize(
    ("states", "expected_state", "exact"),
    (
        (
            (CommitStepState.NOT_COMMITTED,),
            MutationState.NOT_COMMITTED,
            True,
        ),
        (
            (CommitStepState.COMMITTED,),
            MutationState.COMMITTED,
            True,
        ),
        (
            (
                CommitStepState.COMMITTED,
                CommitStepState.NOT_COMMITTED,
            ),
            MutationState.PARTIALLY_COMMITTED,
            True,
        ),
        (
            (CommitStepState.UNKNOWN,),
            MutationState.INDETERMINATE,
            False,
        ),
        (
            (CommitStepState.COMMITTED, CommitStepState.UNKNOWN),
            MutationState.INDETERMINATE,
            False,
        ),
    ),
)
def test_patch_phase_6_complete_step_state_oracle(
    states: tuple[CommitStepState, ...],
    expected_state: MutationState,
    exact: bool,
) -> None:
    """Derive every finite step-state class without an artifact shortcut."""

    async def execute() -> None:
        coordinator, _, identity, digest, plan, grant = await _runtime(
            key="phase-six-state-" + expected_state.value,
            step_count=len(states),
        )
        reservation = await coordinator.reserve(identity, digest)
        result = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(
                WorkerReport(WorkerState.SETTLED, _journal(plan, *states))
            ),
            "controller-a",
        )
        assert isinstance(result, PatchResult)
        assert result.truth.mutation_state is expected_state
        assert result.truth.commit_set_exact is exact

    run(execute())


@pytest.mark.parametrize(
    ("approval_required", "expected"),
    (
        (
            False,
            (
                LifecyclePhase.PARSED,
                LifecyclePhase.SCOPE_BOUND,
                LifecyclePhase.PREFLIGHT_AUTHORIZED,
                LifecyclePhase.PLANNED,
                LifecyclePhase.APPROVED,
                LifecyclePhase.COMMIT_READY,
                LifecyclePhase.COMMIT_STARTED,
                LifecyclePhase.SETTLED,
                LifecyclePhase.REQUEST_COMPLETED,
            ),
        ),
        (
            True,
            (
                LifecyclePhase.PARSED,
                LifecyclePhase.SCOPE_BOUND,
                LifecyclePhase.PREFLIGHT_AUTHORIZED,
                LifecyclePhase.PLANNED,
                LifecyclePhase.APPROVAL_REQUIRED,
                LifecyclePhase.APPROVED,
                LifecyclePhase.COMMIT_READY,
                LifecyclePhase.COMMIT_STARTED,
                LifecyclePhase.SETTLED,
                LifecyclePhase.REQUEST_COMPLETED,
            ),
        ),
    ),
)
def test_patch_phase_6_lifecycle_events_are_total_and_ordered(
    approval_required: bool,
    expected: tuple[LifecyclePhase, ...],
) -> None:
    """Emit every legal lifecycle transition before one terminal event."""

    async def execute() -> None:
        coordinator, _, identity, digest, plan, grant = await _runtime(
            key="phase-six-lifecycle-" + str(approval_required)
        )
        reservation = await coordinator.reserve(identity, digest)
        prepared = await coordinator.prepare(
            reservation,
            plan,
            approval_required=approval_required,
        )
        if approval_required:
            assert prepared is LifecyclePhase.APPROVAL_REQUIRED
            assert (
                await coordinator.advance(reservation, LifecyclePhase.APPROVED)
                is LifecyclePhase.APPROVED
            )
        else:
            assert prepared is LifecyclePhase.APPROVED
        result = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(
                WorkerReport(
                    WorkerState.SETTLED,
                    _journal(plan, CommitStepState.COMMITTED),
                )
            ),
            "controller-a",
        )
        assert isinstance(result, PatchResult)
        events = await coordinator.events(reservation)
        assert tuple(event.event.lifecycle for event in events) == expected
        assert tuple(event.event.sequence.value for event in events) == tuple(
            range(1, len(expected) + 1)
        )

    run(execute())


def test_patch_phase_6_reserves_canonical_identity_before_planning() -> None:
    """Reserve the parsed canonical digest before an in-memory plan exists."""

    async def execute() -> None:
        plan, grant, approvals = await _approved_with_service()
        store = InMemoryCoordinatorStore(approvals)
        reconciler = ScriptedReconciler(_snapshot())
        coordinator = InMemoryPatchCoordinator(
            store,
            InMemoryLeaseManager(store),
            reconciler,
        )
        identity = RuntimeIdentity(
            _subject(),
            PolicyRouteId("route-six"),
            RetransmissionKey("phase-six-preplanning"),
        )
        reservation = await coordinator.reserve(
            identity, plan.binding.request_digest
        )
        assert reservation.digest == plan.binding.request_digest
        result = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(
                WorkerReport(
                    WorkerState.SETTLED,
                    _journal(plan, CommitStepState.COMMITTED),
                )
            ),
            "controller-a",
        )
        assert isinstance(result, PatchResult)
        with pytest.raises(CoordinatorError) as stale:
            await coordinator.prepare(
                reservation,
                replace(
                    plan,
                    binding=replace(
                        plan.binding,
                        request_digest=AlgorithmDigest.from_bytes(b"changed"),
                    ),
                ),
                approval_required=False,
            )
        assert stale.value.code is CoordinatorErrorCode.STALE

    run(execute())


def test_patch_phase_6_shared_backing_domain_serializes_context_aliases() -> (
    None
):
    """Serialize context and lease aliases by their shared backing domain."""

    async def execute() -> None:
        grants = RuntimeGrantStore()
        approvals = ApprovalService(_Broker(), _Clock(), grants)
        plan_one = await _sealed_plan()
        grant_one = await _issue_grant(plan_one, approvals)
        plan_two = await _sealed_plan(
            target=_target(
                suffix="b",
                filesystem_id="filesystem-b",
                persistent_lease_id="persistent-lease-b",
            ),
            context_kind=ContextKind.CONTAINER,
        )
        grant_two = await _issue_grant(plan_two, approvals)
        store = InMemoryCoordinatorStore(approvals)
        leases = InMemoryLeaseManager(store)
        first_reconciler = ScriptedReconciler(_snapshot())
        second_reconciler = ScriptedReconciler(_snapshot())
        first_coordinator = InMemoryPatchCoordinator(
            store,
            leases,
            first_reconciler,
        )
        second_coordinator = InMemoryPatchCoordinator(
            store,
            leases,
            second_reconciler,
        )
        first = await first_coordinator.reserve(
            RuntimeIdentity(
                _subject(),
                PolicyRouteId("route-six"),
                RetransmissionKey("context-local-filesystem-a-lease-a"),
            ),
            plan_one.binding.request_digest,
        )
        second = await second_coordinator.reserve(
            RuntimeIdentity(
                _subject(),
                PolicyRouteId("route-six"),
                RetransmissionKey("context-container-filesystem-b-lease-b"),
            ),
            plan_two.binding.request_digest,
        )
        assert (
            footprint_for(plan_one).domain_id
            == footprint_for(plan_two).domain_id
        )
        assert (
            plan_one.binding.target.context_id
            != plan_two.binding.target.context_id
        )
        assert (
            plan_one.binding.target.filesystem_id
            != plan_two.binding.target.filesystem_id
        )
        assert (
            plan_one.binding.target.persistent_lease_id
            != plan_two.binding.target.persistent_lease_id
        )
        started = Event()
        release = Event()
        first_worker = ScriptedCommitWorker(
            WorkerReport(
                WorkerState.SETTLED,
                _journal(plan_one, CommitStepState.COMMITTED),
            ),
            started,
            release,
        )
        first_task = create_task(
            first_coordinator.execute(
                first,
                plan_one,
                grant_one,
                _snapshot(),
                first_worker,
                "controller-one",
            )
        )
        await started.wait()
        assert first_coordinator.resources == RuntimeResources(
            0, 1, 1, 0, 1, 0
        )
        prepared = await second_coordinator.prepare(
            second,
            plan_two,
            approval_required=True,
        )
        assert prepared is LifecyclePhase.APPROVAL_REQUIRED
        assert (
            await second_coordinator.advance(second, LifecyclePhase.APPROVED)
            is LifecyclePhase.APPROVED
        )
        second_task = create_task(
            second_coordinator.execute(
                second,
                plan_two,
                grant_two,
                _snapshot(),
                ScriptedCommitWorker(
                    WorkerReport(
                        WorkerState.SETTLED,
                        _journal(plan_two, CommitStepState.COMMITTED),
                    )
                ),
                "controller-two",
            )
        )
        await sleep(0)
        assert not second_task.done()
        release.set()
        first_result, second_result = await gather(first_task, second_task)
        assert isinstance(first_result, PatchResult)
        assert isinstance(second_result, PatchResult)
        assert first_worker.commands[0].lease.fence == 1
        assert second_result.truth.mutation_state is MutationState.COMMITTED
        assert first_coordinator.resources == RuntimeResources(
            0, 0, 0, 0, 0, 0
        )

    run(execute())


@pytest.mark.parametrize(
    ("report", "terminal"),
    (
        (WorkerState.LIVE, False),
        (WorkerState.FENCED, True),
    ),
)
def test_patch_phase_6_contact_loss_never_retries_or_guesses(
    report: WorkerState,
    terminal: bool,
) -> None:
    """Keep unprovable workers pending and fence provable workers once."""

    async def execute() -> None:
        (
            coordinator,
            reconciler,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime(key="phase-six-contact-" + report.value)
        reservation = await coordinator.reserve(identity, digest)
        worker = ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None))
        pending = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            worker,
            "controller-a",
        )
        assert not isinstance(pending, PatchResult)
        if report is WorkerState.FENCED:
            reconciler.set_report(
                reservation.request_id,
                WorkerReport(
                    WorkerState.FENCED,
                    _journal(
                        plan,
                        CommitStepState.UNKNOWN,
                        artifact=ArtifactState.UNKNOWN,
                        postcondition=PostconditionState.UNKNOWN,
                    ),
                ),
            )
        settled = await _continue(coordinator, reservation, plan, grant)
        assert isinstance(settled, PatchResult) is terminal
        assert not isinstance(settled, PatchResult) is not terminal
        replay = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            worker,
            "controller-a",
        )
        assert replay == settled
        assert len(worker.commands) == 1

    run(execute())


@pytest.mark.parametrize(
    "transitions",
    (
        (LifecyclePhase.REQUEST_COMPLETED,),
        (LifecyclePhase.PARSED, LifecyclePhase.REQUEST_COMPLETED),
        (
            LifecyclePhase.PARSED,
            LifecyclePhase.SCOPE_BOUND,
            LifecyclePhase.REQUEST_COMPLETED,
        ),
        (
            LifecyclePhase.PARSED,
            LifecyclePhase.SCOPE_BOUND,
            LifecyclePhase.PREFLIGHT_AUTHORIZED,
            LifecyclePhase.REQUEST_COMPLETED,
        ),
        (
            LifecyclePhase.PARSED,
            LifecyclePhase.SCOPE_BOUND,
            LifecyclePhase.PREFLIGHT_AUTHORIZED,
            LifecyclePhase.PLANNED,
            LifecyclePhase.REQUEST_COMPLETED,
        ),
        (
            LifecyclePhase.PARSED,
            LifecyclePhase.SCOPE_BOUND,
            LifecyclePhase.PREFLIGHT_AUTHORIZED,
            LifecyclePhase.PLANNED,
            LifecyclePhase.APPROVAL_REQUIRED,
            LifecyclePhase.REQUEST_COMPLETED,
        ),
    ),
)
def test_patch_phase_6_all_precommit_terminal_paths_are_legal(
    transitions: tuple[LifecyclePhase, ...],
) -> None:
    """Exercise every finite precommit route to the sole terminal state."""

    async def execute() -> None:
        coordinator, _, identity, digest, _, _ = await _runtime(
            key="phase-six-path-"
            + "-".join(phase.value for phase in transitions)
        )
        reservation = await coordinator.reserve(identity, digest)
        for phase in transitions:
            assert await coordinator.advance(reservation, phase) is phase
        assert (
            tuple(
                event.event.lifecycle
                for event in await coordinator.events(reservation)
            )
            == transitions
        )

    run(execute())


@pytest.mark.parametrize(
    "second_keys",
    (
        (
            "workspace",
            "destination-parent",
            "destination.txt",
            "source-parent",
            "source.txt",
        ),
        (
            "workspace",
            "other-destination-parent",
            "other-destination.txt",
            "other-source-parent",
            "other-source.txt",
        ),
    ),
)
def test_patch_phase_6_workspace_lock_fences_overlapping_and_disjoint_paths(
    second_keys: tuple[str, ...],
) -> None:
    """Serialize every path footprint by domain and reject a retired fence."""

    async def execute() -> None:
        plan, _ = await _approved()
        domain = plan.binding.target.domain_id
        store = InMemoryCoordinatorStore()
        leases = InMemoryLeaseManager(store)
        first = await store.reserve(
            RuntimeIdentity(
                _subject(),
                PolicyRouteId("route-six"),
                RetransmissionKey("phase-six-first-footprint"),
            ),
            plan.binding.request_digest,
        )
        second = await store.reserve(
            RuntimeIdentity(
                _subject(),
                PolicyRouteId("route-six"),
                RetransmissionKey("phase-six-second-footprint"),
            ),
            AlgorithmDigest.from_bytes(b"phase-six-second-footprint"),
        )
        first_footprint = LockFootprint(
            domain,
            (
                "workspace",
                "destination-parent",
                "destination.txt",
                "source-parent",
                "source.txt",
            ),
        )
        second_footprint = LockFootprint(domain, second_keys)
        first_lease = await leases.acquire(first_footprint, first)
        next_lease = create_task(leases.acquire(second_footprint, second))
        await sleep(0)
        assert not next_lease.done()
        await leases.release(first_lease)
        second_lease = await next_lease
        assert second_lease.fence == first_lease.fence + 1
        assert not await leases.is_current(first_lease)
        assert await leases.is_current(second_lease)
        await leases.release(second_lease)

    run(execute())


@pytest.mark.parametrize(
    ("label", "changed", "states"),
    (
        (
            "stale",
            RevalidationField.CONTEXT,
            (CommitStepState.COMMITTED,),
        ),
        (
            "commit-failed",
            None,
            (CommitStepState.NOT_COMMITTED,),
        ),
        (
            "committed",
            None,
            (CommitStepState.COMMITTED,),
        ),
        (
            "partial",
            None,
            (
                CommitStepState.COMMITTED,
                CommitStepState.NOT_COMMITTED,
            ),
        ),
        (
            "indeterminate",
            None,
            (CommitStepState.UNKNOWN,),
        ),
        ("live-pending", None, ()),
    ),
)
def test_patch_phase_6_terminal_and_pending_states_never_raw_retry(
    label: str,
    changed: RevalidationField | None,
    states: tuple[CommitStepState, ...],
) -> None:
    """Attach replayed state without issuing another command."""

    async def execute() -> None:
        coordinator, _, identity, digest, plan, grant = await _runtime(
            key="phase-six-no-retry-" + label,
            current=_snapshot(changed),
            step_count=max(len(states), 1),
        )
        reservation = await coordinator.reserve(identity, digest)
        replayed = await coordinator.reserve(identity, digest)
        assert replayed.replayed
        first_worker = ScriptedCommitWorker(
            WorkerReport(
                WorkerState.LIVE if not states else WorkerState.SETTLED,
                None if not states else _journal(plan, *states),
            )
        )
        first = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            first_worker,
            "controller-a",
        )
        second_worker = ScriptedCommitWorker(
            WorkerReport(
                WorkerState.SETTLED,
                _journal(
                    plan,
                    *(CommitStepState.COMMITTED,)
                    * len(plan.candidate.lineages),
                ),
            )
        )
        replay = await coordinator.execute(
            replayed,
            plan,
            grant,
            _snapshot(),
            second_worker,
            "controller-a",
        )
        assert replay == first
        assert not second_worker.commands

    run(execute())


def test_patch_phase_6_inert_provider_history_never_reserves_a_request() -> (
    None
):
    """Treat patch-looking provider history as inert non-authority bytes."""

    async def execute() -> None:
        coordinator, _, identity, digest, plan, grant = await _runtime(
            key="phase-six-inert-history"
        )
        await coordinator.replay_inert_history(
            (b'{"name":"patch.apply","arguments":{"path":"note.txt"}}',)
        )
        reservation = await coordinator.reserve(identity, digest)
        assert reservation.replayed is False
        result = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(
                WorkerReport(
                    WorkerState.SETTLED,
                    _journal(plan, CommitStepState.COMMITTED),
                )
            ),
            "controller-a",
        )
        assert isinstance(result, PatchResult)

    run(execute())


def test_patch_phase_6_await_depth_matrix_has_no_db_or_target_leak() -> None:
    """Record every commit await boundary with only its fenced lease held."""

    async def execute() -> None:
        faults = ScriptedFaultController()
        (
            coordinator,
            _,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime(
            key="phase-six-await-matrix",
            faults=faults,
        )
        reservation = await coordinator.reserve(identity, digest)
        started = Event()
        release = Event()
        task = create_task(
            coordinator.execute(
                reservation,
                plan,
                grant,
                _snapshot(),
                ScriptedCommitWorker(
                    WorkerReport(
                        WorkerState.SETTLED,
                        _journal(plan, CommitStepState.COMMITTED),
                    ),
                    started,
                    release,
                ),
                "controller-a",
            )
        )
        await started.wait()
        assert coordinator.resources == RuntimeResources(0, 1, 1, 0, 1, 0)
        release.set()
        result = await task
        assert isinstance(result, PatchResult)
        assert faults.observed == tuple(CoordinatorBoundary)
        assert len(faults.depths) == len(faults.observed)
        for resources in faults.depths:
            assert resources.transaction_depth == 0
            assert resources.target_handle_depth == 0
            assert resources.approval_depth == 0
            assert resources.lease_depth == 1
        assert coordinator.resources == RuntimeResources(0, 0, 0, 0, 0, 0)

    run(execute())


@pytest.mark.parametrize(
    "boundary",
    tuple(CoordinatorBoundary)[3:],
)
def test_patch_phase_6_cancel_after_every_commit_boundary_stays_owned(
    boundary: CoordinatorBoundary,
) -> None:
    """Record postcommit cancellation without a retry or detached worker."""

    async def execute() -> None:
        faults = ScriptedFaultController(frozenset((boundary,)))
        (
            coordinator,
            reconciler,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime(
            key="phase-six-cancel-" + boundary.value,
            faults=faults,
        )
        reservation = await coordinator.reserve(identity, digest)
        pending = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(
                WorkerReport(
                    WorkerState.SETTLED,
                    _journal(plan, CommitStepState.COMMITTED),
                )
            ),
            "controller-a",
        )
        assert not isinstance(pending, PatchResult)
        assert (
            await coordinator.cancel(reservation, before_commit=False)
            == pending
        )
        reconciler.set_report(
            reservation.request_id,
            WorkerReport(
                WorkerState.FENCED,
                _journal(
                    plan,
                    CommitStepState.UNKNOWN,
                    artifact=ArtifactState.UNKNOWN,
                    postcondition=PostconditionState.UNKNOWN,
                ),
            ),
        )
        result = await _continue(coordinator, reservation, plan, grant)
        assert isinstance(result, PatchResult)
        assert result.status is PatchStatus.INDETERMINATE

    run(execute())


@pytest.mark.parametrize("boundary", tuple(CoordinatorBoundary))
def test_patch_phase_6_fault_boundaries_close_or_retain_settlement(
    boundary: CoordinatorBoundary,
) -> None:
    """Contain every precommit fault or retain its owned postcommit outcome."""

    async def execute() -> None:
        faults = ScriptedFaultController(frozenset((boundary,)))
        (
            coordinator,
            reconciler,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime(
            key="phase-six-fault-" + boundary.value,
            faults=faults,
        )
        reservation = await coordinator.reserve(identity, digest)
        worker = ScriptedCommitWorker(
            WorkerReport(
                WorkerState.SETTLED,
                _journal(plan, CommitStepState.COMMITTED),
            )
        )
        outcome = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            worker,
            "controller-a",
        )
        assert boundary in faults.observed
        if boundary in {
            CoordinatorBoundary.PRIVATE_STAGING,
            CoordinatorBoundary.LEASE,
            CoordinatorBoundary.REVALIDATION,
        }:
            assert isinstance(outcome, PatchResult)
            assert outcome.status is PatchStatus.STALE
            assert not worker.commands
        else:
            assert not isinstance(outcome, PatchResult)
            assert (
                await coordinator.execute(
                    reservation,
                    plan,
                    grant,
                    _snapshot(),
                    worker,
                    "controller-a",
                )
                == outcome
            )
            reconciler.set_report(
                reservation.request_id,
                WorkerReport(
                    WorkerState.FENCED,
                    _journal(
                        plan,
                        CommitStepState.UNKNOWN,
                        artifact=ArtifactState.UNKNOWN,
                        postcondition=PostconditionState.UNKNOWN,
                    ),
                ),
            )
            settled = await _continue(coordinator, reservation, plan, grant)
            assert isinstance(settled, PatchResult)
            assert settled.status is PatchStatus.INDETERMINATE
        assert coordinator.resources == RuntimeResources(0, 0, 0, 0, 0, 0)

    run(execute())


def test_patch_phase_6_pending_retains_one_fenced_owner_until_settlement() -> (
    None
):
    """Keep attached pending private while an owned worker settles once."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        (
            coordinator,
            reconciler,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime()
        reservation = await coordinator.reserve(identity, digest)
        pending = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
            "controller-a",
        )
        assert not isinstance(pending, PatchResult)
        assert coordinator.resources.lease_depth == 1
        assert (
            await coordinator.execute(
                reservation,
                plan,
                grant,
                _snapshot(),
                ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
                "controller-a",
            )
            == pending
        )
        assert (
            await _continue(coordinator, reservation, plan, grant) == pending
        )
        with pytest.raises(CoordinatorError) as denied:
            await _continue(
                coordinator, reservation, plan, grant, "controller-b"
            )
        assert denied.value.code is CoordinatorErrorCode.PENDING_OWNER
        assert (
            await coordinator.cancel(reservation, before_commit=False)
            == pending
        )
        assert (
            await _continue(coordinator, reservation, plan, grant) == pending
        )
        reconciler.set_report(
            reservation.request_id,
            WorkerReport(
                WorkerState.SETTLED,
                _journal(plan, CommitStepState.COMMITTED),
            ),
        )
        result = await _continue(coordinator, reservation, plan, grant)
        assert isinstance(result, PatchResult)
        assert result.status is PatchStatus.COMMITTED
        assert coordinator.resources.lease_depth == 0
        assert (
            await _continue(
                coordinator, reservation, plan, grant, "controller-b"
            )
            == result
        )

    run(execute())


def test_patch_phase_6_precommit_cancellation_and_no_retry_are_closed() -> (
    None
):
    """Settle cancellation without an effect and prohibit later raw replay."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        coordinator, _, identity, digest, plan, _ = await _runtime()
        reservation = await coordinator.reserve(identity, digest)
        assert (
            await coordinator.prepare(
                reservation,
                plan,
                approval_required=False,
            )
            is LifecyclePhase.APPROVED
        )
        cancelled = await coordinator.cancel(reservation, before_commit=True)
        assert isinstance(cancelled, PatchResult)
        assert cancelled.status is PatchStatus.CANCELLED
        assert cancelled.truth.mutation_state is MutationState.NOT_COMMITTED
        assert (
            await coordinator.cancel(reservation, before_commit=True)
            == cancelled
        )
        with pytest.raises(CoordinatorError) as retry:
            await coordinator.advance(reservation, LifecyclePhase.COMMIT_READY)
        assert retry.value.code is CoordinatorErrorCode.INVARIANT

    run(execute())


def test_patch_phase_6_domain_serialization_and_precommit_work() -> None:
    """Serialize shared-domain commits while another plan reaches approval."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        (
            coordinator,
            reconciler,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime()
        first = await coordinator.reserve(identity, digest)
        second_identity = replace(
            identity,
            retransmission_key=RetransmissionKey("phase-six-second"),
        )
        second = await coordinator.reserve(second_identity, digest)
        started = Event()
        release = Event()

        first_worker = ScriptedCommitWorker(
            WorkerReport(
                WorkerState.SETTLED, _journal(plan, CommitStepState.COMMITTED)
            ),
            started,
            release,
        )
        first_task = create_task(
            coordinator.execute(
                first, plan, grant, _snapshot(), first_worker, "controller-a"
            )
        )
        await started.wait()
        await coordinator.prepare(second, plan, approval_required=True)
        await coordinator.advance(second, LifecyclePhase.APPROVED)
        assert coordinator.resources.lease_depth == 1
        second_task = create_task(
            coordinator.execute(
                second,
                plan,
                grant,
                _snapshot(),
                ScriptedCommitWorker(
                    WorkerReport(
                        WorkerState.SETTLED,
                        _journal(plan, CommitStepState.COMMITTED),
                    )
                ),
                "controller-b",
            )
        )
        await sleep(0)
        assert not second_task.done()
        release.set()
        first_result = await first_task
        assert isinstance(first_result, PatchResult)
        with pytest.raises(CoordinatorError) as consumed:
            await second_task
        assert consumed.value.code is CoordinatorErrorCode.GRANT_CONSUMED
        assert reconciler is not None

    run(execute())


def test_patch_phase_6_rejects_non_scripted_workers_and_invalid_journals() -> (
    None
):
    """Keep local targets inactive and reject transient settlement truth."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        coordinator, _, identity, digest, plan, grant = await _runtime()
        reservation = await coordinator.reserve(identity, digest)

        class NonScriptedWorker(ScriptedCommitWorker):
            """Violate the exact scripted worker identity."""

        with pytest.raises(CoordinatorError) as target:
            await coordinator.execute(
                reservation,
                plan,
                grant,
                _snapshot(),
                NonScriptedWorker(
                    WorkerReport(
                        WorkerState.SETTLED,
                        _journal(plan, CommitStepState.COMMITTED),
                    )
                ),
                "controller-a",
            )
        assert target.value.code is CoordinatorErrorCode.SCRIPTED_TARGET_ONLY
        pending = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(
                WorkerReport(
                    WorkerState.SETTLED,
                    _journal(plan, CommitStepState.PLANNED),
                )
            ),
            "controller-a",
        )
        assert not isinstance(pending, PatchResult)

    run(execute())


def test_patch_phase_6_move_update_and_fenced_reconciliation_e2e() -> None:
    """Settle a scripted move-update graph and fenced report once."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        (
            coordinator,
            reconciler,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime(step_count=2)
        reservation = await coordinator.reserve(identity, digest)
        pending = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
            "controller-a",
        )
        assert not isinstance(pending, PatchResult)
        reconciler.set_report(
            reservation.request_id,
            WorkerReport(
                WorkerState.FENCED,
                _journal(
                    plan,
                    CommitStepState.COMMITTED,
                    CommitStepState.UNKNOWN,
                    artifact=ArtifactState.UNKNOWN,
                    postcondition=PostconditionState.UNKNOWN,
                ),
            ),
        )
        result = await _continue(coordinator, reservation, plan, grant)
        assert isinstance(result, PatchResult)
        assert result.status is PatchStatus.INDETERMINATE
        assert result.truth.commit_set_exact is False
        assert await _continue(coordinator, reservation, plan, grant) == result

    run(execute())


def test_patch_phase_6_invalid_boundaries_and_fault_paths_stay_closed() -> (
    None
):
    """Reject malformed state, fenced work, and unowned pending operations."""
    assert callable(InMemoryPatchCoordinator)
    plan, grant, approvals = run(_approved_with_service())
    domain = plan.binding.target.domain_id
    fact = RevalidationFact(RevalidationField.CONTEXT, "key", "value")
    journal = _journal(plan, CommitStepState.COMMITTED)
    identity = RuntimeIdentity(
        plan.binding.subject,
        plan.binding.final.approval.route,
        RetransmissionKey("phase-six-faults"),
    )

    with pytest.raises(CoordinatorError):
        RetransmissionKey("")
    with pytest.raises(CoordinatorError):
        RevalidationFact(RevalidationField.CONTEXT, "", "value")
    with pytest.raises(CoordinatorError):
        RevalidationSnapshot(())
    with pytest.raises(CoordinatorError):
        RevalidationResult(True, fact)
    with pytest.raises(CoordinatorError):
        LockFootprint(domain, ("path",))
    with pytest.raises(PatchValidationError):
        PatchStepId("")

    class DerivedStepId(PatchStepId):
        """Model a noncanonical step identifier subtype at the boundary."""

    with pytest.raises(CoordinatorError):
        JournalStep(
            DerivedStepId(PatchStepId.new().value),
            PatchLineageId.new(),
            CommitStepState.COMMITTED,
        )
    with pytest.raises(CoordinatorError):
        ArtifactJournal("", ArtifactState.ABSENT)
    with pytest.raises(CoordinatorError):
        SettlementJournal((), (), PostconditionState.UNKNOWN)
    with pytest.raises(CoordinatorError):
        WorkerReport(WorkerState.LIVE, journal)
    with pytest.raises(CoordinatorError):
        SealedCommitCommand(
            plan,
            CommitLease(
                PatchDomainId("domain_" + "b" * 16),
                PatchRequestId.new(),
                1,
            ),
            LockFootprint(PatchDomainId("domain_" + "b" * 16), ("workspace",)),
        )

    async def execute() -> None:
        store = InMemoryCoordinatorStore(approvals)
        reservation = await store.reserve(
            identity, plan.binding.request_digest
        )
        invalid = replace(reservation, request_id=PatchRequestId.new())
        with pytest.raises(CoordinatorError):
            await store.record(invalid)
        with pytest.raises(CoordinatorError):
            await store.assign_lease(invalid, domain)
        lease = await store.assign_lease(reservation, domain)
        assert await store.assign_lease(reservation, domain) == lease
        record = await store.record(reservation)
        record.lifecycle = LifecyclePhase.APPROVED
        with pytest.raises(CoordinatorError) as fenced:
            await store.begin_commit(
                reservation,
                plan,
                grant,
                CommitLease(domain, reservation.request_id, lease.fence + 1),
            )
        assert fenced.value.code is CoordinatorErrorCode.FENCED
        with pytest.raises(CoordinatorError) as stale:
            await store.begin_commit(
                reservation,
                plan,
                replace(grant, plan_id=PatchPlanId.new()),
                lease,
            )
        assert stale.value.code is CoordinatorErrorCode.STALE
        with pytest.raises(CoordinatorError):
            await store.append(reservation.request_id, journal)
        await store.begin_commit(reservation, plan, grant, lease)
        await store.append(reservation.request_id, journal)
        record.lifecycle = LifecyclePhase.SETTLEMENT_PENDING
        with pytest.raises(CoordinatorError):
            await store.append(reservation.request_id, journal)
        assert await store.terminal(PatchRequestId.new()) is None
        leases = InMemoryLeaseManager(store)
        with pytest.raises(CoordinatorError):
            await leases.release(lease)
        extra = RevalidationFact(RevalidationField.CWD, "extra", "value")
        changed = RevalidationSnapshot(
            tuple(
                sorted(
                    (*_snapshot().facts, extra),
                    key=lambda item: (item.field.value, item.key, item.value),
                )
            )
        )
        mismatch = await ScriptedReconciler(changed).revalidate(_snapshot())
        assert not mismatch.matched and mismatch.mismatch is not None
        (
            coordinator,
            reconciler,
            runtime,
            digest,
            sealed,
            approved,
        ) = await _runtime(
            key="phase-six-owner",
        )
        prepared = await coordinator.reserve(runtime, digest)
        second_plan = await _sealed_plan()
        await coordinator.prepare(prepared, sealed, approval_required=False)
        with pytest.raises(CoordinatorError):
            await coordinator.prepare(
                prepared,
                second_plan,
                approval_required=False,
            )
        with pytest.raises(CoordinatorError):
            await coordinator.execute(
                prepared,
                second_plan,
                approved,
                _snapshot(),
                ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
                "controller-a",
            )
        pending = await coordinator.execute(
            prepared,
            sealed,
            approved,
            _snapshot(),
            ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
            "controller-a",
        )
        assert not isinstance(pending, PatchResult)
        with pytest.raises(CoordinatorError):
            await coordinator.execute(
                prepared,
                sealed,
                approved,
                _snapshot(),
                ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
                "controller-b",
            )
        with pytest.raises(CoordinatorError):
            await _continue(
                coordinator, prepared, sealed, approved, "controller-b"
            )
        reconciler.set_report(
            prepared.request_id,
            WorkerReport(
                WorkerState.SETTLED,
                _journal(plan, CommitStepState.COMMITTED),
            ),
        )
        settled = await _continue(coordinator, prepared, sealed, approved)
        assert isinstance(settled, PatchResult)
        assert (
            await coordinator.execute(
                prepared,
                sealed,
                approved,
                _snapshot(),
                ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
                "controller-a",
            )
            == settled
        )
        record.result = settled
        with pytest.raises(CoordinatorError):
            await store.begin_commit(reservation, plan, grant, lease)
        untouched = await coordinator.reserve(
            replace(
                runtime, retransmission_key=RetransmissionKey("unprepared")
            ),
            digest,
        )
        with pytest.raises(CoordinatorError):
            await coordinator.cancel(untouched, before_commit=True)
        blocked = await coordinator.reserve(
            replace(runtime, retransmission_key=RetransmissionKey("blocked")),
            digest,
        )
        await coordinator.prepare(blocked, sealed, approval_required=True)
        with pytest.raises(CoordinatorError):
            await coordinator.execute(
                blocked,
                sealed,
                approved,
                _snapshot(),
                ScriptedCommitWorker(
                    WorkerReport(WorkerState.SETTLED, journal)
                ),
                "controller-a",
            )
        blocked_record = await coordinator._store.record(blocked)
        blocked_record.lifecycle = LifecyclePhase.COMMIT_STARTED
        with pytest.raises(CoordinatorError):
            await coordinator.cancel(blocked, before_commit=False)
        assert reconciler is not None

    run(execute())


def test_patch_phase_6_lifecycle_controller_rejects_illegal_order() -> None:
    """Advance only the sealed precommit lifecycle order."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        coordinator, _, identity, digest, plan, _ = await _runtime(
            key="phase-six-lifecycle"
        )
        reservation = await coordinator.reserve(identity, digest)
        with pytest.raises(CoordinatorError):
            await coordinator.advance(reservation, LifecyclePhase.COMMIT_READY)
        assert (
            await coordinator.prepare(
                reservation, plan, approval_required=True
            )
            is LifecyclePhase.APPROVAL_REQUIRED
        )
        assert (
            await coordinator.advance(reservation, LifecyclePhase.APPROVED)
            is LifecyclePhase.APPROVED
        )

    run(execute())


def test_patch_phase_6_footprint_is_workspace_first_and_total() -> None:
    """Keep the conservative workspace lock key before every sealed path."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        plan, _ = await _approved()
        footprint = footprint_for(plan)
        assert footprint.domain_id == plan.binding.target.domain_id
        assert footprint.keys[0] == "workspace"
        assert footprint.keys[1:] == tuple(sorted(footprint.keys[1:]))
        assert "note0.txt" in footprint.keys

    run(execute())


def test_patch_phase_6_pending_waits_own_no_transaction_or_target_handle() -> (
    None
):
    """Expose only the fenced lease while an attached settlement is pending."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        (
            coordinator,
            reconciler,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime(key="phase-six-resources")
        reservation = await coordinator.reserve(identity, digest)
        pending = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
            "controller-a",
        )
        assert not isinstance(pending, PatchResult)
        assert coordinator.resources.transaction_depth == 0
        assert coordinator.resources.target_handle_depth == 0
        assert coordinator.resources.worker_depth == 0
        assert coordinator.resources.private_staging_depth == 0
        reconciler.set_report(
            reservation.request_id,
            WorkerReport(
                WorkerState.SETTLED,
                _journal(plan, CommitStepState.NOT_COMMITTED),
            ),
        )
        result = await _continue(coordinator, reservation, plan, grant)
        assert isinstance(result, PatchResult)
        assert result.status is PatchStatus.COMMIT_FAILED

    run(execute())


def test_patch_phase_6_grant_consumption_precedes_only_scripted_effects() -> (
    None
):
    """Consume one grant after revalidation and before its sole command."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        coordinator, _, identity, digest, plan, grant = await _runtime(
            key="phase-six-grant"
        )
        reservation = await coordinator.reserve(identity, digest)
        worker = ScriptedCommitWorker(
            WorkerReport(
                WorkerState.SETTLED, _journal(plan, CommitStepState.COMMITTED)
            )
        )
        result = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            worker,
            "controller-a",
        )
        assert isinstance(result, PatchResult)
        assert len(worker.commands) == 1
        command = worker.commands[0]
        assert command.lease.request_id == reservation.request_id
        assert command.lease.fence == 1

    run(execute())


def test_patch_phase_6_internal_pending_never_creates_a_detached_handle() -> (
    None
):
    """Keep pending inspection and settlement attached to one controller."""
    assert callable(InMemoryPatchCoordinator)

    async def execute() -> None:
        (
            coordinator,
            reconciler,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime(key="phase-six-attached")
        reservation = await coordinator.reserve(identity, digest)
        pending = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
            "controller-a",
        )
        assert not isinstance(pending, PatchResult)
        assert not hasattr(coordinator, "await_terminal")
        assert not hasattr(coordinator, "resume")
        reconciler.set_report(
            reservation.request_id,
            WorkerReport(
                WorkerState.SETTLED,
                _journal(plan, CommitStepState.COMMITTED),
            ),
        )
        assert isinstance(
            await _continue(coordinator, reservation, plan, grant),
            PatchResult,
        )

    run(execute())


def test_patch_phase_6_has_no_public_mutation_tool_or_local_worker() -> None:
    """Keep direct mutation callables and workers inactive by default."""
    assert callable(InMemoryPatchCoordinator)
    import avalan.patch as patch

    assert hasattr(patch, "PatchToolSet")
    assert not hasattr(patch, "patch_edit")
    assert not hasattr(patch, "patch_apply")
    assert InMemoryPatchCoordinator.scheduler_parallel_safe is False


def test_patch_phase_6_defensive_fault_and_digest_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed injected fault values and wrong preplanned digests."""
    monkeypatch.setattr(
        coordinator_module,
        "type",
        lambda value: str,
        raising=False,
    )
    with pytest.raises(CoordinatorError):
        ScriptedFaultController()
    monkeypatch.undo()

    async def execute() -> None:
        controller = ScriptedFaultController()
        monkeypatch.setattr(
            coordinator_module,
            "type",
            lambda value: str,
            raising=False,
        )
        with pytest.raises(CoordinatorError):
            await controller.checkpoint(
                CoordinatorBoundary.LEASE,
                RuntimeResources(0, 0, 0, 0, 0, 0),
            )
        monkeypatch.undo()
        coordinator, _, _, _, _, _ = await _runtime(
            key="phase-six-inert-history"
        )
        monkeypatch.setattr(
            coordinator_module,
            "type",
            lambda value: str,
            raising=False,
        )
        with pytest.raises(CoordinatorError):
            await coordinator.replay_inert_history((b"historical",))
        monkeypatch.undo()
        coordinator, _, identity, _, plan, grant = await _runtime(
            key="phase-six-wrong-digest"
        )
        reservation = await coordinator.reserve(
            identity,
            AlgorithmDigest.from_bytes(b"wrong-preplanning-digest"),
        )
        with pytest.raises(CoordinatorError) as stale:
            await coordinator.execute(
                reservation,
                plan,
                grant,
                _snapshot(),
                ScriptedCommitWorker(
                    WorkerReport(
                        WorkerState.SETTLED,
                        _journal(plan, CommitStepState.COMMITTED),
                    )
                ),
                "controller-a",
            )
        assert stale.value.code is CoordinatorErrorCode.STALE

        async def stale_lease(_: CommitLease) -> bool:
            """Reject a worker whose lease lost authority before commit."""
            return False

        coordinator, _, identity, digest, plan, grant = await _runtime(
            key="phase-six-stale-lease"
        )
        monkeypatch.setattr(coordinator._leases, "is_current", stale_lease)
        reservation = await coordinator.reserve(identity, digest)
        pending = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(
                WorkerReport(
                    WorkerState.SETTLED,
                    _journal(plan, CommitStepState.COMMITTED),
                )
            ),
            "controller-a",
        )
        assert not isinstance(pending, PatchResult)

    run(execute())


def test_patch_phase_6_commit_uses_only_issued_unexpired_grants() -> None:
    """Reject forged, expired, rebound, and replayed grants before work."""

    async def execute() -> None:
        plan, grant, approvals = await _approved_with_service()
        rejected = (
            replace(grant, grant_id=PatchGrantId.new()),
            replace(
                grant,
                diff_digest=AlgorithmDigest.from_bytes(b"wrong-diff"),
            ),
        )
        for index, candidate in enumerate(rejected, start=1):
            store = InMemoryCoordinatorStore(approvals)
            coordinator = InMemoryPatchCoordinator(
                store,
                InMemoryLeaseManager(store),
                ScriptedReconciler(_snapshot()),
            )
            reservation = await coordinator.reserve(
                RuntimeIdentity(
                    plan.binding.subject,
                    plan.binding.final.approval.route,
                    RetransmissionKey("phase-six-forged-" + str(index)),
                ),
                plan.binding.request_digest,
            )
            worker = ScriptedCommitWorker(
                WorkerReport(
                    WorkerState.SETTLED,
                    _journal(plan, CommitStepState.COMMITTED),
                )
            )
            with pytest.raises(CoordinatorError) as rejected_grant:
                await coordinator.execute(
                    reservation,
                    plan,
                    candidate,
                    _snapshot(),
                    worker,
                    "controller-a",
                )
            assert rejected_grant.value.code is CoordinatorErrorCode.STALE
            assert not worker.commands

        store = InMemoryCoordinatorStore(approvals)
        coordinator = InMemoryPatchCoordinator(
            store,
            InMemoryLeaseManager(store),
            ScriptedReconciler(_snapshot()),
        )
        wrong_subject = replace(
            plan.binding.subject,
            principal=PatchPrincipalId("principal-b"),
        )
        reservation = await coordinator.reserve(
            RuntimeIdentity(
                wrong_subject,
                plan.binding.final.approval.route,
                RetransmissionKey("phase-six-wrong-subject"),
            ),
            plan.binding.request_digest,
        )
        worker = ScriptedCommitWorker(
            WorkerReport(
                WorkerState.SETTLED,
                _journal(plan, CommitStepState.COMMITTED),
            )
        )
        with pytest.raises(CoordinatorError) as wrong_subject_error:
            await coordinator.execute(
                reservation,
                plan,
                grant,
                _snapshot(),
                worker,
                "controller-a",
            )
        assert wrong_subject_error.value.code is CoordinatorErrorCode.STALE
        assert not worker.commands

        clock = _Clock()
        expiring = ApprovalService(_Broker(), clock, RuntimeGrantStore())
        expiring_grant = await _issue_grant(plan, expiring)
        clock.tick = 100
        expired_store = InMemoryCoordinatorStore(expiring)
        expired = InMemoryPatchCoordinator(
            expired_store,
            InMemoryLeaseManager(expired_store),
            ScriptedReconciler(_snapshot()),
        )
        reservation = await expired.reserve(
            RuntimeIdentity(
                plan.binding.subject,
                plan.binding.final.approval.route,
                RetransmissionKey("phase-six-expired"),
            ),
            plan.binding.request_digest,
        )
        worker = ScriptedCommitWorker(
            WorkerReport(
                WorkerState.SETTLED,
                _journal(plan, CommitStepState.COMMITTED),
            )
        )
        with pytest.raises(CoordinatorError) as expired_error:
            await expired.execute(
                reservation,
                plan,
                expiring_grant,
                _snapshot(),
                worker,
                "controller-a",
            )
        assert expired_error.value.code is CoordinatorErrorCode.STALE
        assert not worker.commands

        replay_store = InMemoryCoordinatorStore(approvals)
        replay = InMemoryPatchCoordinator(
            replay_store,
            InMemoryLeaseManager(replay_store),
            ScriptedReconciler(_snapshot()),
        )
        first = await replay.reserve(
            RuntimeIdentity(
                plan.binding.subject,
                plan.binding.final.approval.route,
                RetransmissionKey("phase-six-grant-first"),
            ),
            plan.binding.request_digest,
        )
        first_worker = ScriptedCommitWorker(
            WorkerReport(
                WorkerState.SETTLED,
                _journal(plan, CommitStepState.COMMITTED),
            )
        )
        assert isinstance(
            await replay.execute(
                first,
                plan,
                grant,
                _snapshot(),
                first_worker,
                "controller-a",
            ),
            PatchResult,
        )
        second = await replay.reserve(
            RuntimeIdentity(
                plan.binding.subject,
                plan.binding.final.approval.route,
                RetransmissionKey("phase-six-grant-second"),
            ),
            plan.binding.request_digest,
        )
        replay_worker = ScriptedCommitWorker(
            WorkerReport(
                WorkerState.SETTLED,
                _journal(plan, CommitStepState.COMMITTED),
            )
        )
        with pytest.raises(CoordinatorError) as consumed:
            await replay.execute(
                second,
                plan,
                grant,
                _snapshot(),
                replay_worker,
                "controller-a",
            )
        assert consumed.value.code is CoordinatorErrorCode.GRANT_CONSUMED
        assert not replay_worker.commands

    run(execute())


def test_patch_phase_6_journal_vectors_bind_to_the_sealed_graph() -> None:
    """Reject missing, extra, foreign, reordered, and wrong artifacts."""

    async def execute() -> None:
        for label in ("missing", "extra", "foreign", "reordered", "artifact"):
            (
                coordinator,
                reconciler,
                identity,
                digest,
                plan,
                grant,
            ) = await _runtime(
                key="phase-six-journal-" + label,
                step_count=2,
            )
            valid = _journal(
                plan,
                CommitStepState.COMMITTED,
                CommitStepState.NOT_COMMITTED,
            )
            match label:
                case "missing":
                    invalid = SettlementJournal(
                        valid.steps[:1],
                        valid.artifacts,
                        valid.postcondition,
                    )
                case "extra":
                    invalid = SettlementJournal(
                        valid.steps
                        + (
                            JournalStep(
                                PatchStepId.new(),
                                valid.steps[0].lineage,
                                CommitStepState.COMMITTED,
                            ),
                        ),
                        valid.artifacts,
                        valid.postcondition,
                    )
                case "foreign":
                    invalid = SettlementJournal(
                        (
                            JournalStep(
                                PatchStepId.new(),
                                valid.steps[0].lineage,
                                CommitStepState.COMMITTED,
                            ),
                            valid.steps[1],
                        ),
                        valid.artifacts,
                        valid.postcondition,
                    )
                case "reordered":
                    invalid = SettlementJournal(
                        tuple(reversed(valid.steps)),
                        valid.artifacts,
                        valid.postcondition,
                    )
                case "artifact":
                    invalid = SettlementJournal(
                        valid.steps,
                        (
                            ArtifactJournal(
                                "foreign-artifact", ArtifactState.CLEANED
                            ),
                            valid.artifacts[1],
                        ),
                        valid.postcondition,
                    )
                case _:
                    raise AssertionError(label)
            with pytest.raises(CoordinatorError):
                SettlementJournal(
                    valid.steps + (valid.steps[0],),
                    valid.artifacts,
                    valid.postcondition,
                )
            reservation = await coordinator.reserve(identity, digest)
            pending = await coordinator.execute(
                reservation,
                plan,
                grant,
                _snapshot(),
                ScriptedCommitWorker(
                    WorkerReport(WorkerState.SETTLED, invalid)
                ),
                "controller-a",
            )
            assert not isinstance(pending, PatchResult)
            reconciler.set_report(
                reservation.request_id,
                WorkerReport(WorkerState.SETTLED, valid),
            )
            assert isinstance(
                await _continue(coordinator, reservation, plan, grant),
                PatchResult,
            )

    run(execute())


def test_patch_phase_6_poststart_failures_stay_attached_and_settle_once() -> (
    None
):
    """Retain worker failures and serialize continuation settlement."""

    async def execute() -> None:
        (
            coordinator,
            reconciler,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime(key="phase-six-poststart-race")
        reservation = await coordinator.reserve(identity, digest)
        pending = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
            "controller-a",
        )
        assert not isinstance(pending, PatchResult)
        assert not hasattr(coordinator, "inspect")
        assert not hasattr(coordinator, "settle_pending")
        assert not hasattr(pending, "pending_operation_id")
        assert not hasattr(pending, "request_id")
        blocked_reconciler = _BlockedReconciler(_snapshot())
        blocked_reconciler.set_terminal_report(
            WorkerReport(
                WorkerState.SETTLED,
                _journal(plan, CommitStepState.COMMITTED),
            )
        )
        coordinator._reconciler = blocked_reconciler
        first = create_task(_continue(coordinator, reservation, plan, grant))
        await blocked_reconciler.started.wait()
        second = create_task(_continue(coordinator, reservation, plan, grant))
        await sleep(0)
        assert not second.done()
        blocked_reconciler.release.set()
        first_result, second_result = await gather(first, second)
        assert isinstance(first_result, PatchResult)
        assert first_result == second_result
        assert blocked_reconciler.calls == 1
        assert coordinator.resources == RuntimeResources(0, 0, 0, 0, 0, 0)

    run(execute())


def test_patch_phase_6_private_pending_defenses_fail_closed() -> None:
    """Exercise only the coordinator's remaining defensive pending guards."""

    async def execute() -> None:
        with pytest.raises(CoordinatorError):
            coordinator_module._AttachedPending(LifecyclePhase.RECEIVED)

        plan, grant = await _approved()
        untrusted_store = InMemoryCoordinatorStore()
        identity = RuntimeIdentity(
            plan.binding.subject,
            plan.binding.final.approval.route,
            RetransmissionKey("phase-six-no-grant-validator"),
        )
        reservation = await untrusted_store.reserve(
            identity, plan.binding.request_digest
        )
        lease = await untrusted_store.assign_lease(
            reservation, plan.binding.target.domain_id
        )
        record = await untrusted_store.record(reservation)
        record.lifecycle = LifecyclePhase.APPROVED
        with pytest.raises(CoordinatorError) as untrusted:
            await untrusted_store.begin_commit(reservation, plan, grant, lease)
        assert untrusted.value.code is CoordinatorErrorCode.STALE

        (
            coordinator,
            reconciler,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime(key="phase-six-private-pending")
        faulting_reconciler = _FaultingReconciler(_snapshot())
        coordinator._reconciler = faulting_reconciler
        reservation = await coordinator.reserve(identity, digest)
        pending = await coordinator.execute(
            reservation,
            plan,
            grant,
            _snapshot(),
            ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
            "controller-a",
        )
        assert not isinstance(pending, PatchResult)
        record = await coordinator._store.record(reservation)
        with pytest.raises(CoordinatorError):
            await coordinator._enter_pending(
                record, reservation, "controller-b"
            )
        assert (
            await coordinator._enter_pending(
                record, reservation, "controller-a"
            )
            == pending
        )
        record.plan = None
        with pytest.raises(CoordinatorError):
            await coordinator._continue_pending(
                record,
                reservation,
                "controller-a",
                ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
            )
        record.plan = plan
        valid = _journal(plan, CommitStepState.COMMITTED)
        invalid = SettlementJournal(
            (
                JournalStep(
                    PatchStepId.new(),
                    valid.steps[0].lineage,
                    CommitStepState.COMMITTED,
                ),
            ),
            valid.artifacts,
            valid.postcondition,
        )
        faulting_reconciler.set_report(
            reservation.request_id,
            WorkerReport(WorkerState.SETTLED, invalid),
        )
        assert not isinstance(
            await _continue(coordinator, reservation, plan, grant),
            PatchResult,
        )
        faulting_reconciler.unavailable = True
        assert not isinstance(
            await _continue(coordinator, reservation, plan, grant),
            PatchResult,
        )
        faulting_reconciler.unavailable = False
        faulting_reconciler.set_report(
            reservation.request_id,
            WorkerReport(WorkerState.SETTLED, valid),
        )
        coordinator._leases = _UnavailableLeaseManager(coordinator._store)
        assert isinstance(
            await _continue(coordinator, reservation, plan, grant),
            PatchResult,
        )
        assert record.lease is not None
        await coordinator._release_lease(record, record.lease)

        (
            waiting,
            _,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime(key="phase-six-commitstarted-wait")
        reservation = await waiting.reserve(identity, digest)
        started = Event()
        blocked = Event()
        leader = create_task(
            waiting.execute(
                reservation,
                plan,
                grant,
                _snapshot(),
                ScriptedCommitWorker(
                    WorkerReport(WorkerState.SETTLED, valid),
                    started,
                    blocked,
                ),
                "controller-a",
            )
        )
        await started.wait()
        follower = create_task(
            waiting.execute(
                reservation,
                plan,
                grant,
                _snapshot(),
                ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
                "controller-a",
            )
        )
        await sleep(0)
        leader.cancel()
        leader_result = await leader
        follower_result = await follower
        assert not isinstance(leader_result, PatchResult)
        assert not isinstance(follower_result, PatchResult)

        (
            inconsistent,
            _,
            identity,
            digest,
            plan,
            grant,
        ) = await _runtime(key="phase-six-inconsistent-start")
        reservation = await inconsistent.reserve(identity, digest)
        record = await inconsistent._store.record(reservation)
        record.plan = plan
        record.lifecycle = LifecyclePhase.COMMIT_STARTED
        record.state_changed.set()
        with pytest.raises(CoordinatorError):
            await inconsistent.execute(
                reservation,
                plan,
                grant,
                _snapshot(),
                ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
                "controller-a",
            )

    run(execute())
