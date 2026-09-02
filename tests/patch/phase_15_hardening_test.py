"""Exercise bounded Phase 15 adversarial hardening contracts."""

from asyncio import (
    CancelledError,
    Future,
    Lock,
    create_subprocess_exec,
    create_task,
    gather,
    get_running_loop,
    run,
    sleep,
    wait_for,
)
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from hashlib import sha256
from json import dumps, loads
from logging import Logger
from pathlib import Path
from random import Random
from sys import platform as runtime_platform
from types import SimpleNamespace
from typing import TypeVar, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from patch_activation_support import (
    activated_patch_test_profile,
    patch_container_test_image,
    patch_test_activation_factory,
    phase15_local_target_profile,
)

from avalan.agent.loader import OrchestratorLoader
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    OrchestratorSettings,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallError,
    ToolCallResult,
    ToolManagerExecutionMode,
    ToolManagerSettings,
    ToolValue,
)
from avalan.isolation import (
    IsolationMode,
    IsolationProfileSelection,
    IsolationSettings,
    SandboxBackend,
    trusted_isolation_source,
)
from avalan.model.capability import (
    ModelCapabilityCatalog,
    ProviderCapabilityCall,
)
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.patch import activation as activation_module
from avalan.patch.activation import (
    PatchActivationDurableOperation,
    PatchActivationError,
    PatchActivationLease,
    PatchActivationLimits,
    PatchActivationOperationBinding,
    PatchActivationOperationState,
    PatchActivationPlatform,
    PatchActivationProfileKey,
    PatchActivationRegistry,
    PatchActivationRuntime,
    PatchActivationRuntimeAuthority,
    PatchActivationRuntimeFactory,
    PatchActivationRuntimeRecord,
    PatchActivationVerifier,
    PatchCapabilityProfile,
    PatchDeactivationReceipt,
    PatchProductionManifest,
    PatchProductionSource,
    PatchProfileComponent,
    PatchProfileProofs,
    PatchProfileState,
    PatchProtocolDescription,
    PatchSchemaDescription,
    PatchVerifiedActivationReceipt,
    _build_activation_factory,
    _build_activation_verifier,
    _durable_operation,
    _issue_verified_receipt,
    _manifest,
    _new_activation_authority,
    _production_evidence_digest,
    _receipt_matches,
    _runtime_record,
    build_patch_activation_verifier,
    build_patch_production_manifest,
    render_patch_production_manifest,
)
from avalan.patch.container_target import (
    ContainerInspectionTarget,
    ContainerPatchImage,
    ContainerPatchRuntimeBinder,
    ContainerPatchRuntimeContext,
    ContainerPatchRuntimeSettings,
    ContainerPatchTarget,
    ContainerPersistentLeaseAuthority,
    container_protocol_id,
)
from avalan.patch.coordinator import (
    ArtifactJournal,
    CoordinatorBoundary,
    InMemoryCoordinatorStore,
    InMemoryLeaseManager,
    InMemoryPatchCoordinator,
    JournalStep,
    RetransmissionKey,
    RevalidationFact,
    RevalidationField,
    RevalidationSnapshot,
    RuntimeIdentity,
    RuntimeResources,
    ScriptedFaultController,
    ScriptedReconciler,
    SettlementJournal,
    WorkerReport,
    WorkerState,
)
from avalan.patch.domain import (
    AlgorithmDigest,
    ApprovalMode,
    ArtifactState,
    Audience,
    ByteSize,
    Capability,
    CommitStepState,
    CommitTruth,
    ContextKind,
    DurationTicks,
    ErrorStage,
    ExpiryTick,
    FileMode,
    LifecyclePhase,
    LineageState,
    LogicalPath,
    MetadataProfile,
    MutationState,
    OperationType,
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
    PatchInput,
    PatchInvocationOutcome,
    PatchLifecycleEvent,
    PatchLimits,
    PatchLineageId,
    PatchObserverCorrelationId,
    PatchObserverId,
    PatchPending,
    PatchPendingOperationId,
    PatchPlanId,
    PatchProtocolId,
    PatchRequest,
    PatchRequestId,
    PatchResult,
    PatchRetentionKeyId,
    PatchRetentionRecordId,
    PatchStatus,
    PatchStepId,
    PatchTargetId,
    PatchWorkspaceId,
    PostconditionState,
    RequestedEffectOccurrence,
    Retryability,
    SequenceNumber,
    SourceBytes,
    WorkspaceChange,
    coarsen_error_code,
)
from avalan.patch.durable_approval import (
    HmacDurableApprovalAuthority,
    PhaseFiveDurableApprovalIssuer,
)
from avalan.patch.durable_coordinator import (
    DurableArtifactObservation,
    DurablePatchReconciler,
)
from avalan.patch.durable_retention import (
    AesGcmDurableRetentionCipher,
    AesGcmDurableRetentionEnvelopeValidator,
    DurableRetentionBinding,
    DurableRetentionKey,
    InMemoryDurableRetentionKeyResolver,
    StaticDurableRetentionAuthorizer,
)
from avalan.patch.durable_store import (
    DurableApproval,
    DurableArtifactState,
    DurableCommitClaimState,
    DurableCommitLease,
    DurablePendingRequest,
    DurablePlanReference,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableRetentionAccess,
    DurableRetentionKind,
    DurableRetentionPolicy,
    DurableRetentionRecord,
    DurableStepBinding,
    DurableStoreError,
    DurableStoreErrorCode,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.local_commit import LocalCommitTarget
from avalan.patch.parser import (
    CanonicalPatchRequest,
    PatchInputError,
    PatchInputErrorCode,
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderProfile,
    RawToolCallId,
)
from avalan.patch.planner import (
    BoundedPlannerWorker,
    LogicalText,
    PlannerCandidate,
    PlannerError,
    PlannerErrorCode,
    PlannerFacade,
    PlannerFile,
    PlannerLimits,
    PlannerWorkspace,
    find_match,
    plan,
    render_review_diff,
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
    PlanReviewRequest,
    PolicyAuthorizer,
    PolicyBrokerId,
    PolicyError,
    PolicyPathSelector,
    PolicyReviewerRole,
    PolicyRevision,
    PolicyRouteId,
    PolicyRule,
    PreauthorizationClass,
    PreflightRequest,
    ReviewerDecision,
    RuntimeGrantStore,
    TrustedPatchPolicy,
    compose_limits,
    seal_plan,
)
from avalan.patch.protocols import (
    PatchProtocolProfile,
    PatchProtocolSurface,
)
from avalan.patch.rooted_worker import capture_rooted_root_binding
from avalan.patch.sandbox_commit import (
    PatchActivationObserver,
    SandboxChannelId,
    SandboxCommitTarget,
    SandboxContextLifetimeId,
    SandboxExecutionPlanFingerprint,
    SandboxInspectionTarget,
    SandboxPatchRuntimeBinder,
    SandboxPatchRuntimeContext,
    SandboxPatchRuntimeSettings,
    SandboxPatchServiceConfiguration,
    SandboxWorkerImplementationId,
    SandboxWorkerProtocolVersion,
    sandbox_protocol_id,
)
from avalan.patch.sandbox_wire import canonical_sandbox_plan_bytes
from avalan.patch.target import (
    InspectionRequest,
    LocalInspectionTarget,
    LocalPlatformProfile,
    LocalScopeResolver,
    LocalTargetProfile,
    ResolvedMutationScope,
    ScopeSelection,
    TargetHandshake,
    TargetIdentity,
    TargetPrimitive,
)
from avalan.patch.toolset import (
    PATCH_APPLY_SCHEMA,
    PATCH_EDIT_SCHEMA,
    PatchApprovalBinding,
    PatchCoordinatorBinding,
    PatchInvocationCapability,
    PatchInvocationHandle,
    PatchPersistenceBinding,
    PatchRuntimeBinding,
    PatchTestHostProfile,
    PatchToolError,
    PatchToolLoader,
    PatchToolManagerBundle,
    PatchToolSettings,
)
from avalan.sandbox.backend import (
    BubblewrapSandboxBackend,
    SandboxResultStatus,
    SeatbeltSandboxBackend,
)
from avalan.sandbox.planning import (
    SandboxExecutionPlan,
    SandboxPlanRequest,
    SandboxPlanRequestKind,
)
from avalan.tool import Tool, ToolSet
from avalan.tool.code import CodeTool
from avalan.tool.context import ToolSettingsContext
from avalan.tool.manager import ToolManager

_ROOT = Path(__file__).resolve().parents[2]
_FUZZ_FIXTURE = _ROOT / "tests/fixtures/patch/phase15_fuzz_corpus.json"
_ServiceValue = TypeVar("_ServiceValue")


@dataclass(frozen=True, slots=True)
class _FuzzCase:
    """Store one bounded, replayable fuzz input without live generation."""

    identifier: str
    category: str
    payload: bytes
    expected: str


def _resolved_future(value: _ServiceValue) -> Future[_ServiceValue]:
    """Return one current-loop future resolved to a typed service outcome."""
    future: Future[_ServiceValue] = get_running_loop().create_future()
    future.set_result(value)
    return future


def _phase15_limits() -> PatchLimits:
    """Return finite direct-integration limits for public JSON ingress."""
    return PatchLimits(
        ByteSize(128),
        ByteSize(8),
        ByteSize(96),
        ByteSize(8),
        ByteSize(8),
        ByteSize(4096),
        ByteSize(4096),
        ByteSize(4096),
        DurationTicks(100),
        DurationTicks(100),
        DurationTicks(100),
    )


def _phase15_policy() -> TrustedPatchPolicy:
    """Return a preauthorized policy for deterministic service tests."""
    reader = PreauthorizationClass("phase15-public-reader")
    return TrustedPatchPolicy(
        PolicyRevision("phase15-public-v1"),
        frozenset((OperationType.EDIT, OperationType.APPLY)),
        (
            PolicyRule(
                PolicyPathSelector(None),
                tuple(
                    CapabilityMode(
                        capability,
                        ApprovalMode.PREAUTHORIZED,
                        reader,
                    )
                    for capability in Capability
                ),
                atomicity_classes=frozenset(
                    (
                        "single_step",
                        "dependency_ordered",
                    )
                ),
            ),
        ),
        approval=ApprovalRequirements(
            ApprovalMode.PREAUTHORIZED,
            PolicyRouteId("phase15-public-route"),
            PolicyBrokerId("phase15-public-broker"),
            PolicyReviewerRole("phase15-public-reviewer"),
            1,
            reader,
        ),
    )


def _phase15_result(
    request_id: PatchRequestId, plan_id: PatchPlanId
) -> PatchResult:
    """Return one exact terminal result for a durable service request."""
    return PatchResult(
        1,
        request_id,
        plan_id,
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


def _phase15_rejected_result(
    request_id: PatchRequestId, plan_id: PatchPlanId
) -> PatchResult:
    """Return one content-free terminal result for rejected planning."""
    return PatchResult(
        1,
        request_id,
        plan_id,
        LifecyclePhase.REQUEST_COMPLETED,
        PatchStatus.REJECTED,
        CommitTruth(
            MutationState.NOT_COMMITTED,
            LineageState.NOT_COMMITTED,
            RequestedEffectOccurrence.FALSE,
            ArtifactState.ABSENT,
            WorkspaceChange.UNCHANGED,
            True,
            PostconditionState.UNKNOWN,
        ),
        PatchDiagnostic(
            ErrorStage.PLANNING,
            PatchErrorCode.INVALID_PATCH,
            Retryability.NOT_RETRYABLE,
        ),
    )


def _result_mapping(result: ToolCallResult) -> dict[str, ToolValue]:
    """Return one public tool result after its JSON shape is established."""
    assert isinstance(result.result, dict)
    return result.result


class _Phase15Settlement:
    """Expose deterministic settlement only through the production protocol."""

    def __init__(self, service: "_Phase15DurableService") -> None:
        """Bind this port to one deterministic durable service."""
        self._service = service

    def inspect(
        self, handle: PatchInvocationHandle
    ) -> Future["PatchInvocationOutcome"]:
        """Return the host-owned current result or pending envelope."""
        assert type(handle) is PatchInvocationHandle
        outcome = self._service.current_outcome
        assert outcome is not None
        return _resolved_future(outcome)

    def await_terminal(
        self, handle: PatchInvocationHandle, pending: PatchPending
    ) -> Future[PatchResult]:
        """Settle one exact pending request through durable test state."""
        assert type(handle) is PatchInvocationHandle
        return self._service.await_terminal(pending)


@dataclass(frozen=True, slots=True)
class _Phase15DurableClaim:
    """Retain the exact real durable owner for one service invocation."""

    identity: DurableRequestIdentity
    plan: DurablePlanReference
    lease: DurableCommitLease
    correlation_id: PatchObserverCorrelationId
    artifact_id: PatchArtifactId | None


class _Phase15DurableService:
    """Exercise public JSON handling against real durable reservations."""

    def __init__(
        self,
        *,
        pending: bool,
        ambiguous_after_reservation: bool = False,
        hold_pending: bool = False,
        initial_document: str = "before\n",
        parser_limits: PatchInputLimits | None = None,
        planning_delay_ticks: int = 0,
        approval_delay_ticks: int = 0,
        commit_delay_ticks: int = 0,
        document_mode: FileMode = FileMode(0o644),
        planner_limits: PlannerLimits | None = None,
        additional_files: tuple[PlannerFile, ...] = (),
        max_in_flight: int = 64,
        durable_step_count: int = 1,
        local_executor: (
            Callable[[OperationType, bytes], Awaitable[PatchResult]] | None
        ) = None,
    ) -> None:
        """Initialize one isolated durable backend and pending policy."""
        self._approval_authority = HmacDurableApprovalAuthority.random()
        self._backend = InMemoryDurablePatchBackend(
            approval_verifier=self._approval_authority
        )
        self.store = InMemoryDurablePatchStore(self._backend)
        self._pending_mode = pending
        self._ambiguous_after_reservation = ambiguous_after_reservation
        self._hold_pending = hold_pending
        self._local_executor = local_executor
        self._parser_limits = parser_limits or PatchInputLimits(
            max_raw_bytes=128
        )
        if (
            type(planning_delay_ticks) is not int
            or type(approval_delay_ticks) is not int
            or type(commit_delay_ticks) is not int
            or min(
                planning_delay_ticks,
                approval_delay_ticks,
                commit_delay_ticks,
            )
            < 0
            or type(document_mode) is not FileMode
            or planner_limits is not None
            and type(planner_limits) is not PlannerLimits
            or type(additional_files) is not tuple
            or any(type(item) is not PlannerFile for item in additional_files)
            or type(max_in_flight) is not int
            or max_in_flight < 1
            or type(durable_step_count) is not int
            or durable_step_count < 1
        ):
            raise ValueError("phase15 durable service limits are invalid")
        self._planning_delay_ticks = planning_delay_ticks
        self._approval_delay_ticks = approval_delay_ticks
        self._commit_delay_ticks = commit_delay_ticks
        self._document_mode = document_mode
        self._planner_limits = planner_limits
        self._additional_files = additional_files
        self._max_in_flight = max_in_flight
        self._durable_step_count = durable_step_count
        self._in_flight = 0
        self._capacity_lock = Lock()
        self._runtime_binding: PatchRuntimeBinding | None = None
        self.current_outcome: PatchResult | PatchPending | None = None
        self._terminal: PatchResult | None = None
        self.events: list[PatchLifecycleEvent] = []
        self.invocations = 0
        self.approvals = 0
        self.cleanup_count = 0
        self.document = initial_document
        self.last_planned_snapshot_bytes = 0
        self.parse_attempts = 0
        self.limit_timeouts: list[str] = []
        self.settlement = _Phase15Settlement(self)
        self._activation_observer: PatchActivationObserver | None = None
        self._pending_future: Future[PatchResult] | None = None
        self._claims: dict[PatchRequestId, _Phase15DurableClaim] = {}
        self._retransmissions: dict[
            PatchObserverCorrelationId,
            tuple[PatchRequestId, AlgorithmDigest],
        ] = {}
        self.last_access: DurableRequestAccess | None = None

    async def _reserve_capacity(self) -> bool:
        """Reserve one bounded pre-claim slot without a durable side effect."""
        async with self._capacity_lock:
            if self._in_flight >= self._max_in_flight:
                return False
            self._in_flight += 1
            return True

    async def _release_capacity(self) -> None:
        """Release one exact terminal or rejected pre-claim slot."""
        async with self._capacity_lock:
            if self._in_flight < 1:
                raise RuntimeError("phase15 capacity release is invalid")
            self._in_flight -= 1

    def bind_runtime(self, binding: PatchRuntimeBinding) -> None:
        """Retain the exact loader-bound policy and limit authority."""
        existing = self._runtime_binding
        if (
            type(binding) is not PatchRuntimeBinding
            or binding.service is not self
        ):
            raise RuntimeError("phase15 runtime binding is invalid")
        if existing is not None and (
            binding.scope != existing.scope
            or binding.handshake != existing.handshake
            or binding.policy != existing.policy
            or binding.approval != existing.approval
            or binding.coordinator != existing.coordinator
            or binding.persistence != existing.persistence
        ):
            raise RuntimeError("phase15 runtime binding is invalid")
        self._runtime_binding = binding

    async def _await_limit(
        self, phase: str, delay_ticks: int, limit: DurationTicks
    ) -> None:
        """Apply one trusted finite phase budget before durable dispatch."""
        if delay_ticks == 0:
            return
        try:
            await wait_for(
                sleep(delay_ticks / 1_000), timeout=limit.value / 1_000
            )
        except TimeoutError:
            self.limit_timeouts.append(phase)
            raise

    async def _authorize_candidate(
        self,
        operation: OperationType,
        candidate: PlannerCandidate,
        limits: PatchLimits,
    ) -> None:
        """Apply real policy capabilities and post-plan resource limits."""
        binding = self._runtime_binding
        if binding is None:
            raise RuntimeError("phase15 runtime binding is unavailable")
        lineages = candidate.lineages
        paths = tuple(
            sorted(
                {
                    path
                    for lineage in lineages
                    for path in (
                        lineage.initial.path,
                        lineage.final.path,
                        lineage.source_path,
                        lineage.destination_path,
                    )
                    if path is not None
                },
                key=lambda value: value.value,
            )
        )
        effects = frozenset(
            capability
            for lineage in lineages
            for capability in lineage.capabilities
        )
        preflight = await PolicyAuthorizer(
            binding.policy
        ).authorize_preinspection(
            PreflightRequest(
                operation,
                paths,
                effects,
                frozenset(paths),
                compose_limits(limits, limits, limits, limits, limits),
            )
        )
        await PolicyAuthorizer(binding.policy).authorize_final(
            preflight, candidate, binding.handshake
        )

    def set_activation_observer(
        self, observer: PatchActivationObserver
    ) -> None:
        """Attach the one activation observer owned by the loaded host."""
        if self._activation_observer is not None or not isinstance(
            observer, PatchActivationObserver
        ):
            raise RuntimeError("phase15 activation observer is invalid")
        self._activation_observer = observer

    async def _claim_durable_owner(
        self,
        identity: DurableRequestIdentity,
        request_id: PatchRequestId,
        correlation_id: PatchObserverCorrelationId,
        canonical_digest: AlgorithmDigest,
    ) -> _Phase15DurableClaim:
        """Persist a plan and claim its real owner before observer binding."""
        reservation = await self.store.reserve(
            identity, canonical_digest, request_id
        )
        assert (
            not reservation.replayed and reservation.request_id == request_id
        )
        suffix = request_id.value.removeprefix("request_")
        plan = DurablePlanReference(
            PatchPlanId.new(),
            canonical_digest,
            AlgorithmDigest.from_bytes(
                b"phase15-fingerprint" + canonical_digest.value.encode()
            ),
            AlgorithmDigest.from_bytes(
                b"phase15-review" + canonical_digest.value.encode()
            ),
            PatchContextId("context_" + "f" * 16),
            PatchWorkspaceId("workspace_" + "f" * 16),
            PatchDomainId("domain_" + suffix),
            tuple(
                DurableStepBinding(PatchStepId.new(), PatchLineageId.new())
                for _ in range(self._durable_step_count)
            ),
        )
        await self.store.persist_plan(reservation, plan)
        approval = self._approval_authority.seal(
            DurableApproval(
                PatchGrantId.new(),
                PatchApprovalId.new(),
                identity,
                canonical_digest,
                plan.plan_id,
                plan.fingerprint_digest,
                plan.review_digest,
                plan.context_id,
                plan.workspace_id,
                plan.domain_id,
                "phase15-public-v1",
                PolicyBrokerId("phase15-public-broker"),
                PolicyReviewerRole("phase15-public-reviewer"),
                (PatchPrincipalId("phase15-public-principal"),),
                ExpiryTick(100),
                b"\x00" * 32,
            )
        )
        artifact_id = (
            PatchArtifactId.new() if self._local_executor is not None else None
        )
        claim = await self.store.claim_commit(
            reservation,
            plan,
            approval,
            PatchCommitOwnerId.new(),
            ExpiryTick(1),
            DurationTicks(50),
            () if artifact_id is None else (artifact_id,),
        )
        assert (
            claim.state is DurableCommitClaimState.OWNER
            and claim.lease is not None
        )
        self.last_access = DurableRequestAccess(request_id, identity)
        observer = self._activation_observer
        if observer is None:
            raise RuntimeError("phase15 activation observer is unavailable")
        await observer.bind_durable_commit(claim.lease)
        return _Phase15DurableClaim(
            identity, plan, claim.lease, correlation_id, artifact_id
        )

    async def _settle_durable_claim(
        self, claim: _Phase15DurableClaim, result: PatchResult
    ) -> PatchResult:
        """Settle the exact owner then release it from activation state."""
        access = DurableRequestAccess(result.request_id, claim.identity)
        artifact = None
        if claim.artifact_id is not None:
            artifact_state = {
                ArtifactState.ABSENT: DurableArtifactState.NOT_CREATED,
                ArtifactState.CLEANED: DurableArtifactState.REMOVED,
                ArtifactState.LEAKED: DurableArtifactState.LEAKED,
                ArtifactState.UNKNOWN: DurableArtifactState.UNKNOWN,
            }.get(result.truth.artifact_state)
            if artifact_state is None:
                raise RuntimeError("phase15 durable artifact state is invalid")
            artifact = DurableArtifactObservation(
                "phase15-local-artifact",
                claim.artifact_id,
                artifact_state,
            )
        report = WorkerReport(
            WorkerState.SETTLED,
            SettlementJournal(
                tuple(
                    JournalStep(
                        step.step_id,
                        step.lineage_id,
                        CommitStepState.COMMITTED,
                    )
                    for step in claim.plan.steps
                ),
                (
                    ()
                    if artifact is None
                    else (
                        ArtifactJournal(
                            artifact.worker_identifier,
                            result.truth.artifact_state,
                        ),
                    )
                ),
                result.truth.postcondition,
            ),
        )
        terminal = await DurablePatchReconciler(self.store).reconcile(
            access,
            claim.lease,
            report,
            result,
            claim.correlation_id,
            ExpiryTick(2),
            artifacts=() if artifact is None else (artifact,),
        )
        if not isinstance(terminal, PatchResult):
            raise RuntimeError("phase15 durable settlement remained pending")
        observer = self._activation_observer
        if observer is None:
            raise RuntimeError("phase15 activation observer is unavailable")
        await observer.release_durable_commit(claim.lease)
        self._claims.pop(result.request_id, None)
        return terminal

    async def invoke(
        self,
        operation: OperationType,
        raw_arguments: bytes,
        capability: PatchInvocationCapability,
        request_id: PatchRequestId,
        correlation_id: PatchObserverCorrelationId,
    ) -> PatchResult | PatchPending:
        """Parse canonical provider JSON before reserving its durable tuple."""
        assert type(capability) is PatchInvocationCapability
        kind = (
            RawPatchInputKind.EDIT_JSON
            if operation is OperationType.EDIT
            else RawPatchInputKind.APPLY_JSON
        )
        try:
            request = PatchRequestParser(self._parser_limits).parse(
                RawPatchIngress(
                    RawProviderProfile("phase15-public-provider"),
                    RawToolCallId(correlation_id.value),
                    kind,
                    RawPatchInputState.COMPLETE,
                    raw_arguments,
                )
            )
        except PatchInputError:
            self._terminal = _phase15_rejected_result(
                request_id, PatchPlanId.new()
            )
            self.current_outcome = self._terminal
            self._emit(
                request_id,
                correlation_id,
                LifecyclePhase.REQUEST_COMPLETED,
            )
            return self._terminal
        self.parse_attempts += 1
        identity = DurableRequestIdentity(
            PatchTenantId("phase15-public-tenant"),
            PatchPrincipalId("phase15-public-principal"),
            PatchExecutionId("execution_" + "f" * 16),
            PolicyRouteId("phase15-public-route"),
            RetransmissionKey(correlation_id.value),
        )
        canonical_digest = AlgorithmDigest.from_bytes(raw_arguments)
        retransmission = self._retransmissions.get(correlation_id)
        if retransmission is not None:
            original_request, original_digest = retransmission
            if (
                original_request != request_id
                or original_digest != canonical_digest
                or self.current_outcome is None
            ):
                raise RuntimeError(
                    "phase15 retransmission identity is invalid"
                )
            return self.current_outcome
        if self._local_executor is None:
            binding = self._runtime_binding
            if binding is None:
                raise RuntimeError("phase15 runtime binding is unavailable")
            limits = binding.scope.limits
            try:
                await self._await_limit(
                    "planning",
                    self._planning_delay_ticks,
                    limits.planning_duration,
                )
                snapshot = self.document.encode("utf-8")
                self.last_planned_snapshot_bytes = len(snapshot)
                planner_limits = self._planner_limits or PlannerLimits(
                    max_file_snapshot_bytes=limits.snapshot_bytes.value,
                    max_snapshot_bytes=limits.snapshot_bytes.value,
                    max_file_proposed_bytes=limits.proposed_bytes.value,
                    max_proposed_bytes=limits.proposed_bytes.value,
                    max_changed_bytes=limits.proposed_bytes.value,
                    max_match_candidates=limits.operation_count.value,
                    max_diff_work_bytes=1_048_576,
                    max_diff_bytes=limits.review_diff_bytes.value,
                    max_memory_bytes=100_000_000,
                )
                candidate = plan(
                    request,
                    _workspace(
                        _file("note.txt", snapshot, self._document_mode),
                        *self._additional_files,
                    ),
                    planner_limits,
                )
                await self._authorize_candidate(operation, candidate, limits)
                await self._await_limit(
                    "approval",
                    self._approval_delay_ticks,
                    limits.approval_duration,
                )
                await self._await_limit(
                    "commit",
                    self._commit_delay_ticks,
                    limits.commit_duration,
                )
            except (PlannerError, PolicyError, TimeoutError):
                self._terminal = _phase15_rejected_result(
                    request_id, PatchPlanId.new()
                )
                self.current_outcome = self._terminal
                self._emit(
                    request_id,
                    correlation_id,
                    LifecyclePhase.REQUEST_COMPLETED,
                )
                return self._terminal
        if not await self._reserve_capacity():
            rejected = _phase15_rejected_result(request_id, PatchPlanId.new())
            if not isinstance(self.current_outcome, PatchPending):
                self._terminal = rejected
                self.current_outcome = rejected
            self._emit(
                request_id,
                correlation_id,
                LifecyclePhase.REQUEST_COMPLETED,
            )
            return rejected
        try:
            claim = await self._claim_durable_owner(
                identity,
                request_id,
                correlation_id,
                canonical_digest,
            )
        except BaseException:
            await self._release_capacity()
            raise
        self._claims[request_id] = claim
        self._retransmissions[correlation_id] = (
            request_id,
            canonical_digest,
        )
        self.invocations += 1
        if self._local_executor is None:
            self._terminal = _phase15_result(request_id, claim.plan.plan_id)
        else:
            local_result = await self._local_executor(operation, raw_arguments)
            self._terminal = PatchResult(
                1,
                request_id,
                claim.plan.plan_id,
                LifecyclePhase.REQUEST_COMPLETED,
                local_result.status,
                local_result.truth,
                local_result.diagnostic,
            )
        if self._ambiguous_after_reservation:
            self.current_outcome = await self._settle_durable_claim(
                claim, self._terminal
            )
            await self._release_capacity()
            self._emit(
                request_id, correlation_id, LifecyclePhase.REQUEST_COMPLETED
            )
            raise RuntimeError("phase15 ambiguous durable response")
        if not self._pending_mode:
            self.current_outcome = await self._settle_durable_claim(
                claim, self._terminal
            )
            await self._release_capacity()
            self._emit(
                request_id, correlation_id, LifecyclePhase.REQUEST_COMPLETED
            )
            return self._terminal
        pending = PatchPending(
            1,
            PatchPendingOperationId("pending_" + "f" * 16),
            request_id,
            correlation_id,
            LifecyclePhase.SETTLEMENT_PENDING,
        )
        await self.store.suspend(
            claim.lease,
            DurablePendingRequest(
                pending.pending_operation_id,
                pending.correlation_id,
                DurationTicks(10),
            ),
            ExpiryTick(2),
        )
        observer = self._activation_observer
        if observer is None:
            raise RuntimeError("phase15 activation observer is unavailable")
        await observer.retain_durable_commit(claim.lease)
        self.current_outcome = pending
        self._emit(
            request_id, correlation_id, LifecyclePhase.SETTLEMENT_PENDING
        )
        return pending

    async def settle(self, pending: PatchPending) -> PatchResult:
        """Settle one pending provider call after bounded approval evidence."""
        assert self.current_outcome == pending and self._terminal is not None
        claim = self._claims[pending.request_id]
        self.approvals += 1
        self.current_outcome = await self._settle_durable_claim(
            claim, self._terminal
        )
        await self._release_capacity()
        self.document = "after\n"
        self._emit(
            self._terminal.request_id,
            pending.correlation_id,
            LifecyclePhase.REQUEST_COMPLETED,
        )
        self.cleanup_count += 1
        if (
            self._hold_pending
            and self._pending_future is not None
            and not self._pending_future.done()
        ):
            self._pending_future.set_result(self._terminal)
        return self._terminal

    def await_terminal(self, pending: PatchPending) -> Future[PatchResult]:
        """Return retained pending settlement or the exact terminal result."""
        if not self._hold_pending:
            future: Future[PatchResult] = get_running_loop().create_future()
            create_task(self._settle_pending(pending, future))
            return future
        if self._pending_future is None:
            self._pending_future = get_running_loop().create_future()
        return self._pending_future

    async def _settle_pending(
        self, pending: PatchPending, future: Future[PatchResult]
    ) -> None:
        """Bridge a real durable settlement into the sealed Future port."""
        try:
            result = await self.settle(pending)
            if not future.done():
                future.set_result(result)
        except Exception as error:
            future.set_exception(error)

    async def review(self, handle: PatchInvocationHandle) -> dict[str, str]:
        """Return a fixed non-content review projection for the host."""
        assert type(handle) is PatchInvocationHandle
        return {"kind": "phase15_durable_review"}

    async def approve(
        self, handle: PatchInvocationHandle
    ) -> PatchResult | PatchPending:
        """Return only the current durable service state for one handle."""
        assert type(handle) is PatchInvocationHandle
        assert self.current_outcome is not None
        return self.current_outcome

    async def subscribe(
        self, handle: PatchInvocationHandle
    ) -> AsyncIterator[PatchLifecycleEvent]:
        """Yield only bounded lifecycle labels for the current request."""
        assert type(handle) is PatchInvocationHandle
        for event in self.events:
            yield event

    def _emit(
        self,
        request_id: PatchRequestId,
        correlation_id: PatchObserverCorrelationId,
        lifecycle: LifecyclePhase,
    ) -> None:
        """Append one content-free lifecycle observation in sequence."""
        self.events.append(
            PatchLifecycleEvent(
                1,
                PatchEventId.new(),
                PatchObserverId.new(),
                correlation_id,
                request_id,
                SequenceNumber(len(self.events) + 1),
                lifecycle,
            )
        )


@dataclass(frozen=True, slots=True)
class _Phase15PublicBinder:
    """Return one complete concrete local binding for public tool tests."""

    service: _Phase15DurableService
    limits: PatchLimits = dataclass_field(default_factory=_phase15_limits)
    policy: TrustedPatchPolicy = dataclass_field(
        default_factory=_phase15_policy
    )

    async def bind(self) -> PatchRuntimeBinding:
        """Return one authenticated local binding over the service store."""
        identity = TargetIdentity(
            PatchContextId("context_" + "f" * 16),
            PatchWorkspaceId("workspace_" + "f" * 16),
            PatchDomainId("domain_" + "f" * 16),
            PatchTargetId("target_" + "f" * 16),
            PatchProtocolId("protocol_" + "f" * 16),
            "phase15-filesystem",
            "phase15-mount",
            "phase15-public-v1",
            "phase15-persistent-lease",
            PatchApprovalId("approval_" + "f" * 16),
        )
        scope = ResolvedMutationScope(
            ContextKind.LOCAL,
            identity,
            None,
            self.limits,
            frozenset(Capability),
            frozenset(TargetPrimitive),
        )
        handshake = TargetHandshake(
            identity,
            frozenset(TargetPrimitive),
            (),
            platform=LocalPlatformProfile.DARWIN,
        )
        binding = PatchRuntimeBinding(
            scope,
            handshake,
            self.policy,
            PatchApprovalBinding(True),
            PatchCoordinatorBinding(True, self.service.store),
            PatchPersistenceBinding(True, self.service.store),
            self.service,
        )
        self.service.bind_runtime(binding)
        return binding


async def _phase15_public_bundle(
    service: _Phase15DurableService,
    *,
    ordinary_toolsets: tuple[ToolSet, ...] = (),
    limits: PatchLimits | None = None,
    policy: TrustedPatchPolicy | None = None,
) -> PatchToolManagerBundle:
    """Build one public ToolManager through the production patch loader."""
    return await PatchToolLoader(
        _Phase15PublicBinder(
            service,
            limits or _phase15_limits(),
            policy or _phase15_policy(),
        ),
        activated_patch_test_profile(),
    ).load(
        enable_tools=["patch.edit", "patch.apply", "phase15.read"],
        ordinary_toolsets=ordinary_toolsets,
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES
        ),
    )


def _ingress(
    payload: bytes,
    kind: RawPatchInputKind = RawPatchInputKind.APPLY_JSON,
) -> RawPatchIngress:
    """Return one closed raw ingress with no target authority."""
    return RawPatchIngress(
        RawProviderProfile("phase15-fuzz"),
        RawToolCallId("phase15-fuzz"),
        kind,
        RawPatchInputState.COMPLETE,
        payload,
    )


def _edit(old: str, new: str, path: str = "note.txt") -> CanonicalPatchRequest:
    """Parse one exact JSON edit request for a pure planner workload."""
    payload = dumps(
        {
            "path": path,
            "edits": [{"old_text": old, "new_text": new}],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return PatchRequestParser().parse(
        _ingress(payload, RawPatchInputKind.EDIT_JSON)
    )


def _apply(lines: tuple[str, ...]) -> CanonicalPatchRequest:
    """Parse one exact Version 1 apply document for a pure planner workload."""
    payload = dumps(
        {"patch": "\n".join(lines)},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return PatchRequestParser().parse(_ingress(payload))


def _file(
    path: str,
    value: bytes,
    mode: FileMode = FileMode(0o644),
) -> PlannerFile:
    """Return one abstract rooted regular-file snapshot."""
    view = LogicalText.from_bytes(value)
    return PlannerFile(
        LogicalPath(path),
        SourceBytes(value),
        MetadataProfile(
            mode,
            view.has_bom,
            (
                view.representation.value
                if view.representation.value != "none"
                else "lf"
            ),
        ),
        LogicalPath(path.rsplit("/", 1)[0]) if "/" in path else None,
        "phase15-mount",
        "phase15-identity-" + path,
    )


def _workspace(*files: PlannerFile) -> PlannerWorkspace:
    """Return one fixed abstract workspace without unrelated snapshots."""
    return PlannerWorkspace(tuple(files), frozenset())


def _fuzz_cases() -> tuple[_FuzzCase, ...]:
    """Decode the finite corpus and verify its canonical replay fingerprint."""
    value = loads(_FUZZ_FIXTURE.read_text(encoding="utf-8"))
    assert value["schema_version"] == 1
    assert type(value["seed"]) is int
    assert type(value["cases"]) is list
    canonical = dumps(
        {"seed": value["seed"], "cases": value["cases"]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert value["corpus_sha256"] == sha256(canonical).hexdigest()
    cases: list[_FuzzCase] = []
    for item in value["cases"]:
        assert type(item) is dict
        identifier = item.get("id")
        category = item.get("category")
        payload_hex = item.get("payload_hex")
        expected = item.get("expected")
        assert type(identifier) is str
        assert type(category) is str
        assert type(payload_hex) is str
        assert type(expected) is str
        cases.append(
            _FuzzCase(
                identifier,
                category,
                bytes.fromhex(payload_hex),
                expected,
            )
        )
    return tuple(cases)


def test_patch_phase_15_fixed_seed_fuzz_corpus_is_bounded_and_replayable() -> (
    None
):
    """Replay every malformed category through its owning typed boundary."""
    parser = PatchRequestParser(PatchInputLimits(max_raw_bytes=1024))
    cases = _fuzz_cases()
    assert len(cases) == 11
    order = list(cases)
    Random(20260829).shuffle(order)
    observed: list[str] = []
    for case in order:
        observed.append(case.identifier)
        match case.category:
            case "grammar" | "json" | "unicode" | "declaration":
                if case.expected == "accept":
                    assert parser.parse(_ingress(case.payload))
                else:
                    with pytest.raises(PatchInputError) as error:
                        parser.parse(_ingress(case.payload))
                    assert error.value.code in PatchInputErrorCode
            case "path":
                with pytest.raises(PatchInputError) as error:
                    parser.parse(
                        _ingress(case.payload, RawPatchInputKind.EDIT_JSON)
                    )
                assert error.value.code is PatchInputErrorCode.PATH
            case "matching":
                with pytest.raises(PlannerError):
                    find_match(
                        LogicalText.from_bytes(case.payload), "alpha", 8
                    )
            case "lineage":
                request = parser.parse(_ingress(case.payload))
                with pytest.raises(PlannerError):
                    plan(request, _workspace())
            case "target_rpc":
                decoded = loads(case.payload)
                assert (
                    canonical_sandbox_plan_bytes(decoded)
                    == b'{"request_id":"fuzz","sequence":1}'
                )
            case "event":
                assert LifecyclePhase(case.payload.decode("ascii")) is (
                    LifecyclePhase.SETTLEMENT_PENDING
                )
            case "result":
                code = PatchErrorCode(case.payload.decode("ascii"))
                assert coarsen_error_code(code, Audience.PUBLIC) is (
                    PatchErrorCode.PATH_DENIED
                )
            case "protocol":
                profile = PatchProtocolProfile(
                    PatchProtocolSurface(case.payload.decode("ascii"))
                )
                assert not profile.active
            case _:
                raise AssertionError("unrecognized fixed fuzz corpus category")
    assert set(observed) == {case.identifier for case in cases}


def test_patch_phase_15_n_minus_one_n_n_plus_one_activation_bounds() -> None:
    """Retain each durable owner and block the next bound operation."""

    async def scenario() -> tuple[PatchActivationLease, PatchActivationLease]:
        production = build_patch_production_manifest()
        profile = replace(
            production.profiles[0],
            proofs=PatchProfileProofs(
                context=True,
                platform=True,
                filesystem=True,
                target=True,
                protocol=True,
                policy=True,
                approval=True,
                persistence=True,
                surface=True,
                provider_codec=True,
            ),
            state=PatchProfileState.SELECTED,
            selection_rationale="Phase 15 activation boundary integration.",
        )
        manifest = _manifest(
            sources=production.sources,
            schemas=production.schemas,
            protocols=production.protocols,
            profiles=(profile,),
        )
        verifier = _build_activation_verifier(
            manifest,
            _new_activation_authority(b"a" * 32),
            production=False,
        )
        registry = PatchActivationRegistry(
            manifest,
            verifier,
            PatchActivationLimits(
                max_active_profiles=1, max_operations_per_profile=1
            ),
        )
        record = PatchActivationRuntimeRecord(
            profile.key,
            "a" * 64,
            InMemoryDurablePatchStore(InMemoryDurablePatchBackend()),
        )
        receipt = verifier._runtime_receipt(record)
        assert receipt is not None
        lease = await registry.activate(receipt)
        durable = PatchActivationDurableOperation(
            PatchRequestId.new(),
            PatchCommitOwnerId("owner_" + "a" * 16),
            SequenceNumber(1),
        )
        await registry.bind_operation(
            receipt, durable, PatchActivationOperationState.IN_FLIGHT
        )
        with pytest.raises(PatchActivationError):
            await registry.bind_operation(
                receipt,
                PatchActivationDurableOperation(
                    PatchRequestId.new(),
                    PatchCommitOwnerId("owner_" + "b" * 16),
                    SequenceNumber(2),
                ),
                PatchActivationOperationState.IN_FLIGHT,
            )
        await registry.deactivate(lease.key)
        with pytest.raises(PatchActivationError):
            await registry.activate(receipt)
        await registry.release_operation(lease.key, durable, lease.epoch)
        second = await registry.activate(receipt)
        assert second.epoch == lease.epoch + 1
        return lease, second

    lease, second = run(scenario())
    assert lease.epoch == 1
    assert second.epoch == lease.epoch + 1


def test_patch_phase_15_loader_requires_retained_activation_runtime() -> None:
    """Deny boolean-only admission and stale same-host reconstruction."""

    class UnexpectedBinder:
        """Fail if a boolean-only profile reaches runtime probing."""

        async def bind(self) -> PatchRuntimeBinding:
            """Refuse the unactivated loader path."""
            raise AssertionError("boolean profile reached patch runtime")

    class ProtocolShapedFactory:
        """Imitate the former structural factory protocol without authority."""

        async def activate(self, _binding: PatchRuntimeBinding) -> None:
            """Offer no concrete factory identity."""
            return None

    sealed = patch_test_activation_factory()
    forged = object.__new__(PatchActivationRuntimeFactory)
    for name in ("_manifest", "_verifier", "_issuer"):
        object.__setattr__(forged, name, getattr(sealed, name))
    with pytest.raises(PatchToolError):
        PatchTestHostProfile(
            enabled=True,
            authenticated=True,
            activation_factory=ProtocolShapedFactory(),
        )
    with pytest.raises(PatchToolError):
        PatchTestHostProfile(
            enabled=True,
            authenticated=True,
            activation_factory=forged,
        )

    async def scenario() -> None:
        inactive = await PatchToolLoader(
            UnexpectedBinder(),
            PatchTestHostProfile(enabled=True, authenticated=True),
        ).load(enable_tools=["patch.edit"])
        assert inactive.toolset is None

        service = _Phase15DurableService(pending=False)
        profile = activated_patch_test_profile()
        loader = PatchToolLoader(_Phase15PublicBinder(service), profile)
        substituted = await loader.load(
            enable_tools=["patch.edit"],
            activation_factory=patch_test_activation_factory(),
        )
        assert substituted.toolset is None
        active = await loader.load(enable_tools=["patch.edit"])
        assert active.toolset is not None
        active.toolset._snapshot = replace(
            active.toolset._snapshot, stale=True
        )
        rebuilt = await loader.rebuild_if_stale(
            active.toolset,
            enable_tools=["patch.edit"],
        )
        assert rebuilt.toolset is None
        assert rebuilt.runtime_binding is not None

        forged_service = _Phase15DurableService(pending=False)
        forged_loader = PatchToolLoader(
            _Phase15PublicBinder(forged_service),
            activated_patch_test_profile(),
        )
        forged_active = await forged_loader.load(enable_tools=["patch.edit"])
        assert forged_active.toolset is not None
        forged_active.toolset._snapshot = replace(
            forged_active.toolset._snapshot, stale=True
        )
        forged_active.toolset._activation_factory = forged
        forged_rebuild = await forged_loader.rebuild_if_stale(
            forged_active.toolset,
            enable_tools=["patch.edit"],
        )
        assert forged_rebuild.toolset is None
        assert forged_rebuild.runtime_binding is None

    run(scenario())


def test_patch_e2e_041_incomplete_default_profiles_load_no_authority() -> None:
    """Deny generated dormant profiles through production admission."""
    manifest = build_patch_production_manifest()
    default = manifest.profiles[0]
    assert default.state is PatchProfileState.INCOMPLETE
    inactive = replace(
        default,
        key=replace(
            default.key,
            context=ContextKind.LOCAL,
            platform=PatchActivationPlatform.MACOS,
        ),
        selection_rationale="Generated inactive Phase 15 profile.",
    )
    unsupported = replace(
        default,
        key=replace(
            default.key,
            context=ContextKind.LOCAL,
            platform=PatchActivationPlatform.LINUX,
        ),
        state=PatchProfileState.NOT_SELECTED,
        selection_rationale="Generated unsupported Phase 15 profile.",
    )
    not_selected = replace(
        default,
        state=PatchProfileState.NOT_SELECTED,
        selection_rationale="Generated not-selected Phase 15 profile.",
    )
    inventory = (
        ("default", default),
        ("inactive", inactive),
        ("unsupported", unsupported),
        ("not-selected", not_selected),
    )
    assert {profile.state for _, profile in inventory} == {
        PatchProfileState.INCOMPLETE,
        PatchProfileState.NOT_SELECTED,
    }
    edit_parameters = PATCH_EDIT_SCHEMA["function"]["parameters"]
    apply_parameters = PATCH_APPLY_SCHEMA["function"]["parameters"]
    assert edit_parameters["additionalProperties"] is False
    assert apply_parameters["additionalProperties"] is False
    edits = edit_parameters["properties"]["edits"]
    assert edits["items"]["additionalProperties"] is False
    parser = PatchRequestParser()
    model_controlled_fields: tuple[tuple[str, object], ...] = (
        ("workspace", "model-controlled"),
        ("cwd", "/private/model-controlled"),
        ("backend", "model-controlled"),
        (
            "capabilities",
            [
                Capability.READ_FOR_MUTATION.value,
                Capability.OBSERVE_MUTATION_PRECONDITIONS.value,
                Capability.UPDATE_EXECUTABLE.value,
            ],
        ),
        ("approval", "model-controlled"),
        ("overwrite", True),
        ("policy", "model-controlled"),
        ("limits", {"input_bytes": 1_000_000}),
        ("disclosure", "logical_paths"),
        ("validators", ["model-controlled"]),
    )
    for field, value in model_controlled_fields:
        raw = dumps(
            {
                "path": "note.txt",
                "edits": [{"old_text": "before", "new_text": "after"}],
                field: value,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        with pytest.raises(PatchInputError):
            parser.parse(_ingress(raw, RawPatchInputKind.EDIT_JSON))
    with pytest.raises(PatchInputError):
        parser.parse(
            _ingress(
                b'{"path":"note.txt","path":"other.txt","edits":['
                b'{"old_text":"before","new_text":"after"}]}',
                RawPatchInputKind.EDIT_JSON,
            )
        )

    class ProfileBoundBinder:
        """Expose one real binding to every loader admission attempt."""

        def __init__(self, service: _Phase15DurableService) -> None:
            """Bind one inert durable service through the normal handshake."""
            self.service = service
            self.calls = 0
            self.binding: PatchRuntimeBinding | None = None

        async def bind(self) -> PatchRuntimeBinding:
            """Expose the normal trusted handshake only if required."""
            self.calls += 1
            self.binding = await _Phase15PublicBinder(self.service).bind()
            return self.binding

    def denied_policy(capability: Capability) -> TrustedPatchPolicy:
        """Deny one independent trusted effect without model input."""
        policy = _phase15_policy()
        rule = policy.rules[0]
        return replace(
            policy,
            rules=(
                replace(
                    rule,
                    modes=tuple(
                        (
                            CapabilityMode(item.value, ApprovalMode.DENY)
                            if item.value is capability
                            else item
                        )
                        for item in rule.modes
                    ),
                ),
            ),
        )

    async def scenario() -> None:
        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="ai://local/phase15-incomplete",
            engine_config={},
            tools=["patch.edit", "patch.apply"],
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=False,
        )
        stack = AsyncExitStack()
        with (
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch(
                "avalan.agent.loader.ModelManager", return_value=MagicMock()
            ),
            patch(
                "avalan.agent.loader.DefaultOrchestrator", return_value=None
            ) as orchestrator,
            patch(
                "avalan.agent.loader.EventManager", return_value=MagicMock()
            ),
            patch("avalan.agent.loader.HAS_GRAPH_DEPENDENCIES", False),
            patch("avalan.agent.loader.HAS_CODE_DEPENDENCIES", False),
            patch("avalan.agent.loader.HAS_BROWSER_DEPENDENCIES", False),
            patch(
                "avalan.agent.loader.MathToolSet",
                side_effect=lambda *, namespace: ToolSet(
                    namespace=namespace, tools=[]
                ),
            ),
            patch(
                "avalan.agent.loader.MemoryToolSet",
                side_effect=lambda _memory, *, namespace: ToolSet(
                    namespace=namespace, tools=[]
                ),
            ),
        ):
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            operation_requests = (
                (
                    "update",
                    "patch.edit",
                    (
                        b'{"path":"note.txt","edits":['
                        b'{"old_text":"before","new_text":"after"}]}'
                    ),
                ),
                (
                    "create",
                    "patch.apply",
                    (
                        b'{"patch":"*** Begin Patch v1\\n*** Add File: '
                        b'created.txt\\n+created\\n*** End Patch"}'
                    ),
                ),
                (
                    "delete",
                    "patch.apply",
                    (
                        b'{"patch":"*** Begin Patch v1\\n*** Delete File: '
                        b'note.txt\\n*** End Patch"}'
                    ),
                ),
                (
                    "move",
                    "patch.apply",
                    (
                        b'{"patch":"*** Begin Patch v1\\n*** Update File: '
                        b"note.txt\\n*** Move to: moved.txt\\n@@\\n-before\\n"
                        b'+after\\n*** End Patch"}'
                    ),
                ),
            )
            assert tuple(item[0] for item in operation_requests) == (
                "update",
                "create",
                "delete",
                "move",
            )
            for profile_label, generated_profile in inventory:
                assert (
                    generated_profile.state is not PatchProfileState.SELECTED
                )
                factory = (
                    None
                    if profile_label == "default"
                    else patch_test_activation_factory((generated_profile,))
                )
                for selected_tools in (
                    ["patch.edit"],
                    ["patch.apply"],
                    ["patch.edit", "patch.apply"],
                ):
                    for host_profile, expected_binds in (
                        (PatchTestHostProfile(), 0),
                        (
                            PatchTestHostProfile(
                                enabled=True, authenticated=True
                            ),
                            1,
                        ),
                    ):
                        service = _Phase15DurableService(pending=False)
                        binder = ProfileBoundBinder(service)
                        await loader.from_settings(
                            replace(settings, tools=selected_tools),
                            tool_settings=ToolSettingsContext(
                                patch=PatchToolSettings(
                                    binder,
                                    host_profile,
                                    activation_factory=factory,
                                )
                            ),
                        )
                        assert orchestrator.call_args_list
                        tool = orchestrator.call_args_list[-1].args[4]
                        assert tool.list_tools() == []
                        seed = tool.export_model_capability_seed()
                        assert seed["descriptors"] == []
                        assert "patch." not in str(seed)
                        assert binder.calls == expected_binds
                        if expected_binds:
                            assert binder.binding is not None
                            assert service._runtime_binding is binder.binding
                        else:
                            assert binder.binding is None
                            assert service._runtime_binding is None
                        for (
                            operation,
                            name,
                            raw_arguments,
                        ) in operation_requests:
                            rejected = await tool.execute_call(
                                ToolCall(
                                    id=(
                                        "phase15-inert-"
                                        + profile_label
                                        + "-"
                                        + operation
                                    ),
                                    name=name,
                                    raw_arguments=raw_arguments,
                                ),
                                ToolCallContext(),
                            )
                            assert isinstance(rejected, ToolCallDiagnostic)
                            assert (
                                rejected.code
                                is ToolCallDiagnosticCode.UNKNOWN_TOOL
                            )
                            assert rejected.requested_name == name
                        assert service.invocations == 0

            for (
                capability,
                operation,
                name,
                raw_arguments,
                service_options,
                dispatches,
            ) in (
                (
                    Capability.UPDATE,
                    "update",
                    "patch.apply",
                    (
                        b'{"patch":"*** Begin Patch v1\\n*** Update File: '
                        b'note.txt\\n@@\\n-before\\n+after\\n*** End Patch"}'
                    ),
                    {},
                    True,
                ),
                (
                    Capability.CREATE,
                    "create",
                    "patch.apply",
                    operation_requests[1][2],
                    {},
                    True,
                ),
                (
                    Capability.DELETE,
                    "delete",
                    "patch.apply",
                    operation_requests[2][2],
                    {},
                    True,
                ),
                (
                    Capability.MOVE,
                    "move",
                    "patch.apply",
                    operation_requests[3][2],
                    {},
                    True,
                ),
                (
                    Capability.OBSERVE_MUTATION_PRECONDITIONS,
                    "precondition",
                    "patch.edit",
                    operation_requests[0][2],
                    {},
                    False,
                ),
                (
                    Capability.UPDATE_EXECUTABLE,
                    "executable",
                    "patch.edit",
                    operation_requests[0][2],
                    {"document_mode": FileMode(0o755)},
                    True,
                ),
            ):
                service = _Phase15DurableService(
                    pending=False, **service_options
                )
                bundle = await _phase15_public_bundle(
                    service,
                    policy=denied_policy(capability),
                )
                assert bundle.toolset is not None
                async with bundle.manager:
                    rejected = await bundle.manager.execute_call(
                        ToolCall(
                            id="phase15-independent-" + operation,
                            name=name,
                            raw_arguments=raw_arguments,
                        ),
                        ToolCallContext(
                            patch_capability=bundle.toolset.capability
                        ),
                    )
                if dispatches:
                    assert isinstance(rejected, ToolCallResult)
                    assert _result_mapping(rejected)["status"] == "rejected"
                    assert service.parse_attempts == 1
                else:
                    assert isinstance(rejected, ToolCallDiagnostic)
                    assert rejected.code is ToolCallDiagnosticCode.UNKNOWN_TOOL
                    assert service.parse_attempts == 0
                assert service.invocations == 0
                assert not service._backend.records

            shell_service = _Phase15DurableService(pending=False)
            await loader.from_settings(
                replace(
                    settings,
                    tools=["shell.date", "patch.edit", "patch.apply"],
                ),
                tool_settings=ToolSettingsContext(
                    patch=PatchToolSettings(
                        _Phase15PublicBinder(shell_service),
                        PatchTestHostProfile(),
                    )
                ),
            )
            shell_manager = orchestrator.call_args_list[-1].args[4]
            shell_names = tuple(
                descriptor.name for descriptor in shell_manager.list_tools()
            )
            assert shell_names == ("shell.date",)
            assert all(not name.startswith("patch.") for name in shell_names)
            assert shell_service.invocations == 0
        await stack.aclose()

    run(scenario())


def test_patch_e2e_038_public_tool_manager_fuzz_bounds_privacy_cleanup() -> (
    None
):
    """Drive public adversarial input through an activated ToolManager."""
    assert PATCH_EDIT_SCHEMA["function"]["name"] == "patch.edit"
    assert PATCH_APPLY_SCHEMA["function"]["name"] == "patch.apply"

    async def scenario() -> None:
        service = _Phase15DurableService(pending=False)
        bundle = await _phase15_public_bundle(service)
        assert bundle.toolset is not None
        assert isinstance(bundle.manager, ToolManager)
        context = ToolCallContext(patch_capability=bundle.toolset.capability)
        valid_edit = (
            b'{"path":"note.txt","edits":['
            b'{"old_text":"before","new_text":"after"}]}'
        )

        async with bundle.manager:
            corpus = _fuzz_cases()
            assert tuple(case.identifier for case in corpus) == (
                "PATCH-F15-001",
                "PATCH-F15-002",
                "PATCH-F15-003",
                "PATCH-F15-004",
                "PATCH-F15-005",
                "PATCH-F15-006",
                "PATCH-F15-007",
                "PATCH-F15-008",
                "PATCH-F15-009",
                "PATCH-F15-010",
                "PATCH-F15-011",
            )
            outcomes = await gather(
                *(
                    bundle.manager.execute_call(
                        ToolCall(
                            id="phase15-corpus-" + str(index),
                            name=(
                                "patch.edit"
                                if case.category == "path"
                                else "patch.apply"
                            ),
                            raw_arguments=case.payload,
                        ),
                        context,
                    )
                    for index, case in enumerate(corpus)
                )
            )
            assert len(outcomes) == len(corpus)
            for case, outcome in zip(corpus, outcomes, strict=True):
                assert isinstance(outcome, (ToolCallError, ToolCallResult))
                projection: object = (
                    _result_mapping(outcome)
                    if isinstance(outcome, ToolCallResult)
                    else outcome.error
                )
                if case.identifier == "PATCH-F15-001":
                    assert isinstance(outcome, ToolCallResult)
                    assert _result_mapping(outcome)["status"] == "committed"
                else:
                    assert "status" not in projection or (
                        projection["status"] == "rejected"
                    )
                assert case.payload.hex() not in str(projection)

            invalid_dispatches = service.invocations
            assert invalid_dispatches == 1
            assert len(service._backend.records) == 1

            large_deletion = (
                b'{"patch":"*** Begin Patch v1\\n*** Delete File: note.txt\\n'
                + b"# bounded-deletion-complexity\\n" * 16
                + b'*** End Patch"}'
            )
            oversized = await bundle.manager.execute_call(
                ToolCall(
                    id="phase15-large-deletion",
                    name="patch.apply",
                    raw_arguments=large_deletion,
                ),
                context,
            )
            assert isinstance(oversized, ToolCallResult)
            oversized_projection = _result_mapping(oversized)
            assert oversized_projection["status"] == "rejected"
            assert large_deletion.hex() not in str(oversized_projection)
            assert service.invocations == invalid_dispatches

            large_document = "before\n" + "a" * 2_048
            large_service = _Phase15DurableService(
                pending=False,
                initial_document=large_document,
            )
            large_bundle = await _phase15_public_bundle(
                large_service,
                limits=replace(
                    _phase15_limits(),
                    snapshot_bytes=ByteSize(8_192),
                    proposed_bytes=ByteSize(8_192),
                    review_diff_bytes=ByteSize(8_192),
                ),
            )
            assert large_bundle.toolset is not None
            large_edit = await large_bundle.manager.execute_call(
                ToolCall(
                    id="phase15-large-file-small-edit",
                    name="patch.edit",
                    raw_arguments=(
                        b'{"path":"note.txt","edits":['
                        b'{"old_text":"before\\n","new_text":"after\\n"}]}'
                    ),
                ),
                ToolCallContext(
                    patch_capability=large_bundle.toolset.capability
                ),
            )
            assert isinstance(large_edit, ToolCallResult)
            assert _result_mapping(large_edit)["status"] == "committed"
            assert large_service.invocations == 1
            assert large_service.last_planned_snapshot_bytes == len(
                large_document.encode("utf-8")
            )

            denied_projections: list[dict[str, ToolValue]] = []
            for index, denied_path in enumerate(
                ("/private/phase15.txt", "../private/phase15.txt")
            ):
                denied = await wait_for(
                    bundle.manager.execute_call(
                        ToolCall(
                            id="phase15-denied-path-" + str(index),
                            name="patch.edit",
                            raw_arguments=(
                                b'{"path":"'
                                + denied_path.encode("ascii")
                                + b'","edits":[{"old_text":"before",'
                                b'"new_text":"after"}]}'
                            ),
                        ),
                        context,
                    ),
                    timeout=0.5,
                )
                assert isinstance(denied, ToolCallResult)
                projection = _result_mapping(denied)
                assert projection["status"] == "rejected"
                assert denied_path not in str(projection)
                denied_projections.append(projection)
            assert denied_projections[0] == denied_projections[1]
            assert service.invocations == invalid_dispatches

            for delta, expected_status in (
                (-1, "rejected"),
                (0, "committed"),
                (1, "committed"),
            ):
                bounded_service = _Phase15DurableService(pending=False)
                bounded_bundle = await _phase15_public_bundle(
                    bounded_service,
                    limits=replace(
                        _phase15_limits(),
                        input_bytes=ByteSize(len(valid_edit) + delta),
                    ),
                )
                assert bounded_bundle.toolset is not None
                bounded_context = ToolCallContext(
                    patch_capability=bounded_bundle.toolset.capability
                )
                bounded = await bounded_bundle.manager.execute_call(
                    ToolCall(
                        id="phase15-bound-" + str(delta),
                        name="patch.edit",
                        raw_arguments=valid_edit,
                    ),
                    bounded_context,
                )
                assert isinstance(bounded, ToolCallResult)
                assert _result_mapping(bounded)["status"] == expected_status
                assert bounded_service.invocations == (0 if delta < 0 else 1)

            concurrent = await gather(
                *(
                    bundle.manager.execute_call(
                        ToolCall(
                            id="phase15-concurrent-" + str(index),
                            name="patch.edit",
                            raw_arguments=valid_edit,
                        ),
                        context,
                    )
                    for index in range(4)
                )
            )
            for item in concurrent:
                assert isinstance(item, ToolCallResult)
                assert _result_mapping(item)["status"] == "committed"

            assert service.invocations == invalid_dispatches + len(concurrent)

        assert service.cleanup_count == 0
        assert all(
            "secret.txt" not in str(value)
            for value in (
                *service.events,
                service.current_outcome,
            )
        )

        ambiguous_service = _Phase15DurableService(
            pending=False, ambiguous_after_reservation=True
        )
        ambiguous_bundle = await _phase15_public_bundle(ambiguous_service)
        assert ambiguous_bundle.toolset is not None
        ambiguous = await ambiguous_bundle.manager.execute_call(
            ToolCall(
                id="phase15-ambiguous",
                name="patch.edit",
                raw_arguments=valid_edit,
            ),
            ToolCallContext(
                patch_capability=ambiguous_bundle.toolset.capability
            ),
        )
        assert isinstance(ambiguous, ToolCallResult)
        assert _result_mapping(ambiguous)["status"] == "committed"
        assert ambiguous_service.invocations == 1
        assert len(ambiguous_service._backend.records) == 1

        interrupted_service = _Phase15DurableService(
            pending=True, hold_pending=True
        )
        interrupted_bundle = await _phase15_public_bundle(interrupted_service)
        assert interrupted_bundle.toolset is not None
        interrupted = create_task(
            interrupted_bundle.manager.execute_call(
                ToolCall(
                    id="phase15-interrupted",
                    name="patch.edit",
                    raw_arguments=valid_edit,
                ),
                ToolCallContext(
                    patch_capability=interrupted_bundle.toolset.capability
                ),
            )
        )
        while interrupted_service.current_outcome is None:
            await sleep(0)
        interrupted.cancel()
        with pytest.raises(CancelledError):
            await interrupted
        assert isinstance(interrupted_service.current_outcome, PatchPending)
        assert interrupted_service.invocations == 1
        assert interrupted_service.cleanup_count == 0

        pending_service = _Phase15DurableService(pending=True)
        pending_bundle = await _phase15_public_bundle(pending_service)
        assert pending_bundle.toolset is not None
        pending = await pending_bundle.manager.execute_call(
            ToolCall(
                id="phase15-pending-cleanup",
                name="patch.edit",
                raw_arguments=valid_edit,
            ),
            ToolCallContext(
                patch_capability=pending_bundle.toolset.capability
            ),
        )
        assert isinstance(pending, ToolCallResult)
        assert _result_mapping(pending)["status"] == "committed"
        assert pending_service.cleanup_count == 1
        assert pending_service.current_outcome is not None

    run(scenario())


def test_patch_phase_15_adversarial_workloads() -> None:
    """Contain large, repeated, tombstoned, and multi-lineage pure work."""
    source = b"a" * 1024 + b"needle" + b"b" * 1024
    limits = PlannerLimits(
        max_file_snapshot_bytes=16_384,
        max_snapshot_bytes=16_384,
        max_file_proposed_bytes=16_384,
        max_proposed_bytes=16_384,
        max_changed_bytes=64,
        max_match_candidates=8,
        max_diff_work_bytes=32_768,
        max_diff_bytes=16_384,
        max_memory_bytes=8_388_608,
    )
    request = _edit("needle", "N", "large.txt")
    workspace = _workspace(_file("large.txt", source))
    first = plan(request, workspace, limits)
    second = plan(request, workspace, limits)
    assert first.diff == second.diff
    assert render_review_diff(first, len(first.diff.rendered)) == (
        first.diff.rendered
    )
    with pytest.raises(PlannerError):
        find_match(LogicalText.from_bytes(b"x" * 32), "x", 8)
    deleted = plan(
        _apply(
            (
                "*** Begin Patch v1",
                "*** Delete File: large.txt",
                "*** End Patch",
            )
        ),
        workspace,
        replace(limits, max_changed_bytes=4096),
    )
    assert not deleted.lineages[0].final.present
    tombstone = _apply(
        (
            "*** Begin Patch v1",
            "*** Add File: tomb.txt",
            "+value",
            "*** Delete File: tomb.txt",
            "*** End Patch",
        )
    )
    with pytest.raises(PlannerError):
        plan(tombstone, _workspace(), limits)
    multi = plan(
        _apply(
            (
                "*** Begin Patch v1",
                "*** Update File: note.txt",
                "@@",
                "-before",
                "+after",
                "*** Add File: created.txt",
                "+made",
                "*** End Patch",
            )
        ),
        _workspace(_file("note.txt", b"before\n")),
        limits,
    )
    assert len(multi.lineages) == 2
    assert len({lineage.lineage_id for lineage in multi.lineages}) == 2


def test_patch_phase_15_coarse_denials_and_default_inertness() -> None:
    """Remove path detail and any default-off advertisement predicate."""
    denied = (
        PatchErrorCode.PATH_DENIED,
        PatchErrorCode.ALIAS_DENIED,
        PatchErrorCode.MOUNT_DENIED,
    )
    assert {coarsen_error_code(code, Audience.PUBLIC) for code in denied} == {
        PatchErrorCode.PATH_DENIED
    }
    assert {coarsen_error_code(code, Audience.MODEL) for code in denied} == {
        PatchErrorCode.PATH_DENIED
    }
    manifest = build_patch_production_manifest()
    registry = PatchActivationRegistry(
        manifest, build_patch_activation_verifier(manifest)
    )

    async def unavailable() -> tuple[tuple[str, ...], int]:
        return (
            await registry.advertised_tools(manifest.profiles[0].key),
            await registry.active_binding_count(manifest.profiles[0].key),
        )

    tools, bindings = run(unavailable())
    assert tools == ()
    assert bindings == 0
    assert all(
        not PatchProtocolProfile(surface).active
        for surface in PatchProtocolSurface
    )


def test_patch_phase_15_parser_limit_matrix_is_exact_and_bounded() -> None:
    """Exercise each parser-owned N-1/N/N+1 limit without target authority."""
    payload = dumps(
        {
            "path": "alpha.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    parser = PatchRequestParser(PatchInputLimits(max_raw_bytes=len(payload)))
    assert parser.parse(_ingress(payload, RawPatchInputKind.EDIT_JSON))
    with pytest.raises(PatchInputError) as raw_error:
        PatchRequestParser(
            PatchInputLimits(max_raw_bytes=len(payload) - 1)
        ).parse(_ingress(payload, RawPatchInputKind.EDIT_JSON))
    assert raw_error.value.code is PatchInputErrorCode.OVERSIZED
    assert PatchRequestParser(
        PatchInputLimits(max_raw_bytes=len(payload) + 1)
    ).parse(_ingress(payload, RawPatchInputKind.EDIT_JSON))

    for field, size in (
        ("max_path_characters", len("alpha.txt")),
        ("max_path_bytes", len(b"alpha.txt")),
        ("max_component_characters", len("alpha.txt")),
        ("max_component_bytes", len(b"alpha.txt")),
        ("max_path_components", 1),
        ("max_edits", 1),
        ("max_json_depth", 3),
    ):
        accepted = PatchInputLimits(**{field: size})
        assert PatchRequestParser(accepted).parse(
            _ingress(payload, RawPatchInputKind.EDIT_JSON)
        )
        with pytest.raises(PatchInputError):
            PatchRequestParser(PatchInputLimits(**{field: size - 1})).parse(
                _ingress(payload, RawPatchInputKind.EDIT_JSON)
            )
        assert PatchRequestParser(PatchInputLimits(**{field: size + 1})).parse(
            _ingress(payload, RawPatchInputKind.EDIT_JSON)
        )

    document = dumps(
        {
            "patch": "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: alpha.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            )
        },
        separators=(",", ":"),
    ).encode("utf-8")
    for field, size in (
        ("max_paths", 1),
        ("max_declarations", 1),
        ("max_hunks", 1),
    ):
        assert PatchRequestParser(PatchInputLimits(**{field: size})).parse(
            _ingress(document)
        )
        with pytest.raises(PatchInputError):
            PatchRequestParser(PatchInputLimits(**{field: size - 1})).parse(
                _ingress(document)
            )
        assert PatchRequestParser(PatchInputLimits(**{field: size + 1})).parse(
            _ingress(document)
        )


def test_patch_phase_15_planner_limit_matrix_is_exact_and_deterministic() -> (
    None
):
    """Bound snapshots, candidates, proposals, diffs, and planner memory."""
    source = b"before\n"
    request = _edit("before", "after")
    workspace = _workspace(_file("note.txt", source))
    baseline = PlannerLimits(max_memory_bytes=1_000_000)
    accepted = plan(request, workspace, baseline)
    assert (
        accepted.diff.digest.value
        == sha256(accepted.diff.rendered).hexdigest()
    )

    def with_limit(field: str, value: int) -> PlannerLimits:
        """Return baseline limits with one exact integer ceiling changed."""
        match field:
            case "max_file_snapshot_bytes":
                return replace(baseline, max_file_snapshot_bytes=value)
            case "max_snapshot_bytes":
                return replace(baseline, max_snapshot_bytes=value)
            case "max_file_proposed_bytes":
                return replace(baseline, max_file_proposed_bytes=value)
            case "max_proposed_bytes":
                return replace(baseline, max_proposed_bytes=value)
            case "max_changed_bytes":
                return replace(baseline, max_changed_bytes=value)
            case "max_diff_bytes":
                return replace(baseline, max_diff_bytes=value)
            case "max_diff_work_bytes":
                return replace(baseline, max_diff_work_bytes=value)
            case "max_memory_bytes":
                return replace(baseline, max_memory_bytes=value)
            case _:
                raise AssertionError("unknown planner limit")

    def minimum(field: str) -> int:
        """Return the first accepted integer ceiling for one planner field."""
        lower = 1
        upper = getattr(baseline, field)
        while lower < upper:
            midpoint = (lower + upper) // 2
            try:
                plan(request, workspace, with_limit(field, midpoint))
            except PlannerError as error:
                assert error.code is PlannerErrorCode.LIMIT
                lower = midpoint + 1
            else:
                upper = midpoint
        return lower

    for field in (
        "max_file_snapshot_bytes",
        "max_snapshot_bytes",
        "max_file_proposed_bytes",
        "max_proposed_bytes",
        "max_changed_bytes",
        "max_diff_bytes",
        "max_diff_work_bytes",
        "max_memory_bytes",
    ):
        size = minimum(field)
        assert (
            plan(request, workspace, with_limit(field, size)).diff
            == accepted.diff
        )
        with pytest.raises(PlannerError) as below:
            plan(
                request,
                workspace,
                with_limit(field, size - 1),
            )
        assert below.value.code is PlannerErrorCode.LIMIT
        assert (
            plan(
                request,
                workspace,
                with_limit(field, size + 1),
            ).diff
            == accepted.diff
        )

    repeated = LogicalText.from_bytes(b"a" * 12)
    for maximum in (11, 12, 13):
        if maximum < 12:
            with pytest.raises(PlannerError) as candidates:
                find_match(repeated, "a", maximum)
            assert candidates.value.code is PlannerErrorCode.LIMIT
        else:
            with pytest.raises(PlannerError) as candidates:
                find_match(repeated, "a", maximum)
            assert candidates.value.code is PlannerErrorCode.AMBIGUOUS_MATCH
    with pytest.raises(PlannerError) as memory:
        plan(request, workspace, PlannerLimits(max_memory_bytes=1))
    assert memory.value.code is PlannerErrorCode.LIMIT


def test_patch_phase_15_rooted_candidate_preflight_is_not_commit_e2e() -> None:
    """Reject escaped paths before a pure planner can produce a candidate."""
    workspace = _workspace(_file("note.txt", b"before\n"))
    candidate = plan(_edit("before", "after"), workspace)
    assert candidate.lineages[0].source_path == LogicalPath("note.txt")
    with pytest.raises(PatchInputError) as escaped:
        _edit("before", "after", "../outside.txt")
    assert escaped.value.code is PatchInputErrorCode.PATH
    with pytest.raises(PlannerError) as absent:
        plan(_edit("before", "after", "other.txt"), workspace)
    assert absent.value.code is PlannerErrorCode.SOURCE_MISSING


def test_patch_phase_15_inert_schema_boundary_is_not_public_e2e() -> None:
    """Keep inert schemas separate from a public ToolManager execution."""
    parser = PatchRequestParser(PatchInputLimits(max_raw_bytes=128))
    malformed = b'{"patch":"*** Begin Patch v1\\n*** End Patch"}'
    with pytest.raises(PatchInputError):
        parser.parse(_ingress(malformed))
    assert PATCH_EDIT_SCHEMA["function"]["name"] == "patch.edit"
    assert PATCH_APPLY_SCHEMA["function"]["name"] == "patch.apply"
    assert all(
        not PatchProtocolProfile(surface).active
        for surface in PatchProtocolSurface
    )


def test_patch_phase_15_container_profile_remains_incomplete() -> None:
    """Keep pure planning separate from an incomplete container profile."""
    workspace = _workspace(_file("note.txt", b"before\n"))
    edit = plan(_edit("before", "after"), workspace)
    apply = plan(
        _apply(
            (
                "*** Begin Patch v1",
                "*** Update File: note.txt",
                "@@",
                "-before",
                "+after",
                "*** End Patch",
            )
        ),
        workspace,
    )
    assert edit.diff == apply.diff
    manifest = build_patch_production_manifest()
    assert manifest.profiles[0].state is PatchProfileState.INCOMPLETE
    assert not manifest.profiles[0].proven


class _Phase15AdapterClock(ApprovalClock):
    """Provide the fixed live tick for selected adapter endpoints."""

    async def now(self) -> ExpiryTick:
        """Return an unexpired timestamp for the selected context."""
        return ExpiryTick(1)


class _Phase15AdapterBroker:
    """Approve the adapter endpoint's exact sealed review request."""

    async def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Return one policy-matching durable reviewer decision."""
        return BrokerDecision(
            request.requirements.broker,
            (
                ReviewerDecision(
                    PatchPrincipalId("phase15-adapter-reviewer"),
                    request.subject.tenant,
                    request.requirements.reviewer_role,
                    ApprovalDecisionState.APPROVED,
                ),
            ),
        )


def _phase15_adapter_subject() -> ExecutionSubject:
    """Return the host subject bound to the selected adapter contexts."""
    return ExecutionSubject(
        PatchPrincipalId("phase15-adapter-principal"),
        PatchTenantId("phase15-adapter-tenant"),
        PatchRunId("phase15-adapter-run"),
        PatchSessionId("phase15-adapter-session"),
        PatchTaskId("phase15-adapter-task"),
        PatchAgentId("phase15-adapter-agent"),
    )


def _phase15_adapter_revalidation_snapshot() -> RevalidationSnapshot:
    """Return the complete coordinator witness for a selected adapter run."""
    return RevalidationSnapshot(
        tuple(
            sorted(
                (
                    RevalidationFact(
                        field, "phase15-adapter-" + field.value, "bound"
                    )
                    for field in RevalidationField
                ),
                key=lambda fact: (fact.field.value, fact.key, fact.value),
            )
        )
    )


async def _execute_phase15_local_worker_operation(
    profile: LocalTargetProfile,
    scope: ResolvedMutationScope,
    operation: OperationType,
    ingress: RawPatchIngress,
    suffix: str,
) -> PatchResult:
    """Commit one parsed operation through the selected local worker."""
    target = LocalCommitTarget(profile)
    inspection = LocalInspectionTarget(profile)
    request = PatchRequestParser(PatchInputLimits()).parse(ingress)
    assert request.operation is operation
    candidate = plan(
        request,
        (
            await inspection.inspect(
                InspectionRequest(scope, (LogicalPath("note.txt"),))
            )
        ).planner_workspace(),
    )
    policy = _phase15_policy()
    authorizer = PolicyAuthorizer(policy)
    paths = tuple(lineage.final.path for lineage in candidate.lineages)
    effects = frozenset(
        capability
        for lineage in candidate.lineages
        for capability in lineage.capabilities
    )
    preflight = await authorizer.authorize_preinspection(
        PreflightRequest(
            operation,
            paths,
            effects,
            frozenset(paths),
            compose_limits(
                profile.limits,
                profile.limits,
                profile.limits,
                profile.limits,
                profile.limits,
            ),
        )
    )
    final = await authorizer.authorize_final(
        preflight, candidate, await target.handshake(scope)
    )
    subject = _phase15_adapter_subject()
    sealed = seal_plan(
        PatchPlanId("plan_" + suffix),
        PlanBinding(
            PatchRequest(
                1,
                PatchRequestId("request_" + suffix),
                PatchExecutionId("execution_" + suffix),
                operation,
                PatchInput(b"phase15-local-" + suffix.encode()),
                paths,
            ),
            candidate.request_digest,
            subject,
            ContextKind.LOCAL,
            profile.identity,
            None,
            preflight,
            final,
        ),
        candidate,
        ExpiryTick(100),
    )
    approvals = ApprovalService(
        _Phase15AdapterBroker(), _Phase15AdapterClock(), RuntimeGrantStore()
    )
    decision = await approvals.await_review(
        PlanReviewRequest(sealed, subject, final.approval)
    )
    assert decision.grant is not None
    store = InMemoryCoordinatorStore(approvals)
    coordinator = InMemoryPatchCoordinator(
        store,
        InMemoryLeaseManager(store),
        ScriptedReconciler(_phase15_adapter_revalidation_snapshot()),
    )
    reservation = await coordinator.reserve(
        RuntimeIdentity(
            subject,
            final.approval.route,
            RetransmissionKey("phase15-local-" + suffix),
        ),
        candidate.request_digest,
    )
    result = await coordinator.execute(
        reservation,
        sealed,
        decision.grant,
        _phase15_adapter_revalidation_snapshot(),
        await target.worker(scope),
        "phase15-local-controller",
    )
    assert type(result) is PatchResult
    return result


def _phase15_adapter_configuration() -> (
    tuple[SandboxPatchServiceConfiguration, InMemoryDurablePatchStore]
):
    """Create production planning, approval, and durable services."""
    clock = _Phase15AdapterClock()
    approvals = ApprovalService(
        _Phase15AdapterBroker(), clock, RuntimeGrantStore()
    )
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    return (
        SandboxPatchServiceConfiguration(
            _phase15_adapter_subject(),
            PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
            approvals,
            PhaseFiveDurableApprovalIssuer(approvals, authority),
            clock,
            DurationTicks(10),
            DurationTicks(10),
        ),
        store,
    )


def _phase15_sandbox_settings(
    root: Path, namespace: Path
) -> SandboxPatchRuntimeSettings:
    """Build the selected production sandbox settings with no write root."""
    backend = (
        SandboxBackend.SEATBELT
        if runtime_platform == "darwin"
        else SandboxBackend.BUBBLEWRAP
    )
    settings = IsolationSettings.from_dict(
        {
            "mode": "sandbox",
            "sandbox": {
                "backend": backend.value,
                "default_profile": "phase15-selected",
                "allowed_profiles": ["phase15-selected"],
                "profiles": {
                    "phase15-selected": {
                        "trusted_executables": ["/bin/sh"],
                        "executable_search_roots": ["/bin"],
                        "read_roots": [str(root)],
                        "write_roots": [],
                        "deny_roots": [],
                        "scratch_roots": [str(namespace)],
                        "output_roots": [],
                        "environment": {"variables": {}, "allowlist": []},
                        "network": {"mode": "none", "egress_allowlist": []},
                        "resources": {"timeout_seconds": 10, "pids": None},
                        "output": {
                            "max_stdout_bytes": 4096,
                            "max_stderr_bytes": 4096,
                            "allow_artifacts": False,
                            "max_artifact_bytes": 0,
                        },
                        "child_processes": (
                            "deny"
                            if backend is SandboxBackend.SEATBELT
                            else "allow"
                        ),
                        "inherited_fds": "stdio",
                        "cleanup": "delete",
                    }
                },
                "profile_registry_id": "phase15-selected",
                "policy_version": "phase15-public-v1",
            },
        },
        source=trusted_isolation_source("sdk"),
    ).select_profile(
        IsolationProfileSelection(
            mode=IsolationMode.SANDBOX,
            profile="phase15-selected",
            required=True,
        )
    )
    assert settings.sandbox is not None
    plan_value = SandboxExecutionPlan(
        request=SandboxPlanRequest(
            request_kind=SandboxPlanRequestKind.AGENT_SESSION,
            logical_name="phase15-adapter",
            command="/bin/sh",
            argv=("/bin/sh", "-c", "exit 0"),
            cwd=str(root),
        ),
        settings=settings.sandbox,
    )
    witness, _ = capture_rooted_root_binding(root)
    implementation = SandboxWorkerImplementationId(
        backend.value + "-phase15-adapter-v1"
    )
    token = sha256(str(root).encode()).hexdigest()[:16]
    return SandboxPatchRuntimeSettings(
        plan_value,
        SandboxPatchRuntimeContext(
            TargetIdentity(
                PatchContextId("context_" + token),
                PatchWorkspaceId("workspace_" + token),
                PatchDomainId("domain_" + token),
                PatchTargetId("target_" + token),
                sandbox_protocol_id(
                    SandboxWorkerProtocolVersion("sandbox-patch-runtime-v2")
                ),
                witness.filesystem_id,
                witness.mount_id,
                "phase15-public-v1",
                "persistent-lease-" + token,
                PatchApprovalId("approval_" + token),
                implementation,
            ),
            _phase15_limits(),
            ByteSize(4096),
            None,
            SandboxChannelId("phase15-sandbox-channel"),
            SandboxContextLifetimeId("phase15-sandbox-context"),
            implementation,
        ),
    )


def _phase15_container_settings(root: Path) -> ContainerPatchRuntimeSettings:
    """Build a fresh pinned container adapter profile for one test run."""
    token = uuid4().hex[:16]
    implementation = SandboxWorkerImplementationId("container-phase15-v1")
    return ContainerPatchRuntimeSettings(
        ContainerPatchImage(
            "python:3.11-slim-bookworm@sha256:"
            "2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91"
        ),
        ContainerPatchRuntimeContext(
            TargetIdentity(
                PatchContextId("context_" + token),
                PatchWorkspaceId("workspace_" + token),
                PatchDomainId("domain_" + token),
                PatchTargetId("target_" + token),
                container_protocol_id(),
                "docker-volume-" + token,
                "docker-mount-" + token,
                "phase15-public-v1",
                "persistent-lease-" + token,
                PatchApprovalId("approval_" + token),
                implementation,
            ),
            _phase15_limits(),
            ByteSize(4096),
            None,
            SandboxChannelId("phase15-container-channel"),
            SandboxContextLifetimeId("phase15-container-context"),
            implementation,
        ),
        root,
        SandboxExecutionPlanFingerprint("phase15-container-plan-v1"),
        ContainerPersistentLeaseAuthority.from_bytes(b"p" * 32),
        test_profile=True,
    )


def test_patch_e2e_039_local_sandbox_container_conformance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply one corpus through each activated selected adapter endpoint."""
    local_root = tmp_path / "selected-local-root"
    sandbox_root = tmp_path / "selected-sandbox-root"
    container_root = tmp_path / "selected-container-root"
    sandbox_namespace = tmp_path / "selected-sandbox-private"
    container_namespace = tmp_path / "selected-container-private"
    for root in (local_root, sandbox_root, container_root):
        root.mkdir()
        (root / "note.txt").write_bytes(b"before\n")
        (root / "second.txt").write_bytes(b"second-before\n")
    sandbox_namespace.mkdir()
    container_namespace.mkdir()
    local_host_canary = tmp_path / "local-host-canary.txt"
    sandbox_host_canary = tmp_path / "sandbox-host-canary.txt"
    container_host_canary = tmp_path / "container-host-canary.txt"
    for canary in (
        local_host_canary,
        sandbox_host_canary,
        container_host_canary,
    ):
        canary.write_bytes(b"host remains private\n")
    sandbox_namespace_canary = sandbox_namespace / "namespace-canary.txt"
    container_namespace_canary = container_namespace / "namespace-canary.txt"
    sandbox_namespace_canary.write_bytes(b"namespace remains private\n")
    container_namespace_canary.write_bytes(b"namespace remains private\n")
    local_profile = phase15_local_target_profile(local_root, monkeypatch)
    local_profile = replace(
        local_profile,
        identity=replace(
            local_profile.identity,
            policy_revision=_phase15_policy().revision.value,
        ),
    )
    sandbox_configuration, sandbox_store = _phase15_adapter_configuration()
    sandbox_settings = _phase15_sandbox_settings(
        sandbox_root, sandbox_namespace
    )
    sandbox_binder = SandboxPatchRuntimeBinder.from_settings(
        sandbox_settings,
        sandbox_configuration,
        _phase15_policy(),
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True, sandbox_store),
        PatchPersistenceBinding(True, sandbox_store),
    )
    container_configuration, container_store = _phase15_adapter_configuration()
    container_settings = replace(
        _phase15_container_settings(container_root),
        image=ContainerPatchImage(patch_container_test_image()),
    )
    container_binder = ContainerPatchRuntimeBinder(
        container_settings.create_runtime(),
        container_configuration,
        _phase15_policy(),
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True, container_store),
        PatchPersistenceBinding(True, container_store),
    )
    container_volume: str | None = None

    async def assert_code_nonwriter(path: Path) -> None:
        """Require the ordinary code tool to leave a canary absent."""
        with pytest.raises(NameError):
            await CodeTool()(
                "def run():\n    return open(" + repr(str(path)) + ", 'w')\n",
                context=ToolCallContext(),
            )
        assert not path.exists()

    async def read_local() -> ContextKind:
        """Commit through the activated public local ToolManager boundary."""
        scope = await LocalScopeResolver(local_profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(local_profile)
        worker = await target.worker(scope)
        assert worker.__class__.__name__ == "RootedLocalCommitWorker"

        async def execute_local(
            operation: OperationType, raw_arguments: bytes
        ) -> PatchResult:
            """Delegate the loaded service to its real rooted target path."""
            result = await _execute_phase15_local_worker_operation(
                local_profile,
                scope,
                operation,
                _ingress(
                    raw_arguments,
                    (
                        RawPatchInputKind.EDIT_JSON
                        if operation is OperationType.EDIT
                        else RawPatchInputKind.APPLY_JSON
                    ),
                ),
                "a" * 16 if operation is OperationType.EDIT else "b" * 16,
            )
            return result

        service = _Phase15DurableService(
            pending=False, local_executor=execute_local
        )
        binding = PatchRuntimeBinding(
            scope,
            await target.handshake(scope),
            _phase15_policy(),
            PatchApprovalBinding(True),
            PatchCoordinatorBinding(True, service.store),
            PatchPersistenceBinding(True, service.store),
            service,
        )

        class LocalBinder:
            """Return the authenticated local target binding already probed."""

            async def bind(self) -> PatchRuntimeBinding:
                """Return the exact retained local binding for activation."""
                return binding

        bundle = await PatchToolLoader(
            LocalBinder(), activated_patch_test_profile()
        ).load(enable_tools=["patch.edit", "patch.apply"])
        assert bundle.toolset is not None
        assert bundle.runtime_binding is binding
        async with bundle.manager:
            context = ToolCallContext(
                patch_capability=bundle.toolset.capability
            )
            first = await bundle.manager.execute_call(
                ToolCall(
                    id="phase15-local-edit",
                    name="patch.edit",
                    raw_arguments=dumps(
                        {
                            "path": "note.txt",
                            "edits": [
                                {
                                    "old_text": "before",
                                    "new_text": "after",
                                }
                            ],
                        },
                        separators=(",", ":"),
                    ).encode(),
                ),
                context,
            )
            assert isinstance(first, ToolCallResult)
            assert _result_mapping(first)["status"] == "committed"
            second = await bundle.manager.execute_call(
                ToolCall(
                    id="phase15-local-apply",
                    name="patch.apply",
                    raw_arguments=dumps(
                        {
                            "patch": "\n".join(
                                (
                                    "*** Begin Patch v1",
                                    "*** Update File: note.txt",
                                    "@@",
                                    "-after",
                                    "+after-apply",
                                    "*** End Patch",
                                )
                            )
                        },
                        separators=(",", ":"),
                    ).encode(),
                ),
                context,
            )
            assert isinstance(second, ToolCallResult)
            assert _result_mapping(second)["status"] == "committed"
            assert service.invocations == 2
            assert service.last_access is not None
            journal = (
                await service.store.inspect(service.last_access)
            ).journal
            assert [entry.state for entry in journal.steps] == [
                CommitStepState.PLANNED,
                CommitStepState.COMMITTED,
            ]
            assert [entry.state for entry in journal.artifacts] == [
                DurableArtifactState.INTENDED,
                DurableArtifactState.PRESENT,
                DurableArtifactState.REMOVED,
            ]
            assert journal.cursor.revision.value == 5
            event_stream = sorted(
                [
                    (entry.cursor.revision.value, "step", entry.state.value)
                    for entry in journal.steps
                ]
                + [
                    (
                        entry.cursor.revision.value,
                        "artifact",
                        entry.state.value,
                    )
                    for entry in journal.artifacts
                ]
            )
            assert event_stream == [
                (1, "artifact", DurableArtifactState.INTENDED.value),
                (2, "step", CommitStepState.PLANNED.value),
                (3, "step", CommitStepState.COMMITTED.value),
                (4, "artifact", DurableArtifactState.PRESENT.value),
                (5, "artifact", DurableArtifactState.REMOVED.value),
            ]
            committed = [
                sequence
                for sequence, kind, state in event_stream
                if kind == "step" and state == CommitStepState.COMMITTED.value
            ]
            settlement_artifacts = [
                sequence
                for sequence, kind, state in event_stream
                if kind == "artifact"
                and state != DurableArtifactState.INTENDED.value
            ]
            assert committed and settlement_artifacts
            assert max(committed) < min(settlement_artifacts)
            runtime = service._activation_observer
            assert runtime is bundle.toolset._activation_runtime
            assert runtime is not None
            assert (
                await runtime.registry.active_binding_count(runtime.lease.key)
                == 0
            )
        later = await LocalInspectionTarget(local_profile).inspect(
            InspectionRequest(scope, (LogicalPath("note.txt"),))
        )
        assert later.snapshots[0].bytes_value is not None
        assert later.snapshots[0].bytes_value._value == b"after-apply\n"
        assert local_host_canary.read_bytes() == b"host remains private\n"
        assert local_profile.commit_namespace is not None
        assert not tuple(local_profile.commit_namespace.iterdir())
        await assert_code_nonwriter(local_root / "ordinary-code-write.txt")
        return scope.context_kind

    async def read_sandbox() -> PatchRuntimeBinding:
        """Bind, edit, apply, and inspect the selected sandbox endpoint."""
        bundle = await PatchToolLoader(
            sandbox_binder,
            activated_patch_test_profile(),
        ).load(enable_tools=["patch.edit", "patch.apply"])
        assert (
            bundle.runtime_binding is not None and bundle.toolset is not None
        )
        async with bundle.manager:
            binding = bundle.runtime_binding
            assert binding.scope.context_kind is ContextKind.SANDBOX
            worker = await SandboxCommitTarget(sandbox_binder.runtime).worker(
                binding.scope
            )
            assert worker.__class__.__name__ == "RootedSandboxCommitWorker"
            host = bundle.toolset.sdk_host()
            first = await host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "before", "new_text": "after"}],
                },
            )
            assert first.status is PatchStatus.COMMITTED
            second = await host.invoke_json(
                OperationType.APPLY,
                {
                    "patch": "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: note.txt",
                            "@@",
                            "-after",
                            "+after-apply",
                            "*** End Patch",
                        )
                    )
                },
            )
            assert second.status is PatchStatus.COMMITTED
            later = await SandboxInspectionTarget(
                sandbox_binder.runtime
            ).inspect(
                InspectionRequest(binding.scope, (LogicalPath("note.txt"),))
            )
            assert later.snapshots[0].bytes_value is not None
            assert later.snapshots[0].bytes_value._value == b"after-apply\n"
            ordinary = replace(
                sandbox_settings.execution_plan,
                request=SandboxPlanRequest(
                    request_kind=SandboxPlanRequestKind.AGENT_SESSION,
                    logical_name="phase15-ordinary-shell",
                    command="/bin/sh",
                    argv=(
                        "/bin/sh",
                        "-c",
                        "printf ordinary > ordinary-shell-write.txt",
                    ),
                    cwd=str(sandbox_root),
                ),
            )
            nonwriter = (
                SeatbeltSandboxBackend()
                if runtime_platform == "darwin"
                else BubblewrapSandboxBackend()
            )
            ordinary_result = await nonwriter.execute(ordinary)
            assert ordinary_result.status in {
                SandboxResultStatus.DENIED,
                SandboxResultStatus.FAILED,
            }
            assert not (sandbox_root / "ordinary-shell-write.txt").exists()
            await assert_code_nonwriter(
                sandbox_root / "ordinary-code-write.txt"
            )
            assert (
                sandbox_host_canary.read_bytes() == b"host remains private\n"
            )
            assert (
                sandbox_namespace_canary.read_bytes()
                == b"namespace remains private\n"
            )
            return binding

    async def read_container() -> PatchRuntimeBinding:
        """Bind, edit, apply, and inspect the selected container endpoint."""
        nonlocal container_volume
        bundle = await PatchToolLoader(
            container_binder,
            activated_patch_test_profile(),
        ).load(enable_tools=["patch.edit", "patch.apply"])
        assert (
            bundle.runtime_binding is not None and bundle.toolset is not None
        )
        async with bundle.manager:
            binding = bundle.runtime_binding
            assert binding.scope.context_kind is ContextKind.CONTAINER
            worker = await ContainerPatchTarget(
                container_binder.runtime
            ).worker(binding.scope)
            assert worker.__class__.__name__ == "RootedSandboxCommitWorker"
            host = bundle.toolset.sdk_host()
            first = await host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "before", "new_text": "after"}],
                },
            )
            assert first.status is PatchStatus.COMMITTED
            second = await host.invoke_json(
                OperationType.APPLY,
                {
                    "patch": "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: note.txt",
                            "@@",
                            "-after",
                            "+after-apply",
                            "*** End Patch",
                        )
                    )
                },
            )
            assert second.status is PatchStatus.COMMITTED
            later = await ContainerInspectionTarget(
                container_binder.runtime
            ).inspect(
                InspectionRequest(binding.scope, (LogicalPath("note.txt"),))
            )
            assert later.snapshots[0].bytes_value is not None
            assert later.snapshots[0].bytes_value._value == b"after-apply\n"
            volume = container_binder.runtime._process.volume_name
            assert volume is not None
            container_volume = volume
            ordinary_writer = await create_subprocess_exec(
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--mount",
                "type=volume,source=" + volume + ",target=/workspace,readonly",
                container_settings.image.reference,
                "/bin/sh",
                "-c",
                "printf ordinary > /workspace/ordinary-shell-write.txt",
            )
            assert await ordinary_writer.wait() != 0
            assert not (container_root / "ordinary-shell-write.txt").exists()
            await assert_code_nonwriter(
                container_root / "ordinary-code-write.txt"
            )
            assert (
                container_host_canary.read_bytes() == b"host remains private\n"
            )
            assert (
                container_namespace_canary.read_bytes()
                == b"namespace remains private\n"
            )
            return binding

    async def collect() -> (
        tuple[ContextKind | PatchRuntimeBinding | BaseException, ...]
    ):  # noqa: E501
        """Start all selected endpoints even when one host capability fails."""
        try:
            return await gather(
                read_local(),
                read_sandbox(),
                read_container(),
                return_exceptions=True,
            )
        finally:
            await container_binder.runtime.dispose()

    outcomes = run(collect())
    failures = tuple(
        value for value in outcomes if isinstance(value, BaseException)
    )
    assert not failures, tuple(
        type(value).__name__ + ": " + str(value) for value in failures
    )
    assert outcomes[0] is ContextKind.LOCAL
    assert all(
        isinstance(value, PatchRuntimeBinding) for value in outcomes[1:]
    )
    assert container_volume is not None
    assert container_binder.runtime._process.volume_name is None
    assert local_root.joinpath("note.txt").read_bytes() == b"after-apply\n"
    assert sandbox_root.joinpath("note.txt").read_bytes() == b"after-apply\n"
    assert container_root.joinpath("note.txt").read_bytes() == b"before\n"
    assert (
        sandbox_namespace_canary.read_bytes() == b"namespace remains private\n"
    )
    assert (
        container_namespace_canary.read_bytes()
        == b"namespace remains private\n"
    )


def test_patch_phase_15_proposed_state_is_not_provider_commit_e2e() -> None:
    """Preserve proposed bytes without claiming a provider commit lifecycle."""
    parser = PatchRequestParser()
    document = dumps(
        {
            "patch": "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: note.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            )
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = parser.parse(_ingress(document))
    candidate = plan(request, _workspace(_file("note.txt", b"before\n")))
    assert PATCH_APPLY_SCHEMA["function"]["name"] == "patch.apply"
    assert candidate.lineages[0].final.bytes_value is not None
    assert (
        candidate.lineages[0].final.bytes_value.digest().value
        == sha256(b"after\n").hexdigest()
    )
    assert candidate.diff.rendered.count(b"after") == 1
    assert build_patch_production_manifest().profiles[0].state is (
        PatchProfileState.INCOMPLETE
    )


def test_patch_phase_15_incomplete_profile_is_inert() -> None:
    """Expose no tools or capability detail for the incomplete profile."""
    manifest = build_patch_production_manifest()
    profile = manifest.profiles[0]
    registry = PatchActivationRegistry(
        manifest, build_patch_activation_verifier(manifest)
    )

    async def inspect() -> tuple[tuple[str, ...], int]:
        """Observe only the coarse inert state of the incomplete profile."""
        return (
            await registry.advertised_tools(profile.key),
            await registry.active_binding_count(profile.key),
        )

    assert run(inspect()) == ((), 0)
    assert not profile.proven


def test_patch_phase_15_approval_and_artifact_expiry_remain_closed() -> None:
    """Keep a constructed approval service outside dormant activation."""

    class Clock:
        """Expose a deterministic clock without a real approval request."""

        async def now(self) -> ExpiryTick:
            """Return no value because the service must not be invoked."""
            raise AssertionError("unselected approval service was invoked")

    class Broker:
        """Expose a deterministic broker without a real approval request."""

        async def decide(self, _request: PlanReviewRequest) -> BrokerDecision:
            """Reject any attempted broker call before external approval."""
            raise AssertionError("unselected approval broker was invoked")

    service = ApprovalService(Broker(), Clock(), RuntimeGrantStore())
    assert service._broker is not None
    assert not build_patch_production_manifest().profiles[0].proven


def test_patch_phase_15_coordination_faults_are_fenced_and_nonblocking() -> (
    None
):
    """Record concurrent fault checkpoints without retaining resources."""
    faults = ScriptedFaultController(
        frozenset({CoordinatorBoundary.FENCE, CoordinatorBoundary.CLEANUP})
    )
    resources = RuntimeResources(0, 0, 0, 0, 0, 0)

    async def checkpoint(boundary: CoordinatorBoundary) -> bool:
        """Pass one zero-depth real coordination checkpoint."""
        return await faults.checkpoint(boundary, resources)

    async def checkpoints() -> list[bool]:
        """Observe concurrent fault categories through the real controller."""
        return list(
            await gather(
                checkpoint(CoordinatorBoundary.LEASE),
                checkpoint(CoordinatorBoundary.FENCE),
                checkpoint(CoordinatorBoundary.CLEANUP),
            )
        )

    outcomes = run(checkpoints())
    assert outcomes == [False, True, True]
    assert faults.observed == (
        CoordinatorBoundary.LEASE,
        CoordinatorBoundary.FENCE,
        CoordinatorBoundary.CLEANUP,
    )
    assert all(depth == resources for depth in faults.depths)


def test_patch_phase_15_durable_faults_preserve_truth_and_cleanup() -> None:
    """Deduplicate concurrent retries and delete authenticated expiry data."""
    key = DurableRetentionKey(
        PatchRetentionKeyId("retention_" + "f" * 16), b"p" * 32
    )
    cipher = AesGcmDurableRetentionCipher(
        InMemoryDurableRetentionKeyResolver(key.key_id, {key.key_id: key})
    )
    backend = InMemoryDurablePatchBackend(
        retention_authorizer=StaticDurableRetentionAuthorizer(
            frozenset((Audience.APPROVER,))
        ),
        retention_validator=AesGcmDurableRetentionEnvelopeValidator(cipher),
    )
    store = InMemoryDurablePatchStore(backend)
    identity = DurableRequestIdentity(
        PatchTenantId("phase15-tenant"),
        PatchPrincipalId("phase15-principal"),
        PatchExecutionId("execution_" + "f" * 16),
        PolicyRouteId("phase15-route"),
        RetransmissionKey("phase15-retry"),
    )
    digest = AlgorithmDigest.from_bytes(b"phase15-canonical-request")

    async def reserve() -> tuple[int, int, int]:
        """Persist once, replay matching retries, and expire ciphertext."""
        reservations = await gather(
            *(store.reserve(identity, digest) for _ in range(4))
        )
        initial = next(
            reservation
            for reservation in reservations
            if not reservation.replayed
        )
        assert sum(not item.replayed for item in reservations) == 1
        assert len(backend.records) == 1
        with pytest.raises(DurableStoreError) as conflicting:
            await store.reserve(
                identity, AlgorithmDigest.from_bytes(b"phase15-conflict")
            )
        assert (
            conflicting.value.code
            is DurableStoreErrorCode.IDEMPOTENCY_CONFLICT
        )
        retention_id = PatchRetentionRecordId("retained_" + "f" * 16)
        binding = DurableRetentionBinding(
            initial.request_id,
            retention_id,
            DurableRetentionKind.PRIVATE_STAGING,
        )
        encrypted = await cipher.seal(b"phase15-private-retention", binding)
        record = DurableRetentionRecord(
            retention_id,
            DurableRetentionKind.PRIVATE_STAGING,
            encrypted.key_id,
            encrypted.value,
            DurableRetentionPolicy(ExpiryTick(2), False),
        )
        await store.put_retention(initial, record)
        access = DurableRetentionAccess(
            DurableRequestAccess(initial.request_id, identity)
        )
        assert (
            await store.get_retention(access, retention_id, ExpiryTick(1))
            == record
        )
        cleanup = await store.cleanup_retention(ExpiryTick(2))
        assert cleanup.records_deleted == 1
        assert cleanup.bytes_deleted.value == encrypted.value.size().value
        with pytest.raises(DurableStoreError) as expired:
            await store.get_retention(access, retention_id, ExpiryTick(3))
        assert expired.value.code is DurableStoreErrorCode.RETENTION_DENIED
        return (
            sum(item.replayed for item in reservations),
            len(backend.records),
            cleanup.records_deleted,
        )

    assert run(reserve()) == (3, 1, 1)


def test_patch_e2e_040_provider_json_durable_reconciliation() -> None:
    """Settle provider JSON, re-inject its result, then permit a later read."""
    assert PATCH_APPLY_SCHEMA["function"]["name"] == "patch.apply"

    class LaterReadTool(Tool):
        """Read the post-settlement deterministic service state.

        Returns:
            The bounded current document state.
        """

        def __init__(self, service: _Phase15DurableService) -> None:
            """Bind the read tool to the settled deterministic service."""
            Tool.__init__(self)
            self.__name__ = "read"
            self._service = service

        async def __call__(self, context: ToolCallContext) -> dict[str, str]:
            """Return the only later-read state.

            Returns:
                The document state after durable settlement.
            """
            assert context.patch_capability is None
            return {"document": self._service.document}

    async def scenario() -> None:
        service = _Phase15DurableService(pending=True)
        ordinary = ToolSet(namespace="phase15", tools=[LaterReadTool(service)])
        bundle = await _phase15_public_bundle(
            service, ordinary_toolsets=(ordinary,)
        )
        assert bundle.toolset is not None
        assert isinstance(bundle.manager, ToolManager)
        catalog = ModelCapabilityCatalog.create(
            bundle.manager.export_model_capability_seed()
        )
        provider_name = catalog.provider_name(
            "patch.edit", provider_family="phase15-provider"
        )
        decoded = catalog.decode_call(
            ProviderCapabilityCall(
                call_id="phase15-provider-json",
                provider_name=provider_name,
                arguments=(
                    '{"path":"note.txt","edits":['
                    '{"old_text":"before","new_text":"after"}]}'
                ),
            ),
            provider_family="phase15-provider",
        )
        assert isinstance(decoded, ToolCall)
        provider = await bundle.manager.execute_call(
            decoded,
            ToolCallContext(patch_capability=bundle.toolset.capability),
        )
        assert isinstance(provider, ToolCallResult)
        assert _result_mapping(provider)["status"] == "committed"
        assert service.invocations == 1
        assert service.approvals == 1
        assert service.cleanup_count == 1
        assert len(service._backend.records) == 1
        assert [event.lifecycle for event in service.events] == [
            LifecyclePhase.SETTLEMENT_PENDING,
            LifecyclePhase.REQUEST_COMPLETED,
        ]
        assert all("note.txt" not in str(event) for event in service.events)
        assert (
            service._activation_observer is bundle.toolset._activation_runtime
        )
        assert not service._claims
        runtime = service._activation_observer
        assert runtime is not None
        assert (
            await runtime.registry.active_binding_count(runtime.lease.key) == 0
        )
        assert service.last_access is not None
        durable_events = await service.store.outbox(
            service.last_access, SequenceNumber(0), 8
        )
        assert [event.lifecycle for event in durable_events] == [
            LifecyclePhase.SETTLEMENT_PENDING,
            LifecyclePhase.REQUEST_COMPLETED,
        ]
        assert all("note.txt" not in str(event) for event in durable_events)

        reinjected = OrchestratorResponse._tool_observation_messages(
            provider, call=decoded, json_output=True
        )
        assert len(reinjected) == 2
        assert reinjected[0].tool_calls is not None
        assert reinjected[1].tool_call_result is not None

        continuation = await bundle.manager.execute_call(
            ToolCall(id="phase15-later-read", name="phase15.read"),
            ToolCallContext(patch_capability=bundle.toolset.capability),
        )
        assert isinstance(continuation, ToolCallResult)
        assert continuation.result == {"document": "after\n"}

        inert_manager = ToolManager.create_instance(
            available_toolsets=(), enable_tools=[]
        )
        inert_catalog = ModelCapabilityCatalog.create(
            inert_manager.export_model_capability_seed()
        )
        assert inert_catalog.descriptors == ()

        retry_service = _Phase15DurableService(pending=True, hold_pending=True)
        retry_bundle = await _phase15_public_bundle(retry_service)
        assert retry_bundle.toolset is not None
        assert decoded.raw_arguments is not None
        async with retry_bundle.manager:
            host = retry_bundle.toolset.sdk_host()
            pending = await host.invoke_raw(
                OperationType.EDIT, decoded.raw_arguments
            )
            assert isinstance(pending, PatchPending)
            review = await host.prepare_approval_review()
            detached_host = retry_bundle.toolset.sdk_host()
            with pytest.raises(PatchToolError):
                await detached_host.approve_review(review)
            assert await host.approve_review(review) == pending
            host.validate_invocation_review(review)
            retries = await gather(
                host.retransmit_raw(
                    OperationType.EDIT,
                    decoded.raw_arguments,
                    pending.request_id,
                    pending.correlation_id,
                ),
                host.retransmit_raw(
                    OperationType.EDIT,
                    decoded.raw_arguments,
                    pending.request_id,
                    pending.correlation_id,
                ),
            )
            assert retries == [pending, pending]
            assert retry_service.invocations == 1
            assert len(retry_service._backend.records) == 1
            resumed_host = retry_bundle.toolset.sdk_host()
            suspended = create_task(resumed_host.await_terminal(pending))
            await sleep(0)
            assert not suspended.done()
            settled = await retry_service.settle(pending)
            assert await suspended == settled
            runtime = retry_service._activation_observer
            assert runtime is retry_bundle.toolset._activation_runtime
            assert runtime is not None
            assert (
                await runtime.registry.active_binding_count(runtime.lease.key)
                == 0
            )

    run(scenario())


def test_patch_e2e_042_provider_limit_matrix_uses_exact_owner_boundaries() -> (
    None
):
    """Bind every configured limit to its provider, policy, or host owner."""
    base = replace(_phase15_limits(), input_bytes=ByteSize(512))
    stricter = PatchLimits(
        ByteSize(511),
        ByteSize(7),
        ByteSize(95),
        ByteSize(7),
        ByteSize(7),
        ByteSize(4095),
        ByteSize(4095),
        ByteSize(4095),
        DurationTicks(99),
        DurationTicks(99),
        DurationTicks(99),
    )
    effective = compose_limits(
        base,
        stricter,
        base,
        base,
        base,
    ).value
    limit_fields = (
        "input_bytes",
        "path_count",
        "path_length",
        "file_count",
        "operation_count",
        "snapshot_bytes",
        "proposed_bytes",
        "review_diff_bytes",
        "planning_duration",
        "approval_duration",
        "commit_duration",
    )
    assert tuple(
        getattr(effective, field).value for field in limit_fields
    ) == tuple(getattr(stricter, field).value for field in limit_fields)

    async def scenario() -> None:
        def planner_ceiling(
            *,
            file_snapshot: int = 100_000,
            snapshot: int = 100_000,
            file_proposed: int = 100_000,
            proposed: int = 100_000,
            changed: int = 100_000,
            candidates: int = 100_000,
            diff_work: int = 100_000,
            diff: int = 100_000,
            memory: int = 1_000_000,
        ) -> PlannerLimits:
            """Return an explicit ceiling for each planner-owned resource."""
            return PlannerLimits(
                file_snapshot,
                snapshot,
                file_proposed,
                proposed,
                changed,
                candidates,
                diff_work,
                diff,
                memory,
            )

        def parser_ceiling(
            *, hunks: int = 32, replacement: int = 512
        ) -> PatchInputLimits:
            """Return a parser configuration with bounded hunk content."""
            return PatchInputLimits(
                max_raw_bytes=512,
                max_hunks=hunks,
                max_content_bytes=replacement,
            )

        async def execute_provider(
            bundle: PatchToolManagerBundle,
            *,
            operation: OperationType,
            raw: bytes,
            call_id: str,
        ) -> ToolCallResult:
            """Decode and dispatch one provider call through ToolManager."""
            assert bundle.toolset is not None
            catalog = ModelCapabilityCatalog.create(
                bundle.manager.export_model_capability_seed()
            )
            name = (
                "patch.edit"
                if operation is OperationType.EDIT
                else "patch.apply"
            )
            decoded = catalog.decode_call(
                ProviderCapabilityCall(
                    call_id=call_id,
                    provider_name=catalog.provider_name(
                        name,
                        provider_family="phase15-limit-provider",
                    ),
                    arguments=raw.decode("utf-8"),
                ),
                provider_family="phase15-limit-provider",
            )
            outcome = await bundle.manager.execute_call(
                decoded,
                ToolCallContext(patch_capability=bundle.toolset.capability),
            )
            assert isinstance(outcome, ToolCallResult)
            return outcome

        async def invoke_provider(
            *,
            limits: PatchLimits,
            operation: OperationType,
            raw: bytes,
            document: str = "before\n",
            parser_limits: PatchInputLimits | None = None,
            planner_limits: PlannerLimits | None = None,
            additional_files: tuple[PlannerFile, ...] = (),
            policy: TrustedPatchPolicy | None = None,
            pending: bool = False,
            hold_pending: bool = False,
            max_in_flight: int = 64,
            durable_step_count: int = 1,
            planning_delay_ticks: int = 0,
            approval_delay_ticks: int = 0,
            commit_delay_ticks: int = 0,
        ) -> tuple[ToolCallResult, _Phase15DurableService]:
            """Decode one provider call through the durable public tool."""
            service = _Phase15DurableService(
                pending=pending,
                hold_pending=hold_pending,
                initial_document=document,
                parser_limits=(
                    parser_limits
                    or PatchInputLimits(max_raw_bytes=limits.input_bytes.value)
                ),
                planner_limits=planner_limits,
                additional_files=additional_files,
                max_in_flight=max_in_flight,
                durable_step_count=durable_step_count,
                planning_delay_ticks=planning_delay_ticks,
                approval_delay_ticks=approval_delay_ticks,
                commit_delay_ticks=commit_delay_ticks,
            )
            bundle = await _phase15_public_bundle(
                service,
                limits=limits,
                policy=policy,
            )
            assert isinstance(bundle.manager, ToolManager)
            assert bundle.toolset is not None
            assert bundle.toolset._snapshot.settlement_duration == (
                limits.commit_duration
            )
            outcome = await execute_provider(
                bundle,
                operation=operation,
                raw=raw,
                call_id="phase15-limit-" + operation.value,
            )
            return outcome, service

        edit = (
            b'{"path":"note.txt","edits":['
            b'{"old_text":"before","new_text":"after"}]}'
        )
        rejected, rejected_service = await invoke_provider(
            limits=replace(base, input_bytes=ByteSize(len(edit) - 1)),
            operation=OperationType.EDIT,
            raw=edit,
        )
        assert _result_mapping(rejected)["status"] == "rejected"
        assert rejected_service.parse_attempts == 0
        committed, committed_service = await invoke_provider(
            limits=replace(base, input_bytes=ByteSize(len(edit))),
            operation=OperationType.EDIT,
            raw=edit,
        )
        assert _result_mapping(committed)["status"] == "committed"
        assert committed_service.parse_attempts == 1
        assert committed_service.invocations == 1
        assert len(committed_service._backend.records) == 1

        path_rejected, path_rejected_service = await invoke_provider(
            limits=replace(base, path_length=ByteSize(len("note.txt") - 1)),
            operation=OperationType.EDIT,
            raw=edit,
        )
        assert _result_mapping(path_rejected)["status"] == "rejected"
        assert path_rejected_service.parse_attempts == 0
        path_committed, path_committed_service = await invoke_provider(
            limits=replace(base, path_length=ByteSize(len("note.txt"))),
            operation=OperationType.EDIT,
            raw=edit,
        )
        assert _result_mapping(path_committed)["status"] == "committed"
        assert path_committed_service.invocations == 1

        two_edits = (
            b'{"path":"note.txt","edits":['
            b'{"old_text":"before","new_text":"after"},'
            b'{"old_text":"middle","new_text":"mid"}]}'
        )
        operations_rejected, operations_rejected_service = (
            await invoke_provider(
                limits=replace(base, operation_count=ByteSize(1)),
                operation=OperationType.EDIT,
                raw=two_edits,
                document="before\nmiddle\n",
            )
        )
        assert _result_mapping(operations_rejected)["status"] == "rejected"
        assert operations_rejected_service.parse_attempts == 0
        operations_committed, operations_committed_service = (
            await invoke_provider(
                limits=replace(base, operation_count=ByteSize(2)),
                operation=OperationType.EDIT,
                raw=two_edits,
                document="before\nmiddle\n",
            )
        )
        assert _result_mapping(operations_committed)["status"] == "committed"
        assert operations_committed_service.invocations == 1

        proposed = (
            b'{"path":"note.txt","edits":['
            b'{"old_text":"before","new_text":"abcdefgh"}]}'
        )
        proposed_rejected, proposed_rejected_service = await invoke_provider(
            limits=replace(base, proposed_bytes=ByteSize(7)),
            operation=OperationType.EDIT,
            raw=proposed,
        )
        assert _result_mapping(proposed_rejected)["status"] == "rejected"
        assert proposed_rejected_service.parse_attempts == 0
        proposed_committed, proposed_committed_service = await invoke_provider(
            limits=replace(base, proposed_bytes=ByteSize(9)),
            operation=OperationType.EDIT,
            raw=proposed,
        )
        assert _result_mapping(proposed_committed)["status"] == "committed"
        assert proposed_committed_service.invocations == 1

        two_paths = (
            b'{"patch":"*** Begin Patch v1\\n*** Add File: one.txt\\n'
            b'+one\\n*** Add File: two.txt\\n+two\\n*** End Patch"}'
        )
        paths_rejected, paths_rejected_service = await invoke_provider(
            limits=replace(base, path_count=ByteSize(1)),
            operation=OperationType.APPLY,
            raw=two_paths,
        )
        assert _result_mapping(paths_rejected)["status"] == "rejected"
        assert paths_rejected_service.parse_attempts == 0
        paths_committed, paths_committed_service = await invoke_provider(
            limits=replace(base, path_count=ByteSize(2)),
            operation=OperationType.APPLY,
            raw=two_paths,
        )
        assert _result_mapping(paths_committed)["status"] == "committed"
        assert paths_committed_service.invocations == 1

        files_rejected, files_rejected_service = await invoke_provider(
            limits=replace(base, file_count=ByteSize(1)),
            operation=OperationType.APPLY,
            raw=two_paths,
        )
        assert _result_mapping(files_rejected)["status"] == "rejected"
        assert files_rejected_service.parse_attempts == 1
        assert files_rejected_service.invocations == 0
        assert not files_rejected_service._backend.records
        files_committed, files_committed_service = await invoke_provider(
            limits=replace(base, file_count=ByteSize(2)),
            operation=OperationType.APPLY,
            raw=two_paths,
            durable_step_count=2,
        )
        assert _result_mapping(files_committed)["status"] == "committed"
        assert files_committed_service.parse_attempts == 1
        assert files_committed_service.invocations == 1
        assert files_committed_service.last_access is not None
        artifact_free_journal = (
            await files_committed_service.store.inspect(
                files_committed_service.last_access
            )
        ).journal
        assert [entry.state for entry in artifact_free_journal.steps] == [
            CommitStepState.PLANNED,
            CommitStepState.COMMITTED,
            CommitStepState.PLANNED,
            CommitStepState.COMMITTED,
        ]
        assert artifact_free_journal.artifacts == ()
        assert [
            entry.cursor.revision.value
            for entry in artifact_free_journal.steps
        ] == [1, 2, 3, 4]
        assert artifact_free_journal.cursor.revision.value == 4

        snapshot_rejected, snapshot_rejected_service = await invoke_provider(
            limits=replace(base, snapshot_bytes=ByteSize(6)),
            operation=OperationType.EDIT,
            raw=edit,
        )
        assert _result_mapping(snapshot_rejected)["status"] == "rejected"
        assert snapshot_rejected_service.last_planned_snapshot_bytes == 7
        assert snapshot_rejected_service.parse_attempts == 1
        assert snapshot_rejected_service.invocations == 0
        snapshot_committed, snapshot_committed_service = await invoke_provider(
            limits=replace(base, snapshot_bytes=ByteSize(7)),
            operation=OperationType.EDIT,
            raw=edit,
        )
        assert _result_mapping(snapshot_committed)["status"] == "committed"
        assert snapshot_committed_service.last_planned_snapshot_bytes == 7
        assert snapshot_committed_service.invocations == 1

        review_rejected, review_rejected_service = await invoke_provider(
            limits=replace(base, review_diff_bytes=ByteSize(52)),
            operation=OperationType.EDIT,
            raw=edit,
        )
        assert _result_mapping(review_rejected)["status"] == "rejected"
        assert review_rejected_service.parse_attempts == 1
        assert review_rejected_service.invocations == 0
        review_committed, review_committed_service = await invoke_provider(
            limits=replace(base, review_diff_bytes=ByteSize(53)),
            operation=OperationType.EDIT,
            raw=edit,
        )
        assert _result_mapping(review_committed)["status"] == "committed"
        assert review_committed_service.parse_attempts == 1
        assert review_committed_service.invocations == 1

        async def assert_planner_boundary(
            *,
            operation: OperationType,
            raw: bytes,
            rejected_limits: PlannerLimits,
            accepted_limits: PlannerLimits,
            document: str = "before\n",
            additional_files: tuple[PlannerFile, ...] = (),
        ) -> None:
            """Assert one real planner resource N-minus-one and N boundary."""
            rejected, rejected_service = await invoke_provider(
                limits=base,
                operation=operation,
                raw=raw,
                document=document,
                planner_limits=rejected_limits,
                additional_files=additional_files,
            )
            assert _result_mapping(rejected)["status"] == "rejected"
            assert rejected_service.parse_attempts == 1
            assert rejected_service.invocations == 0
            assert not rejected_service._backend.records
            accepted, accepted_service = await invoke_provider(
                limits=base,
                operation=operation,
                raw=raw,
                document=document,
                planner_limits=accepted_limits,
                additional_files=additional_files,
            )
            assert _result_mapping(accepted)["status"] == "committed"
            assert accepted_service.parse_attempts == 1
            assert accepted_service.invocations == 1
            assert len(accepted_service._backend.records) == 1

        two_hunks = (
            b'{"patch":"*** Begin Patch v1\\n*** Update File: note.txt\\n@@\\n'
            b"-before\\n+after\\n*** Update File: other.txt\\n@@\\n"
            b'-second\\n+next\\n*** End Patch"}'
        )
        hunks_rejected, hunks_rejected_service = await invoke_provider(
            limits=base,
            operation=OperationType.APPLY,
            raw=two_hunks,
            additional_files=(_file("other.txt", b"second\n"),),
            parser_limits=parser_ceiling(hunks=1),
        )
        assert _result_mapping(hunks_rejected)["status"] == "rejected"
        assert hunks_rejected_service.parse_attempts == 0
        assert not hunks_rejected_service._backend.records
        hunks_committed, hunks_committed_service = await invoke_provider(
            limits=base,
            operation=OperationType.APPLY,
            raw=two_hunks,
            additional_files=(_file("other.txt", b"second\n"),),
            parser_limits=parser_ceiling(hunks=2),
        )
        assert _result_mapping(hunks_committed)["status"] == "committed"
        assert hunks_committed_service.invocations == 1

        replacement_rejected, replacement_rejected_service = (
            await invoke_provider(
                limits=base,
                operation=OperationType.EDIT,
                raw=proposed,
                parser_limits=parser_ceiling(replacement=7),
            )
        )
        assert _result_mapping(replacement_rejected)["status"] == "rejected"
        assert replacement_rejected_service.parse_attempts == 0
        assert not replacement_rejected_service._backend.records
        replacement_committed, replacement_committed_service = (
            await invoke_provider(
                limits=base,
                operation=OperationType.EDIT,
                raw=proposed,
                parser_limits=parser_ceiling(replacement=8),
            )
        )
        assert _result_mapping(replacement_committed)["status"] == "committed"
        assert replacement_committed_service.invocations == 1

        await assert_planner_boundary(
            operation=OperationType.EDIT,
            raw=edit,
            rejected_limits=planner_ceiling(file_snapshot=6),
            accepted_limits=planner_ceiling(file_snapshot=7),
        )
        await assert_planner_boundary(
            operation=OperationType.EDIT,
            raw=edit,
            rejected_limits=planner_ceiling(snapshot=13),
            accepted_limits=planner_ceiling(snapshot=14),
            additional_files=(_file("other.txt", b"second\n"),),
        )
        await assert_planner_boundary(
            operation=OperationType.EDIT,
            raw=proposed,
            rejected_limits=planner_ceiling(file_proposed=8),
            accepted_limits=planner_ceiling(file_proposed=9),
        )
        await assert_planner_boundary(
            operation=OperationType.APPLY,
            raw=two_paths,
            rejected_limits=planner_ceiling(file_proposed=4, proposed=7),
            accepted_limits=planner_ceiling(file_proposed=4, proposed=8),
        )
        await assert_planner_boundary(
            operation=OperationType.EDIT,
            raw=edit,
            rejected_limits=planner_ceiling(changed=5),
            accepted_limits=planner_ceiling(changed=6),
        )
        ambiguous_match = (
            b'{"path":"note.txt","edits":[{"old_text":"x","new_text":"y"}]}'
        )
        candidates_rejected, candidates_rejected_service = (
            await invoke_provider(
                limits=base,
                operation=OperationType.EDIT,
                raw=ambiguous_match,
                document="x x x\n",
                planner_limits=planner_ceiling(candidates=1),
            )
        )
        assert _result_mapping(candidates_rejected)["status"] == "rejected"
        assert candidates_rejected_service.parse_attempts == 1
        assert candidates_rejected_service.invocations == 0
        assert not candidates_rejected_service._backend.records
        candidates_committed, candidates_committed_service = (
            await invoke_provider(
                limits=base,
                operation=OperationType.EDIT,
                raw=edit,
                planner_limits=planner_ceiling(candidates=1),
            )
        )
        assert _result_mapping(candidates_committed)["status"] == "committed"
        assert candidates_committed_service.invocations == 1
        await assert_planner_boundary(
            operation=OperationType.EDIT,
            raw=edit,
            rejected_limits=planner_ceiling(diff_work=108),
            accepted_limits=planner_ceiling(diff_work=109),
        )
        await assert_planner_boundary(
            operation=OperationType.EDIT,
            raw=edit,
            rejected_limits=planner_ceiling(diff=52),
            accepted_limits=planner_ceiling(diff=53),
        )
        await assert_planner_boundary(
            operation=OperationType.EDIT,
            raw=edit,
            rejected_limits=planner_ceiling(memory=51_730),
            accepted_limits=planner_ceiling(memory=51_731),
        )

        policy = _phase15_policy()
        staging_denied = replace(
            policy,
            rules=(
                replace(
                    policy.rules[0], staging_classes=frozenset(("untrusted",))
                ),
            ),
        )
        staging_rejected, staging_rejected_service = await invoke_provider(
            limits=base,
            operation=OperationType.EDIT,
            raw=edit,
            policy=staging_denied,
        )
        assert _result_mapping(staging_rejected)["status"] == "rejected"
        assert staging_rejected_service.parse_attempts == 1
        assert staging_rejected_service.invocations == 0
        assert not staging_rejected_service._backend.records
        staging_committed, staging_committed_service = await invoke_provider(
            limits=base,
            operation=OperationType.EDIT,
            raw=edit,
            policy=policy,
        )
        assert _result_mapping(staging_committed)["status"] == "committed"
        assert staging_committed_service.invocations == 1

        capacity_service = _Phase15DurableService(
            pending=True,
            hold_pending=True,
            parser_limits=PatchInputLimits(
                max_raw_bytes=base.input_bytes.value
            ),
            max_in_flight=1,
        )
        capacity_bundle = await _phase15_public_bundle(
            capacity_service, limits=base
        )
        first = create_task(
            execute_provider(
                capacity_bundle,
                operation=OperationType.EDIT,
                raw=edit,
                call_id="phase15-capacity-first",
            )
        )
        while capacity_service.current_outcome is None:
            await sleep(0)
        capacity_rejected = await execute_provider(
            capacity_bundle,
            operation=OperationType.EDIT,
            raw=edit,
            call_id="phase15-capacity-second",
        )
        assert _result_mapping(capacity_rejected)["status"] == "rejected"
        assert capacity_service._in_flight == 1
        assert capacity_service.invocations == 1
        assert len(capacity_service._backend.records) == 1
        pending_outcome = capacity_service.current_outcome
        assert isinstance(pending_outcome, PatchPending)
        assert (await capacity_service.settle(pending_outcome)).status is (
            PatchStatus.COMMITTED
        )
        assert _result_mapping(await first)["status"] == "committed"
        assert capacity_service._in_flight == 0

        for phase, delay_keyword in (
            ("planning", "planning_delay_ticks"),
            ("approval", "approval_delay_ticks"),
            ("commit", "commit_delay_ticks"),
        ):
            elapsed_start = get_running_loop().time()
            rejected, rejected_service = await invoke_provider(
                limits=replace(
                    base, **{phase + "_duration": DurationTicks(1)}
                ),
                operation=OperationType.EDIT,
                raw=edit,
                **{delay_keyword: 8},
            )
            elapsed = get_running_loop().time() - elapsed_start
            assert _result_mapping(rejected)["status"] == "rejected"
            assert rejected_service.limit_timeouts == [phase]
            assert rejected_service.parse_attempts == 1
            assert rejected_service.invocations == 0
            assert 0.0005 <= elapsed < 0.25
            committed, committed_service = await invoke_provider(
                limits=replace(
                    base, **{phase + "_duration": DurationTicks(20)}
                ),
                operation=OperationType.EDIT,
                raw=edit,
                **{delay_keyword: 1},
            )
            assert _result_mapping(committed)["status"] == "committed"
            assert committed_service.limit_timeouts == []
            assert committed_service.invocations == 1

    run(scenario())


def test_patch_phase_15_privacy_and_protocol_boundaries_are_coarse() -> None:
    """Coarsen denials without a wall-clock disclosure claim."""
    assert (
        coarsen_error_code(PatchErrorCode.ALIAS_DENIED, Audience.PUBLIC)
        is PatchErrorCode.PATH_DENIED
    )
    assert all(
        coarsen_error_code(code, Audience.PUBLIC) is PatchErrorCode.PATH_DENIED
        for code in (
            PatchErrorCode.PATH_DENIED,
            PatchErrorCode.ALIAS_DENIED,
            PatchErrorCode.MOUNT_DENIED,
        )
    )
    assert all(
        not PatchProtocolProfile(surface).active
        for surface in PatchProtocolSurface
    )


@dataclass(frozen=True)
class _ActivationCoverageArtifacts:
    """Keep one selected test manifest and its sealed receipt together."""

    manifest: PatchProductionManifest
    profile: PatchCapabilityProfile
    authority: PatchActivationRuntimeAuthority
    verifier: PatchActivationVerifier
    store: InMemoryDurablePatchStore
    record: PatchActivationRuntimeRecord
    receipt: PatchVerifiedActivationReceipt


def _activation_coverage_artifacts() -> _ActivationCoverageArtifacts:
    """Return a complete non-production manifest with one selected profile."""
    production = build_patch_production_manifest()
    profile = replace(
        production.profiles[0],
        proofs=PatchProfileProofs(
            context=True,
            platform=True,
            filesystem=True,
            target=True,
            protocol=True,
            policy=True,
            approval=True,
            persistence=True,
            surface=True,
            provider_codec=True,
        ),
        state=PatchProfileState.SELECTED,
        selection_rationale="Phase 15 activation coverage profile.",
    )
    manifest = _manifest(
        sources=production.sources,
        schemas=production.schemas,
        protocols=production.protocols,
        profiles=(profile,),
    )
    authority = _new_activation_authority(b"a" * 32)
    verifier = _build_activation_verifier(
        manifest, authority, production=False
    )
    store = InMemoryDurablePatchStore(InMemoryDurablePatchBackend())
    record = PatchActivationRuntimeRecord(profile.key, "b" * 64, store)
    receipt = verifier._runtime_receipt(record)
    assert receipt is not None
    return _ActivationCoverageArtifacts(
        manifest, profile, authority, verifier, store, record, receipt
    )


def _activation_coverage_binding(
    store: InMemoryDurablePatchStore,
    service: object,
    platform: object = "macos",
) -> object:
    """Return the exact completed handshake shape required by activation."""
    identity = SimpleNamespace(
        target_id=SimpleNamespace(value="target_activation_coverage"),
        workspace_id=SimpleNamespace(value="workspace_activation_coverage"),
        domain_id=SimpleNamespace(value="domain_activation_coverage"),
    )
    return SimpleNamespace(
        scope=SimpleNamespace(
            context_kind=ContextKind.SANDBOX,
            identity=identity,
        ),
        handshake=SimpleNamespace(platform=SimpleNamespace(value=platform)),
        coordinator=SimpleNamespace(durable_store=store),
        persistence=SimpleNamespace(durable_store=store),
        policy=SimpleNamespace(revision=SimpleNamespace(value="policy-v1")),
        service=service,
    )


def test_patch_phase_15_activation_value_and_receipt_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject forged activation values, receipts, and manifest coordinates."""
    artifacts = _activation_coverage_artifacts()
    key = artifacts.profile.key
    with pytest.raises(PatchActivationError):
        PatchActivationLimits(max_active_profiles=0)
    with pytest.raises(PatchActivationError):
        PatchProfileComponent("not a component")
    with pytest.raises(PatchActivationError):
        replace(key, context=cast(ContextKind, "sandbox"))
    assert key.digest == key.digest
    with pytest.raises(PatchActivationError):
        PatchProfileProofs(
            context=True,
            platform=True,
            filesystem=True,
            target=True,
            protocol=True,
            policy=True,
            approval=True,
            persistence=True,
            surface=True,
            provider_codec=cast(bool, 1),
        )
    with pytest.raises(PatchActivationError):
        replace(artifacts.profile, capability_inventory=())
    with pytest.raises(PatchActivationError):
        replace(
            artifacts.profile,
            proofs=replace(artifacts.profile.proofs, platform=False),
        )
    with pytest.raises(PatchActivationError):
        replace(artifacts.profile, state=PatchProfileState.INCOMPLETE)
    with pytest.raises(PatchActivationError):
        PatchSchemaDescription("patch.edit", "{}", "0" * 64)
    with pytest.raises(PatchActivationError):
        PatchProtocolDescription(PatchProtocolSurface.MCP, ("patch.read",))
    with pytest.raises(PatchActivationError):
        PatchProductionSource("not.production", "source")
    with pytest.raises(PatchActivationError):
        replace(artifacts.manifest, manifest_sha256="0" * 64)
    with pytest.raises(PatchActivationError):
        artifacts.manifest.profile_for(
            cast(PatchActivationProfileKey, object())
        )
    with pytest.raises(PatchActivationError):
        artifacts.manifest.profile_for(
            replace(key, version=PatchProfileComponent("v2"))
        )
    with pytest.raises(PatchActivationError):
        PatchActivationLease(key, 0, ())
    with pytest.raises(PatchActivationError):
        PatchActivationRuntimeRecord(key, "not-a-digest", artifacts.store)
    with pytest.raises(PatchActivationError):
        PatchActivationDurableOperation(
            PatchRequestId.new(),
            PatchCommitOwnerId("owner_" + "a" * 16),
            SequenceNumber(0),
        )
    with pytest.raises(PatchActivationError):
        PatchActivationOperationBinding(
            cast(PatchRequestId, object()),
            PatchCommitOwnerId("owner_" + "a" * 16),
            SequenceNumber(1),
            PatchActivationLease(key, 1, ("patch.edit",)),
            PatchActivationOperationState.IN_FLIGHT,
        )
    with pytest.raises(PatchActivationError):
        PatchDeactivationReceipt(
            cast(PatchActivationProfileKey, object()), None
        )
    with pytest.raises(PatchActivationError):
        PatchActivationRuntimeAuthority()
    forged_authority = object.__new__(PatchActivationRuntimeAuthority)
    object.__setattr__(forged_authority, "_key", b"a" * 32)
    object.__setattr__(forged_authority, "_issuer", object())
    with pytest.raises(PatchActivationError):
        forged_authority._sign(b"payload")
    with pytest.raises(PatchActivationError):
        _new_activation_authority(b"short")
    with pytest.raises(PatchActivationError):
        PatchVerifiedActivationReceipt()
    with pytest.raises(PatchActivationError):
        copy(artifacts.receipt)
    with pytest.raises(PatchActivationError):
        deepcopy(artifacts.receipt)
    with pytest.raises(PatchActivationError):
        PatchActivationVerifier()
    forged_verifier = object.__new__(PatchActivationVerifier)
    for name in ("_manifest", "_authority", "_production"):
        object.__setattr__(
            forged_verifier, name, getattr(artifacts.verifier, name)
        )
    object.__setattr__(forged_verifier, "_issuer", object())
    with pytest.raises(PatchActivationError):
        forged_verifier._runtime_receipt(artifacts.record)
    with pytest.raises(PatchActivationError):
        artifacts.verifier._runtime_receipt(
            cast(PatchActivationRuntimeRecord, object())
        )
    assert (
        artifacts.verifier._runtime_receipt(
            PatchActivationRuntimeRecord(
                replace(key, version=PatchProfileComponent("v2")),
                "c" * 64,
                artifacts.store,
            )
        )
        is None
    )
    production = build_patch_production_manifest()
    production_verifier = build_patch_activation_verifier(production)
    monkeypatch.setattr(
        "avalan.patch.activation.build_patch_production_manifest",
        lambda: artifacts.manifest,
    )
    with pytest.raises(PatchActivationError):
        production_verifier._runtime_receipt(artifacts.record)
    monkeypatch.undo()
    with pytest.raises(PatchActivationError):
        build_patch_activation_verifier(artifacts.manifest)
    with pytest.raises(PatchActivationError):
        _build_activation_verifier(
            cast(PatchProductionManifest, object()),
            artifacts.authority,
            production=False,
        )
    with pytest.raises(PatchActivationError):
        _issue_verified_receipt(
            cast(PatchProductionManifest, object()),
            artifacts.profile,
            artifacts.record,
            artifacts.authority,
        )
    with pytest.raises(PatchActivationError):
        _issue_verified_receipt(
            artifacts.manifest,
            replace(
                artifacts.profile,
                selection_rationale="substituted profile",
            ),
            artifacts.record,
            artifacts.authority,
        )
    assert not _receipt_matches(
        cast(PatchVerifiedActivationReceipt, object()),
        artifacts.manifest,
        artifacts.verifier,
    )
    missing = _activation_coverage_artifacts()
    object.__setattr__(
        missing.receipt,
        "profile_key",
        replace(key, version=PatchProfileComponent("v2")),
    )
    assert not _receipt_matches(
        missing.receipt, artifacts.manifest, artifacts.verifier
    )
    mismatched = _activation_coverage_artifacts()
    object.__setattr__(mismatched.receipt, "profile_sha256", "0" * 64)
    assert not _receipt_matches(
        mismatched.receipt, mismatched.manifest, mismatched.verifier
    )
    unproven = _activation_coverage_artifacts()
    object.__setattr__(unproven.profile, "state", PatchProfileState.INCOMPLETE)
    object.__setattr__(
        unproven.receipt, "profile_sha256", unproven.profile.digest
    )
    assert not _receipt_matches(
        unproven.receipt, unproven.manifest, unproven.verifier
    )
    with pytest.raises(PatchActivationError):
        render_patch_production_manifest(
            cast(PatchProductionManifest, object())
        )
    with pytest.raises(PatchActivationError):
        _production_evidence_digest(
            cast(PatchProductionManifest, object()), artifacts.profile
        )
    with pytest.raises(PatchActivationError):
        activation_module._schema_description({"function": {}})


def test_patch_phase_15_activation_registry_and_factory_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fence stale durable owners and unsealed activation factory outcomes."""

    class ObservedService:
        """Retain an observer attached by a successful factory activation."""

        def __init__(self) -> None:
            """Initialize one empty activation observer slot."""
            self.observer: object | None = None

        def set_activation_observer(self, observer: object) -> None:
            """Attach the concrete runtime exactly once."""
            self.observer = observer

    async def scenario() -> None:
        artifacts = _activation_coverage_artifacts()
        with pytest.raises(PatchActivationError):
            PatchActivationRegistry(
                cast(PatchProductionManifest, object()), artifacts.verifier
            )
        registry = PatchActivationRegistry(
            artifacts.manifest, artifacts.verifier
        )
        with pytest.raises(PatchActivationError):
            await registry.activate(
                cast(PatchVerifiedActivationReceipt, object())
            )
        with pytest.raises(PatchActivationError):
            await registry.bind_operation(
                artifacts.receipt,
                cast(PatchActivationDurableOperation, object()),
                PatchActivationOperationState.IN_FLIGHT,
            )
        dormant = PatchActivationRegistry(
            artifacts.manifest, artifacts.verifier
        )
        durable = PatchActivationDurableOperation(
            PatchRequestId.new(),
            PatchCommitOwnerId("owner_" + "a" * 16),
            SequenceNumber(1),
        )
        with pytest.raises(PatchActivationError):
            await dormant.bind_operation(
                artifacts.receipt,
                durable,
                PatchActivationOperationState.IN_FLIGHT,
            )
        lease = await registry.activate(artifacts.receipt)
        binding = await registry.bind_operation(
            artifacts.receipt,
            durable,
            PatchActivationOperationState.IN_FLIGHT,
        )
        with pytest.raises(PatchActivationError):
            await registry.bind_operation(
                artifacts.receipt,
                PatchActivationDurableOperation(
                    durable.request_id,
                    PatchCommitOwnerId("owner_" + "b" * 16),
                    durable.fence,
                ),
                PatchActivationOperationState.IN_FLIGHT,
            )
        assert (
            await registry.bind_operation(
                artifacts.receipt,
                durable,
                PatchActivationOperationState.IN_FLIGHT,
            )
            is binding
        )
        with pytest.raises(PatchActivationError):
            await registry.retain_operation(
                lease.key,
                durable,
                PatchActivationOperationState.IN_FLIGHT,
            )
        with pytest.raises(PatchActivationError):
            await registry.retain_operation(
                lease.key,
                PatchActivationDurableOperation(
                    PatchRequestId.new(), durable.owner, durable.fence
                ),
                PatchActivationOperationState.PARTIAL,
            )
        with pytest.raises(PatchActivationError):
            await registry.active_binding_count(
                cast(PatchActivationProfileKey, object())
            )
        assert await registry.advertised_tools(lease.key) == (
            "patch.edit",
            "patch.apply",
        )
        with pytest.raises(PatchActivationError):
            await registry.advertised_tools(
                cast(PatchActivationProfileKey, object())
            )
        with pytest.raises(PatchActivationError):
            await registry.deactivate(
                cast(PatchActivationProfileKey, object())
            )
        with pytest.raises(PatchActivationError):
            await registry.release_operation(lease.key, durable, 0)
        with pytest.raises(PatchActivationError):
            await registry.release_operation(
                lease.key,
                PatchActivationDurableOperation(
                    durable.request_id,
                    PatchCommitOwnerId("owner_" + "c" * 16),
                    durable.fence,
                ),
                lease.epoch,
            )
        assert await registry.release_operation(
            lease.key, durable, lease.epoch
        )
        assert (
            await registry.deactivate(lease.key)
        ).retired_epoch == lease.epoch

        with pytest.raises(PatchActivationError):
            PatchActivationRuntimeFactory()
        factory = _build_activation_factory(
            artifacts.manifest, artifacts.verifier
        )
        forged_factory = object.__new__(PatchActivationRuntimeFactory)
        for name in ("_manifest", "_verifier"):
            object.__setattr__(forged_factory, name, getattr(factory, name))
        object.__setattr__(forged_factory, "_issuer", object())
        with pytest.raises(PatchActivationError):
            await forged_factory.activate(
                _activation_coverage_binding(artifacts.store, object())
            )
        assert _runtime_record(artifacts.manifest, object()) is None
        invalid_platform = _activation_coverage_binding(
            artifacts.store, object(), platform="unsupported"
        )
        assert _runtime_record(artifacts.manifest, invalid_platform) is None
        assert (
            await factory.activate(
                _activation_coverage_binding(artifacts.store, object())
            )
            is None
        )

        no_receipt_store = InMemoryDurablePatchStore(
            InMemoryDurablePatchBackend()
        )

        def no_receipt(
            _: PatchActivationVerifier, __: PatchActivationRuntimeRecord
        ) -> PatchVerifiedActivationReceipt | None:
            """Model a profile whose runtime receipt is absent."""
            return None

        monkeypatch.setattr(
            PatchActivationVerifier, "_runtime_receipt", no_receipt
        )
        assert (
            await factory.activate(
                _activation_coverage_binding(no_receipt_store, object())
            )
            is None
        )
        monkeypatch.undo()

        observed_store = InMemoryDurablePatchStore(
            InMemoryDurablePatchBackend()
        )
        observed = ObservedService()
        runtime = await factory.activate(
            _activation_coverage_binding(observed_store, observed)
        )
        assert isinstance(runtime, PatchActivationRuntime)
        assert observed.observer is runtime
        assert not activation_module.validates_patch_activation_runtime(
            factory, object(), runtime
        )
        await runtime.deactivate()
        with pytest.raises(PatchActivationError):
            await runtime.bind_durable_commit(
                DurableCommitLease(
                    PatchRequestId.new(),
                    PatchDomainId("domain_" + "a" * 16),
                    PatchCommitOwnerId("owner_" + "d" * 16),
                    SequenceNumber(1),
                    ExpiryTick(1),
                )
            )
        with pytest.raises(PatchActivationError):
            _durable_operation(cast(DurableCommitLease, object()))
        with pytest.raises(PatchActivationError):
            _build_activation_factory(
                artifacts.manifest,
                cast(PatchActivationVerifier, object()),
            )

    run(scenario())
