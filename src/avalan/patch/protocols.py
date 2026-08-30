"""Define fail-closed protocol and orchestration patch activation.

The production server deliberately does not import this module.  It is the
shared typed boundary that a future explicit protocol test profile must bind
before MCP, A2A, flow, task, multi-agent, or provider projections may expose
patch authority.  Incomplete profiles remain inert.
"""

from asyncio import Future, Task, create_task, get_running_loop, shield
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Protocol, TypeVar, final

from avalan.flow.flow import FlowSuspension
from avalan.patch.coordinator import RetransmissionKey, WorkerReport
from avalan.patch.domain import (
    AlgorithmDigest,
    DurationTicks,
    ExpiryTick,
    LifecyclePhase,
    LogicalPath,
    OperationType,
    PatchCommitOwnerId,
    PatchContextId,
    PatchDomainId,
    PatchExecutionId,
    PatchInvocationOutcome,
    PatchObserverCorrelationId,
    PatchPending,
    PatchPendingOperationId,
    PatchRequestId,
    PatchResult,
    PatchWorkspaceId,
)
from avalan.patch.durable_coordinator import (
    DurableArtifactObservation,
    DurablePatchReconciler,
)
from avalan.patch.durable_store import (
    DurableApproval,
    DurableCommitClaimState,
    DurableCommitLease,
    DurableCoordinationAccess,
    DurableCoordinationAdmission,
    DurablePatchStore,
    DurablePendingRequest,
    DurablePlanReference,
    DurableProtocolOrigin,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableRequestSnapshot,
    DurableReservation,
    DurableStoreError,
)
from avalan.patch.parser import (
    CanonicalPatchRequest,
    PatchDocumentSyntax,
    PatchInputAccumulator,
    PatchInputError,
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderProfile,
    RawToolCallId,
    StructuredEditSyntax,
    UpdateDeclarationSyntax,
)
from avalan.patch.policy import (
    ApprovalDecisionState,
    BrokerDecision,
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PlanReviewRequest,
    PolicyRouteId,
    ReviewerDecision,
)
from avalan.patch.sandbox_commit import SandboxPatchSdkService
from avalan.patch.toolset import PatchSdkHost, PatchToolSet

_TEST_PROFILE_NAME = "authenticated-protocol-patch-test"
_ProtocolPlanValue = TypeVar("_ProtocolPlanValue")


class PatchProtocolError(RuntimeError):
    """Report one privacy-safe protocol activation or continuation failure."""


class PatchProtocolSurface(str, Enum):
    """Name the closed patch projection or orchestration surface."""

    MCP = "mcp"
    A2A = "a2a"
    FLOW = "flow"
    TASK = "task"
    MULTI_AGENT = "multi_agent"
    PROVIDER_FREEFORM = "provider_freeform"
    PROVIDER_NATIVE = "provider_native"


class PatchProtocolProviderItemOrigin(str, Enum):
    """Name the provider stream location of one candidate patch item."""

    CURRENT = "current"
    HISTORY = "history"
    REPLAY = "replay"
    COMPACTION = "compaction"
    TOOL_HISTORY = "tool_history"
    MCP = "mcp"
    A2A = "a2a"


class PatchProtocolCoordinationPolicy(str, Enum):
    """Name the one declared multi-agent workspace scheduling policy."""

    SERIAL = "serial"


class PatchProtocolContinuationKind(str, Enum):
    """Distinguish approval, settlement, and terminal continuation truth."""

    APPROVAL_REQUIRED = "approval_required"
    SETTLEMENT_PENDING = "settlement_pending"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class PatchProtocolChecklist:
    """Require every protocol property before an MCP or A2A advertisement."""

    canonical_input: bool = False
    trusted_authority: bool = False
    plan_approval: bool = False
    detached_resume: bool = False
    retransmission_reservation: bool = False
    owner_fence_before_effect: bool = False
    structured_terminal_result: bool = False
    branch_suspension: bool = False
    privacy_safe_events: bool = False

    def __post_init__(self) -> None:
        """Reject non-boolean activation checklist values."""
        if any(
            type(value) is not bool
            for value in (
                self.canonical_input,
                self.trusted_authority,
                self.plan_approval,
                self.detached_resume,
                self.retransmission_reservation,
                self.owner_fence_before_effect,
                self.structured_terminal_result,
                self.branch_suspension,
                self.privacy_safe_events,
            )
        ):
            raise PatchProtocolError("patch protocol profile is invalid")

    @property
    def complete(self) -> bool:
        """Return whether every protocol precondition has been proved."""
        return all(
            (
                self.canonical_input,
                self.trusted_authority,
                self.plan_approval,
                self.detached_resume,
                self.retransmission_reservation,
                self.owner_fence_before_effect,
                self.structured_terminal_result,
                self.branch_suspension,
                self.privacy_safe_events,
            )
        )


@dataclass(frozen=True, slots=True)
class PatchOrchestrationChecklist:
    """Require every retry and dependency property before orchestration."""

    shared_coordinator: bool = False
    originating_identity: bool = False
    approval_or_denial: bool = False
    retry_blocked_after_commit: bool = False
    durable_resume: bool = False
    dependent_suspension: bool = False
    coordinated_parallelism: bool = False
    committed_state_visibility: bool = False

    def __post_init__(self) -> None:
        """Reject non-boolean orchestration checklist values."""
        if any(
            type(value) is not bool
            for value in (
                self.shared_coordinator,
                self.originating_identity,
                self.approval_or_denial,
                self.retry_blocked_after_commit,
                self.durable_resume,
                self.dependent_suspension,
                self.coordinated_parallelism,
                self.committed_state_visibility,
            )
        ):
            raise PatchProtocolError("patch orchestration profile is invalid")

    @property
    def complete(self) -> bool:
        """Return whether every orchestration precondition has been proved."""
        return all(
            (
                self.shared_coordinator,
                self.originating_identity,
                self.approval_or_denial,
                self.retry_blocked_after_commit,
                self.durable_resume,
                self.dependent_suspension,
                self.coordinated_parallelism,
                self.committed_state_visibility,
            )
        )


@dataclass(frozen=True, slots=True)
class PatchProviderCodecChecklist:
    """Require every optional provider-codec property before projection."""

    advertised: bool = False
    complete_buffering: bool = False
    grammar_and_limits: bool = False
    stable_correlation: bool = False
    replay_fencing: bool = False
    result_injection: bool = False
    approval_suspension: bool = False
    idempotency_and_resume: bool = False
    authority_and_disclosure: bool = False

    def __post_init__(self) -> None:
        """Reject non-boolean provider-codec checklist values."""
        if any(
            type(value) is not bool
            for value in (
                self.advertised,
                self.complete_buffering,
                self.grammar_and_limits,
                self.stable_correlation,
                self.replay_fencing,
                self.result_injection,
                self.approval_suspension,
                self.idempotency_and_resume,
                self.authority_and_disclosure,
            )
        ):
            raise PatchProtocolError("patch provider profile is invalid")

    @property
    def complete(self) -> bool:
        """Return whether every optional provider-codec proof is present."""
        return all(
            (
                self.advertised,
                self.complete_buffering,
                self.grammar_and_limits,
                self.stable_correlation,
                self.replay_fencing,
                self.result_injection,
                self.approval_suspension,
                self.idempotency_and_resume,
                self.authority_and_disclosure,
            )
        )


@dataclass(frozen=True, slots=True)
class PatchProtocolProfile:
    """Bind one exact test-only protocol or orchestration profile."""

    surface: PatchProtocolSurface
    enabled: bool = False
    authenticated: bool = False
    loopback_only: bool = False
    name: str = _TEST_PROFILE_NAME
    protocol: PatchProtocolChecklist = PatchProtocolChecklist()
    orchestration: PatchOrchestrationChecklist = PatchOrchestrationChecklist()
    provider_codec: PatchProviderCodecChecklist = PatchProviderCodecChecklist()

    def __post_init__(self) -> None:
        """Reject malformed profile fields without activating a surface."""
        if (
            type(self.surface) is not PatchProtocolSurface
            or type(self.enabled) is not bool
            or type(self.authenticated) is not bool
            or type(self.loopback_only) is not bool
            or type(self.name) is not str
            or type(self.protocol) is not PatchProtocolChecklist
            or type(self.orchestration) is not PatchOrchestrationChecklist
            or type(self.provider_codec) is not PatchProviderCodecChecklist
        ):
            raise PatchProtocolError("patch protocol profile is invalid")

    @property
    def active(self) -> bool:
        """Return whether this complete exact profile may expose authority."""
        if not (
            self.enabled
            and self.authenticated
            and self.loopback_only
            and self.name == _TEST_PROFILE_NAME
            and self.protocol.complete
        ):
            return False
        if self.surface in {
            PatchProtocolSurface.MCP,
            PatchProtocolSurface.A2A,
        }:
            return True
        if self.surface in {
            PatchProtocolSurface.FLOW,
            PatchProtocolSurface.TASK,
            PatchProtocolSurface.MULTI_AGENT,
        }:
            return self.orchestration.complete
        return self.provider_codec.complete


@dataclass(frozen=True, slots=True)
class PatchProtocolIdentity:
    """Bind one originating execution identity to a durable reservation."""

    tenant: PatchTenantId
    principal: PatchPrincipalId
    execution: PatchExecutionId
    run: PatchRunId
    session: PatchSessionId
    task: PatchTaskId
    agent: PatchAgentId
    route: PolicyRouteId
    context: PatchContextId
    workspace: PatchWorkspaceId

    def __post_init__(self) -> None:
        """Require every authenticated execution coordinate to be typed."""
        if (
            type(self.tenant) is not PatchTenantId
            or type(self.principal) is not PatchPrincipalId
            or type(self.execution) is not PatchExecutionId
            or type(self.run) is not PatchRunId
            or type(self.session) is not PatchSessionId
            or type(self.task) is not PatchTaskId
            or type(self.agent) is not PatchAgentId
            or type(self.route) is not PolicyRouteId
            or type(self.context) is not PatchContextId
            or type(self.workspace) is not PatchWorkspaceId
        ):
            raise PatchProtocolError("patch protocol identity is invalid")

    def durable_identity(
        self, retransmission_key: RetransmissionKey
    ) -> DurableRequestIdentity:
        """Return the durable retry tuple bound before planning begins."""
        if type(retransmission_key) is not RetransmissionKey:
            raise PatchProtocolError("patch protocol identity is invalid")
        return DurableRequestIdentity(
            tenant_id=self.tenant,
            principal_id=self.principal,
            execution_id=self.execution,
            route_id=self.route,
            retransmission_key=retransmission_key,
        )

    def durable_origin(self) -> DurableProtocolOrigin:
        """Return every originating authority fact for a durable plan."""
        return DurableProtocolOrigin(
            self.tenant,
            self.principal,
            self.execution,
            self.run,
            self.session,
            self.task,
            self.agent,
            self.route,
            self.context,
            self.workspace,
        )


@dataclass(frozen=True, slots=True)
class PatchProtocolReservation:
    """Store the sole pre-planning durable protocol reservation."""

    surface: PatchProtocolSurface
    identity: PatchProtocolIdentity
    operation: OperationType
    correlation: PatchObserverCorrelationId
    durable: DurableReservation

    def __post_init__(self) -> None:
        """Require one coherent surface, identity, and reservation tuple."""
        if (
            type(self.surface) is not PatchProtocolSurface
            or type(self.identity) is not PatchProtocolIdentity
            or type(self.operation) is not OperationType
            or type(self.correlation) is not PatchObserverCorrelationId
            or type(self.durable) is not DurableReservation
        ):
            raise PatchProtocolError("patch protocol reservation is invalid")

    @property
    def request_id(self) -> PatchRequestId:
        """Return the server-owned request identity selected by the store."""
        return self.durable.request_id

    @property
    def digest(self) -> AlgorithmDigest:
        """Return the canonical digest attached to the durable reservation."""
        return self.durable.canonical_digest


@dataclass(frozen=True, slots=True)
class PatchProtocolContinuation:
    """Project one authenticated continuation without terminal fabrication."""

    kind: PatchProtocolContinuationKind
    reservation: PatchProtocolReservation
    pending: PatchPending | None = None
    result: PatchResult | None = None

    def __post_init__(self) -> None:
        """Keep approval, pending, and terminal forms mutually exclusive."""
        if (
            type(self.kind) is not PatchProtocolContinuationKind
            or type(self.reservation) is not PatchProtocolReservation
            or (
                self.pending is not None
                and type(self.pending) is not PatchPending
            )
            or (
                self.result is not None
                and type(self.result) is not PatchResult
            )
        ):
            raise PatchProtocolError("patch protocol continuation is invalid")
        if self.kind is PatchProtocolContinuationKind.APPROVAL_REQUIRED:
            valid = self.pending is None and self.result is None
        elif self.kind is PatchProtocolContinuationKind.SETTLEMENT_PENDING:
            valid = self.pending is not None and self.result is None
        else:
            valid = self.pending is None and self.result is not None
        if not valid:
            raise PatchProtocolError("patch protocol continuation is invalid")

    @property
    def completed(self) -> bool:
        """Return whether this continuation has one terminal patch result."""
        return self.kind is PatchProtocolContinuationKind.TERMINAL


class PatchProtocolPlanPort(Protocol):
    """Build one sealed durable plan after reservation and before approval."""

    async def plan(
        self,
        reservation: PatchProtocolReservation,
        operation: OperationType,
        raw_arguments: bytes,
    ) -> DurablePlanReference:
        """Return the complete plan owned by the selected patch runtime."""


class PatchProtocolApprovalPort(Protocol):
    """Consume one authenticated detached approval for a sealed plan."""

    async def approve(
        self,
        reservation: PatchProtocolReservation,
        plan: DurablePlanReference,
    ) -> DurableApproval:
        """Return a broker-attested approval bound to the exact reservation."""


@final
class PatchProtocolApprovalGate:
    """Gate exact test-profile review through the selected approval service."""

    def __init__(self) -> None:
        """Initialize no caller-visible review or decision state."""
        self._reviews: dict[PatchRequestId, Future[PlanReviewRequest]] = {}
        self._decisions: dict[PatchRequestId, Future[BrokerDecision]] = {}

    def review_future(
        self, request_id: PatchRequestId
    ) -> Future[PlanReviewRequest]:
        """Return the service-owned future that records one sealed review."""
        if type(request_id) is not PatchRequestId:
            raise PatchProtocolError("patch protocol approval is unavailable")
        review = self._reviews.get(request_id)
        if review is None:
            review = get_running_loop().create_future()
            self._reviews[request_id] = review
        return review

    async def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Wait for one detached protocol approval of the sealed review."""
        if type(request) is not PlanReviewRequest:
            raise PatchProtocolError("patch protocol approval is unavailable")
        request_id = request.plan.binding.request.request_id
        review = self.review_future(request_id)
        if review.done():
            raise PatchProtocolError("patch protocol approval is unavailable")
        decision = get_running_loop().create_future()
        if request_id in self._decisions:
            raise PatchProtocolError("patch protocol approval is unavailable")
        self._decisions[request_id] = decision
        review.set_result(request)
        try:
            return await shield(decision)
        finally:
            self._decisions.pop(request_id, None)

    async def approve(self, reservation: PatchProtocolReservation) -> None:
        """Approve only the exact service review bound to one reservation."""
        if type(reservation) is not PatchProtocolReservation:
            raise PatchProtocolError("patch protocol approval is unavailable")
        review = self.review_future(reservation.request_id)
        if not review.done() or review.cancelled():
            raise PatchProtocolError("patch protocol approval is unavailable")
        try:
            request = review.result()
        except Exception as error:
            raise PatchProtocolError(
                "patch protocol approval is unavailable"
            ) from error
        binding = request.plan.binding
        if (
            binding.request.request_id != reservation.request_id
            or binding.request.execution_id
            != reservation.durable.identity.execution_id
            or request.subject.tenant != reservation.identity.tenant
            or request.subject.principal != reservation.identity.principal
            or request.subject.run != reservation.identity.run
            or request.subject.session != reservation.identity.session
            or request.subject.task != reservation.identity.task
            or request.subject.agent != reservation.identity.agent
            or binding.target.context_id != reservation.identity.context
            or binding.target.workspace_id != reservation.identity.workspace
            or request.requirements.route != reservation.identity.route
        ):
            raise PatchProtocolError("patch protocol approval is unavailable")
        decision = self._decisions.get(reservation.request_id)
        if decision is None or decision.done():
            raise PatchProtocolError("patch protocol approval is unavailable")
        decision.set_result(
            BrokerDecision(
                request.requirements.broker,
                (
                    ReviewerDecision(
                        PatchPrincipalId("protocol-reviewer"),
                        request.subject.tenant,
                        request.requirements.reviewer_role,
                        ApprovalDecisionState.APPROVED,
                    ),
                ),
            )
        )

    def fail(self, request_id: PatchRequestId, error: BaseException) -> None:
        """Release a waiting planner when its selected runtime has failed."""
        if type(request_id) is not PatchRequestId:
            raise PatchProtocolError("patch protocol approval is unavailable")
        review = self.review_future(request_id)
        if not review.done():
            review.set_exception(
                PatchProtocolError("patch protocol runtime is unavailable")
            )
        decision = self._decisions.get(request_id)
        if decision is not None and not decision.done():
            decision.set_exception(error)


@final
class PatchProtocolSelectedRuntime:
    """Drive protocol stages through one selected sandbox SDK service."""

    def __init__(
        self,
        toolset: PatchToolSet,
        service: SandboxPatchSdkService,
        store: DurablePatchStore,
        approvals: PatchProtocolApprovalGate,
    ) -> None:
        """Bind only one loaded toolset and its selected sandbox service."""
        if (
            type(toolset) is not PatchToolSet
            or type(service) is not SandboxPatchSdkService
            or type(approvals) is not PatchProtocolApprovalGate
            or not callable(getattr(store, "inspect", None))
            or not callable(getattr(store, "reserve", None))
        ):
            raise PatchProtocolError("patch protocol runtime is unavailable")
        host = toolset.sdk_host()
        if type(host) is not PatchSdkHost or host._service is not service:
            raise PatchProtocolError("patch protocol runtime is unavailable")
        self._toolset = toolset
        self._service = service
        self._store = store
        self._approvals = approvals
        self._hosts: dict[PatchRequestId, PatchSdkHost] = {}
        self._tasks: dict[PatchRequestId, Task[PatchInvocationOutcome]] = {}
        self._requests: dict[
            PatchRequestId,
            tuple[
                PatchProtocolReservation,
                OperationType,
                bytes,
            ],
        ] = {}

    async def plan(
        self,
        reservation: PatchProtocolReservation,
        operation: OperationType,
        raw_arguments: bytes,
    ) -> None:
        """Start canonical parse and planning before detached approval."""
        self._validate_request(reservation, operation, raw_arguments)
        self._validate_service_authority(reservation)
        self._requests[reservation.request_id] = (
            reservation,
            operation,
            raw_arguments,
        )
        snapshot = await self._store.inspect(self._access(reservation))
        self._validate_snapshot(reservation, snapshot)
        if snapshot.terminal is not None:
            return
        if snapshot.lifecycle in {
            LifecyclePhase.COMMIT_STARTED,
            LifecyclePhase.SETTLEMENT_PENDING,
        }:
            return
        review = self._approvals.review_future(reservation.request_id)
        if reservation.request_id not in self._tasks:
            host = self._toolset.sdk_host()
            if (
                type(host) is not PatchSdkHost
                or host._service is not self._service
            ):
                raise PatchProtocolError(
                    "patch protocol runtime is unavailable"
                )
            self._hosts[reservation.request_id] = host
            task = create_task(
                host.invoke_remote_raw(
                    operation,
                    raw_arguments,
                    reservation.request_id,
                    reservation.correlation,
                    reservation.durable.identity,
                    reservation.identity.durable_origin(),
                )
            )
            self._tasks[reservation.request_id] = task
            task.add_done_callback(
                lambda value: self._complete_task(
                    reservation.request_id, value
                )
            )
        try:
            await shield(review)
        except Exception as error:
            raise PatchProtocolError(
                "patch protocol runtime is unavailable"
            ) from error
        snapshot = await self._store.inspect(self._access(reservation))
        self._validate_snapshot(reservation, snapshot)
        plan = snapshot.plan
        if (
            type(plan) is not DurablePlanReference
            or plan.canonical_digest != reservation.digest
            or plan.context_id != reservation.identity.context
            or plan.workspace_id != reservation.identity.workspace
        ):
            raise PatchProtocolError("patch protocol runtime is unavailable")

    async def approve(self, reservation: PatchProtocolReservation) -> None:
        """Release selected-service review before durable effect ownership."""
        record = self._requests.get(reservation.request_id)
        if record is None or record[0] != reservation:
            raise PatchProtocolError("patch protocol runtime is unavailable")
        self._validate_service_authority(reservation)
        await self._approvals.approve(reservation)
        await self._service._await_protocol_claim(reservation.request_id)
        task = self._tasks.get(reservation.request_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except Exception as error:
                raise PatchProtocolError(
                    "patch protocol runtime is unavailable"
                ) from error

    async def await_result(
        self, reservation: PatchProtocolReservation
    ) -> None:
        """Attach selected-service recovery without a second target effect."""
        record = self._requests.get(reservation.request_id)
        if record is None or record[0] != reservation:
            raise PatchProtocolError("patch protocol runtime is unavailable")
        self._validate_service_authority(reservation)
        host = self._hosts.get(reservation.request_id)
        if host is None:
            host = self._toolset.sdk_host()
            if (
                type(host) is not PatchSdkHost
                or host._service is not self._service
            ):
                raise PatchProtocolError(
                    "patch protocol runtime is unavailable"
                )
            self._hosts[reservation.request_id] = host
        task = self._tasks.get(reservation.request_id)
        try:
            if task is not None and not task.done():
                await host.inspect()
            else:
                _, operation, raw_arguments = record
                await host.invoke_remote_raw(
                    operation,
                    raw_arguments,
                    reservation.request_id,
                    reservation.correlation,
                    reservation.durable.identity,
                    reservation.identity.durable_origin(),
                )
        except Exception as error:
            raise PatchProtocolError(
                "patch protocol runtime is unavailable"
            ) from error

    async def inspect(
        self, reservation: PatchProtocolReservation
    ) -> DurableRequestSnapshot:
        """Read exact selected-runtime durable truth without dispatching."""
        if type(reservation) is not PatchProtocolReservation:
            raise PatchProtocolError("patch protocol runtime is unavailable")
        self._validate_service_authority(reservation)
        snapshot = await self._store.inspect(self._access(reservation))
        if (
            type(snapshot) is not DurableRequestSnapshot
            or snapshot.reservation.request_id != reservation.request_id
            or snapshot.reservation.identity != reservation.durable.identity
            or snapshot.reservation.canonical_digest != reservation.digest
        ):
            raise PatchProtocolError("patch protocol runtime is unavailable")
        self._validate_snapshot(reservation, snapshot)
        return snapshot

    @staticmethod
    def _access(
        reservation: PatchProtocolReservation,
    ) -> DurableRequestAccess:
        """Return the exact non-bearer durable access for this runtime."""
        return DurableRequestAccess(
            reservation.request_id, reservation.durable.identity
        )

    @staticmethod
    def _validate_request(
        reservation: PatchProtocolReservation,
        operation: OperationType,
        raw_arguments: bytes,
    ) -> None:
        """Reject any replacement input or noncanonical runtime request."""
        if (
            type(reservation) is not PatchProtocolReservation
            or type(operation) is not OperationType
            or type(raw_arguments) is not bytes
            or reservation.operation is not operation
            or reservation.identity.durable_identity(
                reservation.durable.identity.retransmission_key
            )
            != reservation.durable.identity
            or type(reservation.identity.durable_origin())
            is not DurableProtocolOrigin
        ):
            raise PatchProtocolError("patch protocol runtime is unavailable")

    @staticmethod
    def _validate_snapshot(
        reservation: PatchProtocolReservation,
        snapshot: DurableRequestSnapshot,
    ) -> None:
        """Reject a plan not sealed for this full originating authority."""
        if type(snapshot) is not DurableRequestSnapshot:
            raise PatchProtocolError("patch protocol runtime is unavailable")
        plan = snapshot.plan
        if (
            plan is not None
            and plan.origin != reservation.identity.durable_origin()
        ):
            raise PatchProtocolError("patch protocol runtime is unavailable")

    def _validate_service_authority(
        self, reservation: PatchProtocolReservation
    ) -> None:
        """Bind every protocol continuation to the fixed service subject."""
        try:
            self._service._validate_protocol_origin(
                reservation.identity.durable_origin(),
                reservation.durable.identity,
            )
        except Exception as error:
            raise PatchProtocolError(
                "patch protocol runtime is unavailable"
            ) from error

    def _complete_task(
        self, request_id: PatchRequestId, task: Task[PatchInvocationOutcome]
    ) -> None:
        """Release stage waiters only when the selected runtime faults."""
        if task.cancelled():
            self._approvals.fail(
                request_id,
                PatchProtocolError("patch protocol runtime is unavailable"),
            )
            self._service._fail_protocol_claim(request_id)
            return
        error = task.exception()
        if error is not None:
            self._approvals.fail(request_id, error)
            self._service._fail_protocol_claim(request_id)


@dataclass(frozen=True, slots=True)
class PatchProtocolEffectReceipt:
    """Carry only target-owned recovery evidence after one fenced effect."""

    report: WorkerReport
    result: PatchResult
    now: ExpiryTick
    pending: DurablePendingRequest | None = None
    artifacts: tuple[DurableArtifactObservation, ...] = ()

    def __post_init__(self) -> None:
        """Require typed settlement evidence without an effect retry token."""
        if (
            type(self.report) is not WorkerReport
            or type(self.result) is not PatchResult
            or type(self.now) is not ExpiryTick
            or (
                self.pending is not None
                and type(self.pending) is not DurablePendingRequest
            )
            or type(self.artifacts) is not tuple
            or any(
                type(item) is not DurableArtifactObservation
                for item in self.artifacts
            )
        ):
            raise PatchProtocolError("patch protocol effect is invalid")


class PatchProtocolEffectPort(Protocol):
    """Execute and reconcile only the selected target-owned patch effect."""

    async def commit(
        self,
        reservation: PatchProtocolReservation,
        plan: DurablePlanReference,
        lease: DurableCommitLease,
    ) -> PatchProtocolEffectReceipt:
        """Run one effect after the durable owner and fence already exist."""

    async def reconcile(
        self,
        reservation: PatchProtocolReservation,
        plan: DurablePlanReference,
        lease: DurableCommitLease,
    ) -> PatchProtocolEffectReceipt:
        """Read target-owned recovery truth without reissuing an effect."""


class PatchProtocolClock(Protocol):
    """Read trusted durable time for ownership and recovery decisions."""

    async def now(self) -> ExpiryTick:
        """Return the current trusted durable time."""


class PatchProtocolExecutionPort(Protocol):
    """Advance protocol stages through one selected durable runtime."""

    async def plan(
        self,
        reservation: PatchProtocolReservation,
        operation: OperationType,
        raw_arguments: bytes,
    ) -> None:
        """Create a durable reviewable plan after reservation."""

    async def approve(self, reservation: PatchProtocolReservation) -> None:
        """Advance only the exact durable review to settlement."""

    async def await_result(
        self, reservation: PatchProtocolReservation
    ) -> None:
        """Reconcile one durable request without a blind effect retry."""

    async def inspect(
        self, reservation: PatchProtocolReservation
    ) -> DurableRequestSnapshot:
        """Read only the exact durable request bound to this runtime."""


@final
class PatchProtocolDurableCoordinator:
    """Bind protocol calls to the existing durable owner/fence effect path."""

    def __init__(
        self,
        store: DurablePatchStore,
        planner: PatchProtocolPlanPort,
        approvals: PatchProtocolApprovalPort,
        effect: PatchProtocolEffectPort,
        clock: PatchProtocolClock,
        lease_duration: DurationTicks,
    ) -> None:
        """Bind only concrete durable planning, approval, and target ports."""
        required_store_methods = (
            "claim_commit",
            "inspect",
            "persist_plan",
            "replace_expired_owner",
        )
        if (
            type(lease_duration) is not DurationTicks
            or any(
                not callable(getattr(store, name, None))
                for name in required_store_methods
            )
            or not callable(getattr(planner, "plan", None))
            or not callable(getattr(approvals, "approve", None))
            or not callable(getattr(effect, "commit", None))
            or not callable(getattr(effect, "reconcile", None))
            or not callable(getattr(clock, "now", None))
        ):
            raise PatchProtocolError("patch protocol coordinator is invalid")
        self._store = store
        self._planner = planner
        self._approvals = approvals
        self._effect = effect
        self._clock = clock
        self._lease_duration = lease_duration
        self._reconciler = DurablePatchReconciler(store)

    async def plan(
        self,
        reservation: PatchProtocolReservation,
        operation: OperationType,
        raw_arguments: bytes,
    ) -> None:
        """Persist a plan only after canonical reservation is durable."""
        if (
            type(reservation) is not PatchProtocolReservation
            or type(operation) is not OperationType
            or type(raw_arguments) is not bytes
        ):
            raise PatchProtocolError("patch protocol plan is invalid")
        snapshot = await self.inspect(reservation)
        if snapshot.terminal is not None or snapshot.plan is not None:
            return
        plan = await self._planner.plan(reservation, operation, raw_arguments)
        if (
            type(plan) is not DurablePlanReference
            or plan.canonical_digest != reservation.digest
            or plan.context_id != reservation.identity.context
            or plan.workspace_id != reservation.identity.workspace
        ):
            raise PatchProtocolError("patch protocol plan is invalid")
        await self._store.persist_plan(reservation.durable, plan)

    async def approve(self, reservation: PatchProtocolReservation) -> None:
        """Persist owner, fence, and commit start before calling the effect."""
        snapshot = await self.inspect(reservation)
        if snapshot.terminal is not None:
            return
        plan = snapshot.plan
        if type(plan) is not DurablePlanReference:
            raise PatchProtocolError("patch protocol approval is unavailable")
        now = await self._clock.now()
        if type(now) is not ExpiryTick:
            raise PatchProtocolError("patch protocol approval is unavailable")
        approval = await self._approvals.approve(reservation, plan)
        if (
            type(approval) is not DurableApproval
            or approval.identity != reservation.durable.identity
            or approval.canonical_digest != reservation.digest
            or approval.plan_id != plan.plan_id
        ):
            raise PatchProtocolError("patch protocol approval is unavailable")
        claim = await self._store.claim_commit(
            reservation.durable,
            plan,
            approval,
            PatchCommitOwnerId.new(),
            now,
            self._lease_duration,
            (),
        )
        if claim.state is not DurableCommitClaimState.OWNER:
            return
        lease = claim.lease
        if type(lease) is not DurableCommitLease:
            raise PatchProtocolError("patch protocol approval is unavailable")
        await self._reconcile_commit(
            reservation,
            plan,
            lease,
            await self._effect.commit(reservation, plan, lease),
        )

    async def await_result(
        self, reservation: PatchProtocolReservation
    ) -> None:
        """Reconcile durable target truth without dispatching an effect."""
        snapshot = await self.inspect(reservation)
        if snapshot.terminal is not None:
            return
        plan = snapshot.plan
        lease = snapshot.lease
        if (
            type(plan) is not DurablePlanReference
            or type(lease) is not DurableCommitLease
        ):
            raise PatchProtocolError(
                "patch protocol settlement is unavailable"
            )
        now = await self._clock.now()
        if type(now) is not ExpiryTick:
            raise PatchProtocolError(
                "patch protocol settlement is unavailable"
            )
        if now.value >= lease.expires_at.value:
            lease = await self._reconciler.replace_expired_owner(
                self._access(reservation),
                PatchCommitOwnerId.new(),
                now,
                self._lease_duration,
            )
        await self._reconcile_commit(
            reservation,
            plan,
            lease,
            await self._effect.reconcile(reservation, plan, lease),
        )

    async def _reconcile_commit(
        self,
        reservation: PatchProtocolReservation,
        plan: DurablePlanReference,
        lease: DurableCommitLease,
        receipt: PatchProtocolEffectReceipt,
    ) -> None:
        """Commit only journal-derived target receipt truth through storage."""
        if (
            type(receipt) is not PatchProtocolEffectReceipt
            or receipt.result.request_id != reservation.request_id
            or receipt.result.plan_id != plan.plan_id
        ):
            raise PatchProtocolError(
                "patch protocol settlement is unavailable"
            )
        await self._reconciler.reconcile(
            self._access(reservation),
            lease,
            receipt.report,
            receipt.result,
            reservation.correlation,
            receipt.now,
            pending=receipt.pending,
            artifacts=receipt.artifacts,
        )

    async def inspect(
        self, reservation: PatchProtocolReservation
    ) -> DurableRequestSnapshot:
        """Read only the exact durable identity bound to the request."""
        snapshot = await self._store.inspect(self._access(reservation))
        if (
            type(snapshot) is not DurableRequestSnapshot
            or snapshot.reservation.request_id != reservation.request_id
            or snapshot.reservation.identity != reservation.durable.identity
            or snapshot.reservation.canonical_digest != reservation.digest
        ):
            raise PatchProtocolError(
                "patch protocol settlement is unavailable"
            )
        return snapshot

    @staticmethod
    def _access(
        reservation: PatchProtocolReservation,
    ) -> DurableRequestAccess:
        """Return exact non-bearer durable read authority for one request."""
        return DurableRequestAccess(
            reservation.request_id, reservation.durable.identity
        )


@dataclass(frozen=True, slots=True)
class PatchProtocolFlowRequest:
    """Keep one flow node's trusted mutation identity stable across resume."""

    operation: OperationType
    raw_arguments: bytes
    retransmission_key: RetransmissionKey
    correlation: PatchObserverCorrelationId
    mutation_slot: str

    def __post_init__(self) -> None:
        """Require only canonical input and server-owned retry coordinates."""
        if (
            type(self.operation) is not OperationType
            or type(self.raw_arguments) is not bytes
            or type(self.retransmission_key) is not RetransmissionKey
            or type(self.correlation) is not PatchObserverCorrelationId
            or type(self.mutation_slot) is not str
            or not self.mutation_slot
            or len(self.mutation_slot) > 64
            or any(
                not character.isascii()
                or not (character.isalnum() or character == "_")
                for character in self.mutation_slot
            )
        ):
            raise PatchProtocolError("patch protocol flow request is invalid")


@dataclass(frozen=True, slots=True)
class PatchProtocolFlowSuspension(FlowSuspension):
    """Suspend a patch node and every dependent until durable settlement."""

    continuation: PatchProtocolContinuation

    def __post_init__(self) -> None:
        """Require a nonterminal continuation before halting a flow."""
        if (
            type(self.continuation) is not PatchProtocolContinuation
            or self.continuation.completed
        ):
            raise PatchProtocolError(
                "patch protocol flow suspension is invalid"
            )


@final
class PatchProtocolOrchestrationAdapter:
    """Route flow and queued work through one durable patch coordinator."""

    def __init__(
        self,
        profile: PatchProtocolProfile,
        identity: PatchProtocolIdentity,
        store: DurablePatchStore,
        parser: PatchRequestParser,
        coordinator: PatchProtocolExecutionPort,
    ) -> None:
        """Bind one active flow or task profile to exact durable authority."""
        if (
            type(profile) is not PatchProtocolProfile
            or profile.surface
            not in {PatchProtocolSurface.FLOW, PatchProtocolSurface.TASK}
            or type(identity) is not PatchProtocolIdentity
            or type(parser) is not PatchRequestParser
            or not callable(getattr(coordinator, "plan", None))
            or not callable(getattr(coordinator, "approve", None))
            or not callable(getattr(coordinator, "await_result", None))
            or not callable(getattr(store, "inspect", None))
            or not callable(getattr(store, "reserve", None))
        ):
            raise PatchProtocolError("patch protocol adapter is invalid")
        self._protocol = PatchProtocols(profile, identity)
        self._identity = identity
        self._store = store
        self._parser = parser
        self._coordinator = coordinator

    @property
    def surface(self) -> PatchProtocolSurface:
        """Return the exact orchestration surface fixed at construction."""
        return self._protocol.surface

    async def advance(
        self,
        request: PatchProtocolFlowRequest,
        *,
        approve: bool = False,
    ) -> PatchProtocolContinuation:
        """Begin or resume one request without manufacturing a retry key."""
        if type(approve) is not bool:
            raise PatchProtocolError(
                "patch protocol adapter request is invalid"
            )
        reservation = await self._reserve(request)
        continuation = await self._protocol.inspect(self._store, reservation)
        if approve and (
            continuation.kind
            is PatchProtocolContinuationKind.APPROVAL_REQUIRED
        ):
            await self._coordinator.approve(reservation)
            continuation = await self._protocol.inspect(
                self._store, reservation
            )
        return continuation

    async def _reserve(
        self, request: PatchProtocolFlowRequest
    ) -> PatchProtocolReservation:
        """Reserve and replan only the original trusted flow/task identity."""
        if type(request) is not PatchProtocolFlowRequest:
            raise PatchProtocolError(
                "patch protocol adapter request is invalid"
            )
        reservation, _ = await self._protocol.reserve_before_planning(
            self._store,
            request.operation,
            request.raw_arguments,
            request.retransmission_key,
            request.correlation,
            self._parser,
            lambda value: self._coordinator.plan(
                value,
                request.operation,
                request.raw_arguments,
            ),
            request_id=self._request_id(request),
        )
        await self._coordinator.plan(
            reservation, request.operation, request.raw_arguments
        )
        return reservation

    def _request_id(self, request: PatchProtocolFlowRequest) -> PatchRequestId:
        """Derive one stable server-owned identity for a flow/task node."""
        parts = (
            self.surface.value,
            self._identity.tenant.value,
            self._identity.principal.value,
            self._identity.execution.value,
            self._identity.run.value,
            self._identity.session.value,
            self._identity.task.value,
            self._identity.agent.value,
            self._identity.route.value,
            self._identity.context.value,
            self._identity.workspace.value,
            request.mutation_slot,
        )
        value = sha256("\x1f".join(parts).encode()).hexdigest()[:32]
        return PatchRequestId("request_" + value)

    async def reconcile(
        self, request: PatchProtocolFlowRequest
    ) -> PatchProtocolContinuation:
        """Reconcile an existing pending request without a target retry."""
        reservation = await self._reserve(request)
        snapshot = await self._coordinator.inspect(reservation)
        if snapshot.lifecycle in {
            LifecyclePhase.COMMIT_STARTED,
            LifecyclePhase.SETTLEMENT_PENDING,
        }:
            await self._coordinator.await_result(reservation)
        return await self._protocol.inspect(self._store, reservation)


@final
class PatchProtocolFlowAdapter:
    """Suspend real flow routing until a patch terminal result is durable."""

    def __init__(self, adapter: PatchProtocolOrchestrationAdapter) -> None:
        """Bind only a flow-surface orchestration adapter."""
        if (
            type(adapter) is not PatchProtocolOrchestrationAdapter
            or adapter.surface is not PatchProtocolSurface.FLOW
        ):
            raise PatchProtocolError("patch protocol flow adapter is invalid")
        self._adapter = adapter

    async def execute(
        self,
        request: PatchProtocolFlowRequest,
        *,
        approve: bool = False,
    ) -> PatchProtocolFlowSuspension | PatchResult:
        """Return terminal truth or halt the flow before dependent routing."""
        continuation = await self._adapter.advance(request, approve=approve)
        if continuation.result is not None:
            return continuation.result
        return PatchProtocolFlowSuspension(continuation)

    async def resume(
        self, request: PatchProtocolFlowRequest
    ) -> PatchProtocolFlowSuspension | PatchResult:
        """Reconcile the original request and route one terminal result."""
        continuation = await self._adapter.reconcile(request)
        if continuation.result is not None:
            return continuation.result
        return PatchProtocolFlowSuspension(continuation)


@final
class PatchProtocolQueuedTaskAdapter:
    """Use one durable request identity for each queued worker attempt."""

    def __init__(self, adapter: PatchProtocolOrchestrationAdapter) -> None:
        """Bind only a task-surface orchestration adapter."""
        if (
            type(adapter) is not PatchProtocolOrchestrationAdapter
            or adapter.surface is not PatchProtocolSurface.TASK
        ):
            raise PatchProtocolError("patch protocol task adapter is invalid")
        self._adapter = adapter

    async def execute(
        self,
        request: PatchProtocolFlowRequest,
        *,
        approve: bool = False,
    ) -> PatchProtocolContinuation:
        """Run one claimed task without converting a retry into mutation."""
        return await self._adapter.advance(request, approve=approve)

    async def recover(
        self, request: PatchProtocolFlowRequest
    ) -> PatchProtocolContinuation:
        """Reconcile a lost worker only from durable target-owned evidence."""
        return await self._adapter.reconcile(request)


@final
class PatchProtocols:
    """Gate protocol reservations and authenticated durable continuations."""

    def __init__(
        self,
        profile: PatchProtocolProfile,
        identity: PatchProtocolIdentity,
    ) -> None:
        """Bind one profile and its server-derived execution identity."""
        if (
            type(profile) is not PatchProtocolProfile
            or type(identity) is not PatchProtocolIdentity
        ):
            raise PatchProtocolError("patch protocol binding is invalid")
        self._profile = profile
        self._identity = identity

    @property
    def active(self) -> bool:
        """Return whether this exact protocol instance can mutate."""
        return self._profile.active

    @property
    def surface(self) -> PatchProtocolSurface:
        """Return the exact profile surface fixed at construction."""
        return self._profile.surface

    def advertised_tools(self) -> tuple[str, ...]:
        """Return only tools allowed by the fully bound exact profile."""
        if not self.active:
            return ()
        if self._profile.surface in {
            PatchProtocolSurface.MCP,
            PatchProtocolSurface.A2A,
        }:
            return ("patch.edit", "patch.apply")
        if self._profile.surface in {
            PatchProtocolSurface.PROVIDER_FREEFORM,
            PatchProtocolSurface.PROVIDER_NATIVE,
        }:
            return ("patch.apply",)
        return ()

    async def reserve_before_planning(
        self,
        store: DurablePatchStore,
        operation: OperationType,
        raw_arguments: bytes,
        retransmission_key: RetransmissionKey,
        correlation: PatchObserverCorrelationId,
        parser: PatchRequestParser,
        planner: Callable[
            [PatchProtocolReservation], Awaitable[_ProtocolPlanValue]
        ],
        *,
        request_id: PatchRequestId | None = None,
    ) -> tuple[PatchProtocolReservation, _ProtocolPlanValue | None]:
        """Reserve a canonical retry identity before invoking a new planner."""
        reservation = await self.reserve(
            store,
            operation,
            raw_arguments,
            retransmission_key,
            correlation,
            parser,
            request_id=request_id,
        )
        if reservation.durable.replayed:
            return reservation, None
        return reservation, await planner(reservation)

    async def reserve(
        self,
        store: DurablePatchStore,
        operation: OperationType,
        raw_arguments: bytes,
        retransmission_key: RetransmissionKey,
        correlation: PatchObserverCorrelationId,
        parser: PatchRequestParser,
        *,
        request_id: PatchRequestId | None = None,
    ) -> PatchProtocolReservation:
        """Reserve an exact canonical request before any planning callback."""
        self._require_active()
        if (
            type(operation) is not OperationType
            or type(raw_arguments) is not bytes
            or type(retransmission_key) is not RetransmissionKey
            or type(correlation) is not PatchObserverCorrelationId
            or type(parser) is not PatchRequestParser
            or (
                request_id is not None
                and type(request_id) is not PatchRequestId
            )
            or not callable(getattr(store, "reserve", None))
        ):
            raise PatchProtocolError("patch protocol request is invalid")
        kind = (
            RawPatchInputKind.EDIT_JSON
            if operation is OperationType.EDIT
            else RawPatchInputKind.APPLY_JSON
        )
        try:
            request = parser.parse(
                RawPatchIngress(
                    RawProviderProfile(_TEST_PROFILE_NAME),
                    RawToolCallId(correlation.value),
                    kind,
                    RawPatchInputState.COMPLETE,
                    raw_arguments,
                )
            )
        except PatchInputError as error:
            raise PatchProtocolError(
                "patch protocol request is invalid"
            ) from error
        durable = await store.reserve(
            self._identity.durable_identity(retransmission_key),
            request.digest,
            request_id,
        )
        return PatchProtocolReservation(
            self._profile.surface,
            self._identity,
            operation,
            correlation,
            durable,
        )

    async def inspect(
        self,
        store: DurablePatchStore,
        reservation: PatchProtocolReservation,
    ) -> PatchProtocolContinuation:
        """Read one authenticated continuation without bearer-handle access."""
        self._require_reservation(reservation)
        if not callable(getattr(store, "inspect", None)):
            raise PatchProtocolError(
                "patch protocol continuation is unavailable"
            )
        snapshot = await store.inspect(
            DurableRequestAccess(
                reservation.request_id,
                reservation.durable.identity,
            )
        )
        return self._continuation(reservation, snapshot)

    def _continuation(
        self,
        reservation: PatchProtocolReservation,
        snapshot: DurableRequestSnapshot,
    ) -> PatchProtocolContinuation:
        """Map durable truth to one non-interchangeable continuation type."""
        if (
            type(snapshot) is not DurableRequestSnapshot
            or snapshot.reservation.request_id != reservation.request_id
            or snapshot.reservation.identity != reservation.durable.identity
            or snapshot.reservation.canonical_digest != reservation.digest
        ):
            raise PatchProtocolError(
                "patch protocol continuation is unavailable"
            )
        if snapshot.terminal is not None:
            return PatchProtocolContinuation(
                PatchProtocolContinuationKind.TERMINAL,
                reservation,
                result=snapshot.terminal.result,
            )
        if (
            snapshot.lifecycle is LifecyclePhase.SETTLEMENT_PENDING
            and snapshot.pending is not None
        ):
            return PatchProtocolContinuation(
                PatchProtocolContinuationKind.SETTLEMENT_PENDING,
                reservation,
                pending=PatchPending(
                    schema_version=1,
                    pending_operation_id=snapshot.pending.pending_operation_id,
                    request_id=reservation.request_id,
                    correlation_id=snapshot.pending.correlation_id,
                    lifecycle=LifecyclePhase.SETTLEMENT_PENDING,
                ),
            )
        if snapshot.lifecycle is LifecyclePhase.COMMIT_STARTED:
            return PatchProtocolContinuation(
                PatchProtocolContinuationKind.SETTLEMENT_PENDING,
                reservation,
                pending=PatchPending(
                    schema_version=1,
                    pending_operation_id=PatchPendingOperationId(
                        "pending_"
                        + sha256(
                            b"patch-protocol-attached\x00"
                            + reservation.request_id.value.encode()
                        ).hexdigest()[:32]
                    ),
                    request_id=reservation.request_id,
                    correlation_id=reservation.correlation,
                    lifecycle=LifecyclePhase.SETTLEMENT_PENDING,
                ),
            )
        if snapshot.lifecycle in {
            LifecyclePhase.RECEIVED,
            LifecyclePhase.PLANNED,
        }:
            return PatchProtocolContinuation(
                PatchProtocolContinuationKind.APPROVAL_REQUIRED,
                reservation,
            )
        raise PatchProtocolError("patch protocol continuation is unavailable")

    def _require_active(self) -> None:
        """Reject unadvertised or incomplete surfaces before parsing or I/O."""
        if not self.active:
            raise PatchProtocolError("patch protocol surface is unavailable")

    def _require_reservation(
        self, reservation: PatchProtocolReservation
    ) -> None:
        """Reject cross-principal, cross-agent, or cross-route calls."""
        self._require_active()
        if (
            type(reservation) is not PatchProtocolReservation
            or reservation.surface is not self._profile.surface
            or reservation.identity != self._identity
        ):
            raise PatchProtocolError(
                "patch protocol continuation is unavailable"
            )


@final
class PatchProtocolCoordinationDomain:
    """Bind every workspace mutation to one durable serial domain owner."""

    def __init__(
        self,
        workspace: PatchWorkspaceId,
        owner: PatchDomainId,
        policy: PatchProtocolCoordinationPolicy = (
            PatchProtocolCoordinationPolicy.SERIAL
        ),
    ) -> None:
        """Bind a server-selected domain owner to one exact workspace."""
        if (
            type(workspace) is not PatchWorkspaceId
            or type(owner) is not PatchDomainId
            or type(policy) is not PatchProtocolCoordinationPolicy
        ):
            raise PatchProtocolError("patch coordination domain is invalid")
        self._workspace = workspace
        self._owner = owner
        self._policy = policy

    @classmethod
    def for_workspace(
        cls, workspace: PatchWorkspaceId
    ) -> "PatchProtocolCoordinationDomain":
        """Return the deterministic sole owner for one backing workspace."""
        if type(workspace) is not PatchWorkspaceId:
            raise PatchProtocolError("patch coordination domain is invalid")
        owner = PatchDomainId(
            "domain_" + sha256(workspace.value.encode()).hexdigest()[:32]
        )
        return cls(workspace, owner)

    @property
    def owner(self) -> PatchDomainId:
        """Return the one domain owner shared by every bound agent."""
        return self._owner

    @property
    def workspace(self) -> PatchWorkspaceId:
        """Return the workspace selected for this coordination domain."""
        return self._workspace

    @property
    def policy(self) -> PatchProtocolCoordinationPolicy:
        """Return the declared scheduling policy without caller override."""
        return self._policy

    async def admit(
        self,
        store: DurablePatchStore,
        identity: PatchProtocolIdentity,
        reservation: PatchProtocolReservation,
        paths: frozenset[LogicalPath],
    ) -> None:
        """Durably admit only one nonterminal workspace mutation at a time."""
        admission = self._admission(identity, reservation, paths)
        if not callable(getattr(store, "admit_coordination", None)):
            raise PatchProtocolError("patch coordination admission is denied")
        try:
            await store.admit_coordination(admission)
        except DurableStoreError as error:
            raise PatchProtocolError(
                "patch coordination admission is denied"
            ) from error

    async def release(
        self,
        store: DurablePatchStore,
        identity: PatchProtocolIdentity,
        reservation: PatchProtocolReservation,
    ) -> None:
        """Release only an exact terminal workspace mutation admission."""
        access = self._access(identity, reservation)
        if not callable(getattr(store, "release_coordination", None)):
            raise PatchProtocolError("patch coordination admission is denied")
        try:
            await store.release_coordination(access)
        except DurableStoreError as error:
            raise PatchProtocolError(
                "patch coordination admission is denied"
            ) from error

    async def is_admitted(
        self,
        store: DurablePatchStore,
        identity: PatchProtocolIdentity,
        reservation: PatchProtocolReservation,
    ) -> bool:
        """Return whether a request remains the durable workspace owner."""
        access = self._access(identity, reservation)
        if not callable(getattr(store, "is_coordination_admitted", None)):
            raise PatchProtocolError("patch coordination admission is denied")
        try:
            return await store.is_coordination_admitted(access)
        except DurableStoreError as error:
            raise PatchProtocolError(
                "patch coordination admission is denied"
            ) from error

    def _admission(
        self,
        identity: PatchProtocolIdentity,
        reservation: PatchProtocolReservation,
        paths: frozenset[LogicalPath],
    ) -> DurableCoordinationAdmission:
        """Bind protocol authority to one durable workspace admission row."""
        return DurableCoordinationAdmission(
            self._access(identity, reservation), paths
        )

    def _access(
        self,
        identity: PatchProtocolIdentity,
        reservation: PatchProtocolReservation,
    ) -> DurableCoordinationAccess:
        """Bind protocol authority to one durable coordination row."""
        if (
            type(identity) is not PatchProtocolIdentity
            or type(reservation) is not PatchProtocolReservation
            or reservation.identity != identity
            or identity.workspace != self._workspace
        ):
            raise PatchProtocolError("patch coordination admission is denied")
        try:
            return DurableCoordinationAccess(
                DurableReservation(
                    reservation.request_id,
                    reservation.durable.identity,
                    reservation.durable.canonical_digest,
                    False,
                ),
                identity.run,
                identity.session,
                identity.task,
                identity.agent,
                identity.context,
                identity.workspace,
                self._owner,
            )
        except DurableStoreError as error:
            raise PatchProtocolError(
                "patch coordination admission is denied"
            ) from error


@final
class PatchProtocolMultiAgentAdapter:
    """Coordinate exact multi-agent requests through one durable owner."""

    def __init__(
        self,
        profile: PatchProtocolProfile,
        identity: PatchProtocolIdentity,
        domain: PatchProtocolCoordinationDomain,
        store: DurablePatchStore,
        parser: PatchRequestParser,
        coordinator: PatchProtocolExecutionPort,
    ) -> None:
        """Bind one agent to the workspace-wide domain and durable runtime."""
        if (
            type(profile) is not PatchProtocolProfile
            or profile.surface is not PatchProtocolSurface.MULTI_AGENT
            or type(identity) is not PatchProtocolIdentity
            or type(domain) is not PatchProtocolCoordinationDomain
            or identity.workspace != domain.workspace
            or type(parser) is not PatchRequestParser
            or not callable(getattr(store, "inspect", None))
            or not callable(getattr(store, "reserve", None))
            or not callable(getattr(store, "admit_coordination", None))
            or not callable(getattr(store, "release_coordination", None))
            or not callable(getattr(store, "is_coordination_admitted", None))
            or not callable(getattr(coordinator, "plan", None))
            or not callable(getattr(coordinator, "approve", None))
            or not callable(getattr(coordinator, "await_result", None))
            or not callable(getattr(coordinator, "inspect", None))
        ):
            raise PatchProtocolError("patch multi-agent adapter is invalid")
        self._protocol = PatchProtocols(profile, identity)
        self._identity = identity
        self._domain = domain
        self._store = store
        self._parser = parser
        self._coordinator = coordinator

    @property
    def domain_owner(self) -> PatchDomainId:
        """Return the owner every agent targeting this workspace must share."""
        return self._domain.owner

    async def execute(
        self,
        request: PatchProtocolFlowRequest,
        *,
        approve: bool = False,
    ) -> PatchProtocolContinuation:
        """Plan and optionally approve one serially coordinated mutation."""
        if type(approve) is not bool:
            raise PatchProtocolError("patch multi-agent request is invalid")
        reservation = await self._reserve(request)
        continuation = await self._protocol.inspect(self._store, reservation)
        if (
            approve
            and continuation.kind
            is PatchProtocolContinuationKind.APPROVAL_REQUIRED
        ):
            await self._coordinator.approve(reservation)
            continuation = await self._protocol.inspect(
                self._store, reservation
            )
        return await self._release_terminal(continuation)

    async def resume(
        self, request: PatchProtocolFlowRequest
    ) -> PatchProtocolContinuation:
        """Reconcile one owned request without a provider or effect retry."""
        reservation = await self._reserve(request)
        snapshot = await self._coordinator.inspect(reservation)
        if snapshot.lifecycle in {
            LifecyclePhase.COMMIT_STARTED,
            LifecyclePhase.SETTLEMENT_PENDING,
        }:
            await self._coordinator.await_result(reservation)
        return await self._release_terminal(
            await self._protocol.inspect(self._store, reservation)
        )

    async def inspect(
        self, reservation: PatchProtocolReservation
    ) -> PatchProtocolContinuation:
        """Read only the originating agent's exact durable continuation."""
        return await self._release_terminal(
            await self._protocol.inspect(self._store, reservation)
        )

    async def _reserve(
        self, request: PatchProtocolFlowRequest
    ) -> PatchProtocolReservation:
        """Reserve, coordinate, and plan one stable agent-owned request."""
        if type(request) is not PatchProtocolFlowRequest:
            raise PatchProtocolError("patch multi-agent request is invalid")
        canonical = _canonical_json_request(
            self._parser,
            request.operation,
            request.raw_arguments,
            request.correlation,
        )
        reservation = await self._protocol.reserve(
            self._store,
            request.operation,
            request.raw_arguments,
            request.retransmission_key,
            request.correlation,
            self._parser,
            request_id=self._request_id(request),
        )
        snapshot = await self._coordinator.inspect(reservation)
        if snapshot.terminal is not None:
            return reservation
        paths = _canonical_paths(canonical)
        await self._domain.admit(
            self._store, self._identity, reservation, paths
        )
        try:
            snapshot = await self._coordinator.inspect(reservation)
            if snapshot.terminal is not None:
                return reservation
            await self._coordinator.plan(
                reservation, request.operation, request.raw_arguments
            )
            snapshot = await self._coordinator.inspect(reservation)
            plan = snapshot.plan
            if (
                type(plan) is not DurablePlanReference
                or plan.domain_id != self._domain.owner
                or plan.workspace_id != self._identity.workspace
                or plan.context_id != self._identity.context
            ):
                raise PatchProtocolError("patch multi-agent plan is invalid")
            return reservation
        except BaseException:
            await self._release_unplanned(reservation)
            raise

    async def _release_unplanned(
        self, reservation: PatchProtocolReservation
    ) -> None:
        """Release only an admission whose request never received a plan."""
        snapshot = await self._store.inspect(
            DurableRequestAccess(
                reservation.request_id, reservation.durable.identity
            )
        )
        if type(snapshot) is not DurableRequestSnapshot:
            raise PatchProtocolError(
                "patch multi-agent release is unavailable"
            )
        if snapshot.terminal is not None or snapshot.plan is None:
            await self._domain.release(
                self._store, self._identity, reservation
            )

    def _request_id(self, request: PatchProtocolFlowRequest) -> PatchRequestId:
        """Derive an agent-specific durable identity without caller UUIDs."""
        parts = (
            self._identity.tenant.value,
            self._identity.principal.value,
            self._identity.execution.value,
            self._identity.run.value,
            self._identity.session.value,
            self._identity.task.value,
            self._identity.agent.value,
            self._identity.route.value,
            self._identity.context.value,
            self._identity.workspace.value,
            self._domain.owner.value,
            request.mutation_slot,
        )
        return PatchRequestId(
            "request_" + sha256("\x1f".join(parts).encode()).hexdigest()[:32]
        )

    async def _release_terminal(
        self, continuation: PatchProtocolContinuation
    ) -> PatchProtocolContinuation:
        """Release the shared domain after durable terminal truth exists."""
        if type(continuation) is not PatchProtocolContinuation:
            raise PatchProtocolError(
                "patch multi-agent continuation is invalid"
            )
        if continuation.completed:
            await self._domain.release(
                self._store,
                self._identity,
                continuation.reservation,
            )
        return continuation


@dataclass(frozen=True, slots=True)
class PatchProtocolProviderCall:
    """Bind complete selected-provider chunks to exact patch retry identity."""

    provider_profile: RawProviderProfile
    tool_call_id: RawToolCallId
    correlation: PatchObserverCorrelationId
    retransmission_key: RetransmissionKey
    grammar_version: str
    chunks: tuple[bytes, ...]
    complete: bool

    def __post_init__(self) -> None:
        """Reject partial, unbounded, or ambiguous provider-call metadata."""
        if (
            type(self.provider_profile) is not RawProviderProfile
            or type(self.tool_call_id) is not RawToolCallId
            or type(self.correlation) is not PatchObserverCorrelationId
            or type(self.retransmission_key) is not RetransmissionKey
            or type(self.grammar_version) is not str
            or self.grammar_version != "grammar-v1"
            or type(self.chunks) is not tuple
            or not self.chunks
            or any(type(chunk) is not bytes for chunk in self.chunks)
            or type(self.complete) is not bool
            or not self.complete
        ):
            raise PatchProtocolError("patch provider call is invalid")


@dataclass(frozen=True, slots=True)
class PatchProtocolResultInjection:
    """Carry one content-free terminal result back to the selected provider."""

    request_id: PatchRequestId
    correlation: PatchObserverCorrelationId
    lifecycle: LifecyclePhase
    status: str

    def __post_init__(self) -> None:
        """Require exact terminal result metadata with no patch payload."""
        if (
            type(self.request_id) is not PatchRequestId
            or type(self.correlation) is not PatchObserverCorrelationId
            or self.lifecycle is not LifecyclePhase.REQUEST_COMPLETED
            or type(self.status) is not str
            or not self.status
        ):
            raise PatchProtocolError("patch provider result is invalid")


@final
class PatchProtocolProviderAdapter:
    """Project selected freeform or native calls onto canonical JSON only."""

    def __init__(
        self,
        profile: PatchProtocolProfile,
        identity: PatchProtocolIdentity,
        provider_profile: RawProviderProfile,
        store: DurablePatchStore,
        parser: PatchRequestParser,
        runtime: PatchProtocolExecutionPort,
    ) -> None:
        """Bind one complete provider profile to the selected patch runtime."""
        if (
            type(profile) is not PatchProtocolProfile
            or profile.surface
            not in {
                PatchProtocolSurface.PROVIDER_FREEFORM,
                PatchProtocolSurface.PROVIDER_NATIVE,
            }
            or type(identity) is not PatchProtocolIdentity
            or type(provider_profile) is not RawProviderProfile
            or type(parser) is not PatchRequestParser
            or not callable(getattr(store, "inspect", None))
            or not callable(getattr(store, "reserve", None))
            or not callable(getattr(runtime, "plan", None))
            or not callable(getattr(runtime, "approve", None))
            or not callable(getattr(runtime, "await_result", None))
            or not callable(getattr(runtime, "inspect", None))
        ):
            raise PatchProtocolError("patch provider adapter is invalid")
        self._protocol = PatchProtocols(profile, identity)
        self._identity = identity
        self._provider_profile = provider_profile
        self._store = store
        self._parser = parser
        self._runtime = runtime
        self._calls: dict[
            RawToolCallId,
            tuple[
                AlgorithmDigest,
                RetransmissionKey,
                PatchObserverCorrelationId,
            ],
        ] = {}
        self._requests: dict[PatchRequestId, PatchProtocolReservation] = {}

    @property
    def advertised_tools(self) -> tuple[str, ...]:
        """Return only selected complete optional provider projections."""
        return self._protocol.advertised_tools()

    def correlation_for(
        self, tool_call_id: RawToolCallId, key: RetransmissionKey
    ) -> PatchObserverCorrelationId:
        """Derive correlation from server identity, call ID, and retry key."""
        if (
            type(tool_call_id) is not RawToolCallId
            or type(key) is not RetransmissionKey
        ):
            raise PatchProtocolError("patch provider call is invalid")
        material = "\x00".join(
            (
                self._identity.tenant.value,
                self._identity.principal.value,
                self._identity.execution.value,
                self._identity.route.value,
                self._provider_profile.value,
                tool_call_id.value,
                key.value,
            )
        ).encode()
        return PatchObserverCorrelationId(
            "correlation_" + sha256(material).hexdigest()[:32]
        )

    async def apply_json(
        self,
        raw_arguments: bytes,
        retransmission_key: RetransmissionKey,
        correlation: PatchObserverCorrelationId,
        *,
        origin: PatchProtocolProviderItemOrigin = (
            PatchProtocolProviderItemOrigin.CURRENT
        ),
    ) -> PatchProtocolContinuation:
        """Dispatch portable JSON only from a selected current item."""
        canonical = self._current_json(
            raw_arguments, retransmission_key, correlation, origin
        )
        return await self._dispatch(
            canonical, raw_arguments, retransmission_key, correlation
        )

    async def apply_freeform(
        self,
        call: PatchProtocolProviderCall,
        *,
        origin: PatchProtocolProviderItemOrigin = (
            PatchProtocolProviderItemOrigin.CURRENT
        ),
    ) -> PatchProtocolContinuation:
        """Buffer a complete selected freeform item before JSON projection."""
        if (
            type(call) is not PatchProtocolProviderCall
            or type(origin) is not PatchProtocolProviderItemOrigin
            or origin is not PatchProtocolProviderItemOrigin.CURRENT
            or call.provider_profile != self._provider_profile
            or call.correlation
            != self.correlation_for(call.tool_call_id, call.retransmission_key)
        ):
            raise PatchProtocolError("patch provider dispatch is unavailable")
        accumulator = PatchInputAccumulator(self._limits())
        try:
            for chunk in call.chunks:
                accumulator.append(chunk)
            canonical = self._parser.parse(
                accumulator.finish(
                    call.provider_profile,
                    call.tool_call_id,
                    RawPatchInputKind.VERIFIED_FREEFORM,
                )
            )
        except PatchInputError as error:
            raise PatchProtocolError(
                "patch provider dispatch is unavailable"
            ) from error
        if canonical.operation is not OperationType.APPLY:
            raise PatchProtocolError("patch provider dispatch is unavailable")
        existing = self._calls.get(call.tool_call_id)
        witness = (
            canonical.digest,
            call.retransmission_key,
            call.correlation,
        )
        if existing is not None and existing != witness:
            raise PatchProtocolError("patch provider dispatch is unavailable")
        self._calls[call.tool_call_id] = witness
        raw_arguments = _apply_json_arguments(canonical)
        json_canonical = self._current_json(
            raw_arguments,
            call.retransmission_key,
            call.correlation,
            origin,
        )
        if (
            json_canonical.canonical_bytes != canonical.canonical_bytes
            or json_canonical.digest != canonical.digest
        ):
            raise PatchProtocolError("patch provider dispatch is unavailable")
        return await self._dispatch(
            canonical,
            raw_arguments,
            call.retransmission_key,
            call.correlation,
        )

    async def approve(
        self, reservation: PatchProtocolReservation
    ) -> PatchProtocolContinuation:
        """Approve only one selected provider reservation before pending."""
        self._require_owned(reservation)
        continuation = await self._protocol.inspect(self._store, reservation)
        if (
            continuation.kind
            is PatchProtocolContinuationKind.APPROVAL_REQUIRED
        ):
            await self._runtime.approve(reservation)
        return await self._protocol.inspect(self._store, reservation)

    async def resume(
        self, reservation: PatchProtocolReservation
    ) -> PatchProtocolContinuation:
        """Resume selected settlement without issuing another effect."""
        self._require_owned(reservation)
        snapshot = await self._runtime.inspect(reservation)
        if snapshot.lifecycle in {
            LifecyclePhase.COMMIT_STARTED,
            LifecyclePhase.SETTLEMENT_PENDING,
        }:
            await self._runtime.await_result(reservation)
        return await self._protocol.inspect(self._store, reservation)

    def reinject(
        self, continuation: PatchProtocolContinuation
    ) -> PatchProtocolResultInjection:
        """Return only exact terminal truth for the original provider call."""
        if type(continuation) is not PatchProtocolContinuation:
            raise PatchProtocolError("patch provider result is unavailable")
        self._require_owned(continuation.reservation)
        result = continuation.result
        if not continuation.completed or type(result) is not PatchResult:
            raise PatchProtocolError("patch provider result is unavailable")
        return PatchProtocolResultInjection(
            result.request_id,
            continuation.reservation.correlation,
            result.lifecycle,
            result.status.value,
        )

    async def _dispatch(
        self,
        canonical: CanonicalPatchRequest,
        raw_arguments: bytes,
        retransmission_key: RetransmissionKey,
        correlation: PatchObserverCorrelationId,
    ) -> PatchProtocolContinuation:
        """Reserve and plan JSON projection through selected runtime."""
        try:
            reservation = await self._protocol.reserve(
                self._store,
                OperationType.APPLY,
                raw_arguments,
                retransmission_key,
                correlation,
                self._parser,
            )
        except DurableStoreError as error:
            raise PatchProtocolError(
                "patch provider dispatch is unavailable"
            ) from error
        if reservation.digest != canonical.digest:
            raise PatchProtocolError("patch provider dispatch is unavailable")
        self._requests[reservation.request_id] = reservation
        if not reservation.durable.replayed:
            await self._runtime.plan(
                reservation, OperationType.APPLY, raw_arguments
            )
        return await self._protocol.inspect(self._store, reservation)

    def _current_json(
        self,
        raw_arguments: bytes,
        retransmission_key: RetransmissionKey,
        correlation: PatchObserverCorrelationId,
        origin: PatchProtocolProviderItemOrigin,
    ) -> CanonicalPatchRequest:
        """Parse only selected current JSON apply input before reservation."""
        if (
            type(raw_arguments) is not bytes
            or type(retransmission_key) is not RetransmissionKey
            or type(correlation) is not PatchObserverCorrelationId
            or type(origin) is not PatchProtocolProviderItemOrigin
            or origin is not PatchProtocolProviderItemOrigin.CURRENT
        ):
            raise PatchProtocolError("patch provider dispatch is unavailable")
        return _canonical_json_request(
            self._parser,
            OperationType.APPLY,
            raw_arguments,
            correlation,
        )

    def _limits(self) -> PatchInputLimits:
        """Return the parser-owned finite input limits for chunk buffering."""
        return self._parser.limits

    def _require_owned(self, reservation: PatchProtocolReservation) -> None:
        """Reject opaque foreign reservations before status or approval I/O."""
        if type(
            reservation
        ) is not PatchProtocolReservation or not _same_reservation(
            self._requests.get(reservation.request_id), reservation
        ):
            raise PatchProtocolError("patch provider result is unavailable")
        self._protocol._require_reservation(reservation)


def _canonical_json_request(
    parser: PatchRequestParser,
    operation: OperationType,
    raw_arguments: bytes,
    correlation: PatchObserverCorrelationId,
) -> CanonicalPatchRequest:
    """Parse closed portable JSON without permitting freeform edit input."""
    if (
        type(parser) is not PatchRequestParser
        or type(operation) is not OperationType
        or type(raw_arguments) is not bytes
        or type(correlation) is not PatchObserverCorrelationId
    ):
        raise PatchProtocolError("patch protocol request is invalid")
    kind = (
        RawPatchInputKind.EDIT_JSON
        if operation is OperationType.EDIT
        else RawPatchInputKind.APPLY_JSON
    )
    try:
        return parser.parse(
            RawPatchIngress(
                RawProviderProfile(_TEST_PROFILE_NAME),
                RawToolCallId(correlation.value),
                kind,
                RawPatchInputState.COMPLETE,
                raw_arguments,
            )
        )
    except PatchInputError as error:
        raise PatchProtocolError(
            "patch protocol request is invalid"
        ) from error


def _same_reservation(
    first: PatchProtocolReservation | None,
    second: PatchProtocolReservation | None,
) -> bool:
    """Compare durable request authority while ignoring retry provenance."""
    return (
        type(first) is PatchProtocolReservation
        and type(second) is PatchProtocolReservation
        and first.surface is second.surface
        and first.identity == second.identity
        and first.operation is second.operation
        and first.correlation == second.correlation
        and first.request_id == second.request_id
        and first.digest == second.digest
    )


def _canonical_paths(
    request: CanonicalPatchRequest,
) -> frozenset[LogicalPath]:
    """Return the conservative complete logical footprint of a request."""
    if type(request) is not CanonicalPatchRequest:
        raise PatchProtocolError("patch coordination admission is denied")
    syntax = request.syntax
    if type(syntax) is StructuredEditSyntax:
        return frozenset((syntax.path,))
    if type(syntax) is not PatchDocumentSyntax:
        raise PatchProtocolError("patch coordination admission is denied")
    paths: set[LogicalPath] = set()
    for declaration in syntax.declarations:
        paths.add(declaration.path)
        if type(declaration) is UpdateDeclarationSyntax:
            if declaration.move_to is not None:
                paths.add(declaration.move_to)
    if not paths:
        raise PatchProtocolError("patch coordination admission is denied")
    return frozenset(paths)


def _apply_json_arguments(request: CanonicalPatchRequest) -> bytes:
    """Encode freeform syntax as the byte-identical portable JSON apply."""
    if (
        type(request) is not CanonicalPatchRequest
        or request.operation is not OperationType.APPLY
        or type(request.syntax) is not PatchDocumentSyntax
    ):
        raise PatchProtocolError("patch provider dispatch is unavailable")
    document = request.syntax.canonical_bytes.decode("utf-8")
    return b'{"patch":' + _json_string(document).encode("utf-8") + b"}"


def _json_string(value: str) -> str:
    """Encode one string for the fixed portable apply JSON projection."""
    if type(value) is not str:
        raise PatchProtocolError("patch provider dispatch is unavailable")
    return (
        '"'
        + value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\b", "\\b")
        .replace("\f", "\\f")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        + '"'
    )
