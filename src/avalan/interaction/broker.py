"""Coordinate async interaction delivery over an authoritative store."""

from .entities import (
    AnswerProvenance,
    BranchId,
    ContinuationId,
    DeadlineScheduleRevision,
    ExecutionOrigin,
    InputQuestion,
    InputRequest,
    InputRequestId,
    PrincipalScope,
    RequestState,
    RequirementMode,
    ResolutionStatus,
    ResumeInputContinuation,
    RunId,
    TerminateInputContinuation,
    _is_input_question_variant,
)
from .error import (
    InputContractError,
    InputErrorCode,
    InputValidationError,
    InteractionNotFoundError,
    InteractionStoreClosedError,
)
from .handler import (
    InputDisconnectReason,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerDisconnected,
    InputHandlerResolution,
    InputResumer,
    InputResumptionNotification,
    _InputHandler,
    _InputHandlerOutcome,
    _TrustedInputHandlerResolution,
    _validate_trusted_input_handler_resolution,
)
from .policy import (
    HandlerLossDisposition,
    InteractionActor,
    InteractionClock,
    InteractionIdFactory,
    InteractionPolicy,
    TaskInputClassifier,
)
from .state import InputTransitionError, project_resolution_to_model
from .store import (
    CancelInteractionApplied,
    CancelInteractionCommand,
    CancelInteractionRejected,
    CancelInteractionResult,
    ControllerActivityApplied,
    ControllerActivityRejected,
    ControllerActivityResult,
    ControllerLeaseExpiredApplied,
    CreateInteractionApplied,
    CreateInteractionRejected,
    CreateInteractionResult,
    DetachInteractionCommand,
    DueInteractionsApplied,
    DueInteractionsRejected,
    DueInteractionsResult,
    InteractionBranchRegistrationApplied,
    InteractionBranchRegistrationRejected,
    InteractionBranchRegistrationReplayed,
    InteractionBranchRegistrationResult,
    InteractionBranchRoot,
    InteractionBranchRootLookup,
    InteractionCorrelation,
    InteractionDisclosureProjection,
    InteractionPresentationApplied,
    InteractionPresentationRejected,
    InteractionPresentationResult,
    InteractionPresentationState,
    InteractionRecord,
    InteractionResolutionResult,
    InteractionStore,
    InteractionStoreReplayed,
    ListInteractionsCommand,
    PresentInteractionCommand,
    RecordControllerActivityCommand,
    RegisterInteractionBranchCommand,
    ResolutionDecisionStage,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    ScopeCancellationApplied,
    ScopeCancellationRejected,
    ScopeCancellationReplayed,
    ScopeCancellationResult,
    ScopedInteractionLookup,
    ScopeSupersessionApplied,
    ScopeSupersessionRejected,
    ScopeSupersessionReplayed,
    ScopeSupersessionResult,
    SupersedeInteractionScopeCommand,
    TerminalizeDueInteractionsCommand,
    TerminalizeInteractionApplied,
    TerminalizeInteractionCommand,
    TerminalizeInteractionRejected,
    TerminalizeInteractionResult,
    TerminalizeInteractionScopeCommand,
    TrustedDefaultResolutionApplied,
    TrustedDefaultResolutionRequest,
    TrustedDefaultResolutionResult,
    WaitForDeadlineChangeCommand,
    WaitForInteractionChangeCommand,
    _CandidateResolutionCommand,
    _InteractionAdmissionCleanupCommand,
    _InteractionAdmissionCleanupDisposition,
    _InteractionAdmissionCreateCommand,
    _new_interaction_admission_commands,
    _new_trusted_default_resolution_command,
    _new_trusted_policy_resolution_command,
    _validate_interaction_admission_cleanup_command,
    _validate_interaction_admission_cleanup_result,
    _validate_interaction_admission_create_command,
)
from .validation import (
    MAX_STATE_REVISION,
    validate_bool,
    validate_int,
    validate_opaque_id,
    validate_presentation_text,
    validate_state_revision,
)

from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    Lock,
    Queue,
    QueueFull,
    Task,
    create_task,
    current_task,
    gather,
    get_running_loop,
    shield,
    wait,
)
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from inspect import iscoroutinefunction
from typing import Callable, Protocol, TypeAlias, cast, final

_MAX_ATTACHED_HANDLER_RESOLUTION_ATTEMPTS = 2


class InteractionObserverEventKind(StrEnum):
    """Identify one content-safe best-effort broker observation."""

    CREATED = "created"
    PRESENTED = "presented"
    DETACHED = "detached"
    TERMINAL = "terminal"
    HANDLER_LOST = "handler_lost"
    RESUMER_FAILED = "resumer_failed"
    DEADLINE_REARMED = "deadline_rearmed"
    DEADLINE_PUMP_FAILED = "deadline_pump_failed"
    OVERFLOW = "overflow"
    HEARTBEAT = "heartbeat"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionObserverEvent:
    """Expose bounded lifecycle metadata without request or answer content."""

    kind: InteractionObserverEventKind
    request_id: InputRequestId | None = None
    status: ResolutionStatus | None = None
    schedule_revision: DeadlineScheduleRevision | None = None
    dropped_events: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, InteractionObserverEventKind):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "observer.kind",
                "value must be an interaction observer event kind",
            )
        if self.request_id is not None:
            object.__setattr__(
                self,
                "request_id",
                InputRequestId(
                    validate_opaque_id(
                        self.request_id,
                        "observer.request_id",
                    )
                ),
            )
        if self.status is not None and not isinstance(
            self.status,
            ResolutionStatus,
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "observer.status",
                "value must be a resolution status",
            )
        if self.schedule_revision is not None:
            object.__setattr__(
                self,
                "schedule_revision",
                DeadlineScheduleRevision(
                    validate_state_revision(
                        self.schedule_revision,
                        "observer.schedule_revision",
                    )
                ),
            )
        object.__setattr__(
            self,
            "dropped_events",
            validate_int(
                self.dropped_events,
                "observer.dropped_events",
                minimum=0,
                maximum=MAX_STATE_REVISION,
            ),
        )
        if self.kind is InteractionObserverEventKind.OVERFLOW:
            if (
                self.dropped_events < 1
                or self.request_id is not None
                or self.status is not None
                or self.schedule_revision is not None
            ):
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "observer",
                    "overflow events carry only a positive drop count",
                )
            return
        if self.dropped_events != 0:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "observer.dropped_events",
                "only overflow events may carry a drop count",
            )
        if self.kind is InteractionObserverEventKind.TERMINAL:
            if self.request_id is None or self.status is None:
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "observer",
                    "terminal events require request identity and status",
                )
        elif self.status is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "observer.status",
                "only terminal events may carry a resolution status",
            )
        if self.kind is InteractionObserverEventKind.DEADLINE_REARMED:
            if self.schedule_revision is None:
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "observer.schedule_revision",
                    "deadline events require a schedule revision",
                )
        elif self.schedule_revision is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "observer.schedule_revision",
                "only deadline events may carry a schedule revision",
            )
        request_scoped_kinds = {
            InteractionObserverEventKind.CREATED,
            InteractionObserverEventKind.PRESENTED,
            InteractionObserverEventKind.DETACHED,
            InteractionObserverEventKind.HANDLER_LOST,
            InteractionObserverEventKind.RESUMER_FAILED,
        }
        if self.kind in request_scoped_kinds and self.request_id is None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "observer.request_id",
                "request lifecycle events require request identity",
            )
        broker_scoped_kinds = {
            InteractionObserverEventKind.DEADLINE_PUMP_FAILED,
            InteractionObserverEventKind.HEARTBEAT,
        }
        if self.kind in broker_scoped_kinds and self.request_id is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "observer.request_id",
                "broker lifecycle events cannot carry request identity",
            )


class InteractionObserver(Protocol):
    """Consume one best-effort content-safe broker observation."""

    async def __call__(self, event: InteractionObserverEvent) -> None:
        """Observe one lifecycle event without affecting correctness."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionBrokerRequest:
    """Describe one broker-minted canonical interaction request."""

    actor: InteractionActor
    origin: ExecutionOrigin
    mode: RequirementMode
    reason: str
    questions: tuple[InputQuestion, ...]
    handler: _InputHandler | None = field(default=None, repr=False)
    resumer: InputResumer | None = field(default=None, repr=False)
    continuation_ttl_seconds: int = 86_400
    advisory_wait_seconds: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.actor, InteractionActor):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "broker_request.actor",
                "value must be an interaction actor",
            )
        if not isinstance(self.origin, ExecutionOrigin):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "broker_request.origin",
                "value must be an execution origin",
            )
        if self.actor.principal != self.origin.principal:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "broker_request.origin.principal",
                "request origin must belong to the authenticated actor",
            )
        if not isinstance(self.mode, RequirementMode):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "broker_request.mode",
                "value must be a requirement mode",
            )
        object.__setattr__(
            self,
            "reason",
            validate_presentation_text(
                self.reason,
                "broker_request.reason",
                minimum=1,
                maximum=500,
                maximum_bytes=2_000,
            ),
        )
        if not isinstance(self.questions, tuple):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "broker_request.questions",
                "questions must be a tuple",
            )
        if len(self.questions) < 1 or len(self.questions) > 3:
            raise InputValidationError(
                InputErrorCode.OUT_OF_BOUNDS,
                "broker_request.questions",
                "a request must contain one to three questions",
            )
        question_ids: set[object] = set()
        for question in self.questions:
            if not _is_input_question_variant(question):
                raise InputValidationError(
                    InputErrorCode.INVALID_TYPE,
                    "broker_request.questions",
                    "questions must contain typed question variants",
                )
            if question.question_id in question_ids:
                raise InputValidationError(
                    InputErrorCode.DUPLICATE,
                    "broker_request.questions",
                    "question identifiers must be unique",
                )
            question_ids.add(question.question_id)
        object.__setattr__(
            self,
            "continuation_ttl_seconds",
            validate_int(
                self.continuation_ttl_seconds,
                "broker_request.continuation_ttl_seconds",
                minimum=60,
                maximum=604_800,
            ),
        )
        if self.mode is RequirementMode.REQUIRED:
            if self.advisory_wait_seconds is not None:
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "broker_request.advisory_wait_seconds",
                    "required requests cannot have an advisory wait",
                )
        else:
            object.__setattr__(
                self,
                "advisory_wait_seconds",
                (
                    60
                    if self.advisory_wait_seconds is None
                    else validate_int(
                        self.advisory_wait_seconds,
                        "broker_request.advisory_wait_seconds",
                        minimum=1,
                        maximum=3_600,
                    )
                ),
            )
        if self.handler is not None:
            _validate_async_callable(self.handler, "broker_request.handler")
        if self.resumer is not None:
            _validate_async_callable(self.resumer, "broker_request.resumer")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionDelivery:
    """Return the latest authoritative state after broker delivery."""

    correlation: InteractionCorrelation
    record: InteractionRecord
    handler_attempts: int
    resumer_failed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.correlation, InteractionCorrelation):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "delivery.correlation",
                "value must be an interaction correlation",
            )
        if not isinstance(self.record, InteractionRecord):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "delivery.record",
                "value must be an interaction record",
            )
        if self.record.correlation != self.correlation:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "delivery.record",
                "delivery record does not match its correlation",
            )
        object.__setattr__(
            self,
            "handler_attempts",
            validate_int(
                self.handler_attempts,
                "delivery.handler_attempts",
                minimum=0,
                maximum=MAX_STATE_REVISION,
            ),
        )
        validate_bool(self.resumer_failed, "delivery.resumer_failed")


InteractionBrokerStoreResult: TypeAlias = (
    CreateInteractionResult
    | InteractionPresentationResult
    | InteractionResolutionResult
    | TrustedDefaultResolutionResult
    | TerminalizeInteractionResult
    | CancelInteractionResult
    | ScopeCancellationResult
    | ScopeSupersessionResult
    | InteractionBranchRegistrationResult
    | ControllerActivityResult
    | DueInteractionsResult
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionBrokerResult:
    """Return one authoritative store result and callback delivery status."""

    store_result: InteractionBrokerStoreResult
    resumer_failed: bool = False

    def __post_init__(self) -> None:
        if not _is_broker_store_result(self.store_result):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "broker_result.store_result",
                "value must be a supported interaction store result",
            )
        validate_bool(self.resumer_failed, "broker_result.resumer_failed")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionRequestResult:
    """Return atomic admission and optional completed delivery."""

    create_result: CreateInteractionResult
    delivery: InteractionDelivery | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.create_result,
            (CreateInteractionApplied, CreateInteractionRejected),
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "request_result.create_result",
                "value must be a create interaction result",
            )
        if isinstance(self.create_result, CreateInteractionRejected):
            if self.delivery is not None:
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "request_result.delivery",
                    "rejected admission cannot have a delivery",
                )
            return
        if not isinstance(self.delivery, InteractionDelivery):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "request_result.delivery",
                "applied admission requires an interaction delivery",
            )
        if self.delivery.correlation != self.create_result.record.correlation:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "request_result.delivery",
                "delivery does not match the admitted request",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionBrokerHeartbeat:
    """Return one monotonic broker-local liveness sequence."""

    sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            validate_int(
                self.sequence,
                "heartbeat.sequence",
                minimum=1,
                maximum=MAX_STATE_REVISION,
            ),
        )


class InteractionBroker(Protocol):
    """Coordinate interaction delivery without owning lifecycle truth."""

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        """Persist, arbitrate, and deliver one broker-minted request."""
        ...

    async def inspect(
        self,
        query: ScopedInteractionLookup,
    ) -> InteractionDisclosureProjection | None:
        """Return one authorized store projection or absence."""
        ...

    async def list(
        self,
        command: ListInteractionsCommand,
    ) -> tuple[InteractionDisclosureProjection, ...]:
        """Return authorized projections in one execution scope."""
        ...

    async def resolve(
        self,
        command: ResolveInteractionCommand,
    ) -> InteractionBrokerResult:
        """Submit one candidate resolution atomically."""
        ...

    async def resolve_trusted_default(
        self,
        request: TrustedDefaultResolutionRequest,
    ) -> InteractionBrokerResult:
        """Resolve declared defaults through store-owned authority."""
        ...

    async def terminalize(
        self,
        command: TerminalizeInteractionCommand,
    ) -> InteractionBrokerResult:
        """Apply one explicit unavailable or superseded transition."""
        ...

    async def cancel(
        self,
        command: CancelInteractionCommand,
    ) -> InteractionBrokerResult:
        """Cancel one exact request."""
        ...

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        """Cancel one complete authorized execution scope."""
        ...

    async def supersede(
        self,
        command: SupersedeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        """Supersede one complete authorized execution scope."""
        ...

    async def register_branch(
        self,
        command: RegisterInteractionBranchCommand,
    ) -> InteractionBrokerResult:
        """Persist one child branch edge."""
        ...

    async def record_activity(
        self,
        command: RecordControllerActivityCommand,
    ) -> InteractionBrokerResult:
        """Record authenticated active-control activity."""
        ...

    async def wait(
        self,
        command: WaitForInteractionChangeCommand,
    ) -> InteractionDisclosureProjection:
        """Wait independently for one newer authorized projection."""
        ...

    async def heartbeat(self) -> InteractionBrokerHeartbeat:
        """Return broker-local liveness without consulting lifecycle state."""
        ...

    async def aclose(self) -> None:
        """Idempotently close the broker and its owned store handle."""
        ...


_RootKey: TypeAlias = tuple[PrincipalScope, RunId, BranchId]


@dataclass(slots=True, eq=False)
class _PresentationTicket:
    """Queue one owning call for a root presentation slot."""

    key: _RootKey
    ready: Future[None]


@dataclass(frozen=True, slots=True)
class _ResumptionRegistration:
    """Bind one broker-owned delivery task to its handoff barrier."""

    task: Task[bool] | None
    handoff: Future[None]


@final
class AsyncInteractionBroker:
    """Implement non-authoritative async interaction coordination."""

    def __init__(
        self,
        *,
        store: InteractionStore,
        clock: InteractionClock,
        id_factory: InteractionIdFactory,
        policy: InteractionPolicy,
        classifier: TaskInputClassifier,
        observer: InteractionObserver | None = None,
        observer_queue_capacity: int = 64,
    ) -> None:
        _validate_async_methods(
            store,
            (
                "create",
                "create_admission",
                "cleanup_admission",
                "lookup_scoped",
                "lookup_branch_root",
                "list_scoped",
                "mark_presented",
                "mark_detached",
                "resolve",
                "resolve_trusted_default",
                "terminalize",
                "cancel",
                "terminalize_scope",
                "supersede_scope",
                "register_branch",
                "record_activity",
                "wait_for_change",
                "next_deadline",
                "wait_for_deadline_change",
                "terminalize_due",
                "aclose",
            ),
            "broker.store",
        )
        _validate_async_methods(
            clock,
            ("read", "wait_until"),
            "broker.clock",
        )
        _validate_async_methods(
            id_factory,
            (
                "new_request_id",
                "new_continuation_id",
                "new_idempotency_key",
                "new_active_control_lease_nonce",
            ),
            "broker.id_factory",
        )
        if not isinstance(policy, InteractionPolicy):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "broker.policy",
                "value must be an interaction policy",
            )
        _validate_async_methods(
            classifier,
            ("classify_task_input",),
            "broker.classifier",
        )
        if observer is not None:
            _validate_async_callable(observer, "broker.observer")
        capacity = validate_int(
            observer_queue_capacity,
            "broker.observer_queue_capacity",
            minimum=1,
            maximum=1_024,
        )
        self._store = store
        self._clock = clock
        self._id_factory = id_factory
        self._policy = policy
        self._classifier = classifier
        self._observer = observer
        self._observer_queue = (
            Queue[InteractionObserverEvent](maxsize=capacity)
            if observer is not None
            else None
        )
        self._observer_dropped = 0
        self._observer_task: Task[None] | None = None
        self._deadline_task: Task[None] | None = None
        self._close_task: Task[None] | None = None
        self._close_owned_tasks: tuple[Task[object], ...] = ()
        self._close_excluded_task: Task[object] | None = None
        self._state_lock = Lock()
        self._start_lock = Lock()
        self._closed = False
        self._closing_settlements = False
        self._started = False
        self._heartbeat_sequence = 0
        self._root_queues: dict[
            _RootKey,
            deque[_PresentationTicket],
        ] = {}
        self._create_tasks: dict[
            InputRequestId,
            Task[CreateInteractionResult],
        ] = {}
        self._admission_cleanups: dict[
            InputRequestId,
            _InteractionAdmissionCleanupCommand,
        ] = {}
        self._handler_tasks: dict[
            InputRequestId,
            Task[_InputHandlerOutcome],
        ] = {}
        self._watcher_tasks: dict[InputRequestId, Task[None]] = {}
        self._resumers: dict[ContinuationId, InputResumer] = {}
        self._request_resumers: dict[InputRequestId, ContinuationId] = {}
        self._resumer_tasks: dict[InputRequestId, Task[bool]] = {}
        self._resumer_handoffs: dict[InputRequestId, Future[None]] = {}
        self._store_handoffs_acknowledged: set[InputRequestId] = set()
        self._cleanup_tasks: set[Task[None]] = set()
        self._resumer_failures: set[InputRequestId] = set()
        self._settlement_started: set[InputRequestId] = set()
        self._terminal_observed: set[InputRequestId] = set()

    @property
    def is_closed(self) -> bool:
        """Return whether this broker handle has closed."""
        return self._closed

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        """Persist, arbitrate, and deliver one broker-minted request."""
        if not isinstance(request, InteractionBrokerRequest):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "broker.request",
                "value must be an interaction broker request",
            )
        await self._ensure_started()
        observed_at = await self._clock.read()
        request_id = await self._id_factory.new_request_id()
        continuation_id = await self._id_factory.new_continuation_id()
        canonical = InputRequest(
            request_id=request_id,
            continuation_id=continuation_id,
            origin=request.origin,
            mode=request.mode,
            reason=request.reason,
            questions=request.questions,
            created_at=observed_at.wall_time,
            continuation_ttl_seconds=request.continuation_ttl_seconds,
            advisory_wait_seconds=request.advisory_wait_seconds,
        )
        store_resumer = _StoreBoundResumer(
            broker=self,
            request_id=request_id,
        )
        admission_command, cleanup_command = (
            _new_interaction_admission_commands(
                actor=request.actor,
                request=canonical,
                resumer=store_resumer,
            )
        )
        if request.resumer is not None:
            await self._bind_resumer(
                request_id,
                continuation_id,
                request.resumer,
            )
        async with self._state_lock:
            self._ensure_open()
            self._admission_cleanups[request_id] = cleanup_command
            create_operation = create_task(
                self._create_and_track(admission_command),
                name=f"interaction-create-{request_id}",
            )
            self._create_tasks[request_id] = create_operation
        try:
            try:
                create_result = await shield(create_operation)
            except CancelledError:
                create_result = await create_operation
                if isinstance(create_result, CreateInteractionApplied):
                    self._emit_record(
                        InteractionObserverEventKind.CREATED,
                        create_result.record,
                    )
                    await self._start_watcher(
                        request.actor,
                        create_result.record,
                    )
                    await self._await_cancellation_cleanup(
                        request,
                        create_result.record,
                    )
                else:
                    await self._release_admission_cleanup(
                        request_id,
                        cleanup_command,
                    )
                    await self._unbind_resumer(
                        request_id,
                        continuation_id,
                    )
                raise
        except CancelledError:
            if create_operation.cancelled():
                await self._unbind_resumer(request_id, continuation_id)
            raise
        except BaseException:
            await self._cleanup_failed_admission(
                request_id,
                cleanup_command,
            )
            await self._unbind_resumer(request_id, continuation_id)
            raise
        finally:
            async with self._state_lock:
                if self._create_tasks.get(request_id) is create_operation:
                    self._create_tasks.pop(request_id, None)
        if isinstance(create_result, CreateInteractionRejected):
            await self._release_admission_cleanup(
                request_id,
                cleanup_command,
            )
            await self._unbind_resumer(request_id, continuation_id)
            return InteractionRequestResult(
                create_result=create_result,
                delivery=None,
            )
        self._emit_record(
            InteractionObserverEventKind.CREATED,
            create_result.record,
        )
        await self._start_watcher(request.actor, create_result.record)
        ticket: _PresentationTicket | None = None
        try:
            root_branch_id = await self._resolve_root_branch(
                request.actor,
                create_result.record,
            )
            if root_branch_id is None:
                latest = await self._apply_handler_loss(
                    request.actor,
                    create_result.record,
                )
                return InteractionRequestResult(
                    create_result=create_result,
                    delivery=await self._delivery(
                        create_result.record.correlation,
                        latest,
                        0,
                    ),
                )
            ticket = await self._acquire_presentation_slot(
                request.origin,
                root_branch_id,
            )
            delivery = await self._deliver_request(
                request,
                create_result.record,
            )
            return InteractionRequestResult(
                create_result=create_result,
                delivery=delivery,
            )
        except CancelledError:
            await self._await_cancellation_cleanup(
                request,
                create_result.record,
            )
            raise
        finally:
            if ticket is not None:
                await self._release_presentation_slot(ticket)

    async def inspect(
        self,
        query: ScopedInteractionLookup,
    ) -> InteractionDisclosureProjection | None:
        """Return one authorized store projection or absence."""
        await self._ensure_started()
        return await self._store.lookup_scoped(query)

    async def list(
        self,
        command: ListInteractionsCommand,
    ) -> tuple[InteractionDisclosureProjection, ...]:
        """Return authorized projections in one execution scope."""
        await self._ensure_started()
        return await self._store.list_scoped(command)

    async def resolve(
        self,
        command: ResolveInteractionCommand,
    ) -> InteractionBrokerResult:
        """Submit one candidate resolution atomically."""
        await self._ensure_started()
        result = await self._store.resolve(command)
        failed = await self._settle_store_result(result)
        return InteractionBrokerResult(
            store_result=result,
            resumer_failed=failed,
        )

    async def resolve_trusted_default(
        self,
        request: TrustedDefaultResolutionRequest,
    ) -> InteractionBrokerResult:
        """Resolve declared defaults through store-owned authority."""
        await self._ensure_started()
        result = await self._store.resolve_trusted_default(
            _new_trusted_default_resolution_command(request)
        )
        failed = await self._settle_store_result(result)
        return InteractionBrokerResult(
            store_result=result,
            resumer_failed=failed,
        )

    async def terminalize(
        self,
        command: TerminalizeInteractionCommand,
    ) -> InteractionBrokerResult:
        """Apply one explicit unavailable or superseded transition."""
        await self._ensure_started()
        result = await self._store.terminalize(command)
        failed = await self._settle_store_result(result)
        return InteractionBrokerResult(
            store_result=result,
            resumer_failed=failed,
        )

    async def cancel(
        self,
        command: CancelInteractionCommand,
    ) -> InteractionBrokerResult:
        """Cancel one exact request."""
        await self._ensure_started()
        result = await self._store.cancel(command)
        failed = await self._settle_store_result(result)
        return InteractionBrokerResult(
            store_result=result,
            resumer_failed=failed,
        )

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        """Cancel one complete authorized execution scope."""
        await self._ensure_started()
        result = await self._store.terminalize_scope(command)
        failed = await self._settle_store_result(result)
        return InteractionBrokerResult(
            store_result=result,
            resumer_failed=failed,
        )

    async def supersede(
        self,
        command: SupersedeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        """Supersede one complete authorized execution scope."""
        await self._ensure_started()
        result = await self._store.supersede_scope(command)
        failed = await self._settle_store_result(result)
        return InteractionBrokerResult(
            store_result=result,
            resumer_failed=failed,
        )

    async def register_branch(
        self,
        command: RegisterInteractionBranchCommand,
    ) -> InteractionBrokerResult:
        """Persist one child branch edge."""
        await self._ensure_started()
        result = await self._store.register_branch(command)
        return InteractionBrokerResult(store_result=result)

    async def record_activity(
        self,
        command: RecordControllerActivityCommand,
    ) -> InteractionBrokerResult:
        """Record authenticated active-control activity."""
        await self._ensure_started()
        result = await self._store.record_activity(command)
        failed = await self._settle_store_result(result)
        return InteractionBrokerResult(
            store_result=result,
            resumer_failed=failed,
        )

    async def wait(
        self,
        command: WaitForInteractionChangeCommand,
    ) -> InteractionDisclosureProjection:
        """Wait independently for one newer authorized projection."""
        await self._ensure_started()
        return await self._store.wait_for_change(command)

    async def heartbeat(self) -> InteractionBrokerHeartbeat:
        """Return broker-local liveness without consulting lifecycle state."""
        await self._ensure_started()
        async with self._state_lock:
            self._heartbeat_sequence += 1
            sequence = self._heartbeat_sequence
        self._emit(
            InteractionObserverEvent(
                kind=InteractionObserverEventKind.HEARTBEAT,
            )
        )
        return InteractionBrokerHeartbeat(sequence=sequence)

    async def aclose(self) -> None:
        """Idempotently close the broker and its owned store handle.

        A broker-owned task reentering an existing close cooperates by
        returning to the outer barrier that already owns its final join.
        Independent callers await the complete shared close barrier.
        """
        async with self._state_lock:
            close_task = self._close_task
            if (
                close_task is not None
                and close_task.done()
                and (
                    close_task.cancelled()
                    or close_task.exception() is not None
                )
            ):
                self._close_task = None
                close_task = None
            invoking_task = current_task()
            if (
                close_task is not None
                and not close_task.done()
                and invoking_task is not None
            ):
                active_owned_tasks = (
                    *self._close_owned_tasks,
                    *self._create_tasks.values(),
                    *self._handler_tasks.values(),
                    *self._watcher_tasks.values(),
                    *self._resumer_tasks.values(),
                    *self._cleanup_tasks,
                    self._deadline_task,
                    self._observer_task,
                )
                if invoking_task in active_owned_tasks:
                    return
            if close_task is None:
                self._closed = True
                self._closing_settlements = True
                deadline_task = self._deadline_task
                observer_task = self._observer_task
                create_tasks = tuple(self._create_tasks.values())
                handler_tasks = tuple(self._handler_tasks.values())
                watcher_tasks = tuple(self._watcher_tasks.values())
                resumer_tasks = tuple(self._resumer_tasks.values())
                cleanup_tasks = tuple(self._cleanup_tasks)
                tickets = tuple(
                    ticket
                    for queue in self._root_queues.values()
                    for ticket in queue
                )
                excluded_task = current_task()
                self._close_excluded_task = excluded_task
                captured_tasks = cast(
                    tuple[Task[object], ...],
                    tuple(
                        task
                        for task in (
                            deadline_task,
                            observer_task,
                            *create_tasks,
                            *handler_tasks,
                            *watcher_tasks,
                            *resumer_tasks,
                            *cleanup_tasks,
                        )
                        if task is not None
                    ),
                )
                self._close_owned_tasks = tuple(
                    dict.fromkeys((*self._close_owned_tasks, *captured_tasks))
                )
                self._deadline_task = None
                self._observer_task = None
                self._root_queues.clear()
                close_task = create_task(
                    self._finish_close(
                        tickets,
                        create_tasks,
                        excluded_task,
                    ),
                    name="interaction-broker-close",
                )
                self._close_task = close_task
        assert close_task is not None
        await shield(close_task)

    async def _finish_close(
        self,
        tickets: tuple[_PresentationTicket, ...],
        create_tasks: tuple[Task[CreateInteractionResult], ...],
        excluded_task: Task[object] | None,
    ) -> None:
        """Finish one shared close after atomic admission shutdown."""
        for ticket in tickets:
            if not ticket.ready.done():
                ticket.ready.set_exception(InteractionStoreClosedError())
        active_create_tasks = tuple(
            task for task in create_tasks if task is not excluded_task
        )
        for create_operation in active_create_tasks:
            create_operation.cancel()
        if active_create_tasks:
            await gather(*active_create_tasks, return_exceptions=True)
        async with self._state_lock:
            admission_cleanups = tuple(self._admission_cleanups.items())
        for request_id, cleanup_command in sorted(
            admission_cleanups,
            key=lambda item: str(item[0]),
        ):
            cleanup_command = _validate_interaction_admission_cleanup_command(
                cleanup_command
            )
            result = await self._store.cleanup_admission(cleanup_command)
            _validate_interaction_admission_cleanup_result(
                result,
                path="broker.store.cleanup_admission",
            )
            if (
                result.disposition
                is not _InteractionAdmissionCleanupDisposition.ABSENT
            ):
                await self._release_admission_cleanup(
                    request_id,
                    cleanup_command,
                )
        async with self._state_lock:
            current = current_task()
            owned_tasks = cast(
                tuple[Task[object], ...],
                tuple(
                    task
                    for task in (
                        *self._close_owned_tasks,
                        *self._handler_tasks.values(),
                        *self._watcher_tasks.values(),
                        *self._resumer_tasks.values(),
                        *self._cleanup_tasks,
                    )
                    if task is not current and task is not excluded_task
                ),
            )
        unique_tasks = tuple(dict.fromkeys(owned_tasks))
        for owned_task in unique_tasks:
            owned_task.cancel()
        if unique_tasks:
            await gather(*unique_tasks, return_exceptions=True)
        await self._store.aclose()
        finalization = create_task(
            self._finalize_close(),
            name="interaction-broker-close-finalization",
        )
        await _await_shielded_cleanup(finalization)

    async def _finalize_close(self) -> None:
        """Release retained state after store close forbids late commits."""
        async with self._state_lock:
            self._create_tasks.clear()
            self._handler_tasks.clear()
            self._watcher_tasks.clear()
            self._resumers.clear()
            self._request_resumers.clear()
            self._resumer_tasks.clear()
            self._resumer_handoffs.clear()
            self._store_handoffs_acknowledged.clear()
            self._cleanup_tasks.clear()
            self._admission_cleanups.clear()
            self._close_owned_tasks = ()
            self._close_excluded_task = None
            self._settlement_started.clear()
            self._terminal_observed.clear()
            self._closing_settlements = False

    async def _create_and_track(
        self,
        command: _InteractionAdmissionCreateCommand,
    ) -> CreateInteractionResult:
        """Create one request while close retains its cleanup capability."""
        command = _validate_interaction_admission_create_command(command)
        return await self._store.create_admission(command)

    async def _release_admission_cleanup(
        self,
        request_id: InputRequestId,
        command: _InteractionAdmissionCleanupCommand,
    ) -> None:
        """Release authority only after a conclusive store outcome."""
        async with self._state_lock:
            if self._admission_cleanups.get(request_id) is command:
                self._admission_cleanups.pop(request_id, None)

    async def _cleanup_failed_admission(
        self,
        request_id: InputRequestId,
        command: _InteractionAdmissionCleanupCommand,
    ) -> None:
        """Best-effort cleanup after a create call fails outside close."""
        try:
            command = _validate_interaction_admission_cleanup_command(command)
            result = await self._store.cleanup_admission(command)
        except (CancelledError, InteractionStoreClosedError):
            return
        try:
            _validate_interaction_admission_cleanup_result(
                result,
                path="broker.store.cleanup_admission",
            )
        except InputValidationError:
            return
        await self._release_admission_cleanup(request_id, command)

    async def _ensure_started(self) -> None:
        self._ensure_open()
        if self._started:
            return
        async with self._start_lock:
            self._ensure_open()
            if self._started:
                return
            if self._observer is not None:
                self._observer_task = create_task(
                    self._observer_loop(),
                    name="interaction-observer",
                )
            self._deadline_task = create_task(
                self._deadline_loop(),
                name="interaction-deadline-pump",
            )
            self._started = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise InteractionStoreClosedError()

    async def _bind_resumer(
        self,
        request_id: InputRequestId,
        continuation_id: ContinuationId,
        resumer: InputResumer,
    ) -> None:
        async with self._state_lock:
            self._ensure_open()
            if continuation_id in self._resumers:
                raise InputValidationError(
                    InputErrorCode.DUPLICATE,
                    "broker.resumer.continuation_id",
                    "continuation already has an in-process resumer",
                )
            self._resumers[continuation_id] = resumer
            self._request_resumers[request_id] = continuation_id

    async def _unbind_resumer(
        self,
        request_id: InputRequestId,
        continuation_id: ContinuationId,
    ) -> None:
        async with self._state_lock:
            if self._request_resumers.get(request_id) == continuation_id:
                self._request_resumers.pop(request_id, None)
                self._resumers.pop(continuation_id, None)

    async def _acquire_presentation_slot(
        self,
        origin: ExecutionOrigin,
        root_branch_id: BranchId,
    ) -> _PresentationTicket:
        ready = get_running_loop().create_future()
        async with self._state_lock:
            self._ensure_open()
            key = (
                origin.principal,
                origin.run_id,
                root_branch_id,
            )
            ticket = _PresentationTicket(key=key, ready=ready)
            queue = self._root_queues.setdefault(key, deque())
            queue.append(ticket)
            if len(queue) == 1:
                ready.set_result(None)
        try:
            await ready
            return ticket
        except BaseException:
            await self._await_presentation_release(ticket)
            raise

    async def _await_presentation_release(
        self,
        ticket: _PresentationTicket,
    ) -> None:
        cleanup = create_task(
            self._release_presentation_slot(ticket),
            name="interaction-presentation-release",
        )
        await _await_shielded_cleanup(cleanup)

    async def _resolve_root_branch(
        self,
        actor: InteractionActor,
        record: InteractionRecord,
    ) -> BranchId | None:
        """Resolve one content-free authoritative presentation root."""
        origin = record.request.origin
        if origin.parent_branch_id is None:
            return origin.branch_id
        root = await self._store.lookup_branch_root(
            InteractionBranchRootLookup(
                actor=actor,
                run_id=origin.run_id,
                branch_id=origin.branch_id,
            )
        )
        if (
            not isinstance(root, InteractionBranchRoot)
            or root.run_id != origin.run_id
            or root.branch_id != origin.branch_id
        ):
            return None
        return root.root_branch_id

    async def _release_presentation_slot(
        self,
        ticket: _PresentationTicket,
    ) -> None:
        async with self._state_lock:
            queue = self._root_queues.get(ticket.key)
            if queue is None:
                return
            was_head = bool(queue) and queue[0] is ticket
            try:
                queue.remove(ticket)
            except ValueError:
                return
            if not queue:
                self._root_queues.pop(ticket.key, None)
                return
            if was_head and not queue[0].ready.done():
                queue[0].ready.set_result(None)

    async def _deliver_request(
        self,
        request: InteractionBrokerRequest,
        created: InteractionRecord,
    ) -> InteractionDelivery:
        correlation = created.correlation
        latest = await self._latest_record(request.actor, correlation)
        if _is_terminal_record(latest):
            await self._settle_record(latest)
            return await self._delivery(correlation, latest, 0)
        if latest.presentation is not InteractionPresentationState.QUEUED:
            return await self._delivery(correlation, latest, 0)
        if request.handler is None:
            latest = await self._apply_handler_loss(
                request.actor,
                latest,
            )
            return await self._delivery(correlation, latest, 0)
        presentation_result = await self._store.mark_presented(
            PresentInteractionCommand(
                actor=request.actor,
                correlation=correlation,
                expected_store_revision=latest.store_revision,
            )
        )
        await self._settle_store_result(presentation_result)
        if isinstance(presentation_result, InteractionPresentationApplied):
            latest = presentation_result.record
            self._emit_record(
                InteractionObserverEventKind.PRESENTED,
                latest,
            )
        else:
            result_record = _single_result_record(presentation_result)
            latest = (
                result_record
                if result_record is not None
                else await self._latest_record(request.actor, correlation)
            )
        if (
            _is_terminal_record(latest)
            or latest.presentation
            is not InteractionPresentationState.PRESENTED
        ):
            return await self._delivery(correlation, latest, 0)
        latest, attempts = await self._drive_handler(
            request.actor,
            correlation,
            latest,
            request.handler,
        )
        return await self._delivery(correlation, latest, attempts)

    async def _drive_handler(
        self,
        actor: InteractionActor,
        correlation: InteractionCorrelation,
        record: InteractionRecord,
        handler: _InputHandler,
    ) -> tuple[InteractionRecord, int]:
        validation_error: InputTransitionError | None = None
        attempts = 0
        latest = record
        while True:
            if _is_terminal_record(latest):
                await self._settle_record(latest)
                return latest, attempts
            if (
                validation_error is not None
                and attempts >= _MAX_ATTACHED_HANDLER_RESOLUTION_ATTEMPTS
            ):
                latest = await self._apply_handler_loss(actor, latest)
                return latest, attempts
            context = InputHandlerContext(
                request=latest.request,
                validation_error=validation_error,
            )
            attempts += 1
            try:
                outcome = await self._invoke_handler(
                    latest.request.request_id,
                    handler,
                    context,
                )
            except CancelledError:
                task = current_task()
                if task is not None and task.cancelling():
                    raise
                latest = await self._latest_record(actor, correlation)
                if _is_terminal_record(latest):
                    await self._settle_record(latest)
                    return latest, attempts
                latest = await self._apply_handler_loss(actor, latest)
                return latest, attempts
            except Exception:
                latest = await self._apply_handler_loss(actor, latest)
                return latest, attempts
            if isinstance(
                outcome,
                (InputHandlerResolution, _TrustedInputHandlerResolution),
            ):
                try:
                    command: _CandidateResolutionCommand
                    if isinstance(
                        outcome,
                        _TrustedInputHandlerResolution,
                    ):
                        trusted_outcome = (
                            _validate_trusted_input_handler_resolution(outcome)
                        )
                        if trusted_outcome.trusted_default:
                            default_result = (
                                await self._store.resolve_trusted_default(
                                    _new_trusted_default_resolution_command(
                                        TrustedDefaultResolutionRequest(
                                            actor=actor,
                                            correlation=correlation,
                                            expected_state_revision=(
                                                latest.request.state_revision
                                            ),
                                        )
                                    )
                                )
                            )
                            await self._settle_store_result(default_result)
                            result_record = _single_result_record(
                                default_result
                            )
                            latest = (
                                result_record
                                if result_record is not None
                                else await self._latest_record(
                                    actor,
                                    correlation,
                                )
                            )
                            return latest, attempts
                        resolution = trusted_outcome.resolution
                        assert resolution is not None
                        command = _new_trusted_policy_resolution_command(
                            actor=actor,
                            correlation=correlation,
                            expected_state_revision=(
                                latest.request.state_revision
                            ),
                            idempotency_key=(
                                await self._id_factory.new_idempotency_key()
                            ),
                            proposed_resolution=resolution,
                        )
                    else:
                        command = ResolveInteractionCommand(
                            actor=actor,
                            correlation=correlation,
                            expected_state_revision=(
                                latest.request.state_revision
                            ),
                            idempotency_key=(
                                await self._id_factory.new_idempotency_key()
                            ),
                            proposed_resolution=outcome.resolution,
                        )
                except InputContractError as error:
                    validation_error = InputTransitionError(
                        code=error.code,
                        path=error.path,
                        message=error.safe_message,
                    )
                    latest = await self._latest_record(actor, correlation)
                    continue
                result = await self._store.resolve(command)
                await self._settle_store_result(result)
                if (
                    isinstance(result, ResolveInteractionRejected)
                    and result.decision_stage
                    is ResolutionDecisionStage.VALIDATION
                ):
                    latest = await self._latest_record(actor, correlation)
                    if _is_terminal_record(latest):
                        await self._settle_record(latest)
                        return latest, attempts
                    validation_error = result.error
                    continue
                result_record = _single_result_record(result)
                latest = (
                    result_record
                    if result_record is not None
                    else await self._latest_record(actor, correlation)
                )
                return latest, attempts
            if (
                isinstance(outcome, InputHandlerDisconnected)
                and outcome.reason is InputDisconnectReason.HANDLER_CANCELLED
            ):
                latest = await self._apply_handler_cancellation(actor, latest)
                return latest, attempts
            if isinstance(
                outcome,
                (InputHandlerDetached, InputHandlerDisconnected),
            ):
                latest = await self._apply_handler_loss(actor, latest)
                return latest, attempts
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.outcome",
                "attached handler returned an unsupported outcome",
            )

    async def _invoke_handler(
        self,
        request_id: InputRequestId,
        handler: _InputHandler,
        context: InputHandlerContext,
    ) -> _InputHandlerOutcome:
        async with self._state_lock:
            self._ensure_open()
            previous = self._handler_tasks.get(request_id)
            if previous is not None and not previous.done():
                raise InputValidationError(
                    InputErrorCode.DUPLICATE,
                    "handler",
                    "request already has an active attached handler",
                )
            task = create_task(
                handler(context),
                name=f"interaction-handler-{request_id}",
            )
            self._handler_tasks[request_id] = task
        try:
            outcome = await task
        finally:
            async with self._state_lock:
                if self._handler_tasks.get(request_id) is task:
                    self._handler_tasks.pop(request_id, None)
        if not isinstance(
            outcome,
            (
                InputHandlerResolution,
                InputHandlerDetached,
                InputHandlerDisconnected,
                _TrustedInputHandlerResolution,
            ),
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.outcome",
                "attached handler returned an unsupported outcome",
            )
        return outcome

    async def _apply_handler_cancellation(
        self,
        actor: InteractionActor,
        record: InteractionRecord,
    ) -> InteractionRecord:
        latest = await self._latest_record(actor, record.correlation)
        if _is_terminal_record(latest):
            await self._settle_record(latest)
            return latest
        result = await self._store.cancel(
            CancelInteractionCommand(
                actor=actor,
                correlation=latest.correlation,
                provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                expected_state_revision=latest.request.state_revision,
            )
        )
        await self._settle_store_result(result)
        result_record = _single_result_record(result)
        if result_record is not None:
            return result_record
        return await self._latest_record(actor, latest.correlation)

    async def _apply_handler_loss(
        self,
        actor: InteractionActor,
        record: InteractionRecord,
    ) -> InteractionRecord:
        latest = await self._latest_record(actor, record.correlation)
        if _is_terminal_record(latest):
            await self._settle_record(latest)
            return latest
        self._emit_record(InteractionObserverEventKind.HANDLER_LOST, latest)
        has_resumer = await self._has_resumer(latest.request.request_id)
        disposition = (
            self._policy.attached_loss_with_resumer
            if has_resumer
            else self._policy.attached_loss_without_resumer
        )
        if disposition is HandlerLossDisposition.DETACH:
            result: InteractionBrokerStoreResult = (
                await self._store.mark_detached(
                    DetachInteractionCommand(
                        actor=actor,
                        correlation=latest.correlation,
                        expected_store_revision=latest.store_revision,
                    )
                )
            )
            if isinstance(result, InteractionPresentationApplied):
                self._emit_record(
                    InteractionObserverEventKind.DETACHED,
                    result.record,
                )
        elif disposition is HandlerLossDisposition.CANCEL_REQUEST:
            result = await self._store.cancel(
                CancelInteractionCommand(
                    actor=actor,
                    correlation=latest.correlation,
                    provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                    expected_state_revision=latest.request.state_revision,
                )
            )
        else:
            result = await self._store.terminalize(
                TerminalizeInteractionCommand(
                    actor=actor,
                    correlation=latest.correlation,
                    status=ResolutionStatus.UNAVAILABLE,
                    provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                    expected_state_revision=latest.request.state_revision,
                )
            )
        await self._settle_store_result(result)
        result_record = _single_result_record(result)
        if result_record is not None:
            return result_record
        return await self._latest_record(actor, latest.correlation)

    async def _delivery(
        self,
        correlation: InteractionCorrelation,
        record: InteractionRecord,
        handler_attempts: int,
    ) -> InteractionDelivery:
        async with self._state_lock:
            failed = record.request.request_id in self._resumer_failures
        return InteractionDelivery(
            correlation=correlation,
            record=record,
            handler_attempts=handler_attempts,
            resumer_failed=failed,
        )

    async def _latest_record(
        self,
        actor: InteractionActor,
        correlation: InteractionCorrelation,
    ) -> InteractionRecord:
        projection = await self._store.lookup_scoped(
            ScopedInteractionLookup(
                actor=actor,
                correlation=correlation,
            )
        )
        if not isinstance(projection, InteractionRecord):
            raise InteractionNotFoundError()
        return projection

    async def _start_watcher(
        self,
        actor: InteractionActor,
        record: InteractionRecord,
    ) -> None:
        request_id = record.request.request_id
        async with self._state_lock:
            if self._closed or request_id in self._watcher_tasks:
                return
            task = create_task(
                self._watch_terminal(actor, record),
                name=f"interaction-watcher-{request_id}",
            )
            self._watcher_tasks[request_id] = task

    async def _watch_terminal(
        self,
        actor: InteractionActor,
        record: InteractionRecord,
    ) -> None:
        request_id = record.request.request_id
        revision = record.store_revision
        try:
            while True:
                projection = await self._store.wait_for_change(
                    WaitForInteractionChangeCommand(
                        actor=actor,
                        correlation=record.correlation,
                        after_store_revision=revision,
                    )
                )
                if not isinstance(projection, InteractionRecord):
                    return
                if _is_terminal_record(projection):
                    await self._settle_record(projection)
                    return
                revision = projection.store_revision
        except (CancelledError, InteractionStoreClosedError):
            return
        except Exception:
            return
        finally:
            async with self._state_lock:
                if self._watcher_tasks.get(request_id) is current_task():
                    self._watcher_tasks.pop(request_id, None)

    async def _settle_store_result(
        self,
        result: InteractionBrokerStoreResult,
    ) -> bool:
        failed = False
        for record in _result_records(result):
            if _is_terminal_record(record):
                failed = await self._settle_record(record) or failed
        return failed

    async def _settle_record(self, record: InteractionRecord) -> bool:
        if not _is_terminal_record(record):
            return False
        request_id = record.request.request_id
        notification = self._resumption_notification(record)
        registration = await self._start_resumption(
            request_id,
            notification,
        )
        if registration is not None:
            if registration.task is not None:
                return await shield(registration.task)
        async with self._state_lock:
            return request_id in self._resumer_failures

    @staticmethod
    def _resumption_notification(
        record: InteractionRecord,
    ) -> InputResumptionNotification:
        """Project one terminal record to its continuation notification."""
        outcome = project_resolution_to_model(
            record.request,
            containing_run_exists=True,
        )
        return InputResumptionNotification(
            continuation_id=record.request.continuation_id,
            state_revision=record.request.state_revision,
            outcome=outcome,
        )

    async def _start_resumption(
        self,
        request_id: InputRequestId,
        notification: InputResumptionNotification,
    ) -> _ResumptionRegistration | None:
        """Start or return the sole tracked delivery for one request."""
        async with self._state_lock:
            if self._closed and not self._closing_settlements:
                return None
            resumer_task = self._resumer_tasks.get(request_id)
            if resumer_task is not None:
                handoff = self._resumer_handoffs[request_id]
                return _ResumptionRegistration(
                    task=resumer_task,
                    handoff=handoff,
                )
            if request_id in self._settlement_started:
                completed_handoff = self._resumer_handoffs.get(request_id)
                if completed_handoff is not None:
                    return _ResumptionRegistration(
                        task=None,
                        handoff=completed_handoff,
                    )
                return None
            self._settlement_started.add(request_id)
            continuation_id = self._request_resumers.get(request_id)
            resumer: InputResumer | None = None
            if continuation_id == notification.continuation_id:
                assert continuation_id is not None
                self._request_resumers.pop(request_id, None)
                resumer = self._resumers.pop(continuation_id)
            handoff = get_running_loop().create_future()
            resumer_task = create_task(
                self._deliver_resumption(
                    request_id,
                    notification,
                    resumer,
                    handoff,
                ),
                name=f"interaction-resumer-{request_id}",
            )
            self._resumer_tasks[request_id] = resumer_task
            self._resumer_handoffs[request_id] = handoff
            return _ResumptionRegistration(
                task=resumer_task,
                handoff=handoff,
            )

    async def _deliver_resumption(
        self,
        request_id: InputRequestId,
        notification: InputResumptionNotification,
        resumer: InputResumer | None,
        handoff: Future[None],
    ) -> bool:
        try:
            await self._settle_local_tasks(request_id, notification)
            if not handoff.done():
                handoff.set_result(None)
            if resumer is not None:
                await resumer(notification)
        except CancelledError:
            if self._closed:
                raise
            await self._record_resumer_failure(request_id)
        except Exception:
            await self._record_resumer_failure(request_id)
        finally:
            if not handoff.done():
                handoff.set_result(None)
            async with self._state_lock:
                if self._resumer_tasks.get(request_id) is current_task():
                    self._resumer_tasks.pop(request_id, None)
                if request_id in self._store_handoffs_acknowledged:
                    self._store_handoffs_acknowledged.remove(request_id)
                    if self._resumer_handoffs.get(request_id) is handoff:
                        self._resumer_handoffs.pop(request_id, None)
        async with self._state_lock:
            return request_id in self._resumer_failures

    async def _settle_local_tasks(
        self,
        request_id: InputRequestId,
        notification: InputResumptionNotification,
    ) -> None:
        """Cancel and join local delivery tasks after terminal commit."""
        async with self._state_lock:
            handler_task = self._handler_tasks.pop(request_id, None)
            watcher_task = self._watcher_tasks.pop(request_id, None)
            excluded_task = self._close_excluded_task
            emit_terminal = request_id not in self._terminal_observed
            self._terminal_observed.add(request_id)
        tasks = tuple(
            task
            for task in (handler_task, watcher_task)
            if task is not None
            and task is not current_task()
            and task is not excluded_task
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await gather(*tasks, return_exceptions=True)
        if emit_terminal:
            self._emit(
                InteractionObserverEvent(
                    kind=InteractionObserverEventKind.TERMINAL,
                    request_id=request_id,
                    status=self._notification_status(notification),
                )
            )

    @staticmethod
    def _notification_status(
        notification: InputResumptionNotification,
    ) -> ResolutionStatus:
        """Return content-safe terminal status from a store notification."""
        outcome = notification.outcome
        if isinstance(outcome, TerminateInputContinuation):
            return outcome.status
        assert isinstance(outcome, ResumeInputContinuation)
        return ResolutionStatus(outcome.result.kind.value)

    async def _record_resumer_failure(
        self,
        request_id: InputRequestId,
    ) -> None:
        async with self._state_lock:
            self._resumer_failures.add(request_id)
        self._emit(
            InteractionObserverEvent(
                kind=InteractionObserverEventKind.RESUMER_FAILED,
                request_id=request_id,
            )
        )

    async def _complete_store_resumption_handoff(
        self,
        request_id: InputRequestId,
        handoff: Future[None],
    ) -> None:
        """Release admission authority after exact bridge registration."""
        async with self._state_lock:
            if self._resumer_handoffs.get(request_id) is handoff:
                self._admission_cleanups.pop(request_id, None)
                if request_id in self._resumer_tasks:
                    self._store_handoffs_acknowledged.add(request_id)
                else:
                    self._resumer_handoffs.pop(request_id, None)

    async def _has_resumer(self, request_id: InputRequestId) -> bool:
        async with self._state_lock:
            continuation_id = self._request_resumers.get(request_id)
            return request_id in self._resumer_tasks or (
                continuation_id is not None
                and continuation_id in self._resumers
            )

    async def _cancel_handler(self, request_id: InputRequestId) -> None:
        async with self._state_lock:
            task = self._handler_tasks.pop(request_id, None)
        if task is not None and task is not current_task():
            task.cancel()
            await gather(task, return_exceptions=True)

    async def _await_cancellation_cleanup(
        self,
        request: InteractionBrokerRequest,
        record: InteractionRecord,
    ) -> None:
        async with self._state_lock:
            if self._closed:
                return
            cleanup = create_task(
                self._cleanup_cancelled_request(request, record),
                name=f"interaction-cancel-cleanup-{record.request.request_id}",
            )
            self._cleanup_tasks.add(cleanup)
        try:
            await _await_shielded_cleanup(cleanup)
        finally:
            async with self._state_lock:
                self._cleanup_tasks.discard(cleanup)

    async def _cleanup_cancelled_request(
        self,
        request: InteractionBrokerRequest,
        record: InteractionRecord,
    ) -> None:
        await self._cancel_handler(record.request.request_id)
        if self._closed:
            return
        try:
            await self._apply_handler_loss(request.actor, record)
        except (InteractionNotFoundError, InteractionStoreClosedError):
            return

    async def _deadline_loop(self) -> None:
        snapshot = None
        consecutive_failures = 0
        while True:
            try:
                if snapshot is None:
                    snapshot = await self._store.next_deadline()
                self._emit(
                    InteractionObserverEvent(
                        kind=InteractionObserverEventKind.DEADLINE_REARMED,
                        request_id=(
                            snapshot.deadline.request_id
                            if snapshot.deadline is not None
                            else None
                        ),
                        schedule_revision=snapshot.schedule_revision,
                    )
                )
                if snapshot.deadline is None:
                    snapshot = await self._store.wait_for_deadline_change(
                        WaitForDeadlineChangeCommand(
                            after_schedule_revision=snapshot.schedule_revision,
                        )
                    )
                    consecutive_failures = 0
                    continue
                timer_task = create_task(
                    self._clock.wait_until(
                        snapshot.deadline.monotonic_deadline,
                    ),
                    name="interaction-deadline-timer",
                )
                change_task = create_task(
                    self._store.wait_for_deadline_change(
                        WaitForDeadlineChangeCommand(
                            after_schedule_revision=snapshot.schedule_revision,
                        )
                    ),
                    name="interaction-deadline-change",
                )
                try:
                    done, pending = await wait(
                        (timer_task, change_task),
                        return_when=FIRST_COMPLETED,
                    )
                except BaseException:
                    timer_task.cancel()
                    change_task.cancel()
                    await gather(
                        timer_task,
                        change_task,
                        return_exceptions=True,
                    )
                    raise
                for task in pending:
                    task.cancel()
                if pending:
                    await gather(*pending, return_exceptions=True)
                timer_won = timer_task in done
                if timer_task in done:
                    timer_task.result()
                changed_snapshot = (
                    change_task.result() if change_task in done else None
                )
                if timer_won:
                    result = await self._store.terminalize_due(
                        TerminalizeDueInteractionsCommand()
                    )
                    await self._settle_store_result(result)
                    snapshot = await self._store.next_deadline()
                else:
                    assert changed_snapshot is not None
                    snapshot = changed_snapshot
                consecutive_failures = 0
            except (CancelledError, InteractionStoreClosedError):
                return
            except Exception:
                snapshot = None
                consecutive_failures += 1
                self._emit(
                    InteractionObserverEvent(
                        kind=(
                            InteractionObserverEventKind.DEADLINE_PUMP_FAILED
                        ),
                    )
                )
                if consecutive_failures >= 2:
                    await self.aclose()
                    return

    async def _observer_loop(self) -> None:
        assert self._observer is not None
        assert self._observer_queue is not None
        try:
            while True:
                event = await self._observer_queue.get()
                try:
                    await self._observer(event)
                except Exception:
                    pass
                dropped = self._observer_dropped
                self._observer_dropped = 0
                if dropped:
                    overflow = InteractionObserverEvent(
                        kind=InteractionObserverEventKind.OVERFLOW,
                        dropped_events=dropped,
                    )
                    try:
                        await self._observer(overflow)
                    except Exception:
                        pass
                if self._closed:
                    return
        except CancelledError:
            return

    def _emit_record(
        self,
        kind: InteractionObserverEventKind,
        record: InteractionRecord,
    ) -> None:
        self._emit(
            InteractionObserverEvent(
                kind=kind,
                request_id=record.request.request_id,
            )
        )

    def _emit(self, event: InteractionObserverEvent) -> None:
        queue = self._observer_queue
        if queue is None:
            return
        try:
            queue.put_nowait(event)
        except QueueFull:
            self._observer_dropped += 1


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _StoreBoundResumer:
    """Bridge atomic store delivery into one broker-owned callback task."""

    broker: AsyncInteractionBroker = field(repr=False)
    request_id: InputRequestId

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        """Start tracked delivery without extending the store mutation."""
        registration = await self.broker._start_resumption(
            self.request_id,
            notification,
        )
        if registration is not None:
            await shield(registration.handoff)
            await self.broker._complete_store_resumption_handoff(
                self.request_id,
                registration.handoff,
            )


async def _await_shielded_cleanup(task: Task[None]) -> None:
    """Finish one bounded cleanup despite repeated caller cancellation."""
    while True:
        try:
            await shield(task)
            return
        except CancelledError:
            if task.done():
                await task


def _validate_async_callable(value: object, path: str) -> None:
    if not callable(value):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be an async callable",
        )
    callback = cast(Callable[..., object], value)
    if iscoroutinefunction(callback):
        return
    bound_call = getattr(value, "__call__", None)
    if callable(bound_call) and iscoroutinefunction(
        cast(Callable[..., object], bound_call)
    ):
        return
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        path,
        "value must be an async callable",
    )


def _validate_async_methods(
    value: object,
    methods: tuple[str, ...],
    path: str,
) -> None:
    for method_name in methods:
        method = getattr(value, method_name, None)
        if not callable(method) or not iscoroutinefunction(
            cast(Callable[..., object], method)
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                path,
                f"value must provide async {method_name}",
            )


def _is_terminal_record(record: InteractionRecord) -> bool:
    return (
        record.request.state is not RequestState.PENDING
        and record.request.resolution is not None
    )


def _is_broker_store_result(value: object) -> bool:
    return isinstance(
        value,
        (
            CreateInteractionApplied,
            CreateInteractionRejected,
            InteractionPresentationApplied,
            InteractionPresentationRejected,
            ResolveInteractionApplied,
            ResolveInteractionRejected,
            InteractionStoreReplayed,
            TrustedDefaultResolutionApplied,
            TerminalizeInteractionApplied,
            TerminalizeInteractionRejected,
            CancelInteractionApplied,
            CancelInteractionRejected,
            ScopeCancellationApplied,
            ScopeCancellationReplayed,
            ScopeCancellationRejected,
            ScopeSupersessionApplied,
            ScopeSupersessionReplayed,
            ScopeSupersessionRejected,
            InteractionBranchRegistrationApplied,
            InteractionBranchRegistrationReplayed,
            InteractionBranchRegistrationRejected,
            ControllerActivityApplied,
            ControllerActivityRejected,
            ControllerLeaseExpiredApplied,
            DueInteractionsApplied,
            DueInteractionsRejected,
        ),
    )


def _single_result_record(
    result: InteractionBrokerStoreResult,
) -> InteractionRecord | None:
    if isinstance(
        result,
        (
            CreateInteractionApplied,
            InteractionPresentationApplied,
            ResolveInteractionApplied,
            InteractionStoreReplayed,
            TrustedDefaultResolutionApplied,
            TerminalizeInteractionApplied,
            CancelInteractionApplied,
            ControllerActivityApplied,
            ControllerLeaseExpiredApplied,
        ),
    ):
        return result.record
    return None


def _result_records(
    result: InteractionBrokerStoreResult,
) -> tuple[InteractionRecord, ...]:
    single = _single_result_record(result)
    if single is not None:
        return (single,)
    if isinstance(
        result,
        (
            ScopeCancellationApplied,
            ScopeSupersessionApplied,
            DueInteractionsApplied,
        ),
    ):
        return result.records
    return ()
