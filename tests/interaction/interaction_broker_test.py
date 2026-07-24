"""Exercise async broker delivery, races, callbacks, and cleanup."""

from asyncio import (
    CancelledError,
    Event,
    Future,
    Task,
    all_tasks,
    create_task,
    current_task,
    get_running_loop,
    run,
)
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, fields, replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from avalan.interaction import (
    AcquireControllerActivity,
    ActiveControlLeaseNonce,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    AsyncInteractionBroker,
    BranchId,
    CancelInteractionApplied,
    CancelInteractionCommand,
    CancelInteractionRejected,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    ControllerActivityApplied,
    ControllerId,
    CreateInteractionApplied,
    CreateInteractionRejected,
    DeadlineScheduleRevision,
    DeclinedResolution,
    DetachInteractionCommand,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    HandlerLossDisposition,
    InputAnswer,
    InputErrorCode,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerDisconnected,
    InputHandlerResolution,
    InputQuestion,
    InputRequestId,
    InputResumer,
    InputResumptionNotification,
    InputTransitionError,
    InputValidationError,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionBranchRegistration,
    InteractionBranchRegistrationApplied,
    InteractionBranchRoot,
    InteractionBranchRootLookup,
    InteractionBrokerHeartbeat,
    InteractionBrokerRequest,
    InteractionBrokerResult,
    InteractionClock,
    InteractionCorrelation,
    InteractionDelivery,
    InteractionDisclosure,
    InteractionExecutionScope,
    InteractionIdFactory,
    InteractionNotFoundError,
    InteractionObserverEvent,
    InteractionObserverEventKind,
    InteractionOperation,
    InteractionPolicy,
    InteractionPresentationResult,
    InteractionPresentationState,
    InteractionRecord,
    InteractionRequestResult,
    InteractionStore,
    InteractionStoreClosedError,
    InteractionStoreRevision,
    InteractionTerminalMetadata,
    InteractionTime,
    ListInteractionsCommand,
    ModelCallId,
    PresentInteractionCommand,
    PrincipalScope,
    QuestionId,
    RecordControllerActivityCommand,
    RegisterInteractionBranchCommand,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    RunId,
    ScopeCancellationApplied,
    ScopedInteractionLookup,
    ScopeSupersessionApplied,
    StreamSessionId,
    SupersedeInteractionScopeCommand,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TaskInputClassifier,
    TerminalizeInteractionApplied,
    TerminalizeInteractionCommand,
    TerminalizeInteractionScopeCommand,
    TextAnswer,
    TrustedDefaultResolutionApplied,
    TrustedDefaultResolutionRequest,
    TurnId,
    UserId,
    WaitForInteractionChangeCommand,
)
from avalan.interaction.broker import _StoreBoundResumer
from avalan.interaction.handler import (
    InputDisconnectReason,
    InputHandler,
    InputHandlerOutcome,
)
from avalan.interaction.store import (
    CreateInteractionResult,
    _InteractionAdmissionCleanupCommand,
    _InteractionAdmissionCleanupResult,
    _InteractionAdmissionCreateCommand,
)
from avalan.interaction.stores import MemoryInteractionStoreFactory

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


async def _yield_once() -> None:
    """Yield one explicit scheduler turn without a timer."""
    loop = get_running_loop()
    ready = loop.create_future()
    loop.call_soon(ready.set_result, None)
    await ready


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    turns: int = 100,
) -> None:
    """Wait for one deterministic predicate without sleeping."""
    for _ in range(turns):
        if predicate():
            return
        await _yield_once()
    raise AssertionError("event-loop predicate did not become true")


class _Clock(InteractionClock):
    """Provide coherent time and manually releasable deadline waits."""

    def __init__(self) -> None:
        self.wall_time = _NOW
        self.monotonic_seconds = 0.0
        self.wait_calls: list[float] = []
        self.cancelled_waits = 0
        self._waiters: list[tuple[float, Future[None]]] = []

    async def read(self) -> InteractionTime:
        """Return one trusted observation."""
        return InteractionTime.from_clock(
            wall_time=self.wall_time,
            monotonic_seconds=self.monotonic_seconds,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        """Wait until manual advancement reaches one deadline."""
        self.wait_calls.append(monotonic_deadline)
        if self.monotonic_seconds >= monotonic_deadline:
            return
        future = get_running_loop().create_future()
        entry = (monotonic_deadline, future)
        self._waiters.append(entry)
        try:
            await future
        except CancelledError:
            self.cancelled_waits += 1
            raise
        finally:
            if entry in self._waiters:
                self._waiters.remove(entry)

    def advance(self, seconds: float) -> None:
        """Advance wall and monotonic time and wake reached deadlines."""
        assert seconds >= 0.0
        self.wall_time += timedelta(seconds=seconds)
        self.monotonic_seconds += seconds
        for deadline, future in tuple(self._waiters):
            if self.monotonic_seconds >= deadline and not future.done():
                future.set_result(None)


class _IdFactory(InteractionIdFactory):
    """Mint deterministic broker and store identities."""

    def __init__(self) -> None:
        self.sequence = 0
        self.request_ids: list[InputRequestId] = []
        self.continuation_ids: list[ContinuationId] = []

    def _next(self, kind: str) -> str:
        self.sequence += 1
        return f"broker-{kind}-{self.sequence}"

    async def new_request_id(self) -> InputRequestId:
        """Return one request identifier."""
        value = InputRequestId(self._next("request"))
        self.request_ids.append(value)
        return value

    async def new_continuation_id(self) -> ContinuationId:
        """Return one continuation identifier."""
        value = ContinuationId(self._next("continuation"))
        self.continuation_ids.append(value)
        return value

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        """Return one idempotency key."""
        return ResolutionIdempotencyKey(self._next("key"))

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        """Return one controller lease nonce."""
        return ActiveControlLeaseNonce(self._next("lease"))


class _Classifier(TaskInputClassifier):
    """Allow every normalized value under the configured policy identity."""

    def __init__(self, policy: InteractionPolicy) -> None:
        self.policy = policy
        self.sequence = 0

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Return one exact echoed allow classification."""
        self.sequence += 1
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=self.policy.task_input_classifier_id,
            classification_id=f"broker-classification-{self.sequence}",
            policy_revision=self.policy.task_input_policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


class _Authorizer:
    """Return full exact-echo authorization and record operation calls."""

    def __init__(self) -> None:
        self.operations: list[InteractionOperation] = []
        self.allowed = True
        self.disclosure = InteractionDisclosure.FULL
        self.block_operation: InteractionOperation | None = None
        self.entered = Event()
        self.release = Event()
        self.failure: BaseException | None = None

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        """Authorize one exact operation with full owner disclosure."""
        self.operations.append(operation)
        if operation is self.block_operation:
            self.entered.set()
            await self.release.wait()
        if self.failure is not None:
            failure = self.failure
            self.failure = None
            raise failure
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=self.allowed,
            disclosure=(
                self.disclosure if self.allowed else InteractionDisclosure.NONE
            ),
        )


class _Resumer(InputResumer):
    """Record one notification and optionally fail after observation."""

    def __init__(
        self,
        *,
        fail: bool = False,
        block: bool = False,
        cancel: bool = False,
    ) -> None:
        self.fail = fail
        self.block = block
        self.cancel = cancel
        self.notifications: list[InputResumptionNotification] = []
        self.called = Event()
        self.release = Event()
        self.stopped = Event()

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        """Record one call before an optional deterministic failure."""
        self.notifications.append(notification)
        self.called.set()
        try:
            if self.block:
                await self.release.wait()
            if self.cancel:
                raise CancelledError
            if self.fail:
                raise RuntimeError("private callback failure")
        finally:
            self.stopped.set()


class _ClosingResumer(InputResumer):
    """Close the broker reentrantly from continuation delivery."""

    def __init__(self) -> None:
        self.broker: AsyncInteractionBroker | None = None
        self.called = Event()
        self.completed = Event()

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        """Close the configured broker from its tracked resumer task."""
        assert isinstance(notification, InputResumptionNotification)
        broker = self.broker
        assert broker is not None
        self.called.set()
        await broker.aclose()
        self.completed.set()


def _pause_store_bound_resumer(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Event, Event]:
    """Pause every extracted store bridge before broker handoff."""
    entered = Event()
    release = Event()
    original_call = _StoreBoundResumer.__call__

    async def paused_call(
        bridge: _StoreBoundResumer,
        notification: InputResumptionNotification,
    ) -> None:
        """Pause an extracted bridge immediately before broker handoff."""
        entered.set()
        await release.wait()
        await original_call(bridge, notification)

    monkeypatch.setattr(_StoreBoundResumer, "__call__", paused_call)
    return entered, release


class _BlockingHandler(InputHandler):
    """Block presentation behind explicit events and report cancellation."""

    def __init__(
        self,
        outcome: InputHandlerOutcome | None = None,
    ) -> None:
        self.outcome = outcome or InputHandlerDetached()
        self.started = Event()
        self.release = Event()
        self.stopped = Event()
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Wait until release and then return one fixed outcome."""
        self.contexts.append(context)
        self.started.set()
        try:
            await self.release.wait()
            return self.outcome
        finally:
            self.stopped.set()


class _ClosingOnCancellationHandler(InputHandler):
    """Join broker close reentrantly while unwinding cancellation."""

    def __init__(self) -> None:
        self.broker: AsyncInteractionBroker | None = None
        self.started = Event()
        self.close_entered = Event()
        self.close_completed = Event()
        self.stopped = Event()

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Wait until close cancellation, then join that same close."""
        assert isinstance(context, InputHandlerContext)
        self.started.set()
        try:
            await Event().wait()
            return InputHandlerDetached()
        except CancelledError:
            broker = self.broker
            assert broker is not None
            self.close_entered.set()
            await broker.aclose()
            self.close_completed.set()
            raise
        finally:
            self.stopped.set()


class _InitiatingCloseHandler(InputHandler):
    """Initiate broker close and expose its full completion barrier."""

    def __init__(self) -> None:
        self.broker: AsyncInteractionBroker | None = None
        self.close_entered = Event()
        self.close_completed = Event()
        self.stopped = Event()

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Await the close initiated from this broker-owned handler."""
        assert isinstance(context, InputHandlerContext)
        broker = self.broker
        assert broker is not None
        self.close_entered.set()
        try:
            await broker.aclose()
            self.close_completed.set()
            return InputHandlerDetached()
        finally:
            self.stopped.set()


class _LossHandler(InputHandler):
    """Return or raise one deterministic handler-loss variant."""

    def __init__(self, outcome: InputHandlerOutcome | Exception) -> None:
        self.outcome = outcome

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return or raise the configured variant."""
        assert isinstance(context, InputHandlerContext)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class _CorrectionHandler(InputHandler):
    """Return one invalid answer followed by one valid correction."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Resolve after receiving one typed correction error."""
        self.contexts.append(context)
        if len(self.contexts) == 1:
            answer: InputAnswer = TextAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value="wrong semantic type",
            )
        else:
            answer = ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            )
        return InputHandlerResolution(
            resolution=AnsweredResolution(
                request_id=context.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
                answers=(answer,),
            )
        )


class _ContractCorrectionHandler(InputHandler):
    """Correct a correlation-invalid resolution command on retry."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return a mismatched request identity, then a valid answer."""
        self.contexts.append(context)
        request_id = (
            InputRequestId("wrong-request")
            if len(self.contexts) == 1
            else context.request.request_id
        )
        return InputHandlerResolution(
            resolution=DeclinedResolution(
                request_id=request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
            )
        )


class _InvalidOutcomeHandler(InputHandler):
    """Return one runtime-invalid handler outcome for boundary testing."""

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return an object hidden behind the declared outcome union."""
        assert isinstance(context, InputHandlerContext)
        return cast(InputHandlerOutcome, object())


class _Observer:
    """Record observer events with optional blocking and failure."""

    def __init__(
        self,
        *,
        block_first: bool = False,
        fail: bool = False,
        fail_overflow: bool = False,
    ) -> None:
        self.block_first = block_first
        self.fail = fail
        self.fail_overflow = fail_overflow
        self.events: list[InteractionObserverEvent] = []
        self.entered = Event()
        self.release = Event()
        self.overflow = Event()

    async def __call__(self, event: InteractionObserverEvent) -> None:
        """Record one event, optionally block once, then optionally fail."""
        self.events.append(event)
        if event.kind is InteractionObserverEventKind.OVERFLOW:
            self.overflow.set()
            if self.fail_overflow:
                raise RuntimeError("private overflow failure")
        if self.block_first and len(self.events) == 1:
            self.entered.set()
            await self.release.wait()
        if self.fail:
            raise RuntimeError("private observer failure")


class _ClosingObserver:
    """Close the broker reentrantly from its observer task."""

    def __init__(self) -> None:
        self.broker: AsyncInteractionBroker | None = None
        self.entered = Event()
        self.completed = Event()
        self.task: Task[None] | None = None

    async def __call__(self, event: InteractionObserverEvent) -> None:
        """Close exactly once while recording the invoking task."""
        assert isinstance(event, InteractionObserverEvent)
        if self.entered.is_set():
            return
        broker = self.broker
        assert broker is not None
        self.task = cast(Task[None], current_task())
        self.entered.set()
        await broker.aclose()
        self.completed.set()


@dataclass(slots=True)
class _Harness:
    """Hold one broker and its deterministic concrete dependencies."""

    broker: AsyncInteractionBroker
    clock: _Clock
    ids: _IdFactory
    policy: InteractionPolicy
    authorizer: _Authorizer
    classifier: _Classifier
    factory: MemoryInteractionStoreFactory
    store: InteractionStore


class _StoreProxy:
    """Forward public store operations except explicitly overridden hooks."""

    def __init__(self, store: InteractionStore) -> None:
        self.store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)


class _RejectPresentationStore(_StoreProxy):
    """Inject one stale presentation revision through the real store."""

    async def mark_presented(
        self,
        command: PresentInteractionCommand,
    ) -> InteractionPresentationResult:
        """Submit a deliberately stale presentation command."""
        return await self.store.mark_presented(
            replace(
                command,
                expected_store_revision=InteractionStoreRevision(0),
            )
        )


class _RejectDetachStore(_StoreProxy):
    """Inject one stale detach revision through the real store."""

    async def mark_detached(
        self,
        command: DetachInteractionCommand,
    ) -> InteractionPresentationResult:
        """Submit a deliberately stale detach command."""
        return await self.store.mark_detached(
            replace(
                command,
                expected_store_revision=InteractionStoreRevision(0),
            )
        )


class _RejectCancellationStore(_StoreProxy):
    """Reject one cancellation without mutating the authoritative record."""

    async def cancel(
        self,
        command: CancelInteractionCommand,
    ) -> CancelInteractionRejected:
        """Return one content-safe cancellation rejection."""
        return CancelInteractionRejected(
            command=command,
            error=InputTransitionError(
                code=InputErrorCode.STALE_REVISION,
                path="request.state_revision",
                message="state revision changed",
            ),
        )


class _DeadlineErrorStore(_StoreProxy):
    """Fail the initial deadline snapshot deterministically."""

    async def next_deadline(self) -> Any:
        """Raise one raw store adapter failure."""
        raise RuntimeError("deadline adapter failed")


class _WaitErrorStore(_StoreProxy):
    """Fail record waiting deterministically."""

    async def wait_for_change(
        self,
        command: WaitForInteractionChangeCommand,
    ) -> Any:
        """Raise one raw store adapter failure."""
        assert isinstance(command, WaitForInteractionChangeCommand)
        raise RuntimeError("wait adapter failed")


class _OneShotWaitErrorStore(_StoreProxy):
    """Fail the first record wait, then forward later waits."""

    def __init__(self, store: InteractionStore) -> None:
        super().__init__(store)
        self.failed = Event()

    async def wait_for_change(
        self,
        command: WaitForInteractionChangeCommand,
    ) -> Any:
        """Raise one transient adapter failure."""
        if not self.failed.is_set():
            self.failed.set()
            raise RuntimeError("transient wait adapter failure")
        return await self.store.wait_for_change(command)


class _RecordingWaitStore(_StoreProxy):
    """Record one wait projection before forwarding it to the broker."""

    def __init__(self, store: InteractionStore) -> None:
        super().__init__(store)
        self.projections: list[Any] = []
        self.returned = Event()

    async def wait_for_change(
        self,
        command: WaitForInteractionChangeCommand,
    ) -> Any:
        """Record the authoritative projected wait result."""
        projection = await self.store.wait_for_change(command)
        self.projections.append(projection)
        self.returned.set()
        return projection


class _ProjectedWaitStore(_StoreProxy):
    """Return one injected full wait projection."""

    def __init__(self, store: InteractionStore, projection: Any) -> None:
        super().__init__(store)
        self.projection = projection

    async def wait_for_change(
        self,
        command: WaitForInteractionChangeCommand,
    ) -> Any:
        """Return the configured wait projection immediately."""
        assert isinstance(command, WaitForInteractionChangeCommand)
        return self.projection


class _OneShotDeadlineErrorStore(_StoreProxy):
    """Fail the first deadline snapshot, then recover."""

    def __init__(self, store: InteractionStore) -> None:
        super().__init__(store)
        self.failed = Event()

    async def next_deadline(self) -> Any:
        """Raise one transient deadline adapter failure."""
        if not self.failed.is_set():
            self.failed.set()
            raise RuntimeError("transient deadline adapter failure")
        return await self.store.next_deadline()


class _BlockingCloseStore(_StoreProxy):
    """Block store close behind an explicit deterministic barrier."""

    def __init__(self, store: InteractionStore) -> None:
        super().__init__(store)
        self.calls = 0
        self.entered = Event()
        self.release = Event()

    async def aclose(self) -> None:
        """Wait before closing the forwarded store handle."""
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        await self.store.aclose()


class _CommitThenBlockCreateStore(_StoreProxy):
    """Commit a sealed admission, then block before returning its result."""

    def __init__(self, store: InteractionStore) -> None:
        super().__init__(store)
        self.entered = Event()
        self.release = Event()
        self.stopped = Event()

    async def create_admission(
        self,
        command: _InteractionAdmissionCreateCommand,
    ) -> CreateInteractionResult:
        """Expose the post-commit, pre-return cancellation window."""
        result = await self.store.create_admission(command)
        self.entered.set()
        try:
            await self.release.wait()
            return result
        finally:
            self.stopped.set()


class _CloseBeforeCommitAdmissionStore(_StoreProxy):
    """Initiate broker close from create before committing admission state."""

    def __init__(
        self,
        store: InteractionStore,
        *,
        fail_first_close: bool,
    ) -> None:
        super().__init__(store)
        self.fail_first_close = fail_first_close
        self.broker: AsyncInteractionBroker | None = None
        self.close_calls = 0
        self.close_failed = Event()
        self.create_cancelled = Event()
        self.commit_release = Event()
        self.commit_attempted = Event()
        self.committed = Event()

    async def create_admission(
        self,
        command: _InteractionAdmissionCreateCommand,
    ) -> CreateInteractionResult:
        """Close first, then commit after failure or test cancellation."""
        broker = self.broker
        assert broker is not None
        try:
            await broker.aclose()
        except RuntimeError:
            self.close_failed.set()
            try:
                await self.commit_release.wait()
            except CancelledError:
                self.create_cancelled.set()
        self.commit_attempted.set()
        result = await self.store.create_admission(command)
        self.committed.set()
        return result

    async def aclose(self) -> None:
        """Fail the first late close or close the real handle."""
        self.close_calls += 1
        if self.fail_first_close and self.close_calls == 1:
            raise RuntimeError("late store close failure")
        await self.store.aclose()


class _BlockingAdmissionCleanupStore(_StoreProxy):
    """Block capability cleanup before forwarding to the real store."""

    def __init__(self, store: InteractionStore) -> None:
        super().__init__(store)
        self.calls = 0
        self.entered = Event()
        self.release = Event()

    async def cleanup_admission(
        self,
        command: _InteractionAdmissionCleanupCommand,
    ) -> _InteractionAdmissionCleanupResult:
        """Wait until an external terminal contender has run."""
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return await self.store.cleanup_admission(command)


class _TransientAdmissionCleanupStore(_StoreProxy):
    """Fail the first capability cleanup and forward its explicit retry."""

    def __init__(self, store: InteractionStore) -> None:
        super().__init__(store)
        self.calls = 0
        self.retry_entered = Event()
        self.retry_release = Event()
        self.retry_release.set()

    async def cleanup_admission(
        self,
        command: _InteractionAdmissionCleanupCommand,
    ) -> _InteractionAdmissionCleanupResult:
        """Raise once without consuming the retained cleanup authority."""
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient admission cleanup failure")
        self.retry_entered.set()
        await self.retry_release.wait()
        return await self.store.cleanup_admission(command)


class _MalformedAdmissionCleanupStore(_StoreProxy):
    """Return one unsealed cleanup response before a valid retry."""

    def __init__(self, store: InteractionStore) -> None:
        super().__init__(store)
        self.calls = 0

    async def cleanup_admission(
        self,
        command: _InteractionAdmissionCleanupCommand,
    ) -> _InteractionAdmissionCleanupResult:
        """Return one malformed response, then forward the retained command."""
        self.calls += 1
        if self.calls == 1:
            return object.__new__(_InteractionAdmissionCleanupResult)
        return await self.store.cleanup_admission(command)


class _FailingAdmissionCreateStore(_StoreProxy):
    """Fail create and its first best-effort cleanup deterministically."""

    def __init__(
        self,
        store: InteractionStore,
        *,
        malformed_cleanup: bool = False,
    ) -> None:
        super().__init__(store)
        self.cleanup_calls = 0
        self.malformed_cleanup = malformed_cleanup

    async def create_admission(
        self,
        command: _InteractionAdmissionCreateCommand,
    ) -> CreateInteractionResult:
        """Raise without committing the sealed admission."""
        assert isinstance(command, _InteractionAdmissionCreateCommand)
        raise RuntimeError("sealed admission create failure")

    async def cleanup_admission(
        self,
        command: _InteractionAdmissionCleanupCommand,
    ) -> _InteractionAdmissionCleanupResult:
        """Return one inconclusive outcome, then prove definitive absence."""
        self.cleanup_calls += 1
        if self.cleanup_calls == 1:
            if self.malformed_cleanup:
                return object.__new__(_InteractionAdmissionCleanupResult)
            raise InteractionStoreClosedError()
        return await self.store.cleanup_admission(command)


class _BlockingLookupStore(_StoreProxy):
    """Block scoped lookup and expose its cancellation."""

    def __init__(self, store: InteractionStore) -> None:
        super().__init__(store)
        self.entered = Event()
        self.stopped = Event()

    async def lookup_scoped(self, query: ScopedInteractionLookup) -> Any:
        """Block until cancellation before forwarding lookup."""
        assert isinstance(query, ScopedInteractionLookup)
        self.entered.set()
        try:
            await Event().wait()
        finally:
            self.stopped.set()
        return await self.store.lookup_scoped(query)


class _BranchRootResultStore(_StoreProxy):
    """Return one injected branch-root lookup result."""

    def __init__(self, store: InteractionStore, result: Any) -> None:
        super().__init__(store)
        self.result = result

    async def lookup_branch_root(
        self,
        query: InteractionBranchRootLookup,
    ) -> InteractionBranchRoot | None:
        """Return the configured content-free root result."""
        assert isinstance(query, InteractionBranchRootLookup)
        return cast(InteractionBranchRoot | None, self.result)


class _TerminalOnValidationStore(_StoreProxy):
    """Commit external cancellation before returning validation rejection."""

    async def resolve(
        self,
        command: ResolveInteractionCommand,
    ) -> Any:
        """Return rejection only after an external terminal commit."""
        result = await self.store.resolve(command)
        if isinstance(result, ResolveInteractionRejected):
            await self.store.cancel(
                CancelInteractionCommand(
                    actor=command.actor,
                    correlation=command.correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=command.expected_state_revision,
                )
            )
        return result


async def _harness(
    *,
    policy: InteractionPolicy | None = None,
    authorizer: _Authorizer | None = None,
    observer: _Observer | None = None,
    observer_queue_capacity: int = 64,
) -> _Harness:
    """Open one broker over a real authoritative memory store."""
    active_policy = policy or InteractionPolicy()
    clock = _Clock()
    ids = _IdFactory()
    active_authorizer = authorizer or _Authorizer()
    classifier = _Classifier(active_policy)
    factory = MemoryInteractionStoreFactory(
        policy=active_policy,
        clock=clock,
        authorizer=active_authorizer,
        id_factory=ids,
        classifier=classifier,
    )
    store = await factory.open()
    broker = AsyncInteractionBroker(
        store=store,
        clock=clock,
        id_factory=ids,
        policy=active_policy,
        classifier=classifier,
        observer=observer,
        observer_queue_capacity=observer_queue_capacity,
    )
    return _Harness(
        broker=broker,
        clock=clock,
        ids=ids,
        policy=active_policy,
        authorizer=active_authorizer,
        classifier=classifier,
        factory=factory,
        store=store,
    )


def _principal(name: str = "owner") -> PrincipalScope:
    return PrincipalScope(user_id=UserId(name))


def _actor(name: str = "owner") -> InteractionActor:
    return InteractionActor(principal=_principal(name))


def _origin(
    *,
    run_id: str = "run",
    branch_id: str = "root",
    parent_branch_id: str | None = None,
    model_call_id: str | None = None,
) -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId(run_id),
        turn_id=TurnId("turn"),
        agent_id=AgentId("agent"),
        branch_id=BranchId(branch_id),
        parent_branch_id=(
            None if parent_branch_id is None else BranchId(parent_branch_id)
        ),
        model_call_id=ModelCallId(
            model_call_id or f"call-{run_id}-{branch_id}"
        ),
        stream_session_id=StreamSessionId("stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://broker-test",
            agent_definition_revision="revision-1",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model-1",
            tool_revision="tools-1",
            capability_revision="capabilities-1",
        ),
        principal=_principal(),
    )


def _request(
    handler: InputHandler | None,
    *,
    origin: ExecutionOrigin | None = None,
    resumer: InputResumer | None = None,
    mode: RequirementMode = RequirementMode.REQUIRED,
    advisory_wait_seconds: int | None = None,
    reason: str = "Confirm broker behavior.",
    questions: tuple[InputQuestion, ...] | None = None,
) -> InteractionBrokerRequest:
    return InteractionBrokerRequest(
        actor=_actor(),
        origin=origin or _origin(),
        mode=mode,
        reason=reason,
        questions=questions
        or (
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        ),
        handler=handler,
        resumer=resumer,
        continuation_ttl_seconds=600,
        advisory_wait_seconds=advisory_wait_seconds,
    )


def _correlation(
    request_id: InputRequestId,
    continuation_id: ContinuationId,
    origin: ExecutionOrigin,
) -> InteractionCorrelation:
    return InteractionCorrelation(
        request_id=request_id,
        continuation_id=continuation_id,
        run_id=origin.run_id,
        turn_id=origin.turn_id,
        task_id=origin.task_id,
        agent_id=origin.agent_id,
        branch_id=origin.branch_id,
        model_call_id=origin.model_call_id,
    )


async def _inspect(
    broker: AsyncInteractionBroker,
    correlation: InteractionCorrelation,
) -> InteractionRecord:
    projection = await broker.inspect(
        ScopedInteractionLookup(actor=_actor(), correlation=correlation)
    )
    assert isinstance(projection, InteractionRecord)
    return projection


def _decline(record: InteractionRecord, key: str) -> ResolveInteractionCommand:
    return ResolveInteractionCommand(
        actor=_actor(),
        correlation=record.correlation,
        expected_state_revision=record.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=DeclinedResolution(
            request_id=record.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        ),
    )


def test_observer_and_request_dtos_are_strict_and_immutable() -> None:
    """Reject malformed metadata and broker request construction."""

    def invalid(factory: Callable[[], object]) -> None:
        with pytest.raises(InputValidationError):
            factory()

    invalid(
        lambda: InteractionObserverEvent(
            kind=cast(Any, "created"),
        )
    )
    invalid(
        lambda: InteractionObserverEvent(
            kind=InteractionObserverEventKind.TERMINAL,
            request_id=InputRequestId("request"),
            status=cast(Any, "answered"),
        )
    )
    invalid(
        lambda: InteractionObserverEvent(
            kind=InteractionObserverEventKind.OVERFLOW,
        )
    )
    invalid(
        lambda: InteractionObserverEvent(
            kind=InteractionObserverEventKind.HEARTBEAT,
            dropped_events=1,
        )
    )
    invalid(
        lambda: InteractionObserverEvent(
            kind=InteractionObserverEventKind.TERMINAL,
        )
    )
    invalid(
        lambda: InteractionObserverEvent(
            kind=InteractionObserverEventKind.CREATED,
            status=ResolutionStatus.ANSWERED,
        )
    )
    invalid(
        lambda: InteractionObserverEvent(
            kind=InteractionObserverEventKind.DEADLINE_REARMED,
        )
    )
    invalid(
        lambda: InteractionObserverEvent(
            kind=InteractionObserverEventKind.CREATED,
            schedule_revision=DeadlineScheduleRevision(1),
        )
    )
    invalid(
        lambda: InteractionObserverEvent(
            kind=InteractionObserverEventKind.CREATED,
        )
    )
    invalid(
        lambda: InteractionObserverEvent(
            kind=InteractionObserverEventKind.HEARTBEAT,
            request_id=InputRequestId("request"),
        )
    )
    invalid(
        lambda: InteractionObserverEvent(
            kind=InteractionObserverEventKind.DEADLINE_PUMP_FAILED,
            request_id=InputRequestId("request"),
        )
    )

    base = _request(_LossHandler(InputHandlerDetached()))
    invalid(lambda: replace(base, actor=cast(Any, object())))
    invalid(lambda: replace(base, origin=cast(Any, object())))
    invalid(
        lambda: replace(
            base,
            origin=replace(base.origin, principal=_principal("other")),
        )
    )
    invalid(lambda: replace(base, mode=cast(Any, "required")))
    invalid(lambda: replace(base, questions=cast(Any, [])))
    invalid(lambda: replace(base, questions=()))
    invalid(
        lambda: replace(
            base,
            questions=(cast(Any, object()),),
        )
    )
    duplicate = ConfirmationQuestion(
        question_id=QuestionId("confirm"),
        prompt="Duplicate?",
        required=True,
    )
    invalid(lambda: replace(base, questions=(base.questions[0], duplicate)))
    invalid(lambda: replace(base, advisory_wait_seconds=1))
    invalid(lambda: replace(base, handler=cast(Any, object())))
    invalid(lambda: replace(base, handler=cast(Any, lambda _: None)))
    invalid(lambda: replace(base, resumer=cast(Any, lambda _: None)))

    async def direct_handler(_: InputHandlerContext) -> InputHandlerOutcome:
        return InputHandlerDetached()

    direct = replace(base, handler=cast(InputHandler, direct_handler))
    assert direct.handler is direct_handler
    with pytest.raises(FrozenInstanceError):
        setattr(direct, "reason", "mutated")


def test_delivery_and_result_dtos_reject_cross_record_confusion() -> None:
    """Keep correlations, admissions, deliveries, and mutations exact."""

    async def exercise() -> None:
        first_harness = await _harness()
        denied_authorizer = _Authorizer()
        denied_authorizer.allowed = False
        denied_harness = await _harness(authorizer=denied_authorizer)
        try:
            first = await first_harness.broker.request(
                _request(_CorrectionHandler())
            )
            second = await first_harness.broker.request(
                _request(
                    _LossHandler(InputHandlerDetached()),
                    origin=_origin(branch_id="dto-second"),
                    reason="Distinct DTO record.",
                )
            )
            rejected = await denied_harness.broker.request(
                _request(_LossHandler(InputHandlerDetached()))
            )
            assert first.delivery is not None
            assert second.delivery is not None
            assert isinstance(first.create_result, CreateInteractionApplied)
            assert isinstance(
                rejected.create_result,
                CreateInteractionRejected,
            )

            with pytest.raises(InputValidationError):
                InteractionDelivery(
                    correlation=cast(Any, object()),
                    record=first.delivery.record,
                    handler_attempts=0,
                )
            with pytest.raises(InputValidationError):
                InteractionDelivery(
                    correlation=first.delivery.correlation,
                    record=cast(Any, object()),
                    handler_attempts=0,
                )
            with pytest.raises(InputValidationError):
                InteractionDelivery(
                    correlation=first.delivery.correlation,
                    record=second.delivery.record,
                    handler_attempts=0,
                )
            with pytest.raises(InputValidationError):
                InteractionBrokerResult(store_result=cast(Any, object()))
            with pytest.raises(InputValidationError):
                InteractionRequestResult(
                    create_result=cast(Any, object()),
                    delivery=None,
                )
            with pytest.raises(InputValidationError):
                InteractionRequestResult(
                    create_result=rejected.create_result,
                    delivery=first.delivery,
                )
            with pytest.raises(InputValidationError):
                InteractionRequestResult(
                    create_result=first.create_result,
                    delivery=None,
                )
            with pytest.raises(InputValidationError):
                InteractionRequestResult(
                    create_result=first.create_result,
                    delivery=second.delivery,
                )
        finally:
            await first_harness.broker.aclose()
            await denied_harness.broker.aclose()

    run(exercise())


def test_constructor_start_and_resumer_guards_are_deterministic() -> None:
    """Validate dependencies, one-time startup, and callback identity."""

    async def exercise() -> None:
        harness = await _harness()
        second = await _harness()
        try:
            with pytest.raises(InputValidationError):
                AsyncInteractionBroker(
                    store=cast(Any, object()),
                    clock=harness.clock,
                    id_factory=harness.ids,
                    policy=harness.policy,
                    classifier=harness.classifier,
                )
            with pytest.raises(InputValidationError):
                AsyncInteractionBroker(
                    store=harness.store,
                    clock=harness.clock,
                    id_factory=harness.ids,
                    policy=cast(Any, object()),
                    classifier=harness.classifier,
                )
            with pytest.raises(InputValidationError):
                await harness.broker.request(cast(Any, object()))

            await harness.broker._start_lock.acquire()
            first_start = create_task(harness.broker.heartbeat())
            second_start = create_task(harness.broker.heartbeat())
            await _yield_once()
            harness.broker._start_lock.release()
            first_heartbeat = await first_start
            second_heartbeat = await second_start
            assert {
                first_heartbeat.sequence,
                second_heartbeat.sequence,
            } == {1, 2}

            request_id = InputRequestId("bound-request")
            continuation_id = ContinuationId("bound-continuation")
            resumer = _Resumer()
            await second.broker._bind_resumer(
                request_id,
                continuation_id,
                resumer,
            )
            with pytest.raises(InputValidationError):
                await second.broker._bind_resumer(
                    InputRequestId("other-request"),
                    continuation_id,
                    resumer,
                )
            await second.broker._unbind_resumer(
                request_id,
                continuation_id,
            )
            assert second.broker._request_resumers == {}
            assert second.broker._resumers == {}

            stale_origin = _origin(branch_id="stale-slot")
            stale_ticket = await second.broker._acquire_presentation_slot(
                stale_origin,
                stale_origin.branch_id,
            )
            stale_queue = second.broker._root_queues[stale_ticket.key]
            stale_queue.clear()
            await second.broker._release_presentation_slot(stale_ticket)
            second.broker._root_queues.pop(stale_ticket.key)
        finally:
            await harness.broker.aclose()
            await second.broker.aclose()

    run(exercise())


@pytest.mark.parametrize("allowed", (True, False), ids=("applied", "rejected"))
def test_owning_cancellation_during_create_settles_admission(
    allowed: bool,
) -> None:
    """Shield admission, then clean applied or rejected cancellation."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        authorizer.allowed = allowed
        authorizer.block_operation = InteractionOperation.CREATE
        harness = await _harness(authorizer=authorizer)
        resumer = _Resumer()
        origin = _origin()
        task = create_task(
            harness.broker.request(
                _request(
                    _LossHandler(InputHandlerDetached()),
                    resumer=resumer,
                    origin=origin,
                )
            )
        )
        await authorizer.entered.wait()
        task.cancel()
        authorizer.release.set()
        with pytest.raises(CancelledError):
            await task
        assert harness.broker._create_tasks == {}
        if allowed:
            assert len(harness.broker._resumers) == 1
            correlation = _correlation(
                harness.ids.request_ids[0],
                harness.ids.continuation_ids[0],
                origin,
            )
            record = await _inspect(harness.broker, correlation)
            assert record.request.state is RequestState.PENDING
            assert record.presentation is InteractionPresentationState.DETACHED
            await harness.broker.cancel(
                CancelInteractionCommand(
                    actor=_actor(),
                    correlation=correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=record.request.state_revision,
                )
            )
            await resumer.called.wait()
        else:
            assert harness.broker._resumers == {}
        await harness.broker.aclose()

    run(exercise())


def test_create_cancellation_and_failure_unbind_resumer() -> None:
    """Unbind callbacks when close cancels create or admission raises."""

    async def exercise() -> None:
        blocking = _Authorizer()
        blocking.block_operation = InteractionOperation.CREATE
        closing = await _harness(authorizer=blocking)
        closing_task = create_task(
            closing.broker.request(
                _request(
                    _LossHandler(InputHandlerDetached()),
                    resumer=_Resumer(),
                )
            )
        )
        await blocking.entered.wait()
        await closing.broker.aclose()
        with pytest.raises(CancelledError):
            await closing_task
        assert closing.broker._create_tasks == {}
        assert closing.broker._resumers == {}

        failing = _Authorizer()
        failing.failure = RuntimeError("admission failed")
        harness = await _harness(authorizer=failing)
        try:
            with pytest.raises(RuntimeError, match="admission failed"):
                await harness.broker.request(
                    _request(
                        _LossHandler(InputHandlerDetached()),
                        resumer=_Resumer(),
                    )
                )
            assert harness.broker._create_tasks == {}
            assert harness.broker._resumers == {}
        finally:
            await harness.broker.aclose()

    run(exercise())


def test_repeated_owning_cancellation_awaits_one_cleanup() -> None:
    """Finish one shielded loss cleanup after repeated owner cancellation."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        harness = await _harness(authorizer=authorizer)
        handler = _BlockingHandler()
        task = create_task(
            harness.broker.request(_request(handler, resumer=_Resumer()))
        )
        await handler.started.wait()
        authorizer.entered.clear()
        authorizer.release.clear()
        authorizer.block_operation = InteractionOperation.DELIVER
        task.cancel()
        await authorizer.entered.wait()
        task.cancel()
        authorizer.release.set()
        with pytest.raises(CancelledError):
            await task
        correlation = _correlation(
            harness.ids.request_ids[0],
            harness.ids.continuation_ids[0],
            _origin(),
        )
        record = await _inspect(harness.broker, correlation)
        assert record.presentation is InteractionPresentationState.DETACHED
        await harness.broker.aclose()

    run(exercise())


def test_broker_corrects_handler_input_and_exposes_strict_results() -> None:
    """Keep one request and root slot through deterministic correction."""

    async def exercise() -> None:
        harness = await _harness()
        handler = _CorrectionHandler()
        try:
            result = await harness.broker.request(_request(handler))
            assert isinstance(result, InteractionRequestResult)
            assert result.delivery is not None
            assert isinstance(result.delivery, InteractionDelivery)
            assert result.delivery.handler_attempts == 2
            assert (
                result.delivery.record.request.state is RequestState.ANSWERED
            )
            assert len(handler.contexts) == 2
            assert handler.contexts[0].validation_error is None
            correction = handler.contexts[1].validation_error
            assert correction is not None
            assert correction.code is InputErrorCode.ANSWER_TYPE_MISMATCH
            projection = await harness.broker.inspect(
                ScopedInteractionLookup(
                    actor=_actor(),
                    correlation=result.delivery.correlation,
                )
            )
            assert projection == result.delivery.record
            listed = await harness.broker.list(
                ListInteractionsCommand(
                    actor=_actor(),
                    scope=InteractionExecutionScope(run_id=RunId("run")),
                )
            )
            assert listed == (result.delivery.record,)
            heartbeat = await harness.broker.heartbeat()
            assert heartbeat == InteractionBrokerHeartbeat(sequence=1)
        finally:
            await harness.broker.aclose()

    run(exercise())


def test_contract_correction_and_handler_runtime_guards() -> None:
    """Correct command construction and reject impossible handler outcomes."""

    async def exercise() -> None:
        correction_harness = await _harness()
        invalid_harness = await _harness()
        active_harness = await _harness()
        terminal_race = await _harness()
        try:
            correction = _ContractCorrectionHandler()
            corrected = await correction_harness.broker.request(
                _request(correction)
            )
            assert corrected.delivery is not None
            assert corrected.delivery.handler_attempts == 2
            assert (
                corrected.delivery.record.request.state
                is RequestState.DECLINED
            )
            assert len(correction.contexts) == 2
            assert correction.contexts[1].validation_error is not None
            assert (
                correction.contexts[1].validation_error.code
                is InputErrorCode.CORRELATION_MISMATCH
            )

            terminal_race.broker._store = cast(
                InteractionStore,
                _TerminalOnValidationStore(terminal_race.store),
            )
            terminal_handler = _CorrectionHandler()
            raced = await terminal_race.broker.request(
                _request(terminal_handler)
            )
            assert raced.delivery is not None
            assert raced.delivery.handler_attempts == 1
            assert (
                raced.delivery.record.request.state is RequestState.CANCELLED
            )
            assert len(terminal_handler.contexts) == 1

            invalid = await invalid_harness.broker.request(
                _request(_InvalidOutcomeHandler())
            )
            assert invalid.delivery is not None
            assert (
                invalid.delivery.record.request.state
                is RequestState.UNAVAILABLE
            )

            blocking = _BlockingHandler()
            active_task = create_task(
                active_harness.broker.request(_request(blocking))
            )
            await blocking.started.wait()
            correlation = _correlation(
                active_harness.ids.request_ids[0],
                active_harness.ids.continuation_ids[0],
                _origin(),
            )
            active = await _inspect(active_harness.broker, correlation)
            context = InputHandlerContext(
                request=active.request,
                validation_error=None,
            )
            with pytest.raises(InputValidationError):
                await active_harness.broker._invoke_handler(
                    active.request.request_id,
                    _LossHandler(InputHandlerDetached()),
                    context,
                )

            original = active_harness.broker._invoke_handler

            async def impossible_outcome(
                request_id: InputRequestId,
                handler: InputHandler,
                handler_context: InputHandlerContext,
            ) -> InputHandlerOutcome:
                assert request_id == active.request.request_id
                assert handler is blocking
                assert handler_context == context
                return cast(InputHandlerOutcome, object())

            setattr(
                active_harness.broker,
                "_invoke_handler",
                impossible_outcome,
            )
            with pytest.raises(InputValidationError):
                await active_harness.broker._drive_handler(
                    _actor(),
                    correlation,
                    active,
                    blocking,
                )
            setattr(active_harness.broker, "_invoke_handler", original)
            await active_harness.broker.cancel(
                CancelInteractionCommand(
                    actor=_actor(),
                    correlation=correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=active.request.state_revision,
                )
            )
            await active_task
        finally:
            await correction_harness.broker.aclose()
            await invalid_harness.broker.aclose()
            await active_harness.broker.aclose()
            await terminal_race.broker.aclose()

    run(exercise())


def test_stale_presentation_and_loss_races_reread_authority() -> None:
    """Re-read records after rejected presentation and detach mutations."""

    async def exercise() -> None:
        presentation = await _harness()
        presentation.broker._store = cast(
            InteractionStore,
            _RejectPresentationStore(presentation.store),
        )
        handler = _BlockingHandler()
        result = await presentation.broker.request(_request(handler))
        assert result.delivery is not None
        assert result.delivery.handler_attempts == 0
        assert (
            result.delivery.record.presentation
            is InteractionPresentationState.QUEUED
        )
        assert not handler.started.is_set()
        await presentation.broker.aclose()

        detached = await _harness()
        detached_request = _request(
            _LossHandler(InputHandlerDetached()),
            resumer=_Resumer(),
        )
        detached_result = await detached.broker.request(detached_request)
        assert detached_result.delivery is not None
        detached_record = detached_result.delivery.record
        repeated = await detached.broker._deliver_request(
            detached_request,
            detached_record,
        )
        assert repeated.record == detached_record
        assert await detached.broker._settle_record(detached_record) is False
        await detached.broker._start_watcher(_actor(), detached_record)
        missing = replace(
            detached_record.correlation,
            request_id=InputRequestId("missing-request"),
        )
        with pytest.raises(InteractionNotFoundError):
            await detached.broker._latest_record(_actor(), missing)
        cancelled = await detached.broker.cancel(
            CancelInteractionCommand(
                actor=_actor(),
                correlation=detached_record.correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=detached_record.request.state_revision,
            )
        )
        terminal_record = cast(
            CancelInteractionApplied,
            cancelled.store_result,
        ).record
        assert (
            await detached.broker._apply_handler_loss(
                _actor(),
                terminal_record,
            )
            == terminal_record
        )
        driven, attempts = await detached.broker._drive_handler(
            _actor(),
            terminal_record.correlation,
            terminal_record,
            _LossHandler(InputHandlerDetached()),
        )
        assert driven == terminal_record
        assert attempts == 0
        await detached.broker.aclose()

        loss = await _harness()
        loss.broker._store = cast(
            InteractionStore,
            _RejectDetachStore(loss.store),
        )
        loss_result = await loss.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=_Resumer(),
            )
        )
        assert loss_result.delivery is not None
        assert (
            loss_result.delivery.record.presentation
            is InteractionPresentationState.PRESENTED
        )
        await loss.broker.cancel(
            CancelInteractionCommand(
                actor=_actor(),
                correlation=loss_result.delivery.correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=(
                    loss_result.delivery.record.request.state_revision
                ),
            )
        )
        await loss.broker.aclose()

    run(exercise())


@pytest.mark.parametrize(
    "loss",
    (
        InputHandlerDetached(),
        InputHandlerDisconnected(
            reason=InputDisconnectReason.CONTROL_CHANNEL_CLOSED
        ),
        RuntimeError("renderer failed"),
    ),
    ids=("detached", "disconnected", "exception"),
)
@pytest.mark.parametrize("with_resumer", (False, True))
def test_handler_loss_matrix_never_fabricates_decline(
    loss: InputHandlerOutcome | Exception,
    with_resumer: bool,
) -> None:
    """Apply declared detach or unavailable policy for every loss path."""

    async def exercise() -> None:
        harness = await _harness()
        resumer = _Resumer() if with_resumer else None
        try:
            result = await harness.broker.request(
                _request(_LossHandler(loss), resumer=resumer)
            )
            assert result.delivery is not None
            record = result.delivery.record
            if with_resumer:
                assert record.request.state is RequestState.PENDING
                assert (
                    record.presentation
                    is InteractionPresentationState.DETACHED
                )
                assert resumer is not None
                assert resumer.notifications == []
            else:
                assert record.request.state is RequestState.UNAVAILABLE
                assert record.request.resolution is not None
                assert (
                    record.request.resolution.status
                    is ResolutionStatus.UNAVAILABLE
                )
        finally:
            await harness.broker.aclose()

    run(exercise())


def test_handler_cancel_is_canonical_cancel_not_handler_loss() -> None:
    """Persist explicit current-input cancellation as a cancelled request."""

    async def exercise() -> None:
        observer = _Observer()
        harness = await _harness(observer=observer)
        try:
            result = await harness.broker.request(
                _request(
                    _LossHandler(
                        InputHandlerDisconnected(
                            reason=InputDisconnectReason.HANDLER_CANCELLED
                        )
                    )
                )
            )
            assert result.delivery is not None
            record = result.delivery.record
            assert record.request.state is RequestState.CANCELLED
            assert record.request.resolution is not None
            assert (
                record.request.resolution.status is ResolutionStatus.CANCELLED
            )
            await _wait_until(
                lambda: any(
                    event.kind is InteractionObserverEventKind.TERMINAL
                    for event in observer.events
                )
            )
            assert not any(
                event.kind is InteractionObserverEventKind.HANDLER_LOST
                for event in observer.events
            )
            assert any(
                event.kind is InteractionObserverEventKind.TERMINAL
                and event.status is ResolutionStatus.CANCELLED
                for event in observer.events
            )
        finally:
            await harness.broker.aclose()

    run(exercise())


def test_handler_cancel_reconciles_terminal_and_rejected_store_races() -> None:
    """Return the authoritative record for both cancellation race outcomes."""

    async def detached_record(harness: _Harness) -> InteractionRecord:
        result = await harness.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=_Resumer(),
            )
        )
        assert result.delivery is not None
        return result.delivery.record

    async def exercise() -> None:
        terminal_harness = await _harness()
        rejected_harness = await _harness()
        try:
            stale_terminal = await detached_record(terminal_harness)
            applied = await terminal_harness.store.cancel(
                CancelInteractionCommand(
                    actor=_actor(),
                    correlation=stale_terminal.correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=(
                        stale_terminal.request.state_revision
                    ),
                )
            )
            assert isinstance(applied, CancelInteractionApplied)
            terminal = (
                await terminal_harness.broker._apply_handler_cancellation(
                    _actor(),
                    stale_terminal,
                )
            )
            assert terminal.request.state is RequestState.CANCELLED

            pending = await detached_record(rejected_harness)
            rejected_harness.broker._store = cast(
                InteractionStore,
                _RejectCancellationStore(rejected_harness.store),
            )
            unchanged = (
                await rejected_harness.broker._apply_handler_cancellation(
                    _actor(),
                    pending,
                )
            )
            assert unchanged.request.state is RequestState.PENDING
            assert unchanged.store_revision == pending.store_revision
        finally:
            await terminal_harness.broker.aclose()
            await rejected_harness.broker.aclose()

    run(exercise())


def test_public_default_terminal_scope_and_activity_wrappers() -> None:
    """Delegate every public lifecycle mutation to the concrete store."""

    async def detached_record(
        harness: _Harness,
        *,
        reason: str,
        default: bool = False,
    ) -> InteractionRecord:
        question = ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
            default_value=True if default else None,
        )
        result = await harness.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=_Resumer(),
                reason=reason,
                questions=(question,),
            )
        )
        assert result.delivery is not None
        return result.delivery.record

    async def exercise() -> None:
        default_harness = await _harness()
        activity_harness = await _harness()
        cancel_harness = await _harness()
        supersede_harness = await _harness()
        no_handler_harness = await _harness()
        cancel_policy_harness = await _harness(
            policy=replace(
                InteractionPolicy(),
                attached_loss_without_resumer=(
                    HandlerLossDisposition.CANCEL_REQUEST
                ),
            )
        )
        try:
            default_record = await detached_record(
                default_harness,
                reason="Resolve the declared default.",
                default=True,
            )
            defaulted = await default_harness.broker.resolve_trusted_default(
                TrustedDefaultResolutionRequest(
                    actor=_actor(),
                    correlation=default_record.correlation,
                    expected_state_revision=(
                        default_record.request.state_revision
                    ),
                )
            )
            assert isinstance(
                defaulted.store_result,
                TrustedDefaultResolutionApplied,
            )

            handler = _BlockingHandler()
            activity_task = create_task(
                activity_harness.broker.request(
                    _request(
                        handler,
                        mode=RequirementMode.ADVISORY,
                        advisory_wait_seconds=30,
                        reason="Record active control.",
                    )
                )
            )
            await handler.started.wait()
            activity_correlation = _correlation(
                activity_harness.ids.request_ids[0],
                activity_harness.ids.continuation_ids[0],
                _origin(),
            )
            activity_record = await _inspect(
                activity_harness.broker,
                activity_correlation,
            )
            activity = await activity_harness.broker.record_activity(
                RecordControllerActivityCommand(
                    actor=_actor(),
                    correlation=activity_correlation,
                    evidence=AcquireControllerActivity(
                        request_id=activity_record.request.request_id,
                        controller_id=ControllerId("controller"),
                    ),
                )
            )
            assert isinstance(activity.store_result, ControllerActivityApplied)
            terminal = await activity_harness.broker.terminalize(
                TerminalizeInteractionCommand(
                    actor=_actor(),
                    correlation=activity_correlation,
                    status=ResolutionStatus.UNAVAILABLE,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=(
                        activity_record.request.state_revision
                    ),
                )
            )
            assert isinstance(
                terminal.store_result,
                TerminalizeInteractionApplied,
            )
            await activity_task

            await detached_record(
                cancel_harness,
                reason="Cancel this complete run.",
            )
            cancelled = await cancel_harness.broker.cancel_scope(
                TerminalizeInteractionScopeCommand(
                    actor=_actor(),
                    scope=InteractionExecutionScope(run_id=RunId("run")),
                    provenance=AnswerProvenance.HUMAN,
                )
            )
            assert isinstance(
                cancelled.store_result,
                ScopeCancellationApplied,
            )

            await detached_record(
                supersede_harness,
                reason="Supersede this complete run.",
            )
            superseded = await supersede_harness.broker.supersede(
                SupersedeInteractionScopeCommand(
                    actor=_actor(),
                    scope=InteractionExecutionScope(run_id=RunId("run")),
                    provenance=AnswerProvenance.HUMAN,
                )
            )
            assert isinstance(
                superseded.store_result,
                ScopeSupersessionApplied,
            )

            no_handler = await no_handler_harness.broker.request(
                _request(None, reason="No renderer is attached.")
            )
            assert no_handler.delivery is not None
            assert (
                no_handler.delivery.record.request.state
                is RequestState.UNAVAILABLE
            )

            loss_cancelled = await cancel_policy_harness.broker.request(
                _request(
                    _LossHandler(InputHandlerDetached()),
                    reason="Cancel after renderer loss.",
                )
            )
            assert loss_cancelled.delivery is not None
            assert (
                loss_cancelled.delivery.record.request.state
                is RequestState.CANCELLED
            )
        finally:
            for harness in (
                default_harness,
                activity_harness,
                cancel_harness,
                supersede_harness,
                no_handler_harness,
                cancel_policy_harness,
            ):
                await harness.broker.aclose()

    run(exercise())


def test_root_fifo_and_cross_root_concurrency_hold_no_external_lock() -> None:
    """Serialize one root while another root and heartbeat continue."""

    async def exercise() -> None:
        harness = await _harness()
        first = _BlockingHandler()
        second = _BlockingHandler()
        other = _BlockingHandler()
        first_task = create_task(
            harness.broker.request(
                _request(
                    first,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="First same-root request.",
                )
            )
        )
        await first.started.wait()
        second_task = create_task(
            harness.broker.request(
                _request(
                    second,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="Second same-root request.",
                )
            )
        )
        other_task = create_task(
            harness.broker.request(
                _request(
                    other,
                    origin=_origin(branch_id="other-root"),
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="Independent root request.",
                )
            )
        )
        await other.started.wait()
        assert not second.started.is_set()
        assert await harness.broker.heartbeat() == InteractionBrokerHeartbeat(
            sequence=1
        )
        first.release.set()
        await second.started.wait()
        second.release.set()
        other.release.set()
        results = await _gather_request_results(
            first_task,
            second_task,
            other_task,
        )
        assert all(result.delivery is not None for result in results)
        await harness.broker.aclose()

    run(exercise())


def test_reopened_broker_serializes_opaque_cousins_by_persisted_root() -> None:
    """Serialize reopened edge-only cousins without ancestor requests."""

    async def exercise() -> None:
        harness = await _harness()
        run_id = RunId("reloaded-run")
        branch_edges = (
            ("branch-b", "branch-a"),
            ("branch-c", "branch-b"),
            ("branch-x", "branch-a"),
            ("branch-d", "branch-x"),
        )
        for child, parent in branch_edges:
            result = await harness.broker.register_branch(
                RegisterInteractionBranchCommand(
                    actor=_actor(),
                    registration=InteractionBranchRegistration(
                        run_id=run_id,
                        branch_id=BranchId(child),
                        parent_branch_id=BranchId(parent),
                        principal=_principal(),
                    ),
                )
            )
            assert isinstance(
                result.store_result,
                InteractionBranchRegistrationApplied,
            )
        await harness.broker.aclose()

        reopened_store = await harness.factory.open()
        reopened = AsyncInteractionBroker(
            store=reopened_store,
            clock=harness.clock,
            id_factory=harness.ids,
            policy=harness.policy,
            classifier=harness.classifier,
        )
        first_handler = _BlockingHandler()
        second_handler = _BlockingHandler()
        first_origin = _origin(
            run_id=str(run_id),
            branch_id="branch-c",
            parent_branch_id="branch-b",
        )
        second_origin = _origin(
            run_id=str(run_id),
            branch_id="branch-d",
            parent_branch_id="branch-x",
        )
        first_task = create_task(
            reopened.request(
                _request(
                    first_handler,
                    origin=first_origin,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="First opaque cousin after reload.",
                )
            )
        )
        await first_handler.started.wait()
        second_task = create_task(
            reopened.request(
                _request(
                    second_handler,
                    origin=second_origin,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="Second opaque cousin after reload.",
                )
            )
        )
        await _wait_until(
            lambda: any(
                len(queue) == 2 for queue in reopened._root_queues.values()
            )
        )
        assert not second_handler.started.is_set()
        assert set(reopened._root_queues) == {
            (_principal(), run_id, BranchId("branch-a"))
        }

        first_handler.release.set()
        await second_handler.started.wait()
        second_handler.release.set()
        first_result = await first_task
        second_result = await second_task
        assert first_result.delivery is not None
        assert second_result.delivery is not None
        await reopened.aclose()

    run(exercise())


@pytest.mark.parametrize(
    "root_result",
    (
        None,
        object(),
        InteractionBranchRoot(
            run_id=RunId("wrong-run"),
            branch_id=BranchId("child"),
            root_branch_id=BranchId("root"),
        ),
        InteractionBranchRoot(
            run_id=RunId("run"),
            branch_id=BranchId("wrong-child"),
            root_branch_id=BranchId("root"),
        ),
    ),
    ids=("absent", "invalid", "wrong-run", "wrong-branch"),
)
def test_branch_root_lookup_failure_never_presents(
    root_result: object,
) -> None:
    """Apply handler-loss policy when authoritative root lookup fails."""

    async def exercise() -> None:
        harness = await _harness()
        registered = await harness.broker.register_branch(
            RegisterInteractionBranchCommand(
                actor=_actor(),
                registration=InteractionBranchRegistration(
                    run_id=RunId("run"),
                    branch_id=BranchId("child"),
                    parent_branch_id=BranchId("root"),
                    principal=_principal(),
                ),
            )
        )
        assert isinstance(
            registered.store_result,
            InteractionBranchRegistrationApplied,
        )
        harness.broker._store = cast(
            InteractionStore,
            _BranchRootResultStore(harness.store, root_result),
        )
        handler = _BlockingHandler()
        result = await harness.broker.request(
            _request(
                handler,
                origin=_origin(
                    branch_id="child",
                    parent_branch_id="root",
                ),
            )
        )
        assert result.delivery is not None
        assert result.delivery.record.request.state is RequestState.UNAVAILABLE
        assert not handler.started.is_set()
        await harness.broker.aclose()

    run(exercise())


def test_queued_cancel_skips_handler_and_external_terminal_stops_active() -> (
    None
):
    """Skip queued delivery and stop active handlers after terminal commit."""

    async def exercise() -> None:
        harness = await _harness()
        first = _BlockingHandler()
        queued = _BlockingHandler()
        origin = _origin()
        first_task = create_task(
            harness.broker.request(
                _request(
                    first,
                    origin=origin,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="Active request.",
                )
            )
        )
        await first.started.wait()
        queued_task = create_task(
            harness.broker.request(
                _request(
                    queued,
                    origin=origin,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="Queued request.",
                )
            )
        )
        await _wait_until(lambda: len(harness.ids.request_ids) == 2)
        queued_correlation = _correlation(
            harness.ids.request_ids[1],
            harness.ids.continuation_ids[1],
            origin,
        )
        await _wait_for_record(harness.broker, queued_correlation)
        queued_record = await _inspect(harness.broker, queued_correlation)
        cancelled = await harness.broker.cancel(
            CancelInteractionCommand(
                actor=_actor(),
                correlation=queued_correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=queued_record.request.state_revision,
            )
        )
        assert isinstance(cancelled.store_result, CancelInteractionApplied)
        first_record = await _inspect(
            harness.broker,
            _correlation(
                harness.ids.request_ids[0],
                harness.ids.continuation_ids[0],
                origin,
            ),
        )
        first_cancel = await harness.broker.cancel(
            CancelInteractionCommand(
                actor=_actor(),
                correlation=first_record.correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=first_record.request.state_revision,
            )
        )
        assert isinstance(first_cancel.store_result, CancelInteractionApplied)
        await first.stopped.wait()
        first_result, queued_result = await _gather_request_results(
            first_task,
            queued_task,
        )
        assert first_result.delivery is not None
        assert queued_result.delivery is not None
        assert (
            queued_result.delivery.record.request.state
            is RequestState.CANCELLED
        )
        assert not queued.started.is_set()
        await harness.broker.aclose()

    run(exercise())


def test_repeated_queued_cancellation_cannot_leave_stale_fifo_ticket() -> None:
    """Finish queued ticket cleanup despite repeated caller cancellation."""

    async def exercise() -> None:
        harness = await _harness()
        first_handler = _BlockingHandler()
        second_handler = _BlockingHandler()
        first_task = create_task(
            harness.broker.request(
                _request(
                    first_handler,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="Hold the root presentation slot.",
                )
            )
        )
        await first_handler.started.wait()
        second_task = create_task(
            harness.broker.request(
                _request(
                    second_handler,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="Cancel this queued presentation.",
                )
            )
        )
        await _wait_until(
            lambda: any(
                len(queue) == 2
                for queue in harness.broker._root_queues.values()
            )
        )

        await harness.broker._state_lock.acquire()
        try:
            second_task.cancel()
            await _yield_once()
            assert not second_task.done()
            second_task.cancel()
            await _yield_once()
            assert not second_task.done()
        finally:
            harness.broker._state_lock.release()
        with pytest.raises(CancelledError):
            await second_task
        assert not second_handler.started.is_set()
        assert all(
            len(queue) == 1 for queue in harness.broker._root_queues.values()
        )

        third_handler = _BlockingHandler()
        third_task = create_task(
            harness.broker.request(
                _request(
                    third_handler,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="Advance after cancelled queue cleanup.",
                )
            )
        )
        await _wait_until(
            lambda: any(
                len(queue) == 2
                for queue in harness.broker._root_queues.values()
            )
        )
        first_handler.release.set()
        await third_handler.started.wait()
        third_handler.release.set()
        await first_task
        await third_task
        await harness.broker.aclose()

    run(exercise())


def test_owning_call_cancellation_cleans_handler_and_detaches() -> None:
    """Cancel the owning call without declining or leaking its handler."""

    async def exercise() -> None:
        harness = await _harness()
        handler = _BlockingHandler()
        resumer = _Resumer()
        task = create_task(
            harness.broker.request(_request(handler, resumer=resumer))
        )
        await handler.started.wait()
        task.cancel()
        with pytest.raises(CancelledError):
            await task
        await handler.stopped.wait()
        correlation = _correlation(
            harness.ids.request_ids[0],
            harness.ids.continuation_ids[0],
            _origin(),
        )
        record = await _inspect(harness.broker, correlation)
        assert record.request.state is RequestState.PENDING
        assert record.presentation is InteractionPresentationState.DETACHED
        assert resumer.notifications == []
        await harness.broker.aclose()

    run(exercise())


def test_handler_child_cancellation_applies_loss_policy() -> None:
    """Treat isolated attached-handler cancellation as handler loss."""

    async def exercise() -> None:
        harness = await _harness()
        handler = _BlockingHandler()
        request_task = create_task(harness.broker.request(_request(handler)))
        await handler.started.wait()
        request_id = harness.ids.request_ids[0]
        handler_task = harness.broker._handler_tasks[request_id]
        handler_task.cancel()
        result = await request_task
        assert result.delivery is not None
        assert result.delivery.record.request.state is RequestState.UNAVAILABLE
        assert handler.stopped.is_set()
        await harness.broker.aclose()

    run(exercise())


def test_wait_cancellation_isolated_and_resumer_runs_at_most_once() -> None:
    """Cancel one observer wait without affecting another or resumption."""

    async def exercise() -> None:
        harness = await _harness()
        resumer = _Resumer()
        result = await harness.broker.request(
            _request(_LossHandler(InputHandlerDetached()), resumer=resumer)
        )
        assert result.delivery is not None
        record = result.delivery.record
        command = WaitForInteractionChangeCommand(
            actor=_actor(),
            correlation=record.correlation,
            after_store_revision=record.store_revision,
        )
        first_wait = create_task(harness.broker.wait(command))
        second_wait = create_task(harness.broker.wait(command))
        await _wait_until(
            lambda: (
                harness.authorizer.operations.count(InteractionOperation.WAIT)
                >= 3
            )
        )
        await _yield_once()
        first_wait.cancel()
        with pytest.raises(CancelledError):
            await first_wait
        mutation = await harness.broker.resolve(
            _decline(record, "decline-key")
        )
        assert isinstance(mutation, InteractionBrokerResult)
        assert isinstance(mutation.store_result, ResolveInteractionApplied)
        assert mutation.resumer_failed is False
        changed = await second_wait
        assert isinstance(changed, InteractionRecord)
        assert changed.request.state is RequestState.DECLINED
        await resumer.called.wait()
        assert len(resumer.notifications) == 1
        replay = await harness.broker.resolve(_decline(record, "decline-key"))
        assert replay.resumer_failed is False
        assert len(resumer.notifications) == 1
        await harness.broker.aclose()

    run(exercise())


@pytest.mark.parametrize(
    "cancelled", (False, True), ids=("error", "cancelled")
)
def test_resumer_failure_is_post_commit_and_never_retried(
    cancelled: bool,
) -> None:
    """Report callback failure without rollback or replay dispatch."""

    async def exercise() -> None:
        harness = await _harness()
        resumer = _Resumer(fail=not cancelled, cancel=cancelled)
        request_result = await harness.broker.request(
            _request(_LossHandler(InputHandlerDetached()), resumer=resumer)
        )
        assert request_result.delivery is not None
        record = request_result.delivery.record
        result = await harness.broker.resolve(_decline(record, "failure-key"))
        assert isinstance(result.store_result, ResolveInteractionApplied)
        assert result.resumer_failed is True
        stored = await _inspect(harness.broker, record.correlation)
        assert stored.request.state is RequestState.DECLINED
        assert len(resumer.notifications) == 1
        replay = await harness.broker.resolve(_decline(record, "failure-key"))
        assert replay.resumer_failed is True
        assert len(resumer.notifications) == 1
        await harness.broker.aclose()

    run(exercise())


def test_shared_resumer_survives_caller_cancel_and_reports_to_replay() -> None:
    """Shield one post-commit callback and share its failure result."""

    async def exercise() -> None:
        harness = await _harness()
        resumer = _Resumer(fail=True, block=True)
        request_result = await harness.broker.request(
            _request(_LossHandler(InputHandlerDetached()), resumer=resumer)
        )
        assert request_result.delivery is not None
        record = request_result.delivery.record
        command = _decline(record, "shared-failure-key")

        first = create_task(harness.broker.resolve(command))
        await resumer.called.wait()
        replay = create_task(harness.broker.resolve(command))
        await _wait_until(
            lambda: (
                harness.authorizer.operations.count(
                    InteractionOperation.RESOLVE
                )
                >= 2
            )
        )
        assert not replay.done()

        first.cancel()
        with pytest.raises(CancelledError):
            await first
        assert not resumer.stopped.is_set()
        assert not replay.done()

        resumer.release.set()
        replayed = await replay
        await resumer.stopped.wait()
        assert replayed.resumer_failed is True
        assert len(resumer.notifications) == 1
        assert harness.broker._resumer_tasks == {}
        await harness.broker.aclose()

    run(exercise())


def test_cancelled_pre_handoff_delivery_unblocks_store_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail one canceled local registration without stranding the bridge."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        entered = Event()
        blocker = Event()
        original_settle = harness.broker._settle_local_tasks

        async def paused_settle(
            request_id: InputRequestId,
            notification: InputResumptionNotification,
        ) -> None:
            """Pause before the broker delivery reaches its handoff point."""
            entered.set()
            await blocker.wait()
            await original_settle(request_id, notification)

        monkeypatch.setattr(
            harness.broker,
            "_settle_local_tasks",
            paused_settle,
        )
        resumer = _Resumer()
        created = await harness.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=resumer,
            )
        )
        assert created.delivery is not None
        record = created.delivery.record

        external_mutation = create_task(
            external.cancel(
                CancelInteractionCommand(
                    actor=_actor(),
                    correlation=record.correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=record.request.state_revision,
                )
            )
        )
        await entered.wait()
        delivery = harness.broker._resumer_tasks[record.request.request_id]
        delivery.cancel()

        mutation = await external_mutation
        assert isinstance(mutation, CancelInteractionApplied)
        assert await delivery is True
        assert record.request.request_id in harness.broker._resumer_failures
        assert record.request.request_id not in harness.broker._resumer_tasks
        assert (
            record.request.request_id not in harness.broker._resumer_handoffs
        )
        assert resumer.notifications == []
        await harness.broker.aclose()
        await external.aclose()

    run(exercise())


def test_close_joins_store_bridge_owned_blocking_resumer() -> None:
    """Cancel and join a callback after the bridge joins its watcher."""

    async def exercise() -> None:
        harness = await _harness()
        resumer = _Resumer(block=True)
        request_result = await harness.broker.request(
            _request(_LossHandler(InputHandlerDetached()), resumer=resumer)
        )
        assert request_result.delivery is not None
        record = request_result.delivery.record

        cancelled = await harness.store.cancel(
            CancelInteractionCommand(
                actor=_actor(),
                correlation=record.correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=record.request.state_revision,
            )
        )
        assert isinstance(cancelled, CancelInteractionApplied)
        await resumer.called.wait()
        assert record.request.request_id in harness.broker._resumer_tasks
        assert record.request.request_id not in harness.broker._watcher_tasks

        await harness.broker.aclose()
        await resumer.stopped.wait()
        assert harness.broker._resumer_tasks == {}
        assert harness.broker._watcher_tasks == {}

    run(exercise())


@pytest.mark.parametrize("block_callback", (False, True))
def test_close_waits_for_extracted_bridge_handoff_before_task_join(
    monkeypatch: pytest.MonkeyPatch,
    block_callback: bool,
) -> None:
    """Accept a paused terminal bridge before closing task ownership."""
    bridge_entered, bridge_release = _pause_store_bound_resumer(monkeypatch)

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        resumer = _Resumer(block=block_callback)
        created = await harness.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=resumer,
            )
        )
        assert created.delivery is not None
        record = created.delivery.record

        external_mutation = create_task(
            external.cancel(
                CancelInteractionCommand(
                    actor=_actor(),
                    correlation=record.correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=record.request.state_revision,
                )
            )
        )
        await bridge_entered.wait()
        binding = next(iter(harness.factory._state.admissions.values()))
        assert not binding.handoff.done()

        closing = create_task(harness.broker.aclose())
        await _wait_until(lambda: harness.broker._closing_settlements)
        await _yield_once()
        assert not closing.done()
        assert record.request.request_id in harness.broker._admission_cleanups

        bridge_release.set()
        mutation = await external_mutation
        assert isinstance(mutation, CancelInteractionApplied)
        await resumer.called.wait()
        await closing

        assert binding.handoff.done()
        assert len(resumer.notifications) == 1
        assert resumer.stopped.is_set()
        assert harness.broker._admission_cleanups == {}
        assert harness.broker._resumer_tasks == {}
        assert harness.broker._resumer_handoffs == {}
        await external.aclose()

    run(exercise())


def test_deadline_pump_rearms_and_cancels_losing_wait() -> None:
    """Rearm changed deadlines and settle the new schedule."""

    async def exercise() -> None:
        harness = await _harness()
        first_resumer = _Resumer()
        second_resumer = _Resumer()
        first_handler = _BlockingHandler()
        second_handler = _BlockingHandler()
        first_task = create_task(
            harness.broker.request(
                _request(
                    first_handler,
                    resumer=first_resumer,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="Long advisory request.",
                )
            )
        )
        await first_handler.started.wait()
        await _wait_until(lambda: len(harness.clock.wait_calls) >= 1)
        second_task = create_task(
            harness.broker.request(
                _request(
                    second_handler,
                    resumer=second_resumer,
                    origin=_origin(branch_id="second-root"),
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=5,
                    reason="Short advisory request.",
                )
            )
        )
        await second_handler.started.wait()
        await _wait_until(
            lambda: (
                harness.clock.wait_calls[-1] == 5.0
                and harness.clock.cancelled_waits >= 1
            )
        )
        harness.clock.advance(5)
        await second_resumer.called.wait()
        await second_handler.stopped.wait()
        second = await second_task
        assert second.delivery is not None
        second_record = await _inspect(
            harness.broker,
            second.delivery.correlation,
        )
        first_correlation = _correlation(
            harness.ids.request_ids[0],
            harness.ids.continuation_ids[0],
            _origin(),
        )
        first_record = await _inspect(
            harness.broker,
            first_correlation,
        )
        assert second_record.request.state is RequestState.TIMED_OUT
        assert first_record.request.state is RequestState.PENDING
        cancelled = await harness.broker.cancel(
            CancelInteractionCommand(
                actor=_actor(),
                correlation=first_correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=first_record.request.state_revision,
            )
        )
        assert isinstance(cancelled.store_result, CancelInteractionApplied)
        await first_task
        await harness.broker.aclose()

    run(exercise())


def test_deadline_pump_cancellation_drains_both_armed_waits() -> None:
    """Drain both child waits when cancellation interrupts their race."""

    async def exercise() -> None:
        harness = await _harness()
        handler = _BlockingHandler()
        request_task = create_task(
            harness.broker.request(
                _request(
                    handler,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                )
            )
        )
        await handler.started.wait()
        child_names = {
            "interaction-deadline-timer",
            "interaction-deadline-change",
        }
        await _wait_until(
            lambda: (
                {task.get_name() for task in all_tasks() if not task.done()}
                >= child_names
            )
        )
        children = tuple(
            task for task in all_tasks() if task.get_name() in child_names
        )
        assert len(children) == 2
        pump = harness.broker._deadline_task
        assert pump is not None

        pump.cancel()
        await pump

        assert all(task.done() for task in children)

        await harness.broker.aclose()
        result = await request_task

        assert result.delivery is not None
        assert result.delivery.record.request.state is RequestState.UNAVAILABLE

    run(exercise())


def test_deadline_pump_recovers_and_progresses_pending_deadline() -> None:
    """Recover one adapter failure and still terminalize a pending request."""

    async def exercise() -> None:
        observer = _Observer()
        harness = await _harness(observer=observer)
        proxy = _OneShotDeadlineErrorStore(harness.store)
        harness.broker._store = cast(InteractionStore, proxy)
        await harness.broker.heartbeat()
        await proxy.failed.wait()
        await _wait_until(
            lambda: any(
                event.kind is InteractionObserverEventKind.DEADLINE_PUMP_FAILED
                for event in observer.events
            )
        )

        handler = _BlockingHandler()
        resumer = _Resumer()
        task = create_task(
            harness.broker.request(
                _request(
                    handler,
                    resumer=resumer,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=5,
                )
            )
        )
        await handler.started.wait()
        await _wait_until(
            lambda: (
                bool(harness.clock.wait_calls)
                and harness.clock.wait_calls[-1] == 5.0
            )
        )
        harness.clock.advance(5)
        await resumer.called.wait()
        result = await task
        assert result.delivery is not None
        assert result.delivery.record.request.state is RequestState.TIMED_OUT
        assert harness.broker.is_closed is False
        await harness.broker.aclose()

    run(exercise())


def test_watcher_and_cancellation_cleanup_fail_closed() -> None:
    """Handle redaction, adapter failure, task cleanup, and closed stores."""

    async def exercise() -> None:
        harness = await _harness()
        request = _request(
            _LossHandler(InputHandlerDetached()),
            resumer=_Resumer(),
        )
        created = await harness.broker.request(request)
        assert created.delivery is not None
        pending = created.delivery.record
        cancelled = await harness.broker.cancel(
            CancelInteractionCommand(
                actor=_actor(),
                correlation=pending.correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=pending.request.state_revision,
            )
        )
        assert isinstance(cancelled.store_result, CancelInteractionApplied)
        terminal = cancelled.store_result.record

        harness.broker._store = cast(
            InteractionStore,
            _ProjectedWaitStore(harness.store, terminal),
        )
        assert isinstance(created.create_result, CreateInteractionApplied)
        await harness.broker._watch_terminal(
            _actor(),
            created.create_result.record,
        )
        harness.broker._store = harness.store

        harness.authorizer.disclosure = InteractionDisclosure.TERMINAL_METADATA
        await harness.broker._watch_terminal(
            _actor(),
            created.create_result.record,
        )

        harness.authorizer.disclosure = InteractionDisclosure.FULL
        harness.broker._store = cast(
            InteractionStore,
            _WaitErrorStore(harness.store),
        )
        await harness.broker._start_watcher(_actor(), terminal)
        await _wait_until(
            lambda: (
                terminal.request.request_id
                not in harness.broker._watcher_tasks
            )
        )

        async def pending_outcome() -> InputHandlerOutcome:
            await Event().wait()
            return InputHandlerDetached()

        synthetic_id = InputRequestId("synthetic-handler")
        synthetic_task = create_task(pending_outcome())
        harness.broker._handler_tasks[synthetic_id] = synthetic_task
        await harness.broker._cancel_handler(synthetic_id)
        assert synthetic_task.cancelled()

        await harness.broker.aclose()
        await harness.broker._cleanup_cancelled_request(request, terminal)
        await harness.broker._await_cancellation_cleanup(request, terminal)

        closed_store = await _harness()
        closed_request = _request(
            _LossHandler(InputHandlerDetached()),
            resumer=_Resumer(),
        )
        closed_created = await closed_store.broker.request(closed_request)
        assert closed_created.delivery is not None
        await closed_store.store.aclose()
        await closed_store.broker._cleanup_cancelled_request(
            closed_request,
            closed_created.delivery.record,
        )
        with pytest.raises(InteractionStoreClosedError):
            await closed_store.broker.aclose()
        assert closed_store.broker._admission_cleanups
        reopened = await closed_store.factory.open()
        closed_store.broker._store = reopened
        await closed_store.broker.aclose()
        assert closed_store.broker._admission_cleanups == {}

    run(exercise())


@pytest.mark.parametrize(
    "watcher_path",
    ("redacted", "transient_error"),
)
def test_store_bound_resumer_survives_watcher_abandonment(
    watcher_path: str,
) -> None:
    """Resume once after redaction or a transient watcher adapter failure."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        resumer = _Resumer()
        if watcher_path == "redacted":
            proxy: _RecordingWaitStore | _OneShotWaitErrorStore = (
                _RecordingWaitStore(harness.store)
            )
        else:
            proxy = _OneShotWaitErrorStore(harness.store)
        harness.broker._store = cast(InteractionStore, proxy)
        try:
            created = await harness.broker.request(
                _request(
                    _LossHandler(InputHandlerDetached()),
                    resumer=resumer,
                )
            )
            assert created.delivery is not None
            pending = created.delivery.record
            if isinstance(proxy, _RecordingWaitStore):
                harness.authorizer.disclosure = (
                    InteractionDisclosure.TERMINAL_METADATA
                )
            else:
                await proxy.failed.wait()
                await _wait_until(
                    lambda: (
                        pending.request.request_id
                        not in harness.broker._watcher_tasks
                    )
                )

            cancelled = await external.cancel(
                CancelInteractionCommand(
                    actor=_actor(),
                    correlation=pending.correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=pending.request.state_revision,
                )
            )
            assert isinstance(cancelled, CancelInteractionApplied)
            await resumer.called.wait()
            if isinstance(proxy, _RecordingWaitStore):
                await proxy.returned.wait()
                assert isinstance(
                    proxy.projections[-1],
                    InteractionTerminalMetadata,
                )
                await _wait_until(
                    lambda: (
                        pending.request.request_id
                        not in harness.broker._watcher_tasks
                    )
                )
            assert len(resumer.notifications) == 1
            assert (
                resumer.notifications[0].continuation_id
                == pending.request.continuation_id
            )
        finally:
            await external.aclose()
            await harness.broker.aclose()

    run(exercise())


@pytest.mark.parametrize("with_resumer", (False, True))
def test_store_bridge_stops_active_handler_after_watcher_failure(
    with_resumer: bool,
) -> None:
    """Join an active handler on a later cross-handle terminal commit."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        proxy = _OneShotWaitErrorStore(harness.store)
        harness.broker._store = cast(InteractionStore, proxy)
        handler = _BlockingHandler()
        resumer = _Resumer() if with_resumer else None
        request_task = create_task(
            harness.broker.request(_request(handler, resumer=resumer))
        )
        try:
            await handler.started.wait()
            await proxy.failed.wait()
            request_id = harness.ids.request_ids[0]
            await _wait_until(
                lambda: request_id not in harness.broker._watcher_tasks
            )
            correlation = _correlation(
                request_id,
                harness.ids.continuation_ids[0],
                _origin(),
            )
            pending = await _inspect(harness.broker, correlation)

            cancelled = await external.cancel(
                CancelInteractionCommand(
                    actor=_actor(),
                    correlation=correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=pending.request.state_revision,
                )
            )
            assert isinstance(cancelled, CancelInteractionApplied)
            await handler.stopped.wait()
            result = await request_task
            assert result.delivery is not None
            assert (
                result.delivery.record.request.state is RequestState.CANCELLED
            )
            if resumer is not None:
                await resumer.called.wait()
                assert len(resumer.notifications) == 1
            await _wait_until(
                lambda: request_id not in harness.broker._resumer_tasks
            )
        finally:
            await external.aclose()
            await harness.broker.aclose()

    run(exercise())


def test_observer_failure_overflow_and_heartbeat_are_non_authoritative() -> (
    None
):
    """Keep observer latency, loss, and failures outside correctness."""

    async def exercise() -> None:
        blocking = _Observer(block_first=True, fail_overflow=True)
        harness = await _harness(
            observer=blocking,
            observer_queue_capacity=1,
        )
        assert await harness.broker.heartbeat() == InteractionBrokerHeartbeat(
            sequence=1
        )
        await blocking.entered.wait()
        for expected in range(2, 7):
            assert await harness.broker.heartbeat() == (
                InteractionBrokerHeartbeat(sequence=expected)
            )
        blocking.release.set()
        await blocking.overflow.wait()
        overflow = next(
            event
            for event in blocking.events
            if event.kind is InteractionObserverEventKind.OVERFLOW
        )
        assert overflow.dropped_events >= 1
        assert tuple(item.name for item in fields(overflow)) == (
            "kind",
            "request_id",
            "status",
            "schedule_revision",
            "dropped_events",
        )
        await harness.broker.aclose()

        failing = _Observer(fail=True)
        harness = await _harness(observer=failing)
        result = await harness.broker.request(_request(_CorrectionHandler()))
        assert result.delivery is not None
        assert result.delivery.record.request.state is RequestState.ANSWERED
        await _wait_until(lambda: len(failing.events) >= 1)
        await harness.broker.aclose()

        deadline_observer = _Observer()
        harness = await _harness(observer=deadline_observer)
        harness.broker._store = cast(
            InteractionStore,
            _DeadlineErrorStore(harness.store),
        )
        await harness.broker.heartbeat()
        await _wait_until(lambda: harness.broker.is_closed)
        await harness.broker.aclose()

    run(exercise())


def test_close_is_idempotent_and_cleans_owned_tasks() -> None:
    """Cancel handlers, watchers, queues, and pumps on local close."""

    async def exercise() -> None:
        observer = _Observer(block_first=True)
        harness = await _harness(observer=observer)
        handler = _BlockingHandler()
        task = create_task(
            harness.broker.request(
                _request(
                    handler,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                )
            )
        )
        await handler.started.wait()
        queued_handler = _BlockingHandler()
        queued_task = create_task(
            harness.broker.request(
                _request(
                    queued_handler,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="Close this queued request.",
                )
            )
        )
        await _wait_until(
            lambda: any(
                len(queue) == 2
                for queue in harness.broker._root_queues.values()
            )
        )
        await harness.broker.aclose()
        await handler.stopped.wait()
        result = await _task_result(task)
        assert isinstance(result, InteractionRequestResult)
        assert result.delivery is not None
        assert result.delivery.record.request.state is RequestState.UNAVAILABLE
        queued_result = await _task_result(queued_task)
        assert isinstance(queued_result, InteractionStoreClosedError)
        assert not queued_handler.started.is_set()
        assert harness.broker.is_closed is True
        assert harness.broker._create_tasks == {}
        assert harness.broker._handler_tasks == {}
        assert harness.broker._watcher_tasks == {}
        assert harness.broker._resumer_tasks == {}
        assert harness.broker._root_queues == {}
        assert harness.broker._deadline_task is None
        assert harness.broker._observer_task is None
        assert harness.clock._waiters == []
        await harness.broker.aclose()
        with pytest.raises(InteractionStoreClosedError):
            await harness.broker.heartbeat()

    run(exercise())


def test_store_bound_terminal_delivery_releases_broker_cleanup_authority() -> (
    None
):
    """Drop broker authority after conclusive terminal bridge extraction."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        resumer = _Resumer()
        created = await harness.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=resumer,
            )
        )
        assert created.delivery is not None
        record = created.delivery.record
        request_id = record.request.request_id
        assert request_id in harness.broker._admission_cleanups

        cancelled = await external.cancel(
            CancelInteractionCommand(
                actor=_actor(),
                correlation=record.correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=record.request.state_revision,
            )
        )
        assert isinstance(cancelled, CancelInteractionApplied)
        await resumer.called.wait()
        await _wait_until(
            lambda: request_id not in harness.broker._admission_cleanups
        )
        assert len(harness.factory._state.admissions) == 1
        await harness.broker.aclose()
        await external.aclose()

    run(exercise())


def test_failed_store_close_retry_settles_late_excluded_create() -> None:
    """Retain create and cleanup authority until retry settles its commit."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        proxy = _CloseBeforeCommitAdmissionStore(
            harness.store,
            fail_first_close=True,
        )
        proxy.broker = harness.broker
        harness.broker._store = cast(InteractionStore, proxy)
        resumer = _Resumer()
        request_task = create_task(
            harness.broker.request(
                _request(
                    _LossHandler(InputHandlerDetached()),
                    resumer=resumer,
                )
            )
        )

        await proxy.close_failed.wait()
        request_id = harness.ids.request_ids[0]
        create_operation = harness.broker._create_tasks[request_id]
        assert request_id in harness.broker._admission_cleanups
        assert create_operation in harness.broker._close_owned_tasks
        assert harness.factory._state.admissions == {}
        assert harness.factory._state.resumers == {}

        retry = create_task(harness.broker.aclose())
        await proxy.create_cancelled.wait()
        await proxy.committed.wait()
        await retry
        await _task_result(request_task)

        correlation = _correlation(
            request_id,
            harness.ids.continuation_ids[0],
            _origin(),
        )
        terminal = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=correlation,
            )
        )
        assert isinstance(terminal, InteractionRecord)
        assert terminal.request.state is RequestState.UNAVAILABLE
        assert len(resumer.notifications) == 1
        assert harness.factory._state.resumers == {}
        assert harness.broker._admission_cleanups == {}
        assert harness.broker._create_tasks == {}
        assert harness.broker._close_owned_tasks == ()
        assert proxy.close_calls == 2
        await external.aclose()

    run(exercise())


def test_successful_store_close_prevents_excluded_create_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release deferred absence only after store close forbids mutation."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        finalization_entered = Event()
        finalization_release = Event()
        original_finalize = harness.broker._finalize_close

        async def paused_finalize() -> None:
            """Pause after store close before retained-state release."""
            finalization_entered.set()
            await finalization_release.wait()
            await original_finalize()

        monkeypatch.setattr(
            harness.broker,
            "_finalize_close",
            paused_finalize,
        )
        proxy = _CloseBeforeCommitAdmissionStore(
            harness.store,
            fail_first_close=False,
        )
        proxy.broker = harness.broker
        harness.broker._store = cast(InteractionStore, proxy)
        request_task = create_task(
            harness.broker.request(
                _request(_LossHandler(InputHandlerDetached()))
            )
        )

        await finalization_entered.wait()
        close_task = harness.broker._close_task
        assert close_task is not None
        close_task.cancel()
        await _yield_once()
        assert not close_task.done()
        assert not request_task.done()
        finalization_release.set()
        await proxy.commit_attempted.wait()
        outcome = await _task_result(request_task)
        assert isinstance(outcome, InteractionStoreClosedError)
        assert not proxy.committed.is_set()
        assert proxy.close_calls == 1
        assert harness.factory._state.admissions == {}
        assert harness.factory._state.resumers == {}
        assert harness.broker._admission_cleanups == {}
        assert harness.broker._create_tasks == {}
        assert harness.broker._close_owned_tasks == ()

        absent = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=_correlation(
                    harness.ids.request_ids[0],
                    harness.ids.continuation_ids[0],
                    _origin(),
                ),
            )
        )
        assert absent is None
        await harness.broker.aclose()
        await external.aclose()

    run(exercise())


def test_close_cleans_admission_committed_before_create_returns() -> None:
    """Close the post-commit create cancellation window without an orphan."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        blocking = _CommitThenBlockCreateStore(harness.store)
        harness.broker._store = cast(InteractionStore, blocking)
        handler = _BlockingHandler()
        request_task = create_task(harness.broker.request(_request(handler)))
        await blocking.entered.wait()

        request_id = harness.ids.request_ids[0]
        continuation_id = harness.ids.continuation_ids[0]
        pending = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=_correlation(
                    request_id,
                    continuation_id,
                    _origin(),
                ),
            )
        )
        assert isinstance(pending, InteractionRecord)
        assert pending.request.state is RequestState.PENDING
        assert request_id in harness.broker._admission_cleanups

        await harness.broker.aclose()
        await blocking.stopped.wait()
        request_result = await _task_result(request_task)
        assert isinstance(request_result, CancelledError)
        assert not handler.started.is_set()
        terminal = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=pending.correlation,
            )
        )
        assert isinstance(terminal, InteractionRecord)
        assert terminal.request.state is RequestState.UNAVAILABLE
        assert harness.broker._admission_cleanups == {}
        assert not harness.factory._state.resumers
        assert len(harness.factory._state.admissions) == 1
        await external.aclose()

    run(exercise())


def test_close_cleanup_ignores_denied_public_terminalization_authority() -> (
    None
):
    """Use exact admission authority when every public operation is denied."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        created = await harness.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=_Resumer(),
            )
        )
        assert created.delivery is not None
        record = created.delivery.record
        authorization_calls = len(harness.authorizer.operations)
        harness.authorizer.allowed = False

        await harness.broker.aclose()
        close_operations = harness.authorizer.operations[authorization_calls:]
        assert InteractionOperation.EXPIRE not in close_operations
        assert InteractionOperation.SUPERSEDE not in close_operations
        assert harness.broker._admission_cleanups == {}
        assert not harness.factory._state.resumers

        harness.authorizer.allowed = True
        terminal = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=record.correlation,
            )
        )
        assert isinstance(terminal, InteractionRecord)
        assert terminal.request.state is RequestState.UNAVAILABLE
        await external.aclose()

    run(exercise())


def test_close_cleanup_loses_safely_to_concurrent_external_resolution() -> (
    None
):
    """Accept an external terminal winner and extract its bridge once."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        resumer = _Resumer()
        created = await harness.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=resumer,
            )
        )
        assert created.delivery is not None
        record = created.delivery.record
        blocking = _BlockingAdmissionCleanupStore(harness.store)
        harness.broker._store = cast(InteractionStore, blocking)

        close_task = create_task(harness.broker.aclose())
        await blocking.entered.wait()
        resolved = await external.resolve(_decline(record, "close-race"))
        assert isinstance(resolved, ResolveInteractionApplied)
        await resumer.called.wait()
        blocking.release.set()
        await close_task

        terminal = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=record.correlation,
            )
        )
        assert isinstance(terminal, InteractionRecord)
        assert terminal.request.state is RequestState.DECLINED
        assert len(resumer.notifications) == 1
        assert blocking.calls == 1
        assert harness.broker._admission_cleanups == {}
        assert not harness.factory._state.resumers
        await external.aclose()

    run(exercise())


def test_failed_close_cleanup_retains_authority_for_explicit_retry() -> None:
    """Retry a failed close barrier without losing cleanup authority."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        created = await harness.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=_Resumer(),
            )
        )
        assert created.delivery is not None
        record = created.delivery.record
        transient = _TransientAdmissionCleanupStore(harness.store)
        harness.broker._store = cast(InteractionStore, transient)

        with pytest.raises(
            RuntimeError,
            match="transient admission cleanup failure",
        ):
            await harness.broker.aclose()
        assert record.request.request_id in harness.broker._admission_cleanups
        assert not cast(Any, harness.store)._closed
        first_exclusion = harness.broker._close_excluded_task
        assert first_exclusion is current_task()

        transient.retry_release.clear()
        retry = create_task(harness.broker.aclose())
        await transient.retry_entered.wait()
        assert harness.broker._close_excluded_task is retry
        transient.retry_release.set()
        await retry
        assert transient.calls == 2
        assert harness.broker._admission_cleanups == {}
        assert harness.broker._close_excluded_task is None
        terminal = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=record.correlation,
            )
        )
        assert isinstance(terminal, InteractionRecord)
        assert terminal.request.state is RequestState.UNAVAILABLE
        await harness.broker.aclose()
        assert transient.calls == 2
        await external.aclose()

    run(exercise())


def test_malformed_close_cleanup_proof_retains_authority_for_retry() -> None:
    """Reject an unsealed store response without abandoning authority."""

    async def exercise() -> None:
        harness = await _harness()
        created = await harness.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=_Resumer(),
            )
        )
        assert created.delivery is not None
        record = created.delivery.record
        malformed = _MalformedAdmissionCleanupStore(harness.store)
        harness.broker._store = cast(InteractionStore, malformed)

        with pytest.raises(InputValidationError) as error:
            await harness.broker.aclose()
        assert error.value.path == "broker.store.cleanup_admission"
        assert record.request.request_id in harness.broker._admission_cleanups

        await harness.broker.aclose()
        assert malformed.calls == 2
        assert harness.broker._admission_cleanups == {}

    run(exercise())


def test_failed_close_retains_original_pumps_until_successful_retry() -> None:
    """Join every pre-failure owned task at the successful close barrier."""

    async def exercise() -> None:
        observer = _Observer(block_first=True)
        harness = await _harness(observer=observer)
        created = await harness.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=_Resumer(),
            )
        )
        assert created.delivery is not None
        await observer.entered.wait()
        observer_task = harness.broker._observer_task
        deadline_task = harness.broker._deadline_task
        assert observer_task is not None
        assert deadline_task is not None
        assert not observer_task.done()
        assert not deadline_task.done()

        malformed = _MalformedAdmissionCleanupStore(harness.store)
        harness.broker._store = cast(InteractionStore, malformed)
        with pytest.raises(InputValidationError):
            await harness.broker.aclose()

        assert harness.broker._observer_task is None
        assert harness.broker._deadline_task is None
        assert observer_task in harness.broker._close_owned_tasks
        assert deadline_task in harness.broker._close_owned_tasks
        assert not observer_task.done()
        assert not deadline_task.done()

        await harness.broker.aclose()

        assert malformed.calls == 2
        assert observer_task.done()
        assert deadline_task.done()
        assert harness.broker._close_owned_tasks == ()

    run(exercise())


@pytest.mark.parametrize("malformed_cleanup", (False, True))
def test_failed_create_cleanup_retains_authority_until_close_retry(
    malformed_cleanup: bool,
) -> None:
    """Keep capability authority when create-error cleanup cannot run."""

    async def exercise() -> None:
        harness = await _harness()
        failing = _FailingAdmissionCreateStore(
            harness.store,
            malformed_cleanup=malformed_cleanup,
        )
        harness.broker._store = cast(InteractionStore, failing)

        with pytest.raises(
            RuntimeError,
            match="sealed admission create failure",
        ):
            await harness.broker.request(_request(_BlockingHandler()))
        request_id = harness.ids.request_ids[0]
        assert request_id in harness.broker._admission_cleanups
        assert failing.cleanup_calls == 1

        await harness.broker.aclose()
        assert failing.cleanup_calls == 2
        assert harness.broker._admission_cleanups == {}
        assert harness.factory._state.admissions == {}

    run(exercise())


def test_close_terminalizes_queued_admission_and_removes_store_bridge() -> (
    None
):
    """Terminalize every admitted request before closing its store handle."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        blocking = _BlockingCloseStore(harness.store)
        harness.broker._store = cast(InteractionStore, blocking)
        active_handler = _BlockingHandler()
        active_resumer = _Resumer()
        queued_handler = _BlockingHandler()
        queued_resumer = _Resumer()
        active_task = create_task(
            harness.broker.request(
                _request(
                    active_handler,
                    resumer=active_resumer,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                )
            )
        )
        await active_handler.started.wait()
        queued_task = create_task(
            harness.broker.request(
                _request(
                    queued_handler,
                    resumer=queued_resumer,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=30,
                    reason="Close this admitted queued request.",
                )
            )
        )
        await _wait_until(
            lambda: any(
                len(queue) == 2
                for queue in harness.broker._root_queues.values()
            )
        )
        active_correlation = _correlation(
            harness.ids.request_ids[0],
            harness.ids.continuation_ids[0],
            _origin(),
        )
        queued_correlation = _correlation(
            harness.ids.request_ids[1],
            harness.ids.continuation_ids[1],
            _origin(),
        )
        active_before = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=active_correlation,
            )
        )
        queued_before = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=queued_correlation,
            )
        )
        assert isinstance(active_before, InteractionRecord)
        assert isinstance(queued_before, InteractionRecord)

        first_close = create_task(harness.broker.aclose())
        await blocking.entered.wait()
        cancelled_close = create_task(harness.broker.aclose())
        await _yield_once()
        cancelled_close.cancel()
        with pytest.raises(CancelledError):
            await cancelled_close
        final_close = create_task(harness.broker.aclose())

        active_record = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=active_correlation,
            )
        )
        queued_record = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=queued_correlation,
            )
        )
        assert isinstance(active_record, InteractionRecord)
        assert isinstance(queued_record, InteractionRecord)
        assert active_record.request.state is RequestState.UNAVAILABLE
        assert queued_record.request.state is RequestState.UNAVAILABLE
        assert active_record.request.state_revision == (
            active_before.request.state_revision + 1
        )
        assert queued_record.request.state_revision == (
            queued_before.request.state_revision + 1
        )
        assert harness.broker._admission_cleanups == {}
        assert harness.broker._request_resumers == {}
        assert harness.broker._resumers == {}
        assert harness.broker._resumer_tasks == {}
        assert not harness.factory._state.resumers
        assert (
            await harness.broker._start_resumption(
                active_record.request.request_id,
                harness.broker._resumption_notification(active_record),
            )
            is None
        )
        notification_counts = (
            len(active_resumer.notifications),
            len(queued_resumer.notifications),
        )

        later = await external.cancel(
            CancelInteractionCommand(
                actor=_actor(),
                correlation=queued_correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=(queued_record.request.state_revision),
            )
        )
        assert isinstance(later, CancelInteractionRejected)
        await _yield_once()
        assert (
            len(active_resumer.notifications),
            len(queued_resumer.notifications),
        ) == notification_counts
        queued_after = await external.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(),
                correlation=queued_correlation,
            )
        )
        assert queued_after == queued_record

        blocking.release.set()
        await first_close
        await final_close
        await harness.broker.aclose()
        assert (
            await harness.broker._start_resumption(
                active_record.request.request_id,
                harness.broker._resumption_notification(active_record),
            )
            is None
        )
        await active_handler.stopped.wait()
        active_result = await active_task
        assert active_result.delivery is not None
        assert (
            active_result.delivery.record.request.state
            is RequestState.UNAVAILABLE
        )
        queued_result = await _task_result(queued_task)
        assert isinstance(queued_result, InteractionStoreClosedError)
        assert not queued_handler.started.is_set()
        assert blocking.calls == 1
        await external.aclose()

    run(exercise())


def test_concurrent_close_callers_share_one_completion_barrier() -> None:
    """Make every concurrent close await the same store-close completion."""

    async def exercise() -> None:
        harness = await _harness()
        blocking = _BlockingCloseStore(harness.store)
        harness.broker._store = cast(InteractionStore, blocking)
        await harness.broker.heartbeat()

        first = create_task(harness.broker.aclose())
        await blocking.entered.wait()
        second = create_task(harness.broker.aclose())
        await _yield_once()
        assert not first.done()
        assert not second.done()
        blocking.release.set()
        await first
        await second
        assert blocking.calls == 1

    run(exercise())


def test_handler_that_initiates_close_awaits_full_store_barrier() -> None:
    """Exclude the initiating handler from every close-driven join."""

    async def exercise() -> None:
        harness = await _harness()
        blocking = _BlockingCloseStore(harness.store)
        harness.broker._store = cast(InteractionStore, blocking)
        handler = _InitiatingCloseHandler()
        handler.broker = harness.broker
        request_task = create_task(harness.broker.request(_request(handler)))

        await handler.close_entered.wait()
        await blocking.entered.wait()
        assert not handler.close_completed.is_set()
        assert not handler.stopped.is_set()
        assert not request_task.done()
        close_task = harness.broker._close_task
        assert close_task is not None
        assert not close_task.done()

        blocking.release.set()
        await handler.close_completed.wait()
        await close_task
        request_outcome = await _task_result(request_task)

        assert handler.stopped.is_set()
        assert not isinstance(request_outcome, CancelledError)
        assert blocking.calls == 1
        assert harness.broker._close_excluded_task is None
        assert harness.broker._handler_tasks == {}
        await harness.broker.aclose()

    run(exercise())


def test_handler_cancelled_by_external_close_joins_without_cycle() -> None:
    """Let one owned handler cooperate with the close already joining it."""

    async def exercise() -> None:
        harness = await _harness()
        handler = _ClosingOnCancellationHandler()
        handler.broker = harness.broker
        resumer = _Resumer()
        request_task = create_task(
            harness.broker.request(_request(handler, resumer=resumer))
        )
        await handler.started.wait()

        closing = create_task(harness.broker.aclose())
        await handler.close_entered.wait()
        await _wait_until(handler.close_completed.is_set)
        await closing
        result = await request_task

        assert handler.stopped.is_set()
        assert result.delivery is not None
        assert result.delivery.record.request.state is RequestState.UNAVAILABLE
        assert len(resumer.notifications) == 1
        assert harness.broker._admission_cleanups == {}
        assert harness.broker._handler_tasks == {}
        assert harness.broker._resumer_tasks == {}

    run(exercise())


def test_observer_can_close_broker_reentrantly_without_deadlock() -> None:
    """Exclude the observer's current task from its own close join."""

    async def exercise() -> None:
        observer = _ClosingObserver()
        harness = await _harness(observer=cast(_Observer, observer))
        observer.broker = harness.broker
        await harness.broker.heartbeat()
        await observer.entered.wait()
        await observer.completed.wait()
        task = observer.task
        assert task is not None
        await _wait_until(task.done)
        assert harness.broker.is_closed is True
        await harness.broker.aclose()

    run(exercise())


def test_resumer_can_close_broker_reentrantly_without_task_leak() -> None:
    """Exclude the tracked resumer from its own close join."""

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        resumer = _ClosingResumer()
        resumer.broker = harness.broker
        created = await harness.broker.request(
            _request(_LossHandler(InputHandlerDetached()), resumer=resumer)
        )
        assert created.delivery is not None
        record = created.delivery.record

        cancelled = await external.cancel(
            CancelInteractionCommand(
                actor=_actor(),
                correlation=record.correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=record.request.state_revision,
            )
        )
        assert isinstance(cancelled, CancelInteractionApplied)
        await resumer.called.wait()
        await resumer.completed.wait()
        await _wait_until(lambda: not harness.broker._resumer_tasks)
        assert harness.broker.is_closed is True
        await harness.broker.aclose()
        await external.aclose()

    run(exercise())


def test_paused_store_handoff_allows_reentrant_resumer_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unblock a callback-owned close through the exact store handoff."""
    bridge_entered, bridge_release = _pause_store_bound_resumer(monkeypatch)

    async def exercise() -> None:
        harness = await _harness()
        external = await harness.factory.open()
        resumer = _ClosingResumer()
        resumer.broker = harness.broker
        created = await harness.broker.request(
            _request(
                _LossHandler(InputHandlerDetached()),
                resumer=resumer,
            )
        )
        assert created.delivery is not None
        record = created.delivery.record

        external_mutation = create_task(
            external.cancel(
                CancelInteractionCommand(
                    actor=_actor(),
                    correlation=record.correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=record.request.state_revision,
                )
            )
        )
        await bridge_entered.wait()
        await resumer.called.wait()
        await _wait_until(lambda: harness.broker._closing_settlements)
        assert not resumer.completed.is_set()
        assert not external_mutation.done()

        bridge_release.set()
        mutation = await external_mutation
        assert isinstance(mutation, CancelInteractionApplied)
        await resumer.completed.wait()
        await _wait_until(lambda: not harness.broker._resumer_tasks)

        assert harness.broker.is_closed is True
        assert harness.broker._admission_cleanups == {}
        assert harness.broker._resumer_handoffs == {}
        await harness.broker.aclose()
        await external.aclose()

    run(exercise())


def test_close_tracks_and_drains_cancellation_cleanup_task() -> None:
    """Cancel and join a broker-created owning-cancellation cleanup task."""

    async def exercise() -> None:
        harness = await _harness()
        handler = _BlockingHandler()
        request_task = create_task(harness.broker.request(_request(handler)))
        await handler.started.wait()
        blocking = _BlockingLookupStore(harness.store)
        harness.broker._store = cast(InteractionStore, blocking)

        request_task.cancel()
        await blocking.entered.wait()
        assert len(harness.broker._cleanup_tasks) == 1
        await harness.broker.aclose()
        await blocking.stopped.wait()
        with pytest.raises(CancelledError):
            await request_task
        assert harness.broker._cleanup_tasks == set()

    run(exercise())


async def _wait_for_record(
    broker: AsyncInteractionBroker,
    correlation: InteractionCorrelation,
) -> None:
    """Wait until one already-minted record is durably visible."""
    for _ in range(100):
        projection = await broker.inspect(
            ScopedInteractionLookup(actor=_actor(), correlation=correlation)
        )
        if isinstance(projection, InteractionRecord):
            return
        await _yield_once()
    raise AssertionError("record did not become visible")


async def _gather_request_results(
    *tasks: Task[InteractionRequestResult],
) -> tuple[InteractionRequestResult, ...]:
    """Await a fixed tuple of request tasks with exact result typing."""
    results: list[InteractionRequestResult] = []
    for task in tasks:
        results.append(await task)
    return tuple(results)


async def _task_result(task: Task[InteractionRequestResult]) -> object:
    """Return a task result or its raised exception as one object."""
    try:
        return await task
    except BaseException as error:
        return error
