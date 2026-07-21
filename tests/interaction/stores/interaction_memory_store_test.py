"""Exercise in-memory interaction-store sharing and deterministic races."""

from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Event,
    Task,
    all_tasks,
    create_task,
    current_task,
    gather,
    get_running_loop,
    run,
)
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from avalan.interaction import (
    MAX_STATE_REVISION,
    AcquireControllerActivity,
    ActiveControlLeaseNonce,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancelInteractionCommand,
    CancelInteractionRejected,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    ControllerActivityApplied,
    ControllerActivityRejected,
    ControllerId,
    CreateInteractionApplied,
    CreateInteractionCommand,
    CreateInteractionRejected,
    DeadlineScheduleRevision,
    DetachInteractionCommand,
    DueInteractionsRejected,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputErrorCode,
    InputRequest,
    InputRequestId,
    InputResumer,
    InputResumptionNotification,
    InputValidationError,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionBranchRegistration,
    InteractionBranchRegistrationApplied,
    InteractionBranchRegistrationRejected,
    InteractionBranchRoot,
    InteractionBranchRootLookup,
    InteractionCorrelation,
    InteractionDisclosure,
    InteractionExecutionScope,
    InteractionIdFactory,
    InteractionNotFoundError,
    InteractionOperation,
    InteractionPolicy,
    InteractionPresentationApplied,
    InteractionPresentationRejected,
    InteractionRecord,
    InteractionReplayKind,
    InteractionStoreClosedError,
    InteractionStoreGeneration,
    InteractionStoreReplayed,
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
    ResolutionDecisionStage,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    RunId,
    ScopeCancellationApplied,
    ScopeCancellationRejected,
    ScopedInteractionLookup,
    ScopeSupersessionApplied,
    ScopeSupersessionRejected,
    StateRevision,
    StreamSessionId,
    SupersedeInteractionScopeCommand,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TaskInputClassifier,
    TerminalizeDueInteractionsCommand,
    TerminalizeInteractionApplied,
    TerminalizeInteractionCommand,
    TerminalizeInteractionRejected,
    TerminalizeInteractionScopeCommand,
    TextAnswer,
    TextQuestion,
    TrustedDefaultResolutionApplied,
    TurnId,
    UserId,
    WaitForDeadlineChangeCommand,
    WaitForInteractionChangeCommand,
    create_input_request,
)
from avalan.interaction.policy import (
    InteractionClock,
    InteractionRequestAuthorizationTarget,
)
from avalan.interaction.store import (
    TrustedDefaultResolutionCommand,
    TrustedDefaultResolutionRequest,
    _InteractionAdmissionCleanupCommand,
    _InteractionAdmissionCleanupDisposition,
    _InteractionAdmissionCleanupResult,
    _InteractionAdmissionCreateCommand,
    _new_interaction_admission_commands,
    _new_interaction_store_backing,
    _new_trusted_default_resolution_command,
    _snapshot_interaction_store_backing,
)
from avalan.interaction.stores import (
    InteractionResumptionDeliveryError,
    MemoryInteractionStore,
    MemoryInteractionStoreFactory,
)

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


async def _yield_once() -> None:
    """Yield one explicit event-loop scheduling barrier."""
    ready = get_running_loop().create_future()
    get_running_loop().call_soon(ready.set_result, None)
    await ready


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    turns: int = 100,
) -> None:
    """Wait for a deterministic in-process predicate without sleeping."""
    for _ in range(turns):
        if predicate():
            return
        await _yield_once()
    raise AssertionError("event-loop predicate did not become true")


class _Clock(InteractionClock):
    """Return trusted observations and optionally block one read."""

    def __init__(self) -> None:
        self.wall_time = _NOW
        self.monotonic_seconds = 0.0
        self.read_count = 0
        self.active_reads = 0
        self.maximum_active_reads = 0
        self._block_next = False
        self.entered = Event()
        self.release = Event()
        self.release.set()

    async def read(self) -> InteractionTime:
        """Return one coherent observation after any configured barrier."""
        self.read_count += 1
        self.active_reads += 1
        self.maximum_active_reads = max(
            self.maximum_active_reads,
            self.active_reads,
        )
        try:
            if self._block_next:
                self._block_next = False
                self.entered.set()
                await self.release.wait()
            return InteractionTime.from_clock(
                wall_time=self.wall_time,
                monotonic_seconds=self.monotonic_seconds,
            )
        finally:
            self.active_reads -= 1

    async def wait_until(self, monotonic_deadline: float) -> None:
        """Reject scheduler use in a store that owns no scheduler tasks."""
        raise AssertionError(
            f"unexpected scheduler wait: {monotonic_deadline}"
        )

    def block_next_read(self) -> None:
        """Block exactly the next trusted observation."""
        self._block_next = True
        self.entered.clear()
        self.release.clear()


class _InvalidObservationClock(_Clock):
    """Return one malformed trusted observation for rejection coverage."""

    async def read(self) -> InteractionTime:
        """Return a value outside the trusted clock capability."""
        return cast(InteractionTime, object())


class _IdFactory(InteractionIdFactory):
    """Mint deterministic server-owned identifiers."""

    def __init__(self) -> None:
        self.sequence = 0

    def _next(self, kind: str) -> str:
        self.sequence += 1
        return f"memory-{kind}-{self.sequence}"

    async def new_request_id(self) -> InputRequestId:
        """Return one request identifier."""
        return InputRequestId(self._next("request"))

    async def new_continuation_id(self) -> ContinuationId:
        """Return one continuation identifier."""
        return ContinuationId(self._next("continuation"))

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        """Return one resolution idempotency key."""
        return ResolutionIdempotencyKey(self._next("key"))

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        """Return one active-control lease nonce."""
        return ActiveControlLeaseNonce(self._next("lease"))


class _Classifier(TaskInputClassifier):
    """Allow exact task-input classifications behind an optional barrier."""

    def __init__(self, policy: InteractionPolicy) -> None:
        self.policy = policy
        self.calls = 0
        self.block_after = 0
        self.entered = Event()
        self.release = Event()
        self.release.set()

    def block(self, *, after_calls: int) -> None:
        """Block once the requested number of calls have entered."""
        self.block_after = after_calls
        self.entered.clear()
        self.release.clear()

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Return an exact allow decision after any configured barrier."""
        self.calls += 1
        if self.block_after:
            if self.calls >= self.block_after:
                self.entered.set()
            await self.release.wait()
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=self.policy.task_input_classifier_id,
            classification_id=f"classification-{self.calls}",
            policy_revision=self.policy.task_input_policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


class _ExplodingClassifier(TaskInputClassifier):
    """Fail if deadline precedence incorrectly invokes classification."""

    def __init__(self) -> None:
        self.calls = 0

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Raise on every unexpected classification call."""
        assert isinstance(request, TaskInputClassificationRequest)
        self.calls += 1
        raise RuntimeError("classifier must not run for a due request")


class _MalformedClassifier(TaskInputClassifier):
    """Return an untrusted value with the wrong runtime type."""

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Return one malformed classifier output."""
        assert isinstance(request, TaskInputClassificationRequest)
        return cast(TaskInputClassification, object())


class _Authorizer:
    """Echo exact authorization and expose operation barriers."""

    def __init__(self) -> None:
        self.blocked_operations: frozenset[InteractionOperation] = frozenset()
        self.expected_blocked_calls = 0
        self.blocked_calls = 0
        self.entered = Event()
        self.release = Event()
        self.release.set()
        self.seen: dict[InteractionOperation, Event] = {}
        self.calls: list[
            tuple[
                InteractionActor,
                InteractionOperation,
                InteractionAuthorizationTarget,
            ]
        ] = []
        self.allowed = True
        self.denied_operations: set[InteractionOperation] = set()
        self.denied_request_ids: set[InputRequestId] = set()
        self.malformed_operations: set[InteractionOperation] = set()
        self.scope_disclosure = InteractionDisclosure.FULL
        self.request_disclosure = InteractionDisclosure.FULL

    def block(
        self,
        operations: frozenset[InteractionOperation],
        *,
        expected_calls: int,
    ) -> None:
        """Block a bounded set of exact authorization calls."""
        self.blocked_operations = operations
        self.expected_blocked_calls = expected_calls
        self.blocked_calls = 0
        self.entered.clear()
        self.release.clear()

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        """Return one full-disclosure decision bound to exact inputs."""
        self.calls.append((actor, operation, target))
        self.seen.setdefault(operation, Event()).set()
        if operation in self.blocked_operations:
            self.blocked_calls += 1
            if self.blocked_calls == self.expected_blocked_calls:
                self.entered.set()
            await self.release.wait()
        if operation in self.malformed_operations:
            return cast(InteractionAuthorizationDecision, object())
        denied_request = (
            isinstance(target, InteractionRequestAuthorizationTarget)
            and target.request_id in self.denied_request_ids
        )
        allowed = (
            self.allowed
            and operation not in self.denied_operations
            and not denied_request
        )
        disclosure = (
            self.request_disclosure
            if isinstance(target, InteractionRequestAuthorizationTarget)
            else self.scope_disclosure
        )
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=allowed,
            disclosure=(disclosure if allowed else InteractionDisclosure.NONE),
        )


class _Resumer:
    """Record every committed in-process resumption notification."""

    def __init__(self) -> None:
        self.notifications: list[InputResumptionNotification] = []
        self.called = Event()

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        """Record one committed notification."""
        self.notifications.append(notification)
        self.called.set()


class _FailingResumer:
    """Raise one private failure from every delivery attempt."""

    def __init__(self) -> None:
        self.attempts = 0

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        """Raise after recording one attempted notification."""
        assert isinstance(notification, InputResumptionNotification)
        self.attempts += 1
        raise RuntimeError("private callback detail")


class _BlockingResumer:
    """Block one observable callback until deterministic release."""

    def __init__(self) -> None:
        self.notifications: list[InputResumptionNotification] = []
        self.entered = Event()
        self.release = Event()

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        """Record and block one callback invocation."""
        self.notifications.append(notification)
        self.entered.set()
        await self.release.wait()


class _CancellingResumer:
    """Raise callback-local cancellation for failure-continuation coverage."""

    def __init__(self) -> None:
        self.attempts = 0

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        """Raise one callback-local cancellation."""
        assert isinstance(notification, InputResumptionNotification)
        self.attempts += 1
        raise CancelledError()


def _principal(name: str = "owner") -> PrincipalScope:
    return PrincipalScope(user_id=UserId(name))


def _origin(
    *,
    run_id: str = "run",
    branch_id: str = "root",
    parent_branch_id: str | None = None,
    principal: PrincipalScope | None = None,
) -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId(run_id),
        turn_id=TurnId("turn"),
        agent_id=AgentId("agent"),
        branch_id=BranchId(branch_id),
        parent_branch_id=(
            None if parent_branch_id is None else BranchId(parent_branch_id)
        ),
        model_call_id=ModelCallId(f"call-{branch_id}"),
        stream_session_id=StreamSessionId(f"stream-{run_id}"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://memory-test",
            agent_definition_revision="revision-1",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model-1",
            tool_revision="tools-1",
            capability_revision="capabilities-1",
        ),
        principal=principal or _principal(),
    )


def _request(
    name: str,
    *,
    origin: ExecutionOrigin | None = None,
    mode: RequirementMode = RequirementMode.REQUIRED,
    default_value: bool | None = None,
) -> InputRequest:
    return create_input_request(
        request_id=InputRequestId(name),
        continuation_id=ContinuationId(f"continuation-{name}"),
        origin=origin or _origin(),
        mode=mode,
        reason=f"Confirm {name}.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
                default_value=default_value,
            ),
        ),
        created_at=_NOW,
        continuation_ttl_seconds=600,
        advisory_wait_seconds=(
            30 if mode is RequirementMode.ADVISORY else None
        ),
    )


def _text_request(name: str) -> InputRequest:
    return create_input_request(
        request_id=InputRequestId(name),
        continuation_id=ContinuationId(f"continuation-{name}"),
        origin=_origin(),
        mode=RequirementMode.ADVISORY,
        reason=f"Enter {name}.",
        questions=(
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Value?",
                required=True,
            ),
        ),
        created_at=_NOW,
        continuation_ttl_seconds=600,
        advisory_wait_seconds=30,
    )


def _actor(request: InputRequest) -> InteractionActor:
    return InteractionActor(principal=request.origin.principal)


def _answer(
    record: InteractionRecord,
    key: str,
) -> ResolveInteractionCommand:
    return ResolveInteractionCommand(
        actor=_actor(record.request),
        correlation=record.correlation,
        expected_state_revision=StateRevision(record.request.state_revision),
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=AnsweredResolution(
            request_id=record.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=(
                ConfirmationAnswer(
                    question_id=QuestionId("confirm"),
                    provenance=AnswerProvenance.HUMAN,
                    value=True,
                ),
            ),
        ),
    )


def _answer_for_correlation(
    record: InteractionRecord,
    key: str,
    correlation: InteractionCorrelation,
    *,
    expected_state_revision: StateRevision | None = None,
) -> ResolveInteractionCommand:
    original = _answer(record, key)
    return ResolveInteractionCommand(
        actor=original.actor,
        correlation=correlation,
        expected_state_revision=(
            original.expected_state_revision
            if expected_state_revision is None
            else expected_state_revision
        ),
        idempotency_key=original.idempotency_key,
        proposed_resolution=replace(
            original.proposed_resolution,
            request_id=correlation.request_id,
        ),
    )


def _text_answer(
    record: InteractionRecord,
    key: str,
) -> ResolveInteractionCommand:
    return ResolveInteractionCommand(
        actor=_actor(record.request),
        correlation=record.correlation,
        expected_state_revision=record.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=AnsweredResolution(
            request_id=record.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=(
                TextAnswer(
                    question_id=QuestionId("text"),
                    provenance=AnswerProvenance.HUMAN,
                    value="classified value",
                ),
            ),
        ),
    )


def _trusted_default(
    record: InteractionRecord,
    *,
    actor: InteractionActor | None = None,
) -> TrustedDefaultResolutionCommand:
    """Mint one sealed command as the trusted broker boundary."""
    return _new_trusted_default_resolution_command(
        TrustedDefaultResolutionRequest(
            actor=actor or _actor(record.request),
            correlation=record.correlation,
            expected_state_revision=record.request.state_revision,
        )
    )


def _factory(
    *,
    policy: InteractionPolicy | None = None,
    clock: _Clock | None = None,
    authorizer: _Authorizer | None = None,
    classifier: TaskInputClassifier | None = None,
) -> tuple[MemoryInteractionStoreFactory, _Clock, _Authorizer]:
    active_clock = clock or _Clock()
    active_authorizer = authorizer or _Authorizer()
    active_policy = policy or InteractionPolicy()
    factory = MemoryInteractionStoreFactory(
        policy=active_policy,
        clock=active_clock,
        authorizer=active_authorizer,
        id_factory=_IdFactory(),
        classifier=classifier,
    )
    return factory, active_clock, active_authorizer


async def _create(
    store: MemoryInteractionStore,
    request: InputRequest,
    *,
    resumer: InputResumer | None = None,
) -> CreateInteractionApplied:
    result = await store.create(
        CreateInteractionCommand(
            actor=_actor(request),
            request=request,
            resumer=resumer,
        )
    )
    assert isinstance(result, CreateInteractionApplied)
    return result


async def _lookup(
    store: MemoryInteractionStore,
    correlation: InteractionCorrelation,
) -> InteractionRecord:
    projection = await store.lookup_scoped(
        ScopedInteractionLookup(
            actor=InteractionActor(principal=_principal()),
            correlation=correlation,
        )
    )
    assert isinstance(projection, InteractionRecord)
    return projection


def test_reopened_handles_share_records_and_ephemeral_resumers() -> None:
    """Keep inventory and resumers in factory-owned shared state."""

    async def exercise() -> None:
        factory, _, _ = _factory()
        first = await factory.open()
        request = _request("reopened")
        resumer = _Resumer()
        created = await _create(first, request, resumer=resumer)
        await first.aclose()

        reopened = await factory.open()
        assert (
            await _lookup(reopened, created.record.correlation)
            == created.record
        )
        resolved = await reopened.resolve(_answer(created.record, "answer"))
        assert isinstance(resolved, ResolveInteractionApplied)
        assert len(resumer.notifications) == 1

        replay = await reopened.resolve(_answer(created.record, "answer"))
        assert isinstance(replay, InteractionStoreReplayed)
        assert replay.replay_kind is InteractionReplayKind.SAME_KEY
        assert len(resumer.notifications) == 1
        await reopened.aclose()

    run(exercise())


def test_admission_cleanup_is_reopen_safe_auth_independent_and_bounded() -> (
    None
):
    """Retain one opaque tombstone per retained admitted record."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        factory, _, _ = _factory(authorizer=authorizer)
        first = await factory.open()
        request = _request("sealed-cleanup")
        resumer = _Resumer()
        create, cleanup = _new_interaction_admission_commands(
            actor=_actor(request),
            request=request,
            resumer=resumer,
        )
        created = await first.create_admission(create)
        assert isinstance(created, CreateInteractionApplied)
        assert len(factory._state.admissions) == 1
        assert (
            len(
                _snapshot_interaction_store_backing(
                    factory._state.backing
                ).records
            )
            == 1
        )
        await first.aclose()

        reopened = await factory.open()
        authorizer.allowed = False
        authorization_calls = len(authorizer.calls)
        settled = await reopened.cleanup_admission(cleanup)
        assert type(settled) is _InteractionAdmissionCleanupResult
        assert (
            settled.disposition
            is _InteractionAdmissionCleanupDisposition.SETTLED
        )
        assert len(authorizer.calls) == authorization_calls
        assert len(resumer.notifications) == 1
        assert not factory._state.resumers

        repeated = await reopened.cleanup_admission(cleanup)
        assert (
            repeated.disposition
            is _InteractionAdmissionCleanupDisposition.TERMINAL
        )
        assert len(resumer.notifications) == 1
        assert len(factory._state.admissions) == 1
        assert (
            len(
                _snapshot_interaction_store_backing(
                    factory._state.backing
                ).records
            )
            == 1
        )

        authorizer.allowed = True
        record = await _lookup(reopened, created.record.correlation)
        assert record.request.state is RequestState.UNAVAILABLE
        await reopened.aclose()

    run(exercise())


def test_admission_cleanup_rejects_unsealed_commands_and_prunes_orphans() -> (
    None
):
    """Fail malformed calls and bind tombstone lifetime to its record."""

    async def exercise() -> None:
        factory, _, authorizer = _factory()
        store = await factory.open()
        with pytest.raises(InputValidationError):
            await store.create_admission(
                cast(_InteractionAdmissionCreateCommand, object())
            )
        with pytest.raises(InputValidationError):
            await store.cleanup_admission(cast(Any, object()))
        unsealed_create = object.__new__(_InteractionAdmissionCreateCommand)
        with pytest.raises(InputValidationError) as create_error:
            await store.create_admission(unsealed_create)
        assert create_error.value.code is InputErrorCode.FORBIDDEN
        unsealed_cleanup = object.__new__(_InteractionAdmissionCleanupCommand)
        with pytest.raises(InputValidationError) as cleanup_error:
            await store.cleanup_admission(unsealed_cleanup)
        assert cleanup_error.value.code is InputErrorCode.FORBIDDEN
        assert authorizer.calls == []

        request = _request("orphaned-cleanup-binding")
        resumer = _Resumer()
        create, cleanup = _new_interaction_admission_commands(
            actor=_actor(request),
            request=request,
            resumer=resumer,
        )
        unsealed_capability = object.__new__(type(create._capability))
        nested_create = object.__new__(_InteractionAdmissionCreateCommand)
        object.__setattr__(nested_create, "_seal", create._seal)
        object.__setattr__(nested_create, "_command", create._command)
        object.__setattr__(
            nested_create,
            "_capability",
            unsealed_capability,
        )
        with pytest.raises(InputValidationError) as nested_create_error:
            await store.create_admission(nested_create)
        assert nested_create_error.value.path == "admission.create.capability"

        nested_cleanup = object.__new__(_InteractionAdmissionCleanupCommand)
        object.__setattr__(nested_cleanup, "_seal", cleanup._seal)
        object.__setattr__(
            nested_cleanup,
            "_capability",
            unsealed_capability,
        )
        with pytest.raises(InputValidationError) as nested_cleanup_error:
            await store.cleanup_admission(nested_cleanup)
        assert (
            nested_cleanup_error.value.path == "admission.cleanup.capability"
        )
        assert authorizer.calls == []

        created = await store.create_admission(create)
        assert isinstance(created, CreateInteractionApplied)
        assert len(factory._state.admissions) == 1
        assert len(factory._state.resumers) == 1
        factory._state.backing = _new_interaction_store_backing()

        absent = await store.cleanup_admission(cleanup)
        assert (
            absent.disposition
            is _InteractionAdmissionCleanupDisposition.ABSENT
        )
        assert factory._state.admissions == {}
        assert factory._state.resumers == {}
        assert resumer.notifications == []

        repeated = await store.cleanup_admission(cleanup)
        assert (
            repeated.disposition
            is _InteractionAdmissionCleanupDisposition.ABSENT
        )
        assert factory._state.admissions == {}
        assert factory._state.resumers == {}
        assert resumer.notifications == []
        await store.aclose()

    run(exercise())


def test_cancelled_admission_cleanup_drains_bridge_once_before_retry() -> None:
    """Finish bridge extraction despite repeated cleanup cancellation."""

    async def exercise() -> None:
        factory, _, _ = _factory()
        store = await factory.open()
        request = _request("cancelled-sealed-cleanup")
        resumer = _BlockingResumer()
        create, cleanup = _new_interaction_admission_commands(
            actor=_actor(request),
            request=request,
            resumer=resumer,
        )
        created = await store.create_admission(create)
        assert isinstance(created, CreateInteractionApplied)

        operation = create_task(store.cleanup_admission(cleanup))
        await resumer.entered.wait()
        operation.cancel()
        await _yield_once()
        assert not operation.done()
        operation.cancel()
        resumer.release.set()
        with pytest.raises(CancelledError):
            await operation

        assert len(resumer.notifications) == 1
        assert not factory._state.resumers
        record = await _lookup(store, created.record.correlation)
        assert record.request.state is RequestState.UNAVAILABLE
        repeated = await store.cleanup_admission(cleanup)
        assert (
            repeated.disposition
            is _InteractionAdmissionCleanupDisposition.TERMINAL
        )
        assert len(resumer.notifications) == 1
        await store.aclose()

    run(exercise())


def test_cancelled_terminal_handoff_wait_preserves_retry_and_delivery() -> (
    None
):
    """Keep the shared bridge barrier live after one cleanup waiter cancels."""

    async def exercise() -> None:
        factory, _, _ = _factory()
        cleanup_store = await factory.open()
        external_store = await factory.open()
        request = _request("cancelled-terminal-handoff-wait")
        resumer = _BlockingResumer()
        create, cleanup = _new_interaction_admission_commands(
            actor=_actor(request),
            request=request,
            resumer=resumer,
        )
        created = await cleanup_store.create_admission(create)
        assert isinstance(created, CreateInteractionApplied)

        external_resolution = create_task(
            external_store.resolve(_answer(created.record, "terminal-answer"))
        )
        await resumer.entered.wait()
        binding = factory._state.admissions[cleanup._capability]
        assert not binding.handoff.done()

        first_cleanup = create_task(cleanup_store.cleanup_admission(cleanup))
        await _yield_once()
        assert not first_cleanup.done()
        first_cleanup.cancel()
        with pytest.raises(CancelledError):
            await first_cleanup
        assert not binding.handoff.cancelled()

        retry = create_task(cleanup_store.cleanup_admission(cleanup))
        await _yield_once()
        assert not retry.done()
        resumer.release.set()
        resolved = await external_resolution
        assert isinstance(resolved, ResolveInteractionApplied)
        proof = await retry

        assert (
            proof.disposition
            is _InteractionAdmissionCleanupDisposition.TERMINAL
        )
        assert binding.handoff.done()
        assert len(resumer.notifications) == 1
        await cleanup_store.aclose()
        await external_store.aclose()

    run(exercise())


def test_admission_cleanup_and_external_resolution_are_first_winner_safe() -> (
    None
):
    """Commit one terminal winner and extract one bridge under a race."""

    async def exercise() -> None:
        factory, _, _ = _factory()
        cleanup_store = await factory.open()
        external_store = await factory.open()
        request = _request("cleanup-resolution-race")
        resumer = _Resumer()
        create, cleanup = _new_interaction_admission_commands(
            actor=_actor(request),
            request=request,
            resumer=resumer,
        )
        created = await cleanup_store.create_admission(create)
        assert isinstance(created, CreateInteractionApplied)

        cleanup_result, external_result = await gather(
            cleanup_store.cleanup_admission(cleanup),
            external_store.resolve(_answer(created.record, "race-answer")),
        )
        assert cleanup_result.disposition in {
            _InteractionAdmissionCleanupDisposition.SETTLED,
            _InteractionAdmissionCleanupDisposition.TERMINAL,
        }
        assert isinstance(
            external_result,
            (ResolveInteractionApplied, ResolveInteractionRejected),
        )
        record = await _lookup(external_store, created.record.correlation)
        assert record.request.state in {
            RequestState.ANSWERED,
            RequestState.UNAVAILABLE,
        }
        assert len(resumer.notifications) == 1
        repeated = await external_store.cleanup_admission(cleanup)
        assert (
            repeated.disposition
            is _InteractionAdmissionCleanupDisposition.TERMINAL
        )
        assert len(resumer.notifications) == 1
        await cleanup_store.aclose()
        await external_store.aclose()

    run(exercise())


def test_admission_cleanup_observes_deadline_and_lease_equalities() -> None:
    """Apply equal temporal winners before capability unavailability."""

    async def exercise() -> None:
        absolute_factory, absolute_clock, _ = _factory()
        absolute_store = await absolute_factory.open()
        absolute_request = _request("cleanup-absolute-equality")
        absolute_create, absolute_cleanup = (
            _new_interaction_admission_commands(
                actor=_actor(absolute_request),
                request=absolute_request,
                resumer=_Resumer(),
            )
        )
        absolute = await absolute_store.create_admission(absolute_create)
        assert isinstance(absolute, CreateInteractionApplied)
        absolute_clock.wall_time += timedelta(seconds=600)
        absolute_clock.monotonic_seconds += 600
        absolute_result = await absolute_store.cleanup_admission(
            absolute_cleanup
        )
        assert (
            absolute_result.disposition
            is _InteractionAdmissionCleanupDisposition.SETTLED
        )
        assert (
            await _lookup(absolute_store, absolute.record.correlation)
        ).request.state is RequestState.EXPIRED
        await absolute_store.aclose()

        advisory_factory, advisory_clock, _ = _factory()
        advisory_store = await advisory_factory.open()
        advisory_request = _request(
            "cleanup-advisory-equality",
            mode=RequirementMode.ADVISORY,
        )
        advisory_create, advisory_cleanup = (
            _new_interaction_admission_commands(
                actor=_actor(advisory_request),
                request=advisory_request,
                resumer=_Resumer(),
            )
        )
        advisory = await advisory_store.create_admission(advisory_create)
        assert isinstance(advisory, CreateInteractionApplied)
        presented = await advisory_store.mark_presented(
            PresentInteractionCommand(
                actor=_actor(advisory_request),
                correlation=advisory.record.correlation,
                expected_store_revision=advisory.record.store_revision,
            )
        )
        assert isinstance(presented, InteractionPresentationApplied)
        advisory_clock.wall_time += timedelta(seconds=30)
        advisory_clock.monotonic_seconds += 30
        await advisory_store.cleanup_admission(advisory_cleanup)
        assert (
            await _lookup(advisory_store, advisory.record.correlation)
        ).request.state is RequestState.TIMED_OUT
        await advisory_store.aclose()

        lease_factory, lease_clock, _ = _factory()
        lease_store = await lease_factory.open()
        lease_request = _request(
            "cleanup-lease-equality",
            mode=RequirementMode.ADVISORY,
        )
        lease_create, lease_cleanup = _new_interaction_admission_commands(
            actor=_actor(lease_request),
            request=lease_request,
            resumer=_Resumer(),
        )
        lease = await lease_store.create_admission(lease_create)
        assert isinstance(lease, CreateInteractionApplied)
        lease_presented = await lease_store.mark_presented(
            PresentInteractionCommand(
                actor=_actor(lease_request),
                correlation=lease.record.correlation,
                expected_store_revision=lease.record.store_revision,
            )
        )
        assert isinstance(lease_presented, InteractionPresentationApplied)
        acquired = await lease_store.record_activity(
            RecordControllerActivityCommand(
                actor=_actor(lease_request),
                correlation=lease.record.correlation,
                evidence=AcquireControllerActivity(
                    request_id=lease.record.request.request_id,
                    controller_id=ControllerId("cleanup-controller"),
                ),
            )
        )
        assert isinstance(acquired, ControllerActivityApplied)
        lease_clock.wall_time += timedelta(seconds=30)
        lease_clock.monotonic_seconds += 30
        await lease_store.cleanup_admission(lease_cleanup)
        lease_record = await _lookup(lease_store, lease.record.correlation)
        assert lease_record.request.state is RequestState.UNAVAILABLE
        assert (
            lease_record.store_revision == acquired.record.store_revision + 2
        )
        assert lease_record.advisory_wait is not None
        assert lease_record.advisory_wait.controller_id is None
        await lease_store.aclose()

    run(exercise())


def test_resumer_failure_reports_committed_delivery_error_without_retry() -> (
    None
):
    """Return the applied result and report one content-safe failure."""

    async def exercise() -> None:
        contexts: list[dict[str, object]] = []

        def capture(
            loop: AbstractEventLoop,
            context: dict[str, object],
        ) -> None:
            assert loop is get_running_loop()
            contexts.append(context)

        get_running_loop().set_exception_handler(capture)
        factory, _, _ = _factory()
        store = await factory.open()
        resumer = _FailingResumer()
        created = await _create(
            store,
            _request("resumer-failure"),
            resumer=resumer,
        )

        applied = await store.resolve(_answer(created.record, "applied"))
        assert isinstance(applied, ResolveInteractionApplied)
        assert resumer.attempts == 1
        assert len(contexts) == 1
        error = contexts[0]["exception"]
        assert isinstance(error, InteractionResumptionDeliveryError)
        assert "private callback detail" not in str(error)
        assert (
            await _lookup(store, created.record.correlation) == applied.record
        )

        replay = await store.resolve(_answer(created.record, "applied"))
        assert isinstance(replay, InteractionStoreReplayed)
        assert resumer.attempts == 1
        assert len(contexts) == 1
        await store.aclose()

    run(exercise())


def test_cancelled_post_commit_delivery_drains_batch_exactly_once() -> None:
    """Drain shielded resumptions before propagating caller cancellation."""

    async def exercise() -> None:
        contexts: list[dict[str, object]] = []

        def capture(
            loop: AbstractEventLoop,
            context: dict[str, object],
        ) -> None:
            assert loop is get_running_loop()
            contexts.append(context)

        get_running_loop().set_exception_handler(capture)
        factory, _, _ = _factory()
        store = await factory.open()
        blocking = _BlockingResumer()
        cancelling = _CancellingResumer()
        following = _Resumer()
        first = await _create(
            store,
            _request("a-cancelled-delivery", mode=RequirementMode.ADVISORY),
            resumer=blocking,
        )
        second = await _create(
            store,
            _request("b-cancelled-delivery", mode=RequirementMode.ADVISORY),
            resumer=cancelling,
        )
        third = await _create(
            store,
            _request("c-cancelled-delivery", mode=RequirementMode.ADVISORY),
            resumer=following,
        )
        operation = create_task(
            store.terminalize_scope(
                TerminalizeInteractionScopeCommand(
                    actor=_actor(first.record.request),
                    scope=InteractionExecutionScope(run_id=RunId("run")),
                    provenance=AnswerProvenance.HUMAN,
                )
            )
        )
        await blocking.entered.wait()

        operation.cancel()
        await _yield_once()
        assert not operation.done()
        assert not following.notifications
        operation.cancel()
        await _yield_once()
        assert not operation.done()
        blocking.release.set()
        with pytest.raises(CancelledError):
            await operation

        assert len(blocking.notifications) == 1
        assert cancelling.attempts == 1
        assert len(following.notifications) == 1
        assert len(contexts) == 1
        assert not store._state.resumers
        assert (
            await _lookup(store, first.record.correlation)
        ).request.resolution is not None
        assert (
            await _lookup(store, second.record.correlation)
        ).request.resolution is not None
        assert (
            await _lookup(store, third.record.correlation)
        ).request.resolution is not None
        await _yield_once()
        assert all(task is current_task() for task in all_tasks())
        await store.aclose()

    run(exercise())


def test_authorization_and_disclosure_paths_fail_closed() -> None:
    """Cover malformed, denied, narrowed, and content-safe projections."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        factory, _, _ = _factory(authorizer=authorizer)
        store = await factory.open()
        denied_request = _request("denied-create")
        authorizer.denied_operations.add(InteractionOperation.CREATE)
        denied = await store.create(
            CreateInteractionCommand(
                actor=_actor(denied_request),
                request=denied_request,
            )
        )
        assert isinstance(denied, CreateInteractionRejected)

        authorizer.denied_operations.clear()
        authorizer.malformed_operations.add(InteractionOperation.CREATE)
        malformed_request = _request("malformed-create")
        malformed = await store.create(
            CreateInteractionCommand(
                actor=_actor(malformed_request),
                request=malformed_request,
            )
        )
        assert isinstance(malformed, CreateInteractionRejected)
        authorizer.malformed_operations.clear()

        created = await _create(
            store,
            _request("disclosure", mode=RequirementMode.ADVISORY),
        )
        query = ScopedInteractionLookup(
            actor=_actor(created.record.request),
            correlation=created.record.correlation,
        )
        authorizer.request_disclosure = InteractionDisclosure.TERMINAL_METADATA
        assert await store.lookup_scoped(query) is None

        listing = ListInteractionsCommand(
            actor=_actor(created.record.request),
            scope=InteractionExecutionScope(run_id=RunId("run")),
        )
        assert await store.list_scoped(listing) == ()
        authorizer.request_disclosure = InteractionDisclosure.FULL
        assert await store.list_scoped(listing) == (created.record,)

        authorizer.denied_request_ids.add(created.record.request.request_id)
        assert await store.list_scoped(listing) == ()
        authorizer.denied_request_ids.clear()
        authorizer.denied_operations.add(InteractionOperation.LIST)
        assert await store.list_scoped(listing) == ()
        authorizer.denied_operations.clear()

        resolved = await store.resolve(_answer(created.record, "disclosed"))
        assert isinstance(resolved, ResolveInteractionApplied)
        authorizer.scope_disclosure = InteractionDisclosure.TERMINAL_METADATA
        authorizer.request_disclosure = InteractionDisclosure.FULL
        metadata = await store.list_scoped(listing)
        assert len(metadata) == 1
        assert isinstance(metadata[0], InteractionTerminalMetadata)
        authorizer.scope_disclosure = InteractionDisclosure.FULL
        authorizer.request_disclosure = InteractionDisclosure.TERMINAL_METADATA
        metadata = await store.list_scoped(listing)
        assert len(metadata) == 1
        assert isinstance(metadata[0], InteractionTerminalMetadata)
        await store.aclose()

    run(exercise())


def test_trusted_default_requires_sealed_authorized_host_request() -> None:
    """Reject forged or unauthorized trusted-default store commands."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        factory, _, _ = _factory(authorizer=authorizer)
        store = await factory.open()
        created = await _create(
            store,
            _request(
                "trusted-authority",
                mode=RequirementMode.ADVISORY,
                default_value=False,
            ),
        )
        request = TrustedDefaultResolutionRequest(
            actor=_actor(created.record.request),
            correlation=created.record.correlation,
            expected_state_revision=created.record.request.state_revision,
        )
        with pytest.raises(InputValidationError) as forged:
            TrustedDefaultResolutionCommand(
                request=request,
                _token=object(),
            )
        assert forged.value.code is InputErrorCode.FORBIDDEN

        command = _new_trusted_default_resolution_command(request)
        authorizer.denied_operations.add(InteractionOperation.TRUSTED_DEFAULT)
        with pytest.raises(InteractionNotFoundError):
            await store.resolve_trusted_default(command)
        stored = await _lookup(store, created.record.correlation)
        assert stored.request.resolution is None
        assert authorizer.calls[-2][1] is InteractionOperation.TRUSTED_DEFAULT

        authorizer.denied_operations.clear()
        applied = await store.resolve_trusted_default(command)
        assert isinstance(applied, TrustedDefaultResolutionApplied)
        await store.aclose()

    run(exercise())


def test_branch_root_lookup_fails_closed_for_graph_corruption_and_toctou() -> (
    None
):
    """Hide missing or corrupt ancestry and reject an authorization race."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        factory, _, _ = _factory(authorizer=authorizer)
        first = await factory.open()
        second = await factory.open()
        seed = _request("branch-root-seed")
        actor = _actor(seed)

        def branch_command(
            child: str,
            parent: str,
        ) -> RegisterInteractionBranchCommand:
            return RegisterInteractionBranchCommand(
                actor=actor,
                registration=InteractionBranchRegistration(
                    run_id=seed.origin.run_id,
                    branch_id=BranchId(child),
                    parent_branch_id=BranchId(parent),
                    principal=actor.principal,
                ),
            )

        child = await first.register_branch(branch_command("C", "B"))
        assert isinstance(child, InteractionBranchRegistrationApplied)
        query = InteractionBranchRootLookup(
            actor=actor,
            run_id=seed.origin.run_id,
            branch_id=BranchId("C"),
        )
        assert await first.lookup_branch_root(query) == InteractionBranchRoot(
            run_id=seed.origin.run_id,
            branch_id=BranchId("C"),
            root_branch_id=BranchId("B"),
        )

        authorizer.block(
            frozenset({InteractionOperation.INSPECT_BRANCH}),
            expected_calls=1,
        )
        raced_lookup = create_task(first.lookup_branch_root(query))
        await authorizer.entered.wait()
        parent = await second.register_branch(branch_command("B", "A"))
        assert isinstance(parent, InteractionBranchRegistrationApplied)
        authorizer.release.set()
        assert await raced_lookup is None
        expected = InteractionBranchRoot(
            run_id=seed.origin.run_id,
            branch_id=BranchId("C"),
            root_branch_id=BranchId("A"),
        )
        assert await first.lookup_branch_root(query) == expected
        assert await second.lookup_branch_root(query) == expected

        authorizer.block(
            frozenset({InteractionOperation.INSPECT_BRANCH}),
            expected_calls=2,
        )
        missing_lookup = create_task(
            first.lookup_branch_root(
                replace(query, branch_id=BranchId("missing"))
            )
        )
        foreign_lookup = create_task(
            first.lookup_branch_root(
                replace(
                    query,
                    actor=InteractionActor(principal=_principal("intruder")),
                )
            )
        )
        await authorizer.entered.wait()
        assert not missing_lookup.done()
        assert not foreign_lookup.done()
        authorizer.release.set()
        assert await missing_lookup is None
        assert await foreign_lookup is None

        snapshot = _snapshot_interaction_store_backing(first._state.backing)
        original = snapshot.branch_records
        object.__setattr__(
            snapshot,
            "branch_records",
            tuple(
                record
                for record in original
                if record.registration.branch_id != BranchId("C")
            ),
        )
        assert await first.lookup_branch_root(query) is None

        cyclic = tuple(
            (
                replace(
                    record,
                    registration=replace(
                        record.registration,
                        parent_branch_id=BranchId("C"),
                    ),
                )
                if record.registration.branch_id == BranchId("B")
                else record
            )
            for record in original
        )
        object.__setattr__(snapshot, "branch_records", cyclic)
        assert await first.lookup_branch_root(query) is None

        object.__setattr__(
            snapshot, "branch_records", original + (original[0],)
        )
        assert await first.lookup_branch_root(query) is None
        object.__setattr__(snapshot, "branch_records", (object(),))
        assert await first.lookup_branch_root(query) is None

        object.__setattr__(snapshot, "branch_records", original)
        authorizer.malformed_operations.add(
            InteractionOperation.INSPECT_BRANCH
        )
        assert await first.lookup_branch_root(query) is None
        assert authorizer.calls[-1] == (
            actor,
            InteractionOperation.INSPECT_BRANCH,
            query.authorization_target,
        )
        await first.aclose()
        await second.aclose()

    run(exercise())


def test_missing_and_out_of_scope_requests_have_lookup_timing_parity() -> None:
    """Scope-filter identities before any request authorization await."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        factory, _, _ = _factory(authorizer=authorizer)
        store = await factory.open()
        created = await _create(store, _request("non-enumerating"))
        missing = replace(
            created.record.correlation,
            request_id=InputRequestId("non-enumerating-missing"),
        )
        intruder = InteractionActor(principal=_principal("intruder"))
        calls_before = len(authorizer.calls)
        authorizer.block(
            frozenset(
                {
                    InteractionOperation.INSPECT,
                    InteractionOperation.RESOLVE,
                }
            ),
            expected_calls=1,
        )

        missing_lookup = create_task(
            store.lookup_scoped(
                ScopedInteractionLookup(
                    actor=_actor(created.record.request),
                    correlation=missing,
                )
            )
        )
        foreign_lookup = create_task(
            store.lookup_scoped(
                ScopedInteractionLookup(
                    actor=intruder,
                    correlation=created.record.correlation,
                )
            )
        )
        missing_resolution = create_task(
            store.resolve(
                _answer_for_correlation(
                    created.record,
                    "missing-scope",
                    missing,
                )
            )
        )
        foreign_resolution = create_task(
            store.resolve(
                replace(
                    _answer(created.record, "foreign-scope"),
                    actor=intruder,
                )
            )
        )
        await _wait_until(
            lambda: all(
                task.done()
                for task in (
                    missing_lookup,
                    foreign_lookup,
                    missing_resolution,
                    foreign_resolution,
                )
            )
        )
        assert not authorizer.entered.is_set()
        assert len(authorizer.calls) == calls_before
        authorizer.release.set()

        assert await missing_lookup is None
        assert await foreign_lookup is None
        missing_result = await missing_resolution
        foreign_result = await foreign_resolution
        assert isinstance(missing_result, ResolveInteractionRejected)
        assert isinstance(foreign_result, ResolveInteractionRejected)
        assert missing_result.error == foreign_result.error
        assert missing_result.decision_stage == foreign_result.decision_stage
        await store.aclose()

    run(exercise())


def test_superseded_terminalization_uses_supersede_authorization() -> None:
    """Authorize explicit supersession independently from expiry."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        factory, _, _ = _factory(authorizer=authorizer)
        store = await factory.open()
        created = await _create(
            store,
            _request("supersede-authorization", mode=RequirementMode.ADVISORY),
        )
        command = TerminalizeInteractionCommand(
            actor=_actor(created.record.request),
            correlation=created.record.correlation,
            status=ResolutionStatus.SUPERSEDED,
            provenance=AnswerProvenance.HUMAN,
        )
        authorizer.denied_operations.add(InteractionOperation.SUPERSEDE)
        rejected = await store.terminalize(command)
        assert isinstance(rejected, TerminalizeInteractionRejected)
        assert authorizer.calls[-1][1] is InteractionOperation.SUPERSEDE
        assert (
            await _lookup(store, created.record.correlation)
        ).request.resolution is None
        await store.aclose()

    run(exercise())


def test_request_mutation_variants_and_typed_rejections() -> None:
    """Exercise presentation, activity, default, cancel, and terminalize."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        factory, _, _ = _factory(authorizer=authorizer)
        store = await factory.open()

        presentation = await _create(
            store,
            _request("presentation", mode=RequirementMode.ADVISORY),
        )
        missing_correlation = replace(
            presentation.record.correlation,
            request_id=InputRequestId("missing"),
        )
        missing_present = await store.mark_presented(
            PresentInteractionCommand(
                actor=_actor(presentation.record.request),
                correlation=missing_correlation,
                expected_store_revision=presentation.record.store_revision,
            )
        )
        assert isinstance(missing_present, InteractionPresentationRejected)
        authorizer.denied_operations.add(InteractionOperation.DELIVER)
        denied_present = await store.mark_presented(
            PresentInteractionCommand(
                actor=_actor(presentation.record.request),
                correlation=presentation.record.correlation,
                expected_store_revision=presentation.record.store_revision,
            )
        )
        assert isinstance(denied_present, InteractionPresentationRejected)
        authorizer.denied_operations.clear()
        wrong_correlation = replace(
            presentation.record.correlation,
            turn_id=TurnId("wrong-turn"),
        )
        wrong_present = await store.mark_presented(
            PresentInteractionCommand(
                actor=_actor(presentation.record.request),
                correlation=wrong_correlation,
                expected_store_revision=presentation.record.store_revision,
            )
        )
        assert isinstance(wrong_present, InteractionPresentationRejected)
        presented = await store.mark_presented(
            PresentInteractionCommand(
                actor=_actor(presentation.record.request),
                correlation=presentation.record.correlation,
                expected_store_revision=presentation.record.store_revision,
            )
        )
        assert isinstance(presented, InteractionPresentationApplied)
        repeated = await store.mark_presented(
            PresentInteractionCommand(
                actor=_actor(presentation.record.request),
                correlation=presentation.record.correlation,
                expected_store_revision=presentation.record.store_revision,
            )
        )
        assert isinstance(repeated, InteractionPresentationRejected)
        detached = await store.mark_detached(
            DetachInteractionCommand(
                actor=_actor(presentation.record.request),
                correlation=presentation.record.correlation,
                expected_store_revision=presented.record.store_revision,
            )
        )
        assert isinstance(detached, InteractionPresentationApplied)

        activity_request = await _create(
            store,
            _request("activity", mode=RequirementMode.ADVISORY),
        )
        activity_presented = await store.mark_presented(
            PresentInteractionCommand(
                actor=_actor(activity_request.record.request),
                correlation=activity_request.record.correlation,
                expected_store_revision=activity_request.record.store_revision,
            )
        )
        assert isinstance(activity_presented, InteractionPresentationApplied)
        activity = RecordControllerActivityCommand(
            actor=_actor(activity_request.record.request),
            correlation=activity_request.record.correlation,
            evidence=AcquireControllerActivity(
                request_id=activity_request.record.request.request_id,
                controller_id=ControllerId("controller"),
            ),
        )
        acquired = await store.record_activity(activity)
        assert isinstance(acquired, ControllerActivityApplied)
        repeated_acquire = await store.record_activity(activity)
        assert isinstance(repeated_acquire, ControllerActivityRejected)

        default_request = await _create(
            store,
            _request(
                "trusted-default",
                mode=RequirementMode.ADVISORY,
                default_value=True,
            ),
        )
        defaulted = await store.resolve_trusted_default(
            _trusted_default(default_request.record)
        )
        assert isinstance(defaulted, TrustedDefaultResolutionApplied)

        cancel_request = await _create(
            store,
            _request("cancel", mode=RequirementMode.ADVISORY),
        )
        cancel_command = CancelInteractionCommand(
            actor=_actor(cancel_request.record.request),
            correlation=cancel_request.record.correlation,
            provenance=AnswerProvenance.HUMAN,
            expected_state_revision=StateRevision(99),
        )
        rejected_cancel = await store.cancel(cancel_command)
        assert isinstance(rejected_cancel, CancelInteractionRejected)
        authorizer.denied_operations.add(InteractionOperation.CANCEL_REQUEST)
        denied_cancel = await store.cancel(
            replace(cancel_command, expected_state_revision=None)
        )
        assert isinstance(denied_cancel, CancelInteractionRejected)
        authorizer.denied_operations.clear()

        terminal_request = await _create(
            store,
            _request("terminalize", mode=RequirementMode.ADVISORY),
        )
        terminal_command = TerminalizeInteractionCommand(
            actor=_actor(terminal_request.record.request),
            correlation=terminal_request.record.correlation,
            status=ResolutionStatus.UNAVAILABLE,
            provenance=AnswerProvenance.HUMAN,
            expected_state_revision=StateRevision(99),
        )
        rejected_terminal = await store.terminalize(terminal_command)
        assert isinstance(
            rejected_terminal,
            TerminalizeInteractionRejected,
        )
        applied_terminal = await store.terminalize(
            replace(terminal_command, expected_state_revision=None)
        )
        assert isinstance(applied_terminal, TerminalizeInteractionApplied)
        await store.aclose()

    run(exercise())


def test_missing_denied_correlated_and_invalid_mutations_reject() -> None:
    """Return operation-specific failures without changing shared state."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        factory, _, _ = _factory(authorizer=authorizer)
        store = await factory.open()
        created = await _create(
            store,
            _request("rejections", mode=RequirementMode.ADVISORY),
        )
        missing = replace(
            created.record.correlation,
            request_id=InputRequestId("missing-rejection"),
        )
        wrong = replace(
            created.record.correlation,
            turn_id=TurnId("wrong-rejection-turn"),
        )

        missing_resolution = await store.resolve(
            _answer_for_correlation(created.record, "missing", missing)
        )
        assert isinstance(missing_resolution, ResolveInteractionRejected)
        authorizer.denied_operations.add(InteractionOperation.RESOLVE)
        denied_resolution = await store.resolve(
            _answer(created.record, "denied")
        )
        assert isinstance(denied_resolution, ResolveInteractionRejected)
        authorizer.denied_operations.clear()
        wrong_resolution = await store.resolve(
            _answer_for_correlation(created.record, "wrong", wrong)
        )
        assert isinstance(wrong_resolution, ResolveInteractionRejected)
        stale_resolution = await store.resolve(
            _answer_for_correlation(
                created.record,
                "stale",
                created.record.correlation,
                expected_state_revision=StateRevision(99),
            )
        )
        assert isinstance(stale_resolution, ResolveInteractionRejected)

        with pytest.raises(InteractionNotFoundError):
            await store.resolve_trusted_default(
                _new_trusted_default_resolution_command(
                    TrustedDefaultResolutionRequest(
                        actor=_actor(created.record.request),
                        correlation=missing,
                        expected_state_revision=StateRevision(1),
                    )
                )
            )

        missing_terminal = TerminalizeInteractionCommand(
            actor=_actor(created.record.request),
            correlation=missing,
            status=ResolutionStatus.UNAVAILABLE,
            provenance=AnswerProvenance.HUMAN,
        )
        assert isinstance(
            await store.terminalize(missing_terminal),
            TerminalizeInteractionRejected,
        )
        terminal = replace(
            missing_terminal,
            correlation=created.record.correlation,
        )
        authorizer.denied_operations.add(InteractionOperation.EXPIRE)
        assert isinstance(
            await store.terminalize(terminal),
            TerminalizeInteractionRejected,
        )
        authorizer.denied_operations.clear()
        assert isinstance(
            await store.terminalize(replace(terminal, correlation=wrong)),
            TerminalizeInteractionRejected,
        )

        missing_cancel = CancelInteractionCommand(
            actor=_actor(created.record.request),
            correlation=missing,
            provenance=AnswerProvenance.HUMAN,
        )
        assert isinstance(
            await store.cancel(missing_cancel),
            CancelInteractionRejected,
        )
        assert isinstance(
            await store.cancel(replace(missing_cancel, correlation=wrong)),
            CancelInteractionRejected,
        )

        branch = RegisterInteractionBranchCommand(
            actor=_actor(created.record.request),
            registration=InteractionBranchRegistration(
                run_id=RunId("run"),
                branch_id=BranchId("denied-child"),
                parent_branch_id=BranchId("root"),
                principal=_principal(),
            ),
        )
        authorizer.denied_operations.add(InteractionOperation.REGISTER_BRANCH)
        branch_result = await store.register_branch(branch)
        assert isinstance(
            branch_result,
            InteractionBranchRegistrationRejected,
        )
        assert branch_result.error.code is InputErrorCode.FORBIDDEN
        authorizer.denied_operations.clear()

        missing_activity = RecordControllerActivityCommand(
            actor=_actor(created.record.request),
            correlation=missing,
            evidence=AcquireControllerActivity(
                request_id=missing.request_id,
                controller_id=ControllerId("missing-controller"),
            ),
        )
        assert isinstance(
            await store.record_activity(missing_activity),
            ControllerActivityRejected,
        )
        activity = RecordControllerActivityCommand(
            actor=_actor(created.record.request),
            correlation=created.record.correlation,
            evidence=AcquireControllerActivity(
                request_id=created.record.request.request_id,
                controller_id=ControllerId("denied-controller"),
            ),
        )
        authorizer.denied_operations.add(InteractionOperation.RECORD_ACTIVITY)
        assert isinstance(
            await store.record_activity(activity),
            ControllerActivityRejected,
        )
        authorizer.denied_operations.clear()
        assert isinstance(
            await store.record_activity(replace(activity, correlation=wrong)),
            ControllerActivityRejected,
        )

        authorizer.request_disclosure = InteractionDisclosure.TERMINAL_METADATA
        with pytest.raises(InteractionNotFoundError):
            await store.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=_actor(created.record.request),
                    correlation=created.record.correlation,
                    after_store_revision=InteractionStoreRevision(0),
                )
            )
        await store.aclose()

        invalid_factory, _, _ = _factory(clock=_InvalidObservationClock())
        invalid_store = await invalid_factory.open()
        due = await invalid_store.terminalize_due(
            TerminalizeDueInteractionsCommand()
        )
        assert isinstance(due, DueInteractionsRejected)
        await invalid_store.aclose()

    run(exercise())


def test_scope_supersession_denial_and_ownership_rejection() -> None:
    """Apply supersession and reject denied or foreign scope commands."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        factory, _, _ = _factory(authorizer=authorizer)
        store = await factory.open()
        created = await _create(store, _request("scope-variants"))
        scope = InteractionExecutionScope(run_id=RunId("run"))
        cancel = TerminalizeInteractionScopeCommand(
            actor=_actor(created.record.request),
            scope=scope,
            provenance=AnswerProvenance.HUMAN,
        )
        authorizer.denied_operations.add(InteractionOperation.CANCEL_SCOPE)
        denied = await store.terminalize_scope(cancel)
        assert isinstance(denied, ScopeCancellationRejected)
        authorizer.denied_operations.clear()

        foreign = await store.terminalize_scope(
            replace(
                cancel,
                actor=InteractionActor(principal=_principal("intruder")),
            )
        )
        assert isinstance(foreign, ScopeCancellationRejected)
        supersede = SupersedeInteractionScopeCommand(
            actor=_actor(created.record.request),
            scope=scope,
            provenance=AnswerProvenance.HUMAN,
        )
        authorizer.denied_operations.add(InteractionOperation.SUPERSEDE)
        denied_supersede = await store.supersede_scope(supersede)
        assert isinstance(denied_supersede, ScopeSupersessionRejected)
        authorizer.denied_operations.clear()
        foreign_supersede = await store.supersede_scope(
            replace(
                supersede,
                actor=InteractionActor(principal=_principal("intruder")),
            )
        )
        assert isinstance(foreign_supersede, ScopeSupersessionRejected)
        superseded = await store.supersede_scope(supersede)
        assert isinstance(superseded, ScopeSupersessionApplied)
        await store.aclose()

    run(exercise())


def test_free_form_resolution_classifies_and_rechecks_replay() -> None:
    """Bind classifier output and recheck replay after external awaits."""

    async def exercise() -> None:
        unbound_factory, _, _ = _factory()
        unbound = await unbound_factory.open()
        unbound_request = await _create(
            unbound,
            _text_request("unbound-classifier"),
        )
        rejected = await unbound.resolve(
            _text_answer(unbound_request.record, "unbound")
        )
        assert isinstance(rejected, ResolveInteractionRejected)
        assert (
            rejected.error.code
            is InputErrorCode.SECRET_CLASSIFICATION_UNAVAILABLE
        )
        await unbound.aclose()

        policy = InteractionPolicy()
        classifier = _Classifier(policy)
        factory, _, _ = _factory(policy=policy, classifier=classifier)
        first = await factory.open()
        second = await factory.open()
        created = await _create(first, _text_request("classified-race"))
        left_command = _text_answer(created.record, "classified-left")
        right_command = replace(
            left_command,
            idempotency_key=ResolutionIdempotencyKey("classified-right"),
        )
        classifier.block(after_calls=2)

        left = create_task(first.resolve(left_command))
        right = create_task(second.resolve(right_command))
        await classifier.entered.wait()
        classifier.release.set()
        results = (await left, await right)

        applied = [
            result
            for result in results
            if isinstance(result, ResolveInteractionApplied)
        ]
        replayed = [
            result
            for result in results
            if isinstance(result, InteractionStoreReplayed)
        ]
        assert len(applied) == len(replayed) == 1
        assert applied[0].classifier_binding is not None
        assert applied[0].classification_proof is not None
        assert (
            replayed[0].replay_kind is InteractionReplayKind.SEMANTIC_NEW_KEY
        )
        assert classifier.calls == 2
        await first.aclose()
        await second.aclose()

    run(exercise())


def test_due_free_form_resolution_never_invokes_classifier() -> None:
    """Settle equal and overdue deadlines before failing or hanging policy."""

    async def exercise() -> None:
        for offset in (0, 1):
            for classifier_kind in ("blocking", "exploding"):
                policy = InteractionPolicy()
                classifier: _Classifier | _ExplodingClassifier
                if classifier_kind == "blocking":
                    classifier = _Classifier(policy)
                    classifier.block(after_calls=1)
                else:
                    classifier = _ExplodingClassifier()
                factory, clock, _ = _factory(
                    policy=policy,
                    classifier=classifier,
                )
                store = await factory.open()
                created = await _create(
                    store,
                    _text_request(f"due-{offset}-{classifier_kind}"),
                )
                clock.wall_time = (
                    created.record.absolute_expires_at
                    + timedelta(seconds=offset)
                )
                clock.monotonic_seconds = 600.0 + offset
                operation = create_task(
                    store.resolve(_text_answer(created.record, "due"))
                )
                await _wait_until(
                    lambda: (
                        operation.done()
                        or (
                            isinstance(classifier, _Classifier)
                            and classifier.entered.is_set()
                        )
                    )
                )
                if isinstance(classifier, _Classifier):
                    classifier.release.set()
                result = await operation

                assert isinstance(result, ResolveInteractionApplied)
                assert result.record.request.resolution is not None
                assert (
                    result.record.request.resolution.status
                    is ResolutionStatus.EXPIRED
                )
                assert classifier.calls == 0
                await store.aclose()

    run(exercise())


def test_stale_free_form_resolution_never_invokes_classifier() -> None:
    """Reject stale lifecycle CAS before blocking or failing classification."""

    async def exercise() -> None:
        for classifier_kind in ("blocking", "exploding"):
            policy = InteractionPolicy()
            classifier: _Classifier | _ExplodingClassifier
            if classifier_kind == "blocking":
                classifier = _Classifier(policy)
                classifier.block(after_calls=1)
            else:
                classifier = _ExplodingClassifier()
            factory, _, _ = _factory(
                policy=policy,
                classifier=classifier,
            )
            store = await factory.open()
            created = await _create(
                store,
                _text_request(f"stale-{classifier_kind}"),
            )
            stale = replace(
                _text_answer(created.record, "stale"),
                expected_state_revision=StateRevision(0),
            )
            operation = create_task(store.resolve(stale))
            await _wait_until(
                lambda: (
                    operation.done()
                    or (
                        isinstance(classifier, _Classifier)
                        and classifier.entered.is_set()
                    )
                )
            )
            if isinstance(classifier, _Classifier):
                classifier.release.set()
            result = await operation

            assert isinstance(result, ResolveInteractionRejected)
            assert result.error.code is InputErrorCode.STALE_REVISION
            assert result.error.path == "expected_state_revision"
            assert (
                result.decision_stage is ResolutionDecisionStage.STATE_REVISION
            )
            assert classifier.calls == 0
            stored = await _lookup(store, created.record.correlation)
            assert stored.request.resolution is None
            await store.aclose()

    run(exercise())


def test_malformed_classifier_output_rejects_without_commit() -> None:
    """Reject untrusted classifier output after the deadline preflight."""

    async def exercise() -> None:
        factory, _, _ = _factory(classifier=_MalformedClassifier())
        store = await factory.open()
        created = await _create(store, _text_request("malformed-classifier"))
        rejected = await store.resolve(
            _text_answer(created.record, "malformed-classifier")
        )

        assert isinstance(rejected, ResolveInteractionRejected)
        assert rejected.error.code is InputErrorCode.INVALID_TYPE
        stored = await _lookup(store, created.record.correlation)
        assert stored.request.resolution is None
        await store.aclose()

    run(exercise())


def test_revision_exhaustion_rejects_without_partial_mutation() -> None:
    """Reject deadline and branch generation exhaustion atomically."""

    async def exercise() -> None:
        factory, _, _ = _factory()
        store = await factory.open()
        store._state.schedule_revision = DeadlineScheduleRevision(
            MAX_STATE_REVISION
        )
        request = _request("schedule-exhausted")
        rejected = await store.create(
            CreateInteractionCommand(
                actor=_actor(request),
                request=request,
            )
        )
        assert isinstance(rejected, CreateInteractionRejected)
        assert rejected.error.code is InputErrorCode.STATE_REVISION_EXHAUSTED
        assert (
            await store.lookup_scoped(
                ScopedInteractionLookup(
                    actor=_actor(request),
                    correlation=InteractionCorrelation.from_request(request),
                )
            )
            is None
        )

        store._state.backing = _new_interaction_store_backing(
            store_generation=InteractionStoreGeneration(MAX_STATE_REVISION)
        )
        branch = RegisterInteractionBranchCommand(
            actor=_actor(request),
            registration=InteractionBranchRegistration(
                run_id=RunId("run"),
                branch_id=BranchId("exhausted-child"),
                parent_branch_id=BranchId("root"),
                principal=_principal(),
            ),
        )
        branch_result = await store.register_branch(branch)
        assert isinstance(
            branch_result,
            InteractionBranchRegistrationRejected,
        )
        assert branch_result.error.code is InputErrorCode.OUT_OF_BOUNDS
        await store.aclose()

        for supersede in (False, True):
            scope_factory, _, _ = _factory()
            scope_store = await scope_factory.open()
            scope_created = await _create(
                scope_store,
                _request(f"scope-exhausted-{supersede}"),
            )
            scope_store._state.schedule_revision = DeadlineScheduleRevision(
                MAX_STATE_REVISION
            )
            scope = InteractionExecutionScope(run_id=RunId("run"))
            if supersede:
                supersession_result = await scope_store.supersede_scope(
                    SupersedeInteractionScopeCommand(
                        actor=_actor(scope_created.record.request),
                        scope=scope,
                        provenance=AnswerProvenance.HUMAN,
                    )
                )
                assert isinstance(
                    supersession_result,
                    ScopeSupersessionRejected,
                )
                error = supersession_result.error
            else:
                cancellation_result = await scope_store.terminalize_scope(
                    TerminalizeInteractionScopeCommand(
                        actor=_actor(scope_created.record.request),
                        scope=scope,
                        provenance=AnswerProvenance.HUMAN,
                    )
                )
                assert isinstance(
                    cancellation_result,
                    ScopeCancellationRejected,
                )
                error = cancellation_result.error
            assert error.code is InputErrorCode.STATE_REVISION_EXHAUSTED
            stored = await _lookup(
                scope_store,
                scope_created.record.correlation,
            )
            assert stored.request.resolution is None
            await scope_store.aclose()

    run(exercise())


def test_commit_leaves_waiters_ahead_of_changed_revisions_pending() -> None:
    """Keep record and deadline waits whose cursors remain ahead."""

    async def exercise() -> None:
        factory, _, _ = _factory()
        store = await factory.open()
        created = await _create(store, _request("ahead-waiters"))
        record_wait = create_task(
            store.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=_actor(created.record.request),
                    correlation=created.record.correlation,
                    after_store_revision=InteractionStoreRevision(
                        MAX_STATE_REVISION
                    ),
                )
            )
        )
        deadline_wait = create_task(
            store.wait_for_deadline_change(
                WaitForDeadlineChangeCommand(
                    after_schedule_revision=DeadlineScheduleRevision(
                        MAX_STATE_REVISION
                    ),
                )
            )
        )
        await _wait_until(
            lambda: (
                bool(store._record_waiters) and bool(store._deadline_waiters)
            )
        )

        resolved = await store.resolve(_answer(created.record, "ahead"))
        assert isinstance(resolved, ResolveInteractionApplied)
        assert not record_wait.done()
        assert not deadline_wait.done()

        await store.aclose()
        with pytest.raises(InteractionStoreClosedError):
            await record_wait
        with pytest.raises(InteractionStoreClosedError):
            await deadline_wait

    run(exercise())


def test_cross_handle_record_and_deadline_waiters_wake() -> None:
    """Wake all open handles through the shared notification path."""

    async def exercise() -> None:
        factory, _, _ = _factory()
        waiter_handle = await factory.open()
        deadline_handle = await factory.open()
        writer = await factory.open()
        created = await _create(writer, _request("cross-handle"))
        deadline = await deadline_handle.next_deadline()

        record_wait = create_task(
            waiter_handle.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=_actor(created.record.request),
                    correlation=created.record.correlation,
                    after_store_revision=created.record.store_revision,
                )
            )
        )
        deadline_wait = create_task(
            deadline_handle.wait_for_deadline_change(
                WaitForDeadlineChangeCommand(
                    after_schedule_revision=deadline.schedule_revision,
                )
            )
        )
        await _wait_until(
            lambda: (
                bool(waiter_handle._record_waiters)
                and bool(deadline_handle._deadline_waiters)
            )
        )

        resolved = await writer.resolve(_answer(created.record, "wake"))
        assert isinstance(resolved, ResolveInteractionApplied)
        projection = await record_wait
        changed_deadline = await deadline_wait
        assert isinstance(projection, InteractionRecord)
        assert projection == resolved.record
        assert changed_deadline.deadline is None

        await waiter_handle.aclose()
        await deadline_handle.aclose()
        await writer.aclose()

    run(exercise())


def test_local_close_isolated_and_close_wins_external_wait_race() -> None:
    """Fail only local waits and reject an operation when close wins."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        factory, _, _ = _factory(authorizer=authorizer)
        closing = await factory.open()
        surviving = await factory.open()
        created = await _create(surviving, _request("close-isolation"))

        closing_wait = create_task(
            closing.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=_actor(created.record.request),
                    correlation=created.record.correlation,
                    after_store_revision=created.record.store_revision,
                )
            )
        )
        surviving_wait = create_task(
            surviving.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=_actor(created.record.request),
                    correlation=created.record.correlation,
                    after_store_revision=created.record.store_revision,
                )
            )
        )
        await _wait_until(
            lambda: (
                bool(closing._record_waiters)
                and bool(surviving._record_waiters)
            )
        )
        await closing.aclose()
        with pytest.raises(InteractionStoreClosedError):
            await closing_wait
        assert not surviving_wait.done()

        resolved = await surviving.resolve(_answer(created.record, "survive"))
        assert isinstance(resolved, ResolveInteractionApplied)
        assert await surviving_wait == resolved.record

        racing = await factory.open()
        pending = await _create(
            surviving,
            _request("close-race", mode=RequirementMode.ADVISORY),
        )
        authorizer.block(
            frozenset({InteractionOperation.RESOLVE}),
            expected_calls=1,
        )
        operation = create_task(
            racing.resolve(_answer(pending.record, "close-wins"))
        )
        await authorizer.entered.wait()
        await racing.aclose()
        authorizer.release.set()
        with pytest.raises(InteractionStoreClosedError):
            await operation
        stored = await _lookup(surviving, pending.record.correlation)
        assert stored.request.resolution is None

        await closing.aclose()
        await surviving.aclose()

    run(exercise())


def test_cancelled_waiters_are_removed_exactly_once() -> None:
    """Remove cancelled record and deadline waiter registrations."""

    async def exercise() -> None:
        factory, _, _ = _factory()
        store = await factory.open()
        created = await _create(store, _request("cancelled-waiters"))
        deadline = await store.next_deadline()
        record_wait: Task[object] = create_task(
            store.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=_actor(created.record.request),
                    correlation=created.record.correlation,
                    after_store_revision=created.record.store_revision,
                )
            )
        )
        deadline_wait: Task[object] = create_task(
            store.wait_for_deadline_change(
                WaitForDeadlineChangeCommand(
                    after_schedule_revision=deadline.schedule_revision,
                )
            )
        )
        await _wait_until(
            lambda: (
                bool(store._record_waiters) and bool(store._deadline_waiters)
            )
        )

        record_wait.cancel()
        deadline_wait.cancel()
        with pytest.raises(CancelledError):
            await record_wait
        with pytest.raises(CancelledError):
            await deadline_wait
        assert not store._record_waiters
        assert not store._deadline_waiters
        await store.aclose()

    run(exercise())


def test_commit_clock_read_serializes_final_mutations() -> None:
    """Serialize final trusted observations under the one backing lock."""

    async def exercise() -> None:
        factory, clock, authorizer = _factory()
        first = await factory.open()
        second = await factory.open()
        one = await _create(
            first,
            _request("clock-one", mode=RequirementMode.ADVISORY),
        )
        two = await _create(
            second,
            _request("clock-two", mode=RequirementMode.ADVISORY),
        )
        baseline_reads = clock.read_count
        clock.block_next_read()

        first_commit = create_task(first.resolve(_answer(one.record, "one")))
        await clock.entered.wait()
        second_commit = create_task(second.resolve(_answer(two.record, "two")))
        await _yield_once()
        assert authorizer.seen[InteractionOperation.RESOLVE].is_set()
        assert not second_commit.done()

        clock.release.set()
        first_result = await first_commit
        second_result = await second_commit
        assert isinstance(first_result, ResolveInteractionApplied)
        assert isinstance(second_result, ResolveInteractionApplied)
        assert clock.read_count == baseline_reads + 2
        assert clock.maximum_active_reads == 1
        await first.aclose()
        await second.aclose()

    run(exercise())


def test_final_idempotency_ledger_slot_is_atomic() -> None:
    """Allow exactly one racing semantic replay into the final ledger slot."""

    async def exercise() -> None:
        policy = InteractionPolicy(maximum_idempotency_keys_per_request=2)
        authorizer = _Authorizer()
        factory, _, _ = _factory(policy=policy, authorizer=authorizer)
        first = await factory.open()
        second = await factory.open()
        created = await _create(first, _request("ledger-race"))
        initial = await first.resolve(_answer(created.record, "initial"))
        assert isinstance(initial, ResolveInteractionApplied)

        authorizer.block(
            frozenset({InteractionOperation.RESOLVE}),
            expected_calls=2,
        )
        left = create_task(first.resolve(_answer(created.record, "left")))
        right = create_task(second.resolve(_answer(created.record, "right")))
        await authorizer.entered.wait()
        authorizer.release.set()
        results = (await left, await right)

        replayed = [
            result
            for result in results
            if isinstance(result, InteractionStoreReplayed)
        ]
        rejected = [
            result
            for result in results
            if isinstance(result, ResolveInteractionRejected)
        ]
        assert len(replayed) == len(rejected) == 1
        assert (
            replayed[0].replay_kind is InteractionReplayKind.SEMANTIC_NEW_KEY
        )
        assert rejected[0].error.code.value == "input.idempotency_ledger_full"
        stored = await _lookup(first, created.record.correlation)
        assert len(stored.idempotency_ledger) == 2
        await first.aclose()
        await second.aclose()

    run(exercise())


def test_scope_and_child_create_race_is_serializable() -> None:
    """Linearize complete scope selection against child admission."""

    async def exercise() -> None:
        authorizer = _Authorizer()
        factory, _, _ = _factory(authorizer=authorizer)
        creator = await factory.open()
        canceller = await factory.open()
        root = await _create(creator, _request("scope-root"))
        registration = RegisterInteractionBranchCommand(
            actor=_actor(root.record.request),
            registration=InteractionBranchRegistration(
                run_id=RunId("run"),
                branch_id=BranchId("child"),
                parent_branch_id=BranchId("root"),
                principal=_principal(),
            ),
        )
        registered = await creator.register_branch(registration)
        assert isinstance(registered, InteractionBranchRegistrationApplied)
        child = _request(
            "scope-child",
            origin=_origin(
                branch_id="child",
                parent_branch_id="root",
            ),
        )
        scope_command = TerminalizeInteractionScopeCommand(
            actor=_actor(root.record.request),
            scope=InteractionExecutionScope(
                run_id=RunId("run"),
                branch_id=BranchId("root"),
                include_descendants=True,
            ),
            provenance=AnswerProvenance.HUMAN,
        )
        authorizer.block(
            frozenset(
                {
                    InteractionOperation.CREATE,
                    InteractionOperation.CANCEL_SCOPE,
                }
            ),
            expected_calls=2,
        )
        create_result = create_task(
            creator.create(
                CreateInteractionCommand(
                    actor=_actor(child),
                    request=child,
                )
            )
        )
        scope_result = create_task(canceller.terminalize_scope(scope_command))
        await authorizer.entered.wait()
        authorizer.release.set()

        created = await create_result
        cancelled = await scope_result
        assert isinstance(created, CreateInteractionApplied)
        assert isinstance(cancelled, ScopeCancellationApplied)
        stored = await _lookup(creator, created.record.correlation)
        selected_ids = {
            record.request.request_id for record in cancelled.records
        }
        if child.request_id in selected_ids:
            assert stored.request.resolution is not None
        else:
            assert stored.request.resolution is None
        await creator.aclose()
        await canceller.aclose()

    run(exercise())
