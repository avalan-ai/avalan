"""Persist authoritative interaction state in process memory."""

from ..entities import (
    ContinuationId,
    DeadlineScheduleRevision,
    InputRequestId,
    InteractionStoreRevision,
    PrincipalScope,
    ResolutionStatus,
)
from ..error import (
    InputContractError,
    InputErrorCode,
    InputValidationError,
    InteractionNotFoundError,
    InteractionStoreClosedError,
)
from ..handler import (
    InputResumer,
    InputResumptionNotification,
)
from ..policy import (
    AcquireControllerActivity,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionAuthorizer,
    InteractionClock,
    InteractionDisclosure,
    InteractionIdFactory,
    InteractionOperation,
    InteractionPolicy,
    InteractionRequestAuthorizationTarget,
    InteractionTime,
    TaskInputClassification,
    TaskInputClassificationRequest,
    TaskInputClassifier,
)
from ..state import InputTransitionError, project_resolution_to_model
from ..store import (
    CancelInteractionCommand,
    CancelInteractionRejected,
    CancelInteractionResult,
    ControllerActivityRejected,
    ControllerActivityResult,
    ControllerLeaseExpiredApplied,
    CreateInteractionCommand,
    CreateInteractionRejected,
    CreateInteractionResult,
    DetachInteractionCommand,
    DueInteractionsRejected,
    DueInteractionsResult,
    InteractionBranchRecord,
    InteractionBranchRegistrationApplied,
    InteractionBranchRegistrationRejected,
    InteractionBranchRegistrationReplayed,
    InteractionBranchRegistrationResult,
    InteractionBranchRoot,
    InteractionBranchRootLookup,
    InteractionCorrelation,
    InteractionDeadlineSnapshot,
    InteractionDisclosureProjection,
    InteractionExecutionScope,
    InteractionPresentationRejected,
    InteractionPresentationResult,
    InteractionRecord,
    InteractionReplayKind,
    InteractionResolutionResult,
    InteractionStoreReplayed,
    ListInteractionsCommand,
    PresentInteractionCommand,
    RecordControllerActivityCommand,
    RegisterInteractionBranchCommand,
    ResolutionDecisionStage,
    ResolutionIdempotencyDisposition,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    ScopeCancellationApplied,
    ScopeCancellationRejected,
    ScopeCancellationResult,
    ScopedInteractionLookup,
    ScopeSupersessionApplied,
    ScopeSupersessionRejected,
    ScopeSupersessionResult,
    SupersedeInteractionScopeCommand,
    TerminalizeDueInteractionsCommand,
    TerminalizeInteractionCommand,
    TerminalizeInteractionRejected,
    TerminalizeInteractionResult,
    TerminalizeInteractionScopeCommand,
    TrustedDefaultResolutionCommand,
    TrustedDefaultResolutionResult,
    WaitForDeadlineChangeCommand,
    WaitForInteractionChangeCommand,
    _apply_scope_cancellation,
    _apply_scope_supersession,
    _begin_scope_transaction,
    _bind_task_input_classifications,
    _insert_interaction_store_backing_branch_record,
    _insert_interaction_store_backing_record,
    _InteractionAdmissionCapability,
    _InteractionAdmissionCleanupCommand,
    _InteractionAdmissionCleanupDisposition,
    _InteractionAdmissionCleanupResult,
    _InteractionAdmissionCreateCommand,
    _new_interaction_admission_cleanup_result,
    _new_interaction_store_backing,
    _new_task_input_classifier_binding,
    _reduce_interaction_admission_cleanup,
    _replace_interaction_store_backing_records,
    _resolve_interaction_branch_root,
    _scope_descendant_branches,
    _snapshot_interaction_store_backing,
    _task_input_classification_requests,
    _validate_interaction_admission_cleanup_command,
    _validate_interaction_admission_create_command,
    _validate_trusted_default_resolution_command,
    apply_candidate_resolution,
    apply_controller_activity,
    apply_create_interaction,
    apply_due_interactions,
    apply_interaction_detachment,
    apply_interaction_presentation,
    apply_request_cancellation,
    apply_request_terminalization,
    apply_semantic_resolution_replay,
    apply_trusted_default_resolution,
    evaluate_resolution_idempotency,
    project_authorized_interaction,
    select_next_interaction_deadline,
    validate_interaction_admission,
)
from ..validation import MAX_STATE_REVISION

from asyncio import (
    CancelledError,
    Future,
    Lock,
    Task,
    create_task,
    get_running_loop,
    shield,
)
from dataclasses import dataclass
from typing import cast, final


@dataclass(frozen=True, slots=True)
class _RecordWaiter:
    """Hold one handle-local record waiter registration."""

    request_id: InputRequestId
    after_store_revision: InteractionStoreRevision
    future: Future[None]


@final
class InteractionResumptionDeliveryError(InputContractError):
    """Report one post-commit in-process resumer delivery failure."""

    def __init__(self) -> None:
        super().__init__(
            InputErrorCode.UNAVAILABLE,
            "resumer.delivery",
            "committed interaction outcome could not be delivered",
        )


@dataclass(frozen=True, slots=True)
class _DeadlineWaiter:
    """Hold one handle-local deadline waiter registration."""

    after_schedule_revision: DeadlineScheduleRevision
    future: Future[None]


@dataclass(frozen=True, slots=True)
class _Resumption:
    """Pair one extracted resumer with its committed notification."""

    resumer: InputResumer
    notification: InputResumptionNotification
    handoff: Future[None] | None = None


@dataclass(frozen=True, slots=True)
class _CommitEffects:
    """Carry work extracted under the lock for completion after unlock."""

    signals: tuple[Future[None], ...] = ()
    resumptions: tuple[_Resumption, ...] = ()


@dataclass(frozen=True, slots=True)
class _MemoryAdmissionBinding:
    """Bind one opaque cleanup capability to retained record identity."""

    request_id: InputRequestId
    continuation_id: ContinuationId
    resumer: InputResumer
    handoff: Future[None]


class _MemoryInteractionBacking:
    """Own state shared by every handle opened from one factory."""

    __slots__ = (
        "authorizer",
        "admissions",
        "backing",
        "classifier",
        "classifier_binding",
        "clock",
        "handles",
        "id_factory",
        "lock",
        "policy",
        "resumers",
        "schedule_revision",
    )

    def __init__(
        self,
        *,
        policy: InteractionPolicy,
        clock: InteractionClock,
        authorizer: InteractionAuthorizer,
        id_factory: InteractionIdFactory,
        classifier: TaskInputClassifier | None,
    ) -> None:
        self.policy = policy
        self.clock = clock
        self.authorizer = authorizer
        self.id_factory = id_factory
        self.classifier = classifier
        self.classifier_binding = (
            None
            if classifier is None
            else _new_task_input_classifier_binding(
                classifier_id=policy.task_input_classifier_id,
                policy_revision=policy.task_input_policy_revision,
            )
        )
        self.backing = _new_interaction_store_backing()
        self.admissions: dict[
            _InteractionAdmissionCapability,
            _MemoryAdmissionBinding,
        ] = {}
        self.lock = Lock()
        self.schedule_revision = DeadlineScheduleRevision(0)
        self.resumers: dict[str, InputResumer] = {}
        self.handles: set[MemoryInteractionStore] = set()


@final
class MemoryInteractionStoreFactory:
    """Open isolated handles over one shared in-memory backing."""

    __slots__ = ("_state",)

    def __init__(
        self,
        *,
        policy: InteractionPolicy,
        clock: InteractionClock,
        authorizer: InteractionAuthorizer,
        id_factory: InteractionIdFactory,
        classifier: TaskInputClassifier | None = None,
    ) -> None:
        assert isinstance(policy, InteractionPolicy)
        assert callable(getattr(clock, "read", None))
        assert callable(getattr(authorizer, "authorize", None))
        assert callable(
            getattr(id_factory, "new_active_control_lease_nonce", None)
        )
        assert classifier is None or callable(
            getattr(classifier, "classify_task_input", None)
        )
        self._state = _MemoryInteractionBacking(
            policy=policy,
            clock=clock,
            authorizer=authorizer,
            id_factory=id_factory,
            classifier=classifier,
        )

    async def open(self) -> "MemoryInteractionStore":
        """Return one independently closable handle over shared state."""
        store = MemoryInteractionStore(self._state)
        async with self._state.lock:
            self._state.handles.add(store)
        return store


@final
class MemoryInteractionStore:
    """Implement the interaction store over one factory-owned backing."""

    __slots__ = (
        "_closed",
        "_deadline_waiters",
        "_record_waiters",
        "_state",
    )

    def __init__(self, state: _MemoryInteractionBacking) -> None:
        self._state = state
        self._closed = False
        self._record_waiters: dict[Future[None], _RecordWaiter] = {}
        self._deadline_waiters: dict[Future[None], _DeadlineWaiter] = {}

    async def create(
        self,
        command: CreateInteractionCommand,
    ) -> CreateInteractionResult:
        """Create, admit, and bind one request atomically."""
        return await self._create(command, capability=None)

    async def create_admission(
        self,
        command: _InteractionAdmissionCreateCommand,
    ) -> CreateInteractionResult:
        """Create one broker admission with sealed cleanup authority."""
        command = _validate_interaction_admission_create_command(command)
        return await self._create(
            command._command,
            capability=command._capability,
        )

    async def _create(
        self,
        command: CreateInteractionCommand,
        *,
        capability: _InteractionAdmissionCapability | None,
    ) -> CreateInteractionResult:
        """Create one request and atomically retain optional cleanup state."""
        target = InteractionRequestAuthorizationTarget(
            request_id=command.request.request_id,
            origin=command.request.origin,
        )
        await self._check_open()
        decision = await self._authorize(
            command.actor,
            InteractionOperation.CREATE,
            target,
        )
        if not self._allows(
            decision,
            command.actor,
            InteractionOperation.CREATE,
            target,
        ):
            await self._check_open()
            return CreateInteractionRejected(
                command=command,
                error=_authorization_error(),
            )
        effects = _CommitEffects()
        async with self._state.lock:
            self._ensure_open_locked()
            observed_at = await self._state.clock.read()
            snapshot = _snapshot_interaction_store_backing(self._state.backing)
            try:
                if any(
                    record.request.continuation_id
                    == command.request.continuation_id
                    for record in snapshot.records
                ):
                    raise InputValidationError(
                        InputErrorCode.DUPLICATE,
                        "command.request.continuation_id",
                        "continuation identifier already exists",
                    )
                validate_interaction_admission(
                    snapshot.records,
                    command,
                    self._state.policy,
                )
                result = apply_create_interaction(
                    command,
                    self._state.policy,
                )
                effects = self._commit_insert_locked(
                    result.record,
                    observed_at,
                )
                if command.resumer is not None:
                    self._state.resumers[
                        str(command.request.continuation_id)
                    ] = command.resumer
                if capability is not None:
                    assert command.resumer is not None
                    self._state.admissions[capability] = (
                        _MemoryAdmissionBinding(
                            request_id=command.request.request_id,
                            continuation_id=command.request.continuation_id,
                            resumer=command.resumer,
                            handoff=get_running_loop().create_future(),
                        )
                    )
            except InputValidationError as exc:
                return CreateInteractionRejected(
                    command=command,
                    error=_transition_error(exc),
                )
        await self._publish(effects)
        return result

    async def cleanup_admission(
        self,
        command: _InteractionAdmissionCleanupCommand,
    ) -> _InteractionAdmissionCleanupResult:
        """Settle or prove one exact capability-bound broker admission."""
        command = _validate_interaction_admission_cleanup_command(command)
        effects = _CommitEffects()
        terminal_handoff: Future[None] | None = None
        async with self._state.lock:
            self._ensure_open_locked()
            binding = self._state.admissions.get(command._capability)
            if binding is None:
                return _new_interaction_admission_cleanup_result(
                    _InteractionAdmissionCleanupDisposition.ABSENT
                )
            snapshot = _snapshot_interaction_store_backing(self._state.backing)
            record = next(
                (
                    item
                    for item in snapshot.records
                    if item.request.request_id == binding.request_id
                    and item.request.continuation_id == binding.continuation_id
                ),
                None,
            )
            if record is None:
                self._state.admissions.pop(command._capability, None)
                continuation_key = str(binding.continuation_id)
                if (
                    self._state.resumers.get(continuation_key)
                    is binding.resumer
                ):
                    self._state.resumers.pop(continuation_key, None)
                return _new_interaction_admission_cleanup_result(
                    _InteractionAdmissionCleanupDisposition.ABSENT
                )
            if record.request.resolution is not None:
                effects = _CommitEffects(
                    resumptions=self._take_resumptions_locked((record,))
                )
                terminal_handoff = binding.handoff
                result = _new_interaction_admission_cleanup_result(
                    _InteractionAdmissionCleanupDisposition.TERMINAL
                )
            else:
                observed_at = await self._state.clock.read()
                settled = _reduce_interaction_admission_cleanup(
                    record,
                    observed_at,
                    self._state.policy,
                )
                effects = self._commit_record_locked(
                    record,
                    settled,
                    observed_at,
                )
                result = _new_interaction_admission_cleanup_result(
                    _InteractionAdmissionCleanupDisposition.SETTLED
                )
        await self._publish(effects)
        if terminal_handoff is not None:
            await shield(terminal_handoff)
        return result

    async def lookup_scoped(
        self,
        query: ScopedInteractionLookup,
    ) -> InteractionDisclosureProjection | None:
        """Return one authorized projection or indistinguishable absence."""
        async with self._state.lock:
            self._ensure_open_locked()
            record = self._find_scoped_record_locked(
                query.actor,
                query.correlation,
            )
            if record is None:
                return None
            target = _request_target(record)
        decision = await self._authorize(
            query.actor,
            InteractionOperation.INSPECT,
            target,
        )
        async with self._state.lock:
            self._ensure_open_locked()
            current = self._find_scoped_record_locked(
                query.actor,
                query.correlation,
            )
            if current is None or not self._allows_disclosure(
                decision,
                query.actor,
                InteractionOperation.INSPECT,
                target,
            ):
                return None
            assert decision is not None
            try:
                return project_authorized_interaction(current, decision)
            except InputValidationError:
                return None

    async def list_scoped(
        self,
        command: ListInteractionsCommand,
    ) -> tuple[InteractionDisclosureProjection, ...]:
        """Return authorized projections in one execution scope."""
        await self._check_open()
        scope_target = command.authorization_target
        scope_decision = await self._authorize(
            command.actor,
            InteractionOperation.LIST,
            scope_target,
        )
        if not self._allows_disclosure(
            scope_decision,
            command.actor,
            InteractionOperation.LIST,
            scope_target,
        ):
            await self._check_open()
            return ()
        assert scope_decision is not None
        async with self._state.lock:
            self._ensure_open_locked()
            snapshot = _snapshot_interaction_store_backing(self._state.backing)
            records = _records_in_scope(
                snapshot.records,
                snapshot.branch_records,
                command.scope,
                command.actor.principal,
            )
        decisions: list[
            tuple[InputRequestId, InteractionAuthorizationDecision | None]
        ] = []
        for record in records:
            target = _request_target(record)
            decision = await self._authorize(
                command.actor,
                InteractionOperation.LIST,
                target,
            )
            decisions.append((record.request.request_id, decision))
        async with self._state.lock:
            self._ensure_open_locked()
            snapshot = _snapshot_interaction_store_backing(self._state.backing)
            selected = {
                record.request.request_id: record
                for record in _records_in_scope(
                    snapshot.records,
                    snapshot.branch_records,
                    command.scope,
                    command.actor.principal,
                )
            }
            projections: list[InteractionDisclosureProjection] = []
            for request_id, decision in decisions:
                selected_record = selected.get(request_id)
                assert selected_record is not None
                target = _request_target(selected_record)
                if not self._allows_disclosure(
                    decision,
                    command.actor,
                    InteractionOperation.LIST,
                    target,
                ):
                    continue
                assert decision is not None
                narrowed = InteractionAuthorizationDecision(
                    actor=command.actor,
                    operation=InteractionOperation.LIST,
                    target=target,
                    allowed=True,
                    disclosure=_narrow_disclosure(
                        scope_decision.disclosure,
                        decision.disclosure,
                    ),
                )
                try:
                    projections.append(
                        project_authorized_interaction(
                            selected_record,
                            narrowed,
                        )
                    )
                except InputValidationError:
                    continue
        return tuple(projections)

    async def lookup_branch_root(
        self,
        query: InteractionBranchRootLookup,
    ) -> InteractionBranchRoot | None:
        """Return one authorized content-free branch root mapping."""
        async with self._state.lock:
            self._ensure_open_locked()
            initial_generation = _snapshot_interaction_store_backing(
                self._state.backing
            ).store_generation
        target = query.authorization_target
        decision = await self._authorize(
            query.actor,
            InteractionOperation.INSPECT_BRANCH,
            target,
        )
        async with self._state.lock:
            self._ensure_open_locked()
            snapshot = _snapshot_interaction_store_backing(self._state.backing)
            try:
                root = _resolve_interaction_branch_root(
                    snapshot.branch_records,
                    query,
                )
            except InputValidationError:
                return None
            if (
                snapshot.store_generation != initial_generation
                or root is None
                or not self._allows(
                    decision,
                    query.actor,
                    InteractionOperation.INSPECT_BRANCH,
                    target,
                )
            ):
                return None
            return root

    async def mark_presented(
        self,
        command: PresentInteractionCommand,
    ) -> InteractionPresentationResult:
        """Record attached presentation through the trusted clock."""
        return await self._mutate_presentation(command)

    async def mark_detached(
        self,
        command: DetachInteractionCommand,
    ) -> InteractionPresentationResult:
        """Record detached handling through the trusted clock."""
        return await self._mutate_presentation(command)

    async def resolve(
        self,
        command: ResolveInteractionCommand,
    ) -> InteractionResolutionResult:
        """Resolve with exact deadline, replay, CAS, and validation order."""
        async with self._state.lock:
            self._ensure_open_locked()
            record = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            if record is None:
                return _resolve_rejected(
                    command,
                    _not_found_error(),
                    ResolutionDecisionStage.AUTHORIZATION,
                )
            target = _request_target(record)
        decision = await self._authorize(
            command.actor,
            InteractionOperation.RESOLVE,
            target,
        )
        if not self._allows(
            decision,
            command.actor,
            InteractionOperation.RESOLVE,
            target,
        ):
            await self._check_open()
            return _resolve_rejected(
                command,
                _authorization_error(),
                ResolutionDecisionStage.AUTHORIZATION,
            )
        effects = _CommitEffects()
        classification_requests: tuple[
            TaskInputClassificationRequest,
            ...,
        ] = ()
        result: InteractionResolutionResult
        async with self._state.lock:
            self._ensure_open_locked()
            current = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            assert current is not None
            immediate, effects = self._resolution_replay_locked(
                current,
                command,
            )
            if immediate is not None:
                result = immediate
            else:
                observed_at = await self._state.clock.read()
                try:
                    result = apply_candidate_resolution(
                        current,
                        command,
                        observed_at,
                        self._state.policy,
                    )
                    effects = self._commit_record_locked(
                        current,
                        result.record,
                        observed_at,
                    )
                except InputValidationError as exc:
                    if (
                        exc.code
                        is not InputErrorCode.SECRET_CLASSIFICATION_UNAVAILABLE
                    ):
                        return _resolve_rejected(
                            command,
                            _transition_error(exc),
                            _resolution_error_stage(exc),
                        )
                    classification_requests = (
                        _task_input_classification_requests(
                            current,
                            command,
                            self._state.policy,
                        )
                    )
                    assert classification_requests
                    if self._state.classifier is None:
                        return _resolve_rejected(
                            command,
                            _transition_error(exc),
                            ResolutionDecisionStage.VALIDATION,
                        )
        if not classification_requests:
            await self._publish(effects)
            return result
        classifier = self._state.classifier
        assert classifier is not None
        classifications = tuple(
            [
                await classifier.classify_task_input(request)
                for request in classification_requests
            ]
        )
        async with self._state.lock:
            self._ensure_open_locked()
            current = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            assert current is not None
            immediate, effects = self._resolution_replay_locked(
                current,
                command,
            )
            if immediate is not None:
                result = immediate
            else:
                observed_at = await self._state.clock.read()
                try:
                    result = self._apply_candidate_locked(
                        current,
                        command,
                        classifications,
                        observed_at,
                    )
                    effects = self._commit_record_locked(
                        current,
                        result.record,
                        observed_at,
                    )
                except InputValidationError as exc:
                    return _resolve_rejected(
                        command,
                        _transition_error(exc),
                        _resolution_error_stage(exc),
                    )
        await self._publish(effects)
        return result

    async def resolve_trusted_default(
        self,
        command: TrustedDefaultResolutionCommand,
    ) -> TrustedDefaultResolutionResult:
        """Derive and commit declared defaults through trusted policy."""
        _validate_trusted_default_resolution_command(command)
        async with self._state.lock:
            self._ensure_open_locked()
            current = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            if current is None:
                raise InteractionNotFoundError()
            target = _request_target(current)
        decision = await self._authorize(
            command.actor,
            InteractionOperation.TRUSTED_DEFAULT,
            target,
        )
        if not self._allows(
            decision,
            command.actor,
            InteractionOperation.TRUSTED_DEFAULT,
            target,
        ):
            await self._check_open()
            raise InteractionNotFoundError()
        async with self._state.lock:
            self._ensure_open_locked()
            current = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            assert current is not None
            observed_at = await self._state.clock.read()
            result = apply_trusted_default_resolution(
                current,
                command,
                observed_at,
                self._state.policy,
            )
            effects = self._commit_record_locked(
                current,
                result.record,
                observed_at,
            )
        await self._publish(effects)
        return result

    async def terminalize(
        self,
        command: TerminalizeInteractionCommand,
    ) -> TerminalizeInteractionResult:
        """Apply one trusted non-answer terminal transition atomically."""
        async with self._state.lock:
            self._ensure_open_locked()
            record = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            if record is None:
                return TerminalizeInteractionRejected(
                    command=command,
                    error=_not_found_error(),
                )
            target = _request_target(record)
        operation = (
            InteractionOperation.SUPERSEDE
            if command.status is ResolutionStatus.SUPERSEDED
            else InteractionOperation.EXPIRE
        )
        decision = await self._authorize(
            command.actor,
            operation,
            target,
        )
        if not self._allows(
            decision,
            command.actor,
            operation,
            target,
        ):
            await self._check_open()
            return TerminalizeInteractionRejected(
                command=command,
                error=_authorization_error(),
            )
        effects = _CommitEffects()
        async with self._state.lock:
            self._ensure_open_locked()
            current = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            assert current is not None
            observed_at = await self._state.clock.read()
            try:
                result = apply_request_terminalization(
                    current,
                    command,
                    observed_at,
                    self._state.policy,
                )
                effects = self._commit_record_locked(
                    current,
                    result.record,
                    observed_at,
                )
            except InputValidationError as exc:
                return TerminalizeInteractionRejected(
                    command=command,
                    error=_transition_error(exc),
                )
        await self._publish(effects)
        return result

    async def cancel(
        self,
        command: CancelInteractionCommand,
    ) -> CancelInteractionResult:
        """Cancel one exact request with request-local scope."""
        async with self._state.lock:
            self._ensure_open_locked()
            record = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            if record is None:
                return CancelInteractionRejected(
                    command=command,
                    error=_not_found_error(),
                )
            target = _request_target(record)
        decision = await self._authorize(
            command.actor,
            InteractionOperation.CANCEL_REQUEST,
            target,
        )
        if not self._allows(
            decision,
            command.actor,
            InteractionOperation.CANCEL_REQUEST,
            target,
        ):
            await self._check_open()
            return CancelInteractionRejected(
                command=command,
                error=_authorization_error(),
            )
        effects = _CommitEffects()
        async with self._state.lock:
            self._ensure_open_locked()
            current = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            assert current is not None
            observed_at = await self._state.clock.read()
            try:
                result = apply_request_cancellation(
                    current,
                    command,
                    observed_at,
                    self._state.policy,
                )
                effects = self._commit_record_locked(
                    current,
                    result.record,
                    observed_at,
                )
            except InputValidationError as exc:
                return CancelInteractionRejected(
                    command=command,
                    error=_transition_error(exc),
                )
        await self._publish(effects)
        return result

    async def terminalize_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> ScopeCancellationResult:
        """Cancel one complete locked-snapshot execution scope."""
        return cast(
            ScopeCancellationResult,
            await self._mutate_scope(
                command,
                InteractionOperation.CANCEL_SCOPE,
            ),
        )

    async def supersede_scope(
        self,
        command: SupersedeInteractionScopeCommand,
    ) -> ScopeSupersessionResult:
        """Supersede one complete locked-snapshot execution scope."""
        return cast(
            ScopeSupersessionResult,
            await self._mutate_scope(
                command,
                InteractionOperation.SUPERSEDE,
            ),
        )

    async def register_branch(
        self,
        command: RegisterInteractionBranchCommand,
    ) -> InteractionBranchRegistrationResult:
        """Register a child edge while rejecting cycles and ownership drift."""
        await self._check_open()
        target = command.authorization_target
        decision = await self._authorize(
            command.actor,
            InteractionOperation.REGISTER_BRANCH,
            target,
        )
        if not self._allows(
            decision,
            command.actor,
            InteractionOperation.REGISTER_BRANCH,
            target,
        ):
            await self._check_open()
            return InteractionBranchRegistrationRejected(
                command=command,
                error=_authorization_error(),
            )
        async with self._state.lock:
            self._ensure_open_locked()
            if command.actor.principal != command.registration.principal:
                return InteractionBranchRegistrationRejected(
                    command=command,
                    error=InputTransitionError(
                        code=InputErrorCode.FORBIDDEN,
                        path="branch.registration.principal",
                        message=(
                            "branch registration ownership differs from "
                            "the actor"
                        ),
                    ),
                )
            snapshot = _snapshot_interaction_store_backing(self._state.backing)
            existing = next(
                (
                    record
                    for record in snapshot.branch_records
                    if record.registration.run_id
                    == command.registration.run_id
                    and record.registration.principal
                    == command.registration.principal
                    and record.registration.branch_id
                    == command.registration.branch_id
                ),
                None,
            )
            if existing is not None:
                if existing.registration == command.registration:
                    return InteractionBranchRegistrationReplayed(
                        command=command,
                        record=existing,
                    )
                return InteractionBranchRegistrationRejected(
                    command=command,
                    error=InputTransitionError(
                        code=InputErrorCode.CORRELATION_MISMATCH,
                        path="branch.registration",
                        message=(
                            "branch registration conflicts with stored "
                            "ancestry"
                        ),
                    ),
                )
            if _branch_registration_creates_cycle(
                snapshot.branch_records,
                command,
            ):
                return InteractionBranchRegistrationRejected(
                    command=command,
                    error=InputTransitionError(
                        code=InputErrorCode.CORRELATION_MISMATCH,
                        path="branch.registration.parent_branch_id",
                        message=(
                            "branch registration creates an ancestry cycle"
                        ),
                    ),
                )
            record = InteractionBranchRecord(
                registration=command.registration,
                store_revision=InteractionStoreRevision(1),
            )
            try:
                _insert_interaction_store_backing_branch_record(
                    self._state.backing,
                    record,
                )
            except InputValidationError as exc:
                return InteractionBranchRegistrationRejected(
                    command=command,
                    error=_transition_error(exc),
                )
            return InteractionBranchRegistrationApplied(
                command=command,
                record=record,
            )

    async def record_activity(
        self,
        command: RecordControllerActivityCommand,
    ) -> ControllerActivityResult:
        """Record authenticated sequenced active-control evidence."""
        async with self._state.lock:
            self._ensure_open_locked()
            record = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            if record is None:
                return ControllerActivityRejected(
                    command=command,
                    error=_not_found_error(),
                )
            target = _request_target(record)
        decision = await self._authorize(
            command.actor,
            InteractionOperation.RECORD_ACTIVITY,
            target,
        )
        if not self._allows(
            decision,
            command.actor,
            InteractionOperation.RECORD_ACTIVITY,
            target,
        ):
            await self._check_open()
            return ControllerActivityRejected(
                command=command,
                error=_authorization_error(),
            )
        lease_nonce = None
        if type(command.evidence) is AcquireControllerActivity:
            lease_nonce = (
                await self._state.id_factory.new_active_control_lease_nonce()
            )
        effects = _CommitEffects()
        async with self._state.lock:
            self._ensure_open_locked()
            current = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            assert current is not None
            observed_at = await self._state.clock.read()
            try:
                result = apply_controller_activity(
                    current,
                    command,
                    observed_at,
                    self._state.policy,
                    lease_nonce=lease_nonce,
                )
                effects = self._commit_record_locked(
                    current,
                    result.record,
                    observed_at,
                )
            except InputValidationError as exc:
                return ControllerActivityRejected(
                    command=command,
                    error=_transition_error(exc),
                )
        await self._publish(effects)
        return result

    async def wait_for_change(
        self,
        command: WaitForInteractionChangeCommand,
    ) -> InteractionDisclosureProjection:
        """Wait without a registration race for a newer projection."""
        while True:
            async with self._state.lock:
                self._ensure_open_locked()
                record = self._find_scoped_record_locked(
                    command.actor,
                    command.correlation,
                )
                if record is None:
                    raise InteractionNotFoundError()
                target = _request_target(record)
            decision = await self._authorize(
                command.actor,
                InteractionOperation.WAIT,
                target,
            )
            async with self._state.lock:
                self._ensure_open_locked()
                current = self._find_scoped_record_locked(
                    command.actor,
                    command.correlation,
                )
                if current is None or not self._allows_disclosure(
                    decision,
                    command.actor,
                    InteractionOperation.WAIT,
                    target,
                ):
                    raise InteractionNotFoundError()
                assert decision is not None
                if current.store_revision > command.after_store_revision:
                    try:
                        return project_authorized_interaction(
                            current,
                            decision,
                        )
                    except InputValidationError as exc:
                        raise InteractionNotFoundError() from exc
                future = get_running_loop().create_future()
                self._record_waiters[future] = _RecordWaiter(
                    request_id=current.request.request_id,
                    after_store_revision=command.after_store_revision,
                    future=future,
                )
            try:
                await future
            except CancelledError:
                async with self._state.lock:
                    self._record_waiters.pop(future, None)
                raise

    async def next_deadline(self) -> InteractionDeadlineSnapshot:
        """Return the earliest deadline from one locked snapshot."""
        async with self._state.lock:
            self._ensure_open_locked()
            observed_at = await self._state.clock.read()
            snapshot = _snapshot_interaction_store_backing(self._state.backing)
            return select_next_interaction_deadline(
                snapshot.records,
                observed_at,
                self._state.schedule_revision,
            )

    async def wait_for_deadline_change(
        self,
        command: WaitForDeadlineChangeCommand,
    ) -> InteractionDeadlineSnapshot:
        """Wait until the shared deadline schedule revision changes."""
        while True:
            async with self._state.lock:
                self._ensure_open_locked()
                if (
                    self._state.schedule_revision
                    > command.after_schedule_revision
                ):
                    observed_at = await self._state.clock.read()
                    snapshot = _snapshot_interaction_store_backing(
                        self._state.backing
                    )
                    return select_next_interaction_deadline(
                        snapshot.records,
                        observed_at,
                        self._state.schedule_revision,
                    )
                future = get_running_loop().create_future()
                self._deadline_waiters[future] = _DeadlineWaiter(
                    after_schedule_revision=command.after_schedule_revision,
                    future=future,
                )
            try:
                await future
            except CancelledError:
                async with self._state.lock:
                    self._deadline_waiters.pop(future, None)
                raise

    async def terminalize_due(
        self,
        command: TerminalizeDueInteractionsCommand,
    ) -> DueInteractionsResult:
        """Settle a bounded due batch from one coherent clock observation."""
        effects = _CommitEffects()
        async with self._state.lock:
            self._ensure_open_locked()
            observed_at = await self._state.clock.read()
            snapshot = _snapshot_interaction_store_backing(self._state.backing)
            try:
                result = apply_due_interactions(
                    snapshot.records,
                    command,
                    observed_at,
                    self._state.policy,
                )
                if result.records:
                    previous_by_id = {
                        record.request.request_id: record
                        for record in snapshot.records
                    }
                    previous = tuple(
                        previous_by_id[record.request.request_id]
                        for record in result.records
                    )
                    effects = self._commit_records_locked(
                        previous,
                        result.records,
                        observed_at,
                    )
            except InputValidationError as exc:
                return DueInteractionsRejected(
                    command=command,
                    error=_transition_error(exc),
                )
        await self._publish(effects)
        return result

    async def aclose(self) -> None:
        """Idempotently close only this handle and fail its waits."""
        futures: tuple[Future[None], ...] = ()
        async with self._state.lock:
            if self._closed:
                return
            self._closed = True
            futures = tuple(self._record_waiters) + tuple(
                self._deadline_waiters
            )
            self._record_waiters.clear()
            self._deadline_waiters.clear()
            self._state.handles.discard(self)
        for future in futures:
            if not future.done():
                future.set_exception(InteractionStoreClosedError())

    async def _mutate_presentation(
        self,
        command: PresentInteractionCommand | DetachInteractionCommand,
    ) -> InteractionPresentationResult:
        async with self._state.lock:
            self._ensure_open_locked()
            record = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            if record is None:
                return InteractionPresentationRejected(
                    command=command,
                    error=_not_found_error(),
                )
            target = _request_target(record)
        decision = await self._authorize(
            command.actor,
            InteractionOperation.DELIVER,
            target,
        )
        if not self._allows(
            decision,
            command.actor,
            InteractionOperation.DELIVER,
            target,
        ):
            await self._check_open()
            return InteractionPresentationRejected(
                command=command,
                error=_authorization_error(),
            )
        effects = _CommitEffects()
        async with self._state.lock:
            self._ensure_open_locked()
            current = self._find_scoped_record_locked(
                command.actor,
                command.correlation,
            )
            assert current is not None
            observed_at = await self._state.clock.read()
            try:
                result: InteractionPresentationResult
                if isinstance(command, PresentInteractionCommand):
                    result = apply_interaction_presentation(
                        current,
                        command,
                        observed_at,
                        self._state.policy,
                    )
                else:
                    result = apply_interaction_detachment(
                        current,
                        command,
                        observed_at,
                        self._state.policy,
                    )
                effects = self._commit_record_locked(
                    current,
                    result.record,
                    observed_at,
                )
            except InputValidationError as exc:
                return InteractionPresentationRejected(
                    command=command,
                    error=_transition_error(exc),
                )
        await self._publish(effects)
        return result

    async def _mutate_scope(
        self,
        command: (
            TerminalizeInteractionScopeCommand
            | SupersedeInteractionScopeCommand
        ),
        operation: InteractionOperation,
    ) -> ScopeCancellationResult | ScopeSupersessionResult:
        await self._check_open()
        target = command.authorization_target
        decision = await self._authorize(command.actor, operation, target)
        if not self._allows(decision, command.actor, operation, target):
            await self._check_open()
            error = _authorization_error()
            if isinstance(command, TerminalizeInteractionScopeCommand):
                return ScopeCancellationRejected(
                    command=command,
                    error=error,
                )
            return ScopeSupersessionRejected(command=command, error=error)
        effects = _CommitEffects()
        result: ScopeCancellationResult | ScopeSupersessionResult
        async with self._state.lock:
            self._ensure_open_locked()
            snapshot = _snapshot_interaction_store_backing(self._state.backing)
            before = snapshot.records
            observed_at = await self._state.clock.read()
            try:
                if (
                    self._state.schedule_revision == MAX_STATE_REVISION
                    and snapshot.store_generation < MAX_STATE_REVISION
                ):
                    preview_backing = _new_interaction_store_backing(
                        records=snapshot.records,
                        branch_records=snapshot.branch_records,
                        store_generation=snapshot.store_generation,
                    )
                    preview_transaction = _begin_scope_transaction(
                        preview_backing,
                        command,
                    )
                    if isinstance(
                        command,
                        TerminalizeInteractionScopeCommand,
                    ):
                        cancellation_preview = _apply_scope_cancellation(
                            preview_transaction,
                            command,
                            observed_at,
                            self._state.policy,
                            backing=preview_backing,
                        )
                        preview_applied = isinstance(
                            cancellation_preview,
                            ScopeCancellationApplied,
                        )
                    else:
                        supersession_preview = _apply_scope_supersession(
                            preview_transaction,
                            command,
                            observed_at,
                            self._state.policy,
                            backing=preview_backing,
                        )
                        preview_applied = isinstance(
                            supersession_preview,
                            ScopeSupersessionApplied,
                        )
                    if preview_applied:
                        preview_snapshot = _snapshot_interaction_store_backing(
                            preview_backing
                        )
                        self._next_schedule_revision_locked(
                            before,
                            preview_snapshot.records,
                            observed_at,
                        )
                transaction = _begin_scope_transaction(
                    self._state.backing,
                    command,
                )
                if isinstance(command, TerminalizeInteractionScopeCommand):
                    result = _apply_scope_cancellation(
                        transaction,
                        command,
                        observed_at,
                        self._state.policy,
                        backing=self._state.backing,
                    )
                else:
                    result = _apply_scope_supersession(
                        transaction,
                        command,
                        observed_at,
                        self._state.policy,
                        backing=self._state.backing,
                    )
                if isinstance(
                    result,
                    (ScopeCancellationApplied, ScopeSupersessionApplied),
                ):
                    effects = self._finish_existing_commit_locked(
                        before,
                        result.previous,
                        result.records,
                        observed_at,
                    )
            except InputValidationError as exc:
                error = _transition_error(exc)
                if isinstance(command, TerminalizeInteractionScopeCommand):
                    return ScopeCancellationRejected(
                        command=command,
                        error=error,
                    )
                return ScopeSupersessionRejected(
                    command=command,
                    error=error,
                )
        await self._publish(effects)
        return result

    async def _authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision | None:
        decision = await self._state.authorizer.authorize(
            actor,
            operation,
            target,
        )
        if not isinstance(decision, InteractionAuthorizationDecision):
            return None
        return decision

    async def _check_open(self) -> None:
        async with self._state.lock:
            self._ensure_open_locked()

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise InteractionStoreClosedError()

    def _find_scoped_record_locked(
        self,
        actor: InteractionActor,
        correlation: InteractionCorrelation,
    ) -> InteractionRecord | None:
        snapshot = _snapshot_interaction_store_backing(self._state.backing)
        return next(
            (
                record
                for record in snapshot.records
                if record.correlation == correlation
                and record.request.origin.principal == actor.principal
            ),
            None,
        )

    @staticmethod
    def _allows(
        decision: InteractionAuthorizationDecision | None,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> bool:
        return (
            isinstance(decision, InteractionAuthorizationDecision)
            and decision.actor == actor
            and decision.operation is operation
            and decision.target == target
            and decision.allowed
        )

    @classmethod
    def _allows_disclosure(
        cls,
        decision: InteractionAuthorizationDecision | None,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> bool:
        if not cls._allows(decision, actor, operation, target):
            return False
        assert decision is not None
        return decision.disclosure is not InteractionDisclosure.NONE

    def _resolution_replay_locked(
        self,
        current: InteractionRecord,
        command: ResolveInteractionCommand,
    ) -> tuple[InteractionResolutionResult | None, _CommitEffects]:
        disposition = evaluate_resolution_idempotency(
            current,
            key=command.idempotency_key,
            resolution_digest=command.resolution_digest,
            maximum_keys=self._state.policy.maximum_idempotency_keys_per_request,
        )
        if disposition is ResolutionIdempotencyDisposition.NEW_RESOLUTION:
            return None, _CommitEffects()
        if disposition is ResolutionIdempotencyDisposition.SAME_KEY:
            return (
                InteractionStoreReplayed(
                    command=command,
                    record=current,
                    replay_kind=InteractionReplayKind.SAME_KEY,
                ),
                _CommitEffects(),
            )
        if disposition is ResolutionIdempotencyDisposition.SEMANTIC_NEW_KEY:
            result = apply_semantic_resolution_replay(
                current,
                command,
                maximum_keys=(
                    self._state.policy.maximum_idempotency_keys_per_request
                ),
            )
            effects = self._commit_replay_locked(current, result.record)
            return result, effects
        if disposition is ResolutionIdempotencyDisposition.SAME_KEY_CONFLICT:
            return (
                _resolve_rejected(
                    command,
                    InputTransitionError(
                        code=InputErrorCode.IDEMPOTENCY_CONFLICT,
                        path="command.idempotency_key",
                        message=(
                            "idempotency key is bound to different content"
                        ),
                    ),
                    ResolutionDecisionStage.IDEMPOTENCY_KEY,
                ),
                _CommitEffects(),
            )
        if disposition is ResolutionIdempotencyDisposition.LEDGER_FULL:
            return (
                _resolve_rejected(
                    command,
                    InputTransitionError(
                        code=InputErrorCode.IDEMPOTENCY_LEDGER_FULL,
                        path="command.idempotency_key",
                        message="idempotency ledger is full",
                    ),
                    ResolutionDecisionStage.SEMANTIC_REPLAY,
                ),
                _CommitEffects(),
            )
        return (
            _resolve_rejected(
                command,
                InputTransitionError(
                    code=InputErrorCode.ALREADY_RESOLVED,
                    path="command.proposed_resolution",
                    message="interaction already has another terminal outcome",
                ),
                ResolutionDecisionStage.SEMANTIC_REPLAY,
            ),
            _CommitEffects(),
        )

    def _apply_candidate_locked(
        self,
        current: InteractionRecord,
        command: ResolveInteractionCommand,
        classifications: tuple[TaskInputClassification, ...],
        observed_at: InteractionTime,
    ) -> ResolveInteractionApplied | ControllerLeaseExpiredApplied:
        try:
            return apply_candidate_resolution(
                current,
                command,
                observed_at,
                self._state.policy,
            )
        except InputValidationError as exc:
            assert exc.code is InputErrorCode.SECRET_CLASSIFICATION_UNAVAILABLE
        binding = self._state.classifier_binding
        assert binding is not None
        assert self._state.classifier is not None
        proof = _bind_task_input_classifications(
            binding,
            current,
            command,
            classifications,
            self._state.policy,
        )
        return apply_candidate_resolution(
            current,
            command,
            observed_at,
            self._state.policy,
            classifier_binding=binding,
            classification_proof=proof,
        )

    def _commit_insert_locked(
        self,
        record: InteractionRecord,
        observed_at: InteractionTime,
    ) -> _CommitEffects:
        snapshot = _snapshot_interaction_store_backing(self._state.backing)
        before = snapshot.records
        after = before + (record,)
        next_revision = self._next_schedule_revision_locked(
            before,
            after,
            observed_at,
        )
        _insert_interaction_store_backing_record(
            self._state.backing,
            record,
        )
        return self._finish_commit_effects_locked(
            (record,),
            next_revision,
        )

    def _commit_record_locked(
        self,
        previous: InteractionRecord,
        record: InteractionRecord,
        observed_at: InteractionTime,
    ) -> _CommitEffects:
        return self._commit_records_locked(
            (previous,),
            (record,),
            observed_at,
        )

    def _commit_records_locked(
        self,
        previous: tuple[InteractionRecord, ...],
        records: tuple[InteractionRecord, ...],
        observed_at: InteractionTime,
    ) -> _CommitEffects:
        snapshot = _snapshot_interaction_store_backing(self._state.backing)
        replacements = {
            record.request.request_id: record for record in records
        }
        after = tuple(
            replacements.get(record.request.request_id, record)
            for record in snapshot.records
        )
        next_revision = self._next_schedule_revision_locked(
            snapshot.records,
            after,
            observed_at,
        )
        _replace_interaction_store_backing_records(
            self._state.backing,
            previous,
            records,
        )
        return self._finish_commit_effects_locked(records, next_revision)

    def _commit_replay_locked(
        self,
        previous: InteractionRecord,
        record: InteractionRecord,
    ) -> _CommitEffects:
        _replace_interaction_store_backing_records(
            self._state.backing,
            (previous,),
            (record,),
        )
        signals = self._take_record_waiters_locked((record,))
        return _CommitEffects(signals=signals)

    def _finish_existing_commit_locked(
        self,
        before: tuple[InteractionRecord, ...],
        previous: tuple[InteractionRecord, ...],
        records: tuple[InteractionRecord, ...],
        observed_at: InteractionTime,
    ) -> _CommitEffects:
        snapshot = _snapshot_interaction_store_backing(self._state.backing)
        next_revision = self._next_schedule_revision_locked(
            before,
            snapshot.records,
            observed_at,
        )
        assert len(previous) == len(records)
        return self._finish_commit_effects_locked(records, next_revision)

    def _next_schedule_revision_locked(
        self,
        before: tuple[InteractionRecord, ...],
        after: tuple[InteractionRecord, ...],
        observed_at: InteractionTime,
    ) -> DeadlineScheduleRevision | None:
        before_deadline = select_next_interaction_deadline(
            before,
            observed_at,
            self._state.schedule_revision,
        ).deadline
        after_deadline = select_next_interaction_deadline(
            after,
            observed_at,
            self._state.schedule_revision,
        ).deadline
        if before_deadline == after_deadline:
            return None
        if self._state.schedule_revision == MAX_STATE_REVISION:
            raise InputValidationError(
                InputErrorCode.STATE_REVISION_EXHAUSTED,
                "deadline.schedule_revision",
                "deadline schedule revision is exhausted",
            )
        return DeadlineScheduleRevision(self._state.schedule_revision + 1)

    def _finish_commit_effects_locked(
        self,
        records: tuple[InteractionRecord, ...],
        next_schedule_revision: DeadlineScheduleRevision | None,
    ) -> _CommitEffects:
        signals = list(self._take_record_waiters_locked(records))
        if next_schedule_revision is not None:
            self._state.schedule_revision = next_schedule_revision
            signals.extend(self._take_deadline_waiters_locked())
        resumptions = self._take_resumptions_locked(records)
        return _CommitEffects(
            signals=tuple(signals),
            resumptions=resumptions,
        )

    def _take_record_waiters_locked(
        self,
        records: tuple[InteractionRecord, ...],
    ) -> tuple[Future[None], ...]:
        revisions = {
            record.request.request_id: record.store_revision
            for record in records
        }
        futures: list[Future[None]] = []
        for handle in tuple(self._state.handles):
            for future, waiter in tuple(handle._record_waiters.items()):
                revision = revisions.get(waiter.request_id)
                if revision is None or revision <= waiter.after_store_revision:
                    continue
                handle._record_waiters.pop(future, None)
                if not future.done():
                    futures.append(future)
        return tuple(futures)

    def _take_deadline_waiters_locked(self) -> tuple[Future[None], ...]:
        futures: list[Future[None]] = []
        for handle in tuple(self._state.handles):
            for future, waiter in tuple(handle._deadline_waiters.items()):
                if (
                    self._state.schedule_revision
                    <= waiter.after_schedule_revision
                ):
                    continue
                handle._deadline_waiters.pop(future, None)
                if not future.done():
                    futures.append(future)
        return tuple(futures)

    def _take_resumptions_locked(
        self,
        records: tuple[InteractionRecord, ...],
    ) -> tuple[_Resumption, ...]:
        resumptions: list[_Resumption] = []
        for record in records:
            if record.request.resolution is None:
                continue
            resumer = self._state.resumers.pop(
                str(record.request.continuation_id),
                None,
            )
            if resumer is None:
                continue
            handoff = self._admission_handoff_locked(record, resumer)
            outcome = project_resolution_to_model(
                record.request,
                containing_run_exists=True,
            )
            resumptions.append(
                _Resumption(
                    resumer=resumer,
                    notification=InputResumptionNotification(
                        continuation_id=record.request.continuation_id,
                        state_revision=record.request.state_revision,
                        outcome=outcome,
                    ),
                    handoff=handoff,
                )
            )
        return tuple(resumptions)

    def _admission_handoff_locked(
        self,
        record: InteractionRecord,
        resumer: InputResumer,
    ) -> Future[None] | None:
        """Return the exact capability-owned bridge handoff barrier."""
        for binding in self._state.admissions.values():
            if (
                binding.request_id == record.request.request_id
                and binding.continuation_id == record.request.continuation_id
                and binding.resumer is resumer
            ):
                return binding.handoff
        return None

    async def _publish(self, effects: _CommitEffects) -> None:
        for future in effects.signals:
            if not future.done():
                future.set_result(None)
        if not effects.resumptions:
            return
        delivery = create_task(self._deliver_resumptions(effects.resumptions))
        try:
            await shield(delivery)
        except CancelledError:
            await self._drain_resumption_delivery(delivery)
            raise

    @staticmethod
    async def _drain_resumption_delivery(delivery: Task[None]) -> None:
        """Finish one shielded post-commit delivery despite cancellation."""
        while not delivery.done():
            try:
                await shield(delivery)
            except CancelledError:
                continue
        delivery.result()

    @staticmethod
    async def _deliver_resumptions(
        resumptions: tuple[_Resumption, ...],
    ) -> None:
        """Deliver every extracted continuation once in deterministic order."""
        for item in resumptions:
            try:
                await item.resumer(item.notification)
            except CancelledError:
                _report_resumption_delivery_failure()
            except Exception:
                _report_resumption_delivery_failure()
            finally:
                if item.handoff is not None and not item.handoff.done():
                    item.handoff.set_result(None)


def _report_resumption_delivery_failure() -> None:
    """Report one content-safe callback failure to the event loop."""
    error = InteractionResumptionDeliveryError()
    get_running_loop().call_exception_handler(
        {
            "message": error.safe_message,
            "exception": error,
        }
    )


def _request_target(
    record: InteractionRecord,
) -> InteractionRequestAuthorizationTarget:
    return InteractionRequestAuthorizationTarget(
        request_id=record.request.request_id,
        origin=record.request.origin,
    )


def _branch_registration_creates_cycle(
    branch_records: tuple[InteractionBranchRecord, ...],
    command: RegisterInteractionBranchCommand,
) -> bool:
    registration = command.registration
    parents = {
        record.registration.branch_id: record.registration.parent_branch_id
        for record in branch_records
        if record.registration.run_id == registration.run_id
        and record.registration.principal == registration.principal
    }
    current = registration.parent_branch_id
    while current in parents:
        current = parents[current]
    return current == registration.branch_id


def _records_in_scope(
    records: tuple[InteractionRecord, ...],
    branch_records: tuple[InteractionBranchRecord, ...],
    scope: InteractionExecutionScope,
    principal: PrincipalScope,
) -> tuple[InteractionRecord, ...]:
    descendants = _scope_descendant_branches(
        scope,
        branch_records,
        principal,
    )
    return tuple(
        record
        for record in records
        if record.request.origin.run_id == scope.run_id
        and record.request.origin.principal == principal
        and (
            scope.turn_id is None
            or record.request.origin.turn_id == scope.turn_id
        )
        and (
            scope.task_id is None
            or record.request.origin.task_id == scope.task_id
        )
        and (
            scope.agent_id is None
            or record.request.origin.agent_id == scope.agent_id
        )
        and (
            scope.branch_id is None
            or record.request.origin.branch_id in descendants
        )
    )


def _narrow_disclosure(
    left: InteractionDisclosure,
    right: InteractionDisclosure,
) -> InteractionDisclosure:
    rank = {
        InteractionDisclosure.NONE: 0,
        InteractionDisclosure.TERMINAL_METADATA: 1,
        InteractionDisclosure.FULL: 2,
    }
    return left if rank[left] <= rank[right] else right


def _transition_error(error: InputValidationError) -> InputTransitionError:
    return InputTransitionError(
        code=error.code,
        path=error.path,
        message=error.safe_message,
    )


def _authorization_error() -> InputTransitionError:
    return InputTransitionError(
        code=InputErrorCode.FORBIDDEN,
        path="authorization",
        message="operation is not authorized",
    )


def _not_found_error() -> InputTransitionError:
    return InputTransitionError(
        code=InputErrorCode.NOT_FOUND,
        path="interaction",
        message="interaction was not found",
    )


def _resolve_rejected(
    command: ResolveInteractionCommand,
    error: InputTransitionError,
    stage: ResolutionDecisionStage,
) -> ResolveInteractionRejected:
    return ResolveInteractionRejected(
        command=command,
        error=error,
        decision_stage=stage,
    )


def _resolution_error_stage(
    error: InputValidationError,
) -> ResolutionDecisionStage:
    if error.code is InputErrorCode.STALE_REVISION:
        return ResolutionDecisionStage.STATE_REVISION
    return ResolutionDecisionStage.VALIDATION
