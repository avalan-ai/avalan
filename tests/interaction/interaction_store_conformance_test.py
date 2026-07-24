"""Run the concrete interaction-store contract against every backend mode."""

from asyncio import (
    AbstractEventLoop,
    Event,
    Task,
    create_task,
    gather,
    get_running_loop,
    run,
)
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast

import pytest

sys_path.append(str(Path(__file__).parent / "stores"))

from pgsql_support import (  # noqa: E402
    FakeInteractionCipher,
    FakePgsqlDatabase,
)

from avalan.interaction import (
    AcquireControllerActivity,
    ActiveControlLeaseNonce,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancelInteractionApplied,
    CancelInteractionCommand,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    ControllerActivityApplied,
    ControllerId,
    ControllerLeaseExpiredApplied,
    CreateInteractionApplied,
    CreateInteractionCommand,
    CreateInteractionRejected,
    DeclinedResolution,
    DueInteractionsApplied,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputErrorCode,
    InputRequest,
    InputRequestId,
    InputResumer,
    InputResumptionNotification,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionBranchRegistration,
    InteractionBranchRegistrationApplied,
    InteractionBranchRegistrationRejected,
    InteractionBranchRegistrationReplayed,
    InteractionBranchRoot,
    InteractionBranchRootLookup,
    InteractionClock,
    InteractionCorrelation,
    InteractionDisclosure,
    InteractionExecutionScope,
    InteractionIdFactory,
    InteractionNotFoundError,
    InteractionOperation,
    InteractionPolicy,
    InteractionPresentationApplied,
    InteractionRecord,
    InteractionReplayKind,
    InteractionRequestAuthorizationTarget,
    InteractionScopeAuthorizationTarget,
    InteractionStore,
    InteractionStoreFactory,
    InteractionStoreReplayed,
    InteractionStoreRevision,
    InteractionTerminalMetadata,
    InteractionTime,
    ListInteractionsCommand,
    ModelCallId,
    PresentInteractionCommand,
    PrincipalScope,
    PulseControllerActivity,
    QuestionId,
    RecordControllerActivityCommand,
    RegisterInteractionBranchCommand,
    ReleaseControllerActivity,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    RunId,
    ScopeCancellationApplied,
    ScopeCancellationRejected,
    ScopeCancellationReplayed,
    ScopedInteractionLookup,
    ScopeSupersessionApplied,
    ScopeSupersessionRejected,
    ScopeSupersessionReplayed,
    StateRevision,
    StreamSessionId,
    SupersedeInteractionScopeCommand,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TaskInputClassifier,
    TerminalizeDueInteractionsCommand,
    TerminalizeInteractionCommand,
    TerminalizeInteractionScopeCommand,
    TurnId,
    UserId,
    WaitForDeadlineChangeCommand,
    WaitForInteractionChangeCommand,
    create_input_request,
)
from avalan.interaction.store import (
    _InteractionAdmissionCleanupDisposition,
    _new_interaction_admission_commands,
)
from avalan.interaction.stores import (
    InteractionResumptionDeliveryError,
    MemoryInteractionStoreFactory,
)
from avalan.interaction.stores.pgsql import (
    PgsqlInteractionStoreFactory,
    PgsqlInteractionStorePolicy,
)

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


async def _yield_once() -> None:
    """Yield exactly one scheduled event-loop turn without a timer."""
    loop = get_running_loop()
    ready = loop.create_future()
    loop.call_soon(ready.set_result, None)
    await ready


async def _yield_until_done(task: Task[object], *, turns: int = 20) -> bool:
    """Return whether a task completes within bounded scheduler turns."""
    for _ in range(turns):
        if task.done():
            return True
        await _yield_once()
    return task.done()


class _Clock(InteractionClock):
    """Provide deterministic trusted observations and injected failures."""

    def __init__(self) -> None:
        self.wall_time = _NOW
        self.monotonic_seconds = 0.0
        self.read_count = 0
        self.failure: BaseException | None = None
        self.changed = Event()

    async def read(self) -> InteractionTime:
        """Return the current coherent observation or one queued failure."""
        self.read_count += 1
        failure = self.failure
        self.failure = None
        if failure is not None:
            raise failure
        return InteractionTime.from_clock(
            wall_time=self.wall_time,
            monotonic_seconds=self.monotonic_seconds,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        """Wait for manual advancement to reach the requested deadline."""
        while self.monotonic_seconds < monotonic_deadline:
            await self.changed.wait()
            self.changed.clear()

    def advance(self, seconds: float) -> None:
        """Advance wall and monotonic time together."""
        assert seconds >= 0.0
        self.wall_time += timedelta(seconds=seconds)
        self.monotonic_seconds += seconds
        self.changed.set()


class _IdFactory(InteractionIdFactory):
    """Mint deterministic, globally distinct runtime-owned identifiers."""

    def __init__(self) -> None:
        self.sequence = 0

    def _next(self, kind: str) -> str:
        self.sequence += 1
        return f"conformance-{kind}-{self.sequence}"

    async def new_request_id(self) -> InputRequestId:
        """Return a request identifier."""
        return InputRequestId(self._next("request"))

    async def new_continuation_id(self) -> ContinuationId:
        """Return a continuation identifier."""
        return ContinuationId(self._next("continuation"))

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        """Return an idempotency key."""
        return ResolutionIdempotencyKey(self._next("key"))

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        """Return a controller lease nonce."""
        return ActiveControlLeaseNonce(self._next("lease"))


class _Classifier(TaskInputClassifier):
    """Allow every value while exactly echoing its classification binding."""

    def __init__(self, policy: InteractionPolicy) -> None:
        self.policy = policy
        self.sequence = 0

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Return one deterministic allow decision."""
        self.sequence += 1
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=self.policy.task_input_classifier_id,
            classification_id=f"classification-{self.sequence}",
            policy_revision=self.policy.task_input_policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


class _RecordingResumer(InputResumer):
    """Record every delivery attempt and optionally raise after recording."""

    def __init__(self, *, fail: bool) -> None:
        self.fail = fail
        self.notifications: list[InputResumptionNotification] = []

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        """Record one delivery and inject a raw callback failure if asked."""
        self.notifications.append(notification)
        if self.fail:
            raise RuntimeError("sensitive raw resumer failure")


class _Authorizer:
    """Record exact inputs and return configurable exact-echo decisions."""

    def __init__(self) -> None:
        self.calls: list[
            tuple[
                InteractionActor,
                InteractionOperation,
                InteractionAuthorizationTarget,
            ]
        ] = []
        self.allowed = True
        self.disclosure = InteractionDisclosure.FULL
        self.block_operation: InteractionOperation | None = None
        self.entered = Event()
        self.release = Event()

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        """Return a decision bound to the exact supplied authorization."""
        self.calls.append((actor, operation, target))
        if operation is self.block_operation:
            self.entered.set()
            await self.release.wait()
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=self.allowed,
            disclosure=(
                self.disclosure if self.allowed else InteractionDisclosure.NONE
            ),
        )


@dataclass(slots=True)
class _Harness:
    """Hold one backend handle and its deterministic trusted dependencies."""

    store: InteractionStore
    factory: InteractionStoreFactory
    policy: InteractionPolicy
    clock: _Clock
    authorizer: _Authorizer
    reopen_after_first_create: bool = False
    reopened_with_state: bool = False


@dataclass(frozen=True, slots=True)
class _Backend:
    """Run one contract against a fresh or reopened in-memory handle."""

    name: str

    def run(
        self,
        contract: Callable[[_Harness], Awaitable[None]],
        *,
        policy: InteractionPolicy | None = None,
    ) -> None:
        """Open, exercise, and close one isolated concrete backend."""

        async def exercise() -> None:
            active_policy = policy or InteractionPolicy()
            clock = _Clock()
            authorizer = _Authorizer()
            if self.name.startswith("pgsql"):
                factory: InteractionStoreFactory = (
                    PgsqlInteractionStoreFactory(
                        FakePgsqlDatabase(),
                        policy=active_policy,
                        clock=clock,
                        authorizer=authorizer,
                        id_factory=_IdFactory(),
                        classifier=_Classifier(active_policy),
                        cipher=FakeInteractionCipher(),
                        store_policy=PgsqlInteractionStorePolicy(
                            poll_interval_seconds=0.000001,
                        ),
                    )
                )
            else:
                factory = MemoryInteractionStoreFactory(
                    policy=active_policy,
                    clock=clock,
                    authorizer=authorizer,
                    id_factory=_IdFactory(),
                    classifier=_Classifier(active_policy),
                )
            store = await factory.open()
            if self.name.endswith("reopened"):
                await store.aclose()
                store = await factory.open()
            harness = _Harness(
                store=store,
                factory=factory,
                policy=active_policy,
                clock=clock,
                authorizer=authorizer,
                reopen_after_first_create=self.name.endswith("reopened"),
            )
            try:
                await contract(harness)
            finally:
                await harness.store.aclose()

        run(exercise())


@pytest.fixture(
    params=(
        "memory-fresh",
        "memory-reopened",
        "pgsql-fresh",
        "pgsql-reopened",
    )
)
def backend(request: pytest.FixtureRequest) -> _Backend:
    """Return every concrete-store lifecycle required by the contract."""
    return _Backend(cast(str, request.param))


def _principal(name: str = "owner") -> PrincipalScope:
    return PrincipalScope(user_id=UserId(name))


def _origin(
    *,
    run_id: str = "run",
    turn_id: str = "turn",
    branch_id: str = "root",
    parent_branch_id: str | None = None,
    principal: PrincipalScope | None = None,
) -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId(run_id),
        turn_id=TurnId(turn_id),
        agent_id=AgentId("agent"),
        branch_id=BranchId(branch_id),
        parent_branch_id=(
            BranchId(parent_branch_id)
            if parent_branch_id is not None
            else None
        ),
        model_call_id=ModelCallId(f"call-{turn_id}-{branch_id}"),
        stream_session_id=StreamSessionId(f"stream-{run_id}"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://conformance",
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
    continuation_id: str | None = None,
    mode: RequirementMode = RequirementMode.REQUIRED,
    reason: str = "Confirm the operation.",
    created_at: datetime = _NOW,
    continuation_ttl_seconds: int = 600,
    advisory_wait_seconds: int = 10,
) -> InputRequest:
    return create_input_request(
        request_id=InputRequestId(name),
        continuation_id=ContinuationId(continuation_id or f"continue-{name}"),
        origin=origin or _origin(),
        mode=mode,
        reason=reason,
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        ),
        created_at=created_at,
        continuation_ttl_seconds=continuation_ttl_seconds,
        advisory_wait_seconds=(
            advisory_wait_seconds if mode is RequirementMode.ADVISORY else None
        ),
    )


def _actor(request: InputRequest) -> InteractionActor:
    return InteractionActor(principal=request.origin.principal)


async def _create(
    harness: _Harness,
    request: InputRequest,
    *,
    resumer: InputResumer | None = None,
) -> CreateInteractionApplied | CreateInteractionRejected:
    result = await harness.store.create(
        CreateInteractionCommand(
            actor=_actor(request),
            request=request,
            resumer=resumer,
        )
    )
    if (
        harness.reopen_after_first_create
        and not harness.reopened_with_state
        and isinstance(result, CreateInteractionApplied)
    ):
        await harness.store.aclose()
        harness.store = await harness.factory.open()
        harness.reopened_with_state = True
    return result


def _answer(
    record: InteractionRecord,
    key: str,
    *,
    value: bool = True,
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
                ConfirmationAnswer(
                    question_id=QuestionId("confirm"),
                    provenance=AnswerProvenance.HUMAN,
                    value=value,
                ),
            ),
        ),
    )


def _decline(
    record: InteractionRecord,
    key: str,
) -> ResolveInteractionCommand:
    return ResolveInteractionCommand(
        actor=_actor(record.request),
        correlation=record.correlation,
        expected_state_revision=record.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=DeclinedResolution(
            request_id=record.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        ),
    )


def _assert_rejected_code(
    result: (
        CreateInteractionRejected
        | ResolveInteractionRejected
        | InteractionBranchRegistrationRejected
    ),
    code: InputErrorCode,
) -> None:
    assert result.error.code is code
    assert not result.store_mutation_applied


def test_request_and_continuation_identifiers_are_unique(
    backend: _Backend,
) -> None:
    """Reject both identity collisions without changing the stored record."""

    async def contract(harness: _Harness) -> None:
        original = _request("request-1")
        created = await _create(harness, original)
        assert isinstance(created, CreateInteractionApplied)

        duplicate_request = await _create(harness, original)
        assert isinstance(duplicate_request, CreateInteractionRejected)
        _assert_rejected_code(duplicate_request, InputErrorCode.DUPLICATE)

        duplicate_continuation = await _create(
            harness,
            _request(
                "request-2",
                continuation_id=str(original.continuation_id),
                reason="A distinct request.",
            ),
        )
        assert isinstance(duplicate_continuation, CreateInteractionRejected)
        _assert_rejected_code(
            duplicate_continuation,
            InputErrorCode.DUPLICATE,
        )

        projection = await harness.store.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(original),
                correlation=InteractionCorrelation.from_request(original),
            )
        )
        assert projection == created.record

    backend.run(contract)


@pytest.mark.parametrize(
    ("policy", "requests", "expected_code"),
    (
        (
            InteractionPolicy(maximum_pending_interactions_per_process=1),
            (
                _request("process-1", reason="First."),
                _request(
                    "process-2",
                    origin=_origin(run_id="other-run", branch_id="other"),
                    reason="Second.",
                ),
            ),
            InputErrorCode.CAPACITY_EXCEEDED,
        ),
        (
            InteractionPolicy(maximum_unresolved_interactions_per_run=1),
            (
                _request("run-1", reason="First."),
                _request(
                    "run-2",
                    origin=_origin(branch_id="sibling"),
                    reason="Second.",
                ),
            ),
            InputErrorCode.CAPACITY_EXCEEDED,
        ),
        (
            InteractionPolicy(
                maximum_unresolved_required_interactions_per_branch=1
            ),
            (
                _request("required-1", reason="First."),
                _request("required-2", reason="Second."),
            ),
            InputErrorCode.CAPACITY_EXCEEDED,
        ),
        (
            InteractionPolicy(maximum_equivalent_interactions_per_branch=1),
            (
                _request(
                    "equivalent-1",
                    mode=RequirementMode.ADVISORY,
                ),
                _request(
                    "equivalent-2",
                    mode=RequirementMode.ADVISORY,
                ),
            ),
            InputErrorCode.INTERACTION_LOOP_LIMIT,
        ),
    ),
    ids=("process", "run", "required-branch", "equivalent-lifetime"),
)
def test_admission_enforces_every_lifetime_limit(
    backend: _Backend,
    policy: InteractionPolicy,
    requests: tuple[InputRequest, InputRequest],
    expected_code: InputErrorCode,
) -> None:
    """Apply process, run, branch, and semantic lifetime bounds."""

    async def contract(harness: _Harness) -> None:
        first = await _create(harness, requests[0])
        assert isinstance(first, CreateInteractionApplied)
        second = await _create(harness, requests[1])
        assert isinstance(second, CreateInteractionRejected)
        _assert_rejected_code(second, expected_code)

    backend.run(contract, policy=policy)


def test_terminal_records_release_capacity_but_retain_loop_budget(
    backend: _Backend,
) -> None:
    """Separate unresolved capacity from branch-lifetime equivalence."""
    policy = InteractionPolicy(
        maximum_pending_interactions_per_process=1,
        maximum_unresolved_interactions_per_run=1,
        maximum_unresolved_required_interactions_per_branch=1,
        maximum_equivalent_interactions_per_branch=1,
    )

    async def contract(harness: _Harness) -> None:
        request = _request("terminal-capacity")
        created = await _create(harness, request)
        assert isinstance(created, CreateInteractionApplied)
        cancelled = await harness.store.cancel(
            CancelInteractionCommand(
                actor=_actor(request),
                correlation=created.record.correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=created.record.request.state_revision,
            )
        )
        assert isinstance(cancelled, CancelInteractionApplied)

        equivalent = await _create(
            harness,
            _request("terminal-equivalent"),
        )
        assert isinstance(equivalent, CreateInteractionRejected)
        _assert_rejected_code(
            equivalent,
            InputErrorCode.INTERACTION_LOOP_LIMIT,
        )

        distinct = await _create(
            harness,
            _request("terminal-distinct", reason="A new semantic request."),
        )
        assert isinstance(distinct, CreateInteractionApplied)

    backend.run(contract, policy=policy)


def _branch_command(
    request: InputRequest,
    child: str,
    parent: str,
    *,
    principal: PrincipalScope | None = None,
) -> RegisterInteractionBranchCommand:
    actor = _actor(request)
    return RegisterInteractionBranchCommand(
        actor=actor,
        registration=InteractionBranchRegistration(
            run_id=request.origin.run_id,
            branch_id=BranchId(child),
            parent_branch_id=BranchId(parent),
            principal=principal or actor.principal,
        ),
    )


def test_branch_graph_replay_conflict_cycle_owner_and_child_ancestry(
    backend: _Backend,
) -> None:
    """Enforce exact immutable ancestry and run ownership in one graph."""

    async def contract(harness: _Harness) -> None:
        root = _request("branch-root")
        root_created = await _create(harness, root)
        assert isinstance(root_created, CreateInteractionApplied)

        child_command = _branch_command(root, "child", "root")
        child = await harness.store.register_branch(child_command)
        assert isinstance(child, InteractionBranchRegistrationApplied)

        replay = await harness.store.register_branch(child_command)
        assert isinstance(replay, InteractionBranchRegistrationReplayed)
        assert replay.record == child.record

        conflict = await harness.store.register_branch(
            _branch_command(root, "child", "other-parent")
        )
        assert isinstance(conflict, InteractionBranchRegistrationRejected)
        _assert_rejected_code(conflict, InputErrorCode.CORRELATION_MISMATCH)

        owner_drift = await harness.store.register_branch(
            _branch_command(
                root,
                "other-child",
                "root",
                principal=_principal("intruder"),
            )
        )
        assert isinstance(
            owner_drift,
            InteractionBranchRegistrationRejected,
        )
        _assert_rejected_code(owner_drift, InputErrorCode.FORBIDDEN)

        grandchild = await harness.store.register_branch(
            _branch_command(root, "grandchild", "child")
        )
        assert isinstance(
            grandchild,
            InteractionBranchRegistrationApplied,
        )
        cycle = await harness.store.register_branch(
            _branch_command(root, "root", "grandchild")
        )
        assert isinstance(cycle, InteractionBranchRegistrationRejected)
        _assert_rejected_code(cycle, InputErrorCode.CORRELATION_MISMATCH)

        wrong_parent = await _create(
            harness,
            _request(
                "wrong-parent",
                origin=_origin(
                    branch_id="child",
                    parent_branch_id="other-parent",
                ),
            ),
        )
        assert isinstance(wrong_parent, CreateInteractionRejected)
        _assert_rejected_code(
            wrong_parent,
            InputErrorCode.CORRELATION_MISMATCH,
        )

        exact_child = await _create(
            harness,
            _request(
                "exact-child",
                origin=_origin(
                    branch_id="child",
                    parent_branch_id="root",
                ),
            ),
        )
        assert isinstance(exact_child, CreateInteractionApplied)

    backend.run(contract)


def test_branch_ids_are_isolated_by_principal_scope(
    backend: _Backend,
) -> None:
    """Allow identical branch identifiers in separate principal scopes."""
    shared_run_id = RunId("branch-scope-isolation")
    first_request = _request(
        "branch-scope-first",
        origin=replace(
            _origin(branch_id="first-root"),
            run_id=shared_run_id,
            principal=_principal("first-owner"),
        ),
    )
    second_request = _request(
        "branch-scope-second",
        origin=replace(
            _origin(branch_id="second-root"),
            run_id=shared_run_id,
            principal=_principal("second-owner"),
        ),
    )
    observed_roots: list[InteractionBranchRoot] = []

    async def contract(harness: _Harness) -> None:
        first = await _create(harness, first_request)
        second = await _create(harness, second_request)
        assert isinstance(first, CreateInteractionApplied)
        assert isinstance(second, CreateInteractionApplied)

        for request in (first_request, second_request):
            registered = await harness.store.register_branch(
                _branch_command(
                    request,
                    "shared-child",
                    str(request.origin.branch_id),
                )
            )
            assert isinstance(
                registered,
                InteractionBranchRegistrationApplied,
            )

        first_root = await harness.store.lookup_branch_root(
            InteractionBranchRootLookup(
                actor=_actor(first_request),
                run_id=shared_run_id,
                branch_id=BranchId("shared-child"),
            )
        )
        second_root = await harness.store.lookup_branch_root(
            InteractionBranchRootLookup(
                actor=_actor(second_request),
                run_id=shared_run_id,
                branch_id=BranchId("shared-child"),
            )
        )
        observed_roots.extend((first_root, second_root))

    backend.run(contract)
    assert observed_roots == [
        InteractionBranchRoot(
            run_id=shared_run_id,
            branch_id=BranchId("shared-child"),
            root_branch_id=first_request.origin.branch_id,
        ),
        InteractionBranchRoot(
            run_id=shared_run_id,
            branch_id=BranchId("shared-child"),
            root_branch_id=second_request.origin.branch_id,
        ),
    ]


def test_branch_root_lookup_reconstructs_edge_only_ancestry_after_reopen(
    backend: _Backend,
) -> None:
    """Resolve opaque cousin roots without relying on interaction records."""

    async def contract(harness: _Harness) -> None:
        seed = _request("branch-root-seed")
        for child, parent in (
            ("B", "A"),
            ("C", "B"),
            ("X", "A"),
            ("D", "X"),
        ):
            registered = await harness.store.register_branch(
                _branch_command(seed, child, parent)
            )
            assert isinstance(registered, InteractionBranchRegistrationApplied)

        actor = _actor(seed)
        c_query = InteractionBranchRootLookup(
            actor=actor,
            run_id=seed.origin.run_id,
            branch_id=BranchId("C"),
        )
        d_query = replace(c_query, branch_id=BranchId("D"))
        expected_c = InteractionBranchRoot(
            run_id=seed.origin.run_id,
            branch_id=BranchId("C"),
            root_branch_id=BranchId("A"),
        )
        expected_d = replace(expected_c, branch_id=BranchId("D"))

        assert await harness.store.lookup_branch_root(c_query) == expected_c
        assert await harness.store.lookup_branch_root(d_query) == expected_d
        assert harness.authorizer.calls[-1] == (
            actor,
            InteractionOperation.INSPECT_BRANCH,
            d_query.authorization_target,
        )

        await harness.store.aclose()
        harness.store = await harness.factory.open()
        assert await harness.store.lookup_branch_root(c_query) == expected_c
        assert await harness.store.lookup_branch_root(d_query) == expected_d

        for absent in (
            replace(c_query, run_id=RunId("wrong-run")),
            replace(c_query, branch_id=BranchId("missing")),
            replace(
                c_query,
                actor=InteractionActor(principal=_principal("intruder")),
            ),
        ):
            call_count = len(harness.authorizer.calls)
            assert await harness.store.lookup_branch_root(absent) is None
            assert len(harness.authorizer.calls) == call_count + 1
            assert (
                harness.authorizer.calls[-1][1]
                is InteractionOperation.INSPECT_BRANCH
            )
            assert (
                harness.authorizer.calls[-1][2] == absent.authorization_target
            )

        harness.authorizer.allowed = False
        call_count = len(harness.authorizer.calls)
        assert await harness.store.lookup_branch_root(c_query) is None
        assert len(harness.authorizer.calls) == call_count + 1

    backend.run(contract)


def test_missing_and_denied_lookups_and_waits_are_indistinguishable(
    backend: _Backend,
) -> None:
    """Expose neither record existence nor authorization through reads."""

    async def contract(harness: _Harness) -> None:
        existing_request = _request("existing")
        created = await _create(harness, existing_request)
        assert isinstance(created, CreateInteractionApplied)
        missing_request = _request("missing", reason="Missing.")

        missing_query = ScopedInteractionLookup(
            actor=_actor(missing_request),
            correlation=InteractionCorrelation.from_request(missing_request),
        )
        assert await harness.store.lookup_scoped(missing_query) is None

        harness.authorizer.allowed = False
        denied_query = ScopedInteractionLookup(
            actor=_actor(existing_request),
            correlation=created.record.correlation,
        )
        assert await harness.store.lookup_scoped(denied_query) is None

        failures: list[InteractionNotFoundError] = []
        for query in (missing_query, denied_query):
            with pytest.raises(InteractionNotFoundError) as captured:
                await harness.store.wait_for_change(
                    WaitForInteractionChangeCommand(
                        actor=query.actor,
                        correlation=query.correlation,
                        after_store_revision=InteractionStoreRevision(1),
                    )
                )
            failures.append(captured.value)
        assert failures[0].args == failures[1].args
        assert failures[0].code is InputErrorCode.NOT_FOUND

    backend.run(contract)


def test_empty_scope_and_list_are_authorized_with_exact_echo(
    backend: _Backend,
) -> None:
    """Authorize even an empty scope through its immutable target."""

    async def contract(harness: _Harness) -> None:
        actor = InteractionActor(principal=_principal())
        command = ListInteractionsCommand(
            actor=actor,
            scope=InteractionExecutionScope(run_id=RunId("empty-run")),
        )
        assert await harness.store.list_scoped(command) == ()
        assert harness.authorizer.calls == [
            (
                actor,
                InteractionOperation.LIST,
                command.authorization_target,
            )
        ]
        assert isinstance(
            harness.authorizer.calls[0][2],
            InteractionScopeAuthorizationTarget,
        )

    backend.run(contract)


@pytest.mark.parametrize("scope_operation", ("cancel", "supersede"))
def test_empty_scope_mutation_is_authorized_before_no_op(
    backend: _Backend,
    scope_operation: str,
) -> None:
    """Authorize an empty scope rather than treating absence as permission."""

    async def contract(harness: _Harness) -> None:
        actor = InteractionActor(principal=_principal())
        scope = InteractionExecutionScope(run_id=RunId("empty-scope"))
        if scope_operation == "cancel":
            cancel_command = TerminalizeInteractionScopeCommand(
                actor=actor,
                scope=scope,
                provenance=AnswerProvenance.HUMAN,
            )
            cancel_result = await harness.store.terminalize_scope(
                cancel_command
            )
            assert isinstance(cancel_result, ScopeCancellationReplayed)
            expected_operation = InteractionOperation.CANCEL_SCOPE
            target = cancel_command.authorization_target
            records = cancel_result.records
            mutation_applied = cancel_result.store_mutation_applied
        else:
            supersede_command = SupersedeInteractionScopeCommand(
                actor=actor,
                scope=scope,
                provenance=AnswerProvenance.HUMAN,
            )
            supersede_result = await harness.store.supersede_scope(
                supersede_command
            )
            assert isinstance(supersede_result, ScopeSupersessionReplayed)
            expected_operation = InteractionOperation.SUPERSEDE
            target = supersede_command.authorization_target
            records = supersede_result.records
            mutation_applied = supersede_result.store_mutation_applied
        assert records == ()
        assert not mutation_applied
        assert harness.authorizer.calls == [
            (actor, expected_operation, target)
        ]

    backend.run(contract)


@pytest.mark.parametrize("scope_operation", ("cancel", "supersede"))
def test_scope_mutation_isolates_mixed_owners_and_rejects_foreign_only(
    backend: _Backend,
    scope_operation: str,
) -> None:
    """Mutate actor rows, reject foreign-only scope, and replay absence."""
    contract_completed = False

    async def contract(harness: _Harness) -> None:
        nonlocal contract_completed
        actor_principal = _principal("scope-actor")
        foreign_principal = _principal("scope-foreign")
        shared_run_id = "scope-shared-owner-run"
        actor_root = _request(
            f"{scope_operation}-actor-root",
            origin=_origin(
                run_id=shared_run_id,
                branch_id="shared-root",
                principal=actor_principal,
            ),
            reason="Actor root request.",
        )
        foreign_root = _request(
            f"{scope_operation}-foreign-root",
            origin=_origin(
                run_id=shared_run_id,
                branch_id="shared-root",
                principal=foreign_principal,
            ),
            reason="Foreign root request.",
        )
        actor_root_created = await _create(harness, actor_root)
        foreign_root_created = await _create(harness, foreign_root)
        assert isinstance(actor_root_created, CreateInteractionApplied)
        assert isinstance(foreign_root_created, CreateInteractionApplied)

        for root in (actor_root, foreign_root):
            branch = await harness.store.register_branch(
                _branch_command(root, "shared-child", "shared-root")
            )
            assert isinstance(
                branch,
                InteractionBranchRegistrationApplied,
            )
        actor_child = _request(
            f"{scope_operation}-actor-child",
            origin=_origin(
                run_id=shared_run_id,
                turn_id="child-turn",
                branch_id="shared-child",
                parent_branch_id="shared-root",
                principal=actor_principal,
            ),
            reason="Actor child request.",
        )
        foreign_child = _request(
            f"{scope_operation}-foreign-child",
            origin=_origin(
                run_id=shared_run_id,
                turn_id="child-turn",
                branch_id="shared-child",
                parent_branch_id="shared-root",
                principal=foreign_principal,
            ),
            reason="Foreign child request.",
        )
        actor_child_created = await _create(harness, actor_child)
        foreign_child_created = await _create(harness, foreign_child)
        assert isinstance(actor_child_created, CreateInteractionApplied)
        assert isinstance(foreign_child_created, CreateInteractionApplied)

        actor = InteractionActor(principal=actor_principal)
        mixed_scope = InteractionExecutionScope(
            run_id=RunId(shared_run_id),
            branch_id=BranchId("shared-root"),
            include_descendants=True,
        )

        async def mutate(
            scope: InteractionExecutionScope,
        ) -> object:
            if scope_operation == "cancel":
                return await harness.store.terminalize_scope(
                    TerminalizeInteractionScopeCommand(
                        actor=actor,
                        scope=scope,
                        provenance=AnswerProvenance.HUMAN,
                    )
                )
            return await harness.store.supersede_scope(
                SupersedeInteractionScopeCommand(
                    actor=actor,
                    scope=scope,
                    provenance=AnswerProvenance.HUMAN,
                )
            )

        applied = await mutate(mixed_scope)
        if scope_operation == "cancel":
            assert isinstance(applied, ScopeCancellationApplied)
            expected_status = ResolutionStatus.CANCELLED
        else:
            assert isinstance(applied, ScopeSupersessionApplied)
            expected_status = ResolutionStatus.SUPERSEDED
        assert {record.request.request_id for record in applied.records} == {
            actor_root.request_id,
            actor_child.request_id,
        }
        assert all(
            record.request.resolution is not None
            and record.request.resolution.status is expected_status
            for record in applied.records
        )

        replayed = await mutate(mixed_scope)
        if scope_operation == "cancel":
            assert isinstance(replayed, ScopeCancellationReplayed)
        else:
            assert isinstance(replayed, ScopeSupersessionReplayed)
        assert replayed.records == ()

        foreign_records = await harness.store.list_scoped(
            ListInteractionsCommand(
                actor=foreign_root_created.command.actor,
                scope=InteractionExecutionScope(
                    run_id=RunId(shared_run_id),
                ),
            )
        )
        foreign_request_ids: set[InputRequestId] = set()
        for record in foreign_records:
            assert isinstance(record, InteractionRecord)
            foreign_request_ids.add(record.request.request_id)
            assert record.request.state is RequestState.PENDING
            assert record.request.resolution is None
        assert foreign_request_ids == {
            foreign_root.request_id,
            foreign_child.request_id,
        }

        branch_precedence_run = f"{scope_operation}-branch-precedence-run"
        actor_branch_seed = _request(
            f"{scope_operation}-actor-requestless-branch",
            origin=_origin(
                run_id=branch_precedence_run,
                branch_id="precedence-root",
                principal=actor_principal,
            ),
        )
        actor_branch = await harness.store.register_branch(
            _branch_command(
                actor_branch_seed,
                "precedence-branch",
                "precedence-root",
            )
        )
        assert isinstance(
            actor_branch,
            InteractionBranchRegistrationApplied,
        )
        foreign_precedence_request = _request(
            f"{scope_operation}-foreign-precedence",
            origin=_origin(
                run_id=branch_precedence_run,
                branch_id="precedence-branch",
                principal=foreign_principal,
            ),
            reason="Foreign record overrides actor branch presence.",
        )
        foreign_precedence_created = await _create(
            harness,
            foreign_precedence_request,
        )
        assert isinstance(
            foreign_precedence_created,
            CreateInteractionApplied,
        )
        precedence = await mutate(
            InteractionExecutionScope(
                run_id=RunId(branch_precedence_run),
                branch_id=BranchId("precedence-branch"),
            )
        )
        if scope_operation == "cancel":
            assert isinstance(precedence, ScopeCancellationRejected)
        else:
            assert isinstance(precedence, ScopeSupersessionRejected)
        assert precedence.error.code is InputErrorCode.FORBIDDEN

        foreign_only_request = _request(
            f"{scope_operation}-foreign-only",
            origin=replace(
                _origin(
                    run_id="scope-foreign-only-run",
                    turn_id="foreign-only-turn",
                    branch_id="foreign-only-root",
                    principal=foreign_principal,
                ),
                agent_id=AgentId("foreign-only-agent"),
            ),
            reason="Foreign-only request.",
        )
        foreign_only_created = await _create(
            harness,
            foreign_only_request,
        )
        assert isinstance(foreign_only_created, CreateInteractionApplied)
        foreign_only_scope = InteractionExecutionScope(
            run_id=foreign_only_request.origin.run_id,
            turn_id=foreign_only_request.origin.turn_id,
            agent_id=foreign_only_request.origin.agent_id,
            branch_id=foreign_only_request.origin.branch_id,
        )

        harness.authorizer.allowed = False
        denied = await mutate(foreign_only_scope)
        harness.authorizer.allowed = True
        if scope_operation == "cancel":
            assert isinstance(denied, ScopeCancellationRejected)
        else:
            assert isinstance(denied, ScopeSupersessionRejected)
        assert denied.error.code is InputErrorCode.FORBIDDEN

        foreign_only = await mutate(foreign_only_scope)
        if scope_operation == "cancel":
            assert isinstance(foreign_only, ScopeCancellationRejected)
        else:
            assert isinstance(foreign_only, ScopeSupersessionRejected)
        assert foreign_only.error.code is InputErrorCode.FORBIDDEN

        empty = await mutate(
            InteractionExecutionScope(run_id=RunId("truly-empty-scope"))
        )
        if scope_operation == "cancel":
            assert isinstance(empty, ScopeCancellationReplayed)
        else:
            assert isinstance(empty, ScopeSupersessionReplayed)
        assert empty.records == ()

        foreign_projection = await harness.store.lookup_scoped(
            ScopedInteractionLookup(
                actor=foreign_only_created.command.actor,
                correlation=foreign_only_created.record.correlation,
            )
        )
        assert foreign_projection == foreign_only_created.record
        contract_completed = True

    backend.run(
        contract,
        policy=InteractionPolicy(
            maximum_pending_interactions_per_process=10,
        ),
    )
    assert contract_completed


def test_full_and_terminal_metadata_projections_follow_exact_decision(
    backend: _Backend,
) -> None:
    """Return full owner data or content-safe terminal metadata only."""

    async def contract(harness: _Harness) -> None:
        request = _request("projection")
        created = await _create(harness, request)
        assert isinstance(created, CreateInteractionApplied)
        resolved = await harness.store.resolve(_decline(created.record, "key"))
        assert isinstance(resolved, ResolveInteractionApplied)
        query = ScopedInteractionLookup(
            actor=_actor(request),
            correlation=resolved.record.correlation,
        )

        full = await harness.store.lookup_scoped(query)
        assert full == resolved.record

        harness.authorizer.disclosure = InteractionDisclosure.TERMINAL_METADATA
        metadata = await harness.store.lookup_scoped(query)
        assert isinstance(metadata, InteractionTerminalMetadata)
        assert metadata.status is ResolutionStatus.DECLINED
        resolution = resolved.record.request.resolution
        assert resolution is not None
        assert metadata.resolved_at == resolution.resolved_at

        _, operation, target = harness.authorizer.calls[-1]
        assert operation is InteractionOperation.INSPECT
        assert isinstance(target, InteractionRequestAuthorizationTarget)
        assert target.request_id == request.request_id
        assert target.origin == request.origin

    backend.run(contract)


def test_slow_authorizer_never_holds_the_backing_lock(
    backend: _Backend,
) -> None:
    """Allow an unrelated commit while one authorization call is blocked."""

    async def contract(harness: _Harness) -> None:
        first_request = _request("slow-authorization")
        first = await _create(harness, first_request)
        assert isinstance(first, CreateInteractionApplied)
        harness.authorizer.block_operation = InteractionOperation.INSPECT
        lookup = create_task(
            harness.store.lookup_scoped(
                ScopedInteractionLookup(
                    actor=_actor(first_request),
                    correlation=first.record.correlation,
                )
            )
        )
        await harness.authorizer.entered.wait()

        second = create_task(
            _create(
                harness,
                _request(
                    "unrelated-commit",
                    origin=_origin(run_id="run-2", branch_id="root-2"),
                    reason="Unrelated.",
                ),
            )
        )
        completed_without_release = await _yield_until_done(
            cast(Task[object], second)
        )
        harness.authorizer.release.set()
        assert completed_without_release
        assert isinstance(await second, CreateInteractionApplied)
        assert await lookup == first.record

    backend.run(contract)


def test_commit_uses_trusted_time_and_clock_failure_has_no_effect(
    backend: _Backend,
) -> None:
    """Read commit time inside the store and fail before write or wake."""

    async def contract(harness: _Harness) -> None:
        request = _request("commit-clock")
        created = await _create(harness, request)
        assert isinstance(created, CreateInteractionApplied)
        waiter = create_task(
            harness.store.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=_actor(request),
                    correlation=created.record.correlation,
                    after_store_revision=created.record.store_revision,
                )
            )
        )
        await _yield_once()

        harness.clock.failure = RuntimeError("injected clock failure")
        cancel_command = CancelInteractionCommand(
            actor=_actor(request),
            correlation=created.record.correlation,
            provenance=AnswerProvenance.HUMAN,
            expected_state_revision=created.record.request.state_revision,
        )
        with pytest.raises(RuntimeError, match="injected clock failure"):
            await harness.store.cancel(cancel_command)
        assert not await _yield_until_done(cast(Task[object], waiter))

        unchanged = await harness.store.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(request),
                correlation=created.record.correlation,
            )
        )
        assert unchanged == created.record

        harness.clock.advance(7)
        cancelled = await harness.store.cancel(cancel_command)
        assert isinstance(cancelled, CancelInteractionApplied)
        assert cancelled.record.request.resolution is not None
        assert (
            cancelled.record.request.resolution.resolved_at
            == _NOW + timedelta(seconds=7)
        )
        assert await waiter == cancelled.record

    backend.run(contract)


def test_resolution_idempotency_replay_conflict_and_bounded_ledger(
    backend: _Backend,
) -> None:
    """Apply same-key and semantic-new-key rules before lifecycle CAS."""
    policy = InteractionPolicy(maximum_idempotency_keys_per_request=2)

    async def contract(harness: _Harness) -> None:
        request = _request("idempotency")
        created = await _create(harness, request)
        assert isinstance(created, CreateInteractionApplied)
        first_command = _answer(created.record, "key-1")
        first = await harness.store.resolve(first_command)
        assert isinstance(first, ResolveInteractionApplied)

        same_key = await harness.store.resolve(first_command)
        assert isinstance(same_key, InteractionStoreReplayed)
        assert same_key.replay_kind is InteractionReplayKind.SAME_KEY
        assert same_key.record == first.record
        assert not same_key.store_mutation_applied

        conflict = await harness.store.resolve(
            _answer(created.record, "key-1", value=False)
        )
        assert isinstance(conflict, ResolveInteractionRejected)
        _assert_rejected_code(conflict, InputErrorCode.IDEMPOTENCY_CONFLICT)

        waiter = create_task(
            harness.store.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=_actor(request),
                    correlation=first.record.correlation,
                    after_store_revision=first.record.store_revision,
                )
            )
        )
        semantic_command = replace(
            _answer(created.record, "key-2"),
            expected_state_revision=StateRevision(0),
        )
        semantic = await harness.store.resolve(semantic_command)
        assert isinstance(semantic, InteractionStoreReplayed)
        assert semantic.replay_kind is InteractionReplayKind.SEMANTIC_NEW_KEY
        assert semantic.store_mutation_applied
        assert semantic.record.request == first.record.request
        assert (
            semantic.record.store_revision == first.record.store_revision + 1
        )
        assert tuple(
            entry.key for entry in semantic.record.idempotency_ledger
        ) == (
            ResolutionIdempotencyKey("key-1"),
            ResolutionIdempotencyKey("key-2"),
        )
        assert await waiter == semantic.record

        full = await harness.store.resolve(
            replace(
                _answer(created.record, "key-3"),
                expected_state_revision=StateRevision(0),
            )
        )
        assert isinstance(full, ResolveInteractionRejected)
        _assert_rejected_code(full, InputErrorCode.IDEMPOTENCY_LEDGER_FULL)
        projection = await harness.store.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(request),
                correlation=first.record.correlation,
            )
        )
        assert projection == semantic.record

    backend.run(contract, policy=policy)


@pytest.mark.parametrize("callback_fails", (False, True), ids=("ok", "fails"))
def test_registered_resumer_is_attempted_once_after_commit_without_rollback(
    backend: _Backend,
    callback_fails: bool,
) -> None:
    """Keep a commit authoritative across resumer delivery failure."""

    async def contract(harness: _Harness) -> None:
        contexts: list[dict[str, Any]] = []

        def exception_handler(
            loop: AbstractEventLoop,
            context: dict[str, Any],
        ) -> None:
            del loop
            contexts.append(context)

        get_running_loop().set_exception_handler(exception_handler)
        resumer = _RecordingResumer(fail=callback_fails)
        request = _request(
            f"resumer-{callback_fails}",
            reason=f"Resumer {callback_fails}.",
        )
        created = await _create(harness, request, resumer=resumer)
        assert isinstance(created, CreateInteractionApplied)
        assert resumer.notifications == []

        command = _answer(created.record, "resumer-key")
        resolved = await harness.store.resolve(command)
        assert isinstance(resolved, ResolveInteractionApplied)
        assert len(resumer.notifications) == 1
        notification = resumer.notifications[0]
        assert notification.continuation_id == request.continuation_id
        assert (
            notification.state_revision
            == resolved.record.request.state_revision
        )

        committed = await harness.store.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(request),
                correlation=resolved.record.correlation,
            )
        )
        assert committed == resolved.record
        replay = await harness.store.resolve(command)
        assert isinstance(replay, InteractionStoreReplayed)
        assert len(resumer.notifications) == 1

        if callback_fails:
            assert len(contexts) == 1
            error = contexts[0].get("exception")
            assert isinstance(error, InteractionResumptionDeliveryError)
            assert contexts[0].get("message") == error.safe_message
            assert "sensitive raw" not in error.safe_message
        else:
            assert contexts == []

    backend.run(contract)


def test_capability_cleanup_is_authoritative_idempotent_and_reopen_safe(
    backend: _Backend,
) -> None:
    """Settle only an exact admission without public authorization."""

    async def contract(harness: _Harness) -> None:
        request = _request(
            "capability-cleanup",
            reason="Capability-bound cleanup.",
        )
        resumer = _RecordingResumer(fail=False)
        create, cleanup = _new_interaction_admission_commands(
            actor=_actor(request),
            request=request,
            resumer=resumer,
        )
        created = await harness.store.create_admission(create)
        assert isinstance(created, CreateInteractionApplied)
        if harness.reopen_after_first_create:
            await harness.store.aclose()
            harness.store = await harness.factory.open()

        unrelated_request = _request(
            "unbound-capability",
            reason="Never submit this admission.",
        )
        _, unrelated_cleanup = _new_interaction_admission_commands(
            actor=_actor(unrelated_request),
            request=unrelated_request,
            resumer=_RecordingResumer(fail=False),
        )
        absent = await harness.store.cleanup_admission(unrelated_cleanup)
        assert (
            absent.disposition
            is _InteractionAdmissionCleanupDisposition.ABSENT
        )
        assert resumer.notifications == []

        harness.authorizer.allowed = False
        authorization_calls = len(harness.authorizer.calls)
        settled = await harness.store.cleanup_admission(cleanup)
        assert (
            settled.disposition
            is _InteractionAdmissionCleanupDisposition.SETTLED
        )
        assert len(harness.authorizer.calls) == authorization_calls
        assert len(resumer.notifications) == 1

        repeated = await harness.store.cleanup_admission(cleanup)
        assert (
            repeated.disposition
            is _InteractionAdmissionCleanupDisposition.TERMINAL
        )
        assert len(harness.authorizer.calls) == authorization_calls
        assert len(resumer.notifications) == 1

        harness.authorizer.allowed = True
        projection = await harness.store.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(request),
                correlation=created.record.correlation,
            )
        )
        assert isinstance(projection, InteractionRecord)
        assert projection.request.state is RequestState.UNAVAILABLE
        assert projection.request.resolution is not None
        assert (
            projection.request.resolution.provenance is AnswerProvenance.POLICY
        )

    backend.run(contract)


def test_rejected_admission_cleanup_proves_definitive_absence(
    backend: _Backend,
) -> None:
    """Return content-free absence after create rejection without probing."""

    async def contract(harness: _Harness) -> None:
        request = _request(
            "rejected-capability-cleanup",
            reason="Reject this sealed admission.",
        )
        create, cleanup = _new_interaction_admission_commands(
            actor=_actor(request),
            request=request,
            resumer=_RecordingResumer(fail=False),
        )
        harness.authorizer.allowed = False
        rejected = await harness.store.create_admission(create)
        assert isinstance(rejected, CreateInteractionRejected)
        authorization_calls = len(harness.authorizer.calls)

        absent = await harness.store.cleanup_admission(cleanup)
        assert (
            absent.disposition
            is _InteractionAdmissionCleanupDisposition.ABSENT
        )
        assert len(harness.authorizer.calls) == authorization_calls

    backend.run(contract)


def test_record_and_deadline_waiters_have_no_registration_race(
    backend: _Backend,
) -> None:
    """Observe commits whether they race before or after waiter enrollment."""

    async def contract(harness: _Harness) -> None:
        seed = _request(
            "waiter-race-seed",
            origin=_origin(run_id="seed-run", branch_id="seed-root"),
            reason="Seed the reopened backing.",
        )
        seeded = await _create(harness, seed)
        assert isinstance(seeded, CreateInteractionApplied)
        seeded_terminal = await harness.store.cancel(
            CancelInteractionCommand(
                actor=_actor(seed),
                correlation=seeded.record.correlation,
                provenance=AnswerProvenance.HUMAN,
            )
        )
        assert isinstance(seeded_terminal, CancelInteractionApplied)
        deadline_before = await harness.store.next_deadline()
        deadline_waiter = create_task(
            harness.store.wait_for_deadline_change(
                WaitForDeadlineChangeCommand(
                    after_schedule_revision=(
                        deadline_before.schedule_revision
                    ),
                )
            )
        )
        request = _request("waiter-race")
        created = await _create(harness, request)
        assert isinstance(created, CreateInteractionApplied)
        changed_deadline = await deadline_waiter
        assert (
            changed_deadline.schedule_revision
            > deadline_before.schedule_revision
        )
        assert changed_deadline.deadline is not None
        assert changed_deadline.deadline.request_id == request.request_id

        record_waiter = create_task(
            harness.store.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=_actor(request),
                    correlation=created.record.correlation,
                    after_store_revision=created.record.store_revision,
                )
            )
        )
        cancelled = await harness.store.cancel(
            CancelInteractionCommand(
                actor=_actor(request),
                correlation=created.record.correlation,
                provenance=AnswerProvenance.HUMAN,
            )
        )
        assert isinstance(cancelled, CancelInteractionApplied)
        assert await record_waiter == cancelled.record

    backend.run(contract)


def test_rejections_preserve_revisions_and_do_not_wake_waiters(
    backend: _Backend,
) -> None:
    """Keep both record and schedule revisions stable on rejected writes."""

    async def contract(harness: _Harness) -> None:
        request = _request("rejected-waiters")
        created = await _create(harness, request)
        assert isinstance(created, CreateInteractionApplied)
        schedule = await harness.store.next_deadline()
        record_waiter = create_task(
            harness.store.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=_actor(request),
                    correlation=created.record.correlation,
                    after_store_revision=created.record.store_revision,
                )
            )
        )
        deadline_waiter = create_task(
            harness.store.wait_for_deadline_change(
                WaitForDeadlineChangeCommand(
                    after_schedule_revision=schedule.schedule_revision,
                )
            )
        )

        rejected = await harness.store.mark_presented(
            PresentInteractionCommand(
                actor=_actor(request),
                correlation=created.record.correlation,
                expected_store_revision=InteractionStoreRevision(0),
            )
        )
        assert not rejected.store_mutation_applied
        assert not await _yield_until_done(cast(Task[object], record_waiter))
        assert not await _yield_until_done(cast(Task[object], deadline_waiter))
        assert await harness.store.next_deadline() == schedule

        cancelled = await harness.store.cancel(
            CancelInteractionCommand(
                actor=_actor(request),
                correlation=created.record.correlation,
                provenance=AnswerProvenance.HUMAN,
            )
        )
        assert isinstance(cancelled, CancelInteractionApplied)
        assert await record_waiter == cancelled.record
        assert (
            await deadline_waiter
        ).schedule_revision == schedule.schedule_revision + 1

    backend.run(contract)


def test_required_deadline_is_only_absolute_and_ties_are_stable(
    backend: _Backend,
) -> None:
    """Never fabricate an advisory timer and order equal deadlines by ID."""

    async def contract(harness: _Harness) -> None:
        root = _request("deadline-root", reason="Deadline root.")
        root_result = await _create(harness, root)
        assert isinstance(root_result, CreateInteractionApplied)
        for name, branch in (
            ("request-b", "branch-b"),
            ("request-a", "branch-a"),
        ):
            registered = await harness.store.register_branch(
                _branch_command(root, branch, "root")
            )
            assert isinstance(
                registered,
                InteractionBranchRegistrationApplied,
            )
            request = _request(
                name,
                origin=_origin(
                    branch_id=branch,
                    parent_branch_id="root",
                ),
                continuation_ttl_seconds=60,
                reason=f"Deadline {branch}.",
            )
            created = await _create(harness, request)
            assert isinstance(created, CreateInteractionApplied)
            assert created.record.advisory_wait is None

        snapshot = await harness.store.next_deadline()
        assert snapshot.deadline is not None
        assert snapshot.deadline.request_id == InputRequestId("request-a")
        assert snapshot.deadline.monotonic_deadline == 60.0

    backend.run(contract)


def test_due_batch_uses_one_observation_bound_and_order(
    backend: _Backend,
) -> None:
    """Settle at most the bound in deadline and request-identifier order."""

    async def contract(harness: _Harness) -> None:
        requests = (
            _request(
                "due-b",
                origin=_origin(run_id="run-b", branch_id="root-b"),
                reason="Due B.",
                continuation_ttl_seconds=60,
            ),
            _request(
                "due-a",
                origin=_origin(run_id="run-a", branch_id="root-a"),
                reason="Due A.",
                continuation_ttl_seconds=60,
            ),
        )
        for request in requests:
            assert isinstance(
                await _create(harness, request),
                CreateInteractionApplied,
            )
        harness.clock.advance(60)
        before_reads = harness.clock.read_count
        first = await harness.store.terminalize_due(
            TerminalizeDueInteractionsCommand(maximum_results=1)
        )
        assert isinstance(first, DueInteractionsApplied)
        assert harness.clock.read_count == before_reads + 1
        assert tuple(
            record.request.request_id for record in first.records
        ) == (InputRequestId("due-a"),)
        assert first.records[0].request.resolution is not None
        assert (
            first.records[0].request.resolution.status
            is ResolutionStatus.EXPIRED
        )

        second = await harness.store.terminalize_due(
            TerminalizeDueInteractionsCommand(maximum_results=1)
        )
        assert isinstance(second, DueInteractionsApplied)
        assert tuple(
            record.request.request_id for record in second.records
        ) == (InputRequestId("due-b"),)

    backend.run(contract)


@pytest.mark.parametrize("scope_operation", ("cancel", "supersede"))
def test_scope_batch_has_mixed_winners_one_generation_and_sibling_isolation(
    backend: _Backend,
    scope_operation: str,
) -> None:
    """Commit one due winner and one scope winner as one atomic batch."""

    async def contract(harness: _Harness) -> None:
        root = _request(
            f"scope-root-{scope_operation}",
            mode=RequirementMode.ADVISORY,
            reason=f"Scope root {scope_operation}.",
        )
        root_created = await _create(harness, root)
        assert isinstance(root_created, CreateInteractionApplied)
        root_resolved = await harness.store.resolve(
            _decline(root_created.record, "root-key")
        )
        assert isinstance(root_resolved, ResolveInteractionApplied)

        selected_requests: list[InputRequest] = []
        for name, ttl in (("scope-due", 60), ("scope-command", 600)):
            branch = f"{name}-{scope_operation}"
            registered = await harness.store.register_branch(
                _branch_command(root, branch, "root")
            )
            assert isinstance(
                registered,
                InteractionBranchRegistrationApplied,
            )
            request = _request(
                f"{name}-{scope_operation}",
                origin=_origin(
                    branch_id=branch,
                    parent_branch_id="root",
                ),
                reason=f"{name} {scope_operation}.",
                continuation_ttl_seconds=ttl,
            )
            assert isinstance(
                await _create(harness, request),
                CreateInteractionApplied,
            )
            selected_requests.append(request)

        sibling = _request(
            f"scope-sibling-{scope_operation}",
            origin=_origin(
                run_id=f"sibling-run-{scope_operation}",
                branch_id="sibling-root",
            ),
            reason=f"Sibling {scope_operation}.",
        )
        sibling_created = await _create(harness, sibling)
        assert isinstance(sibling_created, CreateInteractionApplied)
        schedule_before = await harness.store.next_deadline()
        harness.clock.advance(60)
        reads_before = harness.clock.read_count

        scope = InteractionExecutionScope(
            run_id=root.origin.run_id,
            branch_id=root.origin.branch_id,
            include_descendants=True,
        )
        if scope_operation == "cancel":
            cancel_result = await harness.store.terminalize_scope(
                TerminalizeInteractionScopeCommand(
                    actor=_actor(root),
                    scope=scope,
                    provenance=AnswerProvenance.HUMAN,
                )
            )
            assert isinstance(cancel_result, ScopeCancellationApplied)
            command_status = ResolutionStatus.CANCELLED
            records = cancel_result.records
        else:
            supersede_result = await harness.store.supersede_scope(
                SupersedeInteractionScopeCommand(
                    actor=_actor(root),
                    scope=scope,
                    provenance=AnswerProvenance.HUMAN,
                )
            )
            assert isinstance(supersede_result, ScopeSupersessionApplied)
            command_status = ResolutionStatus.SUPERSEDED
            records = supersede_result.records
        assert harness.clock.read_count == reads_before + 1
        statuses = {
            record.request.request_id: record.request.resolution.status
            for record in records
            if record.request.resolution is not None
        }
        assert statuses == {
            selected_requests[0].request_id: ResolutionStatus.EXPIRED,
            selected_requests[1].request_id: command_status,
        }
        schedule_after = await harness.store.next_deadline()
        assert (
            schedule_after.schedule_revision
            == schedule_before.schedule_revision + 1
        )

        sibling_projection = await harness.store.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(sibling),
                correlation=sibling_created.record.correlation,
            )
        )
        assert sibling_projection == sibling_created.record

    backend.run(contract)


@pytest.mark.parametrize(
    ("deadline_kind", "operation"),
    (
        (deadline_kind, operation)
        for deadline_kind in ("absolute", "advisory", "lease")
        for operation in ("resolve", "cancel", "terminalize")
    ),
)
def test_mutation_precedence_at_absolute_advisory_and_lease_equality(
    backend: _Backend,
    deadline_kind: str,
    operation: str,
) -> None:
    """Apply frozen deadline precedence at exact clock equality."""

    async def contract(harness: _Harness) -> None:
        mode = (
            RequirementMode.REQUIRED
            if deadline_kind == "absolute"
            else RequirementMode.ADVISORY
        )
        request = _request(
            f"precedence-{deadline_kind}-{operation}",
            mode=mode,
            continuation_ttl_seconds=(
                60 if deadline_kind == "absolute" else 600
            ),
            advisory_wait_seconds=10,
            reason=f"Precedence {deadline_kind} {operation}.",
        )
        created = await _create(harness, request)
        assert isinstance(created, CreateInteractionApplied)
        record = created.record
        if deadline_kind in {"advisory", "lease"}:
            presented = await harness.store.mark_presented(
                PresentInteractionCommand(
                    actor=_actor(request),
                    correlation=record.correlation,
                    expected_store_revision=record.store_revision,
                )
            )
            assert isinstance(presented, InteractionPresentationApplied)
            record = presented.record
        if deadline_kind == "lease":
            acquired = await harness.store.record_activity(
                RecordControllerActivityCommand(
                    actor=_actor(request),
                    correlation=record.correlation,
                    evidence=AcquireControllerActivity(
                        request_id=request.request_id,
                        controller_id=ControllerId("controller"),
                    ),
                )
            )
            assert isinstance(acquired, ControllerActivityApplied)
            record = acquired.record

        harness.clock.advance(
            60
            if deadline_kind == "absolute"
            else (10 if deadline_kind == "advisory" else 30)
        )
        result: object
        if operation == "resolve":
            result = await harness.store.resolve(
                _answer(record, "precedence-key")
            )
        elif operation == "cancel":
            result = await harness.store.cancel(
                CancelInteractionCommand(
                    actor=_actor(request),
                    correlation=record.correlation,
                    provenance=AnswerProvenance.HUMAN,
                )
            )
        else:
            result = await harness.store.terminalize(
                TerminalizeInteractionCommand(
                    actor=_actor(request),
                    correlation=record.correlation,
                    status=ResolutionStatus.UNAVAILABLE,
                    provenance=AnswerProvenance.HUMAN,
                )
            )

        if deadline_kind == "lease":
            assert isinstance(result, ControllerLeaseExpiredApplied)
            assert result.record.request.resolution is None
        else:
            assert isinstance(result, ResolveInteractionApplied)
            assert result.record.request.resolution is not None
            assert result.record.request.resolution.status is (
                ResolutionStatus.EXPIRED
                if deadline_kind == "absolute"
                else ResolutionStatus.TIMED_OUT
            )

    backend.run(contract)


def test_next_deadline_tracks_running_paused_and_resumed_advisory_budget(
    backend: _Backend,
) -> None:
    """Track schedule revisions across active-control pause and release."""

    async def contract(harness: _Harness) -> None:
        request = _request(
            "deadline-advisory",
            mode=RequirementMode.ADVISORY,
            continuation_ttl_seconds=600,
            advisory_wait_seconds=10,
            reason="Advisory schedule.",
        )
        created = await _create(harness, request)
        assert isinstance(created, CreateInteractionApplied)
        queued = await harness.store.next_deadline()
        assert queued.deadline is not None
        assert queued.deadline.monotonic_deadline == 600.0

        harness.clock.advance(2)
        presented = await harness.store.mark_presented(
            PresentInteractionCommand(
                actor=_actor(request),
                correlation=created.record.correlation,
                expected_store_revision=created.record.store_revision,
            )
        )
        assert isinstance(presented, InteractionPresentationApplied)
        running = await harness.store.next_deadline()
        assert running.schedule_revision == queued.schedule_revision + 1
        assert running.deadline is not None
        assert running.deadline.monotonic_deadline == 12.0

        harness.clock.advance(1)
        acquired = await harness.store.record_activity(
            RecordControllerActivityCommand(
                actor=_actor(request),
                correlation=presented.record.correlation,
                evidence=AcquireControllerActivity(
                    request_id=request.request_id,
                    controller_id=ControllerId("controller"),
                ),
            )
        )
        assert isinstance(acquired, ControllerActivityApplied)
        paused = await harness.store.next_deadline()
        assert paused.schedule_revision == running.schedule_revision + 1
        assert paused.deadline is not None
        assert paused.deadline.monotonic_deadline == 33.0

        harness.clock.advance(2)
        assert acquired.lease_nonce is not None
        released = await harness.store.record_activity(
            RecordControllerActivityCommand(
                actor=_actor(request),
                correlation=acquired.record.correlation,
                evidence=ReleaseControllerActivity(
                    request_id=request.request_id,
                    controller_id=ControllerId("controller"),
                    lease_nonce=acquired.lease_nonce,
                    sequence=1,
                ),
            )
        )
        assert isinstance(released, ControllerActivityApplied)
        resumed = await harness.store.next_deadline()
        assert resumed.schedule_revision == paused.schedule_revision + 1
        assert resumed.deadline is not None
        assert resumed.deadline.monotonic_deadline == 14.0
        assert await harness.store.next_deadline() == resumed

    backend.run(contract)


def test_barrier_controller_activity_loses_at_lease_expiry_equality(
    backend: _Backend,
) -> None:
    """Expire a lease exactly once when activity and scheduler race."""

    async def contract(harness: _Harness) -> None:
        request = _request(
            "controller-lease-barrier",
            mode=RequirementMode.ADVISORY,
            continuation_ttl_seconds=600,
            advisory_wait_seconds=10,
            reason="Controller lease barrier.",
        )
        created = await _create(harness, request)
        assert isinstance(created, CreateInteractionApplied)
        presented = await harness.store.mark_presented(
            PresentInteractionCommand(
                actor=_actor(request),
                correlation=created.record.correlation,
                expected_store_revision=created.record.store_revision,
            )
        )
        assert isinstance(presented, InteractionPresentationApplied)
        acquired = await harness.store.record_activity(
            RecordControllerActivityCommand(
                actor=_actor(request),
                correlation=presented.record.correlation,
                evidence=AcquireControllerActivity(
                    request_id=request.request_id,
                    controller_id=ControllerId("controller"),
                ),
            )
        )
        assert isinstance(acquired, ControllerActivityApplied)
        lease_nonce = acquired.lease_nonce
        assert lease_nonce is not None
        harness.clock.advance(30)
        start = Event()

        async def pulse() -> object:
            await start.wait()
            return await harness.store.record_activity(
                RecordControllerActivityCommand(
                    actor=_actor(request),
                    correlation=acquired.record.correlation,
                    evidence=PulseControllerActivity(
                        request_id=request.request_id,
                        controller_id=ControllerId("controller"),
                        lease_nonce=lease_nonce,
                        sequence=1,
                    ),
                )
            )

        async def settle_due() -> object:
            await start.wait()
            return await harness.store.terminalize_due(
                TerminalizeDueInteractionsCommand(maximum_results=1)
            )

        pulse_task = create_task(pulse())
        due_task = create_task(settle_due())
        start.set()
        results = await gather(pulse_task, due_task)
        assert (
            sum(
                bool(getattr(result, "store_mutation_applied", False))
                for result in results
            )
            == 1
        )

        projection = await harness.store.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(request),
                correlation=acquired.record.correlation,
            )
        )
        assert isinstance(projection, InteractionRecord)
        assert projection.request.resolution is None
        assert projection.store_revision == acquired.record.store_revision + 1
        assert projection.advisory_wait is not None
        assert projection.advisory_wait.controller_id is None
        assert projection.advisory_wait.lease_nonce is None

    backend.run(contract)


@pytest.mark.parametrize(
    "opponent",
    ("answer", "decline", "cancel", "timeout", "expiry", "supersede"),
)
def test_barrier_first_winner_matrix(
    backend: _Backend,
    opponent: str,
) -> None:
    """Commit exactly one terminal winner for every concurrent opponent."""

    async def contract(harness: _Harness) -> None:
        mode = (
            RequirementMode.ADVISORY
            if opponent == "timeout"
            else RequirementMode.REQUIRED
        )
        request = _request(
            f"barrier-{opponent}",
            mode=mode,
            continuation_ttl_seconds=(60 if opponent == "expiry" else 600),
            advisory_wait_seconds=10,
            reason=f"Barrier {opponent}.",
        )
        created = await _create(harness, request)
        assert isinstance(created, CreateInteractionApplied)
        record = created.record
        if opponent == "timeout":
            presented = await harness.store.mark_presented(
                PresentInteractionCommand(
                    actor=_actor(request),
                    correlation=record.correlation,
                    expected_store_revision=record.store_revision,
                )
            )
            assert isinstance(presented, InteractionPresentationApplied)
            record = presented.record
            harness.clock.advance(10)
        elif opponent == "expiry":
            harness.clock.advance(60)

        start = Event()

        async def gated(operation: Callable[[], Awaitable[object]]) -> object:
            await start.wait()
            return await operation()

        async def answer_operation() -> object:
            return await harness.store.resolve(_answer(record, "answer-key"))

        async def opponent_operation() -> object:
            if opponent == "answer":
                return await harness.store.resolve(
                    _answer(record, "opponent-key", value=False)
                )
            if opponent == "decline":
                return await harness.store.resolve(
                    _decline(record, "opponent-key")
                )
            if opponent == "cancel":
                return await harness.store.cancel(
                    CancelInteractionCommand(
                        actor=_actor(request),
                        correlation=record.correlation,
                        provenance=AnswerProvenance.HUMAN,
                    )
                )
            if opponent in {"timeout", "expiry"}:
                return await harness.store.terminalize_due(
                    TerminalizeDueInteractionsCommand(maximum_results=1)
                )
            return await harness.store.supersede_scope(
                SupersedeInteractionScopeCommand(
                    actor=_actor(request),
                    scope=InteractionExecutionScope(
                        run_id=request.origin.run_id,
                    ),
                    provenance=AnswerProvenance.HUMAN,
                )
            )

        answer_task = create_task(gated(answer_operation))
        opponent_task = create_task(gated(opponent_operation))
        start.set()
        results = await gather(answer_task, opponent_task)
        assert (
            sum(
                bool(getattr(result, "store_mutation_applied", False))
                for result in results
            )
            == 1
        )

        projection = await harness.store.lookup_scoped(
            ScopedInteractionLookup(
                actor=_actor(request),
                correlation=record.correlation,
            )
        )
        assert isinstance(projection, InteractionRecord)
        assert projection.request.resolution is not None
        assert (
            projection.request.state_revision
            == record.request.state_revision + 1
        )
        expected_statuses = {
            "answer": {ResolutionStatus.ANSWERED},
            "decline": {
                ResolutionStatus.ANSWERED,
                ResolutionStatus.DECLINED,
            },
            "cancel": {
                ResolutionStatus.ANSWERED,
                ResolutionStatus.CANCELLED,
            },
            "timeout": {ResolutionStatus.TIMED_OUT},
            "expiry": {ResolutionStatus.EXPIRED},
            "supersede": {
                ResolutionStatus.ANSWERED,
                ResolutionStatus.SUPERSEDED,
            },
        }
        assert (
            projection.request.resolution.status in expected_statuses[opponent]
        )

    backend.run(contract)
