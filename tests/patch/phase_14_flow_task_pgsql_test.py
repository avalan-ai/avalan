"""Exercise Phase 14 selected-runtime task recovery over PostgreSQL."""

from asyncio import Event, run, to_thread, wait_for
from collections.abc import Awaitable, Callable
from dataclasses import replace
from hashlib import sha256
from multiprocessing import get_context
from os import environ
from pathlib import Path
from queue import Empty
from runpy import run_path
from sys import path as sys_path
from time import monotonic
from traceback import extract_tb
from uuid import uuid4

import pytest
from patch_activation_support import activated_patch_test_profile
from phase_8_store_test import (
    _APPROVAL_AUTHORITY,
    _approval,
    _correlation,
    _owner,
    _plan,
    _result,
)

from avalan.patch.coordinator import RetransmissionKey
from avalan.patch.domain import (
    AlgorithmDigest,
    CommitStepState,
    DurationTicks,
    ExpiryTick,
    LogicalPath,
    MutationState,
    OperationType,
    PatchArtifactId,
    PatchCommitOwnerId,
    PatchContextId,
    PatchDomainId,
    PatchExecutionId,
    PatchObserverCorrelationId,
    PatchRequestId,
    PatchStepId,
    PatchWorkspaceId,
)
from avalan.patch.durable_approval import PhaseFiveDurableApprovalIssuer
from avalan.patch.durable_store import (
    DurableApproval,
    DurableCommitClaim,
    DurableCommitLease,
    DurableCoordinationAccess,
    DurableCoordinationAdmission,
    DurableJournal,
    DurableJournalCursor,
    DurablePendingAccess,
    DurablePlanReference,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableRequestSnapshot,
    DurableReservation,
    DurableStoreError,
    DurableStoreErrorCode,
    DurableWorkerBinding,
)
from avalan.patch.parser import PatchInputLimits, PatchRequestParser
from avalan.patch.pgsql_store import (
    PgsqlDurablePatchStore,
    PgsqlDurablePatchStoreSettings,
    _coordination_access_matches,
    _coordination_matches,
    _coordination_parameters,
    _encode_plan,
)
from avalan.patch.policy import (
    ApprovalService,
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PolicyRouteId,
    RuntimeGrantStore,
)
from avalan.patch.protocols import (
    PatchOrchestrationChecklist,
    PatchProtocolApprovalGate,
    PatchProtocolChecklist,
    PatchProtocolError,
    PatchProtocolFlowRequest,
    PatchProtocolIdentity,
    PatchProtocolOrchestrationAdapter,
    PatchProtocolProfile,
    PatchProtocolQueuedTaskAdapter,
    PatchProtocolReservation,
    PatchProtocols,
    PatchProtocolSelectedRuntime,
    PatchProtocolSurface,
    PatchProviderCodecChecklist,
)
from avalan.patch.sandbox_commit import (
    SandboxPatchRuntimeBinder,
    SandboxPatchSdkService,
    SandboxPatchServiceConfiguration,
)
from avalan.patch.toolset import (
    PatchApprovalBinding,
    PatchCoordinatorBinding,
    PatchPersistenceBinding,
    PatchSdkHost,
    PatchToolLoader,
    PatchToolSet,
)
from avalan.pgsql import (
    PgsqlDatabase,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    task_pgsql_upgrade,
)

_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")
_BARRIERS = (
    "reservation",
    "planned",
    "owner_fence_commit_started",
    "first_effect",
)
_RECOVERY_RESULT_SECONDS = 75.0
_RECOVERY_DIAGNOSTIC_JOIN_SECONDS = 5.0

pytestmark = pytest.mark.skipif(
    _DSN is None,
    reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
)


def _profile() -> PatchProtocolProfile:
    """Return the sole complete queued-task test profile."""
    return PatchProtocolProfile(
        surface=PatchProtocolSurface.TASK,
        enabled=True,
        authenticated=True,
        loopback_only=True,
        protocol=PatchProtocolChecklist(
            canonical_input=True,
            trusted_authority=True,
            plan_approval=True,
            detached_resume=True,
            retransmission_reservation=True,
            owner_fence_before_effect=True,
            structured_terminal_result=True,
            branch_suspension=True,
            privacy_safe_events=True,
        ),
        orchestration=PatchOrchestrationChecklist(
            shared_coordinator=True,
            originating_identity=True,
            approval_or_denial=True,
            retry_blocked_after_commit=True,
            durable_resume=True,
            dependent_suspension=True,
            coordinated_parallelism=True,
            committed_state_visibility=True,
        ),
        provider_codec=PatchProviderCodecChecklist(),
    )


def _request() -> PatchProtocolFlowRequest:
    """Return one stable task-owned selected-runtime edit request."""
    return PatchProtocolFlowRequest(
        OperationType.EDIT,
        b'{"path":"note.txt","edits":['
        b'{"old_text":"before\\n","new_text":"after\\n"}]}',
        RetransmissionKey("phase14-pgsql-selected-runtime"),
        PatchObserverCorrelationId("correlation_" + "a" * 32),
        "selected_runtime_task",
    )


def _changed_request() -> PatchProtocolFlowRequest:
    """Return a same-key request whose canonical digest must be rejected."""
    request = _request()
    return PatchProtocolFlowRequest(
        request.operation,
        request.raw_arguments.replace(b"after", b"later"),
        request.retransmission_key,
        request.correlation,
        request.mutation_slot,
    )


def _phase_ten() -> dict[str, object]:
    """Load selected-runtime builders without adding production wiring."""
    sys_path.insert(0, "tests/patch")
    try:
        return run_path("tests/patch/phase_10_contract_test.py")
    finally:
        sys_path.remove("tests/patch")


class _BoundaryPgsqlStore(PgsqlDurablePatchStore):
    """Expose durable worker-loss boundaries around a real SDK effect."""

    def __init__(
        self,
        database: PgsqlDatabase,
        barrier: str | None,
        ready: object | None,
        release: object | None,
        bindings: object | None,
    ) -> None:
        """Bind one SQL store and process-visible boundary witnesses."""
        super().__init__(
            database,
            owns_database=True,
            approval_verifier=_APPROVAL_AUTHORITY,
        )
        self._barrier = barrier
        self._ready = ready
        self._release = release
        self._bindings = bindings
        self._paused = False
        self.reached = Event()

    async def reserve(
        self,
        identity: DurableRequestIdentity,
        canonical_digest: AlgorithmDigest,
        request_id: PatchRequestId | None = None,
    ) -> DurableReservation:
        """Pause only after the initial durable reservation commits."""
        reservation = await super().reserve(
            identity, canonical_digest, request_id
        )
        if not reservation.replayed:
            await self._pause("reservation")
        return reservation

    async def claim_commit(
        self,
        reservation: DurableReservation,
        plan: DurablePlanReference,
        approval: DurableApproval,
        owner_id: PatchCommitOwnerId,
        now: ExpiryTick,
        lease_duration: DurationTicks,
        artifact_ids: tuple[PatchArtifactId, ...],
    ) -> DurableCommitClaim:
        """Pause only after owner, fence, and commit start are durable."""
        claim = await super().claim_commit(
            reservation,
            plan,
            approval,
            owner_id,
            now,
            lease_duration,
            artifact_ids,
        )
        if claim.lease is not None:
            await self._pause("owner_fence_commit_started")
        return claim

    async def persist_plan(
        self,
        reservation: DurableReservation,
        plan: DurablePlanReference,
    ) -> DurableRequestSnapshot:
        """Pause after a durable plan exists but before detached approval."""
        snapshot = await super().persist_plan(reservation, plan)
        await self._pause("planned")
        return snapshot

    async def bind_worker(
        self,
        lease: DurableCommitLease,
        binding: DurableWorkerBinding,
        now: ExpiryTick,
    ) -> None:
        """Publish the exact bound worker before the real SDK effect starts."""
        await super().bind_worker(lease, binding, now)
        put = getattr(self._bindings, "put", None)
        if callable(put):
            put(binding)

    async def append_step(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        step_id: PatchStepId,
        state: CommitStepState,
        now: ExpiryTick,
    ) -> DurableJournal:
        """Pause after the selected worker journaled its completed effect."""
        journal = await super().append_step(
            lease, expected, step_id, state, now
        )
        if state is CommitStepState.COMMITTED:
            await self._pause("first_effect")
        return journal

    async def _pause(self, boundary: str) -> None:
        """Stop at the one configured process-loss boundary."""
        if self._paused or self._barrier != boundary:
            return
        self._paused = True
        self.reached.set()
        await _pause(self._ready, self._release, boundary)


async def _store(
    dsn: str,
    schema: str,
    barrier: str | None = None,
    ready: object | None = None,
    release: object | None = None,
    bindings: object | None = None,
) -> PgsqlDurablePatchStore:
    """Open one independent PostgreSQL durable-store client."""
    settings = PgsqlDurablePatchStoreSettings(
        dsn=dsn,
        schema=schema,
        pool_minimum=1,
        pool_maximum=2,
    )
    store: PgsqlDurablePatchStore
    if barrier is None:
        store = PgsqlDurablePatchStore.from_settings(
            settings, approval_verifier=_APPROVAL_AUTHORITY
        )
    else:
        store = _BoundaryPgsqlStore(
            settings.database(), barrier, ready, release, bindings
        )
    await store.open()
    return store


async def _task(
    store: PgsqlDurablePatchStore,
    root: Path,
    namespace: Path,
    *,
    recovery: bool,
    progress: Callable[[str], None] | None = None,
) -> tuple[
    PatchToolSet,
    PatchProtocolQueuedTaskAdapter,
    PatchProtocolSelectedRuntime,
    PatchProtocolIdentity,
]:
    """Bind one task adapter to the exact selected SDK service and host."""
    if progress is not None:
        progress("phase10-helper-start")
    helpers = _phase_ten()
    if progress is not None:
        progress("phase10-helper-ready")
    settings_factory = helpers["_settings"]
    subject_factory = helpers["_runtime_subject"]
    policy_factory = helpers["_sandbox_corpus_policy"]
    clock_type = helpers["_RuntimeClock"]
    planner_type = helpers["PlannerFacade"]
    worker_type = helpers["BoundedPlannerWorker"]
    limits_type = helpers["PlannerLimits"]
    assert callable(settings_factory)
    assert callable(subject_factory)
    assert callable(policy_factory)
    assert callable(clock_type)
    assert callable(planner_type)
    assert callable(worker_type)
    assert callable(limits_type)
    settings = settings_factory(root, namespace)
    subject = subject_factory()
    policy = policy_factory()
    clock = clock_type()
    if recovery:
        advance = getattr(clock, "advance", None)
        if not callable(advance):
            raise AssertionError("selected runtime clock cannot recover")
        advance(20)
    approvals = PatchProtocolApprovalGate()
    approval_service = ApprovalService(approvals, clock, RuntimeGrantStore())
    configuration = SandboxPatchServiceConfiguration(
        subject,
        planner_type(worker_type(1), limits_type()),
        approval_service,
        PhaseFiveDurableApprovalIssuer(approval_service, _APPROVAL_AUTHORITY),
        clock,
        DurationTicks(10),
        DurationTicks(10),
        execution_id=PatchExecutionId("execution_" + "a" * 32),
    )
    if progress is not None:
        progress("runtime-configured")
    binder = SandboxPatchRuntimeBinder.from_settings(
        settings,
        configuration,
        policy,
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True, store),
        PatchPersistenceBinding(True, store),
    )
    if progress is not None:
        progress("runtime-loader-start")
    bundle = await PatchToolLoader(
        binder, activated_patch_test_profile()
    ).load(enable_tools=["patch.edit", "patch.apply"])
    if progress is not None:
        progress("runtime-loader-ready")
    toolset = bundle.toolset
    binding = bundle.runtime_binding
    assert type(toolset) is PatchToolSet
    assert binding is not None
    assert type(binding.service) is SandboxPatchSdkService
    if progress is not None:
        progress("toolset-entry-start")
    await toolset.__aenter__()
    if progress is not None:
        progress("toolset-entry-ready")
    host = toolset.sdk_host()
    assert type(host) is PatchSdkHost
    assert host._service is binding.service
    identity = PatchProtocolIdentity(
        subject.tenant,
        subject.principal,
        PatchExecutionId("execution_" + "a" * 32),
        subject.run,
        subject.session,
        subject.task,
        subject.agent,
        policy.approval.route,
        settings.context.identity.context_id,
        settings.context.identity.workspace_id,
    )
    runtime = PatchProtocolSelectedRuntime(
        toolset, binding.service, store, approvals
    )
    adapter = PatchProtocolQueuedTaskAdapter(
        PatchProtocolOrchestrationAdapter(
            _profile(),
            identity,
            store,
            PatchRequestParser(PatchInputLimits()),
            runtime,
        )
    )
    if progress is not None:
        progress("recovery-entry-ready")
    return toolset, adapter, runtime, identity


async def _drop_schema(dsn: str, schema: str) -> None:
    """Drop one isolated test-owned schema after all worker pools close."""
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


async def _pause(
    ready: object | None,
    release: object | None,
    boundary: str,
) -> None:
    """Expose one completed durable boundary before process termination."""
    put = getattr(ready, "put", None)
    wait = getattr(release, "wait", None)
    if not callable(put) or not callable(wait):
        raise AssertionError("process synchronization is unavailable")
    await to_thread(put, boundary)
    await to_thread(wait, 30)


def _crash_worker(
    dsn: str,
    schema: str,
    root: str,
    namespace: str,
    barrier: str,
    ready: object,
    release: object,
    bindings: object,
) -> None:
    """Start one selected-runtime task worker and halt at one boundary."""

    async def execute() -> None:
        store = await _store(dsn, schema, barrier, ready, release, bindings)
        toolset: PatchToolSet | None = None
        try:
            toolset, task, _, _ = await _task(
                store, Path(root), Path(namespace), recovery=False
            )
            await task.execute(_request(), approve=True)
            if barrier == "first_effect":
                assert type(store) is _BoundaryPgsqlStore
                try:
                    await wait_for(store.reached.wait(), timeout=15)
                except TimeoutError as error:
                    raise AssertionError(
                        "selected worker did not reach post-effect boundary"
                    ) from error
            raise AssertionError("worker passed its configured crash boundary")
        finally:
            if toolset is not None:
                await toolset.__aexit__(None, None, None)
            await store.aclose()

    put = getattr(ready, "put", None)
    if not callable(put):
        raise AssertionError("process result channel is unavailable")
    try:
        run(execute())
    except BaseException as error:
        frames = extract_tb(error.__traceback__)
        put(
            (
                "error",
                type(error).__name__,
                tuple((frame.name, frame.lineno) for frame in frames[-4:]),
                str(error),
            )
        )
    else:
        put(("completed", barrier))


def _request_id(
    identity: PatchProtocolIdentity,
    request: PatchProtocolFlowRequest,
) -> PatchRequestId:
    """Return the stable queued-task request identity without a retry UUID."""
    parts = (
        PatchProtocolSurface.TASK.value,
        identity.tenant.value,
        identity.principal.value,
        identity.execution.value,
        identity.run.value,
        identity.session.value,
        identity.task.value,
        identity.agent.value,
        identity.route.value,
        identity.context.value,
        identity.workspace.value,
        request.mutation_slot,
    )
    return PatchRequestId(
        "request_" + sha256("\x1f".join(parts).encode()).hexdigest()[:32]
    )


async def _original_reservation(
    store: PgsqlDurablePatchStore,
    identity: PatchProtocolIdentity,
) -> PatchProtocolReservation:
    """Read the original durable authority through a replay reservation."""
    request = _request()
    protocol = PatchProtocols(_profile(), identity)
    return await protocol.reserve(
        store,
        request.operation,
        request.raw_arguments,
        request.retransmission_key,
        request.correlation,
        PatchRequestParser(PatchInputLimits()),
        request_id=_request_id(identity, request),
    )


def _drain_process_messages(channel: object) -> tuple[object, ...]:
    """Return every immediately available process message in order."""
    get_nowait = getattr(channel, "get_nowait", None)
    if not callable(get_nowait):
        raise AssertionError("process message channel is unavailable")
    messages: list[object] = []
    while True:
        try:
            messages.append(get_nowait())
        except Empty:
            return tuple(messages)


def _wrong_identity(identity: PatchProtocolIdentity) -> PatchProtocolIdentity:
    """Return an execution-mismatched authority for restart rejection."""
    return PatchProtocolIdentity(
        identity.tenant,
        identity.principal,
        PatchExecutionId("execution_" + "b" * 32),
        identity.run,
        identity.session,
        identity.task,
        identity.agent,
        identity.route,
        identity.context,
        identity.workspace,
    )


def _recover_worker(
    dsn: str,
    schema: str,
    root: str,
    namespace: str,
    barrier: str,
    worker_binding: DurableWorkerBinding | None,
    progress: object,
    results: object,
) -> None:
    """Attach a fresh selected runtime without a second target effect."""

    async def recover() -> tuple[str, str, str]:
        put_progress = getattr(progress, "put", None)
        if not callable(put_progress):
            raise AssertionError("recovery progress channel is unavailable")
        put_progress("store-open-start")
        store = await _store(dsn, schema)
        put_progress("store-opened")
        toolset: PatchToolSet | None = None
        try:
            toolset, task, runtime, identity = await _task(
                store,
                Path(root),
                Path(namespace),
                recovery=barrier != "planned",
                progress=put_progress,
            )
            put_progress("task")
            reservation = await _original_reservation(store, identity)
            put_progress("reservation")
            assert reservation.durable.replayed
            wrong = PatchProtocolReservation(
                reservation.surface,
                _wrong_identity(identity),
                reservation.operation,
                reservation.correlation,
                reservation.durable,
            )
            with pytest.raises(PatchProtocolError):
                await runtime.plan(
                    wrong, reservation.operation, _request().raw_arguments
                )
            with pytest.raises(DurableStoreError):
                await task.recover(_changed_request())
            put_progress("rejections")
            assert not runtime._tasks
            note = Path(root) / "note.txt"
            expected_before = "after\n" if worker_binding else "before\n"
            assert note.read_text(encoding="utf-8") == expected_before
            if worker_binding is not None:
                snapshot = await store.inspect(
                    DurableRequestAccess(
                        reservation.request_id, reservation.durable.identity
                    )
                )
                assert snapshot.lease is not None
                assert tuple(
                    step.state for step in snapshot.journal.steps
                ) == (
                    CommitStepState.PLANNED,
                    CommitStepState.COMMITTED,
                )
                await store.mark_worker_reaped(snapshot.lease, worker_binding)
                put_progress("reaped")
            put_progress("recover")
            put_progress("recovery-call-start")
            continuation = await task.recover(_request())
            put_progress("recovery-call-ready")
            put_progress("recovered")
            if barrier == "reservation":
                assert continuation.kind.value == "approval_required"
                return ("approval_required", "before", "zero")
            if barrier == "planned":
                assert continuation.kind.value == "approval_required"
                assert continuation.result is None
                approved = await task.execute(_request(), approve=True)
                if approved.result is None:
                    pending = approved.pending
                    if pending is not None:
                        await store.await_terminal(
                            DurablePendingAccess(
                                DurableRequestAccess(
                                    pending.request_id,
                                    reservation.durable.identity,
                                ),
                                pending.pending_operation_id,
                                pending.correlation_id,
                            )
                        )
                    approved = await task.recover(_request())
                assert approved.result is not None
                assert approved.result.status.value == "committed"
                assert note.read_text(encoding="utf-8") == "after\n"
                return ("planned", "committed", "one")
            assert continuation.result is not None
            status = (
                "committed" if worker_binding is not None else "indeterminate"
            )
            assert continuation.result.status.value == status
            if worker_binding is not None:
                settled = await store.inspect(
                    DurableRequestAccess(
                        reservation.request_id, reservation.durable.identity
                    )
                )
                assert tuple(step.state for step in settled.journal.steps) == (
                    CommitStepState.PLANNED,
                    CommitStepState.COMMITTED,
                )
            return (
                "terminal",
                status,
                "one" if worker_binding is not None else "zero",
            )
        finally:
            if toolset is not None:
                await toolset.__aexit__(None, None, None)
            await store.aclose()

    put = getattr(results, "put", None)
    if not callable(put):
        raise AssertionError("process result channel is unavailable")
    try:
        put(("ok", *run(recover())))
    except BaseException as error:
        frames = extract_tb(error.__traceback__)
        put(
            (
                "error",
                type(error).__name__,
                str(error),
                tuple((frame.name, frame.lineno) for frame in frames[-5:]),
            )
        )


@pytest.mark.parametrize("barrier", _BARRIERS)
def test_patch_e2e_034_pgsql_task_selected_runtime_recovery(
    barrier: str,
    tmp_path: Path,
) -> None:
    """Recover each selected-service crash boundary without an effect retry."""
    assert _DSN is not None

    async def scenario() -> tuple[str, str]:
        schema = "patch_phase14_task_" + uuid4().hex
        root = tmp_path / "sandbox-view"
        namespace = tmp_path / "sandbox-private"
        root.mkdir()
        namespace.mkdir()
        note = root / "note.txt"
        note.write_text("before\n", encoding="utf-8")
        await to_thread(
            task_pgsql_upgrade,
            PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
        )
        context = get_context("spawn")
        ready = context.Queue()
        release = context.Event()
        bindings = context.Queue()
        crashed = context.Process(
            target=_crash_worker,
            args=(
                _DSN,
                schema,
                str(root),
                str(namespace),
                barrier,
                ready,
                release,
                bindings,
            ),
        )
        recovered = context.Queue()
        progress = context.Queue()
        fresh: object | None = None
        worker_binding: DurableWorkerBinding | None = None
        try:
            crashed.start()
            try:
                observed = ready.get(timeout=60)
            except Empty as error:
                raise AssertionError(
                    "selected worker did not reach crash boundary"
                ) from error
            assert observed == barrier
            if barrier == "first_effect":
                worker_binding = bindings.get(timeout=20)
                assert type(worker_binding) is DurableWorkerBinding
                assert note.read_text(encoding="utf-8") == "after\n"
            else:
                assert note.read_text(encoding="utf-8") == "before\n"
            crashed.kill()
            crashed.join(30)
            assert crashed.exitcode is not None and crashed.exitcode != 0
            fresh = context.Process(
                target=_recover_worker,
                args=(
                    _DSN,
                    schema,
                    str(root),
                    str(namespace),
                    barrier,
                    worker_binding,
                    progress,
                    recovered,
                ),
            )
            fresh.start()
            expected = (
                ("ok", "approval_required", "before", "zero")
                if barrier == "reservation"
                else (
                    ("ok", "planned", "committed", "one")
                    if barrier == "planned"
                    else (
                        "ok",
                        "terminal",
                        (
                            "committed"
                            if barrier == "first_effect"
                            else "indeterminate"
                        ),
                        "one" if barrier == "first_effect" else "zero",
                    )
                )
            )
            try:
                result_wait_started = monotonic()
                result = await to_thread(
                    recovered.get,
                    True,
                    _RECOVERY_RESULT_SECONDS,
                )
            except Empty as error:
                await to_thread(fresh.join, _RECOVERY_DIAGNOSTIC_JOIN_SECONDS)
                result_messages = _drain_process_messages(recovered)
                progress_messages = _drain_process_messages(progress)
                elapsed = monotonic() - result_wait_started
                raise AssertionError(
                    "fresh selected worker did not report within "
                    + repr(_RECOVERY_RESULT_SECONDS)
                    + " seconds; diagnostic_join_seconds="
                    + repr(elapsed)
                    + " progress="
                    + repr(progress_messages)
                    + " results="
                    + repr(result_messages)
                    + " exitcode="
                    + repr(fresh.exitcode)
                ) from error
            assert result == expected
            fresh.join(75)
            assert fresh.exitcode == 0
            final_contents = note.read_text(encoding="utf-8")
            assert final_contents == (
                "after\n"
                if barrier in {"planned", "first_effect"}
                else "before\n"
            )
        finally:
            if crashed.is_alive():
                crashed.kill()
                crashed.join(20)
            if (
                fresh is not None
                and getattr(fresh, "is_alive", lambda: False)()
            ):
                getattr(fresh, "kill")()
                getattr(fresh, "join")(1)
            await _drop_schema(_DSN, schema)
        return barrier, final_contents

    observed = run(scenario())
    assert observed[0] == barrier
    assert observed[1] == (
        "after\n" if barrier in {"planned", "first_effect"} else "before\n"
    )


def _coordination_admission(
    reservation: DurableReservation,
    agent: str,
    workspace_token: str = "c",
) -> DurableCoordinationAdmission:
    """Return one full agent-owned admission for one shared workspace."""
    return DurableCoordinationAdmission(
        DurableCoordinationAccess(
            reservation,
            PatchRunId("run-phase14-coordination-" + agent),
            PatchSessionId("session-phase14-coordination-" + agent),
            PatchTaskId("task-phase14-coordination-" + agent),
            PatchAgentId("agent-phase14-coordination-" + agent),
            PatchContextId("context_" + agent * 32),
            PatchWorkspaceId("workspace_" + workspace_token * 32),
            PatchDomainId("domain_" + workspace_token * 32),
        ),
        frozenset((LogicalPath("note.txt"),)),
    )


def test_patch_phase_14_pgsql_coordination_conflicts_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain exact SQL ownership across collisions, plans, and terminals."""
    assert _DSN is not None

    async def terminalize(
        store: PgsqlDurablePatchStore,
        reservation: DurableReservation,
        token: str,
    ) -> None:
        """Settle one real request before testing terminal admission denial."""
        plan = _plan(reservation.canonical_digest, token)
        await store.persist_plan(reservation, plan)
        claim = await store.claim_commit(
            reservation,
            plan,
            _approval(
                reservation.identity,
                reservation.canonical_digest,
                plan,
                token,
            ),
            _owner(token),
            ExpiryTick(1),
            DurationTicks(10),
            (),
        )
        assert claim.lease is not None
        snapshot = await store.inspect(
            DurableRequestAccess(reservation.request_id, reservation.identity)
        )
        journal = snapshot.journal
        for step in plan.steps:
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
            _result(reservation.request_id, plan, MutationState.COMMITTED),
            _correlation(token),
            ExpiryTick(4),
        )

    async def scenario() -> None:
        """Exercise durable coordination through a fresh owned schema."""
        schema = "p14_coord_conf_" + uuid4().hex
        await to_thread(
            task_pgsql_upgrade,
            PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
        )
        store = await _store(_DSN, schema)
        try:
            first_identity = DurableRequestIdentity(
                PatchTenantId("tenant-phase14-coordination"),
                PatchPrincipalId("principal-phase14-coordination"),
                PatchExecutionId("execution_" + "a" * 32),
                PolicyRouteId("route-phase14-coordination"),
                RetransmissionKey("phase14-coordination-a"),
            )
            first = await store.reserve(
                first_identity,
                AlgorithmDigest("sha256", "a" * 64),
                PatchRequestId("request_" + "a" * 32),
            )
            conflicting_identity = DurableRequestIdentity(
                PatchTenantId("tenant-phase14-coordination"),
                PatchPrincipalId("principal-phase14-coordination"),
                PatchExecutionId("execution_" + "b" * 32),
                PolicyRouteId("route-phase14-coordination"),
                RetransmissionKey("phase14-coordination-b"),
            )
            with pytest.raises(DurableStoreError):
                await store.reserve(
                    conflicting_identity,
                    AlgorithmDigest("sha256", "b" * 64),
                    first.request_id,
                )
            admission = _coordination_admission(first, "a")
            await store.admit_coordination(admission)
            await store.admit_coordination(admission)
            assert await store.is_coordination_admitted(admission.access)
            second = await store.reserve(
                conflicting_identity,
                AlgorithmDigest("sha256", "b" * 64),
                PatchRequestId("request_" + "b" * 32),
            )
            with pytest.raises(DurableStoreError):
                await store.admit_coordination(
                    _coordination_admission(second, "b")
                )
            with pytest.raises(DurableStoreError):
                await store.release_coordination(
                    _coordination_admission(second, "b").access
                )
            await store.release_coordination(admission.access)
            await store.release_coordination(admission.access)
            assert not await store.is_coordination_admitted(admission.access)

            planned = await store.reserve(
                DurableRequestIdentity(
                    PatchTenantId("tenant-phase14-coordination"),
                    PatchPrincipalId("principal-phase14-coordination"),
                    PatchExecutionId("execution_" + "c" * 32),
                    PolicyRouteId("route-phase14-coordination"),
                    RetransmissionKey("phase14-coordination-c"),
                ),
                AlgorithmDigest("sha256", "c" * 64),
                PatchRequestId("request_" + "c" * 32),
            )
            planned_admission = _coordination_admission(planned, "c")
            await store.admit_coordination(planned_admission)
            await store.persist_plan(
                planned, _plan(planned.canonical_digest, "c")
            )
            with pytest.raises(DurableStoreError):
                await store.release_coordination(planned_admission.access)
            with pytest.raises(DurableStoreError):
                await store.release_terminal_coordination(
                    DurableRequestAccess(planned.request_id, planned.identity)
                )
            with pytest.raises(DurableStoreError):
                _encode_plan(
                    replace(
                        _plan(planned.canonical_digest, "f"),
                        rehydration=b"invalid-without-origin",
                    )
                )

            terminal = await store.reserve(
                DurableRequestIdentity(
                    PatchTenantId("tenant-phase14-coordination"),
                    PatchPrincipalId("principal-phase14-coordination"),
                    PatchExecutionId("execution_" + "d" * 32),
                    PolicyRouteId("route-phase14-coordination"),
                    RetransmissionKey("phase14-coordination-d"),
                ),
                AlgorithmDigest("sha256", "d" * 64),
                PatchRequestId("request_" + "d" * 32),
            )
            terminal_admission = _coordination_admission(terminal, "d", "d")
            await store.admit_coordination(terminal_admission)
            assert await store.is_coordination_admitted(
                terminal_admission.access
            )
            await terminalize(store, terminal, "d")
            assert not await store.is_coordination_admitted(
                terminal_admission.access
            )
            with pytest.raises(DurableStoreError):
                await store.admit_coordination(
                    _coordination_admission(terminal, "d")
                )

            qualified = quote_pgsql_identifier(schema)
            columns = (
                '"workspace_id", "domain_id", "request_id", '
                '"tenant_id", "principal_id", "execution_id", '
                '"run_id", "session_id", "task_id", "agent_id", '
                '"route_id", "context_id", "paths_digest"'
            )
            placeholders = ", ".join(("%s",) * 13)

            async def insert_residue(values: tuple[str, ...]) -> None:
                """Model a crash residue in this test-owned SQL schema."""
                async with store._database.connection() as connection:
                    async with connection.cursor() as cursor:
                        await cursor.execute(
                            "INSERT INTO "
                            + qualified
                            + '."patch_durable_workspace_coordination" ('
                            + columns
                            + ") VALUES ("
                            + placeholders
                            + ")",
                            values,
                        )

            terminal_access = DurableRequestAccess(
                terminal.request_id, terminal.identity
            )
            matching_residue = _coordination_parameters(terminal_admission)
            await insert_residue(matching_residue)
            await store.release_terminal_coordination(terminal_access)
            assert not await store.is_coordination_admitted(
                terminal_admission.access
            )
            mismatched_residue = list(matching_residue)
            mismatched_residue[3] = "tenant-phase14-stale"
            await insert_residue(tuple(mismatched_residue))
            with pytest.raises(DurableStoreError):
                await store.release_terminal_coordination(terminal_access)
            async with store._database.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        "DELETE FROM "
                        + qualified
                        + '."patch_durable_workspace_coordination" '
                        'WHERE "request_id" = %s',
                        (terminal.request_id.value,),
                    )

            malformed_row: dict[str, object] = {}
            assert not _coordination_access_matches(
                malformed_row, admission.access
            )
            assert not _coordination_matches(malformed_row, admission)
            matching_row: dict[str, object] = {
                "workspace_id": admission.access.workspace_id.value,
                "domain_id": admission.access.domain_id.value,
                "request_id": admission.access.reservation.request_id.value,
                "tenant_id": (
                    admission.access.reservation.identity.tenant_id.value
                ),
                "principal_id": (
                    admission.access.reservation.identity.principal_id.value
                ),
                "execution_id": (
                    admission.access.reservation.identity.execution_id.value
                ),
                "run_id": admission.access.run_id.value,
                "session_id": admission.access.session_id.value,
                "task_id": admission.access.task_id.value,
                "agent_id": admission.access.agent_id.value,
                "route_id": (
                    admission.access.reservation.identity.route_id.value
                ),
                "context_id": admission.access.context_id.value,
                "paths_digest": "a" * 64,
            }

            def unavailable_digest(
                _: DurableCoordinationAdmission,
            ) -> AlgorithmDigest:
                """Model an unavailable digest dependency and fail closed."""
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )

            with monkeypatch.context() as patched:
                patched.setattr(
                    "avalan.patch.pgsql_store._coordination_paths_digest",
                    unavailable_digest,
                )
                assert not _coordination_matches(matching_row, admission)
        finally:
            await store.aclose()
            await _drop_schema(_DSN, schema)

    run(scenario())


def test_patch_phase_14_pgsql_reservation_races_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject request-ID and lifecycle races observed after SQL insert loss."""

    class _Cursor:
        """Return finite SQL responses for one public reservation race."""

        def __init__(self, rows: tuple[dict[str, object] | None, ...]) -> None:
            """Retain one semantic database response sequence."""
            self._rows = iter(rows)

        async def execute(self, statement: str, parameters: object) -> None:
            """Accept one parameterized query from the durable store."""
            del statement, parameters

        async def fetchone(self) -> dict[str, object] | None:
            """Return the next row observed by the public transaction."""
            return next(self._rows)

    async def scenario() -> None:
        """Drive public reserve through DB-observed unique and lost races."""
        store = PgsqlDurablePatchStore(
            type("Pool", (), {"connection": lambda self: None})(),
            approval_verifier=_APPROVAL_AUTHORITY,
        )
        cursor: _Cursor

        async def transaction(
            operation: str,
            callback: Callable[[object], Awaitable[object]],
        ) -> object:
            """Execute the public callback against its exact row sequence."""
            del operation
            return await callback(cursor)

        monkeypatch.setattr(store, "_transaction", transaction)

        async def lost_identity(
            cursor_value: object,
            identity_value: DurableRequestIdentity,
        ) -> None:
            """Model a row deleted after conflict but before identity lock."""
            del cursor_value, identity_value
            return None

        monkeypatch.setattr(
            "avalan.patch.pgsql_store._select_identity_for_update",
            lost_identity,
        )
        identity = DurableRequestIdentity(
            PatchTenantId("tenant-phase14-reservation"),
            PatchPrincipalId("principal-phase14-reservation"),
            PatchExecutionId("execution_" + "a" * 32),
            PolicyRouteId("route-phase14-reservation"),
            RetransmissionKey("phase14-reservation-a"),
        )
        cursor = _Cursor((None, {"request_id": "request_existing"}))
        with pytest.raises(DurableStoreError) as request_collision:
            await store.reserve(
                identity,
                AlgorithmDigest("sha256", "a" * 64),
                PatchRequestId("request_" + "a" * 32),
            )
        assert (
            request_collision.value.code
            is DurableStoreErrorCode.IDEMPOTENCY_CONFLICT
        )

        cursor = _Cursor((None,))
        with pytest.raises(DurableStoreError) as lost_reservation:
            await store.reserve(identity, AlgorithmDigest("sha256", "a" * 64))
        assert (
            lost_reservation.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )

    run(scenario())


def _coordination_process(
    dsn: str,
    schema: str,
    agent: str,
    marker: str,
    results: object,
) -> None:
    """Attempt one durable workspace admission from a fresh process."""

    async def execute() -> tuple[str, bool, str]:
        store = await _store(dsn, schema)
        try:
            identity = DurableRequestIdentity(
                PatchTenantId("tenant-phase14-coordination"),
                PatchPrincipalId("principal-phase14-coordination"),
                PatchExecutionId("execution_" + agent * 32),
                PolicyRouteId("route-phase14-coordination"),
                RetransmissionKey("phase14-coordination-" + agent),
            )
            reservation = await store.reserve(
                identity,
                AlgorithmDigest("sha256", agent * 64),
                PatchRequestId("request_" + agent * 32),
            )
            admission = _coordination_admission(reservation, agent)
            try:
                await store.admit_coordination(admission)
            except DurableStoreError:
                return ("denied", reservation.replayed, "zero")
            effect = Path(marker)
            try:
                with effect.open(
                    "x", encoding="ascii", errors="strict"
                ) as file:
                    file.write(agent)
            except FileExistsError:
                return ("admitted", reservation.replayed, "attached")
            return ("admitted", reservation.replayed, "one")
        finally:
            await store.aclose()

    put = getattr(results, "put", None)
    if not callable(put):
        raise AssertionError("process result channel is unavailable")
    try:
        put(("ok", *run(execute())))
    except BaseException as error:
        put(("error", type(error).__name__, str(error)))


def test_patch_e2e_035_pgsql_coordination_survives_process_restart(
    tmp_path: Path,
) -> None:
    """Retain one workspace owner and exactly one effect across processes."""
    assert _DSN is not None

    async def scenario() -> None:
        schema = "patch_phase14_coordination_" + uuid4().hex
        marker = tmp_path / "coordination-effect"
        await to_thread(
            task_pgsql_upgrade,
            PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
        )
        context = get_context("spawn")
        results = context.Queue()
        processes: list[object] = []
        try:
            for agent in ("a", "b", "a"):
                process = context.Process(
                    target=_coordination_process,
                    args=(
                        _DSN,
                        schema,
                        agent,
                        str(marker),
                        results,
                    ),
                )
                processes.append(process)
                process.start()
                try:
                    result = results.get(timeout=40)
                except Empty as error:
                    raise AssertionError(
                        "coordination worker did not report"
                    ) from error
                process.join(40)
                assert process.exitcode == 0
                if agent == "a" and len(processes) == 1:
                    assert result == ("ok", "admitted", False, "one")
                elif agent == "b":
                    assert result == ("ok", "denied", False, "zero")
                else:
                    assert result == ("ok", "admitted", True, "attached")
            assert marker.read_text(encoding="ascii") == "a"
        finally:
            for child in processes:
                if getattr(child, "is_alive", lambda: False)():
                    getattr(child, "kill")()
                    getattr(child, "join")(20)
            await _drop_schema(_DSN, schema)

    run(scenario())
