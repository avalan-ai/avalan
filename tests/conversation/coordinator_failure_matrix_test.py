"""Verify coordinator conflicts, failures, cancellation, and liveness."""

from ast import Attribute, Call, Name, parse, walk
from asyncio import CancelledError, all_tasks, gather
from collections.abc import Callable
from copy import copy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest
from phase2_fixtures import (
    NOW,
    authority,
    binding,
    child_identity,
    coordinator,
    empty_stateless_plan,
    next_stateless_plan,
    request,
    root_identity,
)

import avalan.conversation as conversation

pytestmark = pytest.mark.anyio

_FORBIDDEN_SYNC_EFFECTS = {
    "Popen",
    "connect",
    "execute",
    "executemany",
    "open",
    "read_bytes",
    "read_text",
    "request",
    "run",
    "sleep",
    "urlopen",
    "write_bytes",
    "write_text",
}


@pytest.fixture
def anyio_backend() -> str:
    """Run deterministic failure tests on asyncio only."""
    return "asyncio"


class _BoundaryFailures:
    """Raise a bounded sequence of store-boundary cleanup failures."""

    def __init__(
        self,
        failures: dict[
            conversation.StoreAwaitBoundary,
            list[BaseException],
        ],
    ) -> None:
        self._failures = failures
        self.visited: list[conversation.StoreAwaitBoundary] = []

    async def reach(
        self,
        boundary: conversation.StoreAwaitBoundary,
    ) -> None:
        self.visited.append(boundary)
        failures = self._failures.get(boundary)
        if failures:
            raise failures.pop(0)


class _ScriptedCloseStore:
    """Return exact close/probe effects without unrelated store behavior."""

    def __init__(self, close: object, inspect: object) -> None:
        self._close = close
        self._inspect = inspect
        self.close_calls = 0
        self.inspect_calls = 0

    @staticmethod
    def _resolve(effect: object) -> object:
        if isinstance(effect, BaseException):
            raise effect
        return effect

    async def close(self) -> conversation.StoreCloseResolution:
        self.close_calls += 1
        return cast(
            conversation.StoreCloseResolution,
            self._resolve(self._close),
        )

    async def inspect_close(self) -> conversation.StoreCloseResolution:
        self.inspect_calls += 1
        return cast(
            conversation.StoreCloseResolution,
            self._resolve(self._inspect),
        )


class _ScriptedSettlementStore:
    """Return bounded abandon, inspection, and reconciliation effects."""

    def __init__(
        self,
        *,
        abandon: tuple[object, ...],
        inspect: tuple[object, ...],
        reconcile: tuple[object, ...],
    ) -> None:
        self._abandon = list(abandon)
        self._inspect = list(inspect)
        self._reconcile = list(reconcile)
        self.abandon_calls = 0
        self.inspect_calls = 0
        self.reconcile_calls = 0

    @staticmethod
    def _next(effects: list[object]) -> object:
        assert effects
        effect = effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect

    async def abandon_idempotency(
        self,
        identity: conversation.RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> conversation.IdempotencySettlementResolution:
        self.abandon_calls += 1
        return cast(
            conversation.IdempotencySettlementResolution,
            self._next(self._abandon),
        )

    async def inspect_idempotency_settlement(
        self,
        identity: conversation.RequestIdempotencyIdentity,
        owner_token: str,
    ) -> conversation.IdempotencySettlementResolution:
        self.inspect_calls += 1
        return cast(
            conversation.IdempotencySettlementResolution,
            self._next(self._inspect),
        )

    async def reconcile_idempotency(
        self,
        identity: conversation.RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> conversation.IdempotencySettlementResolution:
        self.reconcile_calls += 1
        return cast(
            conversation.IdempotencySettlementResolution,
            self._next(self._reconcile),
        )


def _runtime(
    lane_binding: conversation.ProviderLaneBinding,
    results: tuple[conversation.ProviderResult, ...],
    *,
    controller: conversation.DeterministicFaultController | None = None,
    profile: conversation.ConversationCapabilityProfile | None = None,
) -> conversation.ConversationLaneRuntime:
    return conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=profile
        or conversation.fake_capability_profile(lane_binding),
        provider_script=conversation.DeterministicFakeProviderScript(
            results=results,
            controller=controller,
        ),
    )


def _synchronous_effect_calls(source: str) -> set[str]:
    """Return forbidden direct and attribute call names in source."""
    observed: set[str] = set()
    for node in walk(parse(source)):
        if not isinstance(node, Call):
            continue
        name: str | None = None
        if isinstance(node.func, Name):
            name = node.func.id
        elif isinstance(node.func, Attribute):
            name = node.func.attr
        if name in _FORBIDDEN_SYNC_EFFECTS:
            assert name is not None
            observed.add(name)
    return observed


async def _seed_root(
    *,
    results_after_root: tuple[conversation.ProviderResult, ...] = (),
) -> tuple[
    conversation.AuthorityScope,
    conversation.InMemoryConversationStore,
    conversation.RunScopedConversationCoordinator,
    conversation.ConversationLaneRuntime,
    conversation.AtomicCommitReceipt,
    conversation.ProviderResult,
]:
    scope = authority()
    lane_binding = binding()
    first_plan = empty_stateless_plan(lane_binding)
    first_result = conversation.fake_provider_result(first_plan, turn=1)
    runtime = _runtime(lane_binding, (first_result,) + results_after_root)
    store = conversation.InMemoryConversationStore()
    engine = coordinator(store=store, scope=scope, runtimes=(runtime,))
    root = await engine.execute(
        request(
            scope=scope,
            identity=root_identity("failure-root"),
            advance=conversation.FirstTurnAdvance(),
            response_suffix="failure-root",
            key="key-failure-root",
        )
    )
    return scope, store, engine, runtime, root, first_result


async def test_named_head_contenders_have_one_success_and_one_conflict(
    record_property: Callable[[str, object], None],
) -> None:
    """Resolve a deterministic named-head CAS race without interleaving."""
    record_property("conversation_acceptance_evidence", "runtime")
    lane_binding = binding()
    first_plan = empty_stateless_plan(lane_binding)
    first_result = conversation.fake_provider_result(first_plan, turn=1)
    next_plan = next_stateless_plan(lane_binding, first_result.items)
    results = (
        conversation.fake_provider_result(next_plan, turn=2, text="head-a"),
        conversation.fake_provider_result(next_plan, turn=3, text="head-b"),
    )
    scope, store, engine, runtime, root, _ = await _seed_root(
        results_after_root=results
    )
    head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("main-race"),
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=root.checkpoint.identity.checkpoint_id,
    )
    await store.create_head(head, scope)
    contenders = tuple(
        request(
            scope=scope,
            identity=child_identity(root.checkpoint, f"head-{suffix}"),
            advance=conversation.NamedHeadAdvance(
                head_id=head.head_id,
                parent_checkpoint_id=root.checkpoint.identity.checkpoint_id,
                expected_revision=head.revision,
            ),
            response_suffix=f"head-{suffix}",
            key=f"key-head-{suffix}",
        )
        for suffix in ("a", "b")
    )
    outcomes = await gather(
        *(engine.execute(item) for item in contenders),
        return_exceptions=True,
    )
    assert (
        sum(
            isinstance(item, conversation.AtomicCommitReceipt)
            for item in outcomes
        )
        == 1
    )
    assert (
        sum(
            isinstance(item, conversation.ConversationConflictError)
            for item in outcomes
        )
        == 1
    )
    advanced = await store.load_head(head.head_id, scope)
    assert advanced.revision == conversation.NamedHeadRevision(1)
    provider = runtime.provider
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert len(provider.plans) == 2


async def test_idempotency_digest_mismatch_has_zero_extra_dispatch() -> None:
    """Reject same-key semantic drift before provider dispatch."""
    (
        scope,
        store,
        _engine,
        _runtime_value,
        root,
        first_result,
    ) = await _seed_root()
    lane_binding = binding()
    plan = next_stateless_plan(lane_binding, first_result.items)
    runtime = _runtime(
        lane_binding,
        (conversation.fake_provider_result(plan, turn=2),),
    )
    engine = coordinator(store=store, scope=scope, runtimes=(runtime,))
    provider = runtime.provider
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    child = request(
        scope=scope,
        identity=child_identity(root.checkpoint, "digest"),
        advance=conversation.OrdinaryChildAdvance(
            parent_checkpoint_id=root.checkpoint.identity.checkpoint_id
        ),
        response_suffix="digest",
        key="key-digest",
    )
    await engine.execute(child)
    before = len(provider.plans)
    drifted = replace(
        child,
        semantics=replace(
            child.semantics,
            semantic_input={"text": "different-input"},
        ),
    )
    with pytest.raises(conversation.ConversationConflictError):
        await engine.execute(drifted)
    assert len(provider.plans) == before


async def test_wrong_authority_and_binding_drift_reject_predispatch() -> None:
    """Reject authority and lane drift without revealing parent existence."""
    (
        scope,
        store,
        _engine,
        _runtime_value,
        root,
        first_result,
    ) = await _seed_root()
    drift_binding = binding()
    drift_binding = replace(
        drift_binding,
        normalized_endpoint="https://different.phase2.test/v1",
    )
    next_plan = next_stateless_plan(drift_binding, first_result.items)
    drift_runtime = _runtime(
        drift_binding,
        (conversation.fake_provider_result(next_plan, turn=2),),
    )
    drift_engine = coordinator(
        store=store, scope=scope, runtimes=(drift_runtime,)
    )
    child = request(
        scope=scope,
        identity=child_identity(root.checkpoint, "drift"),
        advance=conversation.OrdinaryChildAdvance(
            parent_checkpoint_id=root.checkpoint.identity.checkpoint_id
        ),
        response_suffix="drift",
        key="key-drift",
    )
    with pytest.raises(conversation.ConversationBindingDriftError):
        await drift_engine.execute(child)
    provider = drift_runtime.provider
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert provider.plans == ()

    wrong_request = replace(
        child,
        semantics=replace(child.semantics, authority=authority("wrong")),
        idempotency_key=conversation.RequestIdempotencyKey("key-wrong"),
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await drift_engine.execute(wrong_request)
    assert provider.plans == ()

    wrong_agent_binding = binding(agent="different-agent")
    wrong_agent_plan = empty_stateless_plan(wrong_agent_binding)
    wrong_agent_runtime = _runtime(
        wrong_agent_binding,
        (conversation.fake_provider_result(wrong_agent_plan, turn=1),),
    )
    wrong_agent_store = conversation.InMemoryConversationStore()
    wrong_agent_engine = coordinator(
        store=wrong_agent_store,
        scope=scope,
        runtimes=(wrong_agent_runtime,),
    )
    wrong_agent_run = request(
        scope=scope,
        identity=root_identity("wrong-agent"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="wrong-agent",
        key="key-wrong-agent",
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await wrong_agent_engine.execute(wrong_agent_run)
    wrong_agent_provider = wrong_agent_runtime.provider
    assert isinstance(
        wrong_agent_provider,
        conversation.DeterministicFakeProviderDiagnostics,
    )
    assert wrong_agent_provider.plans == ()
    assert wrong_agent_store.diagnostics.idempotency_records == 0


async def test_known_no_dispatch_retries_but_ambiguous_dispatch_fences() -> (
    None
):
    """Retry only an effect-free transport boundary and fence ambiguity."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    retriable_error = conversation.ConversationError(
        conversation.ConversationErrorCode.STORAGE_FAILED,
        boundary=conversation.FailureBoundary.KNOWN_NO_DISPATCH_TRANSPORT,
    )
    retry_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:provider_dispatch",
                exception=retriable_error,
            ),
        )
    )
    retry_runtime = _runtime(lane_binding, (result,))
    retry_store = conversation.InMemoryConversationStore()
    retry_engine = coordinator(
        store=retry_store,
        scope=scope,
        runtimes=(retry_runtime,),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            retry_controller
        ),
    )
    run = request(
        scope=scope,
        identity=root_identity("retry"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="retry",
        key="key-retry",
    )
    await retry_engine.execute(run)
    provider = retry_runtime.provider
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert len(provider.plans) == 1
    assert retry_controller.visited.count("coordinator:provider_dispatch") == 2

    nominal_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:dispatch", exception=retriable_error
            ),
        )
    )
    nominal_runtime = _runtime(
        lane_binding, (result,), controller=nominal_controller
    )
    nominal_store = conversation.InMemoryConversationStore()
    nominal_engine = coordinator(
        store=nominal_store, scope=scope, runtimes=(nominal_runtime,)
    )
    nominal_run = replace(
        run,
        identity=root_identity("nominal-dispatch"),
        idempotency_key=conversation.RequestIdempotencyKey(
            "key-nominal-dispatch"
        ),
        provisional_response_id=conversation.ProvisionalResponseId(
            "provisional-nominal-dispatch"
        ),
        public_response_id=conversation.PublicResponseId(
            "response-nominal-dispatch"
        ),
    )
    with pytest.raises(conversation.ConversationError):
        await nominal_engine.execute(nominal_run)
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        await nominal_engine.execute(nominal_run)
    nominal_provider = nominal_runtime.provider
    assert isinstance(
        nominal_provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert len(nominal_provider.plans) == 1
    assert nominal_store.diagnostics.provisional_responses == 0

    ambiguous_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:dispatch",
                exception=conversation.ConversationAmbiguousDispatchError(),
            ),
        )
    )
    ambiguous_runtime = _runtime(
        lane_binding, (result,), controller=ambiguous_controller
    )
    ambiguous_store = conversation.InMemoryConversationStore()
    ambiguous_engine = coordinator(
        store=ambiguous_store,
        scope=scope,
        runtimes=(ambiguous_runtime,),
    )
    ambiguous_run = replace(
        run,
        identity=root_identity("ambiguous"),
        idempotency_key=conversation.RequestIdempotencyKey("key-ambiguous"),
        provisional_response_id=conversation.ProvisionalResponseId(
            "provisional-ambiguous"
        ),
        public_response_id=conversation.PublicResponseId("response-ambiguous"),
    )
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        await ambiguous_engine.execute(ambiguous_run)
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        await ambiguous_engine.execute(ambiguous_run)
    ambiguous_provider = ambiguous_runtime.provider
    assert isinstance(
        ambiguous_provider,
        conversation.DeterministicFakeProviderDiagnostics,
    )
    assert len(ambiguous_provider.plans) == 1
    assert ambiguous_store.diagnostics.provisional_responses == 0


async def test_stream_failure_after_visible_output_never_retries(
    record_property: Callable[[str, object], None],
) -> None:
    """Forbid retry after a complete visible stream item was staged."""
    record_property("conversation_acceptance_evidence", "negative")
    scope = authority()
    lane_binding = binding(streaming=True)
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:terminal",
                exception=conversation.ConversationError(
                    conversation.ConversationErrorCode.STORAGE_FAILED,
                    boundary=conversation.FailureBoundary.FAILURE_BEFORE_OUTPUT,
                ),
            ),
        )
    )
    runtime = _runtime(lane_binding, (result,), controller=controller)
    store = conversation.InMemoryConversationStore()
    engine = coordinator(store=store, scope=scope, runtimes=(runtime,))
    run = request(
        scope=scope,
        identity=root_identity("visible-failure"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="visible-failure",
        key="key-visible-failure",
    )
    with pytest.raises(conversation.ConversationError):
        await engine.stream(run)
    provider = runtime.provider
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert len(provider.plans) == 1
    assert all(stream.closed for stream in provider.streams)
    assert store.diagnostics.checkpoints == 0


async def test_stream_close_nominal_predispatch_label_never_retries() -> None:
    """Treat every error after stream open as possible dispatch."""
    scope = authority()
    lane_binding = binding(streaming=True)
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    close_error = conversation.ConversationError(
        conversation.ConversationErrorCode.STORAGE_FAILED,
        boundary=conversation.FailureBoundary.KNOWN_NO_DISPATCH_TRANSPORT,
    )
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:close",
                exception=close_error,
            ),
        )
    )
    runtime = _runtime(lane_binding, (result,), controller=controller)
    store = conversation.InMemoryConversationStore()
    engine = coordinator(store=store, scope=scope, runtimes=(runtime,))
    run = request(
        scope=scope,
        identity=root_identity("stream-close-nominal"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="stream-close-nominal",
        key="key-stream-close-nominal",
    )
    with pytest.raises(conversation.ConversationError):
        await engine.stream(run)
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        await engine.stream(run)
    provider = runtime.provider
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert len(provider.plans) == 1
    assert len(provider.streams) == 1
    assert provider.streams[0].closed
    assert store.diagnostics.provisional_responses == 0
    assert store.diagnostics.idempotency_records == 1


async def test_commit_failure_preserves_parent_and_withholds_mapping(
    record_property: Callable[[str, object], None],
) -> None:
    """Roll back a completed child attempt when atomic commit fails."""
    record_property("conversation_acceptance_evidence", "negative")
    (
        scope,
        store,
        _engine,
        _runtime_value,
        root,
        first_result,
    ) = await _seed_root()
    parent_bytes = conversation.ConversationCheckpointCodec().encode(
        root.checkpoint
    )
    lane_binding = binding()
    next_plan = next_stateless_plan(lane_binding, first_result.items)
    runtime = _runtime(
        lane_binding,
        (conversation.fake_provider_result(next_plan, turn=2),),
    )
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:commit_atomic",
                exception=conversation.ConversationStorageError(),
            ),
        )
    )
    failing_store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(controller)
    )
    root_candidate = conversation.OutwardTurnCheckpointCandidate(
        checkpoint=replace(
            root.checkpoint,
            lifecycle=conversation.CheckpointLifecycle.STAGED,
            timestamps=replace(
                root.checkpoint.timestamps,
                committed_at=None,
            ),
        ),
        public_response_id=conversation.PublicResponseId("copied-root"),
    )
    copied_root = await failing_store.create(root_candidate)
    assert conversation.ConversationCheckpointCodec().encode(copied_root) == (
        parent_bytes
    )
    engine = coordinator(store=failing_store, scope=scope, runtimes=(runtime,))
    child = request(
        scope=scope,
        identity=child_identity(copied_root, "commit-failure"),
        advance=conversation.OrdinaryChildAdvance(
            parent_checkpoint_id=copied_root.identity.checkpoint_id
        ),
        response_suffix="commit-failure",
        key="key-commit-failure",
    )
    with pytest.raises(conversation.ConversationStorageError):
        await engine.execute(child)
    restored = await failing_store.load(
        copied_root.identity.checkpoint_id, scope
    )
    assert conversation.ConversationCheckpointCodec().encode(restored) == (
        parent_bytes
    )
    assert failing_store.diagnostics.checkpoints == 1
    assert failing_store.diagnostics.public_responses == 0
    assert failing_store.diagnostics.provisional_responses == 0


@pytest.mark.parametrize(
    "boundary",
    tuple(
        item
        for item in conversation.CoordinatorAwaitBoundary
        if item
        not in {
            conversation.CoordinatorAwaitBoundary.RETRY_WAIT,
            conversation.CoordinatorAwaitBoundary.ROLLBACK,
            conversation.CoordinatorAwaitBoundary.CLOSE,
        }
    ),
)
async def test_cancellation_at_every_run_boundary_is_leak_free(
    boundary: conversation.CoordinatorAwaitBoundary,
) -> None:
    """Cancel each first-turn await boundary without leaking run resources."""
    scope = authority()
    lane_binding = binding(
        streaming=boundary.name.startswith("PROVIDER_STREAM")
    )
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    runtime = _runtime(lane_binding, (result,))
    store = conversation.InMemoryConversationStore()
    hook_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label=f"coordinator:{boundary.value}",
                exception=CancelledError(),
            ),
        )
    )
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(runtime,),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            hook_controller
        ),
    )
    run = request(
        scope=scope,
        identity=root_identity(f"cancel-{boundary.value}"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix=f"cancel-{boundary.value}",
        key=f"key-cancel-{boundary.value}",
    )
    baseline = len(all_tasks())
    with pytest.raises(CancelledError):
        if boundary.name.startswith("PROVIDER_STREAM"):
            await engine.stream(run)
        else:
            await engine.execute(run)
    assert engine.diagnostics.active_attempts == 0
    assert not store.diagnostics.locked
    assert store.diagnostics.provisional_responses == 0
    assert len(all_tasks()) == baseline
    if boundary.name.startswith("PROVIDER_STREAM"):
        provider = runtime.provider
        assert isinstance(
            provider, conversation.DeterministicFakeProviderDiagnostics
        )
        assert all(stream.closed for stream in provider.streams)


async def test_retry_rollback_and_close_cancellation_settle_resources() -> (
    None
):
    """Settle ownership at the three scenario-dependent await boundaries."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    retriable = conversation.ConversationError(
        conversation.ConversationErrorCode.STORAGE_FAILED,
        boundary=conversation.FailureBoundary.KNOWN_NO_DISPATCH_TRANSPORT,
    )
    retry_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:provider_dispatch",
                exception=retriable,
            ),
            conversation.FaultAction(
                label="coordinator:retry_wait",
                exception=CancelledError(),
            ),
        )
    )
    retry_store = conversation.InMemoryConversationStore()
    retry_engine = coordinator(
        store=retry_store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            retry_controller
        ),
    )
    retry_run = request(
        scope=scope,
        identity=root_identity("cancel-retry"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="cancel-retry",
        key="key-cancel-retry",
    )
    with pytest.raises(CancelledError):
        await retry_engine.execute(retry_run)
    assert retry_store.diagnostics.idempotency_records == 0
    assert retry_store.diagnostics.provisional_responses == 0
    assert retry_engine.diagnostics.active_attempts == 0

    rollback_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:validate_plan",
                exception=conversation.ConversationCapabilityError(),
            ),
            conversation.FaultAction(
                label="coordinator:rollback",
                exception=CancelledError(),
            ),
        )
    )
    rollback_store = conversation.InMemoryConversationStore()
    rollback_engine = coordinator(
        store=rollback_store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            rollback_controller
        ),
    )
    rollback_run = replace(
        retry_run,
        identity=root_identity("cancel-rollback"),
        idempotency_key=conversation.RequestIdempotencyKey(
            "key-cancel-rollback"
        ),
        provisional_response_id=conversation.ProvisionalResponseId(
            "provisional-cancel-rollback"
        ),
        public_response_id=conversation.PublicResponseId(
            "response-cancel-rollback"
        ),
    )
    with pytest.raises(CancelledError):
        await rollback_engine.execute(rollback_run)
    assert rollback_store.diagnostics.idempotency_records == 0
    assert rollback_store.diagnostics.provisional_responses == 0
    assert rollback_engine.diagnostics.active_attempts == 0

    original_cancel_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:validate_plan",
                exception=CancelledError(),
            ),
        )
    )
    store_cancel_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:rollback",
                exception=CancelledError(),
            ),
        )
    )
    store_cancel_store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(
            store_cancel_controller
        )
    )
    store_cancel_engine = coordinator(
        store=store_cancel_store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            original_cancel_controller
        ),
    )
    store_cancel_run = replace(
        retry_run,
        identity=root_identity("cancel-store-rollback"),
        idempotency_key=conversation.RequestIdempotencyKey(
            "key-cancel-store-rollback"
        ),
        provisional_response_id=conversation.ProvisionalResponseId(
            "provisional-cancel-store-rollback"
        ),
        public_response_id=conversation.PublicResponseId(
            "response-cancel-store-rollback"
        ),
    )
    with pytest.raises(CancelledError):
        await store_cancel_engine.execute(store_cancel_run)
    assert store_cancel_store.diagnostics.idempotency_records == 0
    assert store_cancel_engine.diagnostics.active_attempts == 0
    await store_cancel_engine._rollback(
        store_cancel_engine._idempotency(store_cancel_run),
        "missing-owner",
        ambiguous=False,
    )

    conflict_cancel_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:rollback",
                exception=CancelledError(),
            ),
        )
    )
    conflict_cancel_engine = coordinator(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            conflict_cancel_controller
        ),
    )
    with pytest.raises(CancelledError):
        await conflict_cancel_engine._rollback(
            conflict_cancel_engine._idempotency(store_cancel_run),
            "missing-owner",
            ambiguous=False,
        )

    close_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:close",
                exception=CancelledError(),
            ),
        )
    )
    close_store = conversation.InMemoryConversationStore()
    close_engine = coordinator(
        store=close_store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            close_controller
        ),
    )
    with pytest.raises(CancelledError):
        await close_engine.close()
    assert close_engine.diagnostics.closed
    assert close_store.diagnostics.closed
    await close_engine.close()


async def test_cancellation_cleanup_failures_reconcile_or_lease_fence() -> (
    None
):
    """Preserve cancellation while cleanup settles or expires safely."""
    baseline_tasks = len(all_tasks())
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)

    provider_cancel = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:dispatch",
                exception=CancelledError(),
            ),
        )
    )
    transient_hook = _BoundaryFailures(
        {
            conversation.StoreAwaitBoundary.ROLLBACK: [
                RuntimeError("rollback-failed")
            ],
            conversation.StoreAwaitBoundary.IDEMPOTENCY_RECONCILE: [
                RuntimeError("first-reconcile-failed")
            ],
        }
    )
    transient_store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW),
        boundary_hook=transient_hook,
    )
    transient_runtime = _runtime(
        lane_binding,
        (result,),
        controller=provider_cancel,
    )
    transient_engine = coordinator(
        store=transient_store,
        scope=scope,
        runtimes=(transient_runtime,),
    )
    transient_run = request(
        scope=scope,
        identity=root_identity("cancel-cleanup-transient"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="cancel-cleanup-transient",
        key="key-cancel-cleanup-transient",
    )
    with pytest.raises(CancelledError):
        await transient_engine.execute(transient_run)
    assert transient_store.diagnostics.idempotency_records == 1
    assert transient_store.diagnostics.provisional_responses == 0
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        await transient_engine.execute(transient_run)
    transient_provider = transient_runtime.provider
    assert isinstance(
        transient_provider,
        conversation.DeterministicFakeProviderDiagnostics,
    )
    assert len(transient_provider.plans) == 1
    assert transient_store.diagnostics.idempotency_waiters == 0
    assert not transient_store.diagnostics.locked

    lease_clock = conversation.DeterministicFakeClock(NOW)
    repeated_provider_cancel = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:dispatch",
                exception=CancelledError(),
            ),
        )
    )
    repeated_hook = _BoundaryFailures(
        {
            conversation.StoreAwaitBoundary.ROLLBACK: [
                RuntimeError("rollback-failed")
            ],
            conversation.StoreAwaitBoundary.IDEMPOTENCY_RECONCILE: [
                RuntimeError("reconcile-one-failed"),
                RuntimeError("reconcile-two-failed"),
            ],
        }
    )
    repeated_store = conversation.InMemoryConversationStore(
        limits=conversation.StoreLimits(idempotency_lease_seconds=1),
        clock=lease_clock,
        boundary_hook=repeated_hook,
    )
    repeated_runtime = _runtime(
        lane_binding,
        (result,),
        controller=repeated_provider_cancel,
    )
    repeated_engine = coordinator(
        store=repeated_store,
        scope=scope,
        runtimes=(repeated_runtime,),
    )
    repeated_run = request(
        scope=scope,
        identity=root_identity("cancel-cleanup-repeated"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="cancel-cleanup-repeated",
        key="key-cancel-cleanup-repeated",
    )
    with pytest.raises(CancelledError) as cancelled:
        await repeated_engine.execute(repeated_run)
    assert isinstance(cancelled.value.__cause__, RuntimeError)
    assert repeated_store.diagnostics.idempotency_records == 1
    assert repeated_store.diagnostics.provisional_responses == 1
    assert "opaque" not in repr(repeated_store.diagnostics)
    lease_clock.set(NOW + timedelta(seconds=2))
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        await repeated_engine.execute(repeated_run)
    repeated_provider = repeated_runtime.provider
    assert isinstance(
        repeated_provider,
        conversation.DeterministicFakeProviderDiagnostics,
    )
    assert len(repeated_provider.plans) == 1
    assert repeated_store.diagnostics.provisional_responses == 0
    assert repeated_store.diagnostics.idempotency_waiters == 0
    assert not repeated_store.diagnostics.locked
    assert len(all_tasks()) == baseline_tasks


async def test_publication_release_failure_expires_without_redispatch() -> (
    None
):
    """Leave a failed release durably leased until safe exact reclaim."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    store_clock = conversation.DeterministicFakeClock(NOW)
    release_failure = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:outbox_release",
                exception=RuntimeError("release-failed"),
            ),
        )
    )
    store = conversation.InMemoryConversationStore(
        clock=store_clock,
        boundary_hook=conversation.FakeStoreBoundaryHook(release_failure),
    )
    runtime = _runtime(lane_binding, (result,))
    publisher = conversation.DeterministicFakePublisher()
    publish_cancel = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:publish",
                exception=CancelledError(),
            ),
        )
    )
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(runtime,),
        publisher=publisher,
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(publish_cancel),
    )
    run = request(
        scope=scope,
        identity=root_identity("cancel-release-failure"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="cancel-release-failure",
        key="key-cancel-release-failure",
    )
    with pytest.raises(CancelledError) as cancelled:
        await engine.execute(run)
    assert isinstance(cancelled.value.__cause__, RuntimeError)
    provider = runtime.provider
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert len(provider.plans) == 1
    assert publisher.published == ()

    replay_engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(runtime,),
        publisher=publisher,
    )
    with pytest.raises(conversation.ConversationPublicationError):
        await replay_engine.execute(run)
    assert len(provider.plans) == 1
    store_clock.set(NOW + timedelta(seconds=31))
    replay = await replay_engine.execute(run)
    assert replay.result is not None
    assert len(provider.plans) == 1
    assert len(publisher.published) == 1
    assert store.diagnostics.idempotency_waiters == 0
    assert not store.diagnostics.locked


async def test_cancelled_close_with_store_failure_is_retryable() -> None:
    """Preserve close cancellation and leave a failed close retryable."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    close_failure = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:close",
                exception=RuntimeError("close-failed"),
            ),
        )
    )
    store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(close_failure)
    )
    coordinator_cancel = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:close",
                exception=CancelledError(),
            ),
        )
    )
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            coordinator_cancel
        ),
    )
    with pytest.raises(CancelledError) as cancelled:
        await engine.close()
    assert isinstance(cancelled.value.__cause__, RuntimeError)
    assert not engine.diagnostics.closed
    assert not store.diagnostics.closed
    assert not store.diagnostics.locked
    await engine.close()
    assert engine.diagnostics.closed
    assert store.diagnostics.closed


async def test_store_close_cancellation_and_failure_preserve_precedence() -> (
    None
):
    """Propagate store cancellation and keep plain failure retryable."""
    scope = authority()
    lane_binding = binding()
    result = conversation.fake_provider_result(
        empty_stateless_plan(lane_binding),
        turn=1,
    )
    store_cancellation = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:close",
                exception=CancelledError(),
            ),
        )
    )
    cancelled_store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(store_cancellation)
    )
    cancelled_engine = coordinator(
        store=cancelled_store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
    )
    with pytest.raises(CancelledError):
        await cancelled_engine.close()
    assert cancelled_engine.diagnostics.closed
    assert cancelled_store.diagnostics.closed

    store_failure = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:close",
                exception=RuntimeError("plain-close-failure"),
            ),
        )
    )
    failed_store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(store_failure)
    )
    failed_engine = coordinator(
        store=failed_store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
    )
    with pytest.raises(RuntimeError, match="plain-close-failure"):
        await failed_engine.close()
    assert not failed_engine.diagnostics.closed
    assert not failed_store.diagnostics.closed
    await failed_engine.close()
    assert failed_engine.diagnostics.closed
    assert failed_store.diagnostics.closed


@pytest.mark.parametrize(
    ("boundary", "closed"),
    (("close_begin", False), ("close_settled", True)),
)
async def test_coordinator_close_probes_interrupted_store_settlement(
    boundary: str,
    closed: bool,
) -> None:
    """Never infer store closure from cancellation without a status probe."""
    scope = authority()
    lane_binding = binding()
    result = conversation.fake_provider_result(
        empty_stateless_plan(lane_binding),
        turn=1,
    )
    store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(
            conversation.DeterministicFaultController(
                (
                    conversation.FaultAction(
                        label=f"store:{boundary}",
                        exception=CancelledError(),
                    ),
                )
            )
        )
    )
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
    )
    with pytest.raises(CancelledError):
        await engine.close()
    assert engine.diagnostics.closed is closed
    assert store.diagnostics.closed is closed
    if not closed:
        await engine.close()
        assert engine.diagnostics.closed


async def test_close_rejects_untrusted_status_and_probe_failure() -> None:
    """Keep close retryable unless a typed closed status is observed."""
    scope = authority()
    lane_binding = binding()
    result = conversation.fake_provider_result(
        empty_stateless_plan(lane_binding),
        turn=1,
    )
    runtime = _runtime(lane_binding, (result,))

    def engine_for(
        store: _ScriptedCloseStore,
        hook: conversation.FakeCoordinatorBoundaryHook | None = None,
    ) -> conversation.RunScopedConversationCoordinator:
        return coordinator(
            store=cast(conversation.ConversationStore, store),
            scope=scope,
            runtimes=(runtime,),
            boundary_hook=hook,
        )

    probe_failure = engine_for(
        _ScriptedCloseStore(
            CancelledError(),
            RuntimeError("close-status-probe-failed"),
        )
    )
    with pytest.raises(CancelledError) as cancelled:
        await probe_failure.close()
    assert isinstance(cancelled.value.__cause__, RuntimeError)
    assert not probe_failure.diagnostics.closed

    invalid = engine_for(
        _ScriptedCloseStore(
            object(),
            conversation.StoreCloseResolution(
                disposition=conversation.StoreCloseDisposition.CLOSED
            ),
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        await invalid.close()
    assert invalid.diagnostics.closed
    await invalid.close()

    cancellation_hook = conversation.FakeCoordinatorBoundaryHook(
        conversation.DeterministicFaultController(
            (
                conversation.FaultAction(
                    label="coordinator:close",
                    exception=CancelledError(),
                ),
            )
        )
    )
    invalid_after_cancel = engine_for(
        _ScriptedCloseStore(object(), object()),
        cancellation_hook,
    )
    with pytest.raises(CancelledError) as cancelled_invalid:
        await invalid_after_cancel.close()
    assert isinstance(
        cancelled_invalid.value.__cause__,
        conversation.ConversationValidationError,
    )
    assert not invalid_after_cancel.diagnostics.closed

    open_store = engine_for(
        _ScriptedCloseStore(
            conversation.StoreCloseResolution(
                disposition=conversation.StoreCloseDisposition.OPEN
            ),
            conversation.StoreCloseResolution(
                disposition=conversation.StoreCloseDisposition.OPEN
            ),
        )
    )
    with pytest.raises(conversation.ConversationConflictError):
        await open_store.close()
    assert not open_store.diagnostics.closed


async def test_close_always_probes_and_requires_exact_corroboration() -> None:
    """Probe every close action and derive closed state only from the probe."""
    scope = authority()
    lane_binding = binding()
    result = conversation.fake_provider_result(
        empty_stateless_plan(lane_binding),
        turn=1,
    )
    runtime = _runtime(lane_binding, (result,))
    closed = conversation.StoreCloseResolution(
        disposition=conversation.StoreCloseDisposition.CLOSED
    )
    open_resolution = conversation.StoreCloseResolution(
        disposition=conversation.StoreCloseDisposition.OPEN
    )

    def engine_for(
        close: object,
        inspect: object,
        hook: conversation.FakeCoordinatorBoundaryHook | None = None,
    ) -> tuple[
        conversation.RunScopedConversationCoordinator,
        _ScriptedCloseStore,
    ]:
        store = _ScriptedCloseStore(close, inspect)
        return (
            coordinator(
                store=cast(conversation.ConversationStore, store),
                scope=scope,
                runtimes=(runtime,),
                boundary_hook=hook,
            ),
            store,
        )

    matched, matched_store = engine_for(closed, closed)
    await matched.close()
    assert matched.diagnostics.closed
    assert (matched_store.close_calls, matched_store.inspect_calls) == (1, 1)

    matched._hook = conversation.FakeCoordinatorBoundaryHook(
        conversation.DeterministicFaultController(
            (
                conversation.FaultAction(
                    label="coordinator:close",
                    exception=CancelledError(),
                ),
            )
        )
    )
    with pytest.raises(CancelledError):
        await matched.close()
    assert (matched_store.close_calls, matched_store.inspect_calls) == (1, 1)

    failed_action, failed_action_store = engine_for(
        RuntimeError("close-action-failed"),
        closed,
    )
    with pytest.raises(RuntimeError, match="close-action-failed"):
        await failed_action.close()
    assert failed_action.diagnostics.closed
    assert (
        failed_action_store.close_calls,
        failed_action_store.inspect_calls,
    ) == (1, 1)
    await failed_action.close()
    assert (
        failed_action_store.close_calls,
        failed_action_store.inspect_calls,
    ) == (1, 1)

    closed_open, closed_open_store = engine_for(closed, open_resolution)
    with pytest.raises(conversation.ConversationConflictError):
        await closed_open.close()
    assert not closed_open.diagnostics.closed
    assert (
        closed_open_store.close_calls,
        closed_open_store.inspect_calls,
    ) == (1, 1)

    open_closed, open_closed_store = engine_for(open_resolution, closed)
    with pytest.raises(conversation.ConversationConflictError):
        await open_closed.close()
    assert open_closed.diagnostics.closed
    assert (
        open_closed_store.close_calls,
        open_closed_store.inspect_calls,
    ) == (1, 1)

    probe_failed, probe_failed_store = engine_for(
        closed,
        RuntimeError("close-probe-failed"),
    )
    with pytest.raises(RuntimeError, match="close-probe-failed"):
        await probe_failed.close()
    assert not probe_failed.diagnostics.closed
    assert (
        probe_failed_store.close_calls,
        probe_failed_store.inspect_calls,
    ) == (1, 1)

    both_failed, both_failed_store = engine_for(
        RuntimeError("close-action-and-probe-action"),
        RuntimeError("close-action-and-probe-probe"),
    )
    with pytest.raises(
        RuntimeError,
        match="close-action-and-probe-action",
    ) as both_failed_error:
        await both_failed.close()
    assert isinstance(both_failed_error.value.__cause__, RuntimeError)
    assert (
        str(both_failed_error.value.__cause__)
        == "close-action-and-probe-probe"
    )
    assert (
        both_failed_store.close_calls,
        both_failed_store.inspect_calls,
    ) == (1, 1)

    probe_cancelled, probe_cancelled_store = engine_for(
        closed,
        CancelledError(),
    )
    with pytest.raises(CancelledError):
        await probe_cancelled.close()
    assert not probe_cancelled.diagnostics.closed
    assert (
        probe_cancelled_store.close_calls,
        probe_cancelled_store.inspect_calls,
    ) == (1, 1)

    action_cancelled, action_cancelled_store = engine_for(
        CancelledError(),
        closed,
    )
    with pytest.raises(CancelledError):
        await action_cancelled.close()
    assert action_cancelled.diagnostics.closed
    assert (
        action_cancelled_store.close_calls,
        action_cancelled_store.inspect_calls,
    ) == (1, 1)

    hook = conversation.FakeCoordinatorBoundaryHook(
        conversation.DeterministicFaultController(
            (
                conversation.FaultAction(
                    label="coordinator:close",
                    exception=CancelledError(),
                ),
            )
        )
    )
    hook_cancelled, hook_cancelled_store = engine_for(
        closed,
        closed,
        hook,
    )
    with pytest.raises(CancelledError):
        await hook_cancelled.close()
    assert hook_cancelled.diagnostics.closed
    assert (
        hook_cancelled_store.close_calls,
        hook_cancelled_store.inspect_calls,
    ) == (1, 1)

    active_hook = conversation.FakeCoordinatorBoundaryHook(
        conversation.DeterministicFaultController(
            (
                conversation.FaultAction(
                    label="coordinator:close",
                    exception=CancelledError(),
                ),
            )
        )
    )
    active, active_store = engine_for(closed, closed, active_hook)
    active._active_attempts.add("active-close-attempt")
    with pytest.raises(CancelledError) as active_cancelled:
        await active.close()
    assert isinstance(
        active_cancelled.value.__cause__,
        conversation.ConversationConflictError,
    )
    assert (active_store.close_calls, active_store.inspect_calls) == (0, 0)


@pytest.mark.parametrize("boundary", ("rollback_begin", "rollback_settled"))
async def test_coordinator_rollback_probes_interrupted_settlement(
    boundary: str,
) -> None:
    """Reconcile an interrupted cleanup without assuming it mutated state."""
    scope = authority()
    lane_binding = binding()
    result = conversation.fake_provider_result(
        empty_stateless_plan(lane_binding),
        turn=1,
    )
    store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(
            conversation.DeterministicFaultController(
                (
                    conversation.FaultAction(
                        label=f"store:{boundary}",
                        exception=CancelledError(),
                    ),
                )
            )
        )
    )
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
    )
    run = request(
        scope=scope,
        identity=root_identity(f"coordinator-{boundary}"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix=f"coordinator-{boundary}",
        key=f"key-coordinator-{boundary}",
    )
    identity = engine._idempotency(run)
    reservation = await store.reserve_idempotency(identity)
    assert reservation.owner_token is not None
    with pytest.raises(CancelledError):
        await engine._rollback(
            identity,
            reservation.owner_token,
            ambiguous=False,
        )
    resolution = await store.inspect_idempotency_settlement(
        identity,
        reservation.owner_token,
    )
    assert resolution.disposition is (
        conversation.IdempotencySettlementDisposition.SETTLED
    )
    assert store.diagnostics.idempotency_records == 0


async def test_rollback_hook_failure_and_reconcile_cancellation_settle() -> (
    None
):
    """Settle ownership after hook failure and preserve cancellation."""
    scope = authority()
    lane_binding = binding()
    result = conversation.fake_provider_result(
        empty_stateless_plan(lane_binding),
        turn=1,
    )
    run = request(
        scope=scope,
        identity=root_identity("rollback-hook-failure"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="rollback-hook-failure",
        key="key-rollback-hook-failure",
    )
    hook_failure = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:rollback",
                exception=RuntimeError("rollback-hook-failure"),
            ),
        )
    )
    settled_store = conversation.InMemoryConversationStore()
    settled_engine = coordinator(
        store=settled_store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(hook_failure),
    )
    identity = settled_engine._idempotency(run)
    reservation = await settled_store.reserve_idempotency(identity)
    assert reservation.owner_token is not None
    await settled_engine._rollback(
        identity,
        reservation.owner_token,
        ambiguous=False,
    )
    assert settled_store.diagnostics.idempotency_records == 0

    reconcile_faults = _BoundaryFailures(
        {
            conversation.StoreAwaitBoundary.ROLLBACK: [
                RuntimeError("rollback-store-failure")
            ],
            conversation.StoreAwaitBoundary.IDEMPOTENCY_RECONCILE: [
                CancelledError()
            ],
        }
    )
    cancelling_store = conversation.InMemoryConversationStore(
        boundary_hook=reconcile_faults
    )
    cancelling_engine = coordinator(
        store=cancelling_store,
        scope=scope,
        runtimes=(_runtime(lane_binding, (result,)),),
    )
    cancelling_identity = cancelling_engine._idempotency(
        replace(
            run,
            identity=root_identity("rollback-reconcile-cancel"),
            idempotency_key=conversation.RequestIdempotencyKey(
                "key-rollback-reconcile-cancel"
            ),
        )
    )
    cancelling_reservation = await cancelling_store.reserve_idempotency(
        cancelling_identity
    )
    assert cancelling_reservation.owner_token is not None
    with pytest.raises(CancelledError):
        await cancelling_engine._rollback(
            cancelling_identity,
            cancelling_reservation.owner_token,
            ambiguous=False,
        )
    assert cancelling_store.diagnostics.idempotency_records == 0


async def test_rollback_defensive_probe_and_reconcile_matrix_is_closed() -> (
    None
):
    """Reject malformed settlement results and preserve cleanup precedence."""
    scope = authority()
    lane_binding = binding()
    result = conversation.fake_provider_result(
        empty_stateless_plan(lane_binding),
        turn=1,
    )
    runtime = _runtime(lane_binding, (result,))
    run = request(
        scope=scope,
        identity=root_identity("rollback-defensive-matrix"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="rollback-defensive-matrix",
        key="key-rollback-defensive-matrix",
    )
    settled = conversation.IdempotencySettlementResolution(
        disposition=conversation.IdempotencySettlementDisposition.SETTLED
    )
    leased = conversation.IdempotencySettlementResolution(
        disposition=conversation.IdempotencySettlementDisposition.LEASED,
        lease_expires_at=NOW + timedelta(seconds=30),
        lease_owner_token="rollback-defensive-owner",
    )
    conflict = conversation.IdempotencySettlementResolution(
        disposition=(
            conversation.IdempotencySettlementDisposition.OWNERSHIP_CONFLICT
        )
    )

    async def outcome(
        store: _ScriptedSettlementStore,
        hook: conversation.FakeCoordinatorBoundaryHook | None = None,
    ) -> BaseException | None:
        engine = coordinator(
            store=cast(conversation.ConversationStore, store),
            scope=scope,
            runtimes=(runtime,),
            boundary_hook=hook,
        )
        try:
            await engine._rollback(
                engine._idempotency(run),
                "rollback-defensive-owner",
                ambiguous=False,
            )
        except BaseException as exc:
            return exc
        return None

    matched_store = _ScriptedSettlementStore(
        abandon=(settled,),
        inspect=(settled,),
        reconcile=(),
    )
    assert await outcome(matched_store) is None
    assert (
        matched_store.abandon_calls,
        matched_store.inspect_calls,
        matched_store.reconcile_calls,
    ) == (1, 1, 0)

    action_failed_store = _ScriptedSettlementStore(
        abandon=(RuntimeError("abandon-interrupted"),),
        inspect=(settled,),
        reconcile=(),
    )
    assert await outcome(action_failed_store) is None
    assert (
        action_failed_store.abandon_calls,
        action_failed_store.inspect_calls,
        action_failed_store.reconcile_calls,
    ) == (1, 1, 0)

    malformed_store = _ScriptedSettlementStore(
        abandon=(object(),),
        inspect=(settled, settled),
        reconcile=(settled,),
    )
    assert await outcome(malformed_store) is None
    assert (
        malformed_store.abandon_calls,
        malformed_store.inspect_calls,
        malformed_store.reconcile_calls,
    ) == (1, 2, 1)

    mismatched_store = _ScriptedSettlementStore(
        abandon=(settled,),
        inspect=(leased, settled),
        reconcile=(settled,),
    )
    assert await outcome(mismatched_store) is None
    assert (
        mismatched_store.abandon_calls,
        mismatched_store.inspect_calls,
        mismatched_store.reconcile_calls,
    ) == (1, 2, 1)
    inspect_cancel = await outcome(
        _ScriptedSettlementStore(
            abandon=(CancelledError(),),
            inspect=(CancelledError(), settled),
            reconcile=(settled,),
        )
    )
    assert isinstance(inspect_cancel, CancelledError)
    assert (
        await outcome(
            _ScriptedSettlementStore(
                abandon=(RuntimeError("abandon-failed"),),
                inspect=(RuntimeError("inspect-failed"), settled),
                reconcile=(settled,),
            )
        )
        is None
    )

    initial_conflict = await outcome(
        _ScriptedSettlementStore(
            abandon=(conflict,),
            inspect=(conflict,),
            reconcile=(),
        )
    )
    assert isinstance(initial_conflict, conversation.ConversationConflictError)
    cancelled_conflict = await outcome(
        _ScriptedSettlementStore(
            abandon=(CancelledError(),),
            inspect=(conflict,),
            reconcile=(),
        )
    )
    assert isinstance(cancelled_conflict, CancelledError)
    assert isinstance(
        cancelled_conflict.__cause__,
        conversation.ConversationConflictError,
    )

    reconcile_cancel_probe_cancel = await outcome(
        _ScriptedSettlementStore(
            abandon=(RuntimeError("abandon-failed"),),
            inspect=(leased, CancelledError(), settled),
            reconcile=(CancelledError(), settled),
        )
    )
    assert isinstance(reconcile_cancel_probe_cancel, CancelledError)
    reconcile_cancel_probe_error = await outcome(
        _ScriptedSettlementStore(
            abandon=(RuntimeError("abandon-failed"),),
            inspect=(leased, RuntimeError("probe-failed"), settled),
            reconcile=(CancelledError(), settled),
        )
    )
    assert isinstance(reconcile_cancel_probe_error, CancelledError)
    reconcile_error_probe_cancel = await outcome(
        _ScriptedSettlementStore(
            abandon=(RuntimeError("abandon-failed"),),
            inspect=(leased, CancelledError(), settled),
            reconcile=(RuntimeError("reconcile-failed"), settled),
        )
    )
    assert isinstance(reconcile_error_probe_cancel, CancelledError)
    assert (
        await outcome(
            _ScriptedSettlementStore(
                abandon=(RuntimeError("abandon-failed"),),
                inspect=(leased, RuntimeError("probe-failed"), settled),
                reconcile=(RuntimeError("reconcile-failed"), settled),
            )
        )
        is None
    )
    assert (
        await outcome(
            _ScriptedSettlementStore(
                abandon=(RuntimeError("abandon-failed"),),
                inspect=(leased, settled, settled),
                reconcile=(object(), settled),
            )
        )
        is None
    )
    reconcile_conflict = await outcome(
        _ScriptedSettlementStore(
            abandon=(RuntimeError("abandon-failed"),),
            inspect=(leased, conflict),
            reconcile=(conflict,),
        )
    )
    assert isinstance(
        reconcile_conflict,
        conversation.ConversationConflictError,
    )

    cancellation_hook = conversation.FakeCoordinatorBoundaryHook(
        conversation.DeterministicFaultController(
            (
                conversation.FaultAction(
                    label="coordinator:rollback",
                    exception=CancelledError(),
                ),
            )
        )
    )
    leased_store = _ScriptedSettlementStore(
        abandon=(RuntimeError("abandon-failed"),),
        inspect=(leased, leased, leased),
        reconcile=(leased, leased),
    )
    leased_after_failure = await outcome(
        leased_store,
        cancellation_hook,
    )
    assert isinstance(leased_after_failure, CancelledError)
    assert isinstance(leased_after_failure.__cause__, RuntimeError)
    assert (
        leased_store.abandon_calls,
        leased_store.inspect_calls,
        leased_store.reconcile_calls,
    ) == (1, 3, 2)

    invalid_probe_store = _ScriptedSettlementStore(
        abandon=(settled,),
        inspect=(object(), object(), object()),
        reconcile=(settled, settled),
    )
    invalid_probe = await outcome(invalid_probe_store)
    assert isinstance(invalid_probe, conversation.ConversationValidationError)
    assert (
        invalid_probe_store.abandon_calls,
        invalid_probe_store.inspect_calls,
        invalid_probe_store.reconcile_calls,
    ) == (1, 3, 2)

    invalid_lease = copy(leased)
    object.__setattr__(
        invalid_lease,
        "lease_owner_token",
        "different-rollback-owner",
    )
    invalid_lease_store = _ScriptedSettlementStore(
        abandon=(invalid_lease,),
        inspect=(invalid_lease, settled),
        reconcile=(settled,),
    )
    assert await outcome(invalid_lease_store) is None
    assert (
        invalid_lease_store.abandon_calls,
        invalid_lease_store.inspect_calls,
        invalid_lease_store.reconcile_calls,
    ) == (1, 2, 1)


async def test_publication_cancellation_releases_lease_for_replay() -> None:
    """Replay a committed response after cancellation while publishing."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    runtime = _runtime(lane_binding, (result,))
    store = conversation.InMemoryConversationStore()
    publisher = conversation.DeterministicFakePublisher()
    hook_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:publish",
                exception=CancelledError(),
            ),
        )
    )
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(runtime,),
        publisher=publisher,
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            hook_controller
        ),
    )
    run = request(
        scope=scope,
        identity=root_identity("cancel-publication"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="cancel-publication",
        key="key-cancel-publication",
    )
    with pytest.raises(CancelledError):
        await engine.execute(run)
    assert store.diagnostics.provisional_responses == 0
    assert store.diagnostics.outbox_records == 1
    assert len(publisher.published) == 0

    replay_engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(runtime,),
        publisher=publisher,
    )
    replay = await replay_engine.execute(run)
    assert replay.result is not None
    assert replay.output_candidates[0].completed_items == result.items
    provider = runtime.provider
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert len(provider.plans) == 1
    assert len(publisher.published) == 1
    assert run.public_response_id is not None
    target = conversation.OutboxClaimTarget(
        authority=scope,
        checkpoint_id=replay.checkpoint.identity.checkpoint_id,
        public_response_id=run.public_response_id,
        intent_id=f"publication-{run.public_response_id}",
    )
    assert (await store.claim_outbox(target)).disposition is (
        conversation.OutboxClaimDisposition.ALREADY_PUBLISHED
    )


async def test_outbox_ack_and_release_faults_leave_replayable_state() -> None:
    """Settle or release every claimed intent after outbox faults."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)

    ack_failure = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:outbox_acknowledge",
                exception=RuntimeError("ack-failure"),
            ),
        )
    )
    ack_store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(ack_failure)
    )
    ack_result = conversation.fake_provider_result(plan, turn=1)
    ack_runtime = _runtime(lane_binding, (ack_result,))
    ack_publisher = conversation.DeterministicFakePublisher()
    ack_engine = coordinator(
        store=ack_store,
        scope=scope,
        runtimes=(ack_runtime,),
        publisher=ack_publisher,
    )
    ack_run = request(
        scope=scope,
        identity=root_identity("ack-failure"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="ack-failure",
        key="key-ack-failure",
    )
    with pytest.raises(conversation.ConversationPublicationError):
        await ack_engine.execute(ack_run)
    ack_replay = await ack_engine.execute(ack_run)
    assert ack_replay.result is not None
    assert len(ack_publisher.published) == 1

    ack_cancel = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:outbox_acknowledge",
                exception=CancelledError(),
            ),
        )
    )
    cancel_store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(ack_cancel)
    )
    cancel_result = conversation.fake_provider_result(plan, turn=2)
    cancel_runtime = _runtime(lane_binding, (cancel_result,))
    cancel_publisher = conversation.DeterministicFakePublisher()
    cancel_engine = coordinator(
        store=cancel_store,
        scope=scope,
        runtimes=(cancel_runtime,),
        publisher=cancel_publisher,
    )
    cancel_run = request(
        scope=scope,
        identity=root_identity("ack-cancel"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="ack-cancel",
        key="key-ack-cancel",
    )
    with pytest.raises(CancelledError):
        await cancel_engine.execute(cancel_run)
    cancel_replay = await cancel_engine.execute(cancel_run)
    assert cancel_replay.result is not None
    assert len(cancel_publisher.published) == 1
    assert cancel_run.public_response_id is not None
    cancel_target = conversation.OutboxClaimTarget(
        authority=scope,
        checkpoint_id=cancel_replay.checkpoint.identity.checkpoint_id,
        public_response_id=cancel_run.public_response_id,
        intent_id=f"publication-{cancel_run.public_response_id}",
    )
    assert (await cancel_store.claim_outbox(cancel_target)).disposition is (
        conversation.OutboxClaimDisposition.ALREADY_PUBLISHED
    )

    release_cancel = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:outbox_release",
                exception=CancelledError(),
            ),
        )
    )
    release_store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(release_cancel)
    )
    publish_cancel = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="publisher:publish",
                exception=CancelledError(),
            ),
        )
    )
    release_result = conversation.fake_provider_result(plan, turn=3)
    release_runtime = _runtime(lane_binding, (release_result,))
    release_publisher = conversation.DeterministicFakePublisher(publish_cancel)
    release_engine = coordinator(
        store=release_store,
        scope=scope,
        runtimes=(release_runtime,),
        publisher=release_publisher,
    )
    release_run = request(
        scope=scope,
        identity=root_identity("release-cancel"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="release-cancel",
        key="key-release-cancel",
    )
    with pytest.raises(CancelledError):
        await release_engine.execute(release_run)
    assert (await release_engine.execute(release_run)).result is not None
    assert len(release_publisher.published) == 1


@pytest.mark.parametrize(
    ("acknowledge_fault", "publish_fault", "expected_error"),
    (
        (
            None,
            RuntimeError("publish-failed"),
            conversation.ConversationPublicationError,
        ),
        (
            RuntimeError("acknowledge-failed"),
            None,
            conversation.ConversationPublicationError,
        ),
        (CancelledError(), None, CancelledError),
    ),
)
async def test_publication_and_release_combined_failures_preserve_primary(
    acknowledge_fault: BaseException | None,
    publish_fault: BaseException | None,
    expected_error: type[BaseException],
) -> None:
    """Preserve publication or cancellation while chaining release failure."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    failures: dict[
        conversation.StoreAwaitBoundary,
        list[BaseException],
    ] = {
        conversation.StoreAwaitBoundary.OUTBOX_RELEASE: [
            RuntimeError("release-after-primary")
        ]
    }
    if acknowledge_fault is not None:
        failures[conversation.StoreAwaitBoundary.OUTBOX_ACKNOWLEDGE] = [
            acknowledge_fault
        ]
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(NOW),
        boundary_hook=_BoundaryFailures(failures),
    )
    publisher_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="publisher:publish",
                exception=publish_fault,
            ),
        )
        if publish_fault is not None
        else ()
    )
    publisher = conversation.DeterministicFakePublisher(publisher_controller)
    runtime = _runtime(lane_binding, (result,))
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(runtime,),
        publisher=publisher,
    )
    run = request(
        scope=scope,
        identity=root_identity(
            f"combined-{expected_error.__name__}-{type(acknowledge_fault).__name__}"
        ),
        advance=conversation.FirstTurnAdvance(),
        response_suffix=f"combined-{expected_error.__name__}-{type(publish_fault).__name__}",
        key=(
            f"key-combined-{type(acknowledge_fault).__name__}-"
            f"{type(publish_fault).__name__}"
        ),
    )

    with pytest.raises(expected_error) as raised:
        await engine.execute(run)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "release-after-primary"
    provider = runtime.provider
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert len(provider.plans) == 1


async def test_internal_segment_and_explicit_reset_are_closed_plans(
    record_property: Callable[[str, object], None],
) -> None:
    """Commit an internal boundary and reset without replaying parent state."""
    record_property("conversation_acceptance_evidence", "runtime")
    (
        scope,
        store,
        _engine,
        _runtime_value,
        root,
        first_result,
    ) = await _seed_root()
    lane_binding = binding()
    next_plan = next_stateless_plan(lane_binding, first_result.items)
    reset_plan = empty_stateless_plan(lane_binding)
    runtime = _runtime(
        lane_binding,
        (
            conversation.fake_provider_result(next_plan, turn=2),
            conversation.fake_provider_result(reset_plan, turn=3),
        ),
    )
    engine = coordinator(store=store, scope=scope, runtimes=(runtime,))
    internal = request(
        scope=scope,
        identity=child_identity(root.checkpoint, "internal"),
        advance=conversation.OrdinaryChildAdvance(
            parent_checkpoint_id=root.checkpoint.identity.checkpoint_id
        ),
        response_suffix="internal",
        key="key-internal",
        boundary=conversation.ConversationCommitBoundary.INTERNAL_SEGMENT,
    )
    internal_receipt = await engine.execute(internal)
    assert internal_receipt.result is None
    assert internal_receipt.checkpoint.kind is (
        conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
    )
    reset = request(
        scope=scope,
        identity=root_identity("reset-new"),
        advance=conversation.ResetAdvance(
            parent_checkpoint_id=root.checkpoint.identity.checkpoint_id
        ),
        response_suffix="reset",
        key="key-reset",
    )
    reset_receipt = await engine.execute(reset)
    assert reset_receipt.checkpoint.identity.parent_checkpoint_id is None
    provider = runtime.provider
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    reset_dispatch = provider.plans[-1]
    assert isinstance(reset_dispatch, conversation.StatelessProviderPlan)
    assert reset_dispatch.ledger.items == ()


def test_execution_closure_has_no_synchronous_io_calls(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject synchronous filesystem, HTTP, DB, subprocess, and sleep calls."""
    record_property("conversation_acceptance_evidence", "audit")
    root = Path(__file__).resolve().parents[2]
    observed: set[str] = set()
    for relative in (
        "src/avalan/conversation/coordinator.py",
        "src/avalan/conversation/store.py",
        "src/avalan/conversation/fakes.py",
    ):
        calls = _synchronous_effect_calls(
            (root / relative).read_text(encoding="utf-8")
        )
        observed.update(f"{relative}:{name}" for name in calls)
    assert observed == set()
    mutations = {
        "direct-open": "open('unsafe')",
        "path-read": "path.read_text()",
        "path-write": "path.write_bytes(b'unsafe')",
        "sleep": "time.sleep(1)",
        "subprocess": "subprocess.run(('unsafe',))",
        "database": "connection.execute('unsafe')",
        "http": "connection.request('GET', '/')",
    }
    for label, source in mutations.items():
        assert _synchronous_effect_calls(source), label
    assert not conversation.InMemoryConversationStore().diagnostics.locked


def test_failure_reducer_exhaustively_fences_visible_effects() -> None:
    """Reduce every frozen boundary under all unsafe effect facts."""
    for boundary in conversation.FailureBoundary:
        baseline = conversation.reduce_failure(
            boundary,
            visible_output=False,
            tool_effect=False,
            committed=False,
            ambiguous=False,
        )
        assert baseline.preserve_parent
        for facts in (
            {"visible_output": True},
            {"tool_effect": True},
            {"committed": True},
            {"ambiguous": True},
        ):
            values = {
                "visible_output": False,
                "tool_effect": False,
                "committed": False,
                "ambiguous": False,
            }
            values.update(facts)
            reduced = conversation.reduce_failure(boundary, **values)
            assert reduced.fence_dispatch
            assert reduced.reconciliation_required
            assert reduced.retry_rule is not (
                conversation.RetryRule.BOUNDED_EFFECT_FREE
            )
