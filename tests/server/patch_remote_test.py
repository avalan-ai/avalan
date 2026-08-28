"""Exercise the inert-by-default authenticated remote patch test protocol."""

from asyncio import (
    CancelledError,
    Event,
    Future,
    all_tasks,
    create_task,
    current_task,
    get_running_loop,
    run,
    sleep,
    wait_for,
)
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from json import dumps
from os import environ
from secrets import token_urlsafe
from socket import AF_INET, SOCK_STREAM, socket
from subprocess import DEVNULL, Popen, TimeoutExpired
from sys import executable
from time import monotonic
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from fastapi import FastAPI, Request

from avalan.patch.domain import (
    AlgorithmDigest,
    ApprovalMode,
    ArtifactState,
    ByteSize,
    Capability,
    CommitTruth,
    ContextKind,
    DurationTicks,
    LifecyclePhase,
    LineageState,
    MutationState,
    OperationType,
    PatchApprovalId,
    PatchContextId,
    PatchDomainId,
    PatchEventId,
    PatchExecutionId,
    PatchInvocationOutcome,
    PatchLifecycleEvent,
    PatchLimits,
    PatchObserverCorrelationId,
    PatchPending,
    PatchPendingOperationId,
    PatchPlanId,
    PatchProtocolId,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PatchTargetId,
    PatchWorkspaceId,
    PostconditionState,
    RequestedEffectOccurrence,
    SequenceNumber,
    WorkspaceChange,
)
from avalan.patch.durable_store import (
    DurableOutboxRecord,
    DurablePatchStore,
    DurablePendingRecord,
    DurableRequestIdentity,
    DurableRequestSnapshot,
    DurableReservation,
    DurableStoreError,
    DurableStoreErrorCode,
    DurableTerminalRecord,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.parser import PatchInputLimits
from avalan.patch.policy import (
    ApprovalRequirements,
    CapabilityMode,
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PolicyBrokerId,
    PolicyDisclosure,
    PolicyPathSelector,
    PolicyReviewerRole,
    PolicyRevision,
    PolicyRouteId,
    PolicyRule,
    PreauthorizationClass,
    TrustedPatchPolicy,
)
from avalan.patch.target import (
    LocalPlatformProfile,
    ResolvedMutationScope,
    TargetHandshake,
    TargetIdentity,
    TargetPrimitive,
)
from avalan.patch.toolset import (
    PatchApprovalBinding,
    PatchCoordinatorBinding,
    PatchInvocationCapability,
    PatchInvocationHandle,
    PatchPersistenceBinding,
    PatchRuntimeBinding,
    PatchSdkHost,
    PatchToolError,
    PatchToolSet,
    RemotePatchRuntimeWitness,
)
from avalan.server import patch as patch_server
from avalan.server.patch import (
    RemotePatchAuthority,
    RemotePatchAuthorityResolver,
    RemotePatchController,
    RemotePatchEditPart,
    RemotePatchEventStream,
    RemotePatchServerError,
    RemotePatchTestClient,
    RemotePatchTestServerConfiguration,
    RemotePatchTestServerProfile,
    install_remote_patch_test_routes,
)

_LOOPBACK_READY_PATH = "/__avalan_test__/patch/v1/ready"
_LOOPBACK_READY_TIMEOUT_SECONDS = 8.0
_LOOPBACK_READY_POLL_SECONDS = 0.05


async def _await_loopback_ready(
    request: Callable[[float], Awaitable[httpx.Response]],
    expected: Mapping[str, bool | int],
    unavailable: str,
    *,
    clock: Callable[[], float] = monotonic,
    sleeper: Callable[[float], Awaitable[None]] = sleep,
) -> None:
    """Wait for one loopback-ready response without exceeding eight seconds."""
    deadline = clock() + _LOOPBACK_READY_TIMEOUT_SECONDS
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise AssertionError(unavailable)
        try:
            response = await wait_for(request(remaining), timeout=remaining)
        except (TimeoutError, httpx.ConnectError, httpx.TimeoutException):
            pass
        else:
            if response.status_code == 200 and response.json() == expected:
                return
        delay = min(_LOOPBACK_READY_POLL_SECONDS, deadline - clock())
        if delay <= 0:
            raise AssertionError(unavailable)
        await sleeper(delay)


def test_loopback_readiness_timeout_is_hard_bounded() -> None:
    """Bound failed readiness polling by the monotonic deadline."""

    async def scenario() -> None:
        elapsed = 0.0
        timeouts: list[float] = []
        delays: list[float] = []

        def clock() -> float:
            return elapsed

        async def request(timeout: float) -> httpx.Response:
            timeouts.append(timeout)
            raise TimeoutError

        async def sleeper(delay: float) -> None:
            nonlocal elapsed
            delays.append(delay)
            elapsed = min(
                _LOOPBACK_READY_TIMEOUT_SECONDS, round(elapsed + delay, 2)
            )

        with pytest.raises(AssertionError, match="loopback unavailable"):
            await _await_loopback_ready(
                request,
                {"ready": True, "invocations": 0},
                "loopback unavailable",
                clock=clock,
                sleeper=sleeper,
            )
        assert len(timeouts) == 160
        assert timeouts[0] == _LOOPBACK_READY_TIMEOUT_SECONDS
        assert timeouts[-1] == pytest.approx(_LOOPBACK_READY_POLL_SECONDS)
        assert all(
            0 < timeout <= _LOOPBACK_READY_TIMEOUT_SECONDS
            for timeout in timeouts
        )
        assert all(
            0 < delay <= _LOOPBACK_READY_POLL_SECONDS for delay in delays
        )
        assert sum(delays) == pytest.approx(_LOOPBACK_READY_TIMEOUT_SECONDS)

    run(scenario())


@dataclass
class _Binder:
    """Record whether unauthenticated requests reach runtime binding."""

    calls: int = 0

    async def bind(self) -> PatchRuntimeBinding:
        """Fail if a test attempts to bind a target runtime unexpectedly."""
        self.calls += 1
        raise AssertionError("remote target binding must not run")


class _Settlement:
    """Expose only unused host reconciliation futures for test dispatch."""

    def inspect(
        self, _: PatchInvocationHandle
    ) -> Future[PatchInvocationOutcome]:
        """Return one terminal result if host reconciliation is exercised."""
        future: Future[PatchInvocationOutcome] = (
            get_running_loop().create_future()
        )
        future.set_result(_result())
        return future

    def await_terminal(
        self,
        _: PatchInvocationHandle,
        __: PatchPending,
    ) -> Future[PatchResult]:
        """Return one terminal result if host awaiting is exercised."""
        future: Future[PatchResult] = get_running_loop().create_future()
        future.set_result(_result())
        return future


@dataclass
class _RemoteService:
    """Record server-bound invocations without a target side effect."""

    store: InMemoryDurablePatchStore

    def __post_init__(self) -> None:
        """Initialize only the opaque remote-call observation list."""
        self.remote_calls: list[
            tuple[
                OperationType,
                bytes,
                PatchRequestId,
                PatchObserverCorrelationId,
                DurableRequestIdentity,
            ]
        ] = []
        self.settlement = _Settlement()

    async def invoke(
        self,
        operation: OperationType,
        raw_arguments: bytes,
        capability: PatchInvocationCapability,
        request_id: PatchRequestId,
        correlation_id: PatchObserverCorrelationId,
    ) -> PatchResult:
        """Return a terminal result for normal SDK protocol conformance."""
        del operation, raw_arguments, capability, correlation_id
        return _result(request_id)

    async def invoke_remote(
        self,
        operation: OperationType,
        raw_arguments: bytes,
        capability: PatchInvocationCapability,
        request_id: PatchRequestId,
        correlation_id: PatchObserverCorrelationId,
        identity: DurableRequestIdentity,
    ) -> PatchResult:
        """Record the server-owned identity without replacing durable truth."""
        assert isinstance(capability, PatchInvocationCapability)
        self.remote_calls.append(
            (
                operation,
                raw_arguments,
                request_id,
                correlation_id,
                identity,
            )
        )
        return _result(request_id)

    async def review(self, _: PatchInvocationHandle) -> Mapping[str, object]:
        """Return one content-free protocol review projection."""
        return {"object": "review"}

    async def approve(self, _: PatchInvocationHandle) -> PatchResult:
        """Return one content-free terminal protocol result."""
        return _result()

    def subscribe(
        self, _: PatchInvocationHandle
    ) -> AsyncIterator[PatchLifecycleEvent]:
        """Return an empty lifecycle stream for protocol conformance."""
        return _empty_events()


async def _empty_events() -> AsyncIterator[PatchLifecycleEvent]:
    """Yield no synthetic lifecycle records."""
    if False:
        yield _unused_event()


def _unused_event() -> PatchLifecycleEvent:
    """Prevent widening an empty async generator's static element type."""
    raise AssertionError("empty remote test event stream must not yield")


@dataclass
class _RuntimeBinder:
    """Return one exact durable-store-bound local test runtime."""

    authority: RemotePatchAuthority
    service: _RemoteService
    calls: int = 0

    async def bind(self) -> PatchRuntimeBinding:
        """Return a complete local binding with the controller's store."""
        self.calls += 1
        identity = TargetIdentity(
            self.authority.context,
            self.authority.workspace,
            PatchDomainId(self.authority.execution_scope),
            PatchTargetId("target_" + "a" * 16),
            PatchProtocolId("protocol_" + "a" * 16),
            "filesystem-test",
            "mount-test",
            self.authority.policy_revision.value,
            "lease-test",
            PatchApprovalId("approval_" + "a" * 16),
        )
        primitives = frozenset(TargetPrimitive)
        scope = ResolvedMutationScope(
            ContextKind.LOCAL,
            identity,
            None,
            _limits(),
            frozenset(Capability),
            primitives,
        )
        return PatchRuntimeBinding(
            scope,
            TargetHandshake(
                identity,
                primitives,
                (),
                platform=LocalPlatformProfile.DARWIN,
            ),
            _policy(self.authority.policy_revision, self.authority.route),
            PatchApprovalBinding(True),
            PatchCoordinatorBinding(True, self.service.store),
            PatchPersistenceBinding(True, self.service.store),
            self.service,
            remote_witness=_runtime_witness(self.authority),
        )


@dataclass(frozen=True)
class _Resolver(RemotePatchAuthorityResolver):
    """Resolve one exact authenticated authority for a test app."""

    authority: RemotePatchAuthority

    async def __call__(self, _: Request) -> RemotePatchAuthority | None:
        """Return the fixed exact test principal scope."""
        return self.authority


class _NoPrincipalResolver(RemotePatchAuthorityResolver):
    """Resolve no authenticated principal for fail-closed route tests."""

    async def __call__(self, _: Request) -> RemotePatchAuthority | None:
        """Suppress the local test route for an empty principal."""
        return None


@dataclass
class _ContinuationSnapshot:
    """Expose only the durable facts consumed by the remote controller."""

    pending: DurablePendingRecord | None
    terminal: DurableTerminalRecord | None


@dataclass
class _ContinuationStore:
    """Script durable pending, terminal, cancellation, and outbox semantics."""

    snapshot: _ContinuationSnapshot
    records: tuple[DurableOutboxRecord, ...]

    def __post_init__(self) -> None:
        """Initialize exact access observations without target-side work."""
        self.cancellations = 0
        self.await_calls = 0
        self.await_gate = Event()
        self.accesses: list[object] = []
        self.reservations: dict[DurableRequestIdentity, DurableReservation] = (
            {}
        )

    async def reserve(
        self,
        identity: DurableRequestIdentity,
        canonical_digest: AlgorithmDigest,
        request_id: PatchRequestId | None = None,
    ) -> DurableReservation:
        """Record one reservation without changing the scripted snapshot."""
        self.accesses.append((identity, canonical_digest, request_id))
        existing = self.reservations.get(identity)
        if existing is not None:
            return DurableReservation(
                existing.request_id,
                identity,
                canonical_digest,
                True,
            )
        assert request_id is not None
        reservation = DurableReservation(
            request_id,
            identity,
            canonical_digest,
            False,
        )
        self.reservations[identity] = reservation
        return reservation

    async def inspect(self, access: object) -> _ContinuationSnapshot:
        """Return the single sealed request continuation snapshot."""
        self.accesses.append(access)
        return self.snapshot

    async def inspect_pending(self, access: object) -> object:
        """Return the current pending or terminal scripted branch."""
        self.accesses.append(access)
        return self.snapshot.pending or self.snapshot.terminal

    async def await_terminal(self, access: object) -> DurableTerminalRecord:
        """Wait without turning disconnect into cancellation intent."""
        self.await_calls += 1
        self.accesses.append(access)
        await self.await_gate.wait()
        terminal = self.snapshot.terminal
        assert terminal is not None
        return terminal

    async def outbox(
        self,
        access: object,
        after: SequenceNumber,
        limit: int,
    ) -> tuple[DurableOutboxRecord, ...]:
        """Return monotonic at-least-once records after the supplied cursor."""
        self.accesses.append(access)
        return tuple(
            item for item in self.records if item.sequence.value > after.value
        )[:limit]

    async def request_cancellation(
        self, access: object
    ) -> _ContinuationSnapshot:
        """Persist only cancellation intent on the existing pending branch."""
        self.accesses.append(access)
        pending = self.snapshot.pending
        assert pending is not None
        self.cancellations += 1
        self.snapshot = _ContinuationSnapshot(
            replace(pending, cancellation_requested=True),
            None,
        )
        return self.snapshot


def _result(request_id: PatchRequestId | None = None) -> PatchResult:
    """Return one closed committed result for protocol-only service calls."""
    return PatchResult(
        1,
        request_id or PatchRequestId("request_" + "a" * 16),
        PatchPlanId("plan_" + "a" * 16),
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


def _limits() -> PatchLimits:
    """Return bounded capabilities for the complete runtime witness."""
    return PatchLimits(
        ByteSize(1024),
        ByteSize(16),
        ByteSize(256),
        ByteSize(16),
        ByteSize(16),
        ByteSize(4096),
        ByteSize(4096),
        ByteSize(4096),
        DurationTicks(100),
        DurationTicks(100),
        DurationTicks(100),
    )


def _policy(
    revision: PolicyRevision,
    route: PolicyRouteId,
) -> TrustedPatchPolicy:
    """Return a fully preauthorized local policy bound to one route."""
    preauthorization = PreauthorizationClass("remote-test")
    return TrustedPatchPolicy(
        revision,
        frozenset((OperationType.EDIT, OperationType.APPLY)),
        (
            PolicyRule(
                PolicyPathSelector(None),
                tuple(
                    CapabilityMode(
                        capability,
                        ApprovalMode.PREAUTHORIZED,
                        preauthorization,
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
            route,
            PolicyBrokerId("remote-test-broker"),
            PolicyReviewerRole("remote-test-reviewer"),
            1,
            preauthorization,
        ),
    )


def _authority(principal: str = "principal-test") -> RemotePatchAuthority:
    """Return the complete single test principal scope."""
    return RemotePatchAuthority(
        tenant=PatchTenantId("tenant-test"),
        principal=PatchPrincipalId(principal),
        run=PatchRunId("run-test"),
        session=PatchSessionId("session-test"),
        task=PatchTaskId("task-test"),
        agent=PatchAgentId("agent-test"),
        execution_scope="domain_" + "a" * 16,
        route=PolicyRouteId("route-test"),
        context=PatchContextId("context_" + "a" * 16),
        workspace=PatchWorkspaceId("workspace_" + "a" * 16),
        policy_revision=PolicyRevision("policy-test"),
        disclosures=frozenset(),
        approval_route=PolicyRouteId("route-test"),
        correlation="correlation-test",
        capabilities=frozenset(Capability),
    )


def _runtime_witness(
    authority: RemotePatchAuthority,
) -> RemotePatchRuntimeWitness:
    """Return the sealed runtime witness matching one test authority."""
    return RemotePatchRuntimeWitness(
        tenant=authority.tenant,
        principal=authority.principal,
        run=authority.run,
        session=authority.session,
        task=authority.task,
        agent=authority.agent,
        execution_scope=authority.execution_scope,
        route=authority.route,
        context=authority.context,
        workspace=authority.workspace,
        policy_revision=authority.policy_revision,
        disclosures=authority.disclosures,
        approval_route=authority.approval_route,
        capabilities=authority.capabilities,
    )


def _configuration(
    binder: _Binder,
    authority: RemotePatchAuthority,
) -> RemotePatchTestServerConfiguration:
    """Return one complete activation configuration with an in-memory store."""
    return RemotePatchTestServerConfiguration(
        profile=RemotePatchTestServerProfile(
            enabled=True,
            authenticated=True,
            loopback_only=True,
        ),
        authority_resolver=_Resolver(authority),
        expected_authority=authority,
        binder=binder,
        store=InMemoryDurablePatchStore(InMemoryDurablePatchBackend()),
        handle_key=b"a" * 32,
        runtime_witness=_runtime_witness(authority),
    )


def _active_configuration(
    authority: RemotePatchAuthority,
) -> tuple[
    RemotePatchTestServerConfiguration,
    _RuntimeBinder,
    _RemoteService,
]:
    """Return one fully scope-bound local authenticated test server setup."""
    store = InMemoryDurablePatchStore(InMemoryDurablePatchBackend())
    service = _RemoteService(store)
    binder = _RuntimeBinder(authority, service)

    return (
        RemotePatchTestServerConfiguration(
            profile=RemotePatchTestServerProfile(
                enabled=True,
                authenticated=True,
                loopback_only=True,
            ),
            authority_resolver=_Resolver(authority),
            expected_authority=authority,
            binder=binder,
            store=store,
            handle_key=b"a" * 32,
            runtime_witness=_runtime_witness(authority),
        ),
        binder,
        service,
    )


def test_remote_host_rejects_unsealed_identity_and_json_before_dispatch() -> (
    None
):
    """Reject unsealed remote host inputs without starting another effect."""

    async def scenario() -> None:
        authority = _authority()
        with pytest.raises(PatchToolError, match="handshake"):
            replace(_runtime_witness(authority), execution_scope="")
        configuration, _, service = _active_configuration(authority)
        controller = RemotePatchController(configuration)
        arguments = {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        }
        try:
            await controller.start()
            host = controller._host
            assert type(host) is PatchSdkHost
            operation = controller._operation(authority, "host-remote-key")
            outcome = await host.invoke_remote_json(
                OperationType.EDIT,
                arguments,
                operation.request_id,
                operation.correlation_id,
                operation.identity,
            )
            assert isinstance(outcome, PatchResult)
            assert len(service.remote_calls) == 1
            with pytest.raises(PatchToolError, match="arguments"):
                await host.invoke_remote_json(
                    OperationType.EDIT,
                    {"path": object()},
                    PatchRequestId.new(),
                    PatchObserverCorrelationId.new(),
                    operation.identity,
                )
            with pytest.raises(PatchToolError, match="remote invocation"):
                await host.invoke_remote_raw(
                    OperationType.EDIT,
                    dumps(arguments).encode("utf-8"),
                    PatchRequestId.new(),
                    PatchObserverCorrelationId.new(),
                    cast(DurableRequestIdentity, object()),
                )
            with pytest.raises(PatchToolError, match="SDK request"):
                await host._invoke_raw_with_identity(
                    OperationType.EDIT,
                    dumps(arguments).encode("utf-8"),
                    PatchRequestId.new(),
                    PatchObserverCorrelationId.new(),
                    cast(DurableRequestIdentity, object()),
                )
            assert len(service.remote_calls) == 1
        finally:
            await controller.close()

    run(scenario())


def test_remote_host_never_dispatches_through_a_nonremote_service() -> None:
    """Reconcile instead of widening a remote identity to an ordinary SDK."""

    class _NonRemoteService:
        """Expose the base SDK protocol without remote dispatch authority."""

        def __init__(self, delegate: _RemoteService) -> None:
            """Delegate only ordinary SDK and settlement operations."""
            self._delegate = delegate
            self._request_id: PatchRequestId | None = None
            self.store = delegate.store
            self.settlement = self

        def inspect(
            self, _: PatchInvocationHandle
        ) -> Future[PatchInvocationOutcome]:
            """Return only a current terminal reconciliation record."""
            future: Future[PatchInvocationOutcome] = (
                get_running_loop().create_future()
            )
            future.set_result(_result(self._request_id))
            return future

        def await_terminal(
            self,
            _: PatchInvocationHandle,
            __: PatchPending,
        ) -> Future[PatchResult]:
            """Return only a current terminal settlement record."""
            future: Future[PatchResult] = get_running_loop().create_future()
            future.set_result(_result(self._request_id))
            return future

        async def invoke(
            self,
            operation: OperationType,
            raw_arguments: bytes,
            capability: PatchInvocationCapability,
            request_id: PatchRequestId,
            correlation_id: PatchObserverCorrelationId,
        ) -> PatchResult:
            """Delegate only the ordinary SDK invocation contract."""
            return await self._delegate.invoke(
                operation,
                raw_arguments,
                capability,
                request_id,
                correlation_id,
            )

        async def review(
            self, handle: PatchInvocationHandle
        ) -> Mapping[str, object]:
            """Delegate the closed review projection."""
            return await self._delegate.review(handle)

        async def approve(self, handle: PatchInvocationHandle) -> PatchResult:
            """Delegate terminal approval projection."""
            return await self._delegate.approve(handle)

        def subscribe(
            self, handle: PatchInvocationHandle
        ) -> AsyncIterator[PatchLifecycleEvent]:
            """Delegate the content-free lifecycle stream."""
            return self._delegate.subscribe(handle)

    async def scenario() -> None:
        authority = _authority()
        configuration, _, delegate = _active_configuration(authority)
        service = _NonRemoteService(delegate)
        controller = RemotePatchController(
            replace(
                configuration,
                binder=_RuntimeBinder(
                    authority,
                    cast(_RemoteService, service),
                ),
            )
        )
        arguments = {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        }
        try:
            await controller.start()
            host = controller._host
            assert type(host) is PatchSdkHost
            operation = controller._operation(authority, "nonremote-host-key")
            service._request_id = operation.request_id
            outcome = await host._invoke_raw_with_identity(
                OperationType.EDIT,
                dumps(arguments).encode("utf-8"),
                operation.request_id,
                operation.correlation_id,
                operation.identity,
            )
            assert isinstance(outcome, PatchResult)
            assert delegate.remote_calls == []
        finally:
            await controller.close()

    run(scenario())


def _loopback_port() -> int:
    """Reserve and release one ephemeral loopback port for one child server."""
    with socket(AF_INET, SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _continuation_configuration(
    authority: RemotePatchAuthority,
    store: _ContinuationStore,
) -> RemotePatchTestServerConfiguration:
    """Bind a controller to scripted durable continuation truth only."""
    return replace(
        _configuration(_Binder(), authority),
        store=cast(DurablePatchStore, store),
    )


def _pending(
    request_id: PatchRequestId,
    execution_id: PatchExecutionId,
    correlation_id: PatchObserverCorrelationId,
) -> DurablePendingRecord:
    """Return one exact original-identity pending branch record."""
    return DurablePendingRecord(
        request_id,
        execution_id,
        PatchPendingOperationId("pending_" + "a" * 16),
        correlation_id,
        SequenceNumber(1),
        SequenceNumber(1),
        False,
        DurationTicks(1),
    )


def _terminal(
    request_id: PatchRequestId,
    correlation_id: PatchObserverCorrelationId,
) -> tuple[DurableTerminalRecord, DurableOutboxRecord]:
    """Return one exact terminal result and matching durable event record."""
    outbox = DurableOutboxRecord(
        PatchEventId("event_" + "a" * 16),
        request_id,
        SequenceNumber(1),
        LifecyclePhase.REQUEST_COMPLETED,
        correlation_id,
    )
    return (
        DurableTerminalRecord(
            _result(request_id),
            outbox,
            None,
        ),
        outbox,
    )


def _request(authority: RemotePatchAuthority) -> Request:
    """Build one loopback authenticated request for direct controller tests."""
    return Request(
        {
            "client": ("127.0.0.1", 1),
            "headers": [
                (
                    b"x-avalan-correlation",
                    authority.correlation.encode("ascii"),
                )
            ],
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "scheme": "http",
            "server": ("127.0.0.1", 80),
            "type": "http",
        }
    )


def _request_with_retransmission_key(
    authority: RemotePatchAuthority,
    retransmission_key: str,
) -> Request:
    """Build one authenticated request carrying a server-owned key input."""
    return Request(
        {
            "client": ("127.0.0.1", 1),
            "headers": [
                (
                    b"x-avalan-correlation",
                    authority.correlation.encode("ascii"),
                ),
                (b"idempotency-key", retransmission_key.encode("ascii")),
            ],
            "method": "POST",
            "path": "/",
            "query_string": b"",
            "scheme": "http",
            "server": ("127.0.0.1", 80),
            "type": "http",
        }
    )


def test_remote_patch_routes_are_absent_without_explicit_installation() -> (
    None
):
    """Keep default FastAPI applications free of the test-only route."""

    async def scenario() -> None:
        app = FastAPI()
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 1))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/__avalan_test__/patch/v1/tools")
        assert response.status_code == 404

    run(scenario())


def test_remote_client_and_configuration_reject_invalid_values() -> None:
    """Reject malformed local-test configuration and public client input."""

    async def scenario() -> None:
        authority = _authority()
        with pytest.raises(RemotePatchServerError):
            replace(authority, execution_scope="not a scope")
        with pytest.raises(RemotePatchServerError):
            RemotePatchTestServerProfile(name="production")
        with pytest.raises(RemotePatchServerError):
            replace(_configuration(_Binder(), authority), handle_key=b"short")
        async with httpx.AsyncClient() as http_client:
            with pytest.raises(RemotePatchServerError):
                RemotePatchTestClient(http_client, "invalid correlation!")
            client = RemotePatchTestClient(http_client, authority.correlation)
            with pytest.raises(RemotePatchServerError):
                await client.events("opaque", -1)
            with pytest.raises(RemotePatchServerError):
                await client.edit(
                    "note.txt",
                    [RemotePatchEditPart(old_text="before", new_text="after")],
                    "",
                )
        app = FastAPI()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as http_client:
            client = RemotePatchTestClient(http_client, authority.correlation)
            with pytest.raises(RemotePatchServerError):
                await client.tools()
            with pytest.raises(RemotePatchServerError):
                await client.apply("not a patch", "apply-error-key")
            with pytest.raises(RemotePatchServerError):
                await client.cancel_intent("opaque")
        with pytest.raises(RemotePatchServerError):
            patch_server._operation_response(httpx.Response(200, json=[]))
        with pytest.raises(RemotePatchServerError):
            patch_server._operation_response(
                httpx.Response(200, json={"state": "unknown"})
            )

    run(scenario())


def test_remote_client_reads_terminal_result_and_event_replay() -> None:
    """Project a terminal durable continuation through the public client."""

    async def scenario() -> None:
        authority = _authority()
        store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        configuration = _continuation_configuration(authority, store)
        controller = RemotePatchController(configuration)
        operation = controller._operation(authority, "client-terminal-key")
        terminal, outbox = _terminal(
            operation.request_id,
            operation.correlation_id,
        )
        store.snapshot = _ContinuationSnapshot(None, terminal)
        store.records = (outbox,)
        app = FastAPI()
        install_remote_patch_test_routes(
            app,
            configuration,
            controller=controller,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as http_client:
            client = RemotePatchTestClient(http_client, authority.correlation)
            completed = await client.await_result(
                controller._seal_operation(operation)
            )
            async with client.events(
                controller._seal_operation(operation)
            ) as events:
                assert events.__aiter__() is events
                event = await anext(events)
                with pytest.raises(StopAsyncIteration):
                    await anext(events)
        assert completed.state == "completed"
        assert event.event_id == outbox.event_id.value
        assert event.cursor == outbox.sequence.value

    run(scenario())


def test_managed_sse_client_rejects_closed_and_malformed_frames() -> None:
    """Close every bad stream path without retaining a response or cursor."""

    @dataclass
    class _ResponseContext:
        """Expose one close-observed static HTTPX stream response."""

        response: httpx.Response
        closes: int = 0

        async def __aenter__(self) -> httpx.Response:
            """Return the prebuilt test response without buffering it."""
            return self.response

        async def __aexit__(self, *_: object) -> None:
            """Record that the managed stream closed this response."""
            self.closes += 1

    @dataclass
    class _FailingResponseContext:
        """Raise while opening one response context and record cleanup."""

        failure: BaseException
        closes: int = 0

        async def __aenter__(self) -> httpx.Response:
            """Raise the exact configured connection-opening failure."""
            raise self.failure

        async def __aexit__(self, *_: object) -> None:
            """Record that failed stream entry attempted one cleanup."""
            self.closes += 1

    class _FailingLineIterator:
        """Raise a transport-read failure after the response opens."""

        def __aiter__(self) -> "_FailingLineIterator":
            """Return the one failing asynchronous line iterator."""
            return self

        async def __anext__(self) -> str:
            """Raise one non-protocol stream read error."""
            raise RuntimeError("stream read failed")

    class _FailingLineResponse:
        """Expose a syntactically valid SSE response with a failed reader."""

        status_code = 200
        headers: Mapping[str, str] = {"content-type": "text/event-stream"}

        def aiter_lines(self) -> AsyncIterator[str]:
            """Return the one failing asynchronous line reader."""
            return _FailingLineIterator()

    async def rejected(
        response: httpx.Response,
        after: int = 0,
    ) -> None:
        """Require one managed malformed response to close before failure."""
        context = _ResponseContext(response)
        stream = RemotePatchEventStream(context, after)
        with pytest.raises(RemotePatchServerError):
            async with stream:
                await anext(stream)
        assert context.closes == 1

    async def scenario() -> None:
        unopened = RemotePatchEventStream(
            _ResponseContext(httpx.Response(200)),
            0,
        )
        with pytest.raises(RemotePatchServerError):
            unopened.__aiter__()
        with pytest.raises(RemotePatchServerError):
            await anext(unopened)
        await unopened.aclose()
        with pytest.raises(RemotePatchServerError):
            await unopened.__aenter__()
        failed_open = _FailingResponseContext(RuntimeError("connect failed"))
        with pytest.raises(RemotePatchServerError):
            await RemotePatchEventStream(failed_open, 0).__aenter__()
        assert failed_open.closes == 1
        cancelled_open = _FailingResponseContext(CancelledError())
        with pytest.raises(CancelledError):
            await RemotePatchEventStream(cancelled_open, 0).__aenter__()
        assert cancelled_open.closes == 1
        context = _ResponseContext(
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b"id: 1\nevent: patch.lifecycle\ndata: "
                    b'{"cursor":2,"event_id":"event_bad","lifecycle":'
                    b'"planned","object":"patch.event"}\n\n'
                ),
            )
        )
        malformed = RemotePatchEventStream(context, 0)
        await malformed.__aenter__()
        with pytest.raises(RemotePatchServerError):
            await anext(malformed)
        await malformed.aclose()
        assert context.closes == 1
        context = _ResponseContext(
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b"id: 1\nevent: patch.lifecycle\ndata: "
                    b'{"cursor":1,"event_id":"event_done","lifecycle":'
                    b'"request_completed","object":"patch.event"}\n\n'
                ),
            )
        )
        completed = RemotePatchEventStream(context, 0)
        await completed.__aenter__()
        assert (await anext(completed)).event_id == "event_done"
        with pytest.raises(StopAsyncIteration):
            await anext(completed)
        assert context.closes == 1
        context = _ResponseContext(
            cast(httpx.Response, _FailingLineResponse())
        )
        failed_read = RemotePatchEventStream(context, 0)
        await failed_read.__aenter__()
        with pytest.raises(RemotePatchServerError):
            await anext(failed_read)
        assert context.closes == 1
        await rejected(httpx.Response(404))
        await rejected(
            httpx.Response(
                200,
                headers={"content-type": "application/json"},
            )
        )
        await rejected(
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"id: 1\nid: 1\n\n",
            )
        )
        await rejected(
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"id: 1\nevent: patch.lifecycle\ndata: []\n\n",
            )
        )
        await rejected(
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b"id: 1\nevent: patch.lifecycle\ndata: "
                    b'{"cursor":2,"event_id":"event_bad","lifecycle":'
                    b'"planned","object":"patch.event"}\n\n'
                ),
            )
        )
        await rejected(
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b"id: 1\nevent: patch.lifecycle\ndata: "
                    b'{"cursor":1,"event_id":"event_old","lifecycle":'
                    b'"planned","object":"patch.event"}\n\n'
                ),
            ),
            after=2,
        )
        await rejected(
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b"id: 1\nevent: patch.lifecycle\ndata: "
                    b'{"cursor":1,"event_id":"event_new","lifecycle":'
                    b'"planned","object":"patch.event"}\n\n'
                ),
            ),
            after=1,
        )
        with pytest.raises(RemotePatchServerError):
            patch_server._sse_event(None, None, None)

    run(scenario())


def test_remote_controller_covers_start_terminal_timeout_and_cancel_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep invalid starts and detached lifecycle edge cases fail closed."""
    incorrect_bindings: list[PatchRuntimeBinding] = []

    class _IncorrectLoader:
        """Return a correct binding with an invalid selected toolset."""

        def __init__(self, *_: object, **__: object) -> None:
            """Accept the loader constructor arguments without using them."""

        async def load(self, **_: object) -> object:
            """Return an object that violates the exact toolset identity."""
            return SimpleNamespace(
                runtime_binding=incorrect_bindings[0],
                toolset=object(),
            )

    class _CancelledHost:
        """Propagate a caller disconnect through detached dispatch."""

        async def invoke_remote_raw(self, *_: object) -> object:
            """Signal cancellation instead of request cancellation intent."""
            raise CancelledError

    async def scenario() -> None:
        authority = _authority()
        configuration, binder, _ = _active_configuration(authority)
        incorrect_bindings.append(await binder.bind())
        monkeypatch.setattr(patch_server, "PatchToolLoader", _IncorrectLoader)
        incorrect_controller = RemotePatchController(configuration)
        with pytest.raises(RemotePatchServerError):
            await incorrect_controller.start()
        monkeypatch.undo()

        def failing_sdk_host(_: PatchToolSet) -> PatchSdkHost:
            """Fail after toolset entry so startup performs exact cleanup."""
            raise RuntimeError("host construction failed")

        monkeypatch.setattr(PatchToolSet, "sdk_host", failing_sdk_host)
        cleanup_controller = RemotePatchController(configuration)
        with pytest.raises(RuntimeError):
            await cleanup_controller.start()
        assert cleanup_controller._toolset is None
        monkeypatch.undo()

        store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        continuation = RemotePatchController(
            _continuation_configuration(authority, store)
        )
        operation = continuation._operation(authority, "terminal-begin-key")
        terminal, _ = _terminal(
            operation.request_id,
            operation.correlation_id,
        )
        store.snapshot = _ContinuationSnapshot(None, terminal)
        completed = await continuation.begin(
            _request_with_retransmission_key(
                authority,
                "terminal-begin-key",
            ),
            OperationType.EDIT,
            {
                "path": "note.txt",
                "edits": [{"old_text": "before", "new_text": "after"}],
            },
        )
        assert completed.state == "completed"

        pending_operation = continuation._operation(
            authority,
            "timeout-pending-key",
        )
        store.snapshot = _ContinuationSnapshot(
            _pending(
                pending_operation.request_id,
                pending_operation.identity.execution_id,
                pending_operation.correlation_id,
            ),
            None,
        )
        monkeypatch.setattr(patch_server, "_AWAIT_POLLS", 1)
        timed_out = await continuation.await_result(
            _request(authority),
            continuation._seal_operation(pending_operation),
        )
        assert timed_out.state == "pending"

        continuation._host = cast(PatchSdkHost, _CancelledHost())
        with pytest.raises(CancelledError):
            await continuation._dispatch(
                OperationType.EDIT,
                b"{}",
                pending_operation,
            )
        await continuation.close()

    run(scenario())


def test_remote_routes_and_malformed_event_identity_remain_closed() -> None:
    """Cover the public error boundary and explicit application lifetime."""

    async def scenario() -> None:
        authority = _authority()
        configuration = _configuration(_Binder(), authority)
        app = FastAPI()
        controller = install_remote_patch_test_routes(app, configuration)
        headers = {"X-Avalan-Correlation": authority.correlation}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            apply = await client.post(
                "/__avalan_test__/patch/v1/apply",
                headers={**headers, "Idempotency-Key": "route-apply-key"},
                json={"patch": "not a canonical patch"},
            )
            paths = (
                ("GET", "/operations/not-a-token"),
                ("POST", "/operations/not-a-token/await"),
                ("POST", "/operations/not-a-token/cancel"),
                ("GET", "/operations/not-a-token/events"),
            )
            responses = [
                await client.request(
                    method,
                    "/__avalan_test__/patch/v1" + path,
                    headers=headers,
                )
                for method, path in paths
            ]
        assert apply.status_code == 404
        assert [response.status_code for response in responses] == [404] * 4
        await controller.close()

        lifecycle_configuration, binder, _ = _active_configuration(authority)
        lifecycle_app = patch_server.remote_patch_test_server(
            lifecycle_configuration
        )
        async with lifecycle_app.router.lifespan_context(lifecycle_app):
            assert binder.calls == 1
            assert lifecycle_app.state.remote_patch_test_controller

        attached_app = FastAPI()
        attached_controller = RemotePatchController(
            _configuration(_Binder(), authority)
        )
        patch_server.install_remote_patch_test_routes_for_controller(
            attached_app,
            attached_controller,
        )
        assert attached_app.state.remote_patch_test_controller is (
            attached_controller
        )

        mismatched_store = _ContinuationStore(
            _ContinuationSnapshot(None, None), ()
        )
        mismatched_controller = RemotePatchController(
            _continuation_configuration(authority, mismatched_store)
        )
        operation = mismatched_controller._operation(
            authority,
            "mismatched-event-key",
        )
        other_operation = mismatched_controller._operation(
            authority,
            "other-event-key",
        )
        terminal, _ = _terminal(
            operation.request_id,
            operation.correlation_id,
        )
        _, mismatched_record = _terminal(
            other_operation.request_id,
            other_operation.correlation_id,
        )
        mismatched_store.snapshot = _ContinuationSnapshot(None, terminal)
        mismatched_store.records = (mismatched_record,)
        response = await mismatched_controller.events(
            _request(authority),
            mismatched_controller._seal_operation(operation),
            0,
        )
        assert [chunk async for chunk in response.body_iterator] == []

        incomplete_handle = mismatched_controller._fernet.encrypt(
            dumps({}).encode("utf-8")
        ).decode("ascii")
        with pytest.raises(RemotePatchServerError):
            mismatched_controller._open_operation(incomplete_handle, authority)

    run(scenario())


def test_remote_controller_coarsens_durable_and_dispatch_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover detached dispatch, pending wait, cancellation, and close paths."""

    class _FailingHost:
        """Raise one detached invocation fault without exposing details."""

        async def invoke_remote_raw(self, *_: object) -> object:
            """Fail the one server-owned dispatch attempt."""
            raise RuntimeError("remote transport failed")

    async def scenario() -> None:
        authority = _authority()
        configuration, binder, _ = _active_configuration(authority)
        controller = RemotePatchController(configuration)
        operation = controller._operation(authority, "controller-error-key")
        handle = controller._seal_operation(operation)
        with pytest.raises(RemotePatchServerError):
            await controller.begin(
                _request(authority),
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "before", "new_text": "after"}],
                },
            )
        with pytest.raises(RemotePatchServerError):
            controller._operation(authority, "")
        for malformed in ("not-a-token", "x" * 8193):
            with pytest.raises(RemotePatchServerError):
                controller._open_operation(malformed, authority)
        invalid_payload = {
            "authority": authority.canonical(),
            "correlation_id": operation.correlation_id.value,
            "identity": {},
            "request_id": operation.request_id.value,
            "version": 0,
        }
        invalid_handle = controller._fernet.encrypt(
            dumps(invalid_payload).encode("utf-8")
        ).decode("ascii")
        with pytest.raises(RemotePatchServerError):
            controller._open_operation(invalid_handle, authority)
        with pytest.raises(RemotePatchServerError):
            await controller.events(_request(authority), handle, -1)
        await controller._dispatch(OperationType.EDIT, b"{}", operation)
        controller._host = cast(PatchSdkHost, _FailingHost())
        await controller._dispatch(OperationType.EDIT, b"{}", operation)
        assert binder.calls == 0
        store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        pending_controller = RemotePatchController(
            _continuation_configuration(authority, store)
        )
        pending_operation = pending_controller._operation(
            authority, "pending-error-key"
        )
        store.snapshot = _ContinuationSnapshot(
            _pending(
                pending_operation.request_id,
                pending_operation.identity.execution_id,
                pending_operation.correlation_id,
            ),
            None,
        )
        pending_handle = pending_controller._seal_operation(pending_operation)

        async def fail_cancel(_: object) -> _ContinuationSnapshot:
            """Raise one durable cancellation error for the projection."""
            raise DurableStoreError(DurableStoreErrorCode.FENCED)

        monkeypatch.setattr(patch_server, "_AWAIT_POLLS", 1)
        pending = await pending_controller.await_result(
            _request(authority), pending_handle
        )
        assert pending.state == "pending"
        assert store.await_calls == 0
        monkeypatch.setattr(store, "request_cancellation", fail_cancel)
        with pytest.raises(RemotePatchServerError):
            await pending_controller.cancel_intent(
                _request(authority), pending_handle
            )
        store.snapshot = _ContinuationSnapshot(None, None)

        async def settle_after_one_sleep(_: float) -> None:
            """Turn one no-pending await loop into terminal durable truth."""
            terminal, ignored_record = _terminal(
                pending_operation.request_id,
                pending_operation.correlation_id,
            )
            del ignored_record
            store.snapshot = _ContinuationSnapshot(None, terminal)

        monkeypatch.setattr(patch_server, "sleep", settle_after_one_sleep)
        terminal = await pending_controller.await_result(
            _request(authority), pending_handle
        )
        assert terminal.state == "completed"

        async def wait_for_close() -> None:
            """Block until controller shutdown cancels this detached task."""
            await Event().wait()

        task = create_task(wait_for_close())
        controller._tasks[operation.request_id] = task
        await controller.close()
        assert task.cancelled()

    run(scenario())


def test_remote_helpers_and_event_errors_remain_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed helpers and stop SSE on durable cancellation faults."""

    class _OutboxFailureStore(_ContinuationStore):
        """Raise a durable outbox failure after authenticated access."""

        async def outbox(
            self,
            access: object,
            after: SequenceNumber,
            limit: int,
        ) -> tuple[DurableOutboxRecord, ...]:
            """Reject one event continuation read without partial output."""
            del access, after, limit
            raise DurableStoreError(DurableStoreErrorCode.FENCED)

    async def scenario() -> None:
        authority = _authority()
        controller = RemotePatchController(
            _configuration(_Binder(), authority)
        )
        operation = controller._operation(authority, "helper-error-key")
        handle = controller._seal_operation(operation)
        limits = _configuration(_Binder(), authority).input_limits
        with pytest.raises(RemotePatchServerError):
            patch_server._canonical_arguments({"value": object()})
        with pytest.raises(RemotePatchServerError):
            patch_server._parse_digest(
                OperationType.EDIT,
                b"not json",
                operation.correlation_id,
                limits,
            )
        with pytest.raises(RemotePatchServerError):
            patch_server._reject_forbidden_caller_fields(
                {"safe": [{"policy": "denied"}]}
            )
        with pytest.raises(RemotePatchServerError):
            patch_server._terminal_response(
                handle,
                cast(
                    DurableRequestSnapshot,
                    SimpleNamespace(terminal=None, pending=None),
                ),
            )
        with pytest.raises(RemotePatchServerError):
            patch_server._handle_retransmission_key({"identity": {}})
        with pytest.raises(RemotePatchServerError):
            patch_server._handle_retransmission_key(
                {
                    "identity": {
                        "execution_id": "one",
                        "principal_id": "two",
                        "retransmission_key": 3,
                        "route_id": "four",
                        "tenant_id": "five",
                    }
                }
            )
        no_client = Request(
            {
                "headers": [],
                "method": "GET",
                "path": "/",
                "query_string": b"",
                "scheme": "http",
                "server": ("127.0.0.1", 80),
                "type": "http",
            }
        )
        invalid_client = Request(
            {
                "client": ("not an ip", 1),
                "headers": [],
                "method": "GET",
                "path": "/",
                "query_string": b"",
                "scheme": "http",
                "server": ("127.0.0.1", 80),
                "type": "http",
            }
        )
        assert not patch_server._is_loopback(no_client)
        assert not patch_server._is_loopback(invalid_client)
        with pytest.raises(RemotePatchServerError):
            install_remote_patch_test_routes(
                cast(FastAPI, object()),
                _configuration(_Binder(), authority),
            )
        terminal, _ = _terminal(operation.request_id, operation.correlation_id)
        store = _OutboxFailureStore(_ContinuationSnapshot(None, terminal), ())
        event_controller = RemotePatchController(
            _continuation_configuration(authority, store)
        )
        event_handle = event_controller._seal_operation(operation)
        response = await event_controller.events(
            _request(authority), event_handle, 0
        )
        assert [chunk async for chunk in response.body_iterator] == []

        async def cancel_sleep(_: float) -> None:
            """Interrupt an unfinished SSE polling wait with cancellation."""
            raise CancelledError

        polling_store = _ContinuationStore(
            _ContinuationSnapshot(None, None), ()
        )
        polling_controller = RemotePatchController(
            _continuation_configuration(authority, polling_store)
        )
        monkeypatch.setattr(patch_server, "sleep", cancel_sleep)
        with pytest.raises(CancelledError):
            await anext(
                polling_controller._event_stream(operation, SequenceNumber(0))
            )

    run(scenario())


def test_unauthenticated_or_nonloopback_requests_never_bind_target() -> None:
    """Suppress patch advertisement without runtime or target inspection."""

    async def scenario() -> None:
        authority = _authority()
        binder = _Binder()
        app = FastAPI()
        install_remote_patch_test_routes(
            app, _configuration(binder, authority)
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=app,
                client=("203.0.113.12", 1),
            ),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/__avalan_test__/patch/v1/tools")
        assert response.status_code == 200
        assert response.json() == {"object": "list", "data": []}
        assert binder.calls == 0

    run(scenario())


def test_missing_authenticated_correlation_suppresses_advertisement() -> None:
    """Require trusted correlation before the server binds any runtime."""

    async def scenario() -> None:
        authority = _authority()
        binder = _Binder()
        app = FastAPI()
        install_remote_patch_test_routes(
            app, _configuration(binder, authority)
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/__avalan_test__/patch/v1/tools")
        assert response.status_code == 200
        assert response.json() == {"object": "list", "data": []}
        assert binder.calls == 0

    run(scenario())


def test_empty_principal_suppresses_advertisement_without_binding() -> None:
    """Keep a configured test route inert when authentication finds nobody."""

    async def scenario() -> None:
        authority = _authority()
        binder = _Binder()
        configuration = replace(
            _configuration(binder, authority),
            authority_resolver=_NoPrincipalResolver(),
        )
        app = FastAPI()
        install_remote_patch_test_routes(app, configuration)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            response = await client.get(
                "/__avalan_test__/patch/v1/tools",
                headers={"X-Avalan-Correlation": authority.correlation},
            )
        assert response.status_code == 200
        assert response.json() == {"object": "list", "data": []}
        assert binder.calls == 0

    run(scenario())


def test_mismatched_runtime_durable_store_suppresses_advertisement() -> None:
    """Fail closed when persistence does not name the bound runtime store."""

    async def scenario() -> None:
        authority = _authority()
        configuration, binder, service = _active_configuration(authority)
        app = FastAPI()
        install_remote_patch_test_routes(
            app,
            replace(
                configuration,
                store=InMemoryDurablePatchStore(InMemoryDurablePatchBackend()),
            ),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            response = await client.get(
                "/__avalan_test__/patch/v1/tools",
                headers={"X-Avalan-Correlation": authority.correlation},
            )
        assert response.status_code == 200
        assert response.json() == {"object": "list", "data": []}
        assert binder.calls == 1
        assert service.remote_calls == []

    run(scenario())


def test_runtime_witness_rejects_every_scope_coordinate_before_binding() -> (
    None
):
    """Reject each static runtime mismatch before target probing or loading."""
    authority = _authority()
    binder = _Binder()
    configuration = _configuration(binder, authority)
    witness = configuration.runtime_witness
    other_route = PolicyRouteId("route-other")
    alternatives = (
        replace(witness, tenant=PatchTenantId("tenant-other")),
        replace(witness, principal=PatchPrincipalId("principal-other")),
        replace(witness, run=PatchRunId("run-other")),
        replace(witness, session=PatchSessionId("session-other")),
        replace(witness, task=PatchTaskId("task-other")),
        replace(witness, agent=PatchAgentId("agent-other")),
        replace(witness, execution_scope="domain_" + "b" * 16),
        replace(witness, route=other_route),
        replace(witness, context=PatchContextId("context_" + "b" * 16)),
        replace(
            witness,
            workspace=PatchWorkspaceId("workspace_" + "b" * 16),
        ),
        replace(witness, policy_revision=PolicyRevision("policy-other")),
        replace(
            witness,
            disclosures=frozenset((PolicyDisclosure.SERVER_EXACT_TRUTH,)),
        ),
        replace(witness, approval_route=other_route),
        replace(
            witness,
            capabilities=frozenset((Capability.READ_FOR_MUTATION,)),
        ),
    )
    for alternate in alternatives:
        with pytest.raises(RemotePatchServerError):
            replace(configuration, runtime_witness=alternate)
    assert binder.calls == 0


def test_selected_runtime_witness_mismatch_suppresses_advertisement() -> None:
    """Refuse an inspected runtime whose selected service witness differs."""

    async def scenario() -> None:
        authority = _authority()
        configuration, binder, service = _active_configuration(authority)
        binder.authority = replace(
            authority,
            capabilities=frozenset((Capability.READ_FOR_MUTATION,)),
        )
        app = FastAPI()
        install_remote_patch_test_routes(app, configuration)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            response = await client.get(
                "/__avalan_test__/patch/v1/tools",
                headers={"X-Avalan-Correlation": authority.correlation},
            )
        assert response.json() == {"object": "list", "data": []}
        assert binder.calls == 1
        assert service.remote_calls == []

    run(scenario())


def test_closed_bodies_reject_authority_controls_before_runtime_binding() -> (
    None
):
    """Reject caller workspace and schema widening fields without binding."""

    async def scenario() -> None:
        authority = _authority()
        binder = _Binder()
        app = FastAPI()
        install_remote_patch_test_routes(
            app, _configuration(binder, authority)
        )
        headers = {
            "Idempotency-Key": "same-key",
            "X-Avalan-Correlation": authority.correlation,
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/__avalan_test__/patch/v1/edit",
                headers=headers,
                json={
                    "path": "note.txt",
                    "edits": [{"old_text": "before", "new_text": "after"}],
                    "workspace": "caller-selected",
                },
            )
        assert response.status_code == 400
        assert binder.calls == 0

    run(scenario())


def test_every_forbidden_authority_field_fails_before_runtime_binding() -> (
    None
):
    """Reject all documented caller controls under the closed edit schema."""

    async def scenario() -> None:
        authority = _authority()
        store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        app = FastAPI()
        install_remote_patch_test_routes(
            app, _continuation_configuration(authority, store)
        )
        headers = {
            "Idempotency-Key": "forbidden-field-key",
            "X-Avalan-Correlation": authority.correlation,
        }
        forbidden = (
            "approval",
            "approvals",
            "backend",
            "capabilities",
            "capability",
            "confirmation",
            "container_profile",
            "cwd",
            "disclosure",
            "limit",
            "limits",
            "matching_mode",
            "native_item_shape",
            "policy",
            "policy_version",
            "schema",
            "validator",
            "worker",
            "workspace",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            for field in forbidden:
                response = await client.post(
                    "/__avalan_test__/patch/v1/edit",
                    headers=headers,
                    json={
                        "path": "note.txt",
                        "edits": [{"old_text": "before", "new_text": "after"}],
                        field: "caller-controlled",
                    },
                )
                assert response.status_code == 400
        assert store.accesses == []

    run(scenario())


def test_raw_body_limit_precedes_auth_json_reservation_and_inspection() -> (
    None
):
    """Bound streamed raw bytes before any route-authenticated server work."""

    async def chunks(*parts: bytes) -> AsyncIterator[bytes]:
        """Yield one deliberately chunked body without a length header."""
        for part in parts:
            yield part

    async def scenario() -> None:
        authority = _authority()
        store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        configuration = replace(
            _continuation_configuration(authority, store),
            input_limits=PatchInputLimits(max_raw_bytes=32),
        )
        app = FastAPI()
        install_remote_patch_test_routes(app, configuration)
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": "raw-body-limit-key",
        }
        oversized_part = b'{"patch":"' + b"x" * 24 + b'"}'
        oversized_nested = b"[" * 33
        oversized_parts = (
            b'{"path":"x","edits":[',
            b'{"old_text":"x","new_text":""},',
            b'{"old_text":"x","new_text":""}]}',
        )
        exact = b'{"patch":"x"}' + b" " * (32 - len(b'{"patch":"x"}'))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            no_length = await client.post(
                "/__avalan_test__/patch/v1/apply",
                headers=headers,
                content=chunks(oversized_part[:11], oversized_part[11:]),
            )
            false_length = await client.post(
                "/__avalan_test__/patch/v1/apply",
                headers={**headers, "Content-Length": "0"},
                content=oversized_nested,
            )
            nested_parts = await client.post(
                "/__avalan_test__/patch/v1/edit",
                headers=headers,
                content=chunks(*oversized_parts),
            )
            malformed = await client.post(
                "/__avalan_test__/patch/v1/apply",
                headers=headers,
                content=b'{"patch":',
            )
            boundary = await client.post(
                "/__avalan_test__/patch/v1/apply",
                headers=headers,
                content=exact,
            )
            del app.state.remote_patch_test_input_bytes
            unavailable_limit = await client.post(
                "/__avalan_test__/patch/v1/apply",
                headers=headers,
                content=b"{}",
            )
        assert [
            no_length.status_code,
            false_length.status_code,
            nested_parts.status_code,
            malformed.status_code,
        ] == [400, 400, 400, 400]
        assert boundary.status_code == 404
        assert unavailable_limit.status_code == 400
        assert store.accesses == []

    run(scenario())


def test_tcp_attestation_is_required_before_authentication_or_binding() -> (
    None
):
    """Require the process-bound test secret before any remote test action."""

    async def scenario() -> None:
        authority = _authority()
        binder = _Binder()
        configuration = replace(
            _configuration(binder, authority),
            attestation_secret="attestation-test-secret",
        )
        app = FastAPI()
        install_remote_patch_test_routes(app, configuration)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            denied = await client.get(
                "/__avalan_test__/patch/v1/tools",
                headers={"X-Avalan-Correlation": authority.correlation},
            )
            client_with_secret = RemotePatchTestClient(
                client,
                authority.correlation,
                attestation_secret="attestation-test-secret",
            )
            allowed = await client_with_secret.tools()
        assert denied.json() == {"object": "list", "data": []}
        assert allowed.data == []
        assert binder.calls == 1

    run(scenario())


def test_encrypted_handle_binds_every_authority_coordinate() -> None:
    """Reject cross-principal handle replay without an operation oracle."""
    authority = _authority()
    controller = RemotePatchController(_configuration(_Binder(), authority))
    operation = controller._operation(authority, "replay-key")
    handle = controller._seal_operation(operation)

    assert controller._open_operation(handle, authority) == operation
    with pytest.raises(RemotePatchServerError):
        controller._open_operation(handle, _authority("principal-other"))


def test_handle_replay_rejects_each_exact_scope_coordinate() -> None:
    """Bind an opaque operation to every authenticated identity coordinate."""
    authority = _authority()
    controller = RemotePatchController(_configuration(_Binder(), authority))
    handle = controller._seal_operation(
        controller._operation(authority, "authority-replay-key")
    )
    alternate_route = PolicyRouteId("route-other")
    alternatives = (
        replace(authority, tenant=PatchTenantId("tenant-other")),
        replace(authority, principal=PatchPrincipalId("principal-other")),
        replace(authority, run=PatchRunId("run-other")),
        replace(authority, session=PatchSessionId("session-other")),
        replace(authority, task=PatchTaskId("task-other")),
        replace(authority, agent=PatchAgentId("agent-other")),
        replace(authority, execution_scope="scope-other"),
        replace(
            authority,
            route=alternate_route,
            approval_route=alternate_route,
        ),
        replace(authority, context=PatchContextId("context_" + "b" * 16)),
        replace(
            authority,
            workspace=PatchWorkspaceId("workspace_" + "b" * 16),
        ),
        replace(authority, policy_revision=PolicyRevision("policy-other")),
        replace(
            authority,
            disclosures=frozenset((PolicyDisclosure.SERVER_EXACT_TRUTH,)),
        ),
        replace(authority, correlation="correlation-other"),
    )
    for alternate in alternatives:
        with pytest.raises(RemotePatchServerError):
            controller._open_operation(handle, alternate)


def test_retransmission_reserves_before_one_remote_dispatch() -> None:
    """Use the public route and retain one server-derived durable identity."""

    async def scenario() -> None:
        authority = _authority()
        configuration, binder, service = _active_configuration(authority)
        app = FastAPI()
        controller = install_remote_patch_test_routes(app, configuration)
        headers = {
            "Idempotency-Key": "same-retransmission-key",
            "X-Avalan-Correlation": authority.correlation,
        }
        body = {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/__avalan_test__/patch/v1/edit",
                headers=headers,
                json=body,
            )
            second = await client.post(
                "/__avalan_test__/patch/v1/edit",
                headers=headers,
                json=body,
            )
        await sleep(0)
        try:
            assert first.status_code == 200
            assert second.status_code == 200
            assert (
                first.json()["operation_handle"]
                != second.json()["operation_handle"]
            )
            assert binder.calls == 1
            assert len(service.remote_calls) == 1
            remote_call = service.remote_calls[0]
            assert remote_call[0] is OperationType.EDIT
            assert remote_call[4].tenant_id == authority.tenant
            assert remote_call[4].principal_id == authority.principal
            assert remote_call[4].route_id == authority.route
        finally:
            await controller.close()

    run(scenario())


def test_retransmission_body_conflict_never_dispatches_a_second_effect() -> (
    None
):
    """Coarsen an idempotency collision before a second target invocation."""

    async def scenario() -> None:
        authority = _authority()
        configuration, _, service = _active_configuration(authority)
        app = FastAPI()
        controller = install_remote_patch_test_routes(app, configuration)
        headers = {
            "Idempotency-Key": "collision-key",
            "X-Avalan-Correlation": authority.correlation,
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/__avalan_test__/patch/v1/edit",
                headers=headers,
                json={
                    "path": "note.txt",
                    "edits": [{"old_text": "before", "new_text": "after"}],
                },
            )
            await sleep(0)
            collision = await client.post(
                "/__avalan_test__/patch/v1/edit",
                headers=headers,
                json={
                    "path": "note.txt",
                    "edits": [{"old_text": "before", "new_text": "changed"}],
                },
            )
        try:
            assert first.status_code == 200
            assert collision.status_code == 404
            assert len(service.remote_calls) == 1
        finally:
            await controller.close()

    run(scenario())


def test_failed_dispatch_same_key_is_pruned_without_blind_redispatch() -> None:
    """Use durable ownership before another effect attempt."""

    class _FailingRemoteService(_RemoteService):
        """Record one attempted remote dispatch before transport failure."""

        async def invoke_remote(
            self,
            operation: OperationType,
            raw_arguments: bytes,
            capability: PatchInvocationCapability,
            request_id: PatchRequestId,
            correlation_id: PatchObserverCorrelationId,
            identity: DurableRequestIdentity,
        ) -> PatchResult:
            """Record one call and fail without changing durable state."""
            self.remote_calls.append(
                (
                    operation,
                    raw_arguments,
                    request_id,
                    correlation_id,
                    identity,
                )
            )
            del capability
            raise RuntimeError("test dispatch transport failure")

    async def scenario() -> None:
        authority = _authority()
        store = InMemoryDurablePatchStore(InMemoryDurablePatchBackend())
        service = _FailingRemoteService(store)
        binder = _RuntimeBinder(authority, service)
        configuration = RemotePatchTestServerConfiguration(
            profile=RemotePatchTestServerProfile(
                enabled=True,
                authenticated=True,
                loopback_only=True,
            ),
            authority_resolver=_Resolver(authority),
            expected_authority=authority,
            binder=binder,
            store=store,
            handle_key=b"f" * 32,
            runtime_witness=_runtime_witness(authority),
        )
        controller = RemotePatchController(configuration)
        request = _request_with_retransmission_key(authority, "failed-key")
        arguments = {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        }
        first = await controller.begin(request, OperationType.EDIT, arguments)
        await sleep(0)
        task = controller._tasks[
            controller._operation(authority, "failed-key").request_id
        ]
        await task
        await sleep(0)
        replay = await controller.begin(request, OperationType.EDIT, arguments)
        await sleep(0)
        operation = controller._operation(authority, "failed-key")
        snapshot = await store.inspect(operation.access)
        assert first.state == replay.state == "pending"
        assert len(service.remote_calls) == 1
        assert controller._tasks == {}
        assert snapshot.lifecycle is LifecyclePhase.RECEIVED
        await controller.close()

    run(scenario())


def test_durable_commit_started_snapshot_never_starts_or_redispatches() -> (
    None
):
    """Use durable commit ownership rather than a new retry guess."""

    async def scenario() -> None:
        authority = _authority()
        binder = _Binder()
        store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        configuration = _continuation_configuration(authority, store)
        controller = RemotePatchController(configuration)
        operation = controller._operation(authority, "owned-commit-key")
        store.snapshot = SimpleNamespace(
            pending=_pending(
                operation.request_id,
                operation.identity.execution_id,
                operation.correlation_id,
            ),
            terminal=None,
            lifecycle=LifecyclePhase.COMMIT_STARTED,
        )
        controller = RemotePatchController(
            replace(configuration, binder=binder)
        )
        response = await controller.begin(
            _request_with_retransmission_key(authority, "owned-commit-key"),
            OperationType.EDIT,
            {
                "path": "note.txt",
                "edits": [{"old_text": "before", "new_text": "after"}],
            },
        )
        assert response.state == "pending"
        assert binder.calls == 0
        assert controller._tasks == {}

    run(scenario())


def test_same_controller_recovers_only_a_durably_reaped_worker() -> None:
    """Use the same durable recovery decision for old and new controllers."""

    class _ReapedHost:
        """Record recovery dispatches and retain current durable ownership."""

        def __init__(self, store: _ContinuationStore) -> None:
            """Bind one scripted durable state transition to this host."""
            self.calls = 0
            self._store = store

        async def invoke_remote_raw(self, *_: object) -> object:
            """Record one recovery and make durable worker ownership live."""
            self.calls += 1
            self._store.snapshot = SimpleNamespace(
                pending=None,
                terminal=None,
                lifecycle=LifecyclePhase.COMMIT_STARTED,
                worker_reaped=False,
            )
            return _result()

    async def recover_with(
        controller: RemotePatchController,
        store: _ContinuationStore,
        authority: RemotePatchAuthority,
        key: str,
    ) -> _ReapedHost:
        """Exercise one controller against an identical reaped snapshot."""
        operation = controller._operation(authority, key)
        store.snapshot = SimpleNamespace(
            pending=_pending(
                operation.request_id,
                operation.identity.execution_id,
                operation.correlation_id,
            ),
            terminal=None,
            lifecycle=LifecyclePhase.COMMIT_STARTED,
            worker_reaped=False,
        )
        request = _request_with_retransmission_key(authority, key)
        arguments = {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        }
        pending = await controller.begin(
            request, OperationType.EDIT, arguments
        )
        assert pending.state == "pending"
        assert controller._tasks == {}
        store.snapshot = SimpleNamespace(
            pending=_pending(
                operation.request_id,
                operation.identity.execution_id,
                operation.correlation_id,
            ),
            terminal=None,
            lifecycle=LifecyclePhase.COMMIT_STARTED,
            worker_reaped=True,
        )
        host = _ReapedHost(store)
        controller._host = cast(PatchSdkHost, host)

        async def stale_dispatch() -> None:
            """Complete one obsolete local coalescing task without truth."""

        stale = create_task(stale_dispatch())
        await stale
        controller._tasks[operation.request_id] = stale
        first = await controller.begin(request, OperationType.EDIT, arguments)
        task = controller._tasks[operation.request_id]
        assert task is not stale
        await task
        await sleep(0)
        replay = await controller.begin(request, OperationType.EDIT, arguments)
        assert first.state == replay.state == "pending"
        assert host.calls == 1
        assert controller._tasks == {}
        assert not hasattr(controller, "_dispatch_started")
        return host

    async def scenario() -> None:
        authority = _authority()
        first_store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        first = RemotePatchController(
            _continuation_configuration(authority, first_store)
        )
        fresh_store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        fresh = RemotePatchController(
            _continuation_configuration(authority, fresh_store)
        )
        first_host = await recover_with(first, first_store, authority, "same")
        fresh_host = await recover_with(fresh, fresh_store, authority, "fresh")
        assert first_host.calls == fresh_host.calls == 1
        await first.close()
        await fresh.close()

    run(scenario())


def test_public_client_exposes_only_closed_remote_edit_schema() -> None:
    """Use the public client without a caller-controlled authority field."""

    async def scenario() -> None:
        authority = _authority()
        configuration, _, service = _active_configuration(authority)
        app = FastAPI()
        controller = install_remote_patch_test_routes(app, configuration)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as http_client:
            client = RemotePatchTestClient(
                http_client,
                authority.correlation,
            )
            tools = await client.tools()
            operation = await client.edit(
                "note.txt",
                [RemotePatchEditPart(old_text="before", new_text="after")],
                "public-client-key",
            )
            applied = await client.apply(
                "\n".join(
                    (
                        "*** Begin Patch v1",
                        "*** Update File: note.txt",
                        "@@",
                        "-before",
                        "+after",
                        "*** End Patch",
                    )
                ),
                "public-client-apply-key",
            )
            inspected = await client.inspect(operation.operation_handle)
        await sleep(0)
        try:
            assert [tool.name for tool in tools.data] == [
                "patch.edit",
                "patch.apply",
            ]
            assert operation.state == "pending"
            assert applied.state == "pending"
            assert inspected.operation_handle
            assert [call[0] for call in service.remote_calls] == [
                OperationType.EDIT,
                OperationType.APPLY,
            ]
        finally:
            await controller.close()

    run(scenario())


def test_operation_routes_reject_replacement_payloads_before_reads() -> None:
    """Reject replacement plan, approval, and mutation command payloads."""

    async def scenario() -> None:
        authority = _authority()
        store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        controller = RemotePatchController(
            _continuation_configuration(authority, store)
        )
        handle = controller._seal_operation(
            controller._operation(authority, "replacement-payload-key")
        )
        app = FastAPI()
        install_remote_patch_test_routes(
            app,
            _continuation_configuration(authority, store),
            controller=controller,
        )
        headers = {"X-Avalan-Correlation": authority.correlation}
        payloads = (
            {"body": "caller-replacement"},
            {"plan": "caller-replacement"},
            {"fingerprint": "caller-replacement"},
            {"approval": "caller-replacement"},
            {"mutation": "caller-replacement"},
        )
        paths = (
            ("GET", f"/operations/{handle}"),
            ("POST", f"/operations/{handle}/await"),
            ("POST", f"/operations/{handle}/cancel"),
            ("GET", f"/operations/{handle}/events"),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            responses = [
                await client.request(
                    method,
                    "/__avalan_test__/patch/v1" + path,
                    headers=headers,
                    json=payload,
                )
                for method, path in paths
                for payload in payloads
            ]
        assert [response.status_code for response in responses] == [400] * 20
        assert all(
            response.json()
            == {
                "error": {
                    "code": "patch.operation_unavailable",
                    "message": "Patch operation unavailable.",
                }
            }
            for response in responses
        )
        assert store.accesses == []
        await controller.close()

    run(scenario())


def test_operation_routes_deny_authority_before_reads() -> None:
    """Reject opaque handles when either continuation authority differs."""

    async def scenario() -> None:
        authority = _authority()
        owner = RemotePatchController(_configuration(_Binder(), authority))
        handle = owner._seal_operation(
            owner._operation(authority, "continuation-authority-key")
        )
        alternatives = (
            replace(
                authority,
                correlation="correlation-phase13-other",
            ),
            replace(
                authority,
                approval_route=PolicyRouteId("route-approval-phase13-other"),
            ),
        )
        routes = (
            ("GET", ""),
            ("POST", "/await"),
            ("POST", "/cancel"),
            ("GET", "/events"),
        )
        for alternate in alternatives:
            store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
            configuration = _continuation_configuration(alternate, store)
            controller = RemotePatchController(configuration)
            app = FastAPI()
            install_remote_patch_test_routes(
                app, configuration, controller=controller
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=app,
                    client=("127.0.0.1", 1),
                ),
                base_url="http://testserver",
            ) as client:
                responses = [
                    await client.request(
                        method,
                        "/__avalan_test__/patch/v1/operations/"
                        + handle
                        + suffix,
                        headers={
                            "X-Avalan-Correlation": alternate.correlation,
                        },
                    )
                    for method, suffix in routes
                ]
            assert [response.status_code for response in responses] == [
                404
            ] * 4
            assert all(
                response.json()
                == {
                    "error": {
                        "code": "patch.operation_unavailable",
                        "message": "Patch operation unavailable.",
                    }
                }
                for response in responses
            )
            assert store.accesses == []
            await controller.close()

    run(scenario())


def test_disconnect_await_never_cancels_and_cancel_records_only_intent() -> (
    None
):
    """Keep disconnect separate from later cancellation intent."""

    async def scenario() -> None:
        authority = _authority()
        store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        controller = RemotePatchController(
            _continuation_configuration(authority, store)
        )
        operation = controller._operation(authority, "pending-key")
        store.snapshot = _ContinuationSnapshot(
            _pending(
                operation.request_id,
                operation.identity.execution_id,
                operation.correlation_id,
            ),
            None,
        )
        handle = controller._seal_operation(operation)
        request = _request(authority)
        waiter = create_task(controller.await_result(request, handle))
        await sleep(0.01)
        waiter.cancel()
        with pytest.raises(CancelledError):
            await waiter
        assert store.cancellations == 0
        assert store.await_calls == 0
        assert {
            task
            for task in all_tasks()
            if task is not current_task() and not task.done()
        } == set()
        pending = await controller.cancel_intent(request, handle)
        assert pending.state == "pending"
        assert store.cancellations == 1
        assert store.snapshot.pending is not None
        assert store.snapshot.pending.cancellation_requested
        store.await_gate.set()

    run(scenario())


def test_terminal_http_and_sse_omit_exact_truth_without_plan_proof() -> None:
    """Keep empty, insufficient, and exact scope disclosures equally coarse."""

    async def scenario() -> None:
        exact_fields = {
            "status",
            "mutation_state",
            "artifact_state",
            "postcondition",
            "retryability",
            "error_code",
            "result",
        }
        disclosures = (
            frozenset(),
            frozenset((PolicyDisclosure.MODEL_METADATA,)),
            frozenset((PolicyDisclosure.SERVER_EXACT_TRUTH,)),
        )
        for value in disclosures:
            authority = replace(_authority(), disclosures=value)
            store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
            configuration = _continuation_configuration(authority, store)
            controller = RemotePatchController(configuration)
            operation = controller._operation(authority, "disclosure-key")
            terminal, outbox = _terminal(
                operation.request_id,
                operation.correlation_id,
            )
            store.snapshot = _ContinuationSnapshot(None, terminal)
            store.records = (outbox,)
            app = FastAPI()
            install_remote_patch_test_routes(
                app, configuration, controller=controller
            )
            handle = controller._seal_operation(operation)
            headers = {"X-Avalan-Correlation": authority.correlation}
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=app,
                    client=("127.0.0.1", 1),
                ),
                base_url="http://testserver",
            ) as client:
                terminal_response = await client.get(
                    "/__avalan_test__/patch/v1/operations/" + handle,
                    headers=headers,
                )
                events = await client.get(
                    "/__avalan_test__/patch/v1/operations/"
                    + handle
                    + "/events",
                    headers=headers,
                )
            assert terminal_response.status_code == 200
            assert not exact_fields & set(terminal_response.json())
            assert all(field not in events.text for field in exact_fields)
            await controller.close()

        authority = _authority()
        store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        other = replace(
            authority, principal=PatchPrincipalId("principal-other")
        )
        configuration = replace(
            _continuation_configuration(authority, store),
            authority_resolver=_Resolver(other),
        )
        controller = RemotePatchController(configuration)
        operation = controller._operation(authority, "cross-disclosure-key")
        terminal, outbox = _terminal(
            operation.request_id,
            operation.correlation_id,
        )
        store.snapshot = _ContinuationSnapshot(None, terminal)
        store.records = (outbox,)
        app = FastAPI()
        install_remote_patch_test_routes(
            app,
            configuration,
            controller=controller,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            denied = await client.get(
                "/__avalan_test__/patch/v1/operations/"
                + controller._seal_operation(operation),
                headers={"X-Avalan-Correlation": other.correlation},
            )
        assert denied.status_code == 404
        assert denied.json() == {
            "error": {
                "code": "patch.operation_unavailable",
                "message": "Patch operation unavailable.",
            }
        }

    run(scenario())


def test_remote_privacy_canaries_do_not_escape_closed_transport(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep rejected canaries out of remote observability surfaces."""

    async def scenario() -> None:
        authority = _authority()
        configuration, binder, _ = _active_configuration(authority)
        app = FastAPI()
        controller = install_remote_patch_test_routes(app, configuration)
        canaries = (
            "tenant-canary-remote-13",
            "approval-canary-remote-13",
            "trace-canary-remote-13",
            "crash-canary-remote-13",
        )
        observations: list[str] = []
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=app,
                    client=("127.0.0.1", 1),
                ),
                base_url="http://testserver",
            ) as http_client:
                client = RemotePatchTestClient(
                    http_client, authority.correlation
                )
                for canary in canaries:
                    response = await http_client.post(
                        "/__avalan_test__/patch/v1/edit",
                        headers={
                            "Content-Type": "application/json",
                            "X-Avalan-Correlation": authority.correlation,
                            "X-Avalan-Retransmission-Key": "privacy-key-13",
                        },
                        content=dumps(
                            {
                                "path": "note.txt",
                                "edits": [
                                    {
                                        "old_text": "before",
                                        "new_text": "after",
                                    }
                                ],
                                "tenant": canary,
                            },
                            separators=(",", ":"),
                        ).encode("utf-8"),
                    )
                    assert response.status_code == 400
                    observations.extend(
                        (
                            response.text,
                            dumps(dict(response.headers), sort_keys=True),
                        )
                    )
                    with pytest.raises(RemotePatchServerError) as error:
                        await client.inspect("opaque-" + canary)
                    observations.append(str(error.value))
                event_response = await http_client.get(
                    "/__avalan_test__/patch/v1/operations/opaque-"
                    + canaries[-1]
                    + "/events",
                    headers={"X-Avalan-Correlation": authority.correlation},
                )
                assert event_response.status_code == 404
                observations.extend(
                    (
                        event_response.text,
                        dumps(dict(event_response.headers), sort_keys=True),
                    )
                )
            observations.append(caplog.text)
            observations.extend(
                str(getattr(app.state, name, ""))
                for name in ("metrics", "traces", "crash_data")
            )
            assert all(
                canary not in observation
                for canary in canaries
                for observation in observations
            )
            assert binder.calls == 0
        finally:
            await controller.close()

    run(scenario())


def test_sse_replays_monotonic_terminal_record_without_pending_terminal() -> (
    None
):
    """Deliver one durable terminal event with a stable cursor."""

    async def scenario() -> None:
        authority = _authority()
        store = _ContinuationStore(_ContinuationSnapshot(None, None), ())
        controller = RemotePatchController(
            _continuation_configuration(authority, store)
        )
        operation = controller._operation(authority, "terminal-event-key")
        terminal, outbox = _terminal(
            operation.request_id,
            operation.correlation_id,
        )
        store.snapshot = _ContinuationSnapshot(None, terminal)
        store.records = (outbox,)
        app = FastAPI()
        install_remote_patch_test_routes(
            app,
            _continuation_configuration(authority, store),
            controller=controller,
        )
        handle = controller._seal_operation(operation)
        headers = {"X-Avalan-Correlation": authority.correlation}
        path = f"/__avalan_test__/patch/v1/operations/{handle}/events"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as client:
            first = await client.get(
                path, headers=headers, params={"after": 0}
            )
            replay = await client.get(
                path, headers=headers, params={"after": 0}
            )
            consumed = await client.get(
                path, headers=headers, params={"after": 1}
            )
        assert first.status_code == 200
        assert first.text == replay.text
        assert "id: 1" in first.text
        assert outbox.event_id.value in first.text
        assert consumed.text == ""
        store.snapshot = _ContinuationSnapshot(
            _pending(
                operation.request_id,
                operation.identity.execution_id,
                operation.correlation_id,
            ),
            terminal,
        )
        response = await controller.events(_request(authority), handle, 0)
        assert [chunk async for chunk in response.body_iterator] == []

    run(scenario())


def test_public_client_uses_real_loopback_tcp_server_process() -> None:
    """Stream, reconnect, and settle one attested TCP child invocation."""

    async def scenario() -> None:
        port = _loopback_port()
        secret = token_urlsafe(32)
        process = Popen(
            [
                executable,
                "-m",
                "uvicorn",
                "patch_remote_tcp_fixture:app",
                "--app-dir",
                "tests/server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            stdout=DEVNULL,
            stderr=DEVNULL,
            env={
                **{
                    key: value
                    for key, value in environ.items()
                    if not key.startswith("COV_CORE")
                    and key != "COVERAGE_PROCESS_START"
                },
                "AVALAN_PATCH_TCP_TEST_SECRET": secret,
            },
        )
        base_url = f"http://127.0.0.1:{port}"
        headers = {
            "X-Avalan-Correlation": _authority().correlation,
            "X-Avalan-Test-Attestation": secret,
        }
        try:
            async with httpx.AsyncClient(base_url=base_url) as probe:
                await _await_loopback_ready(
                    lambda timeout: probe.get(
                        _LOOPBACK_READY_PATH,
                        headers=headers,
                        timeout=timeout,
                    ),
                    {"ready": True, "invocations": 0},
                    "loopback remote patch server unavailable",
                )
                denied = await probe.get(
                    _LOOPBACK_READY_PATH,
                    headers={"X-Avalan-Test-Attestation": "invalid-token"},
                )
                assert denied.status_code == 404
            async with httpx.AsyncClient(base_url=base_url) as http_client:
                client = RemotePatchTestClient(
                    http_client,
                    _authority().correlation,
                    attestation_secret=secret,
                )
                operation = await client.edit(
                    "note.txt",
                    [
                        RemotePatchEditPart(
                            old_text="before",
                            new_text="after",
                        )
                    ],
                    "tcp-retransmission-key",
                )
            async with httpx.AsyncClient(base_url=base_url) as http_client:
                client = RemotePatchTestClient(
                    http_client,
                    _authority().correlation,
                    attestation_secret=secret,
                )
                resumed = await client.inspect(operation.operation_handle)
                terminal = await client.await_result(
                    operation.operation_handle
                )
                async with client.events("live") as stream:
                    pending = await anext(stream)
                    duplicate = create_task(anext(stream))
                    await sleep(0.05)
                    assert not duplicate.done()
                    duplicate.cancel()
                    with pytest.raises(CancelledError):
                        await duplicate
            async with httpx.AsyncClient(base_url=base_url) as probe:
                released = await probe.post(
                    "/__avalan_test__/patch/v1/operations/live/release",
                    headers=headers,
                )
                assert released.json() == {"released": True}
            async with httpx.AsyncClient(base_url=base_url) as http_client:
                client = RemotePatchTestClient(
                    http_client,
                    _authority().correlation,
                    attestation_secret=secret,
                )
                async with client.events("live") as replay:
                    replayed = await anext(replay)
                    replay_terminal = await anext(replay)
                async with client.events("live", after=1) as stream:
                    resumed_event = await anext(stream)
                    with pytest.raises(StopAsyncIteration):
                        await anext(stream)
            async with httpx.AsyncClient(base_url=base_url) as probe:
                observed = await probe.get(
                    "/__avalan_test__/patch/v1/ready",
                    headers=headers,
                )
            assert operation.state == "pending"
            assert resumed.state == "completed"
            assert terminal.state == "completed"
            assert pending.cursor == 1
            assert pending.event_id == "event_live_pending"
            assert replayed == pending
            assert replay_terminal.cursor == 2
            assert replay_terminal.event_id == "event_live_terminal"
            assert resumed_event.cursor == 2
            assert resumed_event.event_id == "event_live_terminal"
            assert observed.json() == {"ready": True, "invocations": 1}
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    run(scenario())
