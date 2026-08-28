"""Exercise the remote test server against the selected sandbox runtime."""

from asyncio import run, sleep, wait_for
from collections.abc import Awaitable, Callable, Mapping
from os import environ
from pathlib import Path
from runpy import run_path
from secrets import token_urlsafe
from socket import AF_INET, SOCK_STREAM, socket
from subprocess import DEVNULL, Popen, TimeoutExpired
from sys import executable
from sys import path as sys_path
from time import monotonic

import httpx
import pytest
from fastapi import FastAPI, Request

from avalan.patch.coordinator import RetransmissionKey
from avalan.patch.domain import (
    Capability,
    DurationTicks,
    OperationType,
    PatchExecutionId,
    PatchObserverCorrelationId,
    PatchRequestId,
)
from avalan.patch.durable_approval import (
    HmacDurableApprovalAuthority,
    PhaseFiveDurableApprovalIssuer,
)
from avalan.patch.durable_store import (
    DurableRequestIdentity,
    InMemoryDurablePatchBackend,
)
from avalan.patch.planner import (
    BoundedPlannerWorker,
    PlannerFacade,
    PlannerLimits,
)
from avalan.patch.policy import (
    ApprovalService,
    PatchTenantId,
    RuntimeGrantStore,
)
from avalan.patch.sandbox_commit import (
    SandboxPatchRuntimeBinder,
    SandboxPatchRuntimeSettings,
    SandboxPatchSdkService,
    SandboxPatchServiceConfiguration,
)
from avalan.patch.target import TargetErrorCode, TargetInspectionError
from avalan.patch.toolset import (
    PatchApprovalBinding,
    PatchCoordinatorBinding,
    PatchPersistenceBinding,
    RemotePatchRuntimeWitness,
)
from avalan.server.patch import (
    RemotePatchAuthority,
    RemotePatchAuthorityResolver,
    RemotePatchController,
    RemotePatchEditPart,
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


def test_sandbox_loopback_readiness_timeout_is_hard_bounded() -> None:
    """Bound failed sandbox readiness polling by its monotonic deadline."""

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
                {
                    "ready": True,
                    "commit_claims": 0,
                    "worker_bindings": 0,
                    "settlements": 0,
                },
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


class _Resolver(RemotePatchAuthorityResolver):
    """Return one exact test principal resolved by the local test server."""

    def __init__(self, authority: RemotePatchAuthority) -> None:
        """Bind the complete trusted remote scope to this resolver."""
        self._authority = authority

    async def __call__(self, _: Request) -> RemotePatchAuthority | None:
        """Return the one authenticated test principal scope."""
        return self._authority


class _ForgedDurableRequestIdentity(DurableRequestIdentity):
    """Represent a type-compatible but noncanonical durable identity."""


def _loopback_port() -> int:
    """Reserve and release one ephemeral IPv4 loopback test port."""
    with socket(AF_INET, SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_remote_server_runs_selected_sandbox_edit_to_terminal(
    tmp_path: Path,
) -> None:
    """Run one closed remote edit through the real sandbox test profile."""

    async def scenario() -> None:
        sys_path.insert(0, "tests/patch")
        try:
            phase_ten = run_path("tests/patch/phase_10_contract_test.py")
        finally:
            sys_path.remove("tests/patch")
        settings_factory = phase_ten["_settings"]
        subject_factory = phase_ten["_runtime_subject"]
        policy_factory = phase_ten["_sandbox_corpus_policy"]
        clock_type = phase_ten["_RuntimeClock"]
        broker_type = phase_ten["_RuntimeBroker"]
        blocking_store_type = phase_ten["_BlockingFenceStore"]
        native_probe = phase_ten["_native_probe"]
        assert callable(settings_factory)
        assert callable(subject_factory)
        assert callable(policy_factory)
        assert callable(clock_type)
        assert callable(broker_type)
        assert callable(blocking_store_type)
        assert callable(native_probe)
        if not await native_probe():
            pytest.skip("selected native sandbox backend is unavailable")
        root = tmp_path / "sandbox-view"
        namespace = tmp_path / "sandbox-private"
        root.mkdir()
        namespace.mkdir()
        note = root / "note.txt"
        note.write_text("before\n", encoding="utf-8")
        settings = settings_factory(root, namespace)
        assert isinstance(settings, SandboxPatchRuntimeSettings)
        subject = subject_factory()
        policy = policy_factory()
        authority = HmacDurableApprovalAuthority.random()
        store = blocking_store_type(
            InMemoryDurablePatchBackend(approval_verifier=authority)
        )
        clock = clock_type()
        approvals = ApprovalService(broker_type(), clock, RuntimeGrantStore())
        configuration = SandboxPatchServiceConfiguration(
            subject,
            PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
            approvals,
            PhaseFiveDurableApprovalIssuer(approvals, authority),
            clock,
            DurationTicks(10),
            DurationTicks(10),
        )
        binder = SandboxPatchRuntimeBinder.from_settings(
            settings,
            configuration,
            policy,
            PatchApprovalBinding(True),
            PatchCoordinatorBinding(True, store),
            PatchPersistenceBinding(True, store),
        )
        remote_authority = RemotePatchAuthority(
            tenant=subject.tenant,
            principal=subject.principal,
            run=subject.run,
            session=subject.session,
            task=subject.task,
            agent=subject.agent,
            execution_scope=settings.context.identity.domain_id.value,
            route=policy.approval.route,
            context=settings.context.identity.context_id,
            workspace=settings.context.identity.workspace_id,
            policy_revision=policy.revision,
            disclosures=frozenset(),
            approval_route=policy.approval.route,
            correlation="sandbox-remote-correlation",
            capabilities=frozenset(
                (
                    Capability.READ_FOR_MUTATION,
                    Capability.OBSERVE_MUTATION_PRECONDITIONS,
                )
            ),
        )
        server = RemotePatchTestServerConfiguration(
            profile=RemotePatchTestServerProfile(
                enabled=True,
                authenticated=True,
                loopback_only=True,
            ),
            authority_resolver=_Resolver(remote_authority),
            expected_authority=remote_authority,
            binder=binder,
            store=store,
            handle_key=b"s" * 32,
            runtime_witness=RemotePatchRuntimeWitness(
                tenant=remote_authority.tenant,
                principal=remote_authority.principal,
                run=remote_authority.run,
                session=remote_authority.session,
                task=remote_authority.task,
                agent=remote_authority.agent,
                execution_scope=remote_authority.execution_scope,
                route=remote_authority.route,
                context=remote_authority.context,
                workspace=remote_authority.workspace,
                policy_revision=remote_authority.policy_revision,
                disclosures=remote_authority.disclosures,
                approval_route=remote_authority.approval_route,
                capabilities=remote_authority.capabilities,
            ),
        )
        app = FastAPI()
        controller = install_remote_patch_test_routes(app, server)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=app,
                    client=("127.0.0.1", 1),
                ),
                base_url="http://testserver",
            ) as http_client:
                client = RemotePatchTestClient(
                    http_client,
                    remote_authority.correlation,
                )
                operation = await client.edit(
                    "note.txt",
                    [
                        RemotePatchEditPart(
                            old_text="before\n",
                            new_text="after\n",
                        )
                    ],
                    "sandbox-remote-retransmission",
                )
                assert operation.state == "pending"
                await store.effect_reached.wait()
                host = controller._host
                assert host is not None
                service = host._service
                assert isinstance(service, SandboxPatchSdkService)
                raw_arguments = (
                    b'{"path":"note.txt","edits":['
                    b'{"old_text":"before\\n","new_text":"after\\n"}]}'
                )
                with pytest.raises(TargetInspectionError) as stale_identity:
                    await service.invoke_remote(
                        OperationType.EDIT,
                        raw_arguments,
                        host._capability,
                        PatchRequestId.new(),
                        PatchObserverCorrelationId.new(),
                        DurableRequestIdentity(
                            PatchTenantId("other-tenant"),
                            remote_authority.principal,
                            PatchExecutionId.new(),
                            remote_authority.route,
                            RetransmissionKey("wrong-remote-identity"),
                        ),
                    )
                assert (
                    stale_identity.value.code is TargetErrorCode.WITNESS_STALE
                )
                with pytest.raises(
                    TargetInspectionError
                ) as malformed_identity:
                    await service.invoke_remote(
                        OperationType.EDIT,
                        raw_arguments,
                        host._capability,
                        PatchRequestId.new(),
                        PatchObserverCorrelationId.new(),
                        _ForgedDurableRequestIdentity(
                            remote_authority.tenant,
                            remote_authority.principal,
                            PatchExecutionId.new(),
                            remote_authority.route,
                            RetransmissionKey("forged-remote-identity"),
                        ),
                    )
                assert (
                    malformed_identity.value.code
                    is TargetErrorCode.WITNESS_STALE
                )
                attached = await client.edit(
                    "note.txt",
                    [
                        RemotePatchEditPart(
                            old_text="before\n",
                            new_text="after\n",
                        )
                    ],
                    "sandbox-remote-retransmission",
                )
                assert attached.state == "pending"
                assert store.checks >= 2
                store.release_effect.set()
                terminal = await client.await_result(
                    operation.operation_handle
                )
            assert terminal.state == "completed"
            assert note.read_text(encoding="utf-8") == "after\n"
        finally:
            store.release_effect.set()
            await controller.close()

    run(scenario())


def test_remote_tcp_runs_selected_sandbox_apply_to_terminal(
    tmp_path: Path,
) -> None:
    """Apply one patch through a real child server and selected target."""

    async def scenario() -> None:
        assert RemotePatchController.__name__ == "RemotePatchController"
        sys_path.insert(0, "tests/patch")
        try:
            phase_ten = run_path("tests/patch/phase_10_contract_test.py")
        finally:
            sys_path.remove("tests/patch")
        native_probe = phase_ten["_native_probe"]
        assert callable(native_probe)
        if not await native_probe():
            pytest.skip("selected native sandbox backend is unavailable")
        root = tmp_path / "sandbox-tcp-view"
        namespace = tmp_path / "sandbox-tcp-private"
        root.mkdir()
        namespace.mkdir()
        note = root / "note.txt"
        note.write_text("before\n", encoding="utf-8")
        port = _loopback_port()
        secret = token_urlsafe(32)
        process = Popen(
            [
                executable,
                "-m",
                "uvicorn",
                "patch_remote_sandbox_tcp_fixture:app",
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
                "AVALAN_PATCH_SANDBOX_TCP_ROOT": str(root),
                "AVALAN_PATCH_SANDBOX_TCP_NAMESPACE": str(namespace),
                "AVALAN_PATCH_SANDBOX_TCP_SECRET": secret,
            },
        )
        base_url = f"http://127.0.0.1:{port}"
        headers = {
            "X-Avalan-Correlation": "sandbox-tcp-apply-correlation",
            "X-Avalan-Test-Attestation": secret,
        }
        document = "\n".join(
            (
                "*** Begin Patch v1",
                "*** Update File: note.txt",
                "@@",
                "-before",
                "+after",
                "*** End Patch",
            )
        )
        try:
            async with httpx.AsyncClient(base_url=base_url) as probe:
                await _await_loopback_ready(
                    lambda timeout: probe.get(
                        _LOOPBACK_READY_PATH,
                        headers=headers,
                        timeout=timeout,
                    ),
                    {
                        "ready": True,
                        "commit_claims": 0,
                        "worker_bindings": 0,
                        "settlements": 0,
                    },
                    "loopback selected sandbox patch server unavailable",
                )
            async with httpx.AsyncClient(base_url=base_url) as http_client:
                client = RemotePatchTestClient(
                    http_client,
                    "sandbox-tcp-apply-correlation",
                    attestation_secret=secret,
                )
                tools = await client.tools()
                operation = await client.apply(
                    document,
                    "tcp-sandbox-apply-key",
                )
                attached = await client.apply(
                    document,
                    "tcp-sandbox-apply-key",
                )
                terminal = await client.await_result(
                    operation.operation_handle
                )
                later = await client.inspect(operation.operation_handle)
                async with client.events(operation.operation_handle) as events:
                    terminal_event = await anext(events)
                    with pytest.raises(StopAsyncIteration):
                        await anext(events)
                async with client.events(
                    operation.operation_handle,
                    after=terminal_event.cursor,
                ) as resumed:
                    with pytest.raises(StopAsyncIteration):
                        await anext(resumed)
            async with httpx.AsyncClient(base_url=base_url) as probe:
                observed = await probe.get(
                    _LOOPBACK_READY_PATH,
                    headers=headers,
                )
            return (
                [tool.name for tool in tools.data],
                (operation.state, attached.state),
                (terminal.state, later.state),
                (terminal.event_cursor, terminal_event.cursor),
                note.read_text(encoding="utf-8"),
                observed.json(),
            )
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    result = run(scenario())
    assert result[0] == ["patch.edit", "patch.apply"]
    assert result[1] == ("pending", "pending")
    assert result[2] == ("completed", "completed")
    assert result[3][0] == result[3][1]
    assert result[4] == "after\n"
    assert result[5] == {
        "ready": True,
        "commit_claims": 1,
        "worker_bindings": 1,
        "settlements": 1,
    }
