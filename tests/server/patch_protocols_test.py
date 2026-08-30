"""Exercise the explicit Phase 14 loopback MCP and A2A adapters."""

from asyncio import run
from dataclasses import dataclass, replace
from json import dumps, loads

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.types import Message

from avalan.patch.coordinator import RetransmissionKey
from avalan.patch.domain import (
    AlgorithmDigest,
    ArtifactState,
    CommitTruth,
    DurationTicks,
    LifecyclePhase,
    LineageState,
    MutationState,
    OperationType,
    PatchContextId,
    PatchEventId,
    PatchExecutionId,
    PatchObserverCorrelationId,
    PatchPendingOperationId,
    PatchPlanId,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PatchWorkspaceId,
    PostconditionState,
    RequestedEffectOccurrence,
    SequenceNumber,
    WorkspaceChange,
)
from avalan.patch.durable_store import (
    DurableOutboxRecord,
    DurablePendingRecord,
    DurableReservation,
    DurableTerminalRecord,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.policy import (
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PolicyRouteId,
)
from avalan.patch.protocols import (
    PatchOrchestrationChecklist,
    PatchProtocolChecklist,
    PatchProtocolContinuation,
    PatchProtocolContinuationKind,
    PatchProtocolIdentity,
    PatchProtocolProfile,
    PatchProtocolReservation,
    PatchProtocolSurface,
    PatchProviderCodecChecklist,
)
from avalan.server import patch_protocols as patch_protocols_module
from avalan.server.patch_protocols import (
    PatchProtocolAdapter,
    PatchProtocolAdapterConfiguration,
    PatchProtocolAdapterError,
    PatchProtocolIdentityResolver,
    install_patch_protocol_test_routes,
)
from avalan.types import JsonObject

_PREFIX = "/__avalan_test__/patch-protocol/v1"


def _protocol_checklist() -> PatchProtocolChecklist:
    """Return a complete protocol checklist for one active test profile."""
    return PatchProtocolChecklist(
        canonical_input=True,
        trusted_authority=True,
        plan_approval=True,
        detached_resume=True,
        retransmission_reservation=True,
        owner_fence_before_effect=True,
        structured_terminal_result=True,
        branch_suspension=True,
        privacy_safe_events=True,
    )


def _orchestration_checklist() -> PatchOrchestrationChecklist:
    """Return a complete orchestration checklist for one active profile."""
    return PatchOrchestrationChecklist(
        shared_coordinator=True,
        originating_identity=True,
        approval_or_denial=True,
        retry_blocked_after_commit=True,
        durable_resume=True,
        dependent_suspension=True,
        coordinated_parallelism=True,
        committed_state_visibility=True,
    )


def _codec_checklist() -> PatchProviderCodecChecklist:
    """Return a complete codec checklist unused by MCP and A2A dispatch."""
    return PatchProviderCodecChecklist(
        advertised=True,
        complete_buffering=True,
        grammar_and_limits=True,
        stable_correlation=True,
        replay_fencing=True,
        result_injection=True,
        approval_suspension=True,
        idempotency_and_resume=True,
        authority_and_disclosure=True,
    )


def _profile(surface: PatchProtocolSurface) -> PatchProtocolProfile:
    """Return an exact complete loopback-only profile for one protocol."""
    return PatchProtocolProfile(
        surface=surface,
        enabled=True,
        authenticated=True,
        loopback_only=True,
        protocol=_protocol_checklist(),
        orchestration=_orchestration_checklist(),
        provider_codec=_codec_checklist(),
    )


def _identity(suffix: str) -> PatchProtocolIdentity:
    """Return one full authenticated execution identity."""
    return PatchProtocolIdentity(
        tenant=PatchTenantId(f"tenant-protocol-{suffix}"),
        principal=PatchPrincipalId(f"principal-protocol-{suffix}"),
        execution=PatchExecutionId.new(),
        run=PatchRunId(f"run-protocol-{suffix}"),
        session=PatchSessionId(f"session-protocol-{suffix}"),
        task=PatchTaskId(f"task-protocol-{suffix}"),
        agent=PatchAgentId(f"agent-protocol-{suffix}"),
        route=PolicyRouteId(f"route-protocol-{suffix}"),
        context=PatchContextId.new(),
        workspace=PatchWorkspaceId.new(),
    )


def _result(reservation: PatchProtocolReservation) -> PatchResult:
    """Return one durable terminal result without disclosing patch content."""
    return PatchResult(
        schema_version=1,
        request_id=reservation.request_id,
        plan_id=PatchPlanId.new(),
        lifecycle=LifecyclePhase.REQUEST_COMPLETED,
        status=PatchStatus.COMMITTED,
        truth=CommitTruth(
            MutationState.COMMITTED,
            LineageState.COMMITTED,
            RequestedEffectOccurrence.TRUE,
            ArtifactState.CLEANED,
            WorkspaceChange.CHANGED,
            True,
            PostconditionState.ESTABLISHED,
        ),
        diagnostic=None,
    )


@dataclass
class _Executor:
    """Persist an ordered detached review and one terminal test effect."""

    backend: InMemoryDurablePatchBackend
    plan_calls: int = 0
    approval_calls: int = 0
    await_calls: int = 0
    fail_plan: bool = False

    async def plan(
        self,
        reservation: PatchProtocolReservation,
        operation: OperationType,
        raw_arguments: bytes,
    ) -> None:
        """Record the reviewable plan after reservation without an effect."""
        assert operation in {OperationType.EDIT, OperationType.APPLY}
        assert raw_arguments
        self.plan_calls += 1
        record = self.backend.records[reservation.durable.identity]
        assert record.lifecycle is LifecyclePhase.RECEIVED
        if self.fail_plan:
            raise RuntimeError("injected plan fault")
        record.lifecycle = LifecyclePhase.PLANNED

    async def approve(self, reservation: PatchProtocolReservation) -> None:
        """Persist approval transition before pending settlement."""
        self.approval_calls += 1
        record = self.backend.records[reservation.durable.identity]
        assert record.lifecycle is LifecyclePhase.PLANNED
        record.lifecycle = LifecyclePhase.SETTLEMENT_PENDING
        record.pending = DurablePendingRecord(
            reservation.request_id,
            reservation.identity.execution,
            PatchPendingOperationId.new(),
            reservation.correlation,
            SequenceNumber(1),
            SequenceNumber(1),
            False,
            DurationTicks(1),
        )

    async def await_result(
        self, reservation: PatchProtocolReservation
    ) -> None:
        """Settle the pending durable record without a second effect."""
        self.await_calls += 1
        record = self.backend.records[reservation.durable.identity]
        pending = record.pending
        assert record.lifecycle is LifecyclePhase.SETTLEMENT_PENDING
        assert pending is not None
        record.pending = None
        record.lifecycle = LifecyclePhase.REQUEST_COMPLETED
        record.terminal = DurableTerminalRecord(
            _result(reservation),
            DurableOutboxRecord(
                PatchEventId.new(),
                reservation.request_id,
                SequenceNumber(2),
                LifecyclePhase.REQUEST_COMPLETED,
                reservation.correlation,
            ),
            pending.pending_operation_id,
        )


@dataclass(frozen=True)
class _Resolver(PatchProtocolIdentityResolver):
    """Resolve a full identity only from a fixed authenticated test header."""

    owner: PatchProtocolIdentity
    other: PatchProtocolIdentity

    async def __call__(self, request: Request) -> PatchProtocolIdentity | None:
        """Return the named authenticated server identity or no principal."""
        match request.headers.get("X-Patch-Protocol-Principal"):
            case "owner":
                return self.owner
            case "other":
                return self.other
            case _:
                return None


class _OtherConfiguration(PatchProtocolAdapterConfiguration):
    """Represent a rejected configuration subtype at the exact boundary."""


class _OtherApp(FastAPI):
    """Represent a rejected application subtype at the exact boundary."""


class _OtherReservation(DurableReservation):
    """Represent a rejected reservation subtype at the exact boundary."""


def _configuration(
    backend: InMemoryDurablePatchBackend,
    executor: _Executor,
    owner: PatchProtocolIdentity,
    other: PatchProtocolIdentity,
) -> PatchProtocolAdapterConfiguration:
    """Return one complete explicit adapter configuration."""
    return PatchProtocolAdapterConfiguration(
        mcp_profile=_profile(PatchProtocolSurface.MCP),
        a2a_profile=_profile(PatchProtocolSurface.A2A),
        store=InMemoryDurablePatchStore(backend),
        identity_resolver=_Resolver(owner, other),
        executor=executor,
        handle_key=b"p" * 32,
    )


def _mcp_call(
    name: str,
    arguments: JsonObject,
    retransmission_key: str,
) -> JsonObject:
    """Return one canonical MCP tool-call request."""
    return {
        "id": "mcp-call-001",
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "arguments": arguments,
            "name": name,
            "retransmission_key": retransmission_key,
        },
    }


def _a2a_call(
    identity: PatchProtocolIdentity,
    message: JsonObject,
) -> JsonObject:
    """Return one canonical A2A message-send request."""
    return {
        "id": "a2a-call-001",
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {"message": message, "task_id": identity.task.value},
    }


def test_mcp_loopback_lifecycle_is_detached_replay_safe_and_private() -> None:
    """Run PATCH-E2E-031 across discovery, approval, and later read."""

    async def scenario() -> tuple[int, int, int, JsonObject]:
        owner = _identity("owner")
        other = _identity("other")
        backend = InMemoryDurablePatchBackend()
        executor = _Executor(backend)
        app = FastAPI()
        install_patch_protocol_test_routes(
            app, _configuration(backend, executor, owner, other)
        )
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 1))
        headers = {"X-Patch-Protocol-Principal": "owner"}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            discovery = await client.post(
                _PREFIX + "/mcp",
                headers=headers,
                json={
                    "id": "mcp-list-001",
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                },
            )
            assert discovery.status_code == 200
            assert [
                item["name"] for item in discovery.json()["result"]["tools"]
            ] == ["patch.edit", "patch.apply"]
            first = await client.post(
                _PREFIX + "/mcp",
                headers=headers,
                json=_mcp_call(
                    "patch.edit",
                    {
                        "edits": [{"new_text": "after", "old_text": "before"}],
                        "path": "private-supersecret.txt",
                    },
                    "mcp-key-001",
                ),
            )
            replay = await client.post(
                _PREFIX + "/mcp",
                headers=headers,
                json=_mcp_call(
                    "patch.edit",
                    {
                        "edits": [{"new_text": "after", "old_text": "before"}],
                        "path": "private-supersecret.txt",
                    },
                    "mcp-key-001",
                ),
            )
            first_body = first.json()
            replay_body = replay.json()
            handle = first_body["result"]["structuredContent"][
                "operation_handle"
            ]
            assert (
                first_body["result"]["structuredContent"]["state"]
                == "approval_required"
            )
            assert (
                replay_body["result"]["structuredContent"]["state"]
                == "approval_required"
            )
            await client.aclose()
            resumed_client = httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            )
            status = await resumed_client.post(
                _PREFIX + f"/mcp/operations/{handle}/status", headers=headers
            )
            approval = await resumed_client.post(
                _PREFIX + f"/mcp/operations/{handle}/approval", headers=headers
            )
            pending_handle = approval.json()["result"]["structuredContent"][
                "operation_handle"
            ]
            pending = await resumed_client.post(
                _PREFIX + f"/mcp/operations/{pending_handle}/status",
                headers=headers,
            )
            terminal = await resumed_client.post(
                _PREFIX + f"/mcp/operations/{pending_handle}/await",
                headers=headers,
            )
            later = await resumed_client.post(
                _PREFIX + f"/mcp/operations/{pending_handle}/status",
                headers=headers,
            )
            conflict = await resumed_client.post(
                _PREFIX + "/mcp",
                headers=headers,
                json=_mcp_call(
                    "patch.edit",
                    {
                        "edits": [
                            {"new_text": "different", "old_text": "before"}
                        ],
                        "path": "private-supersecret.txt",
                    },
                    "mcp-key-001",
                ),
            )
            denial = await resumed_client.post(
                _PREFIX + f"/mcp/operations/{pending_handle}/status",
                headers={"X-Patch-Protocol-Principal": "other"},
            )
            await resumed_client.aclose()
        assert (
            status.json()["result"]["structuredContent"]["state"]
            == "approval_required"
        )
        assert (
            pending.json()["result"]["structuredContent"]["state"]
            == "settlement_pending"
        )
        assert (
            terminal.json()["result"]["structuredContent"]["state"]
            == "terminal"
        )
        assert later.json()["result"]["structuredContent"]["result"] == {
            "lifecycle": "request_completed",
            "status": "committed",
        }
        assert conflict.status_code == denial.status_code == 404
        assert "private-supersecret" not in dumps(
            [
                first.json(),
                replay.json(),
                terminal.json(),
                denial.json(),
            ]
        )
        return (
            executor.plan_calls,
            executor.approval_calls,
            executor.await_calls,
            later.json(),
        )

    plans, approvals, awaits, later = run(scenario())
    assert (plans, approvals, awaits) == (1, 1, 1)
    result = later["result"]
    assert isinstance(result, dict)
    structured = result["structuredContent"]
    assert isinstance(structured, dict)
    assert structured["state"] == "terminal"


def test_a2a_loopback_restart_preserves_typed_task_correlation() -> None:
    """Run PATCH-E2E-032 across input-required, pending, and restart."""

    async def scenario() -> tuple[int, int, int, JsonObject]:
        owner = _identity("owner")
        other = _identity("other")
        backend = InMemoryDurablePatchBackend()
        executor = _Executor(backend)
        configuration = _configuration(backend, executor, owner, other)
        app = FastAPI()
        install_patch_protocol_test_routes(app, configuration)
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 1))
        headers = {"X-Patch-Protocol-Principal": "owner"}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            initial = await client.post(
                _PREFIX + "/a2a",
                headers=headers,
                json=_a2a_call(
                    owner,
                    {
                        "arguments": {
                            "edits": [
                                {"new_text": "after", "old_text": "before"}
                            ],
                            "path": "a2a-private.txt",
                        },
                        "kind": "patch.call",
                        "name": "patch.edit",
                        "retransmission_key": "a2a-key-001",
                    },
                ),
            )
            handle = initial.json()["result"]["status"]["message"][
                "operation_handle"
            ]
            restarted = PatchProtocolAdapter(configuration)
            restarted_status = await restarted.a2a(
                _request_for_adapter(owner, handle)
            )
            approval = await client.post(
                _PREFIX + "/a2a",
                headers=headers,
                json=_a2a_call(
                    owner,
                    {"kind": "patch.approval", "operation_handle": handle},
                ),
            )
            pending_handle = approval.json()["result"]["status"]["message"][
                "operation_handle"
            ]
            terminal = await client.post(
                _PREFIX + "/a2a",
                headers=headers,
                json=_a2a_call(
                    owner,
                    {
                        "kind": "patch.resume",
                        "operation_handle": pending_handle,
                    },
                ),
            )
            wrong_task = await client.post(
                _PREFIX + "/a2a",
                headers=headers,
                json={
                    "id": "a2a-other-task",
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "params": {
                        "operation_handle": pending_handle,
                        "task_id": other.task.value,
                    },
                },
            )
            wrong_authority = await client.post(
                _PREFIX + "/a2a",
                headers={"X-Patch-Protocol-Principal": "other"},
                json={
                    "id": "a2a-other-authority",
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "params": {
                        "operation_handle": pending_handle,
                        "task_id": other.task.value,
                    },
                },
            )
            completed_output = await client.post(
                _PREFIX + "/a2a",
                headers=headers,
                json=_a2a_call(
                    owner,
                    {
                        "kind": "patch.approval",
                        "operation_handle": pending_handle,
                        "result": {"status": "committed"},
                    },
                ),
            )
        assert initial.json()["result"]["status"]["state"] == "input-required"
        assert restarted_status.status_code == 200
        assert bytes(restarted_status.body).find(b"input-required") >= 0
        assert approval.json()["result"]["status"]["state"] == "working"
        assert terminal.json()["result"]["status"]["state"] == "completed"
        assert (
            wrong_task.status_code
            == wrong_authority.status_code
            == completed_output.status_code
            == 404
        )
        assert "a2a-private" not in dumps(
            [
                initial.json(),
                terminal.json(),
                wrong_authority.json(),
            ]
        )
        return (
            executor.plan_calls,
            executor.approval_calls,
            executor.await_calls,
            terminal.json(),
        )

    plans, approvals, awaits, terminal = run(scenario())
    assert (plans, approvals, awaits) == (1, 1, 1)
    result = terminal["result"]
    assert isinstance(result, dict)
    status = result["status"]
    assert isinstance(status, dict)
    message = status["message"]
    assert isinstance(message, dict)
    assert message["result"] == {
        "lifecycle": "request_completed",
        "status": "committed",
    }


def _request_for_adapter(
    identity: PatchProtocolIdentity, handle: str
) -> Request:
    """Build one loopback A2A status request for restart testing."""
    app = FastAPI()
    scope = {
        "client": ("127.0.0.1", 1),
        "headers": [(b"x-patch-protocol-principal", b"owner")],
        "method": "POST",
        "path": _PREFIX + "/a2a",
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
        "type": "http",
        "app": app,
    }
    body = dumps(
        {
            "id": "a2a-status-001",
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {
                "operation_handle": handle,
                "task_id": identity.task.value,
            },
        }
    ).encode()

    async def receive() -> Message:
        """Return the one complete ASGI body frame for the adapter request."""
        return {"body": body, "more_body": False, "type": "http.request"}

    return Request(scope, receive)


def test_incapable_profiles_and_faulted_replays_stay_inert() -> None:
    """List no tools and never replan incomplete or faulted calls."""

    async def scenario() -> tuple[int, int]:
        owner = _identity("owner")
        other = _identity("other")
        backend = InMemoryDurablePatchBackend()
        executor = _Executor(backend, fail_plan=True)
        configuration = _configuration(backend, executor, owner, other)
        inactive = replace(
            configuration,
            mcp_profile=replace(configuration.mcp_profile, enabled=False),
        )
        app = FastAPI()
        install_patch_protocol_test_routes(app, inactive)
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 1))
        headers = {"X-Patch-Protocol-Principal": "owner"}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            discovery = await client.post(
                _PREFIX + "/mcp",
                headers=headers,
                json={
                    "id": "mcp-inert-list",
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                },
            )
            first = await client.post(
                _PREFIX + "/mcp",
                headers=headers,
                json=_mcp_call(
                    "patch.edit",
                    {
                        "edits": [{"new_text": "after", "old_text": "before"}],
                        "path": "fault.txt",
                    },
                    "fault-key-001",
                ),
            )
        assert discovery.json()["result"] == {"tools": []}
        assert first.status_code == 404
        assert executor.plan_calls == 0
        active_app = FastAPI()
        install_patch_protocol_test_routes(active_app, configuration)
        active_transport = httpx.ASGITransport(
            app=active_app, client=("127.0.0.1", 1)
        )
        async with httpx.AsyncClient(
            transport=active_transport, base_url="http://testserver"
        ) as client:
            first_fault = await client.post(
                _PREFIX + "/mcp",
                headers=headers,
                json=_mcp_call(
                    "patch.edit",
                    {
                        "edits": [{"new_text": "after", "old_text": "before"}],
                        "path": "fault.txt",
                    },
                    "fault-key-001",
                ),
            )
            replay_fault = await client.post(
                _PREFIX + "/mcp",
                headers=headers,
                json=_mcp_call(
                    "patch.edit",
                    {
                        "edits": [{"new_text": "after", "old_text": "before"}],
                        "path": "fault.txt",
                    },
                    "fault-key-001",
                ),
            )
        assert first_fault.status_code == 404
        assert replay_fault.status_code == 200
        return executor.plan_calls, executor.approval_calls

    assert run(scenario()) == (1, 0)


def test_adapter_rejects_invalid_configuration_and_handles() -> None:
    """Reject malformed installation and non-authoritative opaque handles."""
    owner = _identity("owner")
    other = _identity("other")
    backend = InMemoryDurablePatchBackend()
    executor = _Executor(backend)
    configuration = _configuration(backend, executor, owner, other)
    with pytest.raises(PatchProtocolAdapterError):
        PatchProtocolAdapter(replace(configuration, handle_key=b"short"))
    with pytest.raises(PatchProtocolAdapterError):
        PatchProtocolAdapter(
            _OtherConfiguration(
                mcp_profile=configuration.mcp_profile,
                a2a_profile=configuration.a2a_profile,
                store=configuration.store,
                identity_resolver=configuration.identity_resolver,
                executor=configuration.executor,
                handle_key=configuration.handle_key,
            )
        )
    with pytest.raises(PatchProtocolAdapterError):
        install_patch_protocol_test_routes(_OtherApp(), configuration)
    adapter = PatchProtocolAdapter(configuration)
    with pytest.raises(PatchProtocolAdapterError):
        adapter._open("bad-handle")


def test_adapter_closes_all_malformed_protocol_shapes_without_oracles() -> (
    None
):
    """Reject every malformed protocol shape with the same public denial."""

    async def scenario() -> None:
        owner = _identity("owner")
        other = _identity("other")
        backend = InMemoryDurablePatchBackend()
        executor = _Executor(backend)
        configuration = _configuration(backend, executor, owner, other)
        adapter = PatchProtocolAdapter(configuration)
        app = FastAPI()
        install_patch_protocol_test_routes(app, configuration)
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 1))
        headers = {"X-Patch-Protocol-Principal": "owner"}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            malformed = await client.post(
                _PREFIX + "/mcp", headers=headers, content=b"not-json"
            )
            unknown_mcp = await client.post(
                _PREFIX + "/mcp",
                headers=headers,
                json={
                    "id": True,
                    "jsonrpc": "2.0",
                    "method": "unavailable",
                    "params": {},
                },
            )
            unknown_a2a = await client.post(
                _PREFIX + "/a2a",
                headers=headers,
                json={
                    "id": "unknown-a2a",
                    "jsonrpc": "2.0",
                    "method": "unavailable",
                    "params": {},
                },
            )
            invalid_a2a = await client.post(
                _PREFIX + "/a2a",
                headers=headers,
                json={
                    "id": "invalid-a2a",
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {"kind": "patch.call"},
                        "task_id": owner.task.value,
                    },
                },
            )
            invalid_a2a_base = await client.post(
                _PREFIX + "/a2a",
                headers=headers,
                json={
                    "id": "invalid-a2a-base",
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": [],
                        "task_id": owner.task.value,
                    },
                },
            )
            invalid_jsonrpc = await client.post(
                _PREFIX + "/mcp",
                headers=headers,
                json={
                    "id": "invalid-jsonrpc",
                    "jsonrpc": "1.0",
                    "method": "tools/list",
                    "params": {},
                },
            )
            no_principal = await client.post(
                _PREFIX + "/mcp",
                json={
                    "id": "no-principal",
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                },
            )
            apply = await client.post(
                _PREFIX + "/mcp",
                headers=headers,
                json=_mcp_call(
                    "patch.apply",
                    {
                        "patch": "\n".join(
                            (
                                "*** Begin Patch v1",
                                "*** Update File: note.txt",
                                "@@",
                                "-before",
                                "+after",
                                "*** End Patch",
                            )
                        ),
                    },
                    "malformed-call-key",
                ),
            )
        assert malformed.status_code == unknown_mcp.status_code == 404
        assert unknown_a2a.status_code == invalid_a2a.status_code == 404
        assert (
            invalid_a2a_base.status_code == invalid_jsonrpc.status_code == 404
        )
        assert no_principal.status_code == 404
        assert apply.status_code == 200
        handle = apply.json()["result"]["structuredContent"][
            "operation_handle"
        ]
        response = await adapter._mcp_continuation_response(
            _request_for_adapter(owner, handle), handle, "invalid"
        )
        assert response.status_code == 404
        continuation = await adapter._load(
            PatchProtocolSurface.MCP, owner, handle
        )
        with pytest.raises(PatchProtocolAdapterError):
            await adapter._await(continuation)
        pending = await adapter._approve(continuation)
        with pytest.raises(PatchProtocolAdapterError):
            await adapter._approve(pending)
        invalid = object.__new__(PatchProtocolContinuation)
        object.__setattr__(
            invalid, "kind", PatchProtocolContinuationKind.TERMINAL
        )
        object.__setattr__(invalid, "reservation", continuation.reservation)
        object.__setattr__(invalid, "pending", None)
        object.__setattr__(invalid, "result", None)
        with pytest.raises(PatchProtocolAdapterError):
            adapter._continuation_payload(invalid)
        encrypted = adapter._fernet.encrypt(b"{}").decode()
        with pytest.raises(PatchProtocolAdapterError):
            adapter._open(encrypted)
        bad_handle = adapter._seal(continuation)
        payload = adapter._fernet.decrypt(bad_handle.encode())
        altered = adapter._fernet.encrypt(
            payload.replace(b'"replayed":false', b'"replayed":"bad"')
        ).decode()
        with pytest.raises(PatchProtocolAdapterError):
            adapter._open(altered)
        malformed_identity = loads(payload)
        malformed_identity["identity"] = []
        with pytest.raises(PatchProtocolAdapterError):
            adapter._open(
                adapter._fernet.encrypt(
                    dumps(malformed_identity).encode()
                ).decode()
            )
        with pytest.raises(PatchProtocolAdapterError):
            adapter._open("")
        with pytest.raises(PatchProtocolAdapterError):
            await adapter._load(PatchProtocolSurface.A2A, owner, handle)
        for message in (
            {"kind": "patch.approval", "operation_handle": 1},
            {"kind": "patch.resume", "unexpected": True},
            {"kind": "patch.resume", "operation_handle": 1},
            {"kind": "patch.terminal"},
        ):
            with pytest.raises(PatchProtocolAdapterError):
                await adapter._a2a_message(
                    owner,
                    {"message": message, "task_id": owner.task.value},
                )
        remote_scope = {
            "client": ("192.0.2.1", 1),
            "headers": [],
            "method": "POST",
            "path": _PREFIX + "/mcp",
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "type": "http",
        }

        async def receive() -> Message:
            """Return one valid body before the non-loopback denial."""
            return {
                "body": (
                    b'{"id":1,"jsonrpc":"2.0","method":"tools/list",'
                    b'"params":{}}'
                ),
                "more_body": False,
                "type": "http.request",
            }

        assert (
            await adapter.mcp(Request(remote_scope, receive))
        ).status_code == 404

    run(scenario())


def test_adapter_helper_failures_are_closed_and_identity_independent() -> None:
    """Exercise parser, handle, loopback, and schema helper failure paths."""
    owner = _identity("owner")
    with pytest.raises(PatchProtocolAdapterError):
        patch_protocols_module._call({})
    with pytest.raises(PatchProtocolAdapterError):
        patch_protocols_module._call(
            {
                "arguments": {},
                "name": "unknown",
                "retransmission_key": "valid-key",
            }
        )
    with pytest.raises(PatchProtocolAdapterError):
        patch_protocols_module._call(
            {
                "arguments": [],
                "name": "patch.edit",
                "retransmission_key": "valid-key",
            }
        )
    with pytest.raises(PatchProtocolAdapterError):
        patch_protocols_module._call(
            {
                "arguments": {},
                "name": 1,
                "retransmission_key": "valid-key",
            }
        )
    with pytest.raises(PatchProtocolAdapterError):
        patch_protocols_module._call(
            {
                "arguments": {},
                "name": "patch.edit",
                "retransmission_key": "",
            }
        )
    with pytest.raises(PatchProtocolAdapterError):
        patch_protocols_module._canonical_arguments({"not-json": float("nan")})
    assert patch_protocols_module._json_value(None) is None
    assert patch_protocols_module._json_value(True) is True
    assert patch_protocols_module._json_value(1) == 1
    assert patch_protocols_module._json_value(1.0) == 1.0
    with pytest.raises(PatchProtocolAdapterError):
        patch_protocols_module._json_value(object())
    with pytest.raises(PatchProtocolAdapterError):
        patch_protocols_module._tool_schema("unknown")
    with pytest.raises(PatchProtocolAdapterError):
        patch_protocols_module._task_handle(
            {"operation_handle": 1, "task_id": owner.task.value}, owner
        )
    with pytest.raises(ValueError):
        patch_protocols_module._string({"value": 1}, "value")
    assert patch_protocols_module._request_id({"id": True}) is None
    assert patch_protocols_module._request_id(None) is None
    assert patch_protocols_module._correlation(
        owner,
        RetransmissionKey("helper-key"),
        b"p" * 32,
    ).value.startswith("correlation_")
    no_client = Request(
        {
            "headers": [],
            "method": "POST",
            "path": "/",
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "type": "http",
        }
    )
    invalid_client = Request(
        {
            "client": ("not an address", 1),
            "headers": [],
            "method": "POST",
            "path": "/",
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "type": "http",
        }
    )
    localhost = Request(
        {
            "client": ("localhost", 1),
            "headers": [],
            "method": "POST",
            "path": "/",
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "type": "http",
        }
    )
    assert not patch_protocols_module._is_loopback(no_client)
    assert not patch_protocols_module._is_loopback(invalid_client)
    assert patch_protocols_module._is_loopback(localhost)
    reservation = DurableReservation(
        PatchRequestId.new(),
        owner.durable_identity(RetransmissionKey("helper-reservation")),
        AlgorithmDigest.from_bytes(b"helper"),
        False,
    )
    with pytest.raises(PatchProtocolAdapterError):
        patch_protocols_module._ProtocolHandle(
            PatchProtocolSurface.MCP,
            owner,
            OperationType.EDIT,
            PatchObserverCorrelationId.new(),
            _OtherReservation(
                reservation.request_id,
                reservation.identity,
                reservation.canonical_digest,
                reservation.replayed,
            ),
        )
    assert reservation.request_id.value.startswith("request_")
