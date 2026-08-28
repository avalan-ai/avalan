"""Host the isolated remote patch test app for loopback TCP process tests."""

from asyncio import Event
from collections.abc import AsyncIterator
from dataclasses import replace
from json import dumps
from os import environ

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from patch_remote_test import (
    _active_configuration,
    _authority,
    _RemoteService,
    _result,
    _RuntimeBinder,
)

from avalan.patch.domain import (
    LifecyclePhase,
    OperationType,
    PatchEventId,
    PatchObserverCorrelationId,
    PatchRequestId,
    PatchResult,
    SequenceNumber,
)
from avalan.patch.durable_store import (
    DurableOutboxRecord,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableRequestSnapshot,
    DurableTerminalRecord,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.toolset import PatchInvocationCapability
from avalan.server.patch import install_remote_patch_test_routes

_SECRET = environ["AVALAN_PATCH_TCP_TEST_SECRET"]
_AUTHORITY = _authority()


class _TcpSettlingStore(InMemoryDurablePatchStore):
    """Expose deterministic terminal truth only for the TCP child fixture."""

    def __init__(self) -> None:
        """Bind a private in-memory store without an external dependency."""
        super().__init__(InMemoryDurablePatchBackend())
        self._terminals: dict[PatchRequestId, DurableTerminalRecord] = {}

    async def record_terminal(
        self,
        request_id: PatchRequestId,
        correlation_id: PatchObserverCorrelationId,
    ) -> None:
        """Record one deterministic test-only terminal lifecycle event."""
        if request_id not in self._terminals:
            outbox = DurableOutboxRecord(
                PatchEventId.new(),
                request_id,
                SequenceNumber(1),
                LifecyclePhase.REQUEST_COMPLETED,
                correlation_id,
            )
            self._terminals[request_id] = DurableTerminalRecord(
                _result(request_id),
                outbox,
                None,
            )

    async def inspect(
        self, access: DurableRequestAccess
    ) -> DurableRequestSnapshot:
        """Project terminal truth without changing production store logic."""
        snapshot = await super().inspect(access)
        terminal = self._terminals.get(access.request_id)
        if terminal is None:
            return snapshot
        return replace(
            snapshot,
            lifecycle=LifecyclePhase.REQUEST_COMPLETED,
            pending=None,
            terminal=terminal,
            event_cursor=terminal.outbox.sequence,
        )

    async def outbox(
        self,
        access: DurableRequestAccess,
        after: SequenceNumber,
        limit: int,
    ) -> tuple[DurableOutboxRecord, ...]:
        """Return the one fixture terminal event under normal resume rules."""
        terminal = self._terminals.get(access.request_id)
        if terminal is not None:
            return (
                (terminal.outbox,)
                if terminal.outbox.sequence.value > after.value
                else ()
            )
        return await super().outbox(access, after, limit)


class _TcpSettlingService(_RemoteService):
    """Record one remote invocation and expose its deterministic terminal."""

    async def invoke_remote(
        self,
        operation: OperationType,
        raw_arguments: bytes,
        capability: PatchInvocationCapability,
        request_id: PatchRequestId,
        correlation_id: PatchObserverCorrelationId,
        identity: DurableRequestIdentity,
    ) -> PatchResult:
        """Settle exactly one child-process invocation after it is observed."""
        result = await super().invoke_remote(
            operation,
            raw_arguments,
            capability,
            request_id,
            correlation_id,
            identity,
        )
        await self.store.record_terminal(request_id, correlation_id)
        return result


_STORE = _TcpSettlingStore()
_SERVICE = _TcpSettlingService(_STORE)
_BINDER = _RuntimeBinder(_AUTHORITY, _SERVICE)
_BASE_CONFIGURATION, _, _ = _active_configuration(_AUTHORITY)
_CONFIGURATION = replace(
    _BASE_CONFIGURATION,
    binder=_BINDER,
    store=_STORE,
    attestation_secret=_SECRET,
)
_LIVE_RELEASE = Event()
app = FastAPI()


def _require_attestation(
    attestation: str | None,
    correlation: str | None,
) -> None:
    """Reject every un-attested fixture stream request without detail."""
    if attestation != _SECRET or correlation != _AUTHORITY.correlation:
        raise HTTPException(status_code=404)


async def _live_event_stream(after: int) -> AsyncIterator[str]:
    """Yield a duplicate pending event then await child-test settlement."""
    if after < 1:
        pending = (
            "id: 1\nevent: patch.lifecycle\ndata: "
            + dumps(
                {
                    "cursor": 1,
                    "event_id": "event_live_pending",
                    "lifecycle": "planned",
                    "object": "patch.event",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n\n"
        )
        yield pending
        yield pending
    await _LIVE_RELEASE.wait()
    if after < 2:
        yield (
            "id: 2\nevent: patch.lifecycle\ndata: "
            + dumps(
                {
                    "cursor": 2,
                    "event_id": "event_live_terminal",
                    "lifecycle": "request_completed",
                    "object": "patch.event",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n\n"
        )


@app.get("/__avalan_test__/patch/v1/operations/live/events")
async def live_events(
    after: int = 0,
    x_avalan_test_attestation: str | None = Header(default=None),
    x_avalan_correlation: str | None = Header(default=None),
) -> StreamingResponse:
    """Stream fixture live events before the terminal response exists."""
    _require_attestation(x_avalan_test_attestation, x_avalan_correlation)
    return StreamingResponse(
        _live_event_stream(after),
        media_type="text/event-stream",
    )


@app.post("/__avalan_test__/patch/v1/operations/live/release")
async def release_live_events(
    x_avalan_test_attestation: str | None = Header(default=None),
    x_avalan_correlation: str | None = Header(default=None),
) -> dict[str, bool]:
    """Release the fixture terminal event only after caller confirmation."""
    _require_attestation(x_avalan_test_attestation, x_avalan_correlation)
    _LIVE_RELEASE.set()
    return {"released": True}


install_remote_patch_test_routes(app, _CONFIGURATION)


@app.get("/__avalan_test__/patch/v1/ready")
async def ready(
    x_avalan_test_attestation: str | None = Header(default=None),
) -> dict[str, int | bool]:
    """Attest this child process and expose only invocation cardinality."""
    if x_avalan_test_attestation != _SECRET:
        raise HTTPException(status_code=404)
    return {"ready": True, "invocations": len(_SERVICE.remote_calls)}
