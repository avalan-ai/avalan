"""Expose test-only loopback MCP and A2A patch protocol adapters.

The normal server does not import or register these routes.  A caller must
explicitly bind this adapter to complete authenticated test profiles before
either protocol can advertise a patch tool.
"""

from base64 import urlsafe_b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from ipaddress import ip_address
from json import JSONDecodeError, dumps, loads
from typing import Protocol, final

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from avalan.patch.coordinator import CoordinatorError, RetransmissionKey
from avalan.patch.domain import (
    AlgorithmDigest,
    OperationType,
    PatchContextId,
    PatchExecutionId,
    PatchObserverCorrelationId,
    PatchRequestId,
    PatchWorkspaceId,
)
from avalan.patch.durable_store import (
    DurablePatchStore,
    DurableReservation,
)
from avalan.patch.parser import PatchInputLimits, PatchRequestParser
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
    PatchProtocolContinuation,
    PatchProtocolContinuationKind,
    PatchProtocolIdentity,
    PatchProtocolProfile,
    PatchProtocolReservation,
    PatchProtocols,
    PatchProtocolSurface,
)
from avalan.types import JsonObject, MutableJsonValue

_PREFIX = "/__avalan_test__/patch-protocol/v1"
_ERROR = "Patch operation unavailable."
_HANDLE_VERSION = 1


class PatchProtocolAdapterError(RuntimeError):
    """Report one privacy-safe protocol adapter failure."""


class PatchProtocolIdentityResolver(Protocol):
    """Resolve the server-derived authenticated protocol identity."""

    async def __call__(self, request: Request) -> PatchProtocolIdentity | None:
        """Return the complete request authority or no authority."""


class PatchProtocolExecutor(Protocol):
    """Execute one durable protocol plan without caller-provided authority."""

    async def plan(
        self,
        reservation: PatchProtocolReservation,
        operation: OperationType,
        raw_arguments: bytes,
    ) -> None:
        """Persist the reviewable plan after the durable reservation."""

    async def approve(self, reservation: PatchProtocolReservation) -> None:
        """Consume detached approval and start the one fenced settlement."""

    async def await_result(
        self, reservation: PatchProtocolReservation
    ) -> None:
        """Reconcile the original pending settlement without retrying it."""


@dataclass(frozen=True, slots=True)
class PatchProtocolAdapterConfiguration:
    """Bind exact MCP and A2A test-only protocol dependencies."""

    mcp_profile: PatchProtocolProfile
    a2a_profile: PatchProtocolProfile
    store: DurablePatchStore
    identity_resolver: PatchProtocolIdentityResolver
    executor: PatchProtocolExecutor
    handle_key: bytes
    input_limits: PatchInputLimits = PatchInputLimits()
    prefix: str = _PREFIX

    def __post_init__(self) -> None:
        """Reject malformed route configuration before advertisement."""
        required_executor_methods = ("plan", "approve", "await_result")
        if (
            type(self.mcp_profile) is not PatchProtocolProfile
            or self.mcp_profile.surface is not PatchProtocolSurface.MCP
            or type(self.a2a_profile) is not PatchProtocolProfile
            or self.a2a_profile.surface is not PatchProtocolSurface.A2A
            or type(self.handle_key) is not bytes
            or len(self.handle_key) != 32
            or type(self.input_limits) is not PatchInputLimits
            or type(self.prefix) is not str
            or self.prefix != _PREFIX
            or not callable(self.identity_resolver)
            or not callable(getattr(self.store, "reserve", None))
            or not callable(getattr(self.store, "inspect", None))
            or any(
                not callable(getattr(self.executor, method, None))
                for method in required_executor_methods
            )
        ):
            raise PatchProtocolAdapterError("patch protocol unavailable")


@dataclass(frozen=True, slots=True)
class _ProtocolHandle:
    """Store the encrypted authenticated continuation coordinates."""

    surface: PatchProtocolSurface
    identity: PatchProtocolIdentity
    operation: OperationType
    correlation: PatchObserverCorrelationId
    reservation: DurableReservation

    def __post_init__(self) -> None:
        """Require an exact durable continuation before serializing it."""
        if (
            type(self.surface) is not PatchProtocolSurface
            or type(self.identity) is not PatchProtocolIdentity
            or type(self.operation) is not OperationType
            or type(self.correlation) is not PatchObserverCorrelationId
            or type(self.reservation) is not DurableReservation
        ):
            raise PatchProtocolAdapterError("patch protocol unavailable")


@final
class PatchProtocolAdapter:
    """Serve explicit MCP and A2A protocol continuations over loopback."""

    def __init__(
        self, configuration: PatchProtocolAdapterConfiguration
    ) -> None:
        """Bind one disabled-by-default adapter configuration."""
        if type(configuration) is not PatchProtocolAdapterConfiguration:
            raise PatchProtocolAdapterError("patch protocol unavailable")
        self._configuration = configuration
        self._fernet = Fernet(urlsafe_b64encode(configuration.handle_key))

    async def mcp(self, request: Request) -> JSONResponse:
        """Serve one MCP JSON-RPC request without a normal router binding."""
        payload = await _request_payload(request)
        request_id = _request_id(payload)
        try:
            identity = await self._identity(request)
            method, params = _jsonrpc_request(payload)
            result: JsonObject
            if method == "tools/list":
                tools: list[MutableJsonValue] = []
                tools.extend(self._mcp_tools(identity))
                result = {"tools": tools}
            elif method == "tools/call":
                continuation = await self._begin(
                    PatchProtocolSurface.MCP,
                    identity,
                    _call(params),
                )
                result = self._mcp_continuation(continuation)
            else:
                raise PatchProtocolAdapterError("patch protocol unavailable")
        except (Exception,):
            return _mcp_error(request_id)
        return JSONResponse(_mcp_result(request_id, result))

    async def mcp_status(self, request: Request, handle: str) -> JSONResponse:
        """Read one authenticated MCP continuation from its opaque handle."""
        return await self._mcp_continuation_response(request, handle, "status")

    async def mcp_approve(self, request: Request, handle: str) -> JSONResponse:
        """Approve one authenticated detached MCP review exactly once."""
        return await self._mcp_continuation_response(
            request, handle, "approve"
        )

    async def mcp_await(self, request: Request, handle: str) -> JSONResponse:
        """Await one authenticated MCP settlement without dispatching again."""
        return await self._mcp_continuation_response(request, handle, "await")

    async def a2a(self, request: Request) -> JSONResponse:
        """Serve one A2A request with typed task continuation state."""
        payload = await _request_payload(request)
        request_id = _request_id(payload)
        try:
            identity = await self._identity(request)
            method, params = _jsonrpc_request(payload)
            if method == "message/send":
                continuation = await self._a2a_message(identity, params)
            elif method == "tasks/get":
                continuation = await self._load(
                    PatchProtocolSurface.A2A,
                    identity,
                    _task_handle(params, identity),
                )
            else:
                raise PatchProtocolAdapterError("patch protocol unavailable")
            result = self._a2a_continuation(continuation)
        except (Exception,):
            return _a2a_error(request_id)
        return JSONResponse(_mcp_result(request_id, result))

    async def _mcp_continuation_response(
        self,
        request: Request,
        handle: str,
        action: str,
    ) -> JSONResponse:
        """Project one MCP status, approval, or await continuation action."""
        try:
            identity = await self._identity(request)
            continuation = await self._load(
                PatchProtocolSurface.MCP, identity, handle
            )
            if action == "approve":
                continuation = await self._approve(continuation)
            elif action == "await":
                continuation = await self._await(continuation)
            elif action != "status":
                raise PatchProtocolAdapterError("patch protocol unavailable")
        except (Exception,):
            return _mcp_error(None)
        return JSONResponse(
            _mcp_result(None, self._mcp_continuation(continuation))
        )

    async def _a2a_message(
        self,
        identity: PatchProtocolIdentity,
        params: Mapping[str, object],
    ) -> PatchProtocolContinuation:
        """Map A2A new-call and approval messages to typed continuations."""
        task_id = params.get("task_id")
        message = params.get("message")
        if (
            set(params) != {"message", "task_id"}
            or task_id != identity.task.value
            or not isinstance(message, Mapping)
            or not all(isinstance(key, str) for key in message)
        ):
            raise PatchProtocolAdapterError("patch protocol unavailable")
        kind = message.get("kind")
        if kind == "patch.call":
            if set(message) != {
                "arguments",
                "kind",
                "name",
                "retransmission_key",
            }:
                raise PatchProtocolAdapterError("patch protocol unavailable")
            return await self._begin(
                PatchProtocolSurface.A2A,
                identity,
                _call(
                    {
                        "arguments": message["arguments"],
                        "name": message["name"],
                        "retransmission_key": message["retransmission_key"],
                    }
                ),
            )
        if kind == "patch.approval":
            if set(message) != {"kind", "operation_handle"}:
                raise PatchProtocolAdapterError("patch protocol unavailable")
            handle = message.get("operation_handle")
            if not isinstance(handle, str):
                raise PatchProtocolAdapterError("patch protocol unavailable")
            return await self._approve(
                await self._load(PatchProtocolSurface.A2A, identity, handle)
            )
        if kind == "patch.resume":
            if set(message) != {"kind", "operation_handle"}:
                raise PatchProtocolAdapterError("patch protocol unavailable")
            handle = message.get("operation_handle")
            if not isinstance(handle, str):
                raise PatchProtocolAdapterError("patch protocol unavailable")
            return await self._await(
                await self._load(PatchProtocolSurface.A2A, identity, handle)
            )
        raise PatchProtocolAdapterError("patch protocol unavailable")

    async def _begin(
        self,
        surface: PatchProtocolSurface,
        identity: PatchProtocolIdentity,
        call: tuple[OperationType, JsonObject, RetransmissionKey],
    ) -> PatchProtocolContinuation:
        """Reserve canonically before the first planner callback can run."""
        operation, arguments, key = call
        protocol = self._protocol(surface, identity)
        raw_arguments = _canonical_arguments(arguments)
        correlation = _correlation(
            identity, key, self._configuration.handle_key
        )
        reservation, _ = await protocol.reserve_before_planning(
            self._configuration.store,
            operation,
            raw_arguments,
            key,
            correlation,
            PatchRequestParser(self._configuration.input_limits),
            lambda value: self._configuration.executor.plan(
                value, operation, raw_arguments
            ),
        )
        return await protocol.inspect(self._configuration.store, reservation)

    async def _load(
        self,
        surface: PatchProtocolSurface,
        identity: PatchProtocolIdentity,
        handle: str,
    ) -> PatchProtocolContinuation:
        """Open one opaque handle only for its originating exact identity."""
        value = self._open(handle)
        if value.surface is not surface or value.identity != identity:
            raise PatchProtocolAdapterError("patch protocol unavailable")
        protocol = self._protocol(surface, identity)
        return await protocol.inspect(
            self._configuration.store,
            PatchProtocolReservation(
                surface,
                identity,
                value.operation,
                value.correlation,
                value.reservation,
            ),
        )

    async def _approve(
        self, continuation: PatchProtocolContinuation
    ) -> PatchProtocolContinuation:
        """Advance only input-required state into authenticated settlement."""
        if (
            continuation.kind
            is not PatchProtocolContinuationKind.APPROVAL_REQUIRED
        ):
            raise PatchProtocolAdapterError("patch protocol unavailable")
        await self._configuration.executor.approve(continuation.reservation)
        return await self._protocol(
            continuation.reservation.surface,
            continuation.reservation.identity,
        ).inspect(self._configuration.store, continuation.reservation)

    async def _await(
        self, continuation: PatchProtocolContinuation
    ) -> PatchProtocolContinuation:
        """Reconcile only a known pending settlement on its original branch."""
        if (
            continuation.kind
            is not PatchProtocolContinuationKind.SETTLEMENT_PENDING
        ):
            raise PatchProtocolAdapterError("patch protocol unavailable")
        await self._configuration.executor.await_result(
            continuation.reservation
        )
        return await self._protocol(
            continuation.reservation.surface,
            continuation.reservation.identity,
        ).inspect(self._configuration.store, continuation.reservation)

    async def _identity(self, request: Request) -> PatchProtocolIdentity:
        """Require loopback transport and complete server-derived authority."""
        if not _is_loopback(request):
            raise PatchProtocolAdapterError("patch protocol unavailable")
        identity = await self._configuration.identity_resolver(request)
        if type(identity) is not PatchProtocolIdentity:
            raise PatchProtocolAdapterError("patch protocol unavailable")
        return identity

    def _mcp_tools(self, identity: PatchProtocolIdentity) -> list[JsonObject]:
        """List only active canonical MCP patch tools without a predicate."""
        tools = self._protocol(
            PatchProtocolSurface.MCP, identity
        ).advertised_tools()
        return [
            {"inputSchema": _tool_schema(name), "name": name} for name in tools
        ]

    def _protocol(
        self,
        surface: PatchProtocolSurface,
        identity: PatchProtocolIdentity,
    ) -> PatchProtocols:
        """Bind one configured protocol profile to current server authority."""
        profile = (
            self._configuration.mcp_profile
            if surface is PatchProtocolSurface.MCP
            else self._configuration.a2a_profile
        )
        return PatchProtocols(profile, identity)

    def _seal(self, continuation: PatchProtocolContinuation) -> str:
        """Encrypt one complete authenticated continuation handle."""
        reservation = continuation.reservation.durable
        identity = continuation.reservation.identity
        payload = {
            "correlation": continuation.reservation.correlation.value,
            "digest": reservation.canonical_digest.value,
            "identity": {
                "agent": identity.agent.value,
                "context": identity.context.value,
                "execution": identity.execution.value,
                "principal": identity.principal.value,
                "route": identity.route.value,
                "run": identity.run.value,
                "session": identity.session.value,
                "task": identity.task.value,
                "tenant": identity.tenant.value,
                "workspace": identity.workspace.value,
            },
            "operation": continuation.reservation.operation.value,
            "key": reservation.identity.retransmission_key.value,
            "replayed": reservation.replayed,
            "request": reservation.request_id.value,
            "surface": continuation.reservation.surface.value,
            "version": _HANDLE_VERSION,
        }
        return self._fernet.encrypt(
            dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).decode()

    def _open(self, handle: str) -> _ProtocolHandle:
        """Decrypt and validate one continuation handle without an oracle."""
        if not isinstance(handle, str) or not handle:
            raise PatchProtocolAdapterError("patch protocol unavailable")
        try:
            payload = loads(self._fernet.decrypt(handle.encode()))
            if not isinstance(payload, dict) or set(payload) != {
                "correlation",
                "digest",
                "identity",
                "key",
                "operation",
                "replayed",
                "request",
                "surface",
                "version",
            }:
                raise ValueError("invalid handle")
            values = payload["identity"]
            if not isinstance(values, dict) or set(values) != {
                "agent",
                "context",
                "execution",
                "principal",
                "route",
                "run",
                "session",
                "task",
                "tenant",
                "workspace",
            }:
                raise ValueError("invalid handle")
            identity = PatchProtocolIdentity(
                tenant=PatchTenantId(_string(values, "tenant")),
                principal=PatchPrincipalId(_string(values, "principal")),
                execution=PatchExecutionId(_string(values, "execution")),
                run=PatchRunId(_string(values, "run")),
                session=PatchSessionId(_string(values, "session")),
                task=PatchTaskId(_string(values, "task")),
                agent=PatchAgentId(_string(values, "agent")),
                route=PolicyRouteId(_string(values, "route")),
                context=PatchContextId(_string(values, "context")),
                workspace=PatchWorkspaceId(_string(values, "workspace")),
            )
            if (
                payload["version"] != _HANDLE_VERSION
                or type(payload["replayed"]) is not bool
            ):
                raise ValueError("invalid handle")
            retransmission_key = RetransmissionKey(_string(payload, "key"))
            return _ProtocolHandle(
                PatchProtocolSurface(_string(payload, "surface")),
                identity,
                OperationType(_string(payload, "operation")),
                PatchObserverCorrelationId(_string(payload, "correlation")),
                DurableReservation(
                    PatchRequestId(_string(payload, "request")),
                    identity.durable_identity(retransmission_key),
                    AlgorithmDigest("sha256", _string(payload, "digest")),
                    payload["replayed"],
                ),
            )
        except (
            InvalidToken,
            KeyError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as error:
            raise PatchProtocolAdapterError(
                "patch protocol unavailable"
            ) from error

    def _mcp_continuation(
        self, continuation: PatchProtocolContinuation
    ) -> JsonObject:
        """Project one content-free MCP continuation with a sealed handle."""
        return {
            "content": [{"text": "patch continuation", "type": "text"}],
            "isError": False,
            "structuredContent": self._continuation_payload(continuation),
        }

    def _a2a_continuation(
        self, continuation: PatchProtocolContinuation
    ) -> JsonObject:
        """Project approval and pending as distinct A2A task status states."""
        state = {
            PatchProtocolContinuationKind.APPROVAL_REQUIRED: "input-required",
            PatchProtocolContinuationKind.SETTLEMENT_PENDING: "working",
            PatchProtocolContinuationKind.TERMINAL: "completed",
        }[continuation.kind]
        return {
            "id": continuation.reservation.identity.task.value,
            "status": {
                "message": self._continuation_payload(continuation),
                "state": state,
            },
        }

    def _continuation_payload(
        self, continuation: PatchProtocolContinuation
    ) -> JsonObject:
        """Return bounded metadata without patch content or authority."""
        payload: JsonObject = {
            "operation_handle": self._seal(continuation),
            "state": continuation.kind.value,
        }
        if continuation.kind is PatchProtocolContinuationKind.TERMINAL:
            result = continuation.result
            if result is None:
                raise PatchProtocolAdapterError("patch protocol unavailable")
            payload["result"] = {
                "lifecycle": result.lifecycle.value,
                "status": result.status.value,
            }
        return payload


def install_patch_protocol_test_routes(
    app: FastAPI,
    configuration: PatchProtocolAdapterConfiguration,
) -> PatchProtocolAdapter:
    """Install only explicit loopback MCP and A2A test-profile routes."""
    if (
        type(app) is not FastAPI
        or type(configuration) is not PatchProtocolAdapterConfiguration
    ):
        raise PatchProtocolAdapterError("patch protocol unavailable")
    adapter = PatchProtocolAdapter(configuration)
    router = APIRouter(prefix=configuration.prefix)

    @router.post("/mcp")
    async def mcp(request: Request) -> JSONResponse:
        """Dispatch one test-only MCP request."""
        return await adapter.mcp(request)

    @router.post("/mcp/operations/{handle}/status")
    async def mcp_status(request: Request, handle: str) -> JSONResponse:
        """Read one test-only MCP continuation."""
        return await adapter.mcp_status(request, handle)

    @router.post("/mcp/operations/{handle}/approval")
    async def mcp_approve(request: Request, handle: str) -> JSONResponse:
        """Approve one test-only MCP review."""
        return await adapter.mcp_approve(request, handle)

    @router.post("/mcp/operations/{handle}/await")
    async def mcp_await(request: Request, handle: str) -> JSONResponse:
        """Await one test-only MCP settlement."""
        return await adapter.mcp_await(request, handle)

    @router.post("/a2a")
    async def a2a(request: Request) -> JSONResponse:
        """Dispatch one test-only A2A request."""
        return await adapter.a2a(request)

    app.include_router(router)
    return adapter


def _request_id(payload: object) -> str | int | None:
    """Return a JSON-RPC scalar identifier without granting authority."""
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("id")
    return (
        value
        if isinstance(value, (str, int)) and not isinstance(value, bool)
        else None
    )


async def _request_payload(request: Request) -> object:
    """Decode a JSON request body without exposing parser diagnostics."""
    try:
        return loads(await request.body())
    except (JSONDecodeError, UnicodeError):
        return None


def _jsonrpc_request(payload: object) -> tuple[str, Mapping[str, object]]:
    """Validate one closed JSON-RPC request envelope."""
    if not isinstance(payload, Mapping) or set(payload) != {
        "id",
        "jsonrpc",
        "method",
        "params",
    }:
        raise PatchProtocolAdapterError("patch protocol unavailable")
    method = payload.get("method")
    params = payload.get("params")
    if (
        payload.get("jsonrpc") != "2.0"
        or not isinstance(method, str)
        or not isinstance(params, Mapping)
    ):
        raise PatchProtocolAdapterError("patch protocol unavailable")
    return method, params


def _call(
    params: Mapping[str, object],
) -> tuple[OperationType, JsonObject, RetransmissionKey]:
    """Extract only one canonical tool identity, input, and retry key."""
    if set(params) != {"arguments", "name", "retransmission_key"}:
        raise PatchProtocolAdapterError("patch protocol unavailable")
    name = params.get("name")
    arguments = _json_object(params.get("arguments"))
    retransmission_key = params.get("retransmission_key")
    if not isinstance(name, str) or not isinstance(retransmission_key, str):
        raise PatchProtocolAdapterError("patch protocol unavailable")
    try:
        operation = {
            "patch.edit": OperationType.EDIT,
            "patch.apply": OperationType.APPLY,
        }[name]
        return operation, arguments, RetransmissionKey(retransmission_key)
    except (CoordinatorError, KeyError, ValueError) as error:
        raise PatchProtocolAdapterError(
            "patch protocol unavailable"
        ) from error


def _json_object(value: object) -> JsonObject:
    """Validate one JSON object without admitting caller-owned authority."""
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise PatchProtocolAdapterError("patch protocol unavailable")
    return {key: _json_value(item) for key, item in value.items()}


def _json_value(value: object) -> MutableJsonValue:
    """Validate one finite JSON-compatible transport value recursively."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return _json_object(value)
    raise PatchProtocolAdapterError("patch protocol unavailable")


def _canonical_arguments(arguments: JsonObject) -> bytes:
    """Encode canonical tool input before durable reservation or planning."""
    try:
        return dumps(
            arguments,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, UnicodeError, ValueError) as error:
        raise PatchProtocolAdapterError(
            "patch protocol unavailable"
        ) from error


def _correlation(
    identity: PatchProtocolIdentity,
    key: RetransmissionKey,
    handle_key: bytes,
) -> PatchObserverCorrelationId:
    """Derive a server-owned correlation from authority and retry identity."""
    material = "\x00".join(
        (
            identity.tenant.value,
            identity.principal.value,
            identity.execution.value,
            identity.route.value,
            key.value,
        )
    ).encode()
    return PatchObserverCorrelationId(
        "correlation_" + sha256(handle_key + material).hexdigest()[:32]
    )


def _tool_schema(name: str) -> JsonObject:
    """Return the fixed canonical schema for one advertised patch tool."""
    if name == "patch.edit":
        return {
            "additionalProperties": False,
            "properties": {
                "edits": {"items": {"type": "object"}, "type": "array"},
                "path": {"type": "string"},
            },
            "required": ["path", "edits"],
            "type": "object",
        }
    if name == "patch.apply":
        return {
            "additionalProperties": False,
            "properties": {"patch": {"type": "string"}},
            "required": ["patch"],
            "type": "object",
        }
    raise PatchProtocolAdapterError("patch protocol unavailable")


def _task_handle(
    params: Mapping[str, object], identity: PatchProtocolIdentity
) -> str:
    """Require A2A task reads to preserve the originating task identity."""
    if (
        set(params) != {"operation_handle", "task_id"}
        or params.get("task_id") != identity.task.value
    ):
        raise PatchProtocolAdapterError("patch protocol unavailable")
    handle = params.get("operation_handle")
    if not isinstance(handle, str):
        raise PatchProtocolAdapterError("patch protocol unavailable")
    return handle


def _mcp_result(
    request_id: str | int | None, result: JsonObject
) -> JsonObject:
    """Return one JSON-RPC success envelope with no protected diagnostics."""
    return {"id": request_id, "jsonrpc": "2.0", "result": result}


def _mcp_error(request_id: str | int | None) -> JSONResponse:
    """Return one non-oracular MCP error for every denied continuation."""
    return JSONResponse(
        status_code=404,
        content={
            "error": {"code": -32001, "message": _ERROR},
            "id": request_id,
            "jsonrpc": "2.0",
        },
    )


def _a2a_error(request_id: str | int | None) -> JSONResponse:
    """Return one non-oracular A2A error for every denied continuation."""
    return _mcp_error(request_id)


def _string(value: Mapping[str, object], key: str) -> str:
    """Read one exact string field from encrypted continuation payload."""
    result = value[key]
    if not isinstance(result, str):
        raise ValueError("invalid handle")
    return result


def _is_loopback(request: Request) -> bool:
    """Return whether the adapter request came from the loopback transport."""
    client = request.client
    if client is None:
        return False
    try:
        return ip_address(client.host).is_loopback
    except ValueError:
        return client.host == "localhost"
