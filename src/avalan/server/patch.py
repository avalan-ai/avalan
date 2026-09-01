"""Expose the isolated authenticated loopback patch test-server protocol.

This module is not registered by the normal Avalan server.  It is an explicit
Phase 13 local-test harness for a fully bound patch runtime: the server owns
the retransmission identity, persistence, event cursor, and opaque operation
handle.  Request bodies carry only the two canonical patch inputs.
"""

from asyncio import (
    CancelledError,
    Future,
    Lock,
    Task,
    create_task,
    shield,
    sleep,
)
from base64 import urlsafe_b64encode
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from hashlib import sha256
from ipaddress import ip_address
from json import JSONDecodeError, dumps, loads
from re import fullmatch
from typing import AsyncContextManager, Literal, Protocol, final

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from httpx import AsyncClient
from httpx import Response as HttpResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from avalan.patch.activation import is_patch_activation_runtime_factory
from avalan.patch.coordinator import RetransmissionKey
from avalan.patch.domain import (
    AlgorithmDigest,
    Capability,
    LifecyclePhase,
    OperationType,
    PatchContextId,
    PatchExecutionId,
    PatchObserverCorrelationId,
    PatchRequestId,
    PatchWorkspaceId,
    SequenceNumber,
)
from avalan.patch.durable_store import (
    DurablePatchStore,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableRequestSnapshot,
    DurableStoreError,
)
from avalan.patch.parser import (
    PatchInputError,
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderProfile,
    RawToolCallId,
)
from avalan.patch.policy import (
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PolicyDisclosure,
    PolicyRevision,
    PolicyRouteId,
)
from avalan.patch.toolset import (
    PatchRuntimeBinder,
    PatchSdkHost,
    PatchTestHostProfile,
    PatchToolLoader,
    PatchToolSet,
    RemotePatchRuntimeWitness,
)
from avalan.server.authority import remote_runtime_authority_key

_DEFAULT_PREFIX = "/__avalan_test__/patch/v1"
_MAX_EVENT_PAGE = 128
_AWAIT_POLLS = 200
_POLL_SECONDS = 0.05
_CORRELATION_PATTERN = r"[A-Za-z0-9_-]{8,128}"
_OPERATION_HANDLE_VERSION = 1
_FORBIDDEN_CALLER_FIELDS = frozenset(
    {
        "approval",
        "approvals",
        "backend",
        "capabilities",
        "capability",
        "confirmation",
        "containerprofile",
        "cwd",
        "disclosure",
        "limit",
        "limits",
        "matchingmode",
        "nativeitemshape",
        "policy",
        "policyversion",
        "schema",
        "validator",
        "worker",
        "workspace",
    }
)


class RemotePatchServerError(RuntimeError):
    """Report the one coarse remote patch protocol failure."""


@dataclass(frozen=True, slots=True)
class RemotePatchAuthority:
    """Bind every server-trusted remote mutation scope coordinate."""

    tenant: PatchTenantId
    principal: PatchPrincipalId
    run: PatchRunId
    session: PatchSessionId
    task: PatchTaskId
    agent: PatchAgentId
    execution_scope: str
    route: PolicyRouteId
    context: PatchContextId
    workspace: PatchWorkspaceId
    policy_revision: PolicyRevision
    disclosures: frozenset[PolicyDisclosure]
    approval_route: PolicyRouteId
    correlation: str
    capabilities: frozenset[Capability]

    def __post_init__(self) -> None:
        """Require one complete immutable authenticated route scope."""
        if (
            type(self.tenant) is not PatchTenantId
            or type(self.principal) is not PatchPrincipalId
            or type(self.run) is not PatchRunId
            or type(self.session) is not PatchSessionId
            or type(self.task) is not PatchTaskId
            or type(self.agent) is not PatchAgentId
            or not isinstance(self.execution_scope, str)
            or fullmatch(_CORRELATION_PATTERN, self.execution_scope) is None
            or type(self.route) is not PolicyRouteId
            or type(self.context) is not PatchContextId
            or type(self.workspace) is not PatchWorkspaceId
            or type(self.policy_revision) is not PolicyRevision
            or type(self.disclosures) is not frozenset
            or any(
                type(value) is not PolicyDisclosure
                for value in self.disclosures
            )
            or type(self.approval_route) is not PolicyRouteId
            or not isinstance(self.correlation, str)
            or fullmatch(_CORRELATION_PATTERN, self.correlation) is None
            or type(self.capabilities) is not frozenset
            or any(
                type(value) is not Capability for value in self.capabilities
            )
        ):
            raise RemotePatchServerError("remote patch configuration invalid")

    def canonical(self) -> dict[str, object]:
        """Return the non-secret trusted scope used for handle binding."""
        return {
            "agent": self.agent.value,
            "approval_route": self.approval_route.value,
            "context": self.context.value,
            "correlation": self.correlation,
            "capabilities": sorted(value.value for value in self.capabilities),
            "disclosures": sorted(value.value for value in self.disclosures),
            "execution_scope": self.execution_scope,
            "policy_revision": self.policy_revision.value,
            "principal": self.principal.value,
            "route": self.route.value,
            "run": self.run.value,
            "session": self.session.value,
            "task": self.task.value,
            "tenant": self.tenant.value,
            "workspace": self.workspace.value,
        }


class RemotePatchAuthorityResolver(Protocol):
    """Resolve a trusted patch authority from one server request."""

    async def __call__(self, request: Request) -> RemotePatchAuthority | None:
        """Return the exact authenticated authority or no principal."""


@dataclass(frozen=True, slots=True)
class RemotePatchTestServerProfile:
    """Require the isolated authenticated loopback activation profile."""

    enabled: bool = False
    authenticated: bool = False
    loopback_only: bool = False
    name: str = "authenticated-local-patch-test-server"

    def __post_init__(self) -> None:
        """Reject every production or partially configured profile."""
        if (
            type(self.enabled) is not bool
            or type(self.authenticated) is not bool
            or type(self.loopback_only) is not bool
            or self.name != "authenticated-local-patch-test-server"
        ):
            raise RemotePatchServerError("remote patch configuration invalid")

    @property
    def active(self) -> bool:
        """Return whether the sole test-only activation condition holds."""
        return self.enabled and self.authenticated and self.loopback_only


@dataclass(frozen=True, slots=True)
class RemotePatchTestServerConfiguration:
    """Bind the complete trusted remote test-server dependency set."""

    profile: RemotePatchTestServerProfile
    authority_resolver: RemotePatchAuthorityResolver
    expected_authority: RemotePatchAuthority
    binder: PatchRuntimeBinder
    activation_factory: object
    store: DurablePatchStore
    handle_key: bytes = field(repr=False)
    runtime_witness: RemotePatchRuntimeWitness
    prefix: str = _DEFAULT_PREFIX
    input_limits: PatchInputLimits = PatchInputLimits()
    attestation_secret: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Reject incomplete configuration before routes can be installed."""
        required_store_methods = (
            "reserve",
            "inspect",
            "inspect_pending",
            "await_terminal",
            "outbox",
            "request_cancellation",
        )
        if (
            type(self.profile) is not RemotePatchTestServerProfile
            or not self.profile.active
            or not callable(self.authority_resolver)
            or type(self.expected_authority) is not RemotePatchAuthority
            or type(self.runtime_witness) is not RemotePatchRuntimeWitness
            or self.runtime_witness
            != _expected_runtime_witness(self.expected_authority)
            or not isinstance(self.binder, PatchRuntimeBinder)
            or not is_patch_activation_runtime_factory(self.activation_factory)
            or any(
                not callable(getattr(self.store, name, None))
                for name in required_store_methods
            )
            or type(self.handle_key) is not bytes
            or len(self.handle_key) != 32
            or not isinstance(self.prefix, str)
            or not self.prefix.startswith("/")
            or self.prefix.rstrip("/") != self.prefix
            or type(self.input_limits) is not PatchInputLimits
            or (
                self.attestation_secret is not None
                and (
                    not isinstance(self.attestation_secret, str)
                    or fullmatch(
                        _CORRELATION_PATTERN,
                        self.attestation_secret,
                    )
                    is None
                )
            )
        ):
            raise RemotePatchServerError("remote patch configuration invalid")


class _RemotePatchRoute(APIRoute):
    """Coarsen closed-schema validation failures before transport delivery."""

    def get_route_handler(
        self,
    ) -> Callable[[Request], Awaitable[Response]]:
        """Return the route handler with the fixed error projection."""
        handler = super().get_route_handler()

        async def remote_patch_route(request: Request) -> Response:
            try:
                maximum = getattr(
                    request.app.state,
                    "remote_patch_test_input_bytes",
                    None,
                )
                if type(maximum) is not int or maximum < 1:
                    return _error_response(400)
                body = await _read_bounded_body(request, maximum)
                if body and not _patch_input_route(request):
                    raise RemotePatchServerError(
                        "remote patch operation unavailable"
                    )
                return await handler(request)
            except (RequestValidationError, RemotePatchServerError):
                return _error_response(400)

        return remote_patch_route


async def _read_bounded_body(request: Request, maximum: int) -> bytes:
    """Cache one streamed request body only after its raw byte bound holds."""
    size = 0
    parts: list[bytes] = []
    try:
        async for part in request.stream():
            size += len(part)
            if size > maximum:
                raise RemotePatchServerError(
                    "remote patch operation unavailable"
                )
            parts.append(part)
    except (RuntimeError, UnicodeError) as error:
        raise RemotePatchServerError(
            "remote patch operation unavailable"
        ) from error
    request._body = b"".join(parts)
    return request._body


def _patch_input_route(request: Request) -> bool:
    """Return whether one route is permitted to receive patch input bytes."""
    return request.method == "POST" and request.url.path.endswith(
        (
            "/edit",
            "/apply",
        )
    )


class _RemotePatchEdit(BaseModel):
    """Carry the closed canonical edit operation body."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(min_length=1, max_length=1024)
    edits: list["_RemotePatchEditPart"] = Field(min_length=1, max_length=16384)


class _RemotePatchEditPart(BaseModel):
    """Carry one exact structured text replacement declaration."""

    model_config = ConfigDict(extra="forbid", strict=True)

    old_text: str = Field(min_length=1, max_length=1_048_576)
    new_text: str = Field(max_length=1_048_576)


class _RemotePatchApply(BaseModel):
    """Carry the closed complete patch-document operation body."""

    model_config = ConfigDict(extra="forbid", strict=True)

    patch: str = Field(min_length=1, max_length=1_048_576)


class _RemotePatchTool(BaseModel):
    """Describe one closed advertised remote patch function."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["function"] = "function"
    name: Literal["patch.edit", "patch.apply"]
    strict: Literal[True] = True
    parameters: dict[str, object]


class _RemotePatchTools(BaseModel):
    """Return the complete remote test-profile tool advertisement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object: Literal["list"] = "list"
    data: list[_RemotePatchTool]


class _RemotePatchPendingResponse(BaseModel):
    """Return the only nonterminal remote operation projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object: Literal["patch.operation"] = "patch.operation"
    state: Literal["pending"] = "pending"
    operation_handle: str
    event_cursor: int


class _RemotePatchTerminalResponse(BaseModel):
    """Return a coarse terminal lifecycle projection without exact truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object: Literal["patch.operation"] = "patch.operation"
    state: Literal["completed"] = "completed"
    operation_handle: str
    event_cursor: int


class RemotePatchEvent(BaseModel):
    """Expose one content-free resumable patch lifecycle event."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    object: Literal["patch.event"]
    cursor: int = Field(ge=1)
    event_id: str = Field(min_length=1, max_length=256)
    lifecycle: LifecyclePhase = Field(strict=False)


RemotePatchEditPart = _RemotePatchEditPart
RemotePatchPendingOperation = _RemotePatchPendingResponse
RemotePatchTerminalOperation = _RemotePatchTerminalResponse
RemotePatchOperation = (
    RemotePatchPendingOperation | RemotePatchTerminalOperation
)


@final
class RemotePatchEventStream:
    """Manage one cancellable at-least-once remote SSE response."""

    def __init__(
        self,
        response_context: AsyncContextManager[HttpResponse],
        after: int,
    ) -> None:
        """Bind an unopened stream to its durable resume cursor."""
        self._response_context: AsyncContextManager[HttpResponse] | None = (
            response_context
        )
        self._lines: AsyncIterator[str] | None = None
        self._last_cursor = after
        self._last_event_id: str | None = None

    async def __aenter__(self) -> "RemotePatchEventStream":
        """Open and validate the streaming response before iteration."""
        context = self._response_context
        if context is None:
            raise RemotePatchServerError("remote patch operation unavailable")
        try:
            response = await context.__aenter__()
        except CancelledError:
            await self.aclose()
            raise
        except Exception as error:
            await self.aclose()
            raise RemotePatchServerError(
                "remote patch operation unavailable"
            ) from error
        if response.status_code >= 400:
            await self.aclose()
            raise RemotePatchServerError("remote patch operation unavailable")
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("text/event-stream"):
            await self.aclose()
            raise RemotePatchServerError("remote patch operation unavailable")
        self._lines = response.aiter_lines()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the underlying network response on every caller exit."""
        await self.aclose()

    def __aiter__(self) -> "RemotePatchEventStream":
        """Return the managed stream iterator after it has been opened."""
        if self._lines is None:
            raise RemotePatchServerError("remote patch operation unavailable")
        return self

    async def __anext__(self) -> RemotePatchEvent:
        """Read one monotonic SSE event and coalesce an immediate replay."""
        lines = self._lines
        if lines is None:
            raise RemotePatchServerError("remote patch operation unavailable")
        frame_id: str | None = None
        frame_event: str | None = None
        frame_data: str | None = None
        try:
            while True:
                line = await anext(lines)
                if not line:
                    event = _sse_event(frame_id, frame_event, frame_data)
                    frame_id = None
                    frame_event = None
                    frame_data = None
                    if event.cursor < self._last_cursor:
                        raise RemotePatchServerError(
                            "remote patch operation unavailable"
                        )
                    if event.cursor == self._last_cursor:
                        if event.event_id == self._last_event_id:
                            continue
                        raise RemotePatchServerError(
                            "remote patch operation unavailable"
                        )
                    self._last_cursor = event.cursor
                    self._last_event_id = event.event_id
                    return event
                if line.startswith("id: ") and frame_id is None:
                    frame_id = line.removeprefix("id: ")
                elif line.startswith("event: ") and frame_event is None:
                    frame_event = line.removeprefix("event: ")
                elif line.startswith("data: ") and frame_data is None:
                    frame_data = line.removeprefix("data: ")
                else:
                    raise RemotePatchServerError(
                        "remote patch operation unavailable"
                    )
        except CancelledError:
            await self.aclose()
            raise
        except StopAsyncIteration:
            await self.aclose()
            raise
        except RemotePatchServerError:
            await self.aclose()
            raise
        except Exception as error:
            await self.aclose()
            raise RemotePatchServerError(
                "remote patch operation unavailable"
            ) from error

    async def aclose(self) -> None:
        """Close the response exactly once without retaining stream state."""
        context = self._response_context
        self._response_context = None
        self._lines = None
        self._last_event_id = None
        if context is not None:
            await context.__aexit__(None, None, None)


@final
class RemotePatchTestClient:
    """Call only the closed authenticated local patch test protocol."""

    def __init__(
        self,
        client: AsyncClient,
        correlation: str,
        prefix: str = _DEFAULT_PREFIX,
        attestation_secret: str | None = None,
    ) -> None:
        """Bind one HTTP client to one trusted correlation and route prefix."""
        if (
            type(client) is not AsyncClient
            or not isinstance(correlation, str)
            or fullmatch(_CORRELATION_PATTERN, correlation) is None
            or not isinstance(prefix, str)
            or not prefix.startswith("/")
            or prefix.rstrip("/") != prefix
            or (
                attestation_secret is not None
                and (
                    not isinstance(attestation_secret, str)
                    or fullmatch(_CORRELATION_PATTERN, attestation_secret)
                    is None
                )
            )
        ):
            raise RemotePatchServerError("remote patch client invalid")
        self._client = client
        self._correlation = correlation
        self._prefix = prefix
        self._attestation_secret = attestation_secret

    async def tools(self) -> _RemotePatchTools:
        """List only the server-authorized remote test tools."""
        response = await self._request("GET", "/tools")
        return _RemotePatchTools.model_validate(response.json())

    async def edit(
        self,
        path: str,
        edits: list[RemotePatchEditPart],
        retransmission_key: str,
    ) -> RemotePatchOperation:
        """Begin or attach to one closed structured edit operation."""
        body = _RemotePatchEdit(path=path, edits=edits)
        return await self._begin(
            "/edit",
            body.model_dump(mode="python"),
            retransmission_key,
        )

    async def apply(
        self,
        patch: str,
        retransmission_key: str,
    ) -> RemotePatchOperation:
        """Begin or attach to one closed complete patch document operation."""
        body = _RemotePatchApply(patch=patch)
        return await self._begin(
            "/apply",
            body.model_dump(mode="python"),
            retransmission_key,
        )

    async def inspect(self, handle: str) -> RemotePatchOperation:
        """Inspect one opaque current-authority operation handle."""
        return _operation_response(
            await self._request("GET", f"/operations/{handle}")
        )

    async def await_result(self, handle: str) -> RemotePatchOperation:
        """Await one opaque operation without cancelling its work."""
        return _operation_response(
            await self._request("POST", f"/operations/{handle}/await")
        )

    async def cancel_intent(self, handle: str) -> RemotePatchOperation:
        """Record only cancellation intent for one opaque operation."""
        return _operation_response(
            await self._request("POST", f"/operations/{handle}/cancel")
        )

    def events(self, handle: str, after: int = 0) -> RemotePatchEventStream:
        """Open one managed at-least-once SSE stream from a durable cursor."""
        if type(after) is not int or after < 0:
            raise RemotePatchServerError("remote patch client invalid")
        return RemotePatchEventStream(
            self._client.stream(
                "GET",
                self._prefix + f"/operations/{handle}/events",
                headers=self._headers(),
                params={"after": str(after)},
            ),
            after,
        )

    async def _begin(
        self,
        suffix: str,
        body: dict[str, object],
        retransmission_key: str,
    ) -> RemotePatchOperation:
        """Post one validated canonical body under a retransmission key."""
        if not isinstance(retransmission_key, str) or not retransmission_key:
            raise RemotePatchServerError("remote patch client invalid")
        return _operation_response(
            await self._request(
                "POST",
                suffix,
                content=dumps(
                    body,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8"),
                retransmission_key=retransmission_key,
            )
        )

    async def _request(
        self,
        method: str,
        suffix: str,
        *,
        content: bytes | None = None,
        params: Mapping[str, str] | None = None,
        retransmission_key: str | None = None,
    ) -> HttpResponse:
        """Send one fixed-prefix correlation-bound request."""
        headers = self._headers()
        if content is not None:
            headers["Content-Type"] = "application/json"
        if retransmission_key is not None:
            headers["Idempotency-Key"] = retransmission_key
        response = await self._client.request(
            method,
            self._prefix + suffix,
            content=content,
            headers=headers,
            params=params,
        )
        if response.status_code >= 400:
            raise RemotePatchServerError("remote patch operation unavailable")
        return response

    def _headers(self) -> dict[str, str]:
        """Return the fixed authenticated headers for one client request."""
        headers = {"X-Avalan-Correlation": self._correlation}
        if self._attestation_secret is not None:
            headers["X-Avalan-Test-Attestation"] = self._attestation_secret
        return headers


def _operation_response(response: HttpResponse) -> RemotePatchOperation:
    """Validate one closed pending or terminal public operation response."""
    payload = response.json()
    if not isinstance(payload, dict):
        raise RemotePatchServerError("remote patch operation unavailable")
    state = payload.get("state")
    if state == "pending":
        return RemotePatchPendingOperation.model_validate(payload)
    if state == "completed":
        return RemotePatchTerminalOperation.model_validate(payload)
    raise RemotePatchServerError("remote patch operation unavailable")


def _sse_event(
    frame_id: str | None,
    frame_event: str | None,
    frame_data: str | None,
) -> RemotePatchEvent:
    """Validate one complete content-free lifecycle SSE frame."""
    if (
        frame_id is None
        or frame_event != "patch.lifecycle"
        or frame_data is None
    ):
        raise RemotePatchServerError("remote patch operation unavailable")
    try:
        cursor = int(frame_id)
        payload = loads(frame_data)
        if not isinstance(payload, dict):
            raise ValueError("SSE payload must be an object")
        event = RemotePatchEvent.model_validate(payload)
    except (JSONDecodeError, TypeError, ValidationError, ValueError) as error:
        raise RemotePatchServerError(
            "remote patch operation unavailable"
        ) from error
    if event.cursor != cursor:
        raise RemotePatchServerError("remote patch operation unavailable")
    return event


@dataclass(frozen=True, slots=True)
class _RemotePatchOperation:
    """Keep the encrypted handle payload server-owned and exact."""

    authority: RemotePatchAuthority
    request_id: PatchRequestId
    correlation_id: PatchObserverCorrelationId
    identity: DurableRequestIdentity

    @property
    def access(self) -> DurableRequestAccess:
        """Return the exact read authority bound into this opaque handle."""
        return DurableRequestAccess(self.request_id, self.identity)

    def to_bytes(self) -> bytes:
        """Return canonical encrypted-handle payload bytes."""
        payload = {
            "authority": self.authority.canonical(),
            "correlation_id": self.correlation_id.value,
            "identity": {
                "execution_id": self.identity.execution_id.value,
                "principal_id": self.identity.principal_id.value,
                "retransmission_key": self.identity.retransmission_key.value,
                "route_id": self.identity.route_id.value,
                "tenant_id": self.identity.tenant_id.value,
            },
            "request_id": self.request_id.value,
            "version": _OPERATION_HANDLE_VERSION,
        }
        return dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


@final
class RemotePatchController:
    """Own one authenticated local test-server durable route boundary."""

    def __init__(
        self, configuration: RemotePatchTestServerConfiguration
    ) -> None:
        """Bind configuration without starting a target or worker."""
        self._configuration = configuration
        self._fernet = Fernet(urlsafe_b64encode(configuration.handle_key))
        self._lock = Lock()
        self._toolset: PatchToolSet | None = None
        self._host: PatchSdkHost | None = None
        self._tasks: dict[PatchRequestId, Task[None]] = {}

    async def start(self) -> None:
        """Probe and bind the full runtime only for this activated profile."""
        async with self._lock:
            if self._host is not None:
                return
            try:
                bundle = await PatchToolLoader(
                    self._configuration.binder,
                    PatchTestHostProfile(
                        enabled=True,
                        authenticated=True,
                        activation_factory=(
                            self._configuration.activation_factory
                        ),
                    ),
                ).load(enable_tools=["patch.edit", "patch.apply"])
                binding = bundle.runtime_binding
                if (
                    binding is None
                    or binding.coordinator.durable_store
                    is not self._configuration.store
                    or binding.persistence.durable_store
                    is not self._configuration.store
                    or binding.remote_witness
                    != self._configuration.runtime_witness
                ):
                    raise RemotePatchServerError(
                        "remote patch configuration invalid"
                    )
                toolset = bundle.toolset
                if type(toolset) is not PatchToolSet or tuple(
                    tool.__name__
                    for tool in toolset.available_tools_for_enabled_tools(
                        [
                            "patch.edit",
                            "patch.apply",
                        ]
                    )
                ) != ("edit", "apply"):
                    raise RemotePatchServerError(
                        "remote patch configuration invalid"
                    )
                await toolset.__aenter__()
                self._toolset = toolset
                self._host = toolset.sdk_host()
            except BaseException:
                if self._toolset is not None:
                    await self._toolset.__aexit__(None, None, None)
                    self._toolset = None
                raise

    async def close(self) -> None:
        """Close host-owned tasks without changing durable operation truth."""
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await shield(task)
            except (CancelledError, Exception):
                pass
        toolset = self._toolset
        self._toolset = None
        self._host = None
        if toolset is not None:
            await toolset.__aexit__(None, None, None)

    async def authenticate(self, request: Request) -> RemotePatchAuthority:
        """Return only the preconfigured loopback authority and correlation."""
        if not _is_loopback(request):
            raise RemotePatchServerError("remote patch operation unavailable")
        if (
            self._configuration.attestation_secret is not None
            and request.headers.get("X-Avalan-Test-Attestation")
            != self._configuration.attestation_secret
        ):
            raise RemotePatchServerError("remote patch operation unavailable")
        authority = await self._configuration.authority_resolver(request)
        if (
            type(authority) is not RemotePatchAuthority
            or authority != self._configuration.expected_authority
            or request.headers.get("X-Avalan-Correlation")
            != authority.correlation
        ):
            raise RemotePatchServerError("remote patch operation unavailable")
        return authority

    async def tools(self, request: Request) -> _RemotePatchTools:
        """Advertise both tools only after exact authentication and binding."""
        await self.authenticate(request)
        try:
            await self.start()
        except Exception as error:
            raise RemotePatchServerError(
                "remote patch operation unavailable"
            ) from error
        return _RemotePatchTools(
            data=[
                _RemotePatchTool(
                    name="patch.edit",
                    parameters=_edit_schema(),
                ),
                _RemotePatchTool(
                    name="patch.apply",
                    parameters=_apply_schema(),
                ),
            ]
        )

    async def begin(
        self,
        request: Request,
        operation: OperationType,
        arguments: dict[str, object],
    ) -> _RemotePatchPendingResponse | _RemotePatchTerminalResponse:
        """Reserve then dispatch one exact request without retries."""
        authority = await self.authenticate(request)
        _reject_forbidden_caller_fields(arguments)
        retransmission_key = request.headers.get("Idempotency-Key")
        if not retransmission_key:
            raise RemotePatchServerError("remote patch operation unavailable")
        resolved = self._operation(authority, retransmission_key)
        raw_arguments = _canonical_arguments(arguments)
        digest = _parse_digest(
            operation,
            raw_arguments,
            resolved.correlation_id,
            self._configuration.input_limits,
        )
        try:
            reservation = await self._configuration.store.reserve(
                resolved.identity,
                digest,
                resolved.request_id,
            )
        except DurableStoreError as error:
            raise RemotePatchServerError(
                "remote patch operation unavailable"
            ) from error
        snapshot = await self._snapshot(resolved)
        if snapshot.terminal is not None:
            return _terminal_response(self._seal_operation(resolved), snapshot)
        if not _dispatch_permitted(snapshot):
            return _pending_response(self._seal_operation(resolved), snapshot)
        if reservation.replayed and not _reaped_worker_recovery(snapshot):
            return _pending_response(self._seal_operation(resolved), snapshot)
        await self.start()
        async with self._lock:
            task = self._tasks.get(resolved.request_id)
            if task is not None and task.done():
                # A completed task is only a stale coalescing entry.  Its
                # result is never recovery authority; the current durable
                # snapshot below is the sole source for another dispatch.
                self._tasks.pop(resolved.request_id, None)
                task = None
            current = await self._snapshot(resolved)
            if (
                task is None
                and _dispatch_permitted(current)
                and (
                    not reservation.replayed
                    or _reaped_worker_recovery(current)
                )
            ):
                task = create_task(
                    self._dispatch(
                        operation,
                        raw_arguments,
                        resolved,
                    )
                )
                self._tasks[resolved.request_id] = task
                task.add_done_callback(
                    partial(self._drop_dispatch_task, resolved.request_id)
                )
        return _pending_response(self._seal_operation(resolved), snapshot)

    async def inspect(
        self,
        request: Request,
        handle: str,
    ) -> _RemotePatchPendingResponse | _RemotePatchTerminalResponse:
        """Read one original operation without accepting replacement input."""
        authority = await self.authenticate(request)
        operation = self._open_operation(handle, authority)
        snapshot = await self._snapshot(operation)
        sealed = self._seal_operation(operation)
        return (
            _terminal_response(sealed, snapshot)
            if snapshot.terminal is not None
            else _pending_response(sealed, snapshot)
        )

    async def await_result(
        self,
        request: Request,
        handle: str,
    ) -> _RemotePatchPendingResponse | _RemotePatchTerminalResponse:
        """Wait within the fixed server bound without cancelling work."""
        authority = await self.authenticate(request)
        operation = self._open_operation(handle, authority)
        for _ in range(_AWAIT_POLLS):
            snapshot = await self._snapshot(operation)
            if snapshot.terminal is not None:
                return _terminal_response(
                    self._seal_operation(operation), snapshot
                )
            await sleep(_POLL_SECONDS)
        snapshot = await self._snapshot(operation)
        return (
            _terminal_response(self._seal_operation(operation), snapshot)
            if snapshot.terminal is not None
            else _pending_response(self._seal_operation(operation), snapshot)
        )

    async def cancel_intent(
        self,
        request: Request,
        handle: str,
    ) -> _RemotePatchPendingResponse | _RemotePatchTerminalResponse:
        """Persist intent only; never cancel, retry, or roll back a commit."""
        authority = await self.authenticate(request)
        operation = self._open_operation(handle, authority)
        snapshot = await self._snapshot(operation)
        if snapshot.terminal is None and snapshot.pending is not None:
            try:
                await self._configuration.store.request_cancellation(
                    operation.access
                )
            except DurableStoreError as error:
                raise RemotePatchServerError(
                    "remote patch operation unavailable"
                ) from error
            snapshot = await self._snapshot(operation)
        sealed = self._seal_operation(operation)
        return (
            _terminal_response(sealed, snapshot)
            if snapshot.terminal is not None
            else _pending_response(sealed, snapshot)
        )

    async def events(
        self,
        request: Request,
        handle: str,
        after: int,
    ) -> StreamingResponse:
        """Stream at-least-once durable event records from one cursor."""
        authority = await self.authenticate(request)
        operation = self._open_operation(handle, authority)
        if type(after) is not int or after < 0:
            raise RemotePatchServerError("remote patch operation unavailable")
        return StreamingResponse(
            self._event_stream(operation, SequenceNumber(after)),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _dispatch(
        self,
        operation: OperationType,
        raw_arguments: bytes,
        value: _RemotePatchOperation,
    ) -> None:
        """Dispatch once and reconcile durable truth without blind retries."""
        host = self._host
        if host is None:
            return
        try:
            await host.invoke_remote_raw(
                operation,
                raw_arguments,
                value.request_id,
                value.correlation_id,
                value.identity,
            )
        except CancelledError:
            raise
        except Exception:
            # Durable inspection is the only reconciliation source.  A server
            # or transport fault must not trigger another provider/target call.
            try:
                await self._snapshot(value)
            except (DurableStoreError, RemotePatchServerError):
                return

    def _drop_dispatch_task(
        self,
        request_id: PatchRequestId,
        task: Future[None],
    ) -> None:
        """Discard only the completed dispatch task this controller owns."""
        if self._tasks.get(request_id) is task:
            self._tasks.pop(request_id)

    async def _snapshot(
        self, operation: _RemotePatchOperation
    ) -> DurableRequestSnapshot:
        """Read exact durable truth through the operation-bound authority."""
        try:
            return await self._configuration.store.inspect(operation.access)
        except DurableStoreError as error:
            raise RemotePatchServerError(
                "remote patch operation unavailable"
            ) from error

    def _operation(
        self,
        authority: RemotePatchAuthority,
        retransmission_key: str,
    ) -> _RemotePatchOperation:
        """Derive server-owned durable identities from the trusted tuple."""
        try:
            key = RetransmissionKey(retransmission_key)
        except Exception as error:
            raise RemotePatchServerError(
                "remote patch operation unavailable"
            ) from error
        material = dumps(
            {
                "authority": authority.canonical(),
                "retransmission_key": key.value,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        request_id = PatchRequestId(
            "request_"
            + sha256(self._configuration.handle_key + material).hexdigest()[
                :32
            ]
        )
        correlation = PatchObserverCorrelationId(
            "correlation_"
            + sha256(material + self._configuration.handle_key).hexdigest()[
                :32
            ]
        )
        execution_id = PatchExecutionId(
            "execution_" + sha256(request_id.value.encode()).hexdigest()[:32]
        )
        return _RemotePatchOperation(
            authority,
            request_id,
            correlation,
            DurableRequestIdentity(
                authority.tenant,
                authority.principal,
                execution_id,
                authority.route,
                key,
            ),
        )

    def _seal_operation(self, operation: _RemotePatchOperation) -> str:
        """Return an encrypted opaque operation handle."""
        return self._fernet.encrypt(operation.to_bytes()).decode("ascii")

    def _open_operation(
        self,
        handle: str,
        authority: RemotePatchAuthority,
    ) -> _RemotePatchOperation:
        """Open one exact current-authority operation handle or coarsen it."""
        if not isinstance(handle, str) or len(handle) > 8192:
            raise RemotePatchServerError("remote patch operation unavailable")
        try:
            payload = loads(self._fernet.decrypt(handle.encode("ascii")))
        except (
            InvalidToken,
            UnicodeError,
            JSONDecodeError,
            ValueError,
        ) as error:
            raise RemotePatchServerError(
                "remote patch operation unavailable"
            ) from error
        if not isinstance(payload, dict) or set(payload) != {
            "authority",
            "correlation_id",
            "identity",
            "request_id",
            "version",
        }:
            raise RemotePatchServerError("remote patch operation unavailable")
        if payload["version"] != _OPERATION_HANDLE_VERSION:
            raise RemotePatchServerError("remote patch operation unavailable")
        operation = self._operation(
            authority,
            _handle_retransmission_key(payload),
        )
        if (
            payload["authority"] != authority.canonical()
            or payload["request_id"] != operation.request_id.value
            or payload["correlation_id"] != operation.correlation_id.value
            or payload["identity"]
            != {
                "execution_id": operation.identity.execution_id.value,
                "principal_id": operation.identity.principal_id.value,
                "retransmission_key": (
                    operation.identity.retransmission_key.value
                ),
                "route_id": operation.identity.route_id.value,
                "tenant_id": operation.identity.tenant_id.value,
            }
        ):
            raise RemotePatchServerError("remote patch operation unavailable")
        return operation

    async def _event_stream(
        self,
        operation: _RemotePatchOperation,
        cursor: SequenceNumber,
    ) -> AsyncIterator[str]:
        """Yield monotonic durable records without acknowledging delivery."""
        current = cursor
        while True:
            try:
                records = await self._configuration.store.outbox(
                    operation.access,
                    current,
                    _MAX_EVENT_PAGE,
                )
            except DurableStoreError:
                return
            for record in records:
                if (
                    record.sequence.value <= current.value
                    or record.request_id != operation.request_id
                    or record.correlation_id != operation.correlation_id
                ):
                    return
                current = record.sequence
                payload: dict[str, object] = {
                    "cursor": record.sequence.value,
                    "event_id": record.event_id.value,
                    "lifecycle": record.lifecycle.value,
                    "object": "patch.event",
                }
                if record.lifecycle is LifecyclePhase.REQUEST_COMPLETED:
                    snapshot = await self._snapshot(operation)
                    if (
                        snapshot.terminal is None
                        or snapshot.pending is not None
                    ):
                        return
                yield _sse(record.sequence, payload)
            snapshot = await self._snapshot(operation)
            if snapshot.terminal is not None:
                return
            try:
                await sleep(_POLL_SECONDS)
            except CancelledError:
                raise


def install_remote_patch_test_routes(
    app: FastAPI,
    configuration: RemotePatchTestServerConfiguration,
    *,
    controller: RemotePatchController | None = None,
) -> RemotePatchController:
    """Install only the explicit authenticated local test-server routes."""
    if type(app) is not FastAPI:
        raise RemotePatchServerError("remote patch configuration invalid")
    controller = controller or RemotePatchController(configuration)
    app.state.remote_patch_test_input_bytes = (
        configuration.input_limits.max_raw_bytes
    )
    router = APIRouter(
        prefix=configuration.prefix, route_class=_RemotePatchRoute
    )

    @router.get("/tools", response_model=_RemotePatchTools)
    async def tools(request: Request) -> _RemotePatchTools | JSONResponse:
        """List patch tools only for the exact authenticated test authority."""
        try:
            return await controller.tools(request)
        except RemotePatchServerError:
            return _tools_absent_response()

    @router.post(
        "/edit",
        response_model=_RemotePatchPendingResponse
        | _RemotePatchTerminalResponse,
    )
    async def edit(
        body: _RemotePatchEdit,
        request: Request,
    ) -> (
        _RemotePatchPendingResponse
        | _RemotePatchTerminalResponse
        | JSONResponse
    ):
        """Start or attach the closed structured edit operation."""
        arguments: dict[str, object] = body.model_dump(mode="python")
        return await _begin_response(
            controller,
            request,
            OperationType.EDIT,
            arguments,
        )

    @router.post(
        "/apply",
        response_model=_RemotePatchPendingResponse
        | _RemotePatchTerminalResponse,
    )
    async def apply(
        body: _RemotePatchApply,
        request: Request,
    ) -> (
        _RemotePatchPendingResponse
        | _RemotePatchTerminalResponse
        | JSONResponse
    ):
        """Start or attach the closed complete patch-document operation."""
        arguments = body.model_dump(mode="python")
        return await _begin_response(
            controller,
            request,
            OperationType.APPLY,
            arguments,
        )

    @router.get(
        "/operations/{handle}",
        response_model=_RemotePatchPendingResponse
        | _RemotePatchTerminalResponse,
    )
    async def inspect(
        handle: str,
        request: Request,
    ) -> (
        _RemotePatchPendingResponse
        | _RemotePatchTerminalResponse
        | JSONResponse
    ):
        """Inspect only an original authority-bound opaque operation."""
        try:
            return await controller.inspect(request, handle)
        except RemotePatchServerError:
            return _error_response(404)

    @router.post(
        "/operations/{handle}/await",
        response_model=_RemotePatchPendingResponse
        | _RemotePatchTerminalResponse,
    )
    async def await_result(
        handle: str,
        request: Request,
    ) -> (
        _RemotePatchPendingResponse
        | _RemotePatchTerminalResponse
        | JSONResponse
    ):
        """Await one operation without replacing or cancelling it."""
        try:
            return await controller.await_result(request, handle)
        except RemotePatchServerError:
            return _error_response(404)

    @router.post(
        "/operations/{handle}/cancel",
        response_model=_RemotePatchPendingResponse
        | _RemotePatchTerminalResponse,
    )
    async def cancel_intent(
        handle: str,
        request: Request,
    ) -> (
        _RemotePatchPendingResponse
        | _RemotePatchTerminalResponse
        | JSONResponse
    ):
        """Record cancellation intent without cancelling a transport task."""
        try:
            return await controller.cancel_intent(request, handle)
        except RemotePatchServerError:
            return _error_response(404)

    @router.get("/operations/{handle}/events", response_model=None)
    async def events(
        handle: str,
        request: Request,
        after: int = 0,
    ) -> StreamingResponse | JSONResponse:
        """Resume at-least-once durable SSE from an inclusive caller cursor."""
        try:
            return await controller.events(request, handle, after)
        except RemotePatchServerError:
            return _error_response(404)

    app.include_router(router)
    app.state.remote_patch_test_controller = controller
    return controller


def remote_patch_test_server(
    configuration: RemotePatchTestServerConfiguration,
) -> FastAPI:
    """Build one explicit loopback-only test application."""
    controller = RemotePatchController(configuration)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await controller.start()
        try:
            yield
        finally:
            await controller.close()

    app = FastAPI(lifespan=lifespan)
    install_remote_patch_test_routes(
        app,
        configuration,
        controller=controller,
    )
    return app


def install_remote_patch_test_routes_for_controller(
    app: FastAPI,
    controller: RemotePatchController,
) -> None:
    """Install routes for one already-owned controller lifetime."""
    install_remote_patch_test_routes(
        app,
        controller._configuration,
        controller=controller,
    )


async def _begin_response(
    controller: RemotePatchController,
    request: Request,
    operation: OperationType,
    arguments: dict[str, object],
) -> _RemotePatchPendingResponse | _RemotePatchTerminalResponse | JSONResponse:
    """Project all begin failures to the one closed non-oracular response."""
    try:
        return await controller.begin(request, operation, arguments)
    except RemotePatchServerError:
        return _error_response(404)


def _canonical_arguments(arguments: dict[str, object]) -> bytes:
    """Return canonical transport bytes without carrying authority controls."""
    try:
        return dumps(
            arguments,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, UnicodeError, ValueError) as error:
        raise RemotePatchServerError(
            "remote patch operation unavailable"
        ) from error


def _expected_runtime_witness(
    authority: RemotePatchAuthority,
) -> RemotePatchRuntimeWitness:
    """Convert the configured remote authority into its exact runtime seal."""
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


def _dispatch_permitted(snapshot: DurableRequestSnapshot) -> bool:
    """Allow initial work or reaped-worker recovery from durable facts only."""
    lifecycle = getattr(snapshot, "lifecycle", None)
    if lifecycle in {
        LifecyclePhase.COMMIT_STARTED,
        LifecyclePhase.SETTLEMENT_PENDING,
    }:
        return _reaped_worker_recovery(snapshot)
    return snapshot.terminal is None and lifecycle not in {
        LifecyclePhase.SETTLED,
        LifecyclePhase.REQUEST_COMPLETED,
    }


def _reaped_worker_recovery(snapshot: DurableRequestSnapshot) -> bool:
    """Allow recovery only after durable proof that its worker was reaped."""
    return bool(getattr(snapshot, "worker_reaped", False))


def _parse_digest(
    operation: OperationType,
    raw_arguments: bytes,
    correlation_id: PatchObserverCorrelationId,
    limits: PatchInputLimits,
) -> AlgorithmDigest:
    """Parse canonical input before durable reservation and inspection."""
    kind = (
        RawPatchInputKind.EDIT_JSON
        if operation is OperationType.EDIT
        else RawPatchInputKind.APPLY_JSON
    )
    try:
        request = PatchRequestParser(limits).parse(
            RawPatchIngress(
                RawProviderProfile("remote-patch-test-server"),
                RawToolCallId(correlation_id.value),
                kind,
                RawPatchInputState.COMPLETE,
                raw_arguments,
            )
        )
    except PatchInputError as error:
        raise RemotePatchServerError(
            "remote patch operation unavailable"
        ) from error
    return request.digest


def _reject_forbidden_caller_fields(value: object) -> None:
    """Reject authority-like caller fields before durable or target work."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = "".join(
                character
                for character in str(key).lower()
                if character.isalnum()
            )
            if (
                normalized in _FORBIDDEN_CALLER_FIELDS
                or remote_runtime_authority_key(key)
            ):
                raise RemotePatchServerError(
                    "remote patch operation unavailable"
                )
            _reject_forbidden_caller_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_caller_fields(item)


def _pending_response(
    handle: str,
    snapshot: DurableRequestSnapshot,
) -> _RemotePatchPendingResponse:
    """Return a nonterminal response without fabricating commit truth."""
    cursor = snapshot.pending.event_cursor.value if snapshot.pending else 0
    return _RemotePatchPendingResponse(
        operation_handle=handle,
        event_cursor=cursor,
    )


def _terminal_response(
    handle: str,
    snapshot: DurableRequestSnapshot,
) -> _RemotePatchTerminalResponse:
    """Return terminal lifecycle only when no pending record remains."""
    terminal = snapshot.terminal
    if terminal is None or snapshot.pending is not None:
        raise RemotePatchServerError("remote patch operation unavailable")
    return _RemotePatchTerminalResponse(
        operation_handle=handle,
        event_cursor=terminal.outbox.sequence.value,
    )


def _edit_schema() -> dict[str, object]:
    """Return the closed public edit schema without private configuration."""
    return {
        "additionalProperties": False,
        "properties": {
            "edits": {
                "items": {
                    "additionalProperties": False,
                    "properties": {
                        "new_text": {"type": "string"},
                        "old_text": {"minLength": 1, "type": "string"},
                    },
                    "required": ["old_text", "new_text"],
                    "type": "object",
                },
                "minItems": 1,
                "type": "array",
            },
            "path": {"type": "string"},
        },
        "required": ["path", "edits"],
        "type": "object",
    }


def _apply_schema() -> dict[str, object]:
    """Return the closed public complete patch-document schema."""
    return {
        "additionalProperties": False,
        "properties": {"patch": {"type": "string"}},
        "required": ["patch"],
        "type": "object",
    }


def _handle_retransmission_key(payload: dict[str, object]) -> str:
    """Read one opaque-handle retransmission key without a status oracle."""
    identity = payload.get("identity")
    if not isinstance(identity, dict) or set(identity) != {
        "execution_id",
        "principal_id",
        "retransmission_key",
        "route_id",
        "tenant_id",
    }:
        raise RemotePatchServerError("remote patch operation unavailable")
    value = identity.get("retransmission_key")
    if not isinstance(value, str):
        raise RemotePatchServerError("remote patch operation unavailable")
    return value


def _is_loopback(request: Request) -> bool:
    """Return whether transport reached this route from a loopback peer."""
    client = request.client
    if client is None:
        return False
    try:
        return ip_address(client.host).is_loopback
    except ValueError:
        return client.host == "localhost"


def _sse(sequence: SequenceNumber, payload: dict[str, object]) -> str:
    """Encode one monotonic at-least-once SSE frame with stable identity."""
    data = dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"id: {sequence.value}\nevent: patch.lifecycle\ndata: {data}\n\n"


def _error_response(status_code: int) -> JSONResponse:
    """Return the sole coarse remote operation error without an oracle."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": "patch.operation_unavailable",
                "message": "Patch operation unavailable.",
            }
        },
    )


def _tools_absent_response() -> JSONResponse:
    """Return no tool advertisement for unauthenticated or invalid profiles."""
    return JSONResponse(
        status_code=200, content={"object": "list", "data": []}
    )
