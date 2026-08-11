"""Expose the capability-bound public patch tool integration.

The semantic patch pipeline deliberately lives behind a small typed host
protocol.  The public tools only accept the two normative JSON forms; trusted
scope, target, approval, coordination, persistence, and disclosure state are
bound before this module is constructed and never come from a model call.
"""

from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    Queue,
    Task,
    create_task,
    current_task,
    gather,
    get_running_loop,
    sleep,
    wait,
)
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from inspect import currentframe
from json import dumps
from types import MappingProxyType, TracebackType
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from avalan._patch_authority import (
    _PatchAuthorityValidator,
)
from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallOutcome,
    ToolCapabilities,
    ToolDomainApprovalKind,
    ToolDomainExecutionContract,
    ToolDomainInputKind,
    ToolDomainParallelismKind,
    ToolDomainPendingKind,
    ToolDomainProjectionKind,
    ToolDomainRetryKind,
    ToolManagerSettings,
)
from avalan.patch.domain import (
    ApprovalMode,
    Audience,
    Capability,
    ContextKind,
    DurationTicks,
    OperationType,
    PatchInvocationOutcome,
    PatchLifecycleEvent,
    PatchLimits,
    PatchObserverCorrelationId,
    PatchPending,
    PatchRequestId,
    PatchResult,
    coarsen_error_code,
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
from avalan.patch.policy import TrustedPatchPolicy
from avalan.patch.target import ResolvedMutationScope, TargetHandshake
from avalan.tool import Tool, ToolSet
from avalan.tool.display import (
    patch_tool_call_display_projection,
    patch_tool_outcome_display_projection,
)
from avalan.tool.manager import ToolManager
from avalan.tool.names import matches_tool_namespace

PATCH_EDIT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "patch.edit",
        "description": "Edit exact text ranges in one existing file.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "path": {"type": "string"},
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "old_text": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "new_text": {"type": "string"},
                        },
                        "required": ["old_text", "new_text"],
                    },
                },
            },
            "required": ["path", "edits"],
        },
        "return": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "enum": ["patch_result"]},
                "status": {"type": "string"},
                "mutation_state": {"type": "string"},
                "lineage_state": {"type": "string"},
                "requested_effect_occurred": {"type": "string"},
                "artifact_state": {"type": "string"},
                "commit_set_exact": {"type": "boolean"},
                "workspace_changed": {"type": "string"},
                "postcondition": {"type": "string"},
                "lifecycle": {
                    "type": "string",
                    "enum": ["request_completed"],
                },
                "code": {"type": ["string", "null"]},
            },
            "required": [
                "kind",
                "status",
                "mutation_state",
                "lineage_state",
                "requested_effect_occurred",
                "artifact_state",
                "commit_set_exact",
                "workspace_changed",
                "postcondition",
                "lifecycle",
                "code",
            ],
        },
    },
}

_SettlementValue = TypeVar("_SettlementValue")

PATCH_APPLY_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "patch.apply",
        "description": "Apply one complete versioned multi-file patch.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"patch": {"type": "string"}},
            "required": ["patch"],
        },
        "return": PATCH_EDIT_SCHEMA["function"]["return"],
    },
}


def _freeze_schema(value: object) -> object:
    """Freeze every nested public-schema container for private reuse."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_schema(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_schema(item) for item in value)
    return value


def _copy_frozen_schema(value: object) -> object:
    """Return one fresh mutable JSON-compatible schema tree."""
    if isinstance(value, Mapping):
        return {key: _copy_frozen_schema(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy_frozen_schema(item) for item in value]
    return value


_PATCH_EDIT_SCHEMA_TEMPLATE = _freeze_schema(PATCH_EDIT_SCHEMA)
_PATCH_APPLY_SCHEMA_TEMPLATE = _freeze_schema(PATCH_APPLY_SCHEMA)
PATCH_EDIT_SCHEMA = cast(
    dict[str, Any],
    _copy_frozen_schema(_PATCH_EDIT_SCHEMA_TEMPLATE),
)
PATCH_APPLY_SCHEMA = cast(
    dict[str, Any],
    _copy_frozen_schema(_PATCH_APPLY_SCHEMA_TEMPLATE),
)

_PATCH_DOMAIN_EXECUTION_CONTRACT = ToolDomainExecutionContract(
    input_kind=ToolDomainInputKind.STRICT_RAW_JSON,
    approval_kind=ToolDomainApprovalKind.SEALED_PLAN,
    retry_kind=ToolDomainRetryKind.DOMAIN,
    parallelism_kind=ToolDomainParallelismKind.COORDINATOR,
    pending_kind=ToolDomainPendingKind.HOST,
    projection_kind=ToolDomainProjectionKind.DOMAIN,
)


def _project_patch_tool_display(*args: object, **kwargs: object) -> object:
    """Project only closed patch display metadata from a call or outcome."""
    del kwargs
    if len(args) == 1:
        call = args[0]
        if isinstance(call, ToolCall):
            return patch_tool_call_display_projection(call)
    if len(args) == 2:
        outcome = args[1]
        if isinstance(outcome, ToolCallOutcome):
            return patch_tool_outcome_display_projection(outcome)
    return None


class PatchToolError(ValueError):
    """Report a content-free patch public-boundary failure."""


class PatchAdmissionDecision(StrEnum):
    """Name the closed admission outcomes available before parsing."""

    ALLOW = "allow"
    SUPPRESS = "suppress"


@dataclass(frozen=True, slots=True)
class PatchAdmissionView:
    """Expose only a name and random correlation to an admission filter."""

    tool_name: str
    correlation_id: PatchObserverCorrelationId

    def __post_init__(self) -> None:
        """Require exactly one canonical public patch name."""
        if self.tool_name not in {"patch.edit", "patch.apply"}:
            raise PatchToolError("patch admission tool name is invalid")


class PatchAdmissionFilter(Protocol):
    """Admit or suppress one content-free patch tool invocation."""

    async def admit(self, view: PatchAdmissionView) -> PatchAdmissionDecision:
        """Return the closed admission decision for a content-free view."""


@runtime_checkable
class PatchSettlementPort(Protocol):
    """Expose fenced service-owned settlement observations."""

    def inspect(
        self, handle: "PatchInvocationHandle"
    ) -> Future[PatchInvocationOutcome]:
        """Return the service-owned future for one current observation."""

    def await_terminal(
        self,
        handle: "PatchInvocationHandle",
        pending: PatchPending,
    ) -> Future[PatchResult]:
        """Return the service-owned future for one durable settlement."""


@runtime_checkable
class PatchSdkService(Protocol):
    """Describe the trusted host operations backing public patch tools."""

    @property
    def settlement(self) -> PatchSettlementPort:
        """Return the fenced service-owned settlement operation port."""

    async def invoke(
        self,
        operation: OperationType,
        raw_arguments: bytes,
        capability: "PatchInvocationCapability",
        request_id: PatchRequestId,
        correlation_id: PatchObserverCorrelationId,
    ) -> PatchInvocationOutcome:
        """Run a fully bound patch pipeline from complete raw JSON input."""

    async def review(
        self, handle: "PatchInvocationHandle"
    ) -> dict[str, object]:
        """Return the privileged review projection for the current request."""

    async def approve(
        self, handle: "PatchInvocationHandle"
    ) -> PatchInvocationOutcome:
        """Resolve only the host-owned sealed-plan approval state."""

    def subscribe(
        self, handle: "PatchInvocationHandle"
    ) -> AsyncIterator[PatchLifecycleEvent]:
        """Yield only content-free semantic lifecycle events."""


@dataclass(frozen=True, slots=True)
class PatchTestHostProfile:
    """Require the explicit authenticated local profile for activation."""

    enabled: bool = False
    authenticated: bool = False

    def __post_init__(self) -> None:
        """Reject all non-boolean activation witnesses."""
        if (
            type(self.enabled) is not bool
            or type(self.authenticated) is not bool
        ):
            raise PatchToolError("patch test-host profile is invalid")


@dataclass(frozen=True, slots=True)
class PatchApprovalBinding:
    """Witness one probed host-owned sealed-plan approval service."""

    ready: bool

    def __post_init__(self) -> None:
        """Reject unavailable or untyped approval bindings."""
        if type(self.ready) is not bool or not self.ready:
            raise PatchToolError("patch approval binding is not ready")


@dataclass(frozen=True, slots=True)
class PatchCoordinatorBinding:
    """Witness one probed host-owned patch coordinator service."""

    ready: bool

    def __post_init__(self) -> None:
        """Reject unavailable or untyped coordinator bindings."""
        if type(self.ready) is not bool or not self.ready:
            raise PatchToolError("patch coordinator binding is not ready")


@dataclass(frozen=True, slots=True)
class PatchPersistenceBinding:
    """Witness one probed host-owned patch persistence service."""

    ready: bool

    def __post_init__(self) -> None:
        """Reject unavailable or untyped persistence bindings."""
        if type(self.ready) is not bool or not self.ready:
            raise PatchToolError("patch persistence binding is not ready")


@dataclass(frozen=True, slots=True)
class PatchRuntimeBinding:
    """Carry the complete already-probed trusted public host handshake."""

    scope: ResolvedMutationScope
    handshake: TargetHandshake
    policy: TrustedPatchPolicy
    approval: PatchApprovalBinding
    coordinator: PatchCoordinatorBinding
    persistence: PatchPersistenceBinding
    service: PatchSdkService

    def __post_init__(self) -> None:
        """Require target, policy, approval, coordinator, and store binding."""
        if (
            type(self.scope) is not ResolvedMutationScope
            or type(self.handshake) is not TargetHandshake
            or type(self.policy) is not TrustedPatchPolicy
            or type(self.approval) is not PatchApprovalBinding
            or type(self.coordinator) is not PatchCoordinatorBinding
            or type(self.persistence) is not PatchPersistenceBinding
            or not isinstance(self.service, PatchSdkService)
            or not isinstance(self.service.settlement, PatchSettlementPort)
        ):
            raise PatchToolError("patch runtime handshake is incomplete")


@runtime_checkable
class PatchRuntimeBinder(Protocol):
    """Probe and bind the complete patch runtime before tool construction."""

    async def bind(self) -> PatchRuntimeBinding:
        """Return one immutable, complete, already-probed runtime binding."""


@dataclass(frozen=True, slots=True)
class PatchToolSettings:
    """Configure the explicit trusted host integration for patch tools."""

    binder: PatchRuntimeBinder
    profile: PatchTestHostProfile

    def __post_init__(self) -> None:
        """Require one typed binder and a closed activation profile."""
        if (
            not isinstance(self.binder, PatchRuntimeBinder)
            or type(self.profile) is not PatchTestHostProfile
        ):
            raise PatchToolError("patch tool settings are invalid")


@dataclass(frozen=True, slots=True)
class PatchToolManagerBundle:
    """Return the selected manager and its context-bound patch toolset."""

    manager: ToolManager
    toolset: "PatchToolSet | None"


def _seal_patch_authority() -> tuple[object, ...]:
    """Seal all authority issuance state into loader and toolset closures."""

    @dataclass(frozen=True, slots=True)
    class _Reservation:
        """Bind one one-shot loader construction to its task context."""

        token: object
        service: object
        snapshot: object
        task: object

    @dataclass(frozen=True, slots=True)
    class _CapabilityToken:
        """Keep a one-shot marker and epoch private to construction."""

        marker: object
        epoch: object

    @dataclass(slots=True)
    class _CapabilityIssue:
        """Bind one opaque capability epoch to its host and owner."""

        marker: object
        epoch: object
        service: object
        owner: object
        snapshot: object
        capability: object | None = None
        active: bool = True

    @dataclass(frozen=True, slots=True)
    class _PatchToolsetRegistration:
        """Store one exact loader-owned public toolset witness."""

        owner: object
        service: object
        capability: object
        tools: tuple[tuple[str, object], ...]

        def owns(self, canonical_name: str, tool: object) -> bool:
            """Return whether this witness owns the named exact callable."""
            return (canonical_name, tool) in self.tools

    @dataclass(frozen=True, slots=True)
    class _InvocationIssue:
        """Bind one request-local handle to capability and correlation."""

        handle: object
        capability: object
        service: object
        operation: object
        correlation: object
        request_id: PatchRequestId

    reservations: list[_Reservation] = []
    capabilities: list[_CapabilityIssue] = []
    registrations: list[_PatchToolsetRegistration] = []
    invocations: list[_InvocationIssue] = []
    active_reservation: ContextVar[_Reservation | None] = ContextVar(
        "patch_loader_reservation",
        default=None,
    )

    def caller_is(code: object) -> bool:
        """Return whether one sealed helper has the required direct caller."""
        frame = currentframe()
        return (
            frame is not None
            and frame.f_back is not None
            and frame.f_back.f_back is not None
            and frame.f_back.f_back.f_code is code
        )

    def sealed_capability_is_issued(
        capability: object,
        service: object,
        owner: object | None = None,
    ) -> bool:
        """Return whether the exact capability has a sealed issuance record."""
        return any(
            issue.capability is capability
            and issue.active
            and issue.marker is getattr(capability, "_issuer", None)
            and issue.epoch is getattr(capability, "_epoch", None)
            and issue.service is service
            and (owner is None or issue.owner is owner)
            for issue in capabilities
        )

    def sealed_capability_token_is_available(capability: object) -> bool:
        """Return whether a constructor sees one unbound issued token."""
        return any(
            issue.capability is None
            and issue.active
            and issue.marker is getattr(capability, "_issuer", None)
            and issue.epoch is getattr(capability, "_epoch", None)
            for issue in capabilities
        )

    def capability_is_issued(
        capability: object,
        service: object,
        owner: object | None = None,
    ) -> bool:
        """Read whether one exact capability has a sealed issuance record."""
        return sealed_capability_is_issued(capability, service, owner)

    def capability_snapshot(
        capability: object,
        service: object,
    ) -> object | None:
        """Return only the frozen snapshot for one exact active capability."""
        for issue in capabilities:
            if (
                issue.capability is capability
                and issue.active
                and issue.service is service
            ):
                return issue.snapshot
        return None

    def issue_invocation(
        capability: object,
        service: object,
        operation: OperationType,
        request_id: PatchRequestId,
        correlation: PatchObserverCorrelationId,
    ) -> "PatchInvocationHandle":
        """Issue an opaque handle for one exact SDK request correlation."""
        if not sealed_capability_is_issued(capability, service):
            raise PatchToolError("patch invocation capability is invalid")
        handle = PatchInvocationHandle(object())
        invocations.append(
            _InvocationIssue(
                handle,
                capability,
                service,
                operation,
                correlation,
                request_id,
            )
        )
        return handle

    def bind_invocation(
        handle: "PatchInvocationHandle",
        outcome: PatchInvocationOutcome,
    ) -> None:
        """Bind an issued handle to the exact returned request identity."""
        for index, issue in enumerate(invocations):
            if issue.handle is not handle:
                continue
            if outcome.request_id != issue.request_id:
                raise PatchToolError("patch invocation request is invalid")
            if (
                isinstance(outcome, PatchPending)
                and outcome.correlation_id is not issue.correlation
            ):
                raise PatchToolError("patch invocation correlation is invalid")
            invocations[index] = _InvocationIssue(
                issue.handle,
                issue.capability,
                issue.service,
                issue.operation,
                issue.correlation,
                issue.request_id,
            )
            return
        raise PatchToolError("patch invocation handle is invalid")

    def resume_invocation(
        capability: object,
        service: object,
        pending: PatchPending,
    ) -> "PatchInvocationHandle":
        """Recover only an issued handle for its exact durable pending item."""
        for issue in invocations:
            if (
                issue.capability is capability
                and issue.service is service
                and issue.request_id == pending.request_id
                and issue.correlation is pending.correlation_id
            ):
                return cast(PatchInvocationHandle, issue.handle)
        raise PatchToolError("patch pending handle is invalid")

    def invocation_is_issued(
        handle: object,
        capability: object,
        service: object,
    ) -> bool:
        """Return whether a handle remains bound to one active capability."""
        return sealed_capability_is_issued(capability, service) and any(
            issue.handle is handle
            and issue.capability is capability
            and issue.service is service
            for issue in invocations
        )

    def invocation_matches_event(
        handle: object,
        event: object,
    ) -> bool:
        """Return whether an event belongs to an issued request handle."""
        return isinstance(event, PatchLifecycleEvent) and any(
            issue.handle is handle
            and event.request_id == issue.request_id
            and event.correlation_id is issue.correlation
            for issue in invocations
        )

    def sealed_registration_is_issued(
        registration: object,
        owner: object,
    ) -> bool:
        """Return whether the exact toolset witness belongs to its owner."""
        return any(
            issued is registration
            and issued.owner is owner
            and sealed_capability_is_issued(
                issued.capability,
                issued.service,
            )
            for issued in registrations
        )

    def registration_is_issued(registration: object, owner: object) -> bool:
        """Read whether one exact toolset witness belongs to its owner."""
        return sealed_registration_is_issued(registration, owner)

    def sealed_registration_owns(
        registration: object,
        canonical_name: str,
        tool: object,
    ) -> bool:
        """Return whether one sealed witness owns the exact named tool."""
        return any(
            issued is registration and issued.owns(canonical_name, tool)
            for issued in registrations
        )

    def registration_owns(
        registration: object,
        canonical_name: str,
        tool: object,
    ) -> bool:
        """Read whether one sealed witness owns the exact named tool."""
        return sealed_registration_owns(registration, canonical_name, tool)

    authority_validator = cast(Any, _PatchAuthorityValidator)
    authority_validator.capability_is_issued = staticmethod(
        capability_is_issued
    )
    authority_validator.registration_is_issued = staticmethod(
        registration_is_issued
    )
    authority_validator.registration_owns = staticmethod(registration_owns)
    authority_validator.capability_snapshot = staticmethod(capability_snapshot)

    def reserve(service: object, snapshot: object) -> _Reservation:
        """Record one loader-private construction reservation."""
        if not caller_is(loader_load.__code__):
            raise PatchToolError(
                "patch reservation requires the trusted loader"
            )
        task = current_task()
        assert task is not None
        reservation = _Reservation(object(), service, snapshot, task)
        reservations.append(reservation)
        return reservation

    def discard(reservation: _Reservation) -> None:
        """Remove one unconsumed reservation after its loader scope exits."""
        if not caller_is(loader_load.__code__):
            raise PatchToolError(
                "patch reservation requires the trusted loader"
            )
        for index, current in enumerate(reservations):
            if current is reservation:
                del reservations[index]
                return

    def claim(
        service: object,
        snapshot: object,
        owner: object,
    ) -> _CapabilityToken | None:
        """Consume the task-bound exact reservation and issue one marker."""
        if not caller_is(toolset_init.__code__):
            raise PatchToolError("patch issuance requires the trusted loader")
        reservation = active_reservation.get()
        if (
            not isinstance(reservation, _Reservation)
            or reservation.service is not service
            or reservation.snapshot is not snapshot
            or reservation.task is not current_task()
        ):
            return None
        index = next(
            index
            for index, current in enumerate(reservations)
            if current is reservation
        )
        del reservations[index]
        token = _CapabilityToken(object(), object())
        capabilities.append(
            _CapabilityIssue(
                token.marker,
                token.epoch,
                service,
                owner,
                snapshot,
            )
        )
        return token

    def bind_capability(
        capability: object,
        token: _CapabilityToken,
        owner: object,
    ) -> None:
        """Bind the constructed object to its unrepeatable capability epoch."""
        if not caller_is(toolset_init.__code__):
            raise PatchToolError("patch issuance requires the trusted loader")
        matches = [
            issue
            for issue in capabilities
            if (
                issue.marker is token.marker
                and issue.epoch is token.epoch
                and issue.owner is owner
                and issue.capability is None
                and issue.active
            )
        ]
        assert len(matches) == 1, "patch invocation capability is invalid"
        matches[0].capability = capability

    def revoke(marker: object, owner: object) -> None:
        """Remove an issuance that failed before construction completed."""
        if not caller_is(toolset_init.__code__):
            raise PatchToolError("patch issuance requires the trusted loader")
        for index, issue in enumerate(capabilities):
            if (
                isinstance(marker, _CapabilityToken)
                and issue.marker is marker.marker
                and issue.epoch is marker.epoch
                and issue.owner is owner
            ):
                del capabilities[index]
                return

    def revoke_active(capability: object, owner: object) -> None:
        """Revoke one complete capability epoch when its owner exits."""
        for issue in capabilities:
            if issue.capability is capability and issue.owner is owner:
                issue.active = False
                return
        raise PatchToolError("patch invocation capability is invalid")

    def register(
        registration: _PatchToolsetRegistration,
        owner: object,
    ) -> None:
        """Record one exact registration from a sealed toolset method only."""
        if not (
            caller_is(toolset_init.__code__)
            or caller_is(with_enabled_tools.__code__)
        ):
            raise PatchToolError(
                "patch registration requires the trusted loader"
            )
        assert registration.owner is owner
        registrations.append(registration)

    async def loader_load(
        self: "PatchToolLoader",
        *,
        enable_tools: list[str] | None,
        ordinary_toolsets: Sequence[ToolSet] = (),
        settings: ToolManagerSettings | None = None,
    ) -> PatchToolManagerBundle:
        """Bind once and construct a manager without inventory-time probing."""
        if isinstance(enable_tools, str):
            raise PatchToolError("patch tool selection is invalid")
        selects_patch = enable_tools is not None and any(
            matches_tool_namespace("patch.edit", selector)
            or matches_tool_namespace("patch.apply", selector)
            for selector in enable_tools
        )
        if not selects_patch:
            return PatchToolManagerBundle(
                ToolManager.create_instance(
                    available_toolsets=ordinary_toolsets,
                    enable_tools=enable_tools,
                    settings=settings,
                ),
                None,
            )
        if not (self._profile.enabled and self._profile.authenticated):
            raise PatchToolError("patch activation requires local test host")
        binding = await self._binder.bind()
        snapshot = _snapshot_for_binding(binding)
        reservation = reserve(binding.service, snapshot)
        context_token = active_reservation.set(reservation)
        try:
            toolset = PatchToolSet(binding.service, snapshot)
        finally:
            active_reservation.reset(context_token)
            discard(reservation)
        manager = ToolManager.create_instance(
            available_toolsets=(*ordinary_toolsets, toolset),
            enable_tools=enable_tools,
            settings=settings,
        )
        return PatchToolManagerBundle(manager, toolset)

    def capability_post_init(self: "PatchInvocationCapability") -> None:
        """Require one exact loader-issued capability marker."""
        if not sealed_capability_token_is_available(self):
            raise PatchToolError("patch invocation capability is invalid")

    def toolset_init(
        self: "PatchToolSet",
        service: PatchSdkService,
        snapshot: "PatchCapabilitySnapshot",
        *,
        admission_filter: PatchAdmissionFilter | None = None,
        admission_timeout_seconds: float = 1.0,
        owned_resources: Sequence[AbstractAsyncContextManager[object]] = (),
    ) -> None:
        """Bind one already-probed host, inventory, and async resources."""
        if (
            type(snapshot) is not PatchCapabilitySnapshot
            or type(admission_timeout_seconds) not in {int, float}
            or isinstance(admission_timeout_seconds, bool)
            or admission_timeout_seconds <= 0
        ):
            raise PatchToolError("patch toolset configuration is invalid")
        resources = tuple(owned_resources)
        for resource in resources:
            if not isinstance(
                resource, AbstractAsyncContextManager
            ) or hasattr(resource, "__enter__"):
                raise PatchToolError("patch resources must be async-only")
        token = claim(service, snapshot, self)
        if token is None:
            raise PatchToolError(
                "patch toolset construction requires the trusted loader"
            )
        registration: _PatchToolsetRegistration | None = None
        try:
            self._service = service
            self._snapshot = snapshot
            self._admission_filter = admission_filter
            self._admission_timeout_seconds = float(admission_timeout_seconds)
            self._owned_resources = resources
            self._capability_owner = self
            self._capability = PatchInvocationCapability(
                token.marker,
                token.epoch,
            )
            bind_capability(self._capability, token, self)
            resolved_tools = (
                ("patch.edit", _PatchEditTool(self)),
                ("patch.apply", _PatchApplyTool(self)),
            )
            names = snapshot.tool_names()
            tools = [tool for name, tool in resolved_tools if name in names]
            ToolSet.__init__(self, namespace="patch", tools=tools)
            self._all_tools = resolved_tools
            registration = _PatchToolsetRegistration(
                self,
                service,
                self._capability,
                tuple((name, tool) for name, tool in resolved_tools),
            )
            self._registration = registration
            register(registration, self)
        except BaseException:
            revoke(token, self)
            raise

    def with_enabled_tools(
        self: "PatchToolSet", enable_tools: list[str]
    ) -> "PatchToolSet":
        """Return a fresh selection without refreshing state."""
        if not registration_is_issued(
            getattr(self, "_registration", None), self
        ):
            raise PatchToolError("patch toolset registration is invalid")
        selected_tools = self._selected_tools(enable_tools)
        selected = object.__new__(PatchToolSet)
        selected._service = self._service
        selected._snapshot = self._snapshot
        selected._admission_filter = self._admission_filter
        selected._admission_timeout_seconds = self._admission_timeout_seconds
        selected._owned_resources = self._owned_resources
        selected._capability_owner = self._capability_owner
        selected._capability = self._capability
        selected._all_tools = self._all_tools
        ToolSet.__init__(selected, namespace="patch", tools=selected_tools)
        registration = _PatchToolsetRegistration(
            selected,
            selected._service,
            selected._capability,
            tuple(
                (name, tool)
                for name, tool in selected._all_tools
                if tool in selected_tools
            ),
        )
        selected._registration = registration
        register(registration, selected)
        return cast(PatchToolSet, selected)

    def toolset_revoke(self: "PatchToolSet") -> None:
        """Revoke this toolset's capability epoch on close or rebuild."""
        revoke_active(self._capability, self._capability_owner)

    return (
        loader_load,
        capability_post_init,
        toolset_init,
        with_enabled_tools,
        active_reservation,
        discard,
        reserve,
        claim,
        register,
        revoke,
        bind_capability,
        toolset_revoke,
        issue_invocation,
        bind_invocation,
        resume_invocation,
        invocation_is_issued,
        invocation_matches_event,
    )


(
    _sealed_loader_load,
    _sealed_capability_post_init,
    _sealed_toolset_init,
    _sealed_with_enabled_tools,
    _sealed_active_reservation,
    _sealed_discard,
    _sealed_reserve,
    _sealed_claim,
    _sealed_register,
    _sealed_revoke,
    _sealed_bind_capability,
    _sealed_toolset_revoke,
    _sealed_issue_invocation,
    _sealed_bind_invocation,
    _sealed_resume_invocation,
    _sealed_invocation_is_issued,
    _sealed_invocation_matches_event,
) = _seal_patch_authority()
del _seal_patch_authority


def _bind_sealed_patch_methods(
    loader_load: Callable[..., Awaitable[PatchToolManagerBundle]],
    capability_post_init: Callable[[object], None],
    toolset_init: Callable[..., None],
    with_enabled_tools: Callable[..., object],
    active_reservation: ContextVar[object | None],
    discard: Callable[[object], None],
    reserve: Callable[[object, object], object],
    claim: Callable[[object, object, object], object | None],
    register: Callable[[object, object], None],
    revoke: Callable[[object, object], None],
    bind_capability: Callable[[object, object, object], None],
    toolset_revoke: Callable[[object], None],
) -> tuple[object, object, object, object, object]:
    """Bind public methods to the sealed authority closure state."""

    async def bound_loader_load(
        self: object,
        *,
        enable_tools: list[str] | None,
        ordinary_toolsets: Sequence[ToolSet] = (),
        settings: ToolManagerSettings | None = None,
    ) -> PatchToolManagerBundle:
        """Construct a manager through the sealed loader implementation."""
        assert isinstance(active_reservation, ContextVar)
        assert callable(discard)
        assert callable(reserve)
        return await loader_load(
            self,
            enable_tools=enable_tools,
            ordinary_toolsets=ordinary_toolsets,
            settings=settings,
        )

    def bound_capability_post_init(self: object) -> None:
        """Validate one capability through the sealed issuance records."""
        capability_post_init(self)

    def bound_toolset_init(
        self: object,
        service: object,
        snapshot: object,
        *,
        admission_filter: PatchAdmissionFilter | None = None,
        admission_timeout_seconds: float = 1.0,
        owned_resources: Sequence[AbstractAsyncContextManager[object]] = (),
    ) -> None:
        """Construct one toolset through the sealed issuance records."""
        assert callable(claim)
        assert callable(register)
        assert callable(revoke)
        assert callable(bind_capability)
        toolset_init(
            self,
            service,
            snapshot,
            admission_filter=admission_filter,
            admission_timeout_seconds=admission_timeout_seconds,
            owned_resources=owned_resources,
        )

    def bound_with_enabled_tools(
        self: object,
        enable_tools: list[str],
    ) -> object:
        """Select tools through the sealed registration implementation."""
        return with_enabled_tools(self, enable_tools)

    def bound_toolset_revoke(self: object) -> None:
        """Revoke one toolset through the sealed capability epoch."""
        toolset_revoke(self)

    return (
        bound_loader_load,
        bound_capability_post_init,
        bound_toolset_init,
        bound_with_enabled_tools,
        bound_toolset_revoke,
    )


(
    _bound_loader_load,
    _bound_capability_post_init,
    _bound_toolset_init,
    _bound_with_enabled_tools,
    _bound_toolset_revoke,
) = _bind_sealed_patch_methods(
    cast(
        Callable[..., Awaitable[PatchToolManagerBundle]], _sealed_loader_load
    ),
    cast(Callable[[object], None], _sealed_capability_post_init),
    cast(Callable[..., None], _sealed_toolset_init),
    cast(Callable[..., object], _sealed_with_enabled_tools),
    cast(ContextVar[object | None], _sealed_active_reservation),
    cast(Callable[[object], None], _sealed_discard),
    cast(Callable[[object, object], object], _sealed_reserve),
    cast(Callable[[object, object, object], object | None], _sealed_claim),
    cast(Callable[[object, object], None], _sealed_register),
    cast(Callable[[object, object], None], _sealed_revoke),
    cast(Callable[[object, object, object], None], _sealed_bind_capability),
    cast(Callable[[object], None], _sealed_toolset_revoke),
)
del _bind_sealed_patch_methods


def _bind_sealed_sdk_handles(
    issue_invocation: Callable[..., object],
    bind_invocation: Callable[..., None],
    resume_invocation: Callable[..., object],
    invocation_is_issued: Callable[..., bool],
    invocation_matches_event: Callable[..., bool],
) -> tuple[
    Callable[
        [
            object,
            object,
            OperationType,
            PatchRequestId,
            PatchObserverCorrelationId,
        ],
        object,
    ],
    Callable[[object, PatchInvocationOutcome], None],
    Callable[[object, object, PatchPending], object],
    Callable[[object, object, object], bool],
    Callable[[object, object], bool],
]:
    """Bind SDK request-handle operations to sealed authority records."""

    def issue(
        capability: object,
        service: object,
        operation: OperationType,
        request_id: PatchRequestId,
        correlation: PatchObserverCorrelationId,
    ) -> object:
        """Issue one handle without exposing registry state."""
        return issue_invocation(
            capability,
            service,
            operation,
            request_id,
            correlation,
        )

    def bind(handle: object, outcome: PatchInvocationOutcome) -> None:
        """Bind one handle to its returned request identity."""
        bind_invocation(handle, outcome)

    def resume(
        capability: object,
        service: object,
        pending: PatchPending,
    ) -> object:
        """Resume only a matching issued pending handle."""
        return resume_invocation(capability, service, pending)

    def issued(handle: object, capability: object, service: object) -> bool:
        """Validate one handle against its active capability epoch."""
        return invocation_is_issued(handle, capability, service)

    def matches(handle: object, event: object) -> bool:
        """Validate one lifecycle event against its request handle."""
        return invocation_matches_event(handle, event)

    return issue, bind, resume, issued, matches


(
    _bound_issue_invocation,
    _bound_bind_invocation,
    _bound_resume_invocation,
    _bound_invocation_is_issued,
    _bound_invocation_matches_event,
) = _bind_sealed_sdk_handles(
    cast(Callable[..., object], _sealed_issue_invocation),
    cast(Callable[..., None], _sealed_bind_invocation),
    cast(Callable[..., object], _sealed_resume_invocation),
    cast(Callable[..., bool], _sealed_invocation_is_issued),
    cast(Callable[..., bool], _sealed_invocation_matches_event),
)
del _bind_sealed_sdk_handles


class PatchToolLoader:
    """Construct patch tools only after one complete async handshake."""

    def __init__(
        self,
        binder: PatchRuntimeBinder,
        profile: PatchTestHostProfile,
    ) -> None:
        """Bind the runtime loader and test-host activation."""
        if (
            not isinstance(binder, PatchRuntimeBinder)
            or type(profile) is not PatchTestHostProfile
        ):
            raise PatchToolError("patch tool loader is invalid")
        self._binder = binder
        self._profile = profile

    load = cast(
        Callable[..., Awaitable[PatchToolManagerBundle]],
        _bound_loader_load,
    )

    async def rebuild_if_stale(
        self,
        toolset: "PatchToolSet",
        *,
        enable_tools: list[str],
        ordinary_toolsets: Sequence[ToolSet] = (),
    ) -> PatchToolManagerBundle:
        """Rebind a stale patch inventory through the async host loader."""
        if type(toolset) is not PatchToolSet or not toolset.snapshot_stale:
            raise PatchToolError("patch toolset is not stale")
        toolset._revoke(toolset)
        return await self.load(
            enable_tools=enable_tools,
            ordinary_toolsets=ordinary_toolsets,
        )


@dataclass(frozen=True, slots=True, repr=False)
class PatchInvocationCapability:
    """Bind one immutable trusted patch host to a tool-call context."""

    _issuer: object
    _epoch: object

    def __post_init__(self) -> None:
        """Validate the loader-issued capability after initialization."""
        validator = cast(
            Callable[[PatchInvocationCapability], None],
            _bound_capability_post_init,
        )
        validator(self)

    def __copy__(self) -> "PatchInvocationCapability":
        """Refuse copies that could detach trusted call authority."""
        raise TypeError("patch invocation capability is not copyable")

    def __deepcopy__(self, memo: object) -> "PatchInvocationCapability":
        """Refuse deep copies that could detach trusted call authority."""
        del memo
        raise TypeError("patch invocation capability is not copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        """Refuse serialization of trusted patch invocation authority."""
        del protocol
        raise TypeError("patch invocation capability is not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class PatchInvocationHandle:
    """Represent one sealed request-local SDK lifecycle binding."""

    _issuer: object

    def __copy__(self) -> "PatchInvocationHandle":
        """Refuse copies that could detach request-local lifecycle state."""
        raise TypeError("patch invocation handle is not copyable")

    def __deepcopy__(self, memo: object) -> "PatchInvocationHandle":
        """Refuse deep copies that detach request-local lifecycle state."""
        del memo
        raise TypeError("patch invocation handle is not copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        """Refuse serialization of trusted request-local lifecycle state."""
        del protocol
        raise TypeError("patch invocation handle is not serializable")


@dataclass(frozen=True, slots=True)
class PatchCapabilitySnapshot:
    """Store one already-probed immutable public capability inventory."""

    edit_available: bool
    apply_available: bool
    provider_verified_freeform: bool = False
    stale: bool = False
    policy_revision: str = "test-host"
    input_limits: PatchInputLimits = PatchInputLimits()
    settlement_duration: DurationTicks = DurationTicks(60_000)

    def __post_init__(self) -> None:
        """Keep provider freeform optional and disabled by default."""
        if (
            type(self.edit_available) is not bool
            or type(self.apply_available) is not bool
            or type(self.provider_verified_freeform) is not bool
            or type(self.stale) is not bool
            or not isinstance(self.policy_revision, str)
            or not self.policy_revision
            or type(self.input_limits) is not PatchInputLimits
            or type(self.settlement_duration) is not DurationTicks
        ):
            raise PatchToolError("patch capability snapshot is invalid")
        if self.provider_verified_freeform:
            raise PatchToolError("patch freeform is not activated")

    def tool_names(self) -> tuple[str, ...]:
        """Return the frozen advertised canonical names without probing."""
        if self.stale:
            return ()
        names: list[str] = []
        if self.edit_available:
            names.append("patch.edit")
        if self.apply_available:
            names.append("patch.apply")
        return tuple(names)

    def permits(self, operation: OperationType) -> bool:
        """Return whether the frozen snapshot permits one exact operation."""
        if self.stale:
            return False
        return (operation is OperationType.EDIT and self.edit_available) or (
            operation is OperationType.APPLY and self.apply_available
        )


class _PatchEditTool(Tool):
    """Execute one exact file edit through a trusted patch capability.

    Args:
        path: Relative path of one existing file.
        edits: Exact non-overlapping replacement declarations.

    Returns:
        Content-free terminal patch result projection.
    """

    tool_capabilities = ToolCapabilities(
        supports_streaming=False, side_effecting=True, parallel_safe=False
    )
    domain_execution_contract = _PATCH_DOMAIN_EXECUTION_CONTRACT
    tool_display_projector = staticmethod(_project_patch_tool_display)

    def __init__(self, owner: "PatchToolSet") -> None:
        """Bind the immutable public toolset owner."""
        Tool.__init__(self)
        self.__name__ = "edit"
        self._owner = owner

    def json_schema(self, prefix: str | None = None) -> dict[str, Any]:
        """Return the frozen closed public edit schema."""
        del prefix
        return cast(
            dict[str, Any],
            _copy_frozen_schema(_PATCH_EDIT_SCHEMA_TEMPLATE),
        )

    async def __call__(
        self,
        path: str,
        edits: list[dict[str, str]],
        context: ToolCallContext,
    ) -> dict[str, object]:
        """Execute one exact edit through the bound trusted host.

        Args:
            path: Relative path of one existing file.
            edits: Exact old and new text values.

        Returns:
            Content-free terminal patch result projection.
        """
        return await self._owner.invoke_json(
            OperationType.EDIT, {"path": path, "edits": edits}, context
        )

    async def invoke_raw(
        self, raw_arguments: bytes, context: ToolCallContext
    ) -> dict[str, object]:
        """Execute raw provider JSON after patch-specific classification.

        Args:
            raw_arguments: Exact complete provider JSON argument bytes.

        Returns:
            Content-free terminal patch result projection.
        """
        capability = context.patch_capability
        if capability is None:
            return _failure_projection("patch.capability_unavailable")
        return await self._owner.invoke_raw(
            OperationType.EDIT, raw_arguments, capability
        )


class _PatchApplyTool(Tool):
    """Execute one complete versioned patch through a trusted capability.

    Args:
        patch: Complete Version 1 patch-language document.

    Returns:
        Content-free terminal patch result projection.
    """

    tool_capabilities = ToolCapabilities(
        supports_streaming=False, side_effecting=True, parallel_safe=False
    )
    domain_execution_contract = _PATCH_DOMAIN_EXECUTION_CONTRACT
    tool_display_projector = staticmethod(_project_patch_tool_display)

    def __init__(self, owner: "PatchToolSet") -> None:
        """Bind the immutable public toolset owner."""
        Tool.__init__(self)
        self.__name__ = "apply"
        self._owner = owner

    def json_schema(self, prefix: str | None = None) -> dict[str, Any]:
        """Return the frozen closed public apply schema."""
        del prefix
        return cast(
            dict[str, Any],
            _copy_frozen_schema(_PATCH_APPLY_SCHEMA_TEMPLATE),
        )

    async def __call__(
        self, patch: str, context: ToolCallContext
    ) -> dict[str, object]:
        """Execute one complete patch through the bound trusted host.

        Args:
            patch: Complete Version 1 patch-language document.

        Returns:
            Content-free terminal patch result projection.
        """
        return await self._owner.invoke_json(
            OperationType.APPLY, {"patch": patch}, context
        )

    async def invoke_raw(
        self, raw_arguments: bytes, context: ToolCallContext
    ) -> dict[str, object]:
        """Execute raw provider JSON after patch-specific classification.

        Args:
            raw_arguments: Exact complete provider JSON argument bytes.

        Returns:
            Content-free terminal patch result projection.
        """
        capability = context.patch_capability
        if capability is None:
            return _failure_projection("patch.capability_unavailable")
        return await self._owner.invoke_raw(
            OperationType.APPLY, raw_arguments, capability
        )


class PatchToolSet(ToolSet):
    """Expose exactly the capability-bound public patch tools."""

    _service: PatchSdkService
    _snapshot: PatchCapabilitySnapshot
    _admission_filter: PatchAdmissionFilter | None
    _admission_timeout_seconds: float
    _owned_resources: tuple[AbstractAsyncContextManager[object], ...]
    _capability: PatchInvocationCapability
    _capability_owner: object
    _all_tools: tuple[tuple[str, Tool], ...]
    _registration: object
    _revoke = staticmethod(
        cast(Callable[["PatchToolSet"], None], _bound_toolset_revoke)
    )
    __init__ = cast(Callable[..., None], _bound_toolset_init)

    def __setattr__(self, name: str, value: object) -> None:
        """Revoke the shared epoch when a bound snapshot becomes stale."""
        if (
            name == "_snapshot"
            and isinstance(value, PatchCapabilitySnapshot)
            and value.stale
            and hasattr(self, "_capability")
        ):
            self._revoke(self)
        super().__setattr__(name, value)

    @property
    def available_tools(self) -> tuple[Tool, ...]:
        """Return the complete frozen available inventory without effects."""
        names = self._snapshot.tool_names()
        return tuple(tool for name, tool in self._all_tools if name in names)

    @property
    def capability(self) -> PatchInvocationCapability:
        """Return the immutable capability to place in a trusted context."""
        return self._capability

    def _patch_registration(self) -> object:
        """Return the private exact ownership witness for manager binding."""
        return self._registration

    @property
    def snapshot_stale(self) -> bool:
        """Return whether asynchronous reconstruction is required."""
        return self._snapshot.stale

    def available_tools_for_enabled_tools(
        self, enable_tools: Sequence[str]
    ) -> tuple[Tool, ...]:
        """Read selected inventory without probing or mutating the toolset."""
        return self._selected_tools(enable_tools)

    def advertised_tools_for_enabled_tools(
        self, enable_tools: Sequence[str]
    ) -> tuple[Tool, ...]:
        """Read advertised inventory without probes or mutation."""
        return self._selected_tools(enable_tools)

    with_enabled_tools = cast(
        Callable[..., "PatchToolSet"],
        _bound_with_enabled_tools,
    )

    async def __aenter__(self) -> "PatchToolSet":
        """Enter only explicitly owned async resources and child tools."""
        try:
            for resource in self._owned_resources:
                await self._exit_stack.enter_async_context(resource)
            for tool in self.tools:
                await self._exit_stack.enter_async_context(cast(Tool, tool))
        except BaseException:
            try:
                await self._exit_stack.aclose()
            finally:
                self._revoke(self)
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Close owned resources and revoke the active capability epoch."""
        try:
            return await ToolSet.__aexit__(
                self,
                exc_type,
                exc_value,
                traceback,
            )
        finally:
            self._revoke(self)

    async def invoke_json(
        self,
        operation: OperationType,
        arguments: dict[str, object],
        context: ToolCallContext,
    ) -> dict[str, object]:
        """Encode canonical JSON only after trusted patch classification."""
        if operation not in {OperationType.EDIT, OperationType.APPLY}:
            raise PatchToolError("patch operation is invalid")
        if not self._snapshot.permits(operation):
            return _failure_projection("patch.capability_unavailable")
        capability = context.patch_capability
        if (
            capability is not self._capability
            or not self._capability_is_active()
        ):
            return _failure_projection("patch.capability_unavailable")
        try:
            raw = dumps(
                arguments,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, UnicodeError, ValueError):
            return _failure_projection("patch.invalid_request")
        return await self.invoke_raw(operation, raw, capability)

    async def invoke_raw(
        self,
        operation: OperationType,
        raw_arguments: bytes,
        capability: PatchInvocationCapability,
    ) -> dict[str, object]:
        """Run pre-parse admission then return only a terminal model result."""
        if (
            capability is not self._capability
            or type(raw_arguments) is not bytes
            or operation not in {OperationType.EDIT, OperationType.APPLY}
            or not self._snapshot.permits(operation)
            or not self._capability_is_active()
        ):
            return _failure_projection("patch.capability_unavailable")
        tool_name = (
            "patch.edit" if operation is OperationType.EDIT else "patch.apply"
        )
        correlation = PatchObserverCorrelationId.new()
        admitted = await self._admit(tool_name, correlation)
        if not admitted:
            return _failure_projection("patch.admission_unavailable")
        if not self._valid_raw_ingress(
            operation,
            raw_arguments,
            correlation,
            self._snapshot.input_limits,
        ):
            return _failure_projection("patch.invalid_request")
        request_id = PatchRequestId.new()
        handle = cast(
            PatchInvocationHandle,
            _bound_issue_invocation(
                capability,
                self._service,
                operation,
                request_id,
                correlation,
            ),
        )
        try:
            outcome = await self._service.invoke(
                operation,
                raw_arguments,
                capability,
                request_id,
                correlation,
            )
        except CancelledError:
            raise
        except Exception as error:
            outcome = await self._reconcile_after_dispatch(
                capability,
                handle,
                correlation,
                error,
            )
        _bound_bind_invocation(handle, outcome)
        if isinstance(outcome, PatchPending):
            outcome = await self._await_terminal(handle, outcome)
        return project_model_result(outcome)

    async def _reconcile_after_dispatch(
        self,
        capability: PatchInvocationCapability,
        handle: PatchInvocationHandle,
        correlation: PatchObserverCorrelationId,
        error: Exception,
    ) -> PatchInvocationOutcome:
        """Recover truth after a host failure that may follow dispatch."""
        del error
        try:
            outcome = await _await_settlement_future(
                self._service.settlement.inspect(handle),
                self._snapshot.settlement_duration,
            )
        except CancelledError:
            raise
        except Exception as reconciliation_error:
            raise PatchToolError(
                "patch dispatch requires host reconciliation"
            ) from reconciliation_error
        if outcome is None:
            raise PatchToolError("patch reconciliation remains pending")
        if (
            isinstance(outcome, PatchPending)
            and outcome.correlation_id != correlation
        ):
            raise PatchToolError("patch reconciliation correlation is invalid")
        _bound_bind_invocation(handle, outcome)
        return outcome

    async def _await_terminal(
        self,
        handle: PatchInvocationHandle,
        pending: PatchPending,
    ) -> PatchResult:
        """Await settlement without treating cancellation as truth."""
        try:
            result = await _await_settlement_future(
                self._service.settlement.await_terminal(handle, pending),
                self._snapshot.settlement_duration,
            )
            if result is None:
                self._preserve_pending(handle, pending)
                raise PatchToolError("patch settlement remains pending")
            _bound_bind_invocation(handle, result)
            return result
        except CancelledError:
            self._preserve_pending(handle, pending)
            raise

    def _preserve_pending(
        self,
        handle: PatchInvocationHandle,
        pending: PatchPending,
    ) -> None:
        """Retain the exact durable pending envelope without cancellation."""
        _bound_bind_invocation(handle, pending)

    @staticmethod
    def _valid_raw_ingress(
        operation: OperationType,
        raw_arguments: bytes,
        correlation: PatchObserverCorrelationId,
        limits: PatchInputLimits,
    ) -> bool:
        """Validate exact raw JSON before any semantic host invocation."""
        kind = (
            RawPatchInputKind.EDIT_JSON
            if operation is OperationType.EDIT
            else RawPatchInputKind.APPLY_JSON
        )
        try:
            PatchRequestParser(limits).parse(
                RawPatchIngress(
                    RawProviderProfile("patch-toolset"),
                    RawToolCallId(correlation.value),
                    kind,
                    RawPatchInputState.COMPLETE,
                    raw_arguments,
                )
            )
        except PatchInputError:
            return False
        return True

    async def _admit(
        self, tool_name: str, correlation: PatchObserverCorrelationId
    ) -> bool:
        """Run the isolated content-free admission filter with a hard bound."""
        if self._admission_filter is None:
            return True
        admission = create_task(
            self._admission_filter.admit(
                PatchAdmissionView(tool_name, correlation)
            )
        )
        timeout = create_task(sleep(self._admission_timeout_seconds))
        try:
            completed, _ = await wait(
                (admission, timeout),
                return_when=FIRST_COMPLETED,
            )
        except BaseException:
            await _settle_admission_tasks(admission, timeout)
            raise
        if admission not in completed:
            await _settle_admission_tasks(admission, timeout)
            return False
        await _settle_admission_tasks(admission, timeout)
        try:
            decision = admission.result()
        except CancelledError:
            raise
        except Exception:
            return False
        return decision is PatchAdmissionDecision.ALLOW

    def _selected_tools(self, enable_tools: Sequence[str]) -> tuple[Tool, ...]:
        """Return one pure namespace-filtered snapshot selection."""
        if isinstance(enable_tools, str) or not isinstance(
            enable_tools, Sequence
        ):
            raise PatchToolError("patch selection must be a sequence")
        return tuple(
            tool
            for name, tool in self._all_tools
            if name in self._snapshot.tool_names()
            and any(
                matches_tool_namespace(name, enabled)
                for enabled in enable_tools
            )
        )

    def _capability_is_active(self) -> bool:
        """Return whether this owner holds the issued capability epoch."""
        return _PatchAuthorityValidator.capability_is_issued(
            self._capability,
            self._service,
            self._capability_owner,
        )


class PatchSdkHost:
    """Provide direct async host APIs without model-controlled authority."""

    def __init__(
        self, service: PatchSdkService, capability: PatchInvocationCapability
    ) -> None:
        """Bind the trusted service and immutable invocation capability."""
        snapshot = _PatchAuthorityValidator.capability_snapshot(
            capability,
            service,
        )
        if (
            type(capability) is not PatchInvocationCapability
            or not _PatchAuthorityValidator.capability_is_issued(
                capability, service
            )
            or type(snapshot) is not PatchCapabilitySnapshot
        ):
            raise PatchToolError("patch SDK capability is invalid")
        self._service = service
        self._capability = capability
        self._snapshot = snapshot
        self._handle: PatchInvocationHandle | None = None
        self._pending: PatchPending | None = None

    async def invoke_json(
        self,
        operation: OperationType,
        arguments: dict[str, object],
    ) -> PatchInvocationOutcome:
        """Invoke one trusted SDK request encoded as canonical JSON.

        Args:
            operation: The edit or apply operation to execute.
            arguments: The operation's complete JSON object.

        Returns:
            The authenticated terminal or pending SDK outcome.
        """
        if operation not in {OperationType.EDIT, OperationType.APPLY}:
            raise PatchToolError("patch operation is invalid")
        if not self._snapshot.permits(operation) or not self._is_active():
            raise PatchToolError("patch operation is unavailable")
        try:
            raw_arguments = dumps(
                arguments,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, UnicodeError, ValueError) as exc:
            raise PatchToolError("patch SDK arguments are invalid") from exc
        return await self.invoke_raw(operation, raw_arguments)

    async def invoke_raw(
        self,
        operation: OperationType,
        raw_arguments: bytes,
    ) -> PatchInvocationOutcome:
        """Invoke one complete validated raw JSON SDK request.

        Args:
            operation: The edit or apply operation to execute.
            raw_arguments: Exact complete UTF-8 JSON request bytes.

        Returns:
            The authenticated terminal or pending SDK outcome.
        """
        if operation not in {OperationType.EDIT, OperationType.APPLY}:
            raise PatchToolError("patch operation is invalid")
        if not self._snapshot.permits(operation) or not self._is_active():
            raise PatchToolError("patch operation is unavailable")
        if type(raw_arguments) is not bytes:
            raise PatchToolError("patch SDK arguments are invalid")
        correlation = PatchObserverCorrelationId.new()
        if not PatchToolSet._valid_raw_ingress(
            operation,
            raw_arguments,
            correlation,
            self._snapshot.input_limits,
        ):
            raise PatchToolError("patch SDK request is invalid")
        request_id = PatchRequestId.new()
        handle = cast(
            PatchInvocationHandle,
            _bound_issue_invocation(
                self._capability,
                self._service,
                operation,
                request_id,
                correlation,
            ),
        )
        self._handle = handle
        try:
            outcome = await self._service.invoke(
                operation,
                raw_arguments,
                self._capability,
                request_id,
                correlation,
            )
        except CancelledError:
            raise
        except Exception as error:
            outcome = await self._reconcile_after_dispatch(handle, error)
        _bound_bind_invocation(handle, outcome)
        self._pending = outcome if isinstance(outcome, PatchPending) else None
        return outcome

    async def review(self) -> dict[str, object]:
        """Return the complete host-owned review projection.

        Returns:
            Complete privileged review data selected by trusted policy.
        """
        return await self._service.review(self._require_handle())

    async def approve(self) -> PatchInvocationOutcome:
        """Resolve approval for the exact sealed request.

        Returns:
            The current terminal or pending invocation outcome.
        """
        handle = self._require_handle()
        outcome = await self._service.approve(handle)
        _bound_bind_invocation(handle, outcome)
        self._pending = outcome if isinstance(outcome, PatchPending) else None
        return outcome

    async def inspect(self) -> PatchInvocationOutcome:
        """Inspect the current authenticated invocation state.

        Returns:
            The current terminal or pending invocation outcome.
        """
        handle = self._require_handle()
        outcome = await _await_settlement_future(
            self._service.settlement.inspect(handle),
            self._snapshot.settlement_duration,
        )
        if outcome is None:
            raise PatchToolError("patch inspection remains pending")
        _bound_bind_invocation(handle, outcome)
        self._pending = outcome if isinstance(outcome, PatchPending) else None
        return outcome

    async def await_terminal(self, pending: PatchPending) -> PatchResult:
        """Await a pending request on its original trusted capability.

        Args:
            pending: Current nonterminal patch operation envelope.

        Returns:
            The one terminal result for the original request.
        """
        if type(pending) is not PatchPending:
            raise PatchToolError("patch pending handle is invalid")
        handle = cast(
            PatchInvocationHandle,
            _bound_resume_invocation(
                self._capability,
                self._service,
                pending,
            ),
        )
        if self._handle is not None and self._handle is not handle:
            raise PatchToolError("patch pending handle is invalid")
        self._handle = handle
        self._pending = pending
        self._require_handle()
        result = await _await_settlement_future(
            self._service.settlement.await_terminal(handle, pending),
            self._snapshot.settlement_duration,
        )
        if result is None:
            self._preserve_pending(handle, pending)
            raise PatchToolError("patch settlement remains pending")
        _bound_bind_invocation(handle, result)
        self._pending = None
        return result

    async def cancel(self) -> PatchInvocationOutcome:
        """Return the existing durable pending request without cancellation.

        Returns:
            The current terminal or pending invocation outcome.
        """
        handle = self._require_handle()
        pending = self._pending
        if pending is None:
            raise PatchToolError("patch cancellation is unavailable")
        _bound_bind_invocation(handle, pending)
        return pending

    async def lifecycle(self) -> AsyncIterator[PatchLifecycleEvent]:
        """Yield content-free lifecycle events for this trusted request.

        Returns:
            An asynchronous stream of semantic lifecycle events.
        """
        handle = self._require_handle()
        async for event in self._service.subscribe(handle):
            if not _bound_invocation_matches_event(handle, event):
                raise PatchToolError("patch lifecycle event is invalid")
            yield event

    def _is_active(self) -> bool:
        """Return whether the host capability remains in its issued epoch."""
        return _PatchAuthorityValidator.capability_is_issued(
            self._capability,
            self._service,
        )

    async def _reconcile_after_dispatch(
        self,
        handle: PatchInvocationHandle,
        error: Exception,
    ) -> PatchInvocationOutcome:
        """Recover service truth rather than fabricate a zero-write result."""
        del error
        try:
            outcome = await _await_settlement_future(
                self._service.settlement.inspect(handle),
                self._snapshot.settlement_duration,
            )
        except CancelledError:
            raise
        except Exception as reconciliation_error:
            raise PatchToolError(
                "patch dispatch requires host reconciliation"
            ) from reconciliation_error
        if outcome is None:
            raise PatchToolError("patch reconciliation remains pending")
        _bound_bind_invocation(handle, outcome)
        return outcome

    def _preserve_pending(
        self,
        handle: PatchInvocationHandle,
        pending: PatchPending,
    ) -> None:
        """Retain durable state after a terminal wait times out.

        Phase Nine has no service-owned fenced cancellation operation, so it
        must preserve the exact known pending envelope without starting an
        arbitrary service coroutine.
        """
        _bound_bind_invocation(handle, pending)
        self._pending = pending

    def _require_handle(self) -> PatchInvocationHandle:
        """Return the active sealed request handle for lifecycle operations."""
        handle = self._handle
        if (
            handle is None
            or not self._is_active()
            or not _bound_invocation_is_issued(
                handle,
                self._capability,
                self._service,
            )
        ):
            raise PatchToolError("patch request handle is invalid")
        return handle


class InMemoryPatchLifecycleService:
    """Offer a strict test-host lifecycle subscription adapter."""

    def __init__(self) -> None:
        """Initialize no request state and no subscriber authority."""
        self._events: list[PatchLifecycleEvent] = []
        self._subscribers: list[Queue[PatchLifecycleEvent | None]] = []

    async def emit(self, event: PatchLifecycleEvent) -> None:
        """Publish one canonical event to current test-host subscribers."""
        self._events.append(event)
        for subscriber in tuple(self._subscribers):
            await subscriber.put(event)

    async def close(self) -> None:
        """Close subscriptions without publishing a terminal event."""
        for subscriber in tuple(self._subscribers):
            await subscriber.put(None)
        self._subscribers.clear()

    async def subscribe(
        self, handle: PatchInvocationHandle
    ) -> AsyncIterator[PatchLifecycleEvent]:
        """Yield replay-safe events to an authenticated subscriber."""
        if type(handle) is not PatchInvocationHandle:
            raise PatchToolError("patch subscription handle is invalid")
        queue: Queue[PatchLifecycleEvent | None] = Queue()
        for event in self._events:
            await queue.put(event)
        self._subscribers.append(queue)
        try:
            while True:
                next_event = await queue.get()
                if next_event is None:
                    return
                yield next_event
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)


def project_model_result(outcome: PatchInvocationOutcome) -> dict[str, object]:
    """Project a terminal patch result without content-derived disclosure.

    Args:
        outcome: Canonical terminal patch outcome to project.

    Returns:
        Closed bounded model-safe result fields.
    """
    if isinstance(outcome, PatchPending):
        raise PatchToolError("pending patch outcomes are not tool results")
    diagnostic = outcome.diagnostic
    return {
        "kind": "patch_result",
        "status": outcome.status.value,
        "mutation_state": outcome.truth.mutation_state.value,
        "lineage_state": outcome.truth.lineage_state.value,
        "requested_effect_occurred": (
            outcome.truth.requested_effect_occurred.value
        ),
        "artifact_state": outcome.truth.artifact_state.value,
        "commit_set_exact": outcome.truth.commit_set_exact,
        "workspace_changed": outcome.truth.workspace_change.value,
        "postcondition": outcome.truth.postcondition.value,
        "lifecycle": outcome.lifecycle.value,
        "code": (
            None
            if diagnostic is None
            else coarsen_error_code(
                diagnostic.code,
                Audience.MODEL,
            ).value
        ),
    }


def _failure_projection(code: str) -> dict[str, object]:
    """Return a bounded generic-free precommit failure projection."""
    return {
        "kind": "patch_result",
        "status": "rejected",
        "mutation_state": "not_committed",
        "lineage_state": "not_committed",
        "requested_effect_occurred": "false",
        "artifact_state": "absent",
        "commit_set_exact": True,
        "workspace_changed": "unchanged",
        "postcondition": "unknown",
        "lifecycle": "request_completed",
        "code": code,
    }


def _snapshot_for_binding(
    binding: PatchRuntimeBinding,
) -> PatchCapabilitySnapshot:
    """Intersect frozen target and policy authority without target I/O."""
    if (
        binding.scope.context_kind is not ContextKind.LOCAL
        or binding.handshake.identity != binding.scope.identity
        or binding.handshake.identity.policy_revision
        != binding.policy.revision.value
        or not binding.handshake.supports_inspection()
    ):
        raise PatchToolError("patch runtime handshake is incompatible")
    target_operations = binding.handshake.advertised_operations()
    read_required = {
        Capability.READ_FOR_MUTATION,
        Capability.OBSERVE_MUTATION_PRECONDITIONS,
    }
    policy_modes = {
        capability: tuple(
            mode
            for rule in binding.policy.rules
            if (mode := rule.mode_for(capability)) is not None
            and mode.mode is not ApprovalMode.DENY
        )
        for capability in Capability
    }
    inspection_available = all(policy_modes[item] for item in read_required)
    edit_available = (
        OperationType.EDIT in binding.policy.enabled_operations
        and Capability.UPDATE in target_operations
        and inspection_available
        and bool(policy_modes[Capability.UPDATE])
    )
    apply_effects = {
        Capability.CREATE,
        Capability.UPDATE,
        Capability.DELETE,
        Capability.MOVE,
    }
    apply_available = (
        OperationType.APPLY in binding.policy.enabled_operations
        and inspection_available
        and any(
            capability in target_operations and policy_modes[capability]
            for capability in apply_effects
        )
    )
    return PatchCapabilitySnapshot(
        edit_available,
        apply_available,
        policy_revision=binding.policy.revision.value,
        input_limits=_effective_input_limits(binding.scope.limits),
        settlement_duration=binding.scope.limits.commit_duration,
    )


def _effective_input_limits(limits: PatchLimits) -> PatchInputLimits:
    """Derive parser ceilings that cannot exceed trusted scope limits."""
    if type(limits) is not PatchLimits:
        raise PatchToolError("patch input limits are invalid")
    input_bytes = limits.input_bytes
    path_count = limits.path_count
    path_length = limits.path_length
    operation_count = limits.operation_count
    proposed_bytes = limits.proposed_bytes
    defaults = PatchInputLimits()
    return PatchInputLimits(
        max_raw_bytes=min(defaults.max_raw_bytes, input_bytes.value),
        max_paths=min(defaults.max_paths, path_count.value),
        max_declarations=min(
            defaults.max_declarations,
            operation_count.value,
        ),
        max_hunks=min(defaults.max_hunks, operation_count.value),
        max_edits=min(defaults.max_edits, operation_count.value),
        max_path_characters=min(
            defaults.max_path_characters,
            path_length.value,
        ),
        max_path_bytes=min(defaults.max_path_bytes, path_length.value),
        max_component_characters=min(
            defaults.max_component_characters,
            path_length.value,
        ),
        max_component_bytes=min(
            defaults.max_component_bytes,
            path_length.value,
        ),
        max_content_bytes=min(
            defaults.max_content_bytes,
            proposed_bytes.value,
        ),
    )


async def _settle_admission_tasks(
    admission: Task[PatchAdmissionDecision],
    timeout: Task[None],
) -> None:
    """Cancel and join admission helpers before their owner continues."""
    if not admission.done():
        admission.cancel()
    if not timeout.done():
        timeout.cancel()
    await gather(admission, timeout, return_exceptions=True)


async def _await_settlement_future(
    settlement: Future[_SettlementValue],
    duration: DurationTicks,
) -> _SettlementValue | None:
    """Wait for a service-owned future without taking control of its worker.

    The host owns and reaps only the local timeout task.  A pending settlement
    future is intentionally left with its fenced service so cancellation cannot
    detach an arbitrary in-process coroutine or invent terminal truth.
    """
    if (
        type(settlement) is not Future
        or settlement.get_loop() is not get_running_loop()
    ):
        raise PatchToolError("patch settlement future is invalid")
    timeout = create_task(sleep(duration.value / 1_000))
    try:
        completed, _ = await wait(
            (settlement, timeout),
            return_when=FIRST_COMPLETED,
        )
    except BaseException:
        await _reap_settlement_timeout(timeout)
        raise
    await _reap_settlement_timeout(timeout)
    if settlement not in completed:
        return None
    return settlement.result()


async def _reap_settlement_timeout(timeout: Task[None]) -> None:
    """Cancel and join the host-owned finite timeout helper."""
    if not timeout.done():
        timeout.cancel()
    await gather(timeout, return_exceptions=True)
