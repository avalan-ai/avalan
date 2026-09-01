"""Exercise the public capability-bound patch tool integration."""

from asyncio import (
    CancelledError,
    Event,
    Future,
    all_tasks,
    create_task,
    current_task,
    gather,
    get_running_loop,
    run,
    sleep,
    wait_for,
)
from collections.abc import Iterator
from contextlib import AsyncExitStack
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from inspect import getclosurevars, signature
from json import dumps, loads
from logging import Logger
from os import umask
from pathlib import Path
from runpy import run_path
from subprocess import run as run_process
from sys import executable
from typing import Never, TypeVar, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from patch_activation_support import activated_patch_test_profile

import avalan.patch.target as target_module
import avalan.patch.toolset as patch_toolset_module
from avalan.agent.loader import OrchestratorLoader
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    Message,
    MessageRole,
    OrchestratorSettings,
    PreparedToolCall,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallError,
    ToolCallResult,
    ToolDescriptor,
    ToolDomainApprovalKind,
    ToolDomainInputKind,
    ToolDomainParallelismKind,
    ToolDomainPendingKind,
    ToolDomainProjectionKind,
    ToolDomainRetryKind,
    ToolManagerExecutionMode,
    ToolManagerSettings,
)
from avalan.model.capability import (
    ModelCapabilityCatalog,
    ModelCapabilityValidationError,
    ProviderCapabilityCall,
)
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.patch.activation import PatchActivationRuntime
from avalan.patch.domain import (
    ApprovalMode,
    ArtifactState,
    ByteSize,
    Capability,
    CommitTruth,
    ContextKind,
    DurationTicks,
    ErrorStage,
    LifecyclePhase,
    LineageState,
    MutationState,
    OperationType,
    PatchApprovalId,
    PatchContextId,
    PatchDiagnostic,
    PatchDomainId,
    PatchErrorCode,
    PatchEventId,
    PatchLifecycleEvent,
    PatchLimits,
    PatchObserverCorrelationId,
    PatchObserverId,
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
    Retryability,
    SequenceNumber,
    WorkspaceChange,
)
from avalan.patch.durable_store import (
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.parser import (
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderProfile,
    RawToolCallId,
)
from avalan.patch.policy import (
    ApprovalRequirements,
    CapabilityMode,
    PolicyBrokerId,
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
    PATCH_APPLY_SCHEMA,
    PATCH_EDIT_SCHEMA,
    InMemoryPatchLifecycleService,
    PatchAdmissionDecision,
    PatchAdmissionView,
    PatchApprovalBinding,
    PatchCapabilitySnapshot,
    PatchCoordinatorBinding,
    PatchInvocationCapability,
    PatchInvocationHandle,
    PatchPersistenceBinding,
    PatchRuntimeBinding,
    PatchSdkHost,
    PatchTestHostProfile,
    PatchToolError,
    PatchToolLoader,
    PatchToolSet,
    PatchToolSettings,
    project_model_result,
)
from avalan.tool import Tool, ToolSet
from avalan.tool.context import ToolSettingsContext
from avalan.tool.display import (
    TOOL_DISPLAY_PROJECTION_METADATA_KEY,
    tool_call_display_projection_from_metadata,
    tool_outcome_display_projection_from_metadata,
)
from avalan.tool.manager import ToolManager

_PHASE_NINE_TARGET_BASELINE = run_path(target_module.__file__)
_PHASE_NINE_PRODUCTION_WORKER_BOOTSTRAP = _PHASE_NINE_TARGET_BASELINE[
    "_WORKER_BOOTSTRAP"
]


_PHASE_NINE_PRODUCTION_RUNTIME_VERIFIER = _PHASE_NINE_TARGET_BASELINE[
    "_RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES"
]
assert isinstance(_PHASE_NINE_PRODUCTION_WORKER_BOOTSTRAP, str)
assert isinstance(_PHASE_NINE_PRODUCTION_RUNTIME_VERIFIER, bytes)
_SettlementTestValue = TypeVar("_SettlementTestValue")


def _settled_future(
    value: _SettlementTestValue,
) -> Future[_SettlementTestValue]:
    """Return a current-loop future already resolved to one trusted value."""
    future: Future[_SettlementTestValue] = get_running_loop().create_future()
    future.set_result(value)
    return future


class _SettlementPort:
    """Expose only service-owned effect-free settlement futures."""

    def __init__(self, service: "_Service") -> None:
        """Bind one scripted service without starting a helper task."""
        self._service = service

    def inspect(
        self, handle: PatchInvocationHandle
    ) -> Future[PatchResult | PatchPending]:
        """Return the scripted current observation future."""
        return self._service.settlement_inspection(handle)

    def await_terminal(
        self, handle: PatchInvocationHandle, pending: PatchPending
    ) -> Future[PatchResult]:
        """Return the scripted service-owned settlement future."""
        return self._service.settlement_terminal(handle, pending)


def _restore_phase_nine_target_baseline() -> None:
    """Restore the target globals required before one isolated test host."""
    target_module._WORKER_BOOTSTRAP = _PHASE_NINE_PRODUCTION_WORKER_BOOTSTRAP
    target_module._RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES = (
        _PHASE_NINE_PRODUCTION_RUNTIME_VERIFIER
    )


@pytest.fixture(autouse=True)
def _phase_nine_target_baseline() -> Iterator[None]:
    """Reset the process-global target worker before and after every test."""
    _restore_phase_nine_target_baseline()
    yield
    _restore_phase_nine_target_baseline()


@pytest.fixture(autouse=True)
def _phase_nine_file_creation_umask() -> Iterator[None]:
    """Create ordinary Phase 9 fixture files with their sealed 0644 mode."""
    previous = umask(0o022)
    try:
        yield
    finally:
        umask(previous)


def _phase_seven_test_host() -> dict[str, object]:
    """Load Phase Seven with its production worker bootstrap restored.

    Phase Seven deliberately replaces the process-global worker bootstrap with
    its test verifier.  Read the original source in an isolated namespace so
    a preceding host cannot become the next Phase Four baseline.
    """
    _restore_phase_nine_target_baseline()
    return run_path("tests/patch/phase_7_contract_test.py")


async def _phase_seven_scope(
    phase_seven: dict[str, object], profile: object
) -> object:
    """Acquire one Phase Seven test worker before resolving its scope."""
    phase_four = phase_seven["_PHASE4"]
    assert isinstance(phase_four, dict)
    bootstrap_factory = phase_four["_test_worker_bootstrap"]
    verifier = phase_four["_TEST_RUNTIME_AUTHORITY_VERIFIER_BYTES"]
    assert callable(bootstrap_factory)
    assert isinstance(verifier, bytes)
    bootstrap = bootstrap_factory()
    assert isinstance(bootstrap, str)
    target_module._WORKER_BOOTSTRAP = bootstrap
    target_module._RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES = verifier
    resolver = phase_seven["LocalScopeResolver"]
    selection = phase_seven["ScopeSelection"]
    assert callable(resolver)
    assert callable(selection)
    return await resolver(profile).resolve(selection(ContextKind.LOCAL))


def _result(request_id: PatchRequestId | None = None) -> PatchResult:
    """Return one content-free terminal result for public projection tests."""
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


def _policy(revision: str) -> TrustedPatchPolicy:
    """Return a local test-host policy with every required handshake gate."""
    preauthorization = PreauthorizationClass("phase-nine")
    return TrustedPatchPolicy(
        PolicyRevision(revision),
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
            PolicyRouteId("phase-nine-route"),
            PolicyBrokerId("phase-nine-broker"),
            PolicyReviewerRole("phase-nine-reviewer"),
            1,
            preauthorization,
        ),
    )


@dataclass
class _Service:
    """Record public service inputs without parsing or mutating a target."""

    pending: bool = False
    terminal_gate: Event | None = None

    def __post_init__(self) -> None:
        """Initialize all call observations."""
        self.invocations: list[tuple[object, bytes]] = []
        self.terminal_waits = 0
        self.request_id: PatchRequestId | None = None
        self.correlation_id: PatchObserverCorrelationId | None = None
        self.lifecycle = InMemoryPatchLifecycleService()
        self.settlement = _SettlementPort(self)
        self._activation_store = InMemoryDurablePatchStore(
            InMemoryDurablePatchBackend()
        )
        self._activation_observer: object | None = None
        self._activation_observers: list[object] = []

    def set_activation_observer(self, observer: object) -> None:
        """Retain the one loader-issued activation observer for this host."""
        if observer in self._activation_observers:
            raise RuntimeError(
                "phase nine activation observer is already bound"
            )
        self._activation_observers.append(observer)
        self._activation_observer = observer

    def settlement_inspection(
        self, handle: PatchInvocationHandle
    ) -> Future[PatchResult | PatchPending]:
        """Return the scripted terminal inspection without a host task."""
        assert isinstance(handle, PatchInvocationHandle)
        return _settled_future(_result(self.request_id))

    def settlement_terminal(
        self, handle: PatchInvocationHandle, pending: PatchPending
    ) -> Future[PatchResult]:
        """Return the scripted terminal settlement without a host task."""
        assert isinstance(handle, PatchInvocationHandle)
        assert isinstance(pending, PatchPending)
        self.terminal_waits += 1
        return _settled_future(_result(self.request_id))

    async def invoke(
        self,
        operation: object,
        raw_arguments: bytes,
        capability: PatchInvocationCapability,
        request_id: PatchRequestId,
        correlation_id: PatchObserverCorrelationId,
    ) -> PatchResult | PatchPending:
        """Record an authenticated raw input and return a scripted outcome."""
        assert isinstance(capability, PatchInvocationCapability)
        self.invocations.append((operation, raw_arguments))
        self.request_id = request_id
        self.correlation_id = correlation_id
        if self.pending:
            return PatchPending(
                1,
                PatchPendingOperationId("pending_" + "a" * 16),
                request_id,
                correlation_id,
                LifecyclePhase.SETTLEMENT_PENDING,
            )
        return _result(request_id)

    async def review(self, handle: PatchInvocationHandle) -> dict[str, object]:
        """Return a trusted review marker without content."""
        assert isinstance(handle, PatchInvocationHandle)
        return {"kind": "review"}

    async def approve(self, handle: PatchInvocationHandle) -> PatchResult:
        """Return the scripted terminal approval outcome."""
        assert isinstance(handle, PatchInvocationHandle)
        return _result(self.request_id)

    async def inspect(self, handle: PatchInvocationHandle) -> PatchResult:
        """Return the scripted terminal inspection outcome."""
        assert isinstance(handle, PatchInvocationHandle)
        return _result(self.request_id)

    async def await_terminal(
        self,
        handle: PatchInvocationHandle,
        pending: PatchPending,
    ) -> PatchResult:
        """Settle only the exact pending envelope supplied by the toolset."""
        assert isinstance(handle, PatchInvocationHandle)
        assert isinstance(pending, PatchPending)
        self.terminal_waits += 1
        if self.terminal_gate is not None:
            await self.terminal_gate.wait()
        return _result(self.request_id)

    async def cancel(self, handle: PatchInvocationHandle) -> PatchResult:
        """Return the scripted terminal cancellation observation."""
        assert isinstance(handle, PatchInvocationHandle)
        return _result(self.request_id)

    def subscribe(self, handle: PatchInvocationHandle) -> object:
        """Return the only content-free lifecycle stream."""
        assert isinstance(handle, PatchInvocationHandle)
        return self.lifecycle.subscribe(handle)


def _limits() -> PatchLimits:
    """Return finite test-host limits for an already-probed snapshot."""
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


def _binding(
    service: object,
    activation_store: InMemoryDurablePatchStore | None = None,
) -> PatchRuntimeBinding:
    """Build a complete already-probed local test-host binding."""
    if activation_store is None:
        activation_store = getattr(service, "_activation_store", None)
    assert isinstance(activation_store, InMemoryDurablePatchStore)
    identity = TargetIdentity(
        PatchContextId("context_" + "a" * 16),
        PatchWorkspaceId("workspace_" + "a" * 16),
        PatchDomainId("domain_" + "a" * 16),
        PatchTargetId("target_" + "a" * 16),
        PatchProtocolId("protocol_" + "a" * 16),
        "filesystem-nine",
        "mount-nine",
        "policy-nine",
        "lease-nine",
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
    handshake = TargetHandshake(
        identity,
        primitives,
        (),
        platform=LocalPlatformProfile.DARWIN,
    )
    return PatchRuntimeBinding(
        scope,
        handshake,
        _policy(identity.policy_revision),
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True, activation_store),
        PatchPersistenceBinding(True, activation_store),
        service,
    )


async def _toolset_async(
    service: _Service,
    snapshot: PatchCapabilitySnapshot,
    **kwargs: object,
) -> PatchToolSet:
    """Construct one test toolset only through the trusted loader path."""
    activation_store = InMemoryDurablePatchStore(InMemoryDurablePatchBackend())

    class Binder:
        """Expose the complete scripted runtime binding to the loader."""

        async def bind(self) -> PatchRuntimeBinding:
            """Return the complete local authenticated test binding."""
            return _binding(service, activation_store)

    loader = PatchToolLoader(
        Binder(),
        activated_patch_test_profile(),
    )
    bundle = await loader.load(enable_tools=["patch.edit"])
    assert bundle.toolset is not None
    toolset = bundle.toolset
    toolset._snapshot = snapshot
    toolset._tools = [
        tool
        for name, tool in toolset._all_tools
        if name in snapshot.tool_names()
    ]
    fields = {
        "admission_filter": "_admission_filter",
        "admission_timeout_seconds": "_admission_timeout_seconds",
        "owned_resources": "_owned_resources",
    }
    for name, value in kwargs.items():
        field = fields.get(name)
        if field is None:
            raise AssertionError(f"unsupported test toolset option: {name}")
        setattr(toolset, field, value)
    return toolset


def _toolset(
    service: _Service,
    snapshot: PatchCapabilitySnapshot,
    **kwargs: object,
) -> PatchToolSet:
    """Construct one test toolset outside an active event loop."""
    return run(_toolset_async(service, snapshot, **kwargs))


async def _seal_local_sdk_edit(
    phase_seven: dict[str, object],
    profile: object,
    target: object,
    scope: object,
    raw_arguments: bytes,
) -> object:
    """Seal one direct SDK edit with the real local target witness."""
    canonical = phase_seven["PatchRequestParser"](
        phase_seven["PatchInputLimits"]()
    ).parse(
        phase_seven["RawPatchIngress"](
            phase_seven["RawProviderProfile"]("phase-nine-sdk"),
            phase_seven["RawToolCallId"]("phase-nine-sdk-edit"),
            phase_seven["RawPatchInputKind"].EDIT_JSON,
            phase_seven["RawPatchInputState"].COMPLETE,
            raw_arguments,
        )
    )
    source = b"before\n"
    path = "note0.txt"
    root = profile.root._path
    planner_file = phase_seven["PlannerFile"](
        phase_seven["LogicalPath"](path),
        phase_seven["SourceBytes"](source),
        phase_seven["_metadata"](source, phase_seven["FileMode"](0o644)),
        None,
        profile.identity.mount_id,
        "identity-" + path,
        ((root / path).lstat().st_dev, (root / path).lstat().st_ino),
        (root.stat().st_dev, root.stat().st_ino),
        phase_seven["_protected_metadata"](root, path),
    )
    workspace = phase_seven["PlannerWorkspace"](
        (planner_file,),
        frozenset(),
        (
            phase_seven["PlannerParentMount"](
                None,
                profile.identity.mount_id,
                (root.stat().st_dev, root.stat().st_ino),
            ),
        ),
    )
    candidate = phase_seven["plan"](canonical, workspace)
    paths = tuple(
        sorted(
            {
                value
                for lineage in candidate.lineages
                for value in (lineage.source_path, lineage.destination_path)
                if value is not None
            },
            key=lambda value: value.value,
        )
    )
    limits = phase_seven["_limits"]()
    reader = phase_seven["PreauthorizationClass"]("phase-nine-reader")
    rule = phase_seven["PolicyRule"](
        phase_seven["PolicyPathSelector"](None),
        tuple(
            phase_seven["CapabilityMode"](
                item,
                (
                    phase_seven["ApprovalMode"].REQUIRE_REVIEW
                    if item is phase_seven["Capability"].UPDATE
                    else phase_seven["ApprovalMode"].PREAUTHORIZED
                ),
                (None if item is phase_seven["Capability"].UPDATE else reader),
            )
            for item in phase_seven["Capability"]
        ),
        atomicity_classes=frozenset(("single_step", "dependency_ordered")),
    )
    requirements = phase_seven["ApprovalRequirements"](
        phase_seven["ApprovalMode"].REQUIRE_REVIEW,
        phase_seven["PolicyRouteId"]("route-seven"),
        phase_seven["PolicyBrokerId"]("broker-seven"),
        phase_seven["PolicyReviewerRole"]("reviewer-seven"),
        1,
    )
    authorizer = phase_seven["PolicyAuthorizer"](
        phase_seven["TrustedPatchPolicy"](
            phase_seven["PolicyRevision"]("policy-six"),
            frozenset((phase_seven["OperationType"].EDIT,)),
            (rule,),
            limits,
            requirements,
        )
    )
    effects = frozenset(
        item for lineage in candidate.lineages for item in lineage.capabilities
    )
    preflight = await authorizer.authorize_preinspection(
        phase_seven["PreflightRequest"](
            phase_seven["OperationType"].EDIT,
            paths,
            effects,
            frozenset(paths),
            phase_seven["compose_limits"](
                limits, limits, limits, limits, limits
            ),
        )
    )
    final = await authorizer.authorize_final(
        preflight,
        candidate,
        await target.handshake(scope),
    )
    return phase_seven["seal_plan"](
        phase_seven["PatchPlanId"]("plan_" + "d" * 16),
        phase_seven["PlanBinding"](
            phase_seven["PatchRequest"](
                1,
                phase_seven["PatchRequestId"]("request_" + "d" * 16),
                phase_seven["PatchExecutionId"]("execution_" + "d" * 16),
                phase_seven["OperationType"].EDIT,
                phase_seven["PatchInput"](raw_arguments),
                paths,
            ),
            candidate.request_digest,
            phase_seven["ExecutionSubject"](
                phase_seven["PatchPrincipalId"]("principal-seven"),
                phase_seven["PatchTenantId"]("tenant-seven"),
                phase_seven["PatchRunId"]("run-seven"),
                phase_seven["PatchSessionId"]("session-seven"),
                phase_seven["PatchTaskId"]("task-seven"),
                phase_seven["PatchAgentId"]("agent-seven"),
            ),
            phase_seven["ContextKind"].LOCAL,
            profile.identity,
            None,
            preflight,
            final,
        ),
        candidate,
        phase_seven["ExpiryTick"](100),
    )


def test_patch_phase_9_static_public_tools_and_selection() -> None:
    """Expose only frozen schemas and namespace-independent selection."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=True),
    )
    assert tuple(tool.__name__ for tool in toolset.available_tools) == (
        "edit",
        "apply",
    )
    assert (
        PATCH_EDIT_SCHEMA["function"]["parameters"]["additionalProperties"]
        is False
    )
    assert (
        PATCH_APPLY_SCHEMA["function"]["parameters"]["additionalProperties"]
        is False
    )
    assert tuple(
        tool.__name__
        for tool in toolset.available_tools_for_enabled_tools(("patch.edit",))
    ) == ("edit",)
    assert not toolset.available_tools_for_enabled_tools(("shell.*",))
    manager = ToolManager.create_instance(
        available_toolsets=[toolset],
        enable_tools=["patch.apply"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES
        ),
    )
    descriptors = manager.list_tools()
    assert [descriptor.name for descriptor in descriptors] == ["patch.apply"]
    assert descriptors[0].capabilities.side_effecting
    assert not descriptors[0].capabilities.parallel_safe
    assert descriptors[0].domain_execution is not None
    assert descriptors[0].domain_execution.input_kind is (
        ToolDomainInputKind.STRICT_RAW_JSON
    )
    assert descriptors[0].domain_execution.approval_kind is (
        ToolDomainApprovalKind.SEALED_PLAN
    )
    assert descriptors[0].domain_execution.retry_kind is (
        ToolDomainRetryKind.DOMAIN
    )
    assert descriptors[0].domain_execution.parallelism_kind is (
        ToolDomainParallelismKind.COORDINATOR
    )
    assert descriptors[0].domain_execution.pending_kind is (
        ToolDomainPendingKind.HOST
    )
    assert descriptors[0].domain_execution.projection_kind is (
        ToolDomainProjectionKind.DOMAIN
    )


def test_patch_phase_9_schema_exports_and_tool_copies_do_not_share_state() -> (
    None
):
    """Keep public schema mutation outside every advertised tool template."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=True),
    )
    edit = next(
        tool for tool in toolset.available_tools if tool.__name__ == "edit"
    )
    first = edit.json_schema()
    first["function"]["parameters"]["properties"]["path"]["type"] = "integer"
    PATCH_EDIT_SCHEMA["function"]["parameters"]["properties"]["path"][
        "type"
    ] = "boolean"
    second = edit.json_schema()
    assert (
        second["function"]["parameters"]["properties"]["path"]["type"]
        == "string"
    )
    PATCH_EDIT_SCHEMA["function"]["parameters"]["properties"]["path"][
        "type"
    ] = "string"
    apply = next(
        tool for tool in toolset.available_tools if tool.__name__ == "apply"
    )
    apply_schema = apply.json_schema()
    assert (
        apply_schema["function"]["return"] is not second["function"]["return"]
    )
    manager = ToolManager.create_instance(
        available_toolsets=[toolset],
        enable_tools=["patch.*"],
    )
    descriptor = manager.list_tools()[0]
    assert descriptor.schema is not None
    descriptor.schema["function"]["parameters"]["properties"]["path"][
        "type"
    ] = "integer"
    assert descriptor.parameter_schema is not None
    descriptor.parameter_schema["properties"]["path"]["type"] = "boolean"
    model_seed = manager.export_model_capability_seed()
    model_seed["descriptors"][0]["parameter_schema"]["properties"]["path"][
        "type"
    ] = "number"
    refreshed_descriptor = manager.list_tools()[0]
    assert refreshed_descriptor.schema is not None
    assert refreshed_descriptor.parameter_schema is not None
    assert (
        refreshed_descriptor.schema["function"]["parameters"]["properties"][
            "path"
        ]["type"]
        == "string"
    )
    assert (
        refreshed_descriptor.parameter_schema["properties"]["path"]["type"]
        == "string"
    )
    refreshed_seed = manager.export_model_capability_seed()
    assert (
        refreshed_seed["descriptors"][0]["parameter_schema"]["properties"][
            "path"
        ]["type"]
        == "string"
    )


def test_patch_phase_9_runtime_binding_rejects_unready_approval() -> None:
    """Reject host advertisement when approval readiness is unavailable."""
    service = _Service()
    binding = _binding(service)
    with pytest.raises(PatchToolError):
        PatchApprovalBinding(False)
    with pytest.raises(PatchToolError):
        replace(binding, approval=object())  # type: ignore[arg-type]
    assert PatchToolSet is not None


def test_patch_phase_9_runtime_binding_rejects_unready_coordinator() -> None:
    """Reject host advertisement when coordinator readiness is unavailable."""
    service = _Service()
    binding = _binding(service)
    with pytest.raises(PatchToolError):
        PatchCoordinatorBinding(False)
    with pytest.raises(PatchToolError):
        replace(binding, coordinator=object())  # type: ignore[arg-type]
    assert PatchToolSet is not None


def test_patch_phase_9_runtime_binding_rejects_unready_persistence() -> None:
    """Reject host advertisement when persistence readiness is unavailable."""
    service = _Service()
    binding = _binding(service)
    with pytest.raises(PatchToolError):
        PatchPersistenceBinding(False)
    with pytest.raises(PatchToolError):
        replace(binding, persistence=object())  # type: ignore[arg-type]
    assert PatchToolSet is not None


def test_patch_phase_9_runtime_binding_requires_typed_loader_settings() -> (
    None
):
    """Reject runtime tool settings without an authenticated typed binder."""
    service = _Service()

    class Binder:
        """Return the complete immutable local test binding."""

        async def bind(self) -> PatchRuntimeBinding:
            """Return a valid ready runtime binding."""
            return _binding(service)

    assert PatchToolSet is not None
    with pytest.raises(PatchToolError):
        PatchToolSettings(
            object(),  # type: ignore[arg-type]
            activated_patch_test_profile(),
        )
    assert PatchToolSettings(
        Binder(),
        activated_patch_test_profile(),
    )


def test_patch_phase_9_orchestrator_loader_binds_ready_patch_tools() -> None:
    """Advertise patch tools only through the real async loader path."""
    service = _Service()

    class Binder:
        """Count the complete ready host binding used by the loader."""

        def __init__(self) -> None:
            """Initialize a zero-call trusted binding probe."""
            self.calls = 0

        async def bind(self) -> PatchRuntimeBinding:
            """Return one complete typed ready runtime binding."""
            self.calls += 1
            return _binding(service)

    settings = OrchestratorSettings(
        agent_id=uuid4(),
        orchestrator_type=None,
        agent_config={"role": "assistant"},
        uri="ai://local/model",
        engine_config={},
        tools=["patch.edit"],
        call_options=None,
        template_vars=None,
        memory_permanent_message=None,
        permanent_memory=None,
        memory_recent=False,
        sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
        sentence_model_engine_config=None,
        sentence_model_max_tokens=500,
        sentence_model_overlap_size=125,
        sentence_model_window_size=250,
        json_config=None,
        log_events=True,
    )
    binder = Binder()

    async def execute() -> None:
        """Drive the production loader with only trusted runtime settings."""
        stack = AsyncExitStack()
        with (
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch(
                "avalan.agent.loader.ModelManager", return_value=MagicMock()
            ),
            patch(
                "avalan.agent.loader.DefaultOrchestrator",
                return_value="orchestrator",
            ) as orchestrator,
            patch(
                "avalan.agent.loader.EventManager", return_value=MagicMock()
            ),
            patch("avalan.agent.loader.HAS_GRAPH_DEPENDENCIES", False),
            patch("avalan.agent.loader.HAS_CODE_DEPENDENCIES", False),
            patch("avalan.agent.loader.HAS_BROWSER_DEPENDENCIES", False),
            patch(
                "avalan.agent.loader.MathToolSet",
                side_effect=lambda *, namespace: ToolSet(
                    namespace=namespace,
                    tools=[],
                ),
            ),
            patch(
                "avalan.agent.loader.MemoryToolSet",
                side_effect=lambda _memory, *, namespace: ToolSet(
                    namespace=namespace,
                    tools=[],
                ),
            ),
        ):
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_settings(
                settings,
                tool_settings=ToolSettingsContext(
                    patch=PatchToolSettings(
                        binder,
                        PatchTestHostProfile(
                            enabled=True,
                            authenticated=True,
                        ),
                    )
                ),
            )
            assert result == "orchestrator"
            tool = orchestrator.call_args.args[4]
            assert tool.list_tools() == []
        await stack.aclose()

    run(execute())
    assert binder.calls == 1


def test_patch_phase_9_loader_binds_once_and_requires_test_host() -> None:
    """Construct a manager only after the complete local handshake is bound."""
    assert PatchToolSet is not None
    service = _Service()
    binding = _binding(service)
    assert binding.approval.ready is True

    class Binder:
        """Count the one explicit asynchronous runtime binding operation."""

        def __init__(self) -> None:
            """Initialize the bind-call counter."""
            self.calls = 0

        async def bind(self) -> PatchRuntimeBinding:
            """Return the immutable complete test-host binding once."""
            self.calls += 1
            return binding

    binder = Binder()

    async def execute() -> None:
        """Exercise absent, denied, and activated patch manager loading."""
        inactive = PatchToolLoader(binder, activated_patch_test_profile())
        absent = await inactive.load(enable_tools=["shell.*"])
        assert absent.toolset is None
        assert binder.calls == 0
        denied = PatchToolLoader(
            binder, PatchTestHostProfile(enabled=False, authenticated=True)
        )
        denied_bundle = await denied.load(enable_tools=["patch.*"])
        assert denied_bundle.toolset is None
        assert denied_bundle.manager.list_tools() == []
        assert binder.calls == 0
        bundle = await inactive.load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        assert [item.name for item in bundle.manager.list_tools()] == [
            "patch.edit"
        ]
        assert binder.calls == 1
        assert bundle.toolset is not None
        for _ in range(3):
            assert [
                tool.__name__ for tool in bundle.toolset.available_tools
            ] == [
                "edit",
                "apply",
            ]
            assert [item.name for item in bundle.manager.list_tools()] == [
                "patch.edit"
            ]
        assert binder.calls == 1

        denied_policy = replace(
            binding,
            policy=replace(binding.policy, enabled_operations=frozenset()),
        )

        class PolicyDeniedBinder:
            """Return a binding whose policy authorizes no mutation path."""

            async def bind(self) -> PatchRuntimeBinding:
                """Return the policy-denied but otherwise complete binding."""
                return denied_policy

        policy_denied = await PatchToolLoader(
            PolicyDeniedBinder(),
            activated_patch_test_profile(),
        ).load(enable_tools=["patch.*"])
        assert policy_denied.toolset is not None
        assert policy_denied.manager.list_tools() == []

        read_only_handshake = replace(
            binding,
            handshake=TargetHandshake(
                binding.handshake.identity,
                frozenset({TargetPrimitive.BOUNDED_READ}),
                (),
                platform=LocalPlatformProfile.DARWIN,
            ),
        )

        class ReadOnlyBinder:
            """Return a complete target that grants no mutation primitive."""

            async def bind(self) -> PatchRuntimeBinding:
                """Return the read-only trusted target handshake."""
                return read_only_handshake

        with pytest.raises(PatchToolError):
            await PatchToolLoader(
                ReadOnlyBinder(),
                activated_patch_test_profile(),
            ).load(enable_tools=["patch.*"])
        assert tuple(
            tool.__name__ for tool in bundle.toolset.available_tools
        ) == (
            "edit",
            "apply",
        )
        assert binder.calls == 1

    run(execute())


def test_patch_phase_9_manager_bypasses_generic_hooks_and_confirmation() -> (
    None
):
    """Keep generic hooks and confirmation out of patch authority."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    calls: list[str] = []

    def forbidden(*arguments: object) -> object:
        """Record any generic callback that incorrectly observes patch data."""
        del arguments
        calls.append("generic")
        raise AssertionError("patch must bypass generic callback")

    manager = ToolManager.create_instance(
        available_toolsets=[toolset],
        enable_tools=["patch.edit"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES,
            filters=[forbidden],
            transformers=[forbidden],
        ),
    )

    async def execute() -> None:
        """Run a model-shaped edit call through the manager."""
        outcome = await manager.execute_call(
            ToolCall(
                id="call-phase-nine",
                name="patch.edit",
                raw_arguments=(
                    b'{"path":"note.txt","edits":['
                    b'{"old_text":"old","new_text":"new"}]}'
                ),
            ),
            ToolCallContext(patch_capability=toolset.capability),
            confirm=forbidden,
        )
        assert isinstance(outcome, ToolCallResult)
        assert outcome.result == {
            "kind": "patch_result",
            "status": "committed",
            "mutation_state": "committed",
            "lineage_state": "committed",
            "requested_effect_occurred": "true",
            "artifact_state": "absent",
            "commit_set_exact": True,
            "workspace_changed": "changed",
            "postcondition": "established",
            "lifecycle": "request_completed",
            "code": None,
        }

    run(execute())
    assert calls == []
    assert service.invocations
    assert (
        service.invocations[0][1]
        == b'{"path":"note.txt","edits":[{"old_text":"old","new_text":"new"}]}'
    )


def test_patch_phase_9_generic_filter_cannot_rewrite_ordinary_call_to_patch() -> (  # noqa: E501
    None
):
    """Seal an ordinary canonical domain before every generic filter hook."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )

    async def read() -> dict[str, str]:
        """Return one ordinary read result."""
        return {"state": "read"}

    read.__name__ = "read"

    def rewrite(
        call: ToolCall,
        context: ToolCallContext,
    ) -> tuple[ToolCall, ToolCallContext]:
        """Attempt to smuggle captured authority through a filter."""
        assert context.patch_capability is None
        return (
            ToolCall(
                id=call.id,
                name="patch.edit",
                raw_arguments=(
                    b'{"path":"note.txt","edits":['
                    b'{"old_text":"old","new_text":"new"}]}'
                ),
            ),
            ToolCallContext(patch_capability=toolset.capability),
        )

    manager = ToolManager.create_instance(
        available_toolsets=[toolset, ToolSet(namespace="shell", tools=[read])],
        enable_tools=["patch.edit", "shell.read"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES,
            filters=[rewrite],
        ),
    )

    async def execute() -> None:
        """Reject the attempted ordinary-to-patch domain transition."""
        result = await manager.execute_call(
            ToolCall(id="filter-rewrite", name="shell.read"),
            ToolCallContext(patch_capability=toolset.capability),
        )
        assert isinstance(result, ToolCallDiagnostic)
        assert result.code is ToolCallDiagnosticCode.FILTER_SUPPRESSED

    run(execute())
    assert service.invocations == []


def test_patch_phase_9_manager_defensive_patch_boundaries() -> None:
    """Reject malformed or cross-boundary patch manager state."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )

    async def read() -> dict[str, str]:
        """Return one ordinary result after a failed boundary transition."""
        return {"state": "current"}

    read.__name__ = "read"
    manager = ToolManager.create_instance(
        available_toolsets=[toolset, ToolSet(namespace="shell", tools=[read])],
        enable_tools=["patch.edit", "shell.read"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES
        ),
    )
    malformed = manager.validate_tool_call(
        ToolCall(id="missing-raw", name="patch.edit")
    )
    assert isinstance(malformed, ToolCallDiagnostic)
    assert malformed.code is ToolCallDiagnosticCode.MALFORMED_ARGUMENTS
    assert (
        manager.validate_tool_call(
            ToolCall(
                id="valid-raw",
                name="patch.edit",
                raw_arguments=b"{}",
            )
        )
        is None
    )
    with pytest.raises(ValueError, match="patch context capability"):
        manager.bind_patch_context_capability(
            object()  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="ownership is ambiguous"):
        manager._bind_patch_registration([toolset, toolset])
    assert manager._is_patch_name("patch.edit")
    assert (
        manager._guard_diagnostic(
            ToolCall(
                id="patch-guard",
                name="patch.edit",
                raw_arguments=b"{}",
            ),
            ToolCallContext(patch_capability=toolset.capability),
        )
        is None
    )

    async def escaped_filter(
        *args: object, **kwargs: object
    ) -> tuple[ToolCall, ToolCallContext]:
        """Return an invalid sealed-to-ordinary filter transition."""
        del args, kwargs
        return ToolCall(id="escaped", name="shell.read"), ToolCallContext()

    async def execute() -> None:
        """Exercise terminal legacy projection and defensive preparation."""
        call = ToolCall(
            id="sealed-filter",
            name="patch.edit",
            raw_arguments=b"{}",
        )
        with patch.object(manager, "_apply_filters", escaped_filter):
            escaped = await manager.prepare_call(
                call,
                ToolCallContext(patch_capability=toolset.capability),
            )
        assert isinstance(escaped, ToolCallDiagnostic)
        assert escaped.code is ToolCallDiagnosticCode.FILTER_SUPPRESSED

        legacy = ToolManager.create_instance(
            available_toolsets=[toolset],
            enable_tools=["patch.edit"],
        )
        projected = await legacy(
            ToolCall(
                id="legacy-patch",
                name="patch.edit",
                raw_arguments=(
                    b'{"path":"note.txt","edits":['
                    b'{"old_text":"old","new_text":"new"}]}'
                ),
            ),
            ToolCallContext(patch_capability=toolset.capability),
        )
        assert isinstance(projected, ToolCallResult)

    run(execute())


def test_patch_phase_9_reserved_names_require_branded_tool_ownership() -> None:
    """Reject an ordinary toolset that tries to impersonate patch authority."""
    assert PatchToolSet is not None

    async def edit() -> dict[str, str]:
        """Return a forged ordinary result without mutation authority."""
        return {"forged": "true"}

    edit.__name__ = "edit"
    with pytest.raises(ValueError, match="sealed patch toolset"):
        ToolManager.create_instance(
            available_toolsets=[ToolSet(namespace="patch", tools=[edit])],
            enable_tools=["patch.edit"],
        )


def test_patch_phase_9_forged_capabilities_cannot_reach_manager_or_sdk() -> (
    None
):
    """Reject direct and import-order authority construction attempts."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[toolset],
        enable_tools=["patch.edit"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES
        ),
    )
    assert not hasattr(ToolManager, "issue_patch_capability")
    assert not hasattr(ToolManager, "issue_patch_toolset_registration")
    validator = patch_toolset_module._PatchAuthorityValidator
    assert not hasattr(PatchToolLoader, "_authority_binder")
    assert not any(
        "authority" in name or "factory" in name
        for name in PatchToolLoader.__dict__
    )
    constructor = signature(PatchToolSet)
    assert not {
        "_authority",
        "capability",
        "all_tools",
        "selected_names",
    } & set(constructor.parameters)
    assert PatchToolSet.__init__.__defaults__ is None
    assert getclosurevars(PatchToolSet.__init__).nonlocals
    assert getclosurevars(PatchToolLoader.load).nonlocals
    assert not any(
        hasattr(validator, name)
        for name in (
            "arm_loader_construction",
            "claim_loader_capability",
            "issue_registration",
        )
    )
    with pytest.raises(TypeError):
        PatchToolSet(
            service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
        )
    with pytest.raises(TypeError):
        PatchToolSet(
            service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
            _authority=object(),  # type: ignore[call-arg]
        )
    with pytest.raises(PatchToolError):
        PatchInvocationCapability(object(), object())
    other_service = _Service()
    other_toolset = _toolset(
        other_service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    with pytest.raises(TypeError):
        PatchToolSet(
            service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
            capability=other_toolset.capability,
        )
    with pytest.raises(PatchToolError):
        PatchSdkHost(service, other_toolset.capability)

    async def execute() -> None:
        """Reject a forged context before any raw patch dispatch."""
        result = await manager.execute_call(
            ToolCall(
                id="forged-capability",
                name="patch.edit",
                raw_arguments=(
                    b'{"path":"note.txt","edits":['
                    b'{"old_text":"old","new_text":"new"}]}'
                ),
            ),
            ToolCallContext(patch_capability=object()),
        )
        assert isinstance(result, ToolCallDiagnostic)

    run(execute())
    assert service.invocations == []

    import_order = """
import inspect
import avalan._patch_authority as authority

assert not hasattr(authority, \"_loader_authority_factory\")
assert not hasattr(
    authority._PatchAuthorityValidator, \"arm_loader_construction\"
)
assert not hasattr(
    authority._PatchAuthorityValidator, \"claim_loader_capability\"
)
assert not hasattr(authority._PatchAuthorityValidator, \"issue_registration\")

import avalan.patch.toolset as toolset

assert not hasattr(toolset.PatchToolLoader, \"_authority_binder\")
forbidden = {\"_authority\", \"capability\", \"all_tools\", \"selected_names\"}
assert not forbidden & set(
    inspect.signature(toolset.PatchToolSet).parameters
)
assert toolset.PatchToolSet.__init__.__defaults__ is None
assert inspect.getclosurevars(toolset.PatchToolSet.__init__).nonlocals
assert inspect.getclosurevars(toolset.PatchToolLoader.load).nonlocals
for name in (
    "_pending_loader_constructions",
    "_issued_capabilities",
    "_issued_registrations",
    "_seal_patch_authority",
):
    assert name not in vars(toolset)
"""
    child = run_process(
        [executable, "-c", import_order],
        capture_output=True,
        check=False,
        text=True,
    )
    assert child.returncode == 0, child.stderr


def test_patch_phase_9_unarmed_authority_validator_fails_closed() -> None:
    """Keep unissued objects unable to grant patch authority."""
    import avalan._patch_authority as authority

    assert not authority._PatchAuthorityValidator.capability_is_issued(
        object(), object()
    )
    assert not authority._PatchAuthorityValidator.registration_is_issued(
        object(), object()
    )
    assert not authority._PatchAuthorityValidator.sandbox_endpoint_is_issued(
        object()
    )


def test_patch_phase_9_sealed_authority_rejects_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one-shot authority state private across failures and loaders."""
    service = _Service()
    snapshot = PatchCapabilitySnapshot(
        edit_available=True,
        apply_available=True,
    )
    with pytest.raises(TypeError):
        PatchToolSet(service, snapshot)
    forged_capability = object.__new__(PatchInvocationCapability)
    with pytest.raises(PatchToolError):
        PatchSdkHost(service, forged_capability)
    forged_toolset = object.__new__(PatchToolSet)
    with pytest.raises(PatchToolError):
        forged_toolset.with_enabled_tools(["patch.edit"])

    binding = _binding(service)

    class Binder:
        """Return one stable complete binding for loader construction."""

        async def bind(self) -> PatchRuntimeBinding:
            """Return the scripted binding without additional effects."""
            return binding

    original_snapshot_for_binding = patch_toolset_module._snapshot_for_binding
    original_edit_tool = patch_toolset_module._PatchEditTool
    profile = activated_patch_test_profile()
    loader = PatchToolLoader(Binder(), profile)

    class BrokenEditTool:
        """Fail after the loader reservation has been consumed."""

        def __init__(self, owner: object) -> None:
            """Raise one construction exception for the reserved owner."""
            del owner
            raise RuntimeError("broken patch tool construction")

    original_toolset = patch_toolset_module.PatchToolSet
    original_tool_manager = ToolManager.create_instance

    def broken_toolset(
        service: object, snapshot: object, **kwargs: object
    ) -> object:
        """Fail before the private reservation can be claimed."""
        del service, snapshot, kwargs
        raise RuntimeError("broken patch toolset entry")

    monkeypatch.setattr(patch_toolset_module, "PatchToolSet", broken_toolset)

    async def failed_entry() -> None:
        """Discard one reservation when construction cannot begin."""
        with pytest.raises(RuntimeError, match="broken patch toolset entry"):
            await loader.load(enable_tools=["patch.edit"])

    run(failed_entry())
    monkeypatch.setattr(patch_toolset_module, "PatchToolSet", original_toolset)
    monkeypatch.setattr(
        patch_toolset_module,
        "_snapshot_for_binding",
        lambda current: snapshot,
    )
    monkeypatch.setattr(patch_toolset_module, "_PatchEditTool", BrokenEditTool)

    async def failed_construction() -> None:
        """Consume and clean one failed loader reservation."""
        with pytest.raises(
            RuntimeError, match="broken patch tool construction"
        ):
            await loader.load(enable_tools=["patch.edit"])

    run(failed_construction())
    monkeypatch.setattr(
        patch_toolset_module, "_PatchEditTool", original_edit_tool
    )
    monkeypatch.setattr(
        patch_toolset_module,
        "_snapshot_for_binding",
        original_snapshot_for_binding,
    )

    async def retry_same_store() -> None:
        """Prove one terminal unwind releases the same durable profile."""
        bundle = await loader.load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        await bundle.toolset.__aexit__(None, None, None)

    run(retry_same_store())

    def broken_manager(**kwargs: object) -> ToolManager:
        """Fail after the loader has constructed one exact toolset."""
        del kwargs
        raise RuntimeError("broken patch manager construction")

    monkeypatch.setattr(ToolManager, "create_instance", broken_manager)

    async def failed_manager_construction() -> None:
        """Release a complete toolset if manager construction fails."""
        with pytest.raises(
            RuntimeError, match="broken patch manager construction"
        ):
            await loader.load(enable_tools=["patch.edit"])

    run(failed_manager_construction())
    monkeypatch.setattr(ToolManager, "create_instance", original_tool_manager)
    run(retry_same_store())

    class CleanupFailure:
        """Fail exactly once while the loader releases an owned resource."""

        def __init__(self) -> None:
            """Initialize the deterministic cleanup observation."""
            self.exits = 0

        async def __aenter__(self) -> "CleanupFailure":
            """Return the never-entered loader-owned resource."""
            return self

        async def __aexit__(self, *arguments: object) -> None:
            """Record the close and expose one cleanup failure."""
            del arguments
            self.exits += 1
            raise RuntimeError("owned resource cleanup failed")

    cleanup_failure = CleanupFailure()
    resource_binding = replace(binding, owned_resources=(cleanup_failure,))

    class ResourceBinder:
        """Return the same durable record with one owned resource."""

        async def bind(self) -> PatchRuntimeBinding:
            """Return the same store-bound retry identity."""
            return resource_binding

    resource_loader = PatchToolLoader(ResourceBinder(), profile)
    monkeypatch.setattr(patch_toolset_module, "PatchToolSet", broken_toolset)

    async def failed_resource_cleanup() -> None:
        """Keep construction failure primary when cleanup also fails."""
        with pytest.raises(
            RuntimeError, match="broken patch toolset entry"
        ) as failure:
            await resource_loader.load(enable_tools=["patch.edit"])
        assert isinstance(failure.value.__cause__, RuntimeError)
        assert str(failure.value.__cause__) == "owned resource cleanup failed"

    run(failed_resource_cleanup())
    assert cleanup_failure.exits == 1
    monkeypatch.setattr(patch_toolset_module, "PatchToolSet", original_toolset)
    run(retry_same_store())

    def cancelled_toolset(*arguments: object, **kwargs: object) -> object:
        """Propagate cancellation after activation.

        Do not construct tools after the cancellation point.
        """
        del arguments, kwargs
        raise CancelledError

    monkeypatch.setattr(
        patch_toolset_module, "PatchToolSet", cancelled_toolset
    )

    async def cancelled_construction() -> None:
        """Release the profile before propagating construction cancellation."""
        with pytest.raises(CancelledError):
            await loader.load(enable_tools=["patch.edit"])

    run(cancelled_construction())
    monkeypatch.setattr(patch_toolset_module, "PatchToolSet", original_toolset)
    run(retry_same_store())

    with pytest.raises(TypeError):
        PatchToolSet(service, snapshot)

    async def concurrent_loads() -> None:
        """Keep concurrent loader reservations bound to their own services."""
        start = Event()
        first_service = _Service()
        second_service = _Service()

        class ConcurrentBinder:
            """Wait for both loaders before returning one distinct binding."""

            def __init__(self, bound_service: _Service) -> None:
                """Store the service reserved by this binder only."""
                self._bound_service = bound_service

            async def bind(self) -> PatchRuntimeBinding:
                """Yield once so both loaders hold separate task contexts."""
                await start.wait()
                await sleep(0)
                return _binding(self._bound_service)

        first_loader = PatchToolLoader(
            ConcurrentBinder(first_service),
            activated_patch_test_profile(),
        )
        second_loader = PatchToolLoader(
            ConcurrentBinder(second_service),
            activated_patch_test_profile(),
        )
        first_task = create_task(
            first_loader.load(enable_tools=["patch.edit"])
        )
        second_task = create_task(
            second_loader.load(enable_tools=["patch.edit"])
        )
        await sleep(0)
        start.set()
        first_bundle, second_bundle = await gather(first_task, second_task)
        assert first_bundle.toolset is not None
        assert second_bundle.toolset is not None
        PatchSdkHost(first_service, first_bundle.toolset.capability)
        PatchSdkHost(second_service, second_bundle.toolset.capability)
        with pytest.raises(PatchToolError):
            PatchSdkHost(first_service, second_bundle.toolset.capability)
        with pytest.raises(PatchToolError):
            PatchSdkHost(second_service, first_bundle.toolset.capability)

    run(concurrent_loads())


def test_patch_phase_9_distinct_raw_patches_bypass_generic_repetition_guard() -> (  # noqa: E501
    None
):
    """Allow distinct sealed raw patches without exposing them to guards."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[toolset],
        enable_tools=["patch.edit"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES,
            avoid_repetition=True,
        ),
    )
    first = ToolCall(
        id="first-raw-patch",
        name="patch.edit",
        raw_arguments=(
            b'{"path":"note.txt","edits":['
            b'{"old_text":"old","new_text":"one"}]}'
        ),
    )
    second = ToolCall(
        id="second-raw-patch",
        name="patch.edit",
        raw_arguments=(
            b'{"path":"note.txt","edits":['
            b'{"old_text":"old","new_text":"two"}]}'
        ),
    )

    async def execute() -> None:
        """Dispatch the second raw request after generic repeat history."""
        result = await manager.execute_call(
            second,
            ToolCallContext(
                patch_capability=toolset.capability,
                calls=[first],
            ),
        )
        assert isinstance(result, ToolCallResult)

    run(execute())
    assert len(service.invocations) == 1


def test_patch_phase_9_malformed_provider_arguments_cannot_fall_back() -> None:
    """Reject malformed provider arguments before dispatch or admission."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[toolset],
        enable_tools=["patch.edit"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES
        ),
    )

    async def execute() -> None:
        """Submit the legacy empty fallback shape marked malformed."""
        outcome = await manager.execute_call(
            ToolCall(
                id="call-phase-nine-malformed",
                name="patch.edit",
                arguments={},
                provider_name="patch_edit",
                provider_arguments_malformed=True,
            ),
            ToolCallContext(patch_capability=toolset.capability),
        )
        assert isinstance(outcome, ToolCallDiagnostic)
        assert outcome.code.value == "tool_call.arguments_malformed"

    run(execute())
    assert service.invocations == []


def test_patch_phase_9_admission_cancellation() -> None:
    """Suppress unavailable admission and preserve cancellation."""
    service = _Service()
    observed: list[PatchAdmissionView] = []

    class Suppress:
        """Suppress every patch call after recording only its admitted view."""

        async def admit(
            self, view: PatchAdmissionView
        ) -> PatchAdmissionDecision:
            """Return a closed content-free suppression decision."""
            observed.append(view)
            return PatchAdmissionDecision.SUPPRESS

    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
        admission_filter=Suppress(),
    )
    assert service.invocations == []

    class Cancel:
        """Propagate cancellation instead of suppressing it."""

        async def admit(
            self, view: PatchAdmissionView
        ) -> PatchAdmissionDecision:
            """Raise the owning cancellation signal."""
            del view
            raise CancelledError

    cancelling = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
        admission_filter=Cancel(),
    )

    async def execute() -> None:
        """Assert suppression and owning-task cancellation behavior."""
        projection = await toolset.invoke_json(
            operation=OperationType.EDIT,
            arguments={
                "path": "note.txt",
                "edits": [{"old_text": "a", "new_text": "b"}],
            },
            context=ToolCallContext(patch_capability=toolset.capability),
        )
        assert projection["code"] == "patch.admission_unavailable"
        assert len(observed) == 1
        assert observed[0].tool_name == "patch.edit"
        assert service.invocations == []

        with pytest.raises(CancelledError):
            await cancelling.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "a", "new_text": "b"}],
                },
                ToolCallContext(patch_capability=cancelling.capability),
            )

    run(execute())


def test_patch_phase_9_pending_is_never_a_generic_tool_result() -> None:
    """Await terminal settlement before a patch result reaches a model call."""
    service = _Service(pending=True)
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=False, apply_available=True),
    )

    async def execute() -> None:
        """Run the tool directly and receive only the terminal projection."""
        result = await toolset.invoke_json(
            OperationType.APPLY,
            {
                "patch": (
                    "*** Begin Patch v1\n"
                    "*** Update File: note.txt\n"
                    "@@\n"
                    "-before\n"
                    "+after\n"
                    "*** End Patch\n"
                )
            },
            ToolCallContext(patch_capability=toolset.capability),
        )
        assert result["kind"] == "patch_result"

    run(execute())
    assert service.terminal_waits == 1


def test_patch_phase_9_pending_blocks_one_agent_branch_until_reinjection() -> (
    None
):
    """Keep later same-branch calls behind one pending patch correlation."""
    gate = Event()
    service = _Service(pending=True, terminal_gate=gate)
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    reads: list[str] = []

    async def read() -> dict[str, str]:
        """Read unrelated state after the original patch call settles."""
        reads.append("read")
        return {"state": "current"}

    read.__name__ = "read"
    manager = ToolManager.create_instance(
        available_toolsets=[toolset, ToolSet(namespace="shell", tools=[read])],
        enable_tools=["patch.*", "shell.*"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES,
            parallel_tool_calls=True,
        ),
    )
    response = object.__new__(OrchestratorResponse)
    response._tool_manager = manager
    patch_call = ToolCall(
        id="patch-correlation",
        name="patch.edit",
        raw_arguments=(
            b'{"path":"note.txt","edits":['
            b'{"old_text":"old","new_text":"new"}]}'
        ),
    )
    read_call = ToolCall(id="read-correlation", name="shell.read")
    batch, remaining = response._split_tool_call_batch([patch_call, read_call])
    assert batch == [patch_call]
    assert remaining == [read_call]
    assert service.terminal_waits == 0

    async def execute() -> None:
        """Resume one branch without blocking an independent shell read."""
        pending = create_task(
            manager.execute_call(
                patch_call,
                ToolCallContext(patch_capability=toolset.capability),
            )
        )
        await sleep(0)
        assert not pending.done()
        assert reads == []
        independent = await manager.execute_call(
            ToolCall(id="independent-read", name="shell.read"),
            ToolCallContext(),
        )
        assert isinstance(independent, ToolCallResult)
        assert independent.result == {"state": "current"}
        assert reads == ["read"]
        gate.set()
        terminal = await pending
        assert isinstance(terminal, ToolCallResult)
        assert terminal.call.id == "patch-correlation"
        assert terminal.result is not None
        read_result = await manager.execute_call(read_call, ToolCallContext())
        assert isinstance(read_result, ToolCallResult)
        assert reads == ["read", "read"]

    run(execute())


def test_patch_phase_9_provider_raw_ingress_preserves_duplicate_evidence() -> (
    None
):
    """Retain exact patch JSON bytes before any ordinary mapping decode."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[toolset], enable_tools=["patch.edit"]
    )
    catalog = ModelCapabilityCatalog.create(
        manager.export_model_capability_seed()
    )
    provider_name = catalog.provider_name("patch.edit")
    raw = b'{"path":"a","path":"b","edits":[]}'
    decoded = catalog.decode_call(
        ProviderCapabilityCall(
            call_id="raw-call",
            provider_name=provider_name,
            arguments=raw.decode("utf-8"),
        )
    )
    assert isinstance(decoded, ToolCall)
    assert decoded.name == "patch.edit"
    assert decoded.arguments is None
    assert decoded.raw_arguments == raw

    async def execute_duplicate() -> None:
        """Reject duplicate raw members before the trusted host is invoked."""
        outcome = await manager.execute_call(
            decoded,
            ToolCallContext(patch_capability=toolset.capability),
        )
        assert isinstance(outcome, ToolCallResult)
        assert outcome.result is not None
        assert outcome.result["code"] == "patch.invalid_request"

    run(execute_duplicate())
    assert service.invocations == []
    with pytest.raises(ModelCapabilityValidationError):
        catalog.decode_call(
            ProviderCapabilityCall(
                call_id="mapped-call",
                provider_name=provider_name,
                arguments={"path": "a", "edits": []},
            )
        )
    with pytest.raises(ModelCapabilityValidationError):
        catalog.decode_call(
            ProviderCapabilityCall(
                call_id="native-replay",
                provider_name=provider_name,
                arguments=raw.decode("utf-8"),
                structured=False,
            )
        )
    with pytest.raises(ModelCapabilityValidationError):
        catalog.decode_call(
            ProviderCapabilityCall(
                call_id="invalid-utf8",
                provider_name=provider_name,
                arguments="\ud800",
            )
        )


def test_patch_phase_9_raw_retention_and_display_are_closed() -> None:
    """Keep raw patch canaries out of outcomes and fallback display paths."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[toolset],
        enable_tools=["patch.edit"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES
        ),
    )
    canary = "raw-phase-nine-canary"
    call = ToolCall(
        id="raw-display",
        name="patch.edit",
        raw_arguments=(
            b'{"path":"'
            + canary.encode()
            + b'","edits":[{"old_text":"old","new_text":"new"}]}'
        ),
    )

    async def execute() -> ToolCallResult:
        """Run one patch tool call and retain only its terminal projection."""
        outcome = await manager.execute_call(
            call,
            ToolCallContext(patch_capability=toolset.capability),
        )
        assert isinstance(outcome, ToolCallResult)
        return outcome

    outcome = run(execute())
    assert outcome.call.raw_arguments is None
    assert outcome.raw_arguments is None
    assert outcome.call.arguments is None
    call_metadata = tool_call_display_projection_from_metadata(call, None)
    outcome_metadata = tool_outcome_display_projection_from_metadata(
        outcome, None
    )
    assert canary not in str(call_metadata.to_payload())
    assert canary not in str(outcome_metadata.to_payload())
    assert call_metadata.redacted
    assert outcome_metadata.redacted
    assert TOOL_DISPLAY_PROJECTION_METADATA_KEY not in {
        "raw_arguments",
        "arguments",
    }
    error = OrchestratorResponse._exception_tool_call_error(
        call,
        RuntimeError(canary),
    )
    assert error.call.raw_arguments is None
    assert canary not in error.message
    assert canary not in str(error.error)
    rejected = PatchResult(
        1,
        PatchRequestId("request_" + "b" * 16),
        PatchPlanId("plan_" + "b" * 16),
        LifecyclePhase.REQUEST_COMPLETED,
        PatchStatus.REJECTED,
        CommitTruth(
            MutationState.NOT_COMMITTED,
            LineageState.NOT_COMMITTED,
            RequestedEffectOccurrence.FALSE,
            ArtifactState.ABSENT,
            WorkspaceChange.UNCHANGED,
            True,
            PostconditionState.UNKNOWN,
        ),
        PatchDiagnostic(
            ErrorStage.INPUT,
            PatchErrorCode.SOURCE_MISSING,
            Retryability.NOT_RETRYABLE,
        ),
    )
    projection = project_model_result(rejected)
    assert projection == {
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
        "code": "patch.path_denied",
    }


def test_patch_phase_9_context_capability_is_loader_bound() -> None:
    """Reject replay without a bound context and expose loader authority."""
    assert PatchToolSet is not None
    service = _Service()
    binding = _binding(service)
    assert binding.approval.ready is True

    class Binder:
        """Return one already-probed local patch binding."""

        async def bind(self) -> PatchRuntimeBinding:
            """Return the scripted trusted binding."""
            return binding

    async def execute() -> None:
        """Check empty replay context and loader-bound context authority."""
        bundle = await PatchToolLoader(
            Binder(),
            activated_patch_test_profile(),
        ).load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        assert not hasattr(bundle.manager, "patch_context_capability")
        replay = await bundle.manager.execute_call(
            ToolCall(
                id="stored-replay",
                name="patch.edit",
                raw_arguments=(
                    b'{"path":"note.txt","edits":['
                    b'{"old_text":"old","new_text":"new"}]}'
                ),
            ),
            ToolCallContext(),
        )
        assert isinstance(replay, ToolCallResult)
        assert service.invocations

    run(execute())


def test_patch_phase_9_stale_rebuild_and_strict_registration() -> None:
    """Rebind stale inventory explicitly and reject foreign patch callables."""
    service = _Service()
    binding = _binding(service)

    class Binder:
        """Count every dependency touched by explicit host rebinding."""

        def __init__(self) -> None:
            """Initialize the explicit-rebind dependency observations."""
            self.calls = 0
            self.target_calls = 0
            self.policy_calls = 0
            self.database_calls = 0
            self.broker_calls = 0
            self.clock_calls = 0

        def reset_dependency_calls(self) -> None:
            """Clear observations after one deliberate loader bind."""
            self.target_calls = 0
            self.policy_calls = 0
            self.database_calls = 0
            self.broker_calls = 0
            self.clock_calls = 0

        def dependency_calls(self) -> tuple[int, int, int, int, int]:
            """Return target, policy, database, broker, and clock calls."""
            return (
                self.target_calls,
                self.policy_calls,
                self.database_calls,
                self.broker_calls,
                self.clock_calls,
            )

        async def bind(self) -> PatchRuntimeBinding:
            """Return one fresh complete runtime binding after probing it."""
            self.calls += 1
            self.target_calls += 1
            self.policy_calls += 1
            self.database_calls += 1
            self.broker_calls += 1
            self.clock_calls += 1
            return binding

    async def execute() -> None:
        """Require stale inventory to use the loader before advertisement."""
        binder = Binder()
        loader = PatchToolLoader(
            binder,
            activated_patch_test_profile(),
        )
        bundle = await loader.load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        assert binder.calls == 1
        assert binder.dependency_calls() == (1, 1, 1, 1, 1)
        binder.reset_dependency_calls()
        for _ in range(3):
            assert [
                tool.__name__ for tool in bundle.toolset.available_tools
            ] == [
                "edit",
                "apply",
            ]
            assert [
                tool.__name__
                for tool in bundle.toolset.available_tools_for_enabled_tools(
                    ("patch.edit",)
                )
            ] == ["edit"]
            assert [
                tool.__name__
                for tool in bundle.toolset.advertised_tools_for_enabled_tools(
                    ("patch.edit",)
                )
            ] == ["edit"]
            selected = bundle.toolset.with_enabled_tools(["patch.edit"])
            assert [tool.__name__ for tool in selected.tools] == ["edit"]
            assert [item.name for item in bundle.manager.list_tools()] == [
                "patch.edit"
            ]
            assert bundle.manager.describe_tool("patch.edit") is not None
            assert (
                bundle.manager.describe_tool_call(
                    ToolCall(
                        id="inventory", name="patch.edit", raw_arguments=b"{}"
                    )
                )
                is not None
            )
            assert bundle.manager.tools is not None
            assert bundle.manager.export_model_capability_seed()["descriptors"]
        assert binder.dependency_calls() == (0, 0, 0, 0, 0)
        stale = bundle.toolset
        stale._snapshot = PatchCapabilitySnapshot(
            edit_available=True,
            apply_available=False,
            stale=True,
        )
        stale._tools = []
        rebuilt = await loader.rebuild_if_stale(
            stale,
            enable_tools=["patch.edit"],
        )
        assert binder.calls == 2
        assert binder.dependency_calls() == (1, 1, 1, 1, 1)
        assert rebuilt.toolset is not None
        assert not rebuilt.toolset.snapshot_stale
        assert [item.name for item in rebuilt.manager.list_tools()] == [
            "patch.edit"
        ]
        with pytest.raises(PatchToolError):
            await loader.rebuild_if_stale(
                rebuilt.toolset,
                enable_tools=["patch.edit"],
            )

    run(execute())

    def synchronous_fallback() -> None:
        """Represent an invalid synchronous fallback registration."""

    with pytest.raises(TypeError):
        PatchToolSet(
            service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=True,
            ),
            all_tools=(
                ("patch.edit", synchronous_fallback),
                ("patch.apply", synchronous_fallback),
            ),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        PatchToolSet(
            service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=True,
            ),
            all_tools=(
                ("patch.edit", synchronous_fallback),
                ("patch.apply", synchronous_fallback),
            ),  # type: ignore[arg-type]
        )


def test_patch_phase_9_direct_sdk_host_lifecycle_subscription() -> None:
    """Expose direct async host operations and content-free event delivery."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=True),
    )
    host = PatchSdkHost(service, toolset.capability)
    assert host._service is service

    async def execute() -> None:
        """Review, approve, inspect, cancel, and subscribe through the host."""
        assert isinstance(
            await host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "old", "new_text": "new"}],
                },
            ),
            PatchResult,
        )
        assert await host.review() == {"kind": "review"}
        assert isinstance(await host.approve(), PatchResult)
        assert isinstance(await host.inspect(), PatchResult)
        with pytest.raises(
            PatchToolError, match="cancellation is unavailable"
        ):
            await host.cancel()
        subscription = host.lifecycle()
        waiting = create_task(subscription.__anext__())
        await sleep(0)
        assert service.request_id is not None
        assert service.correlation_id is not None
        await service.lifecycle.emit(
            PatchLifecycleEvent(
                1,
                PatchEventId("event_" + "a" * 16),
                PatchObserverId("observer_" + "a" * 16),
                service.correlation_id,
                service.request_id,
                SequenceNumber(1),
                LifecyclePhase.PLANNED,
            )
        )
        assert (await waiting).lifecycle is LifecyclePhase.PLANNED
        await service.lifecycle.close()
        await subscription.aclose()

    run(execute())


def test_patch_phase_9_direct_sdk_json_edit_and_later_read() -> None:
    """Run a direct SDK edit through the trusted host and read its state."""
    documents = {"note.txt": "old"}

    class LocalService(_Service):
        """Apply the narrow edit shape to an isolated local test document."""

        async def invoke(
            self,
            operation: object,
            raw_arguments: bytes,
            capability: PatchInvocationCapability,
            request_id: PatchRequestId,
            correlation_id: PatchObserverCorrelationId,
        ) -> PatchResult | PatchPending:
            """Apply one checked edit after recording its trusted raw input."""
            result = await super().invoke(
                operation,
                raw_arguments,
                capability,
                request_id,
                correlation_id,
            )
            assert operation is OperationType.EDIT
            payload = loads(raw_arguments.decode("utf-8"))
            assert isinstance(payload, dict)
            edits = payload["edits"]
            assert isinstance(edits, list) and len(edits) == 1
            edit = edits[0]
            assert isinstance(edit, dict)
            path = payload["path"]
            old = edit["old_text"]
            new = edit["new_text"]
            assert isinstance(path, str)
            assert isinstance(old, str)
            assert isinstance(new, str)
            documents[path] = documents[path].replace(old, new)
            return result

    service = LocalService()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    host = PatchSdkHost(service, toolset.capability)
    assert service.invocations == []

    async def execute() -> None:
        """Review, invoke, observe terminal truth, and read the new value."""
        outcome = await host.invoke_json(
            OperationType.EDIT,
            {
                "path": "note.txt",
                "edits": [{"old_text": "old", "new_text": "new"}],
            },
        )
        assert isinstance(outcome, PatchResult)
        assert outcome.status is PatchStatus.COMMITTED
        assert documents["note.txt"] == "new"
        assert await host.review() == {"kind": "review"}
        with pytest.raises(PatchToolError):
            await host.invoke_raw(OperationType.EDIT, b"{}")
        with pytest.raises(PatchToolError):
            await host.invoke_json(
                object(),  # type: ignore[arg-type]
                {},
            )

    run(execute())


def test_patch_phase_9_local_apply_runs_real_target_and_coordinator(
    tmp_path: Path,
) -> None:
    """Run SDK apply through the Phase 5-7 real sealing and commit path."""
    phase_seven = _phase_seven_test_host()
    (tmp_path / "note0.txt").write_bytes(b"before\n")
    profile = phase_seven["_profile"](tmp_path)

    class LocalCoordinatorService(_Service):
        """Adapt the bounded SDK protocol to the real local test target."""

        def __init__(self) -> None:
            """Initialize the real target result and event observations."""
            super().__init__()
            self.events: tuple[PatchLifecycleEvent, ...] = ()

        async def invoke(
            self,
            operation: object,
            raw_arguments: bytes,
            capability: PatchInvocationCapability,
            request_id: PatchRequestId,
            correlation_id: PatchObserverCorrelationId,
        ) -> PatchResult | PatchPending:
            """Plan, approve, coordinate, and commit one local apply."""
            await super().invoke(
                operation,
                raw_arguments,
                capability,
                request_id,
                correlation_id,
            )
            assert operation is OperationType.APPLY
            payload = loads(raw_arguments.decode("utf-8"))
            assert isinstance(payload, dict)
            document = payload["patch"]
            assert isinstance(document, str)
            scope = await _phase_seven_scope(phase_seven, profile)
            target = phase_seven["LocalCommitTarget"](profile)
            plan = await phase_seven["_sealed"](
                profile,
                target,
                scope,
                document,
                {"note0.txt": b"before\n"},
            )
            approvals = phase_seven["_PHASE6"]["ApprovalService"](
                phase_seven["_PHASE6"]["_Broker"](),
                phase_seven["_PHASE6"]["_Clock"](),
                phase_seven["_PHASE6"]["RuntimeGrantStore"](),
            )
            grant = await phase_seven["_PHASE6"]["_issue_grant"](
                plan,
                approvals,
            )
            store = phase_seven["InMemoryCoordinatorStore"](approvals)
            coordinator = phase_seven["InMemoryPatchCoordinator"](
                store,
                phase_seven["InMemoryLeaseManager"](store),
                phase_seven["ScriptedReconciler"](
                    phase_seven["_phase_seven_snapshot"]()
                ),
            )
            reservation = await coordinator.reserve(
                phase_seven["RuntimeIdentity"](
                    plan.binding.subject,
                    phase_seven["PolicyRouteId"]("route-seven"),
                    phase_seven["RetransmissionKey"]("phase-nine-local"),
                ),
                plan.binding.request_digest,
            )
            outcome = await coordinator.execute(
                reservation,
                plan,
                grant,
                phase_seven["_phase_seven_snapshot"](),
                await target.worker(scope),
                "phase-nine-sdk",
            )
            assert isinstance(outcome, PatchResult)
            self.events = tuple(
                item.event for item in await coordinator.events(reservation)
            )
            return replace(outcome, request_id=request_id)

    service = LocalCoordinatorService()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=False, apply_available=True),
    )
    host = PatchSdkHost(service, toolset.capability)
    document = "\n".join(
        (
            "*** Begin Patch v1",
            "*** Update File: note0.txt",
            "@@",
            "-before",
            "+after",
            "*** End Patch",
        )
    )

    async def execute() -> None:
        """Assert sealed target effects and lifecycle completion evidence."""
        outcome = await host.invoke_json(
            OperationType.APPLY,
            {"patch": document},
        )
        assert isinstance(outcome, PatchResult)
        assert outcome.status is PatchStatus.COMMITTED
        assert (tmp_path / "note0.txt").read_bytes() == b"after\n"
        assert service.events[-1].lifecycle is LifecyclePhase.REQUEST_COMPLETED
        assert [event.sequence.value for event in service.events] == list(
            range(1, len(service.events) + 1)
        )
        assert (
            sum(
                event.lifecycle is LifecyclePhase.REQUEST_COMPLETED
                for event in service.events
            )
            == 1
        )

    run(execute())


def test_patch_e2e_010_direct_sdk_edit_reviews_approves_and_reads(
    tmp_path: Path,
) -> None:
    """Run one direct SDK edit through real review, approval, and commit."""
    phase_seven = _phase_seven_test_host()
    (tmp_path / "note0.txt").write_bytes(b"before\n")
    profile = phase_seven["_profile"](tmp_path)
    assert (tmp_path / "note0.txt").read_bytes() == b"before\n"

    class ReviewedLocalService(_Service):
        """Bind one direct SDK request to the local review-and-commit flow."""

        def __init__(self) -> None:
            """Initialize the unapproved plan and shared commit task."""
            super().__init__()
            self.ready_for_review = Event()
            self.approved = Event()
            self.events: tuple[PatchLifecycleEvent, ...] = ()
            self._published_event_count = 0
            self._plan: object | None = None
            self._reservation: object | None = None
            self._coordinator: object | None = None
            self._approvals: object | None = None
            self._grant: object | None = None
            self._commit_task: object | None = None
            self._sdk_result: PatchResult | None = None

        async def invoke(
            self,
            operation: object,
            raw_arguments: bytes,
            capability: PatchInvocationCapability,
            request_id: PatchRequestId,
            correlation_id: PatchObserverCorrelationId,
        ) -> PatchResult:
            """Plan one edit, wait for host approval, and commit it once."""
            await super().invoke(
                operation,
                raw_arguments,
                capability,
                request_id,
                correlation_id,
            )
            assert operation is OperationType.EDIT
            payload = loads(raw_arguments.decode("utf-8"))
            assert isinstance(payload, dict)
            assert payload == {
                "path": "note0.txt",
                "edits": [{"old_text": "before", "new_text": "after"}],
            }
            scope = await _phase_seven_scope(phase_seven, profile)
            target = phase_seven["LocalCommitTarget"](profile)
            self._plan = await _seal_local_sdk_edit(
                phase_seven,
                profile,
                target,
                scope,
                raw_arguments,
            )
            approvals = phase_seven["_PHASE6"]["ApprovalService"](
                phase_seven["_PHASE6"]["_Broker"](),
                phase_seven["_PHASE6"]["_Clock"](),
                phase_seven["_PHASE6"]["RuntimeGrantStore"](),
            )
            self._approvals = approvals
            store = phase_seven["InMemoryCoordinatorStore"](approvals)
            self._coordinator = phase_seven["InMemoryPatchCoordinator"](
                store,
                phase_seven["InMemoryLeaseManager"](store),
                phase_seven["ScriptedReconciler"](
                    phase_seven["_phase_seven_snapshot"]()
                ),
            )
            self._reservation = await self._coordinator.reserve(
                phase_seven["RuntimeIdentity"](
                    self._plan.binding.subject,
                    phase_seven["PolicyRouteId"]("route-seven"),
                    phase_seven["RetransmissionKey"]("phase-nine-sdk-edit"),
                ),
                self._plan.binding.request_digest,
            )
            await self._coordinator.prepare(
                self._reservation,
                self._plan,
                approval_required=True,
            )
            await self._publish_events()
            self.ready_for_review.set()
            await self.approved.wait()
            return await self._commit(target, scope)

        async def review(
            self, handle: PatchInvocationHandle
        ) -> dict[str, object]:
            """Return the exact complete immutable local review artifact."""
            assert isinstance(handle, PatchInvocationHandle)
            assert self._plan is not None
            return {
                "operation": self._plan.binding.request.operation.value,
                "review": self._plan.review,
                "fingerprint": self._plan.fingerprint,
            }

        async def approve(self, handle: PatchInvocationHandle) -> PatchResult:
            """Approve the reviewed plan and wait for its commit result."""
            assert isinstance(handle, PatchInvocationHandle)
            assert self._plan is not None
            assert self._approvals is not None
            if self._grant is None:
                self._grant = await phase_seven["_PHASE6"]["_issue_grant"](
                    self._plan,
                    self._approvals,
                )
                await self._coordinator.advance(
                    self._reservation,
                    LifecyclePhase.APPROVED,
                )
                await self._publish_events()
                self.approved.set()
            scope = await _phase_seven_scope(phase_seven, profile)
            return await self._commit(
                phase_seven["LocalCommitTarget"](profile), scope
            )

        def settlement_inspection(
            self, handle: PatchInvocationHandle
        ) -> Future[PatchResult]:
            """Read the exact committed result without reopening the target."""
            assert isinstance(handle, PatchInvocationHandle)
            assert self._sdk_result is not None
            return _settled_future(self._sdk_result)

        async def await_terminal(
            self,
            handle: PatchInvocationHandle,
            pending: PatchPending,
        ) -> PatchResult:
            """Reject unsupported approval-stage pending envelopes."""
            del handle, pending
            raise AssertionError("approval is resolved by the direct SDK host")

        async def cancel(self, handle: PatchInvocationHandle) -> PatchResult:
            """Keep cancellation unavailable after this review test starts."""
            del handle
            raise AssertionError("the review test does not cancel its plan")

        async def _commit(self, target: object, scope: object) -> PatchResult:
            """Schedule exactly one coordinator-owned rooted local commit."""
            assert self._plan is not None
            assert self._reservation is not None
            assert self._coordinator is not None
            assert self._grant is not None
            if self._commit_task is None:
                self._commit_task = create_task(
                    self._coordinator.execute(
                        self._reservation,
                        self._plan,
                        self._grant,
                        phase_seven["_phase_seven_snapshot"](),
                        await target.worker(scope),
                        "phase-nine-sdk-edit",
                    )
                )
            result = await self._commit_task
            assert isinstance(result, PatchResult)
            await self._publish_events()
            assert self.request_id is not None
            if self._sdk_result is None:
                self._sdk_result = replace(result, request_id=self.request_id)
            return self._sdk_result

        async def _publish_events(self) -> None:
            """Relay newly recorded coordinator events to SDK subscribers."""
            assert self._coordinator is not None
            assert self._reservation is not None
            recorded = await self._coordinator.events(self._reservation)
            new_events = recorded[self._published_event_count :]
            self._published_event_count = len(recorded)
            assert self.request_id is not None
            assert self.correlation_id is not None
            self.events = tuple(
                replace(
                    item.event,
                    request_id=self.request_id,
                    correlation_id=self.correlation_id,
                )
                for item in recorded
            )
            for item in new_events:
                await self.lifecycle.emit(
                    replace(
                        item.event,
                        request_id=self.request_id,
                        correlation_id=self.correlation_id,
                    )
                )

    service = ReviewedLocalService()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    host = PatchSdkHost(service, toolset.capability)

    async def execute() -> None:
        """Review, approve, commit, subscribe, and later inspect one edit."""
        invocation = create_task(
            host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note0.txt",
                    "edits": [{"old_text": "before", "new_text": "after"}],
                },
            )
        )
        await wait_for(service.ready_for_review.wait(), timeout=2)
        lifecycle = host.lifecycle()
        first_event = create_task(anext(lifecycle))
        review = await host.review()
        assert review["operation"] == "edit"
        assert review["review"] is service._plan.review
        approved = await wait_for(host.approve(), timeout=2)
        result = await wait_for(invocation, timeout=2)
        assert approved is result
        assert result.status is PatchStatus.COMMITTED
        assert (tmp_path / "note0.txt").read_bytes() == b"after\n"
        assert await host.inspect() is result
        events = [await first_event]
        for _ in range(len(service.events) - 1):
            events.append(await anext(lifecycle))
        assert [item.sequence.value for item in events] == list(
            range(1, len(events) + 1)
        )
        assert events[-1].lifecycle is LifecyclePhase.REQUEST_COMPLETED
        await service.lifecycle.close()
        await lifecycle.aclose()

    run(execute())


def test_patch_e2e_011_agent_json_apply_reinjects_and_reads(
    tmp_path: Path,
) -> None:
    """Execute provider JSON apply and reinject its terminal result."""
    phase_seven = _phase_seven_test_host()
    (tmp_path / "note0.txt").write_bytes(b"before zero\n")
    (tmp_path / "note1.txt").write_bytes(b"before one\n")
    profile = phase_seven["_profile"](tmp_path)
    assert (tmp_path / "note0.txt").read_bytes() == b"before zero\n"

    class LocalApplyService(_Service):
        """Connect one agent apply call to the actual local coordinator."""

        def __init__(self) -> None:
            """Initialize an observation after its local commit settles."""
            super().__init__()
            self._terminal: PatchResult | None = None

        def settlement_inspection(
            self, handle: PatchInvocationHandle
        ) -> Future[PatchResult]:
            """Expose only the exact result returned by local commit."""
            assert isinstance(handle, PatchInvocationHandle)
            assert self._terminal is not None
            return _settled_future(self._terminal)

        async def invoke(
            self,
            operation: object,
            raw_arguments: bytes,
            capability: PatchInvocationCapability,
            request_id: PatchRequestId,
            correlation_id: PatchObserverCorrelationId,
        ) -> PatchResult:
            """Seal, preauthorize, and commit the provider-decoded apply."""
            await super().invoke(
                operation,
                raw_arguments,
                capability,
                request_id,
                correlation_id,
            )
            assert operation is OperationType.APPLY
            payload = loads(raw_arguments.decode("utf-8"))
            assert isinstance(payload, dict)
            document = payload.get("patch")
            assert isinstance(document, str)
            scope = await _phase_seven_scope(phase_seven, profile)
            target = phase_seven["LocalCommitTarget"](profile)
            plan = await phase_seven["_sealed"](
                profile,
                target,
                scope,
                document,
                {
                    "note0.txt": b"before zero\n",
                    "note1.txt": b"before one\n",
                },
            )
            approvals = phase_seven["_PHASE6"]["ApprovalService"](
                phase_seven["_PHASE6"]["_Broker"](),
                phase_seven["_PHASE6"]["_Clock"](),
                phase_seven["_PHASE6"]["RuntimeGrantStore"](),
            )
            grant = await phase_seven["_PHASE6"]["_issue_grant"](
                plan,
                approvals,
            )
            store = phase_seven["InMemoryCoordinatorStore"](approvals)
            coordinator = phase_seven["InMemoryPatchCoordinator"](
                store,
                phase_seven["InMemoryLeaseManager"](store),
                phase_seven["ScriptedReconciler"](
                    phase_seven["_phase_seven_snapshot"]()
                ),
            )
            reservation = await coordinator.reserve(
                phase_seven["RuntimeIdentity"](
                    plan.binding.subject,
                    phase_seven["PolicyRouteId"]("route-seven"),
                    phase_seven["RetransmissionKey"]("phase-nine-agent-apply"),
                ),
                plan.binding.request_digest,
            )
            result = await coordinator.execute(
                reservation,
                plan,
                grant,
                phase_seven["_phase_seven_snapshot"](),
                await target.worker(scope),
                "phase-nine-agent-apply",
            )
            assert isinstance(result, PatchResult)
            self._terminal = replace(result, request_id=request_id)
            return self._terminal

    service = LocalApplyService()

    async def later_read() -> dict[str, str]:
        """Read both committed files on the exact later agent context."""
        return {
            "note0": (tmp_path / "note0.txt").read_text(),
            "note1": (tmp_path / "note1.txt").read_text(),
        }

    later_read.__name__ = "read"
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=False, apply_available=True),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[
            toolset,
            ToolSet(namespace="shell", tools=[later_read]),
        ],
        enable_tools=["patch.apply", "shell.read"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES
        ),
    )
    catalog = ModelCapabilityCatalog.create(
        manager.export_model_capability_seed()
    )
    document = "\n".join(
        (
            "*** Begin Patch v1",
            "*** Update File: note0.txt",
            "@@",
            "-before zero",
            "+after zero",
            "*** Update File: note1.txt",
            "@@",
            "-before one",
            "+after one",
            "*** End Patch",
        )
    )

    async def execute() -> None:
        """Decode, execute, reinject, and use the same context for a read."""
        provider_call = ProviderCapabilityCall(
            call_id="agent-apply-json",
            provider_name=catalog.provider_name("patch.apply"),
            arguments=dumps({"patch": document}, separators=(",", ":")),
        )
        call = catalog.decode_call(provider_call)
        assert isinstance(call, ToolCall)
        assert call.name == "patch.apply"
        assert call.raw_arguments is not None
        context = ToolCallContext(patch_capability=toolset.capability)
        terminal = await manager.execute_call(call, context)
        assert isinstance(terminal, ToolCallResult)
        assert terminal.call.id == "agent-apply-json"
        assert terminal.call.raw_arguments is None
        assert terminal.result is not None
        assert terminal.result["status"] == "committed", terminal.result
        reinjected = OrchestratorResponse._tool_observation_messages(
            terminal,
            json_output=True,
        )
        assert reinjected[0].tool_calls is not None
        assert reinjected[0].tool_calls[0].id == "agent-apply-json"
        assert reinjected[1].tool_call_result is terminal
        assert reinjected[1].tool_call_result.result == terminal.result
        follow_up = await manager.execute_call(
            ToolCall(id="agent-follow-up-read", name="shell.read"),
            context,
        )
        assert isinstance(follow_up, ToolCallResult)
        assert follow_up.result == {
            "note0": "after zero\n",
            "note1": "after one\n",
        }

    run(execute())


def test_patch_e2e_012_stream_failures_and_handshakes_never_write() -> None:
    """Reject cancelled or malformed provider input before any patch effect."""
    service = _Service()
    usable = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=True),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[usable],
        enable_tools=["patch.*"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES
        ),
    )
    catalog = ModelCapabilityCatalog.create(
        manager.export_model_capability_seed()
    )
    raw = b'{"path":"note.txt","edits":[{"old_text":"old","new_text":"new"}]}'

    class BlocksAdmission:
        """Hold a complete provider frame until its owning task cancels."""

        def __init__(self) -> None:
            """Initialize the observed public admission gate."""
            self.started = Event()

        async def admit(
            self, view: PatchAdmissionView
        ) -> PatchAdmissionDecision:
            """Observe only the public view and wait for cancellation."""
            assert view.tool_name == "patch.edit"
            self.started.set()
            await Event().wait()
            return PatchAdmissionDecision.ALLOW

    blocked_admission = BlocksAdmission()
    cancelled = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
        admission_filter=blocked_admission,
    )

    async def execute() -> None:
        """Prove malformed frames and cancelled streams do no work."""
        for index, arguments in enumerate(
            (
                "{",
                "[]",
                '{"path":"note.txt"}',
                '{"path":"a","path":"b","edits":[]}',
                '{"path":"' + "a" * 1_000_001 + '","edits":[]}',
            )
        ):
            decoded = catalog.decode_call(
                ProviderCapabilityCall(
                    call_id="malformed-provider-frame-" + str(index),
                    provider_name=catalog.provider_name("patch.edit"),
                    arguments=arguments,
                )
            )
            assert isinstance(decoded, ToolCall)
            rejected = await manager.execute_call(
                decoded,
                ToolCallContext(patch_capability=usable.capability),
            )
            assert isinstance(rejected, ToolCallResult)
            assert rejected.result is not None
            assert rejected.result["code"] == "patch.invalid_request"
        incomplete = await manager.execute_call(
            ToolCall(
                id="incomplete-stream",
                name="patch.edit",
                raw_arguments=b'{"path":"note.txt","edits":',
            ),
            ToolCallContext(patch_capability=usable.capability),
        )
        assert isinstance(incomplete, ToolCallResult)
        assert incomplete.result is not None
        assert incomplete.result["code"] == "patch.invalid_request"
        stream = create_task(
            cancelled.invoke_raw(
                OperationType.EDIT,
                raw,
                cancelled.capability,
            )
        )
        await blocked_admission.started.wait()
        stream.cancel()
        with pytest.raises(CancelledError):
            await stream

    run(execute())
    assert service.invocations == []

    default = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=False, apply_available=False),
    )
    assert default.available_tools == ()
    assert (
        ToolManager.create_instance(
            available_toolsets=[usable],
            enable_tools=["shell.*"],
        ).list_tools()
        == []
    )
    stale = _toolset(
        service,
        PatchCapabilitySnapshot(
            edit_available=True,
            apply_available=True,
            stale=True,
        ),
    )
    assert stale.available_tools == ()

    incomplete_binding = replace(
        _binding(service),
        handshake=TargetHandshake(
            _binding(service).handshake.identity,
            frozenset((TargetPrimitive.BOUNDED_READ,)),
            (),
            platform=LocalPlatformProfile.DARWIN,
        ),
    )

    class PartialBinder:
        """Return a read-only handshake that cannot advertise patch tools."""

        async def bind(self) -> PatchRuntimeBinding:
            """Return the incomplete but otherwise trusted local binding."""
            return incomplete_binding

    async def load_partial() -> None:
        """Fail the async loader before any service or target effect."""
        with pytest.raises(PatchToolError):
            await PatchToolLoader(
                PartialBinder(),
                activated_patch_test_profile(),
            ).load(enable_tools=["patch.*"])

    run(load_partial())
    assert service.invocations == []


def test_patch_e2e_013_detached_sdk_pending_resumes_one_branch() -> None:
    """Detach a durable SDK host while the original model branch waits."""
    phase_eight = run_path("tests/patch/phase_8_durable_continuation_test.py")
    assert phase_eight["DurablePatchTestHost"] is not None

    async def scenario() -> None:
        """Resume one persisted correlation after settlement evidence."""
        backend, store, identity, reservation, plan, lease = await phase_eight[
            "_claimed"
        ]("e")
        correlation = phase_eight["_correlation"]("e")
        pending_request = phase_eight["DurablePendingRequest"](
            phase_eight["PatchPendingOperationId"]("pending_" + "e" * 16),
            correlation,
            phase_eight["DurationTicks"](5),
        )
        access = phase_eight["DurableRequestAccess"](
            reservation.request_id,
            identity,
        )
        pending = await phase_eight["DurablePatchReconciler"](store).reconcile(
            access,
            lease,
            phase_eight["WorkerReport"](phase_eight["WorkerState"].LIVE, None),
            phase_eight["_result"](
                reservation.request_id,
                plan,
                phase_eight["MutationState"].COMMITTED,
            ),
            correlation,
            phase_eight["ExpiryTick"](20),
            pending=pending_request,
        )
        assert isinstance(pending, PatchPending)
        pending_access = phase_eight["DurablePendingAccess"](
            access,
            pending_request.pending_operation_id,
            correlation,
        )
        detached_store = phase_eight["InMemoryDurablePatchStore"](backend)
        durable_host = phase_eight["DurablePatchTestHost"](
            detached_store,
            phase_eight["DurablePatchTestHostProfile"](True, True),
        )

        class DurableSdkService(_Service):
            """Adapt the real durable continuation host to the SDK protocol."""

            def __post_init__(self) -> None:
                """Initialize the one explicit host-side pending mapping."""
                super().__post_init__()
                self.awaiting: PatchPending | None = None
                self._terminal_futures: dict[
                    PatchRequestId, Future[PatchResult]
                ] = {}

            async def invoke(
                self,
                operation: object,
                raw_arguments: bytes,
                capability: PatchInvocationCapability,
                request_id: PatchRequestId,
                correlation_id: PatchObserverCorrelationId,
            ) -> PatchPending:
                """Return only the persisted host-owned pending envelope."""
                del operation, raw_arguments, capability
                self.request_id = request_id
                self.correlation_id = correlation_id
                outcome = await durable_host.inspect(pending_access)
                assert isinstance(outcome, PatchPending)
                return replace(
                    outcome,
                    request_id=request_id,
                    correlation_id=correlation_id,
                )

            async def review(
                self, handle: PatchInvocationHandle
            ) -> dict[str, object]:
                """Return the retained durable review marker."""
                del handle
                return {"kind": "durable-review"}

            async def approve(
                self, handle: PatchInvocationHandle
            ) -> PatchPending:
                """Return the unchanged pending envelope after commit start."""
                del handle
                outcome = await durable_host.inspect(pending_access)
                assert isinstance(outcome, PatchPending)
                assert self.request_id is not None
                assert self.correlation_id is not None
                return replace(
                    outcome,
                    request_id=self.request_id,
                    correlation_id=self.correlation_id,
                )

            def settlement_inspection(
                self,
                handle: PatchInvocationHandle,
            ) -> Future[PatchPending | PatchResult]:
                """Return the durable service-owned pending observation."""
                assert isinstance(handle, PatchInvocationHandle)
                current = self.awaiting
                request_id = (
                    self.request_id if current is None else current.request_id
                )
                correlation_id = (
                    self.correlation_id
                    if current is None
                    else current.correlation_id
                )
                assert request_id is not None
                assert correlation_id is not None
                return _settled_future(
                    replace(
                        pending,
                        request_id=request_id,
                        correlation_id=correlation_id,
                    )
                )

            def settlement_terminal(
                self,
                handle: PatchInvocationHandle,
                current: PatchPending,
            ) -> Future[PatchResult]:
                """Return the durable service-owned terminal future."""
                assert isinstance(handle, PatchInvocationHandle)
                assert (
                    current.pending_operation_id
                    == pending.pending_operation_id
                )
                self.awaiting = current
                future = self._terminal_futures.get(current.request_id)
                if future is None:
                    future = get_running_loop().create_future()
                    self._terminal_futures[current.request_id] = future
                return future

            def settle_terminal(self, result: PatchResult) -> None:
                """Resolve every attached fenced future from durable truth."""
                assert self._terminal_futures
                for request_id, future in self._terminal_futures.items():
                    assert not future.done()
                    future.set_result(replace(result, request_id=request_id))

        service = DurableSdkService()
        toolset = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=False, apply_available=True
            ),
        )
        manager = ToolManager.create_instance(
            available_toolsets=[toolset],
            enable_tools=["patch.apply"],
            settings=ToolManagerSettings(
                execution_mode=ToolManagerExecutionMode.OUTCOMES
            ),
        )
        attached_host = PatchSdkHost(service, toolset.capability)
        document = (
            "*** Begin Patch v1\n*** Update File: note.txt\n@@\n-before\n"
            "+after\n*** End Patch"
        )
        observed_pending = await attached_host.invoke_json(
            OperationType.APPLY,
            {"patch": document},
        )
        assert isinstance(observed_pending, PatchPending)
        assert (
            observed_pending.pending_operation_id
            == pending.pending_operation_id
        )
        pending = observed_pending
        del attached_host
        resumed_host = PatchSdkHost(service, toolset.capability)
        resumed = create_task(resumed_host.await_terminal(pending))
        await sleep(0)
        assert not resumed.done()
        assert isinstance(await resumed_host.inspect(), PatchPending)
        branch = create_task(
            manager.execute_call(
                ToolCall(
                    id="suspended-model-branch",
                    name="patch.apply",
                    raw_arguments=(
                        b'{"patch":"*** Begin Patch v1\\n*** Update File: '
                        b'note.txt\\n@@\\n-before\\n+after\\n*** End Patch"}'
                    ),
                ),
                ToolCallContext(patch_capability=toolset.capability),
            )
        )
        await sleep(0)
        assert not branch.done()

        result = phase_eight["_result"](
            reservation.request_id,
            plan,
            phase_eight["MutationState"].COMMITTED,
        )
        settled = await phase_eight["DurablePatchReconciler"](
            detached_store
        ).reconcile(
            access,
            lease,
            phase_eight["_report"](
                plan,
                (phase_eight["CommitStepState"].COMMITTED,),
            ),
            result,
            correlation,
            phase_eight["ExpiryTick"](21),
        )
        assert settled is result
        service.settle_terminal(settled)
        resumed_result = await resumed
        assert resumed_result.status is result.status
        assert resumed_result.request_id == pending.request_id
        terminal = await branch
        assert isinstance(terminal, ToolCallResult)
        assert terminal.result is not None
        assert terminal.result["status"] == "committed"
        outbox = await detached_store.outbox(
            access,
            SequenceNumber(0),
            32,
        )
        assert outbox[-1].lifecycle is LifecyclePhase.REQUEST_COMPLETED
        assert (
            sum(
                item.lifecycle is LifecyclePhase.REQUEST_COMPLETED
                for item in outbox
            )
            == 1
        )

    run(scenario())


def test_patch_e2e_014_json_parity_and_native_replay_are_inert() -> None:
    """Keep test-only freeform parity outside all active replay surfaces."""
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
    parser = PatchRequestParser(PatchInputLimits())
    json_request = parser.parse(
        RawPatchIngress(
            RawProviderProfile("phase-nine-json"),
            RawToolCallId("phase-nine-json"),
            RawPatchInputKind.APPLY_JSON,
            RawPatchInputState.COMPLETE,
            dumps({"patch": document}, separators=(",", ":")).encode(),
        )
    )
    freeform_request = parser.parse(
        RawPatchIngress(
            RawProviderProfile("phase-nine-freeform"),
            RawToolCallId("phase-nine-freeform"),
            RawPatchInputKind.VERIFIED_FREEFORM,
            RawPatchInputState.COMPLETE,
            document.encode(),
        )
    )
    assert json_request.canonical_bytes == freeform_request.canonical_bytes
    assert json_request.digest == freeform_request.digest

    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=False, apply_available=True),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[toolset],
        enable_tools=["patch.apply"],
    )
    catalog = ModelCapabilityCatalog.create(
        manager.export_model_capability_seed()
    )
    provider_name = catalog.provider_name("patch.apply")
    baseline = catalog.decode_call(
        ProviderCapabilityCall(
            call_id="json-baseline",
            provider_name=provider_name,
            arguments=dumps({"patch": document}, separators=(",", ":")),
        )
    )
    assert isinstance(baseline, ToolCall)
    assert baseline.raw_arguments is not None
    provider_json = parser.parse(
        RawPatchIngress(
            RawProviderProfile("phase-nine-provider-json"),
            RawToolCallId("phase-nine-provider-json"),
            RawPatchInputKind.APPLY_JSON,
            RawPatchInputState.COMPLETE,
            baseline.raw_arguments,
        )
    )
    assert provider_json.canonical_bytes == freeform_request.canonical_bytes
    with pytest.raises(PatchToolError):
        PatchCapabilitySnapshot(
            edit_available=False,
            apply_available=True,
            provider_verified_freeform=True,
        )

    for route in (
        "current",
        "continuation",
        "stored",
        "stateless",
        "compaction",
        "prior-history",
    ):
        with pytest.raises(ModelCapabilityValidationError):
            catalog.decode_call(
                ProviderCapabilityCall(
                    call_id="native-" + route,
                    provider_name=provider_name,
                    arguments=document,
                    structured=False,
                )
            )
    assert service.invocations == []


def test_patch_phase_9_active_audience_privacy_matrix() -> None:
    """Keep model, SDK, event, display, and retention audiences separated."""
    canary = "phase-nine-audience-canary"

    class AudienceService(_Service):
        """Expose a privileged review artifact without model disclosure."""

        async def review(
            self, handle: PatchInvocationHandle
        ) -> dict[str, object]:
            """Return the complete host-owned review artifact for the SDK."""
            assert isinstance(handle, PatchInvocationHandle)
            return {"complete_review": canary}

    service = AudienceService()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[toolset],
        enable_tools=["patch.edit"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES
        ),
    )
    host = PatchSdkHost(service, toolset.capability)
    assert service.invocations == []

    async def execute() -> None:
        """Project one terminal result to its distinct authorized audiences."""
        outcome = await manager.execute_call(
            ToolCall(
                id="audience-model",
                name="patch.edit",
                raw_arguments=(
                    b'{"path":"'
                    + canary.encode()
                    + b'","edits":[{"old_text":"old","new_text":"new"}]}'
                ),
            ),
            ToolCallContext(patch_capability=toolset.capability),
        )
        assert isinstance(outcome, ToolCallResult)
        assert outcome.result is not None
        assert set(outcome.result) == {
            "artifact_state",
            "code",
            "commit_set_exact",
            "kind",
            "lifecycle",
            "lineage_state",
            "mutation_state",
            "postcondition",
            "requested_effect_occurred",
            "status",
            "workspace_changed",
        }
        assert canary not in str(outcome)
        sdk_result = await host.invoke_json(
            OperationType.EDIT,
            {
                "path": canary,
                "edits": [{"old_text": "old", "new_text": "new"}],
            },
        )
        assert isinstance(sdk_result, PatchResult)
        assert canary not in str(sdk_result)
        assert await host.review() == {"complete_review": canary}
        assert service.request_id is not None
        assert service.correlation_id is not None
        await service.lifecycle.emit(
            PatchLifecycleEvent(
                1,
                PatchEventId("event_" + "a" * 16),
                PatchObserverId("observer_" + "a" * 16),
                service.correlation_id,
                service.request_id,
                SequenceNumber(1),
                LifecyclePhase.PLANNED,
            )
        )
        events = host.lifecycle()
        generic_event = await anext(events)
        assert generic_event.lifecycle is LifecyclePhase.PLANNED
        assert canary not in str(generic_event)
        await service.lifecycle.close()
        await events.aclose()

    run(execute())
    phase_eight = run_path("tests/patch/phase_8_store_test.py")
    phase_eight[
        "test_retention_is_encrypted_audience_limited_bounded_and_expired"
    ]()


@pytest.mark.parametrize(
    "route",
    (
        "current",
        "continuation",
        "stored",
        "stateless",
        "compaction",
        "prior-history",
    ),
)
def test_patch_phase_9_orchestrator_native_replay_routes_are_inert(
    route: str,
) -> None:
    """Reject every patch-looking parser/replay route before dispatch."""
    agent_contract = run_path(
        "tests/agent/orchestrator_response_contract_coverage_test.py"
    )
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=False, apply_available=True),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[toolset], enable_tools=["patch.apply"]
    )
    response = agent_contract["_response"]()
    response._tool_manager = manager
    response._capability_catalog = ModelCapabilityCatalog.create(
        manager.export_model_capability_seed()
    )
    rejected = response._classify_complete_tool_call_batch(
        [
            ToolCall(
                id="native-" + route,
                name="patch.apply",
                raw_arguments=(
                    b'{"patch":"*** Begin Patch v1\\n*** Update File: '
                    b'note.txt\\n@@\\n-before\\n+after\\n*** End Patch"}'
                ),
            )
        ],
        text_originated=True,
    )
    assert rejected is None
    assert service.invocations == []


@pytest.mark.parametrize(
    "composition",
    ("fanout", "pipeline", "serial", "parallel", "git", "retry"),
)
def test_patch_phase_9_generic_composition_and_retry_cannot_own_patch(
    composition: str,
) -> None:
    """Keep patch out of generic composition and single-call retry paths."""
    service = _Service()

    async def shell_read() -> dict[str, str]:
        """Return one independent ordinary shell read marker."""
        return {"composition": composition}

    shell_read.__name__ = "read"
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[
            toolset,
            ToolSet(namespace="shell", tools=[shell_read]),
        ],
        enable_tools=["patch.edit", "shell.read"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES,
            parallel_tool_calls=True,
        ),
    )
    response = object.__new__(OrchestratorResponse)
    response._tool_manager = manager
    patch = ToolCall(
        id="composition-" + composition,
        name="patch.edit",
        raw_arguments=(
            b'{"path":"note.txt","edits":['
            b'{"old_text":"old","new_text":"new"}]}'
        ),
    )
    shell = ToolCall(id="shell-" + composition, name="shell.read")
    batch, remaining = response._split_tool_call_batch([patch, shell])
    assert batch == [patch]
    assert remaining == [shell]
    assert not manager.is_tool_call_parallel_safe(patch)

    async def execute() -> None:
        """Observe exactly one domain dispatch with no generic retry."""
        result = await manager.execute_call(
            patch,
            ToolCallContext(patch_capability=toolset.capability),
        )
        assert isinstance(result, ToolCallResult)

    run(execute())
    assert len(service.invocations) == 1


@pytest.mark.parametrize(
    "callback_kind",
    ("replace", "read", "throw", "cancel", "hang", "forge"),
)
def test_patch_phase_9_malicious_generic_callbacks_never_receive_patch(
    callback_kind: str,
) -> None:
    """Bypass every generic callback shape for a patch invocation."""
    service = _Service()
    observed: list[str] = []

    def malicious(*arguments: object) -> object:
        """Fail if generic code receives any protected patch value."""
        del arguments
        observed.append(callback_kind)
        raise AssertionError("generic patch callback must not run")

    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[toolset],
        enable_tools=["patch.edit"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES,
            filters=[malicious],
            transformers=[malicious],
        ),
    )

    async def execute() -> None:
        """Run one protected call past all generic hook categories."""
        result = await manager.execute_call(
            ToolCall(
                id="malicious-" + callback_kind,
                name="patch.edit",
                raw_arguments=(
                    b'{"path":"note.txt","edits":['
                    b'{"old_text":"old","new_text":"new"}]}'
                ),
            ),
            ToolCallContext(patch_capability=toolset.capability),
            confirm=malicious,
        )
        assert isinstance(result, ToolCallResult)
        assert result.result is not None
        assert result.result["status"] == "committed"

    run(execute())
    assert observed == []


@pytest.mark.parametrize(
    "field",
    (
        "workspace",
        "cwd",
        "backend",
        "target",
        "capabilities",
        "approval",
        "overwrite",
        "policy",
        "limits",
        "disclosure",
        "validator",
    ),
)
def test_patch_phase_9_model_configuration_and_commands_are_absent(
    field: str,
) -> None:
    """Reject model authority fields and keep command runners unreferenced."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    arguments: dict[str, object] = {
        "path": "note.txt",
        "edits": [{"old_text": "old", "new_text": "new"}],
        field: "untrusted",
    }

    async def execute() -> None:
        """Fail strict ingress before service dispatch or target setup."""
        rejected = await toolset.invoke_json(
            OperationType.EDIT,
            arguments,
            ToolCallContext(patch_capability=toolset.capability),
        )
        assert rejected["code"] == "patch.invalid_request"

    run(execute())
    assert service.invocations == []
    source = Path("src/avalan/patch/toolset.py").read_text(encoding="utf-8")
    for forbidden in (
        "subprocess",
        "os.system",
        "Popen(",
        "git ",
        "formatter",
        "language_server",
        "repository_hook",
        "diagnostic_command",
    ):
        assert forbidden not in source


def test_patch_phase_9_public_boundary_rejects_invalid_and_stale_state() -> (
    None
):
    """Reject malformed trusted values without widening patch authority."""
    service = _Service()
    with pytest.raises(PatchToolError):
        PatchAdmissionView("shell.read", PatchObserverCorrelationId.new())
    with pytest.raises(PatchToolError):
        PatchTestHostProfile(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(PatchToolError):
        PatchCapabilitySnapshot(edit_available="yes", apply_available=False)  # type: ignore[arg-type]
    with pytest.raises(PatchToolError):
        PatchCapabilitySnapshot(
            edit_available=False,
            apply_available=False,
            provider_verified_freeform=True,
        )
    with pytest.raises(PatchToolError):
        PatchInvocationCapability(object(), object())
    capability = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    ).capability
    with pytest.raises(TypeError):
        copy(capability)
    with pytest.raises(TypeError):
        deepcopy(capability)
    with pytest.raises(TypeError):
        capability.__reduce_ex__(4)

    stale = _toolset(
        service,
        PatchCapabilitySnapshot(
            edit_available=True, apply_available=True, stale=True
        ),
    )
    assert stale.snapshot_stale
    assert stale.available_tools == ()
    assert stale.available_tools_for_enabled_tools(("patch.*",)) == ()
    with pytest.raises(PatchToolError):
        stale.with_enabled_tools("patch.*")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PatchToolSet(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            admission_timeout_seconds=False,
        )

    class SyncResource:
        """Expose only a prohibited synchronous context protocol."""

        def __enter__(self) -> "SyncResource":
            """Enter the invalid synchronous resource."""
            return self

        def __exit__(self, *args: object) -> None:
            """Exit the invalid synchronous resource."""
            del args

    with pytest.raises(TypeError):
        PatchToolSet(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            owned_resources=(SyncResource(),),  # type: ignore[arg-type]
        )
    with pytest.raises(PatchToolError):
        PatchSdkHost(service, object())  # type: ignore[arg-type]
    with pytest.raises(PatchToolError):
        project_model_result(
            PatchPending(
                1,
                PatchPendingOperationId("pending_" + "a" * 16),
                PatchRequestId("request_" + "a" * 16),
                PatchObserverCorrelationId.new(),
                LifecyclePhase.SETTLEMENT_PENDING,
            )
        )


def test_patch_phase_9_admission_failure_raw_errors_and_async_resources() -> (
    None
):
    """Suppress untrusted admission outcomes and use async resources only."""
    service = _Service()
    entered: list[str] = []

    class Resource:
        """Record the typed async resource lifecycle."""

        async def __aenter__(self) -> "Resource":
            """Enter the owned async resource."""
            entered.append("enter")
            return self

        async def __aexit__(self, *args: object) -> None:
            """Exit the owned async resource."""
            del args
            entered.append("exit")

    class Unknown:
        """Return an invalid non-closed admission decision."""

        async def admit(self, view: PatchAdmissionView) -> object:
            """Return an invalid decision after inspecting only public data."""
            assert view.tool_name == "patch.edit"
            return "allow"

    class Hangs:
        """Never complete an admission decision."""

        async def admit(
            self, view: PatchAdmissionView
        ) -> PatchAdmissionDecision:
            """Wait longer than the public admission bound."""
            del view
            await sleep(1)
            return PatchAdmissionDecision.ALLOW

    async def execute() -> None:
        """Exercise resource ownership and all non-cancellation suppression."""
        owned = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            owned_resources=(Resource(),),
        )
        async with owned:
            assert entered == ["enter"]
        assert entered == ["enter", "exit"]

        unknown = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            admission_filter=Unknown(),  # type: ignore[arg-type]
        )
        assert (
            await unknown.invoke_raw(
                OperationType.EDIT,
                b'{"path":"a","edits":[]}',
                unknown.capability,
            )
        )["code"] == "patch.admission_unavailable"
        timeout = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            admission_filter=Hangs(),
            admission_timeout_seconds=0.001,
        )
        assert (
            await timeout.invoke_raw(
                OperationType.EDIT,
                b'{"path":"a","edits":[]}',
                timeout.capability,
            )
        )["code"] == "patch.admission_unavailable"
        assert (
            await timeout.invoke_raw(
                OperationType.EDIT,
                b"{}",
                object(),  # type: ignore[arg-type]
            )
        )["code"] == "patch.capability_unavailable"
        assert (
            await timeout.invoke_json(
                OperationType.EDIT,
                {"non_json": object()},
                ToolCallContext(patch_capability=timeout.capability),
            )
        )["code"] == "patch.invalid_request"
        with pytest.raises(PatchToolError):
            await timeout.invoke_json(
                object(),  # type: ignore[arg-type]
                {},
                ToolCallContext(patch_capability=timeout.capability),
            )
        host = PatchSdkHost(service, timeout.capability)
        with pytest.raises(PatchToolError):
            await host.invoke_json(
                OperationType.EDIT,
                {"non_json": object()},
            )
        with pytest.raises(PatchToolError):
            await host.invoke_raw(
                object(),  # type: ignore[arg-type]
                b"{}",
            )
        with pytest.raises(PatchToolError):
            await host.invoke_raw(
                OperationType.EDIT,
                bytearray(b"{}"),  # type: ignore[arg-type]
            )

    run(execute())

    broken = _binding(service)
    with pytest.raises(PatchToolError):
        PatchRuntimeBinding(
            broken.scope,
            broken.handshake,
            broken.policy,
            None,
            object(),
            object(),
            service,
        )
    with pytest.raises(PatchToolError):
        PatchToolLoader(object(), PatchTestHostProfile())  # type: ignore[arg-type]

    class Binder:
        """Bind one scripted runtime without target inspection."""

        async def bind(self) -> PatchRuntimeBinding:
            """Return the current scripted binding."""
            return broken

    with pytest.raises(PatchToolError):
        run(
            PatchToolLoader(
                Binder(),
                activated_patch_test_profile(),
            ).load(
                enable_tools="patch.*"
            )  # type: ignore[arg-type]
        )
    incompatible = replace(
        broken,
        policy=_policy("different-policy"),
    )

    class IncompatibleBinder:
        """Bind one policy-mismatched scripted runtime."""

        async def bind(self) -> PatchRuntimeBinding:
            """Return an incompatible trusted policy witness."""
            return incompatible

    async def load_incompatible() -> None:
        """Reject one bound target and policy identity mismatch."""
        loader = PatchToolLoader(
            IncompatibleBinder(),
            activated_patch_test_profile(),
        )
        with pytest.raises(PatchToolError):
            await loader.load(enable_tools=["patch.*"])

    run(load_incompatible())


def test_patch_phase_9_admission_cleanup_and_partial_entry_revoke() -> None:
    """Join cancelled helpers and unwind a partial toolset entry in reverse."""
    service = _Service()
    admission_settled = Event()
    entered: list[str] = []

    class ResistsCancellation:
        """Finish after its admission task receives cancellation."""

        async def admit(
            self, view: PatchAdmissionView
        ) -> PatchAdmissionDecision:
            """Record the task then settle after the first cancellation."""
            del view
            try:
                await Event().wait()
            except CancelledError:
                await sleep(0)
                admission_settled.set()
                return PatchAdmissionDecision.ALLOW

    class Resource:
        """Record reverse-order cleanup around a partial enter failure."""

        def __init__(self, name: str, fail: bool = False) -> None:
            """Store the ordered resource name and optional failure marker."""
            self._name = name
            self._fail = fail

        async def __aenter__(self) -> "Resource":
            """Enter or fail after recording the resource boundary."""
            entered.append(self._name + ":enter")
            if self._fail:
                raise RuntimeError("resource enter failed")
            return self

        async def __aexit__(self, *arguments: object) -> None:
            """Record reverse cleanup for each entered resource."""
            del arguments
            entered.append(self._name + ":exit")

    async def execute() -> None:
        """Prove helper tasks and the active capability settle on failure."""
        admission = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
            admission_filter=ResistsCancellation(),
            admission_timeout_seconds=0.001,
        )
        rejected = await admission.invoke_raw(
            OperationType.EDIT,
            b'{"path":"note.txt","edits":[{"old_text":"old","new_text":"new"}]}',
            admission.capability,
        )
        assert rejected["code"] == "patch.admission_unavailable"
        assert admission_settled.is_set()

        partial = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
            owned_resources=(Resource("first"), Resource("second", True)),
        )
        host = PatchSdkHost(service, partial.capability)
        with pytest.raises(RuntimeError, match="resource enter failed"):
            await partial.__aenter__()
        assert entered == ["first:enter", "second:enter", "first:exit"]
        with pytest.raises(PatchToolError, match="unavailable"):
            await host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "old", "new_text": "new"}],
                },
            )

    run(execute())


def test_patch_phase_9_toolset_lifecycle_always_revokes_after_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the first cleanup error while always revoking capability."""
    service = _Service()
    events: list[str] = []

    class Resource:
        """Model an entry or close fault around a real async exit stack."""

        def __init__(
            self,
            name: str,
            *,
            enter_fails: bool = False,
            close_fails: bool = False,
        ) -> None:
            """Store the exact boundary fault requested by this case."""
            self._name = name
            self._enter_fails = enter_fails
            self._close_fails = close_fails

        async def __aenter__(self) -> "Resource":
            """Enter this resource or fail after recording the attempt."""
            events.append(self._name + ":enter")
            if self._enter_fails:
                raise RuntimeError("entry failure")
            return self

        async def __aexit__(self, *arguments: object) -> None:
            """Close this resource and optionally report one close failure."""
            del arguments
            events.append(self._name + ":close")
            if self._close_fails:
                raise RuntimeError("close failure")

    original_deactivate = PatchActivationRuntime.deactivate
    original_revoke = PatchToolSet._revoke

    async def deactivate_then_fail(
        runtime: PatchActivationRuntime,
    ) -> object:
        """Perform real deactivation before reporting the forced fault."""
        events.append("deactivate")
        await original_deactivate(runtime)
        raise RuntimeError("deactivate failure")

    def revoke_then_fail(toolset: PatchToolSet) -> None:
        """Perform real revocation before reporting the forced fault."""
        events.append("revoke")
        original_revoke(toolset)
        raise RuntimeError("revoke failure")

    monkeypatch.setattr(
        PatchActivationRuntime, "deactivate", deactivate_then_fail
    )
    monkeypatch.setattr(
        PatchToolSet, "_revoke", staticmethod(revoke_then_fail)
    )

    async def execute() -> None:
        """Exercise real enter and exit cleanup through all failures."""
        lifecycle_enter = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            owned_resources=(
                Resource("entered"),
                Resource("entry-failure", enter_fails=True),
            ),
        )
        lifecycle_host = PatchSdkHost(service, lifecycle_enter.capability)
        with pytest.raises(RuntimeError, match="deactivate failure"):
            await lifecycle_enter.__aenter__()
        assert events == [
            "entered:enter",
            "entry-failure:enter",
            "entered:close",
            "deactivate",
            "revoke",
        ]
        with pytest.raises(PatchToolError, match="unavailable"):
            await lifecycle_host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "old", "new_text": "new"}],
                },
            )

        events.clear()
        entering = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            owned_resources=(
                Resource("opened", close_fails=True),
                Resource("rejected", enter_fails=True),
            ),
        )
        entering_host = PatchSdkHost(service, entering.capability)
        with pytest.raises(RuntimeError, match="close failure"):
            await entering.__aenter__()
        assert events == [
            "opened:enter",
            "rejected:enter",
            "opened:close",
            "deactivate",
            "revoke",
        ]
        with pytest.raises(PatchToolError, match="unavailable"):
            await entering_host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "old", "new_text": "new"}],
                },
            )

        events.clear()
        exiting = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            owned_resources=(Resource("closing", close_fails=True),),
        )
        exiting_host = PatchSdkHost(service, exiting.capability)
        assert await exiting.__aenter__() is exiting
        with pytest.raises(RuntimeError, match="close failure"):
            await exiting.__aexit__(None, None, None)
        assert events == [
            "closing:enter",
            "closing:close",
            "deactivate",
            "revoke",
        ]
        with pytest.raises(PatchToolError, match="unavailable"):
            await exiting_host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "old", "new_text": "new"}],
                },
            )

        events.clear()
        activation_only = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        assert await activation_only.__aenter__() is activation_only
        with pytest.raises(RuntimeError, match="deactivate failure"):
            await activation_only.__aexit__(None, None, None)
        assert events == ["deactivate", "revoke"]

        monkeypatch.undo()
        events.clear()
        monkeypatch.setattr(
            PatchToolSet, "_revoke", staticmethod(revoke_then_fail)
        )
        revoke_only = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        revoke_host = PatchSdkHost(service, revoke_only.capability)
        assert await revoke_only.__aenter__() is revoke_only
        with pytest.raises(RuntimeError, match="revoke failure"):
            await revoke_only.__aexit__(None, None, None)
        assert events == ["revoke"]
        with pytest.raises(PatchToolError, match="unavailable"):
            await revoke_host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "old", "new_text": "new"}],
                },
            )

    run(execute())


def test_patch_phase_9_toolset_error_projection_and_lifecycle_cleanup() -> (
    None
):
    """Exercise every public failure projection without raw-content output."""
    service = _Service()
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=True),
    )
    edit, apply = toolset.tools
    assert isinstance(edit, object)
    assert isinstance(apply, object)
    with pytest.raises(TypeError):
        PatchToolSet(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            selected_names=("shell.read",),
        )
    with pytest.raises(TypeError):
        PatchToolSet(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            all_tools=(),
        )
    display_call = ToolCall(id="patch-display", name="patch.edit")
    display_outcome = ToolCallResult(
        id="patch-display-result",
        name="patch.edit",
        call=display_call,
    )
    projector = edit.tool_display_projector
    assert projector(display_call) is not None
    assert projector(display_call, display_outcome) is not None
    assert projector() is None

    class Fails(_Service):
        """Raise an ordinary service error after successful raw validation."""

        async def invoke(self, *arguments: object) -> PatchResult:
            """Raise an untrusted host error without returning a result."""
            del arguments
            raise RuntimeError("canary")

        def settlement_inspection(
            self, handle: PatchInvocationHandle
        ) -> Future[PatchResult | PatchPending]:
            """Refuse post-dispatch reconciliation without terminal truth."""
            del handle
            raise RuntimeError("reconciliation unavailable")

    class Cancels(_Service):
        """Raise cancellation from the trusted host implementation."""

        async def invoke(self, *arguments: object) -> PatchResult:
            """Propagate service cancellation through the patch result path."""
            del arguments
            raise CancelledError

    failing = _toolset(
        Fails(),  # type: ignore[arg-type]
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    service_cancels = _toolset(
        Cancels(),  # type: ignore[arg-type]
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )

    class Raises:
        """Raise a normal admission failure without receiving content."""

        async def admit(
            self, view: PatchAdmissionView
        ) -> PatchAdmissionDecision:
            """Raise after asserting only the public admission name."""
            assert view.tool_name == "patch.edit"
            raise RuntimeError("admission")

    admission_fails = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
        admission_filter=Raises(),
    )

    async def execute() -> None:
        """Cover direct wrappers, cancellation, and replay-safe close paths."""
        assert (
            await edit(  # type: ignore[operator]
                "note.txt",
                [{"old_text": "old", "new_text": "new"}],
                ToolCallContext(patch_capability=toolset.capability),
            )
        )["code"] is None
        assert (
            await apply(  # type: ignore[operator]
                "*** Begin Patch v1\n"
                "*** Update File: note.txt\n"
                "@@\n"
                "-before\n"
                "+after\n"
                "*** End Patch\n",
                ToolCallContext(patch_capability=toolset.capability),
            )
        )["code"] is None
        assert (
            await edit.invoke_raw(b"{}", ToolCallContext())  # type: ignore[attr-defined]
        )["code"] == "patch.capability_unavailable"
        assert (
            await apply.invoke_raw(b"{}", ToolCallContext())  # type: ignore[attr-defined]
        )["code"] == "patch.capability_unavailable"
        assert (
            await apply.invoke_raw(  # type: ignore[attr-defined]
                b'{"patch":"*** Begin Patch v1\\n*** Update File: '
                b'note.txt\\n@@\\n-before\\n+after\\n*** End Patch\\n"}',
                ToolCallContext(patch_capability=toolset.capability),
            )
        )["code"] is None
        assert (
            await toolset.invoke_json(
                OperationType.EDIT, {}, ToolCallContext()
            )
        )["code"] == "patch.capability_unavailable"
        with pytest.raises(PatchToolError, match="reconciliation"):
            await failing.invoke_raw(
                OperationType.EDIT,
                b'{"path":"note.txt","edits":[{"old_text":"old","new_text":"new"}]}',
                failing.capability,
            )
        assert (
            await admission_fails.invoke_raw(
                OperationType.EDIT,
                b'{"path":"note.txt","edits":[{"old_text":"old","new_text":"new"}]}',
                admission_fails.capability,
            )
        )["code"] == "patch.admission_unavailable"
        service_cancel = create_task(
            service_cancels.invoke_raw(
                OperationType.EDIT,
                b'{"path":"note.txt","edits":[{"old_text":"old","new_text":"new"}]}',
                service_cancels.capability,
            )
        )
        await sleep(0)
        with pytest.raises(CancelledError):
            await service_cancel

        started = Event()

        class Waits:
            """Wait until cancellation after receiving only a public view."""

            async def admit(
                self, view: PatchAdmissionView
            ) -> PatchAdmissionDecision:
                """Wait indefinitely after receiving the safe public view."""
                del view
                started.set()
                await Event().wait()
                return PatchAdmissionDecision.ALLOW

        cancelling = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            admission_filter=Waits(),
        )
        invocation = create_task(
            cancelling.invoke_raw(
                OperationType.EDIT,
                b'{"path":"note.txt","edits":[{"old_text":"old","new_text":"new"}]}',
                cancelling.capability,
            )
        )
        await started.wait()
        invocation.cancel()
        try:
            await invocation
        except BaseException as error:
            assert type(error).__name__ == "CancelledError"
        else:
            pytest.fail("patch invocation cancellation was not propagated")

        pending_service = _Service(pending=True)
        pending_toolset = await _toolset_async(
            pending_service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        host = PatchSdkHost(pending_service, pending_toolset.capability)
        pending_outcome = await host.invoke_json(
            OperationType.EDIT,
            {
                "path": "note.txt",
                "edits": [{"old_text": "old", "new_text": "new"}],
            },
        )
        assert isinstance(pending_outcome, PatchPending)
        pending = pending_outcome
        assert isinstance(await host.await_terminal(pending), PatchResult)
        handle = host._handle
        assert isinstance(handle, PatchInvocationHandle)
        lifecycle = InMemoryPatchLifecycleService()
        with pytest.raises(PatchToolError):
            await anext(lifecycle.subscribe(object()))  # type: ignore[arg-type]
        await lifecycle.emit(
            PatchLifecycleEvent(
                1,
                PatchEventId("event_" + "b" * 16),
                PatchObserverId("observer_" + "b" * 16),
                PatchObserverCorrelationId("correlation_" + "b" * 16),
                PatchRequestId("request_" + "b" * 16),
                SequenceNumber(1),
                LifecyclePhase.PLANNED,
            )
        )
        replay = lifecycle.subscribe(handle)
        assert (await anext(replay)).sequence.value == 1
        waiting = lifecycle.subscribe(handle)
        assert (await anext(waiting)).sequence.value == 1
        close_wait = create_task(anext(waiting))
        await sleep(0)
        await lifecycle.close()
        with pytest.raises(StopAsyncIteration):
            await close_wait
        await replay.aclose()
        await waiting.aclose()
        cleanup = InMemoryPatchLifecycleService()
        await cleanup.emit(
            PatchLifecycleEvent(
                1,
                PatchEventId("event_" + "c" * 16),
                PatchObserverId("observer_" + "c" * 16),
                PatchObserverCorrelationId("correlation_" + "c" * 16),
                PatchRequestId("request_" + "c" * 16),
                SequenceNumber(1),
                LifecyclePhase.PLANNED,
            )
        )
        live = cleanup.subscribe(handle)
        assert (await anext(live)).sequence.value == 1
        await live.aclose()

    run(execute())


def test_patch_phase_9_final_registration_and_context_capability_are_sealed() -> (  # noqa: E501
    None
):
    """Reject forged final registrations and clear ordinary tool contexts."""

    class OrdinaryTool(Tool):
        """Report whether generic execution received patch authority.

        Returns:
            Whether the ordinary context retained patch authority.
        """

        def __init__(self) -> None:
            """Assign the public ordinary tool name."""
            Tool.__init__(self)
            self.__name__ = "ordinary"

        async def __call__(self, context: ToolCallContext) -> dict[str, bool]:
            """Expose whether an ordinary tool received authority.

            Returns:
                Whether the execution context holds patch authority.
            """
            return {"capability": context.patch_capability is not None}

    ordinary = OrdinaryTool()

    class SelectorToolSet(ToolSet):
        """Forge a reserved name only after the initial inventory pass."""

        @property
        def available_tools(self) -> tuple[object, ...]:
            """Advertise the selected forged candidate without mutation."""
            return (ordinary, forged)

        def available_tools_for_enabled_tools(
            self, enable_tools: list[str]
        ) -> tuple[object, ...]:
            """Return the selected forged candidate for manager validation."""
            del enable_tools
            return (forged,)

        def with_enabled_tools(self, enable_tools: list[str]) -> ToolSet:
            """Return an impersonating final toolset after selection."""
            del enable_tools
            return ToolSet(tools=[forged])

    async def forged() -> dict[str, str]:
        """Represent an unsealed impersonating patch callable."""
        return {"forged": "true"}

    forged.__name__ = "patch.edit"
    selected = SelectorToolSet(tools=[ordinary])
    with pytest.raises(ValueError, match="sealed patch toolset"):
        ToolManager.create_instance(
            available_toolsets=[selected],
            enable_tools=["patch.edit"],
        )
    manager = ToolManager.create_instance(
        available_toolsets=[ToolSet(namespace="shell", tools=[ordinary])],
        enable_tools=["shell.ordinary"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES
        ),
    )

    async def execute() -> None:
        """Prove captured patch context is cleared before ordinary dispatch."""
        result = await manager.execute_call(
            ToolCall(id="ordinary-context", name="shell.ordinary"),
            ToolCallContext(patch_capability=object()),
        )
        assert isinstance(result, ToolCallResult)
        assert result.result == {"capability": False}

    run(execute())
    with pytest.raises(ValueError, match="sealed patch toolset"):
        manager._register_toolset(ToolSet(tools=[forged]))
    selected_plan = manager._registration_plans[0]
    forged_final_plan = replace(
        selected_plan,
        registrations=(
            replace(
                selected_plan.registrations[0],
                canonical_name="patch.edit",
            ),
        ),
    )
    with pytest.raises(ValueError, match="sealed patch toolset"):
        manager._validate_final_patch_registrations((forged_final_plan,))


def test_patch_phase_9_public_prepared_calls_revalidate_registry() -> None:
    """Reject forged prepared dispatch and strip ordinary patch authority."""
    service = _Service()
    observed: list[bool] = []

    class OrdinaryTool(Tool):
        """Report whether ordinary execution received patch authority.

        Returns:
            Whether the context retains patch authority.
        """

        def __init__(self) -> None:
            """Set the ordinary tool's public name."""
            Tool.__init__(self)
            self.__name__ = "ordinary"

        async def __call__(self, context: ToolCallContext) -> dict[str, bool]:
            """Return whether dispatch retained patch authority.

            Returns:
                Whether the context retains patch authority.
            """
            observed.append(context.patch_capability is not None)
            return {"capability": context.patch_capability is not None}

    ordinary = OrdinaryTool()

    async def forged() -> dict[str, str]:
        """Fail if a public forged prepared call reaches this callable."""
        raise AssertionError("forged callable must not execute")

    forged.__name__ = "patch.edit"
    toolset = _toolset(
        service,
        PatchCapabilitySnapshot(edit_available=True, apply_available=False),
    )
    manager = ToolManager.create_instance(
        available_toolsets=[
            toolset,
            ToolSet(namespace="shell", tools=[ordinary]),
        ],
        enable_tools=["patch.edit", "shell.ordinary"],
        settings=ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES
        ),
    )
    ordinary_descriptor = manager._descriptors["shell.ordinary"]

    async def execute() -> None:
        """Probe real, forged ordinary, and forged patch prepared calls."""
        ordinary_prepared = PreparedToolCall(
            call=ToolCall(
                id="prepared-ordinary",
                name="shell.ordinary",
                arguments={},
            ),
            callable=ordinary_descriptor.callable,
            descriptor=ordinary_descriptor,
            arguments={},
            context=ToolCallContext(patch_capability=toolset.capability),
        )
        ordinary_result = await manager.execute_prepared_call(
            ordinary_prepared
        )
        assert isinstance(ordinary_result, ToolCallResult)
        assert ordinary_result.result == {"capability": False}
        assert observed == [False]

        forged_ordinary = PreparedToolCall(
            call=ToolCall(
                id="forged-ordinary",
                name="shell.ordinary",
                arguments={},
            ),
            callable=forged,
            descriptor=ToolDescriptor(
                name="shell.ordinary",
                callable=forged,
            ),
            arguments={},
            context=ToolCallContext(),
        )
        ordinary_rejection = await manager.execute_prepared_call(
            forged_ordinary
        )
        assert isinstance(ordinary_rejection, ToolCallError)
        assert ordinary_rejection.error_type == "PreparedToolCallRejected"
        assert observed == [False]

        invalid_ordinary = PreparedToolCall(
            call=ToolCall(
                id="invalid-ordinary",
                name="shell.ordinary",
                arguments={"forged": True},
            ),
            callable=ordinary_descriptor.callable,
            descriptor=ordinary_descriptor,
            arguments={"forged": True},
            context=ToolCallContext(),
        )
        invalid_rejection = await manager.execute_prepared_call(
            invalid_ordinary
        )
        assert isinstance(invalid_rejection, ToolCallError)
        assert invalid_rejection.error_type == "PreparedToolCallRejected"
        assert observed == [False]

        stale_registry = PreparedToolCall(
            call=ToolCall(
                id="stale-registry",
                name="shell.removed",
                arguments={},
            ),
            callable=ordinary_descriptor.callable,
            descriptor=ToolDescriptor(
                name="shell.removed",
                callable=ordinary_descriptor.callable,
            ),
            arguments={},
            context=ToolCallContext(),
        )
        stale_rejection = await manager.execute_prepared_call(stale_registry)
        assert isinstance(stale_rejection, ToolCallError)
        assert stale_rejection.error_type == "PreparedToolCallRejected"
        assert observed == [False]

        forged_patch = PreparedToolCall(
            call=ToolCall(
                id="forged-patch",
                name="patch.edit",
                raw_arguments=(
                    b'{"path":"note.txt","edits":['
                    b'{"old_text":"old","new_text":"new"}]}'
                ),
            ),
            callable=forged,
            descriptor=ToolDescriptor(name="patch.edit", callable=forged),
            arguments={},
            context=ToolCallContext(patch_capability=toolset.capability),
        )
        patch_rejection = await manager.execute_prepared_call(forged_patch)
        assert isinstance(patch_rejection, ToolCallError)
        assert patch_rejection.error_type == "PreparedToolCallRejected"
        assert service.invocations == []

    run(execute())


def test_patch_phase_9_epoch_operations_limits_and_request_handles() -> None:
    """Bind raw parsing, lifetime, and SDK lifecycle to sealed authority."""
    service = _Service()
    binding = _binding(service)
    edit_only = replace(
        binding,
        policy=replace(
            binding.policy,
            enabled_operations=frozenset((OperationType.EDIT,)),
        ),
    )

    class Binder:
        """Return one edit-only binding with finite trusted limits."""

        async def bind(self) -> PatchRuntimeBinding:
            """Return the fixed already-probed binding."""
            return edit_only

    async def execute() -> None:
        """Reject stale, widened, forged, and oversized public invocations."""
        loader = PatchToolLoader(
            Binder(),
            activated_patch_test_profile(),
        )
        bundle = await loader.load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        toolset = bundle.toolset
        capability = toolset.capability
        assert not hasattr(capability, "_service")
        assert not hasattr(capability, "_owner")
        host = PatchSdkHost(service, capability)
        with pytest.raises(PatchToolError, match="unavailable"):
            await host.invoke_raw(
                OperationType.APPLY,
                b'{"patch":"*** Begin Patch v1\\n*** End Patch"}',
            )
        oversized = (
            b'{"path":"note.txt","edits":[{"old_text":"'
            + b"x" * 1030
            + b'","new_text":"y"}]}'
        )
        rejected = await bundle.manager.execute_call(
            ToolCall(
                id="effective-input-limit",
                name="patch.edit",
                raw_arguments=oversized,
            ),
            ToolCallContext(),
        )
        assert isinstance(rejected, ToolCallResult)
        assert rejected.result is not None
        assert rejected.result["code"] == "patch.invalid_request"
        assert service.invocations == []
        pending_service = _Service(pending=True)
        pending_toolset = await _toolset_async(
            pending_service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        pending_host = PatchSdkHost(
            pending_service,
            pending_toolset.capability,
        )
        with pytest.raises(PatchToolError, match="request handle"):
            await pending_host.review()
        pending = await pending_host.invoke_json(
            OperationType.EDIT,
            {
                "path": "note.txt",
                "edits": [{"old_text": "old", "new_text": "new"}],
            },
        )
        assert isinstance(pending, PatchPending)
        with pytest.raises(PatchToolError, match="pending handle"):
            await pending_host.await_terminal(
                replace(
                    pending,
                    correlation_id=PatchObserverCorrelationId.new(),
                )
            )
        handle = pending_host._handle
        assert isinstance(handle, PatchInvocationHandle)
        with pytest.raises(TypeError):
            copy(handle)
        toolset._snapshot = replace(toolset._snapshot, stale=True)
        with pytest.raises(PatchToolError, match="unavailable"):
            await host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "old", "new_text": "new"}],
                },
            )
        toolset._tools = []
        await loader.rebuild_if_stale(
            toolset,
            enable_tools=["patch.edit"],
        )
        with pytest.raises(PatchToolError, match="unavailable"):
            await host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "old", "new_text": "new"}],
                },
            )

    run(execute())


def test_patch_phase_9_lifecycle_handles_reject_cross_host_and_wrong_results() -> (  # noqa: E501
    None
):
    """Bind lifecycle operations and terminal results to one issued request."""

    class WrongTerminalService(_Service):
        """Return another request's terminal result after pending work."""

        def settlement_terminal(
            self,
            handle: PatchInvocationHandle,
            pending: PatchPending,
        ) -> Future[PatchResult]:
            """Return a mismatched fenced terminal result for rejection."""
            assert isinstance(handle, PatchInvocationHandle)
            assert isinstance(pending, PatchPending)
            return _settled_future(
                PatchResult(
                    1,
                    PatchRequestId("request_" + "b" * 16),
                    PatchPlanId("plan_" + "b" * 16),
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
            )

    async def execute() -> None:
        """Reject cross-host handles and a mismatched terminal request ID."""
        first_service = _Service()
        second_service = _Service()
        first_toolset = await _toolset_async(
            first_service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
        )
        second_toolset = await _toolset_async(
            second_service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
        )
        first_host = PatchSdkHost(first_service, first_toolset.capability)
        second_host = PatchSdkHost(second_service, second_toolset.capability)
        arguments = {
            "path": "note.txt",
            "edits": [{"old_text": "old", "new_text": "new"}],
        }
        assert isinstance(
            await first_host.invoke_json(OperationType.EDIT, arguments),
            PatchResult,
        )
        assert isinstance(
            await second_host.invoke_json(OperationType.EDIT, arguments),
            PatchResult,
        )
        foreign_handle = second_host._handle
        assert isinstance(foreign_handle, PatchInvocationHandle)
        first_host._handle = foreign_handle
        with pytest.raises(PatchToolError, match="request handle"):
            await first_host.review()

        wrong_service = WrongTerminalService(pending=True)
        wrong_toolset = await _toolset_async(
            wrong_service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
        )
        wrong_host = PatchSdkHost(wrong_service, wrong_toolset.capability)
        pending = await wrong_host.invoke_json(OperationType.EDIT, arguments)
        assert isinstance(pending, PatchPending)
        with pytest.raises(PatchToolError, match="request is invalid"):
            await wrong_host.await_terminal(pending)

    run(execute())


def test_patch_rejects_mismatched_lifecycle_and_reconciliation() -> None:
    """Reject lifecycle and recovery results outside a sealed request."""

    class LostResponseService(_Service):
        """Raise after dispatch and report another request during recovery."""

        async def invoke(
            self,
            operation: object,
            raw_arguments: bytes,
            capability: PatchInvocationCapability,
            request_id: PatchRequestId,
            correlation_id: PatchObserverCorrelationId,
        ) -> PatchResult:
            """Record dispatch before simulating a lost transport response."""
            await super().invoke(
                operation,
                raw_arguments,
                capability,
                request_id,
                correlation_id,
            )
            raise RuntimeError("response lost")

        def settlement_inspection(
            self, handle: PatchInvocationHandle
        ) -> Future[PatchResult]:
            """Return an unissued fenced result for the recovery path."""
            assert isinstance(handle, PatchInvocationHandle)
            return _settled_future(_result(PatchRequestId.new()))

    async def execute() -> None:
        """Reject identity substitutions and close the subscription."""
        arguments = {
            "path": "note.txt",
            "edits": [{"old_text": "old", "new_text": "new"}],
        }
        service = _Service()
        toolset = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
        )
        host = PatchSdkHost(service, toolset.capability)
        outcome = await host.invoke_json(OperationType.EDIT, arguments)
        assert isinstance(outcome, PatchResult)
        events = host.lifecycle()
        waiting = create_task(anext(events))
        await sleep(0)
        assert service.correlation_id is not None
        await service.lifecycle.emit(
            PatchLifecycleEvent(
                1,
                PatchEventId("event_" + "e" * 16),
                PatchObserverId("observer_" + "e" * 16),
                service.correlation_id,
                PatchRequestId.new(),
                SequenceNumber(1),
                LifecyclePhase.PLANNED,
            )
        )
        with pytest.raises(PatchToolError, match="lifecycle event"):
            await waiting
        await events.aclose()
        await service.lifecycle.close()

        lost_service = LostResponseService()
        lost_toolset = await _toolset_async(
            lost_service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
        )
        lost_host = PatchSdkHost(lost_service, lost_toolset.capability)
        with pytest.raises(PatchToolError, match="request is invalid"):
            await lost_host.invoke_json(OperationType.EDIT, arguments)

    run(execute())


def test_patch_phase_9_sdk_settlement_timeouts_preserve_pending_without_leaks() -> (  # noqa: E501
    None
):
    """Bound hung settlement calls without inventing a terminal result."""

    class HungSettlementService(_Service):
        """Expose a legacy cancellation coroutine that never cooperates."""

        def __post_init__(self) -> None:
            """Initialize durable pending and cancellation observations."""
            super().__post_init__()
            self.current: PatchPending | None = None
            self.cancel_started = Event()
            self.terminal_future: Future[PatchResult] | None = None

        async def invoke(
            self,
            operation: object,
            raw_arguments: bytes,
            capability: PatchInvocationCapability,
            request_id: PatchRequestId,
            correlation_id: PatchObserverCorrelationId,
        ) -> PatchPending:
            """Return the durable pending envelope for the test request."""
            outcome = await super().invoke(
                operation,
                raw_arguments,
                capability,
                request_id,
                correlation_id,
            )
            assert isinstance(outcome, PatchPending)
            self.current = outcome
            return outcome

        def settlement_inspection(
            self, handle: PatchInvocationHandle
        ) -> Future[PatchPending]:
            """Return durable nonterminal truth without a host task."""
            assert isinstance(handle, PatchInvocationHandle)
            assert self.current is not None
            return _settled_future(self.current)

        def settlement_terminal(
            self,
            handle: PatchInvocationHandle,
            pending: PatchPending,
        ) -> Future[PatchResult]:
            """Return a durable unresolved future owned by this service."""
            assert isinstance(handle, PatchInvocationHandle)
            assert pending == self.current
            if self.terminal_future is None:
                self.terminal_future = get_running_loop().create_future()
            return self.terminal_future

        async def cancel(self, handle: PatchInvocationHandle) -> PatchPending:
            """Ignore cancellation indefinitely."""
            assert isinstance(handle, PatchInvocationHandle)
            self.cancel_started.set()
            while True:
                try:
                    await Event().wait()
                except CancelledError:
                    continue

    async def execute() -> None:
        """Bound settlement without starting legacy service work."""
        service = HungSettlementService(pending=True)
        toolset = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
                settlement_duration=DurationTicks(1),
            ),
        )
        host = PatchSdkHost(service, toolset.capability)
        pending = await host.invoke_json(
            OperationType.EDIT,
            {
                "path": "note.txt",
                "edits": [{"old_text": "old", "new_text": "new"}],
            },
        )
        assert isinstance(pending, PatchPending)
        before = frozenset(
            task for task in all_tasks() if task is not current_task()
        )
        with pytest.raises(PatchToolError, match="settlement remains pending"):
            await host.await_terminal(pending)
        assert await host.cancel() == pending
        assert not service.cancel_started.is_set()
        await sleep(0)
        after = frozenset(
            task for task in all_tasks() if task is not current_task()
        )
        assert after == before

    run(execute())


def test_patch_phase_9_sdk_approval_updates_cancellation_pending_state() -> (
    None
):
    """Retain only the exact approval outcome for fail-closed cancellation."""

    class ApprovalStateService(_Service):
        """Return either pending or terminal approval without cancellation."""

        def __init__(self, approve_pending: bool) -> None:
            """Initialize one service with its scripted approval outcome."""
            self._approve_pending = approve_pending
            super().__init__(pending=True)

        def __post_init__(self) -> None:
            """Initialize the exact pending observation and legacy sentinel."""
            super().__post_init__()
            self.current: PatchPending | None = None
            self.cancel_started = Event()

        async def invoke(
            self,
            operation: object,
            raw_arguments: bytes,
            capability: PatchInvocationCapability,
            request_id: PatchRequestId,
            correlation_id: PatchObserverCorrelationId,
        ) -> PatchPending:
            """Return and retain the one durable request pending envelope."""
            outcome = await super().invoke(
                operation,
                raw_arguments,
                capability,
                request_id,
                correlation_id,
            )
            assert isinstance(outcome, PatchPending)
            self.current = outcome
            return outcome

        async def approve(
            self, handle: PatchInvocationHandle
        ) -> PatchResult | PatchPending:
            """Return the configured exact approval outcome."""
            assert isinstance(handle, PatchInvocationHandle)
            assert self.current is not None
            if self._approve_pending:
                return self.current
            return _result(self.current.request_id)

        async def cancel(self, handle: PatchInvocationHandle) -> PatchPending:
            """Expose a legacy coroutine that must never be started."""
            assert isinstance(handle, PatchInvocationHandle)
            self.cancel_started.set()
            await Event().wait()
            raise AssertionError("legacy cancellation unexpectedly settled")

    async def execute() -> None:
        """Verify approval replaces, then clears, the host pending envelope."""
        arguments = {
            "path": "note.txt",
            "edits": [{"old_text": "old", "new_text": "new"}],
        }
        pending_service = ApprovalStateService(approve_pending=True)
        pending_toolset = await _toolset_async(
            pending_service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
        )
        pending_host = PatchSdkHost(
            pending_service,
            pending_toolset.capability,
        )
        pending = await pending_host.invoke_json(OperationType.EDIT, arguments)
        assert isinstance(pending, PatchPending)
        approved_pending = await pending_host.approve()
        assert approved_pending is pending
        assert await pending_host.cancel() is pending
        assert not pending_service.cancel_started.is_set()

        terminal_service = ApprovalStateService(approve_pending=False)
        terminal_toolset = await _toolset_async(
            terminal_service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
        )
        terminal_host = PatchSdkHost(
            terminal_service,
            terminal_toolset.capability,
        )
        initial_pending = await terminal_host.invoke_json(
            OperationType.EDIT,
            arguments,
        )
        assert isinstance(initial_pending, PatchPending)
        terminal = await terminal_host.approve()
        assert isinstance(terminal, PatchResult)
        with pytest.raises(
            PatchToolError, match="cancellation is unavailable"
        ):
            await terminal_host.cancel()
        assert not terminal_service.cancel_started.is_set()

    run(execute())


def test_patch_phase_9_sealed_error_and_lifecycle_branches() -> None:
    """Exercise every sealed negative path without fabricating outcomes."""

    def pending(
        correlation: PatchObserverCorrelationId, marker: str
    ) -> PatchPending:
        """Return one correlation-bound pending outcome for a negative path."""
        return PatchPending(
            1,
            PatchPendingOperationId("pending_" + marker * 16),
            PatchRequestId("request_" + marker * 16),
            correlation,
            LifecyclePhase.SETTLEMENT_PENDING,
        )

    class InspectingService(_Service):
        """Return or raise one exact reconciliation outcome."""

        def __init__(
            self,
            outcome: PatchResult | PatchPending | BaseException,
        ) -> None:
            """Store the scripted reconciliation outcome."""
            super().__init__()
            self._outcome = outcome

        def settlement_inspection(
            self, handle: PatchInvocationHandle
        ) -> Future[PatchResult | PatchPending]:
            """Return or raise the scripted fenced inspection result."""
            assert isinstance(handle, PatchInvocationHandle)
            if isinstance(self._outcome, BaseException):
                future: Future[PatchResult | PatchPending] = (
                    get_running_loop().create_future()
                )
                future.set_exception(self._outcome)
                return future
            return _settled_future(self._outcome)

    class DispatchService(InspectingService):
        """Raise one post-handoff failure before reconciliation."""

        def __init__(
            self,
            invocation_error: BaseException,
            inspection: PatchResult | PatchPending | BaseException,
        ) -> None:
            """Store the dispatch failure and reconciliation response."""
            super().__init__(inspection)
            self._invocation_error = invocation_error

        async def invoke(
            self,
            operation: object,
            raw_arguments: bytes,
            capability: PatchInvocationCapability,
            request_id: PatchRequestId,
            correlation_id: PatchObserverCorrelationId,
        ) -> PatchResult:
            """Raise only after receiving an authenticated dispatch request."""
            del operation, raw_arguments, request_id, correlation_id
            assert isinstance(capability, PatchInvocationCapability)
            raise self._invocation_error

    async def execute() -> None:
        """Exercise invalid epochs, handles, and reconciliation paths."""
        raw = (
            b'{"path":"note.txt","edits":['
            b'{"old_text":"old","new_text":"new"}]}'
        )
        service = _Service()
        toolset = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        correlation = PatchObserverCorrelationId.new()
        with pytest.raises(PatchToolError, match="capability is invalid"):
            patch_toolset_module._bound_issue_invocation(
                object(),
                service,
                OperationType.EDIT,
                PatchRequestId.new(),
                correlation,
            )
        request_id = PatchRequestId("request_" + "a" * 16)
        handle = patch_toolset_module._bound_issue_invocation(
            toolset.capability,
            service,
            OperationType.EDIT,
            request_id,
            correlation,
        )
        assert isinstance(handle, PatchInvocationHandle)
        with pytest.raises(PatchToolError, match="request"):
            patch_toolset_module._bound_bind_invocation(
                handle,
                pending(PatchObserverCorrelationId.new(), "b"),
            )
        with pytest.raises(PatchToolError, match="handle is invalid"):
            patch_toolset_module._bound_bind_invocation(
                PatchInvocationHandle(object()),
                _result(),
            )
        with pytest.raises(TypeError, match="not copyable"):
            deepcopy(handle)
        with pytest.raises(TypeError, match="not serializable"):
            handle.__reduce_ex__(4)
        assert isinstance(
            await toolset._reconcile_after_dispatch(
                toolset.capability,
                handle,
                correlation,
                RuntimeError("dispatch lost"),
            ),
            PatchResult,
        )
        forged_toolset = object.__new__(PatchToolSet)
        forged_toolset._capability = object()  # type: ignore[assignment]
        forged_toolset._capability_owner = object()
        with pytest.raises(PatchToolError, match="capability is invalid"):
            PatchToolSet._revoke(forged_toolset)
        toolset._snapshot = replace(toolset._snapshot, stale=True)
        unavailable = await toolset.invoke_json(
            OperationType.EDIT,
            {"path": "note.txt", "edits": []},
            ToolCallContext(patch_capability=toolset.capability),
        )
        assert unavailable["code"] == "patch.capability_unavailable"
        with pytest.raises(PatchToolError, match="input limits"):
            patch_toolset_module._effective_input_limits(object())

        mismatched = InspectingService(
            pending(PatchObserverCorrelationId.new(), "c")
        )
        mismatch_toolset = await _toolset_async(
            mismatched,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        with pytest.raises(PatchToolError, match="reconciliation correlation"):
            mismatch_handle = patch_toolset_module._bound_issue_invocation(
                mismatch_toolset.capability,
                mismatched,
                OperationType.EDIT,
                PatchRequestId("request_" + "c" * 16),
                PatchObserverCorrelationId.new(),
            )
            await mismatch_toolset._reconcile_after_dispatch(
                mismatch_toolset.capability,
                mismatch_handle,
                PatchObserverCorrelationId.new(),
                RuntimeError("dispatch lost"),
            )
        for outcome, expected in (
            (CancelledError(), CancelledError),
            (RuntimeError("inspection lost"), PatchToolError),
        ):
            inspecting = InspectingService(outcome)
            inspected_toolset = await _toolset_async(
                inspecting,
                PatchCapabilitySnapshot(
                    edit_available=True,
                    apply_available=False,
                ),
            )
            with pytest.raises(expected):
                inspection_handle = (
                    patch_toolset_module._bound_issue_invocation(
                        inspected_toolset.capability,
                        inspecting,
                        OperationType.EDIT,
                        PatchRequestId.new(),
                        PatchObserverCorrelationId.new(),
                    )
                )
                await inspected_toolset._reconcile_after_dispatch(
                    inspected_toolset.capability,
                    inspection_handle,
                    PatchObserverCorrelationId.new(),
                    RuntimeError("dispatch lost"),
                )

        durable = pending(PatchObserverCorrelationId.new(), "d")
        preservation_service = _Service()
        preservation_toolset = await _toolset_async(
            preservation_service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
        )
        preservation_handle = patch_toolset_module._bound_issue_invocation(
            preservation_toolset.capability,
            preservation_service,
            OperationType.EDIT,
            durable.request_id,
            durable.correlation_id,
        )
        patch_toolset_module._bound_bind_invocation(
            preservation_handle,
            durable,
        )
        preservation_toolset._preserve_pending(
            preservation_handle,
            durable,
        )

        pending_service = _Service(pending=True)
        pending_toolset = await _toolset_async(
            pending_service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        pending_host = PatchSdkHost(
            pending_service,
            pending_toolset.capability,
        )
        with pytest.raises(PatchToolError, match="pending handle"):
            await pending_host.await_terminal(object())  # type: ignore[arg-type]
        first = await pending_host.invoke_raw(OperationType.EDIT, raw)
        second = await pending_host.invoke_raw(OperationType.EDIT, raw)
        assert isinstance(first, PatchPending)
        assert isinstance(second, PatchPending)
        with pytest.raises(PatchToolError, match="pending handle"):
            await pending_host.await_terminal(first)

        cancelled_dispatch = DispatchService(CancelledError(), _result())
        cancelled_toolset = await _toolset_async(
            cancelled_dispatch,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        cancelled_host = PatchSdkHost(
            cancelled_dispatch,
            cancelled_toolset.capability,
        )
        with pytest.raises(CancelledError):
            await cancelled_host.invoke_raw(OperationType.EDIT, raw)

        wrong_reconciled_dispatch = DispatchService(
            RuntimeError("dispatch lost"),
            _result(),
        )
        wrong_reconciled_toolset = await _toolset_async(
            wrong_reconciled_dispatch,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        wrong_reconciled_host = PatchSdkHost(
            wrong_reconciled_dispatch,
            wrong_reconciled_toolset.capability,
        )
        with pytest.raises(PatchToolError, match="request is invalid"):
            await wrong_reconciled_host.invoke_raw(OperationType.EDIT, raw)
        for outcome, expected in (
            (CancelledError(), CancelledError),
            (RuntimeError("inspection lost"), PatchToolError),
        ):
            inspecting = DispatchService(
                RuntimeError("dispatch lost"), outcome
            )
            inspected_toolset = await _toolset_async(
                inspecting,
                PatchCapabilitySnapshot(
                    edit_available=True,
                    apply_available=False,
                ),
            )
            host = PatchSdkHost(inspecting, inspected_toolset.capability)
            with pytest.raises(expected):
                await host.invoke_raw(OperationType.EDIT, raw)

    run(execute())


def test_patch_phase_9_settlement_rejections_preserve_request_authority() -> (
    None
):
    """Reject unverified settlement state without releasing its request."""

    class HungSettlementService(_Service):
        """Return only fenced futures which never manufacture a result."""

        def __post_init__(self) -> None:
            """Record every pending service-owned settlement future."""
            super().__post_init__()
            self.inspection_result: PatchResult | PatchPending | None = None
            self.inspections: list[Future[PatchResult | PatchPending]] = []
            self.terminals: list[Future[PatchResult]] = []

        def settlement_inspection(
            self, handle: PatchInvocationHandle
        ) -> Future[PatchResult | PatchPending]:
            """Return one unresolved inspection future for the exact handle."""
            assert isinstance(handle, PatchInvocationHandle)
            if self.inspection_result is not None:
                return _settled_future(self.inspection_result)
            future: Future[PatchResult | PatchPending] = (
                get_running_loop().create_future()
            )
            self.inspections.append(future)
            return future

        def settlement_terminal(
            self,
            handle: PatchInvocationHandle,
            pending: PatchPending,
        ) -> Future[PatchResult]:
            """Return one unresolved terminal future for the exact request."""
            assert isinstance(handle, PatchInvocationHandle)
            assert isinstance(pending, PatchPending)
            future: Future[PatchResult] = get_running_loop().create_future()
            self.terminals.append(future)
            return future

    async def execute() -> None:
        """Exercise timeouts, forged correlations, and invalid future ports."""
        service = HungSettlementService(pending=True)
        snapshot = PatchCapabilitySnapshot(
            edit_available=True,
            apply_available=False,
            settlement_duration=DurationTicks(1),
        )
        toolset = await _toolset_async(service, snapshot)
        request_id = PatchRequestId("request_" + "a" * 16)
        correlation = PatchObserverCorrelationId.new()
        handle = patch_toolset_module._bound_issue_invocation(
            toolset.capability,
            service,
            OperationType.EDIT,
            request_id,
            correlation,
        )
        assert isinstance(handle, PatchInvocationHandle)
        with pytest.raises(PatchToolError, match="correlation"):
            patch_toolset_module._bound_bind_invocation(
                handle,
                PatchPending(
                    1,
                    PatchPendingOperationId("pending_" + "b" * 16),
                    request_id,
                    PatchObserverCorrelationId.new(),
                    LifecyclePhase.SETTLEMENT_PENDING,
                ),
            )

        pending = PatchPending(
            1,
            PatchPendingOperationId("pending_" + "a" * 16),
            request_id,
            correlation,
            LifecyclePhase.SETTLEMENT_PENDING,
        )
        patch_toolset_module._bound_bind_invocation(handle, pending)
        with pytest.raises(PatchToolError, match="settlement remains pending"):
            await toolset._await_terminal(handle, pending)
        with pytest.raises(
            PatchToolError, match="reconciliation remains pending"
        ):
            await toolset._reconcile_after_dispatch(
                toolset.capability,
                handle,
                correlation,
                RuntimeError("transport response lost"),
            )
        with pytest.raises(
            PatchToolError, match="selection must be a sequence"
        ):
            toolset._selected_tools("patch.edit")

        host = PatchSdkHost(service, toolset.capability)
        pending_outcome = await host.invoke_raw(
            OperationType.EDIT,
            b'{"path":"note.txt","edits":['
            b'{"old_text":"old","new_text":"new"}]}',
        )
        assert isinstance(pending_outcome, PatchPending)
        with pytest.raises(PatchToolError, match="inspection remains pending"):
            await host.inspect()
        host_handle = host._handle
        assert isinstance(host_handle, PatchInvocationHandle)
        service.inspection_result = _result(service.request_id)
        recovered = await host._reconcile_after_dispatch(
            host_handle,
            RuntimeError("transport response lost"),
        )
        assert isinstance(recovered, PatchResult)
        service.inspection_result = None
        with pytest.raises(
            PatchToolError, match="reconciliation remains pending"
        ):
            await host._reconcile_after_dispatch(
                host_handle,
                RuntimeError("transport response lost"),
            )
        with pytest.raises(
            PatchToolError, match="settlement future is invalid"
        ):
            await patch_toolset_module._await_settlement_future(
                MagicMock(),
                DurationTicks(1),
            )

    run(execute())


def test_patch_phase_9_pending_cancellation_and_orchestrator_guards() -> None:
    """Keep a cancelled pending request durable and out of generic guards."""

    class PendingService(_Service):
        """Hold one durable pending result until the owning task cancels."""

        def __init__(self) -> None:
            """Initialize durable-pending cancellation observations."""
            super().__init__()
            self.started = Event()
            self.current: PatchPending | None = None
            self.terminal_future: Future[PatchResult] | None = None

        async def invoke(
            self,
            operation: object,
            raw_arguments: bytes,
            capability: PatchInvocationCapability,
            request_id: PatchRequestId,
            correlation_id: PatchObserverCorrelationId,
        ) -> PatchPending:
            """Return the only correlation-bound pending envelope."""
            assert isinstance(capability, PatchInvocationCapability)
            self.invocations.append((operation, raw_arguments))
            self.current = PatchPending(
                1,
                PatchPendingOperationId("pending_" + "b" * 16),
                request_id,
                correlation_id,
                LifecyclePhase.SETTLEMENT_PENDING,
            )
            return self.current

        def settlement_terminal(
            self,
            handle: PatchInvocationHandle,
            pending: PatchPending,
        ) -> Future[PatchResult]:
            """Return one service-owned pending terminal future."""
            assert isinstance(handle, PatchInvocationHandle)
            assert pending == self.current
            self.started.set()
            if self.terminal_future is None:
                self.terminal_future = get_running_loop().create_future()
            return self.terminal_future

        async def review(
            self, handle: PatchInvocationHandle
        ) -> dict[str, object]:
            """Return a content-free review projection."""
            assert isinstance(handle, PatchInvocationHandle)
            return {"kind": "review"}

        async def approve(self, handle: PatchInvocationHandle) -> PatchResult:
            """Return a terminal approval projection."""
            assert isinstance(handle, PatchInvocationHandle)
            assert self.current is not None
            return _result(self.current.request_id)

        def subscribe(self, handle: PatchInvocationHandle) -> object:
            """Return the content-free lifecycle stream."""
            assert isinstance(handle, PatchInvocationHandle)
            return self.lifecycle.subscribe(handle)

    service = PendingService()
    raw = b'{"path":"note.txt","edits":[{"old_text":"old","new_text":"new"}]}'

    async def execute() -> None:
        """Cancel only after durable handoff and bypass raw repeat guards."""
        pending_toolset = await _toolset_async(
            service,
            PatchCapabilitySnapshot(
                edit_available=True,
                apply_available=False,
            ),
        )
        assert pending_toolset._service is service
        pending_manager = ToolManager.create_instance(
            available_toolsets=[pending_toolset],
            enable_tools=["patch.edit"],
            settings=ToolManagerSettings(
                execution_mode=ToolManagerExecutionMode.OUTCOMES,
                avoid_repetition=True,
            ),
        )
        assert pending_manager._toolsets is not None
        assert pending_manager._toolsets[0]._service is service
        first = create_task(
            pending_manager.execute_call(
                ToolCall(
                    id="pending-cancel", name="patch.edit", raw_arguments=raw
                ),
                ToolCallContext(),
            )
        )
        await wait_for(service.started.wait(), timeout=1)
        assert not first.done(), first.result()
        first.cancel()
        with pytest.raises(CancelledError):
            await first
        assert service.current is not None
        terminal_service = _Service()
        terminal_manager = ToolManager.create_instance(
            available_toolsets=[
                await _toolset_async(
                    terminal_service,
                    PatchCapabilitySnapshot(
                        edit_available=True,
                        apply_available=False,
                    ),
                )
            ],
            enable_tools=["patch.edit"],
            settings=ToolManagerSettings(
                execution_mode=ToolManagerExecutionMode.OUTCOMES,
                avoid_repetition=True,
            ),
        )
        response = object.__new__(OrchestratorResponse)
        response._tool_manager = terminal_manager
        response._block_repeated_tool_calls = True
        response._attempted_call_signatures = {
            OrchestratorResponse._call_signature(
                ToolCall(id="prior", name="patch.edit", raw_arguments=raw)
            )
        }
        second = await response._execute_tool_call(
            ToolCall(
                id="repeat-after-pending", name="patch.edit", raw_arguments=raw
            ),
            ToolCallContext(),
            confirm=False,
        )
        assert isinstance(second, ToolCallResult)
        assert len(terminal_service.invocations) == 1
        response._tool_cycle_signatures = {"already-observed"}
        response._maximum_tool_cycles = 0
        response._tool_cycle_count = 0
        response._consecutive_non_executed_cycles = 0
        assert response._should_continue_tool_cycle(
            [Message(role=MessageRole.TOOL, content="patch")],
            [second],
        )
        assert response._tool_cycle_signatures == {"already-observed"}
        assert response._tool_cycle_count == 0

    run(execute())


def test_patch_phase_9_activation_and_sdk_review_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject structural activation hooks and forged SDK review values."""

    class MissingActivation:
        """Expose no activation member after an independent seal check."""

    class SynchronousActivation:
        """Expose a synchronous result that cannot mint authority."""

        def activate(self, binding: PatchRuntimeBinding) -> None:
            """Return no awaitable activation receipt for this binding."""
            del binding

    class NonJsonReviewService(_Service):
        """Return a review that cannot be serialized into a sealed binding."""

        async def review(
            self, handle: PatchInvocationHandle
        ) -> dict[str, object]:
            """Return a value rejected by canonical JSON encoding."""
            assert isinstance(handle, PatchInvocationHandle)
            return {"not_json": float("nan")}

    class PendingReviewService(_Service):
        """Preserve the pending outcome until its review is validated."""

        def settlement_inspection(
            self, handle: PatchInvocationHandle
        ) -> Future[PatchResult | PatchPending]:
            """Return a pending invocation instead of a terminal view."""
            assert isinstance(handle, PatchInvocationHandle)
            assert self.request_id is not None
            assert self.correlation_id is not None
            return _settled_future(
                PatchPending(
                    1,
                    PatchPendingOperationId("pending_" + "b" * 16),
                    self.request_id,
                    self.correlation_id,
                    LifecyclePhase.SETTLEMENT_PENDING,
                )
            )

    async def scenario() -> None:
        binding = _binding(_Service())
        assert (
            await patch_toolset_module._activate_sealed_factory(
                object(), binding
            )
            is None
        )
        monkeypatch.setattr(
            patch_toolset_module,
            "_is_sealed_activation_factory",
            lambda _: True,
        )
        assert (
            await patch_toolset_module._activate_sealed_factory(
                MissingActivation(), binding
            )
            is None
        )
        assert (
            await patch_toolset_module._activate_sealed_factory(
                SynchronousActivation(), binding
            )
            is None
        )
        monkeypatch.undo()

        invalid_service = NonJsonReviewService(pending=True)
        invalid_toolset = await _toolset_async(
            invalid_service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        invalid_host = PatchSdkHost(
            invalid_service, invalid_toolset.capability
        )
        assert isinstance(
            await invalid_host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "old", "new_text": "new"}],
                },
            ),
            PatchPending,
        )
        with pytest.raises(PatchToolError):
            await invalid_host.prepare_approval_review()

        pending_service = PendingReviewService(pending=True)
        pending_toolset = await _toolset_async(
            pending_service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        pending_host = PatchSdkHost(
            pending_service, pending_toolset.capability
        )
        assert isinstance(
            await pending_host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "old", "new_text": "new"}],
                },
            ),
            PatchPending,
        )
        review = await pending_host.prepare_approval_review()
        pending_host.validate_approval_review(review)

        terminal_service = _Service()
        terminal_toolset = await _toolset_async(
            terminal_service,
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
        )
        terminal_host = PatchSdkHost(
            terminal_service, terminal_toolset.capability
        )
        assert isinstance(
            await terminal_host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "old", "new_text": "new"}],
                },
            ),
            PatchResult,
        )
        with pytest.raises(PatchToolError):
            await terminal_host.prepare_approval_review()

    run(scenario())
    with pytest.raises(PatchToolError):
        patch_toolset_module.PatchSdkInvocationReview(cast(Never, object()))
    review = object.__new__(patch_toolset_module.PatchSdkInvocationReview)
    assert repr(review) == "PatchSdkInvocationReview(<opaque>)"
    with pytest.raises(PatchToolError):
        copy(review)
    with pytest.raises(PatchToolError):
        deepcopy(review)
    with pytest.raises(PatchToolError):
        review.__reduce__()
    with pytest.raises(PatchToolError):
        review.__reduce_ex__(4)


def test_patch_phase_9_loader_unwinds_all_activation_cleanup_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve construction failure while every cleanup layer fails."""

    class Resource:
        """Model one loader-owned async resource whose close fails."""

        async def __aenter__(self) -> "Resource":
            """Return the resource without a synchronous ownership path."""
            return self

        async def __aexit__(self, *arguments: object) -> None:
            """Raise after recording the loader's cleanup attempt."""
            del arguments
            raise RuntimeError("resource cleanup failed")

    service = _Service()
    binding = replace(_binding(service), owned_resources=(Resource(),))

    class Binder:
        """Return the one resource-owning binding for this loader."""

        async def bind(self) -> PatchRuntimeBinding:
            """Return the bound service and its owned resource."""
            return binding

    async def broken_exit(self: PatchToolSet, *arguments: object) -> None:
        """Fail after the loader has constructed its exact toolset."""
        del self, arguments
        raise RuntimeError("toolset cleanup failed")

    def broken_manager(**kwargs: object) -> ToolManager:
        """Fail after construction so the loader enters all cleanup layers."""
        del kwargs
        raise RuntimeError("manager construction failed")

    def broken_revoke(_: PatchToolSet) -> None:
        """Fail while revoking an incompletely closed toolset capability."""
        raise RuntimeError("revoke failed")

    async def broken_deactivate(self: PatchActivationRuntime) -> object:
        """Fail while deactivating the partially constructed runtime."""
        del self
        raise RuntimeError("activation cleanup failed")

    monkeypatch.setattr(PatchToolSet, "__aexit__", broken_exit)
    monkeypatch.setattr(PatchToolSet, "_revoke", staticmethod(broken_revoke))
    monkeypatch.setattr(ToolManager, "create_instance", broken_manager)
    monkeypatch.setattr(
        PatchActivationRuntime, "deactivate", broken_deactivate
    )
    loader = PatchToolLoader(Binder(), activated_patch_test_profile())

    async def scenario() -> None:
        """Retain the original manager error through all cleanup failures."""
        with pytest.raises(RuntimeError, match="manager construction failed"):
            await loader.load(enable_tools=["patch.edit"])

    run(scenario())


def test_patch_phase_9_loader_deactivates_an_unvalidated_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deactivate a non-null runtime that fails exact factory validation."""

    class Runtime:
        """Record the loader's required deactivation of this fake runtime."""

        def __init__(self) -> None:
            """Initialize an unvalidated runtime that has not been closed."""
            self.deactivated = False

        async def deactivate(self) -> None:
            """Record cleanup before fallback loading continues."""
            self.deactivated = True

    service = _Service()
    runtime = Runtime()

    class Binder:
        """Return the selected local binding for this exact loader."""

        async def bind(self) -> PatchRuntimeBinding:
            """Return the service-bound durable host handshake."""
            return _binding(service)

    async def activate(
        _factory: object, _binding: PatchRuntimeBinding
    ) -> object:
        """Return the runtime whose independent validation will fail."""
        return runtime

    monkeypatch.setattr(
        patch_toolset_module, "_activate_sealed_factory", activate
    )
    monkeypatch.setattr(
        patch_toolset_module, "_validates_activation_runtime", lambda *_: False
    )
    loader = PatchToolLoader(Binder(), activated_patch_test_profile())

    async def scenario() -> None:
        """Fall back to ordinary tools after deactivating the fake runtime."""
        bundle = await loader.load(enable_tools=["patch.edit"])
        assert bundle.toolset is None

    run(scenario())
    assert runtime.deactivated


def test_patch_phase_9_toolset_constructor_rejects_unissued_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid construction before a caller can claim a capability."""
    with pytest.raises(PatchToolError):
        PatchToolSet(
            cast(object, object()),
            cast(PatchCapabilitySnapshot, object()),
            runtime_binding=cast(PatchRuntimeBinding, object()),
            activation_runtime=object(),
            activation_factory=object(),
        )
    monkeypatch.setattr(
        patch_toolset_module,
        "_validates_activation_runtime",
        lambda *_: True,
    )
    with pytest.raises(PatchToolError, match="async-only"):
        PatchToolSet(
            cast(object, object()),
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            runtime_binding=cast(PatchRuntimeBinding, object()),
            activation_runtime=object(),
            activation_factory=object(),
            owned_resources=(object(),),
        )
    with pytest.raises(PatchToolError, match="trusted loader"):
        PatchToolSet(
            cast(object, object()),
            PatchCapabilitySnapshot(
                edit_available=True, apply_available=False
            ),
            runtime_binding=cast(PatchRuntimeBinding, object()),
            activation_runtime=object(),
            activation_factory=object(),
        )


def test_patch_phase_9_requirements(tmp_path: Path) -> None:
    """Exercise the complete public patch-tool contract surface."""
    snapshot = PatchCapabilitySnapshot(
        edit_available=True,
        apply_available=False,
    )
    assert snapshot.tool_names() == ("patch.edit",)
    assert PatchToolSet is not None
    test_patch_phase_9_static_public_tools_and_selection()
    test_patch_phase_9_loader_binds_once_and_requires_test_host()
    test_patch_phase_9_manager_bypasses_generic_hooks_and_confirmation()
    test_patch_phase_9_malformed_provider_arguments_cannot_fall_back()
    test_patch_phase_9_admission_cancellation()
    test_patch_phase_9_pending_is_never_a_generic_tool_result()
    test_patch_phase_9_pending_blocks_one_agent_branch_until_reinjection()
    test_patch_phase_9_provider_raw_ingress_preserves_duplicate_evidence()
    test_patch_phase_9_raw_retention_and_display_are_closed()
    test_patch_phase_9_context_capability_is_loader_bound()
    test_patch_phase_9_stale_rebuild_and_strict_registration()
    test_patch_phase_9_direct_sdk_host_lifecycle_subscription()
    test_patch_phase_9_public_boundary_rejects_invalid_and_stale_state()
    test_patch_phase_9_admission_failure_raw_errors_and_async_resources()
    test_patch_phase_9_admission_cleanup_and_partial_entry_revoke()
    test_patch_phase_9_toolset_error_projection_and_lifecycle_cleanup()
    test_patch_phase_9_public_prepared_calls_revalidate_registry()
    test_patch_phase_9_lifecycle_handles_reject_cross_host_and_wrong_results()
    test_patch_rejects_mismatched_lifecycle_and_reconciliation()
    test_patch_phase_9_sdk_settlement_timeouts_preserve_pending_without_leaks()
    test_patch_phase_9_sdk_approval_updates_cancellation_pending_state()
    test_patch_e2e_010_direct_sdk_edit_reviews_approves_and_reads(tmp_path)
    test_patch_e2e_011_agent_json_apply_reinjects_and_reads(tmp_path)
    test_patch_e2e_012_stream_failures_and_handshakes_never_write()
    test_patch_e2e_013_detached_sdk_pending_resumes_one_branch()
    test_patch_e2e_014_json_parity_and_native_replay_are_inert()
    test_patch_phase_9_active_audience_privacy_matrix()
