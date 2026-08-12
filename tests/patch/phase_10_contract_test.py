"""Exercise the selected persistent sandbox patch runtime."""

from asyncio import (
    CancelledError,
    Event,
    Task,
    create_task,
    run,
    sleep,
    wait_for,
)
from asyncio.subprocess import Process
from base64 import b64encode
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from hmac import digest
from importlib import util as importlib_util
from inspect import getclosurevars
from io import BytesIO
from json import dumps, loads
from logging import getLogger
from os import close, fstat, mkfifo
from os import open as open_fd
from pathlib import Path
from runpy import run_path
from subprocess import run as run_process
from sys import platform as sys_platform
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from avalan.agent.loader import OrchestratorLoader
from avalan.entities import (
    Message,
    OrchestratorSettings,
    ToolCall,
    ToolCallContext,
    ToolCallError,
    ToolCallResult,
)
from avalan.isolation import (
    IsolationEffectiveSettings,
    IsolationMode,
    IsolationProfileSelection,
    IsolationSettings,
    IsolationToolRuntimeSettings,
    SandboxProfileSelection,
    trusted_isolation_source,
)
from avalan.model.call import ModelCall
from avalan.model.deterministic import (
    DeterministicModelManager,
    DeterministicToolPlan,
)
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.patch import coordinator as coordinator_module
from avalan.patch import pgsql_store as pgsql_store_module
from avalan.patch import rooted_worker as rooted_worker_module
from avalan.patch import sandbox_commit as sandbox_commit_module
from avalan.patch import sandbox_wire as sandbox_wire_module
from avalan.patch import sandbox_worker as sandbox_worker_module
from avalan.patch import toolset as patch_toolset_module
from avalan.patch.coordinator import (
    ArtifactJournal,
    CommitLease,
    CoordinatorError,
    CoordinatorErrorCode,
    JournalStep,
    LockFootprint,
    RetransmissionKey,
    RootedCommandAuthorityValidator,
    RootedSandboxCommitChannel,
    RootedSandboxCommitWorker,
    SealedCommitCommand,
    SettlementJournal,
    WorkerReport,
    WorkerState,
    _rooted_sandbox_endpoint,
    _sandbox_worker_for_endpoint,
)
from avalan.patch.domain import (
    AlgorithmDigest,
    ApprovalMode,
    ArtifactState,
    ByteSize,
    Capability,
    CommitStepState,
    CommitTruth,
    ContextKind,
    DurationTicks,
    ErrorStage,
    ExpiryTick,
    LifecyclePhase,
    LineageState,
    LogicalPath,
    MutationState,
    OperationType,
    PatchApprovalId,
    PatchCommitOwnerId,
    PatchContextId,
    PatchDiagnostic,
    PatchDomainId,
    PatchErrorCode,
    PatchExecutionId,
    PatchFingerprint,
    PatchGrantId,
    PatchLimits,
    PatchLineageId,
    PatchObserverCorrelationId,
    PatchPending,
    PatchPendingOperationId,
    PatchPlanId,
    PatchProtocolId,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PatchStepId,
    PatchTargetId,
    PatchWorkspaceId,
    PostconditionState,
    RequestedEffectOccurrence,
    Retryability,
    SequenceNumber,
    WorkspaceChange,
)
from avalan.patch.durable_approval import (
    HmacDurableApprovalAuthority,
    PhaseFiveDurableApprovalIssuer,
)
from avalan.patch.durable_store import (
    DurableApproval,
    DurableApprovalVerifier,
    DurableCommitClaimState,
    DurableCommitLease,
    DurableJournal,
    DurableJournalCursor,
    DurablePatchStore,
    DurablePatchStoreBinding,
    DurablePendingRequest,
    DurablePlanReference,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableReservation,
    DurableRetentionAuthorizer,
    DurableRetentionEnvelopeValidator,
    DurableStepBinding,
    DurableStoreError,
    DurableStoreErrorCode,
    DurableWorkerBinding,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.pgsql_store import (
    PgsqlDurablePatchStore,
    PgsqlDurablePatchStoreFactory,
    PgsqlDurablePatchStoreSettings,
)
from avalan.patch.planner import (
    BoundedPlannerWorker,
    PlannerFacade,
    PlannerLimits,
)
from avalan.patch.policy import (
    ApprovalClock,
    ApprovalDecisionState,
    ApprovalRequirements,
    ApprovalService,
    BrokerDecision,
    CapabilityMode,
    ExecutionSubject,
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PlanReviewRequest,
    PolicyBrokerId,
    PolicyPathSelector,
    PolicyReviewerRole,
    PolicyRevision,
    PolicyRouteId,
    PolicyRule,
    PreauthorizationClass,
    ReviewerDecision,
    RuntimeGrantStore,
    SealedPlan,
    TrustedPatchPolicy,
)
from avalan.patch.sandbox_commit import (
    _MESSAGE_VERSION,
    SandboxChannelId,
    SandboxCommitTarget,
    SandboxContextLifetimeId,
    SandboxInspectionTarget,
    SandboxPatchRuntime,
    SandboxPatchRuntimeBinder,
    SandboxPatchRuntimeContext,
    SandboxPatchRuntimeSettings,
    SandboxPatchSdkService,
    SandboxPatchServiceConfiguration,
    SandboxPatchServiceFactory,
    SandboxSessionId,
    SandboxWorkerImplementationId,
    SandboxWorkerProtocolVersion,
    _identity_payload,
    _response_payload,
    _SandboxDurableCommandAuthority,
    sandbox_protocol_id,
)
from avalan.patch.sandbox_worker import (
    _child_request,
)
from avalan.patch.sandbox_worker import (
    _RuntimeChildConfig as _WorkerChildConfig,
)
from avalan.patch.target import (
    EphemeralWorkerWitness,
    FileIdentity,
    InspectionRequest,
    LocalPlatformProfile,
    ResolvedMutationScope,
    ScopeSelection,
    TargetErrorCode,
    TargetHandshake,
    TargetIdentity,
    TargetInspectionError,
    TargetPrimitive,
    _filesystem_id,
    _open_directory,
    _ProtectedMetadata,
    _root_mount_id,
)
from avalan.patch.toolset import (
    PatchApprovalBinding,
    PatchCoordinatorBinding,
    PatchInvocationCapability,
    PatchInvocationHandle,
    PatchPersistenceBinding,
    PatchRuntimeBinding,
    PatchSdkHost,
    PatchSdkService,
    PatchTestHostProfile,
    PatchToolError,
    PatchToolLoader,
    PatchToolSet,
    PatchToolSettings,
    project_model_result,
)
from avalan.pgsql import PgsqlCursor, PgsqlRow
from avalan.sandbox import (
    BubblewrapSandboxBackend,
    SandboxBackend,
    SandboxBackendProbeResult,
    SandboxExecutionPlan,
    SandboxPlanRequest,
    SandboxPlanRequestKind,
    SandboxResultStatus,
    SeatbeltSandboxBackend,
)
from avalan.tool.code import HAS_CODE_DEPENDENCIES, CodeTool
from avalan.tool.context import ToolSettingsContext
from avalan.tool.shell import ShellGitToolSettings, ShellToolSettings


def _limits() -> PatchLimits:
    """Return bounded limits for a selected sandbox context."""
    return PatchLimits(
        ByteSize(65_536),
        ByteSize(16),
        ByteSize(512),
        ByteSize(16),
        ByteSize(32),
        ByteSize(65_536),
        ByteSize(65_536),
        ByteSize(65_536),
        DurationTicks(10),
        DurationTicks(10),
        DurationTicks(10),
    )


def _plan(
    root: Path,
    namespace: Path,
    *,
    ordinary_write_roots: list[str] | None = None,
    backend_name: str | None = None,
) -> SandboxExecutionPlan:
    """Build the real ordinary read-only sandbox execution plan."""
    backend = backend_name or _native_backend_name()
    raw = {
        "mode": "sandbox",
        "sandbox": {
            "backend": backend,
            "default_profile": "patch-context",
            "allowed_profiles": ["patch-context"],
            "profiles": {
                "patch-context": {
                    "trusted_executables": [
                        "/bin/cat",
                        "/bin/sh",
                        "/usr/bin/git",
                    ],
                    "executable_search_roots": ["/bin", "/usr/bin"],
                    "read_roots": [str(root)],
                    "write_roots": ordinary_write_roots or [],
                    "deny_roots": [],
                    "scratch_roots": [str(namespace)],
                    "output_roots": [],
                    "environment": {"variables": {}, "allowlist": []},
                    "network": {"mode": "none", "egress_allowlist": []},
                    "resources": {"timeout_seconds": 10, "pids": None},
                    "output": {
                        "max_stdout_bytes": 4096,
                        "max_stderr_bytes": 4096,
                        "allow_artifacts": False,
                        "max_artifact_bytes": 0,
                    },
                    "child_processes": (
                        "deny" if backend == "seatbelt" else "allow"
                    ),
                    "inherited_fds": "stdio",
                    "cleanup": "delete",
                }
            },
            "profile_registry_id": "patch-context",
            "policy_version": "patch-runtime-v2",
        },
    }
    settings = IsolationSettings.from_dict(
        raw,
        source=trusted_isolation_source("sdk"),
    ).select_profile(
        IsolationProfileSelection(
            mode=IsolationMode.SANDBOX,
            profile="patch-context",
            required=True,
        )
    )
    assert settings.sandbox is not None
    return SandboxExecutionPlan(
        request=SandboxPlanRequest(
            request_kind=SandboxPlanRequestKind.AGENT_SESSION,
            logical_name="agent",
            command="/bin/sh",
            argv=("/bin/sh", "-c", "exit 0"),
            cwd=str(root),
        ),
        settings=settings.sandbox,
    )


def _settings(
    root: Path,
    namespace: Path,
    *,
    backend_name: str | None = None,
) -> SandboxPatchRuntimeSettings:
    """Create one trusted settings value from the selected execution plan."""
    descriptor = _open_directory(root)
    try:
        status = fstat(descriptor)
        filesystem_id = _filesystem_id(descriptor)
        mount_id = _root_mount_id(descriptor, status)
    finally:
        close(descriptor)
    backend = backend_name or _native_backend_name()
    implementation = SandboxWorkerImplementationId(backend + "-runtime-v2")
    plan = _plan(root, namespace, backend_name=backend)
    token = AlgorithmDigest.from_bytes(str(root).encode()).value[:16]
    context = SandboxPatchRuntimeContext(
        TargetIdentity(
            PatchContextId("context_" + token),
            PatchWorkspaceId("workspace_" + token),
            PatchDomainId("domain_" + token),
            PatchTargetId("target_" + token),
            sandbox_protocol_id(
                SandboxWorkerProtocolVersion("sandbox-patch-runtime-v2")
            ),
            filesystem_id,
            mount_id,
            "policy-v2",
            "persistent-lease-v2",
            PatchApprovalId("approval_" + token),
            implementation,
        ),
        _limits(),
        ByteSize(65_536),
        None,
        SandboxChannelId("seatbelt-patch-channel-v2"),
        SandboxContextLifetimeId("seatbelt-patch-context-v2"),
        implementation,
    )
    return SandboxPatchRuntimeSettings(plan, context)


def _agent_settings() -> OrchestratorSettings:
    """Select patch, ordinary shell, and code through the real loader."""
    return OrchestratorSettings(
        agent_id=uuid4(),
        orchestrator_type=None,
        agent_config={"system": "Use the selected tools exactly as asked."},
        uri="ai://local/sandbox-agent-cycle",
        engine_config={},
        tools=[
            "patch.edit",
            "shell.cat",
            "shell.git_restore",
            "code.run",
        ],
        call_options={"maximum_tool_cycles": 3},
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
        log_events=False,
    )


def _agent_tool_settings(
    root: Path,
    binder: SandboxPatchRuntimeBinder,
) -> ToolSettingsContext:
    """Return trusted settings that select the same sandbox profile."""
    plan = binder.runtime.profile.execution_plan
    return ToolSettingsContext(
        patch=PatchToolSettings(
            binder,
            PatchTestHostProfile(enabled=True, authenticated=True),
        ),
        shell=ShellToolSettings(
            execution_mode="sandbox",
            workspace_root=str(root),
            cwd=".",
            executable_paths={"cat": "/bin/cat"},
            sandbox=SandboxProfileSelection(
                profile="patch-context",
                required=True,
            ),
            git=ShellGitToolSettings(
                workspace_root=str(root),
                cwd=".",
                executable_path="/usr/bin/git",
                capabilities=("read", "worktree"),
                allowed_commands=("restore",),
            ),
        ),
        isolation=IsolationToolRuntimeSettings(
            effective_settings=IsolationEffectiveSettings(
                mode=IsolationMode.SANDBOX,
                source=trusted_isolation_source("sdk"),
                sandbox=plan.settings,
            ),
            sandbox_backend=_native_backend(),
        ),
    )


@dataclass(frozen=True, slots=True)
class _SandboxCorpusCase:
    """Bind one inherited local contract case to sandbox execution."""

    case_id: str
    source_contract: str
    category: str
    operation: OperationType
    arguments: dict[str, object]
    initial_files: tuple[tuple[str, bytes], ...]
    expected_files: tuple[tuple[str, bytes], ...]
    expected_status: PatchStatus
    replace_root: bool = False
    inspection_only: bool = False
    expected_error: bool = False


_SANDBOX_SHARED_CORPUS = (
    _SandboxCorpusCase(
        "P4-nested-exact-match",
        "phase_4_contract_test.py",
        "semantic",
        OperationType.EDIT,
        {
            "path": "nested/note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        },
        (("nested/note.txt", b"before\n"),),
        (("nested/note.txt", b"before\n"),),
        PatchStatus.COMMITTED,
        inspection_only=True,
    ),
    _SandboxCorpusCase(
        "P4-missing-match-fault",
        "phase_4_contract_test.py",
        "fault",
        OperationType.EDIT,
        {
            "path": "note.txt",
            "edits": [{"old_text": "absent", "new_text": "after"}],
        },
        (("note.txt", b"before\n"),),
        (("note.txt", b"before\n"),),
        PatchStatus.REJECTED,
        expected_error=True,
    ),
    _SandboxCorpusCase(
        "P4-root-replacement-race",
        "phase_4_contract_test.py",
        "race",
        OperationType.EDIT,
        {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        },
        (("note.txt", b"before\n"),),
        (("note.txt", b"before\n"),),
        PatchStatus.STALE,
        replace_root=True,
    ),
    _SandboxCorpusCase(
        "P7-operation-matrix",
        "phase_7_contract_test.py",
        "operation_matrix",
        OperationType.APPLY,
        {
            "patch": "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: created.txt",
                    "+created",
                    "*** Delete File: deleted.txt",
                    "*** Update File: source.txt",
                    "*** Move to: moved.txt",
                    "*** Update File: note.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            )
        },
        (
            ("deleted.txt", b"deleted\n"),
            ("note.txt", b"before\n"),
            ("source.txt", b"source\n"),
        ),
        (
            ("created.txt", b"created\n"),
            ("moved.txt", b"source\n"),
            ("note.txt", b"after\n"),
        ),
        PatchStatus.COMMITTED,
    ),
    _SandboxCorpusCase(
        "P9-closed-model-projection",
        "phase_9_contract_test.py",
        "projection",
        OperationType.EDIT,
        {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        },
        (("note.txt", b"before\n"),),
        (("note.txt", b"after\n"),),
        PatchStatus.COMMITTED,
    ),
)


def _sandbox_corpus_policy() -> TrustedPatchPolicy:
    """Authorize every shared operation class through one review route."""
    reader = PreauthorizationClass("sandbox-corpus-read")
    return TrustedPatchPolicy(
        PolicyRevision("policy-v2"),
        frozenset((OperationType.EDIT, OperationType.APPLY)),
        (
            PolicyRule(
                PolicyPathSelector(None),
                (
                    CapabilityMode(
                        Capability.CREATE,
                        ApprovalMode.REQUIRE_REVIEW,
                    ),
                    CapabilityMode(
                        Capability.UPDATE,
                        ApprovalMode.REQUIRE_REVIEW,
                    ),
                    CapabilityMode(
                        Capability.DELETE,
                        ApprovalMode.REQUIRE_REVIEW,
                    ),
                    CapabilityMode(
                        Capability.MOVE,
                        ApprovalMode.REQUIRE_REVIEW,
                    ),
                    CapabilityMode(
                        Capability.READ_FOR_MUTATION,
                        ApprovalMode.PREAUTHORIZED,
                        reader,
                    ),
                    CapabilityMode(
                        Capability.OBSERVE_MUTATION_PRECONDITIONS,
                        ApprovalMode.PREAUTHORIZED,
                        reader,
                    ),
                ),
                atomicity_classes=frozenset(
                    (
                        "single_step",
                        "dependency_ordered",
                    )
                ),
            ),
        ),
        _limits(),
        ApprovalRequirements(
            ApprovalMode.REQUIRE_REVIEW,
            PolicyRouteId("sandbox-runtime-route"),
            PolicyBrokerId("sandbox-runtime-broker"),
            PolicyReviewerRole("sandbox-runtime-reviewer"),
            1,
        ),
    )


def _write_corpus_tree(
    root: Path,
    files: tuple[tuple[str, bytes], ...],
) -> None:
    """Materialize one bounded shared-corpus tree."""
    for path, value in files:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)


def _read_corpus_tree(root: Path) -> tuple[tuple[str, bytes], ...]:
    """Return the complete regular-file tree in stable logical order."""
    return tuple(
        sorted(
            (path.relative_to(root).as_posix(), path.read_bytes())
            for path in root.rglob("*")
            if path.is_file()
        )
    )


def _native_backend_name() -> str:
    """Return the one supported native backend for the active platform."""
    if sys_platform == "darwin":
        return "seatbelt"
    if sys_platform.startswith("linux"):
        return "bubblewrap"
    raise RuntimeError("native sandbox patch runtime is unsupported")


async def _native_probe() -> bool:
    """Probe the backend that the trusted plan selected for this platform."""
    if _native_backend_name() == "seatbelt":
        return (await SeatbeltSandboxBackend().probe()).ok
    return (await BubblewrapSandboxBackend().probe()).ok


def _native_backend() -> SeatbeltSandboxBackend | BubblewrapSandboxBackend:
    """Return the one actual native backend selected by this platform."""
    if _native_backend_name() == "seatbelt":
        return SeatbeltSandboxBackend()
    return BubblewrapSandboxBackend()


def _ordinary_write_plan(root: Path, namespace: Path) -> SandboxExecutionPlan:
    """Build the ordinary selected sandbox plan that attempts a write."""
    base = _plan(root, namespace)
    request = SandboxPlanRequest(
        request_kind=SandboxPlanRequestKind.AGENT_SESSION,
        logical_name="ordinary-shell",
        command="/bin/sh",
        argv=(
            "/bin/sh",
            "-c",
            "printf ordinary > ordinary-shell-write.txt",
        ),
        cwd=str(root),
    )
    return replace(base, request=request)


async def _assert_ordinary_tool_writes_are_denied(
    root: Path, namespace: Path
) -> None:
    """Require real ordinary tools to leave the workspace unchanged."""
    result = await _native_backend().execute(
        _ordinary_write_plan(root, namespace)
    )
    assert result.status in {
        SandboxResultStatus.DENIED,
        SandboxResultStatus.FAILED,
    }
    shell_path = root / "ordinary-shell-write.txt"
    assert not shell_path.exists()
    assert HAS_CODE_DEPENDENCIES
    code_path = root / "ordinary-code-write.txt"
    with pytest.raises(NameError):
        await CodeTool()(
            "def run():\n    return open(" + repr(str(code_path)) + ", 'w')\n",
            context=ToolCallContext(),
        )
    assert not code_path.exists()


def _runtime(root: Path, namespace: Path) -> SandboxPatchRuntime:
    """Create one runtime from its actual selected sandbox plan."""
    return _settings(root, namespace).create_runtime()


class _RuntimeClock(ApprovalClock):
    """Read one fixed nonexpired trusted time for the runtime E2E."""

    def __init__(self) -> None:
        """Start at the first trusted tick."""
        self.value = ExpiryTick(1)

    async def now(self) -> ExpiryTick:
        """Return the stable time used by review and durable fencing."""
        return self.value

    def advance(self, value: int) -> None:
        """Advance monotonically for deterministic recovery exercises."""
        if value <= self.value.value:
            raise ValueError("runtime clock must advance")
        self.value = ExpiryTick(value)


class _BlockingFenceStore(InMemoryDurablePatchStore):
    """Pause the first native effect at its production durable fence check."""

    def __init__(self, backend: InMemoryDurablePatchBackend) -> None:
        """Create independently observable effect and release events."""
        super().__init__(backend)
        self.effect_reached = Event()
        self.release_effect = Event()
        self.checks = 0

    async def is_current_fence(
        self, lease: DurableCommitLease, now: ExpiryTick
    ) -> bool:
        """Block only the first per-effect check after authority issuance."""
        self.checks += 1
        if self.checks == 2:
            self.effect_reached.set()
            await self.release_effect.wait()
        return await super().is_current_fence(lease, now)


class _RuntimeBroker:
    """Approve the selected policy review through the real approval service."""

    async def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Return the one policy-matching reviewer decision."""
        return BrokerDecision(
            request.requirements.broker,
            (
                ReviewerDecision(
                    PatchPrincipalId("sandbox-reviewer"),
                    request.subject.tenant,
                    request.requirements.reviewer_role,
                    ApprovalDecisionState.APPROVED,
                ),
            ),
        )


def _runtime_subject() -> ExecutionSubject:
    """Return the trusted host subject bound to the sandbox session."""
    return ExecutionSubject(
        PatchPrincipalId("sandbox-principal"),
        PatchTenantId("sandbox-tenant"),
        PatchRunId("sandbox-run"),
        PatchSessionId("sandbox-session"),
        PatchTaskId("sandbox-task"),
        PatchAgentId("sandbox-agent"),
    )


def _runtime_policy() -> TrustedPatchPolicy:
    """Return the production review policy matching the selected runtime."""
    reader = PreauthorizationClass("sandbox-runtime-read")
    return TrustedPatchPolicy(
        PolicyRevision("policy-v2"),
        frozenset((OperationType.EDIT,)),
        (
            PolicyRule(
                PolicyPathSelector(None),
                (
                    CapabilityMode(
                        Capability.UPDATE, ApprovalMode.REQUIRE_REVIEW
                    ),
                    CapabilityMode(
                        Capability.READ_FOR_MUTATION,
                        ApprovalMode.PREAUTHORIZED,
                        reader,
                    ),
                    CapabilityMode(
                        Capability.OBSERVE_MUTATION_PRECONDITIONS,
                        ApprovalMode.PREAUTHORIZED,
                        reader,
                    ),
                ),
            ),
        ),
        _limits(),
        ApprovalRequirements(
            ApprovalMode.REQUIRE_REVIEW,
            PolicyRouteId("sandbox-runtime-route"),
            PolicyBrokerId("sandbox-runtime-broker"),
            PolicyReviewerRole("sandbox-runtime-reviewer"),
            1,
        ),
    )


@dataclass(frozen=True, slots=True)
class _UnissuedBinding:
    """Keep only the plan domain needed to test worker authority rejection."""

    target: TargetIdentity
    context_kind: ContextKind
    cwd: LogicalPath | None


@dataclass(frozen=True, slots=True)
class _UnissuedPlan:
    """Provide no mutation contents to an authority-only negative test."""

    binding: _UnissuedBinding


def _unissued_command(
    identity: TargetIdentity,
    fence: int = 1,
    cwd: LogicalPath | None = None,
    request_id: PatchRequestId | None = None,
) -> SealedCommitCommand:
    """Return a sealed command that has no minted durable worker authority."""
    return SealedCommitCommand(
        cast(
            SealedPlan,
            _UnissuedPlan(
                _UnissuedBinding(identity, ContextKind.SANDBOX, cwd)
            ),
        ),
        CommitLease(
            identity.domain_id,
            request_id or PatchRequestId("request_" + "f" * 16),
            fence,
        ),
        LockFootprint(identity.domain_id, ("workspace",)),
    )


async def _assert_wrong_fence_is_rejected(
    runtime: SandboxPatchRuntime,
    scope: ResolvedMutationScope,
    clock: ApprovalClock,
) -> None:
    """Prove a stale command cannot pass the live durable owner check."""
    identity = runtime.profile.identity
    request_id = PatchRequestId("request_" + "e" * 16)
    digest = AlgorithmDigest.from_bytes(b"wrong-fence")
    durable_identity = DurableRequestIdentity(
        PatchTenantId("fence-tenant"),
        PatchPrincipalId("fence-principal"),
        PatchExecutionId("execution_" + "e" * 16),
        PolicyRouteId("fence-route"),
        RetransmissionKey("fence-retransmission"),
    )
    plan = DurablePlanReference(
        PatchPlanId("plan_" + "e" * 16),
        digest,
        digest,
        digest,
        identity.context_id,
        identity.workspace_id,
        identity.domain_id,
        (
            DurableStepBinding(
                PatchStepId("step_" + "e" * 16),
                PatchLineageId("lineage_" + "e" * 16),
            ),
        ),
    )
    signer = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=signer)
    )
    reservation = await store.reserve(durable_identity, digest, request_id)
    await store.persist_plan(reservation, plan)
    approval = signer.seal(
        DurableApproval(
            PatchGrantId("grant_" + "e" * 16),
            PatchApprovalId("approval_" + "e" * 16),
            durable_identity,
            digest,
            plan.plan_id,
            plan.fingerprint_digest,
            plan.review_digest,
            identity.context_id,
            identity.workspace_id,
            identity.domain_id,
            identity.policy_revision,
            PolicyBrokerId("fence-broker"),
            PolicyReviewerRole("fence-reviewer"),
            (PatchPrincipalId("fence-reviewer"),),
            ExpiryTick(100),
            b"\x00" * 32,
        )
    )
    claim = await store.claim_commit(
        reservation,
        plan,
        approval,
        PatchCommitOwnerId("owner_" + "e" * 16),
        ExpiryTick(1),
        DurationTicks(10),
        (),
    )
    assert claim.state is DurableCommitClaimState.OWNER
    assert claim.lease is not None
    validator = _SandboxDurableCommandAuthority(
        runtime,
        scope,
        claim.lease,
        store,
        clock,
    )
    assert await validator.is_rooted_command_current(
        _unissued_command(
            identity,
            claim.lease.fence.value,
            None,
            request_id,
        )
    )
    assert not await validator.is_rooted_command_current(
        _unissued_command(
            identity,
            claim.lease.fence.value + 1,
            None,
            request_id,
        )
    )


async def _claim_domain_owner(
    store: InMemoryDurablePatchStore,
    signer: HmacDurableApprovalAuthority,
    identity: TargetIdentity,
    label: str,
    now: ExpiryTick,
) -> DurableCommitLease:
    """Claim one actual durable owner for the supplied target identity."""
    request_id = PatchRequestId("request_" + label)
    digest = AlgorithmDigest.from_bytes(("domain-" + label).encode())
    durable_identity = DurableRequestIdentity(
        PatchTenantId("tenant-" + label),
        PatchPrincipalId("principal-" + label),
        PatchExecutionId("execution_" + label),
        PolicyRouteId("route-" + label),
        RetransmissionKey("retransmission-" + label),
    )
    plan = DurablePlanReference(
        PatchPlanId("plan_" + label),
        digest,
        digest,
        digest,
        identity.context_id,
        identity.workspace_id,
        identity.domain_id,
        (
            DurableStepBinding(
                PatchStepId("step_" + label),
                PatchLineageId("lineage_" + label),
            ),
        ),
    )
    reservation = await store.reserve(durable_identity, digest, request_id)
    await store.persist_plan(reservation, plan)
    approval = signer.seal(
        DurableApproval(
            PatchGrantId("grant_" + label),
            PatchApprovalId("approval_" + label),
            durable_identity,
            digest,
            plan.plan_id,
            plan.fingerprint_digest,
            plan.review_digest,
            identity.context_id,
            identity.workspace_id,
            identity.domain_id,
            identity.policy_revision,
            PolicyBrokerId("broker-" + label),
            PolicyReviewerRole("reviewer-" + label),
            (PatchPrincipalId("reviewer-" + label),),
            ExpiryTick(now.value + 100),
            b"\x00" * 32,
        )
    )
    claim = await store.claim_commit(
        reservation,
        plan,
        approval,
        PatchCommitOwnerId("owner_" + label),
        now,
        DurationTicks(10),
        (),
    )
    assert claim.state is DurableCommitClaimState.OWNER
    assert claim.lease is not None
    return claim.lease


async def _settle_expired_domain_owner(
    store: InMemoryDurablePatchStore,
    backend: InMemoryDurablePatchBackend,
    lease: DurableCommitLease,
    label: str,
    now: ExpiryTick,
) -> DurableCommitLease:
    """Recover and settle one expired owner from durable unknown truth."""
    record = backend.by_request[lease.request_id]
    plan = record.plan
    assert plan is not None
    recovery = await store.replace_expired_owner(
        record.reservation,
        lease,
        PatchCommitOwnerId("owner_recovery_" + label),
        now,
        DurationTicks(10),
    )
    journal = await store.append_step(
        recovery,
        DurableJournalCursor(lease.request_id, SequenceNumber(0)),
        plan.steps[0].step_id,
        CommitStepState.PLANNED,
        now,
    )
    journal = await store.append_step(
        recovery,
        journal.cursor,
        plan.steps[0].step_id,
        CommitStepState.UNKNOWN,
        now,
    )
    result = PatchResult(
        1,
        lease.request_id,
        plan.plan_id,
        LifecyclePhase.REQUEST_COMPLETED,
        PatchStatus.INDETERMINATE,
        CommitTruth(
            MutationState.INDETERMINATE,
            LineageState.INDETERMINATE,
            RequestedEffectOccurrence.UNKNOWN,
            ArtifactState.ABSENT,
            WorkspaceChange.UNKNOWN,
            False,
            PostconditionState.UNKNOWN,
        ),
        PatchDiagnostic(
            ErrorStage.SETTLEMENT,
            PatchErrorCode.INDETERMINATE,
            Retryability.NOT_RETRYABLE,
        ),
    )
    await store.settle(
        recovery,
        journal.cursor,
        result,
        PatchObserverCorrelationId("correlation_recovery_" + label),
        now,
    )
    return recovery


async def _assert_identity_invalidation_matrix(
    runtime: SandboxPatchRuntime,
    scope: ResolvedMutationScope,
    clock: ApprovalClock,
) -> None:
    """Reject every changed plan, profile, worker, session, or lease fact."""
    signer = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=signer)
    )
    identity = runtime.profile.identity
    lease = await _claim_domain_owner(
        store,
        signer,
        identity,
        "a" * 16,
        ExpiryTick(1),
    )
    validator = _SandboxDurableCommandAuthority(
        runtime,
        scope,
        lease,
        store,
        clock,
    )
    command = _unissued_command(
        identity,
        lease.fence.value,
        scope.cwd,
        lease.request_id,
    )
    assert await validator.is_rooted_command_current(command)
    replacements = (
        replace(
            identity,
            context_id=PatchContextId("context_" + "b" * 16),
        ),
        replace(
            identity,
            workspace_id=PatchWorkspaceId("workspace_" + "b" * 16),
        ),
        replace(
            identity,
            domain_id=PatchDomainId("domain_" + "b" * 16),
        ),
        replace(
            identity,
            target_id=PatchTargetId("target_" + "b" * 16),
        ),
        replace(
            identity,
            protocol_id=PatchProtocolId("protocol_" + "b" * 16),
        ),
        replace(identity, filesystem_id="filesystem-vary"),
        replace(identity, mount_id="mount-vary"),
        replace(identity, policy_revision="policy-vary"),
        replace(identity, persistent_lease_id="lease-vary"),
        replace(
            identity,
            approval_channel_id=PatchApprovalId("approval_" + "b" * 16),
        ),
        replace(identity, implementation_id="implementation-vary"),
    )
    for changed in replacements:
        assert not await validator.is_rooted_command_current(
            _unissued_command(
                changed,
                lease.fence.value,
                scope.cwd,
                lease.request_id,
            )
        )
    assert not await validator.is_rooted_command_current(
        _unissued_command(
            identity,
            lease.fence.value + 1,
            scope.cwd,
            lease.request_id,
        )
    )
    assert not await _SandboxDurableCommandAuthority(
        runtime,
        replace(
            scope,
            worker=EphemeralWorkerWitness(
                "channel-vary",
                scope.worker.worker_instance_id if scope.worker else "worker",
                scope.worker.fence_id if scope.worker else "fence",
            ),
        ),
        lease,
        store,
        clock,
    ).is_rooted_command_current(command)
    assert not await _SandboxDurableCommandAuthority(
        runtime,
        replace(scope, cwd=LogicalPath("view-vary")),
        lease,
        store,
        clock,
    ).is_rooted_command_current(command)
    original_profile = runtime.profile
    runtime.profile = replace(
        original_profile,
        context_lifetime_id=SandboxContextLifetimeId("context-lifetime-vary"),
    )
    assert not await validator.is_rooted_command_current(command)
    runtime.profile = original_profile
    assert await validator.is_rooted_command_current(command)
    assert runtime._receipt is not None
    runtime._receipt = replace(
        runtime._receipt,
        session_id=SandboxSessionId(runtime._receipt.session_id + "-vary"),
    )
    assert not await validator.is_rooted_command_current(command)


def test_patch_phase_10_requirements(tmp_path: Path) -> None:
    """Use only the live selected native view or fail closed."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    host_canary = tmp_path / "host-canary"
    root.mkdir()
    namespace.mkdir()
    host_canary.mkdir()
    (root / "note.txt").write_text("inside\n", encoding="utf-8")
    (root / "host-canary").mkdir()
    (root / "host-canary" / "note.txt").write_text(
        "inside-canary\n",
        encoding="utf-8",
    )
    (host_canary / "note.txt").write_text("outside\n", encoding="utf-8")
    runtime = _runtime(root, namespace)

    async def exercise() -> None:
        if not await _native_probe():
            with pytest.raises(TargetInspectionError) as unavailable:
                await runtime.resolve(ScopeSelection(ContextKind.SANDBOX))
            assert (
                unavailable.value.code
                is TargetErrorCode.CAPABILITY_UNAVAILABLE
            )
            return
        await _assert_ordinary_tool_writes_are_denied(root, namespace)
        scope = await runtime.resolve(ScopeSelection(ContextKind.SANDBOX))
        receipt = runtime._receipt
        attestation = runtime._process._attestation
        assert receipt is not None
        assert attestation is not None
        assert (
            receipt.runtime_command_digest
            == attestation.runtime_command_digest
        )
        assert (
            receipt.backend_policy_digest == attestation.backend_policy_digest
        )
        assert (
            receipt.child_process_identity
            == attestation.child_process_identity
        )
        assert receipt.canary_receipt == attestation.canary_receipt
        assert len(receipt.canary_receipt) == 64
        if _native_backend_name() == "seatbelt":
            bundle = runtime._process._bundle
            assert bundle is not None
            profile = sandbox_commit_module._seatbelt_runtime_profile(
                runtime.profile,
                bundle.root,
            )
            assert '(subpath "/")' not in profile
            assert profile.count('(literal "/")') == 1
        inspection = SandboxInspectionTarget(runtime)
        target = SandboxCommitTarget(runtime)
        handshake = await target.handshake(scope)
        assert handshake.worker is scope.worker
        assert all(item.receipt is not None for item in handshake.probes)
        batch = await inspection.inspect(
            InspectionRequest(scope, (LogicalPath("note.txt"),))
        )
        assert batch.snapshots[0].bytes_value is not None
        assert batch.snapshots[0].bytes_value._value == b"inside\n"
        canary = await inspection.inspect(
            InspectionRequest(
                scope,
                (LogicalPath("host-canary/note.txt"),),
            )
        )
        assert canary.snapshots[0].bytes_value is not None
        assert canary.snapshots[0].bytes_value._value == b"inside-canary\n"
        assert (host_canary / "note.txt").read_text(
            encoding="utf-8"
        ) == "outside\n"
        await _assert_ordinary_tool_writes_are_denied(root, namespace)
        await runtime.close()
        await _assert_ordinary_tool_writes_are_denied(root, namespace)

    run(exercise())


def test_patch_phase_10_public_sdk_uses_durable_worker_and_outbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute a real selected runtime through loader, SDK, and outbox."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    note = root / "note.txt"
    note.write_text("before\n", encoding="utf-8")
    clock = _RuntimeClock()
    approvals = ApprovalService(_RuntimeBroker(), clock, RuntimeGrantStore())
    authority = HmacDurableApprovalAuthority.random()
    backend = InMemoryDurablePatchBackend(approval_verifier=authority)
    store = InMemoryDurablePatchStore(backend)
    configuration = SandboxPatchServiceConfiguration(
        _runtime_subject(),
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, authority),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )
    policy = _runtime_policy()
    binder = SandboxPatchRuntimeBinder.from_settings(
        _settings(root, namespace),
        configuration,
        policy,
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True, store),
        PatchPersistenceBinding(True, store),
    )
    captured_command: dict[str, object] = {}
    original_command_payload = sandbox_commit_module._command_payload

    def capture_command_payload(
        command: SealedCommitCommand,
        profile: sandbox_commit_module.SandboxRuntimeProfile,
        receipt: sandbox_commit_module.SandboxProfileReceipt,
        session_id: SandboxSessionId,
        root_witness: rooted_worker_module.RootWitness,
        implementation_digest: str,
    ) -> Mapping[str, object]:
        """Retain the native wire transaction emitted by the SDK commit."""
        payload = original_command_payload(
            command,
            profile,
            receipt,
            session_id,
            root_witness,
            implementation_digest,
        )
        captured_command.update(
            command=command,
            payload=payload,
            profile=profile,
            receipt=receipt,
            session_id=session_id,
            root=root_witness,
            implementation_digest=implementation_digest,
        )
        return payload

    monkeypatch.setattr(
        sandbox_commit_module, "_command_payload", capture_command_payload
    )

    async def exercise() -> None:
        """Run one public mutation and read only durable lifecycle truth."""
        assert await _native_probe()
        loader = PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        )
        bundle = await loader.load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        scope = await binder.runtime.resolve(
            ScopeSelection(ContextKind.SANDBOX)
        )
        direct_worker = await binder.runtime.worker(scope)
        with pytest.raises(CoordinatorError) as direct_rejected:
            await direct_worker.commit(
                _unissued_command(binder.runtime.profile.identity)
            )
        assert direct_rejected.value.code is CoordinatorErrorCode.FENCED
        unsettled = await direct_worker._reconcile_for_owner(
            PatchRequestId("request_" + "a" * 16)
        )
        assert unsettled == WorkerReport(WorkerState.LIVE, None)
        await _assert_wrong_fence_is_rejected(binder.runtime, scope, clock)
        host = bundle.toolset.sdk_host()
        outcome = await host.invoke_json(
            OperationType.EDIT,
            {
                "path": "note.txt",
                "edits": [{"old_text": "before", "new_text": "after"}],
            },
        )
        assert outcome.lifecycle is LifecyclePhase.REQUEST_COMPLETED
        profile = cast(
            sandbox_commit_module.SandboxRuntimeProfile,
            captured_command["profile"],
        )
        sealed_command = cast(SealedCommitCommand, captured_command["command"])
        original_backend = profile.execution_plan.settings.backend
        object.__setattr__(
            profile.execution_plan.settings, "backend", object()
        )
        try:
            with pytest.raises(TargetInspectionError) as malformed_backend:
                original_command_payload(
                    sealed_command,
                    profile,
                    cast(
                        sandbox_commit_module.SandboxProfileReceipt,
                        captured_command["receipt"],
                    ),
                    cast(SandboxSessionId, captured_command["session_id"]),
                    cast(
                        rooted_worker_module.RootWitness,
                        captured_command["root"],
                    ),
                    cast(str, captured_command["implementation_digest"]),
                )
        finally:
            object.__setattr__(
                profile.execution_plan.settings, "backend", original_backend
            )
        assert (
            malformed_backend.value.code
            is TargetErrorCode.CAPABILITY_UNAVAILABLE
        )

        original_fingerprint = sealed_command.plan.fingerprint
        object.__setattr__(
            sealed_command.plan, "fingerprint", PatchFingerprint(b"tampered")
        )
        validation_patch = pytest.MonkeyPatch()
        validation_patch.setattr(
            sandbox_commit_module, "_validate_sealed_plan", lambda _plan: None
        )
        try:
            with pytest.raises(TargetInspectionError) as tampered_fingerprint:
                original_command_payload(
                    sealed_command,
                    profile,
                    cast(
                        sandbox_commit_module.SandboxProfileReceipt,
                        captured_command["receipt"],
                    ),
                    cast(SandboxSessionId, captured_command["session_id"]),
                    cast(
                        rooted_worker_module.RootWitness,
                        captured_command["root"],
                    ),
                    cast(str, captured_command["implementation_digest"]),
                )
        finally:
            validation_patch.undo()
            object.__setattr__(
                sealed_command.plan, "fingerprint", original_fingerprint
            )
        assert tampered_fingerprint.value.code is TargetErrorCode.WITNESS_STALE
        worker_config = cast(
            _WorkerChildConfig,
            {
                "root": profile.workspace_view_root,
                "namespace": profile.private_view_root,
                "cwd": None if profile.cwd is None else profile.cwd.value,
                "maximum": profile.max_snapshot_bytes.value,
                "aggregate_maximum": profile.limits.snapshot_bytes.value,
                "token": "a" * 64,
                "receipt": captured_command["receipt"],
                "identity": _identity_payload(profile.identity),
                "channel_id": profile.channel_id,
                "implementation_id": profile.implementation_id,
                "implementation_digest": captured_command[
                    "implementation_digest"
                ],
                "source_digest": "source-digest",
                "implementation_root": "/implementation",
                "read_canary": "/outside-canary",
                "session_id": captured_command["session_id"],
                "execution_plan": profile.execution_plan.plan_fingerprint,
                "backend": profile.execution_plan.settings.backend,
                "workspace_view": profile.workspace_view_root,
                "private_view": profile.private_view_root,
                "context_lifetime": profile.context_lifetime_id,
                "protocol": profile.identity.protocol_id.value,
                "persistent_lease": profile.identity.persistent_lease_id,
                "filesystem": profile.identity.filesystem_id,
                "mount": profile.identity.mount_id,
            },
        )
        decoded = sandbox_worker_module._mutation_command(
            cast(Mapping[str, object], captured_command["payload"]),
            worker_config,
            cast(rooted_worker_module.RootWitness, captured_command["root"]),
        )
        assert decoded.plan_id == outcome.plan_id
        assert decoded.effects == frozenset((Capability.UPDATE,))
        assert (
            rooted_worker_module._command_plan_id(decoded) == outcome.plan_id
        )
        assert (
            rooted_worker_module._command_lineages(decoded) == decoded.lineages
        )
        assert (
            rooted_worker_module._command_effects(decoded) == decoded.effects
        )

        def reseal_wire_plan(value: dict[str, object]) -> None:
            """Recompute transport integrity fields after tampering."""
            plan_value = cast(dict[str, object], value["plan"])
            wire_fields = {
                key: item
                for key, item in plan_value.items()
                if key not in {"canonical", "fingerprint"}
            }
            canonical = sandbox_wire_module.canonical_sandbox_plan_bytes(
                wire_fields
            )
            plan_value["canonical"] = b64encode(canonical).decode()
            plan_value["fingerprint"] = b64encode(
                sha256(canonical).digest()
            ).decode()

        def reject_tampered_wire(
            tamper: Callable[[dict[str, object]], None],
            code: TargetErrorCode,
        ) -> None:
            """Require decoder rejection after one targeted wire alteration."""
            tampered = deepcopy(
                cast(dict[str, object], captured_command["payload"])
            )
            tamper(tampered)
            with pytest.raises(TargetInspectionError) as rejected_wire:
                sandbox_worker_module._mutation_command(
                    tampered,
                    worker_config,
                    cast(
                        rooted_worker_module.RootWitness,
                        captured_command["root"],
                    ),
                )
            assert rejected_wire.value.code is code

        def alter_schema(value: dict[str, object]) -> None:
            """Change the closed outer schema tag."""
            value["schema"] = "wrong-schema"

        def alter_runtime(value: dict[str, object]) -> None:
            """Change the bound runtime backend fact."""
            runtime_value = cast(dict[str, object], value["runtime"])
            runtime_value["backend"] = "wrong-backend"

        def remove_footprint(value: dict[str, object]) -> None:
            """Remove every sealed lock footprint entry."""
            command_value = cast(dict[str, object], value["command"])
            command_value["footprint"] = []

        def alter_fingerprint(value: dict[str, object]) -> None:
            """Change a syntactically valid transport digest."""
            plan_value = cast(dict[str, object], value["plan"])
            plan_value["fingerprint"] = b64encode(b"x" * 32).decode()

        def alter_operation(value: dict[str, object]) -> None:
            """Change the sealed request operation to an unrecognized value."""
            plan_value = cast(dict[str, object], value["plan"])
            request_value = cast(dict[str, object], plan_value["request"])
            request_value["operation"] = "delete"
            reseal_wire_plan(value)

        def alter_diff_digest(value: dict[str, object]) -> None:
            """Break the rendered diff digest while retaining integrity."""
            plan_value = cast(dict[str, object], value["plan"])
            diff_value = cast(dict[str, object], plan_value["diff"])
            diff_value["digest"] = AlgorithmDigest.from_bytes(b"wrong").value
            reseal_wire_plan(value)

        def expire_review(value: dict[str, object]) -> None:
            """Set an invalid zero review expiry in the transported plan."""
            plan_value = cast(dict[str, object], value["plan"])
            review_value = cast(dict[str, object], plan_value["review"])
            review_value["expiry"] = 0
            reseal_wire_plan(value)

        def remove_lineages(value: dict[str, object]) -> None:
            """Remove mandatory planned lineage evidence."""
            plan_value = cast(dict[str, object], value["plan"])
            plan_value["lineages"] = []
            reseal_wire_plan(value)

        def alter_effect(value: dict[str, object]) -> None:
            """Add an unrecognized capability to the sealed effect list."""
            plan_value = cast(dict[str, object], value["plan"])
            plan_value["authorized_effects"] = ["unknown-capability"]
            reseal_wire_plan(value)

        def duplicate_final_file(value: dict[str, object]) -> None:
            """Repeat a final-file fact with a valid transport checksum."""
            plan_value = cast(dict[str, object], value["plan"])
            final_files = cast(list[object], plan_value["final_files"])
            final_files.append(deepcopy(final_files[0]))
            reseal_wire_plan(value)

        def final_file(value: dict[str, object]) -> dict[str, object]:
            """Return the first complete final-file payload."""
            plan_value = cast(dict[str, object], value["plan"])
            final_files = cast(list[object], plan_value["final_files"])
            return cast(dict[str, object], final_files[0])

        def replace_matches_with_mapping(value: dict[str, object]) -> None:
            """Replace the closed list of source matches with a mapping."""
            plan_value = cast(dict[str, object], value["plan"])
            lineages = cast(list[object], plan_value["lineages"])
            lineage = cast(dict[str, object], lineages[0])
            lineage["matches"] = {}
            reseal_wire_plan(value)

        def corrupt_parent_identity(value: dict[str, object]) -> None:
            """Supply a malformed filesystem identity tuple."""
            plan_value = cast(dict[str, object], value["plan"])
            lineages = cast(list[object], plan_value["lineages"])
            lineage = cast(dict[str, object], lineages[0])
            lineage["parent_identities"] = [["note.txt", "one", 2]]
            reseal_wire_plan(value)

        def corrupt_lineage_id(value: dict[str, object]) -> None:
            """Supply an invalid opaque lineage identifier."""
            plan_value = cast(dict[str, object], value["plan"])
            lineages = cast(list[object], plan_value["lineages"])
            lineage = cast(dict[str, object], lineages[0])
            lineage["id"] = "invalid"
            reseal_wire_plan(value)

        def corrupt_file_presence(value: dict[str, object]) -> None:
            """Supply a non-boolean planned-file presence field."""
            final = final_file(value)
            final["present"] = "yes"
            reseal_wire_plan(value)

        def corrupt_file_kind(value: dict[str, object]) -> None:
            """Supply an unknown byte wrapper for a planned file."""
            final = final_file(value)
            final["content_kind"] = "unknown"
            reseal_wire_plan(value)

        def corrupt_file_metadata(value: dict[str, object]) -> None:
            """Supply a non-boolean byte-order-mark field."""
            final = final_file(value)
            metadata = cast(dict[str, object], final["metadata"])
            metadata["bom"] = "yes"
            reseal_wire_plan(value)

        def corrupt_file_digest(value: dict[str, object]) -> None:
            """Break a planned-file digest without breaking its wire seal."""
            final = final_file(value)
            final["digest"] = AlgorithmDigest.from_bytes(b"wrong").value
            reseal_wire_plan(value)

        def leak_absent_file_field(value: dict[str, object]) -> None:
            """Leave content behind while declaring a planned file absent."""
            final = final_file(value)
            final["present"] = False
            reseal_wire_plan(value)

        def corrupt_file_identity(value: dict[str, object]) -> None:
            """Supply a malformed planned-file identity tuple."""
            final = final_file(value)
            final["identity"] = ["one", 2]
            reseal_wire_plan(value)

        reject_tampered_wire(alter_schema, TargetErrorCode.WORKER_UNAVAILABLE)
        reject_tampered_wire(alter_runtime, TargetErrorCode.WITNESS_STALE)
        reject_tampered_wire(
            remove_footprint, TargetErrorCode.WORKER_UNAVAILABLE
        )
        reject_tampered_wire(alter_fingerprint, TargetErrorCode.WITNESS_STALE)
        reject_tampered_wire(
            alter_operation, TargetErrorCode.WORKER_UNAVAILABLE
        )
        reject_tampered_wire(alter_diff_digest, TargetErrorCode.WITNESS_STALE)
        reject_tampered_wire(expire_review, TargetErrorCode.WITNESS_STALE)
        reject_tampered_wire(
            remove_lineages, TargetErrorCode.WORKER_UNAVAILABLE
        )
        reject_tampered_wire(alter_effect, TargetErrorCode.WORKER_UNAVAILABLE)
        reject_tampered_wire(
            duplicate_final_file, TargetErrorCode.WITNESS_STALE
        )
        reject_tampered_wire(
            replace_matches_with_mapping, TargetErrorCode.WORKER_UNAVAILABLE
        )
        reject_tampered_wire(
            corrupt_parent_identity, TargetErrorCode.WORKER_UNAVAILABLE
        )
        reject_tampered_wire(
            corrupt_lineage_id, TargetErrorCode.WORKER_UNAVAILABLE
        )
        reject_tampered_wire(
            corrupt_file_presence, TargetErrorCode.WORKER_UNAVAILABLE
        )
        reject_tampered_wire(
            corrupt_file_kind, TargetErrorCode.WORKER_UNAVAILABLE
        )
        reject_tampered_wire(
            corrupt_file_metadata, TargetErrorCode.WORKER_UNAVAILABLE
        )
        reject_tampered_wire(
            corrupt_file_digest, TargetErrorCode.WITNESS_STALE
        )
        reject_tampered_wire(
            leak_absent_file_field, TargetErrorCode.WORKER_UNAVAILABLE
        )
        reject_tampered_wire(
            corrupt_file_identity, TargetErrorCode.WORKER_UNAVAILABLE
        )
        events = [event async for event in host.lifecycle()]
        assert [event.lifecycle for event in events] == [
            LifecyclePhase.REQUEST_COMPLETED
        ]
        assert events[0].request_id == outcome.request_id
        assert [event.sequence.value for event in events] == [1]
        assert len({event.event_id for event in events}) == 1
        assert note.read_text(encoding="utf-8") == "after\n"
        later = await SandboxInspectionTarget(binder.runtime).inspect(
            InspectionRequest(scope, (LogicalPath("note.txt"),))
        )
        assert later.snapshots[0].bytes_value is not None
        assert later.snapshots[0].bytes_value._value == b"after\n"
        staging_canary = namespace / "staging-canary"
        staging_canary.write_text("private\n", encoding="utf-8")
        assert binder.runtime._process._token is not None
        private_markers = (
            str(root),
            str(namespace),
            staging_canary.name,
            binder.runtime.profile.channel_id,
            binder.runtime.profile.implementation_id,
            binder.runtime._process._token.hex(),
            (
                binder.runtime._receipt.profile_receipt
                if binder.runtime._receipt is not None
                else "missing-receipt"
            ),
        )
        public_values = (outcome, events[0], await host.inspect())
        for value in public_values:
            projection = repr(value)
            assert all(marker not in projection for marker in private_markers)
        fresh_bundle = await loader.load(enable_tools=["patch.edit"])
        assert fresh_bundle.toolset is not None
        replayed = await fresh_bundle.toolset.sdk_host().retransmit_json(
            OperationType.EDIT,
            {
                "path": "note.txt",
                "edits": [{"old_text": "before", "new_text": "after"}],
            },
            outcome.request_id,
            events[0].correlation_id,
        )
        assert replayed == outcome
        assert note.read_text(encoding="utf-8") == "after\n"
        await _assert_identity_invalidation_matrix(
            binder.runtime,
            scope,
            clock,
        )
        await binder.runtime.close()

    run(exercise())


def test_patch_phase_10_agent_cycle_uses_selected_sandbox_manager(
    tmp_path: Path,
) -> None:
    """Drive pending settlement and later reads through a real agent cycle."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    note = root / "note.txt"
    note.write_text("before\n", encoding="utf-8")
    assert (
        run_process(
            ("/usr/bin/git", "init", "--quiet", str(root)),
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    assert (
        run_process(
            ("/usr/bin/git", "-C", str(root), "add", "note.txt"),
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    assert (
        run_process(
            (
                "/usr/bin/git",
                "-C",
                str(root),
                "-c",
                "user.name=Sandbox Test",
                "-c",
                "user.email=sandbox@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "-m",
                "baseline",
            ),
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    clock = _RuntimeClock()
    approvals = ApprovalService(_RuntimeBroker(), clock, RuntimeGrantStore())
    authority = HmacDurableApprovalAuthority.random()
    backend = InMemoryDurablePatchBackend(approval_verifier=authority)
    store = _BlockingFenceStore(backend)
    configuration = SandboxPatchServiceConfiguration(
        _runtime_subject(),
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, authority),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )
    runtime_binder = SandboxPatchRuntimeBinder.from_settings(
        _settings(root, namespace),
        configuration,
        _runtime_policy(),
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True, store),
        PatchPersistenceBinding(True, store),
    )
    model_manager = DeterministicModelManager(
        [
            DeterministicToolPlan(
                "patch.edit",
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "before", "new_text": "after"}],
                },
            ),
            DeterministicToolPlan("shell.cat", {"path": "note.txt"}),
            "sandbox agent cycle complete",
        ],
        getLogger(__name__),
    )

    async def exercise() -> None:
        """Run production orchestration with finite fake provider IO."""
        assert await _native_probe()
        stack = AsyncExitStack()
        loader = OrchestratorLoader(
            hub=cast(HuggingfaceHub, object()),
            logger=getLogger(__name__),
            participant_id=uuid4(),
            stack=stack,
            model_manager_factory=(
                lambda _hub, _logger, _events: model_manager
            ),
        )
        agent = await loader.from_settings(
            _agent_settings(),
            tool_settings=_agent_tool_settings(root, runtime_binder),
        )
        await stack.enter_async_context(agent)
        response = await agent("Edit the file, then read it back.")
        consumed = create_task(response.to_str())
        await store.effect_reached.wait()
        assert len(model_manager.calls) == 1
        assert agent.tool._toolsets is not None
        patch_toolsets = [
            item
            for item in agent.tool._toolsets
            if isinstance(item, PatchToolSet)
        ]
        assert len(patch_toolsets) == 1
        service = patch_toolsets[0]._service
        assert isinstance(service, SandboxPatchSdkService)
        consumed.cancel()
        for _ in range(100):
            if service._pending:
                break
            await sleep(0)
        assert service._pending
        assert not consumed.done()
        store.release_effect.set()
        assert await consumed == "sandbox agent cycle complete"
        assert note.read_text(encoding="utf-8") == "after\n"
        assert len(model_manager.calls) == 3
        patch_observation = repr(model_manager.calls[1].context.input)
        assert "patch_result" in patch_observation
        assert LifecyclePhase.REQUEST_COMPLETED.value in patch_observation
        read_messages = model_manager.calls[2].context.input
        assert isinstance(read_messages, list)
        read_message = read_messages[-1]
        assert isinstance(read_message, Message)
        read_result = read_message.tool_call_result
        assert isinstance(read_result, ToolCallResult)
        assert "after" in str(read_result.result), read_result.result

        shell_outcome = await agent.tool.execute_call(
            ToolCall(
                id="sandbox-shell-write-denied",
                name="shell.git_restore",
                arguments={"paths": ["note.txt"]},
            ),
            ToolCallContext(),
        )
        assert isinstance(shell_outcome, ToolCallResult)
        shell_projection = str(shell_outcome.result)
        assert "status: failed" in shell_projection
        assert "execution_mode: sandbox" in shell_projection
        assert "error_code: nonzero_exit" in shell_projection
        assert "exit_code: 1" in shell_projection
        assert note.read_text(encoding="utf-8") == "after\n"
        code_path = root / "ordinary-code-write.txt"
        code_outcome = await agent.tool.execute_call(
            ToolCall(
                id="sandbox-code-write-denied",
                name="code.run",
                arguments={
                    "code": (
                        "def run():\n"
                        f"    return open({str(code_path)!r}, 'w')\n"
                    )
                },
            ),
            ToolCallContext(),
        )
        assert isinstance(code_outcome, ToolCallError)
        assert code_outcome.error_type == "NameError"
        assert code_outcome.message == "name 'open' is not defined"
        assert code_outcome.error == {"type": "NameError"}
        assert not code_path.exists()
        await stack.aclose()
        assert runtime_binder.runtime._process._process is None

    run(exercise())


@pytest.mark.parametrize(
    "case",
    _SANDBOX_SHARED_CORPUS,
    ids=lambda case: case.case_id,
)
def test_patch_phase_10_executes_shared_local_contract_corpus(
    tmp_path: Path,
    case: _SandboxCorpusCase,
) -> None:
    """Execute inherited semantic, race, fault, and projection cases."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    _write_corpus_tree(root, case.initial_files)
    clock = _RuntimeClock()
    approvals = ApprovalService(_RuntimeBroker(), clock, RuntimeGrantStore())
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    configuration = SandboxPatchServiceConfiguration(
        _runtime_subject(),
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, authority),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )
    binder = SandboxPatchRuntimeBinder.from_settings(
        _settings(root, namespace),
        configuration,
        _sandbox_corpus_policy(),
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True, store),
        PatchPersistenceBinding(True, store),
    )

    async def exercise() -> None:
        """Run one mapped case only through the production sandbox adapter."""
        assert await _native_probe()
        bundle = await PatchToolLoader(
            binder,
            PatchTestHostProfile(enabled=True, authenticated=True),
        ).load(enable_tools=["patch.edit", "patch.apply"])
        assert bundle.toolset is not None
        await bundle.manager.__aenter__()
        parked: Path | None = None
        try:
            host = bundle.toolset.sdk_host()
            if case.inspection_only:
                scope = await binder.runtime.resolve(
                    ScopeSelection(ContextKind.SANDBOX)
                )
                inspected = await SandboxInspectionTarget(
                    binder.runtime
                ).inspect(
                    InspectionRequest(
                        scope,
                        tuple(
                            LogicalPath(path)
                            for path, _value in case.initial_files
                        ),
                    )
                )
                assert (
                    tuple(
                        (
                            snapshot.path.value,
                            (
                                b""
                                if snapshot.bytes_value is None
                                else snapshot.bytes_value._value
                            ),
                        )
                        for snapshot in inspected.snapshots
                    )
                    == case.expected_files
                )
                assert _read_corpus_tree(root) == case.expected_files
                return
            if case.replace_root:
                parked = tmp_path / "planned-root"
                root.rename(parked)
                root.mkdir()
                (root / "note.txt").write_bytes(b"canary\n")
                with pytest.raises((PatchToolError, TargetInspectionError)):
                    await host.invoke_json(case.operation, case.arguments)
                assert _read_corpus_tree(parked) == case.expected_files
                assert _read_corpus_tree(root) == (("note.txt", b"canary\n"),)
                return
            if case.expected_error:
                with pytest.raises(PatchToolError):
                    await host.invoke_json(case.operation, case.arguments)
                assert _read_corpus_tree(root) == case.expected_files
                return
            outcome = await host.invoke_json(case.operation, case.arguments)
            assert isinstance(outcome, PatchResult)
            assert outcome.status is case.expected_status
            assert _read_corpus_tree(root) == case.expected_files
            projection = project_model_result(outcome)
            assert set(projection) == {
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
            }
            public = repr(projection)
            private_markers = {
                str(root),
                str(namespace),
                *(path for path, _value in case.initial_files),
                *(
                    value.decode("utf-8", errors="ignore").strip()
                    for _path, value in case.initial_files
                    if value.strip()
                ),
            }
            assert all(marker not in public for marker in private_markers)
        finally:
            await bundle.manager.__aexit__(None, None, None)
            assert binder.runtime._process._process is None

    run(exercise())


def test_patch_phase_10_reaps_a_lost_native_runtime(
    tmp_path: Path,
) -> None:
    """Fence and reap a lost selected child without fabricating settlement."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)

    async def exercise() -> None:
        """Lose the native process and require a closed runtime outcome."""
        assert await _native_probe()
        scope = await runtime.resolve(ScopeSelection(ContextKind.SANDBOX))
        process = runtime._process._process
        assert process is not None
        process.terminate()
        await process.wait()
        with pytest.raises(TargetInspectionError) as lost:
            await runtime.inspect(
                InspectionRequest(scope, (LogicalPath("note.txt"),))
            )
        assert lost.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        await runtime.close()
        assert runtime._process._process is None

    run(exercise())


def test_patch_phase_10_cancellation_reconciles_from_fresh_client(
    tmp_path: Path,
) -> None:
    """Persist pending on cancellation and deliver exact terminal truth."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    note = root / "note.txt"
    note.write_text("before\n", encoding="utf-8")
    clock = _RuntimeClock()
    approvals = ApprovalService(_RuntimeBroker(), clock, RuntimeGrantStore())
    authority = HmacDurableApprovalAuthority.random()
    backend = InMemoryDurablePatchBackend(approval_verifier=authority)
    store = _BlockingFenceStore(backend)
    configuration = SandboxPatchServiceConfiguration(
        _runtime_subject(),
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, authority),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )
    binder = SandboxPatchRuntimeBinder.from_settings(
        _settings(root, namespace),
        configuration,
        _runtime_policy(),
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True, store),
        PatchPersistenceBinding(True, store),
    )

    async def exercise() -> None:
        """Cancel at the real effect fence and attach a new public host."""
        assert await _native_probe()
        loader = PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        )
        bundle = await loader.load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        await bundle.toolset.__aenter__()
        arguments: dict[str, object] = {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        }
        host = bundle.toolset.sdk_host()
        invocation = create_task(
            host.invoke_json(OperationType.EDIT, arguments)
        )
        await store.effect_reached.wait()
        invocation.cancel()
        pending = await invocation
        assert isinstance(pending, PatchPending)
        assert pending.lifecycle is LifecyclePhase.SETTLEMENT_PENDING
        assert note.read_text(encoding="utf-8") == "before\n"
        fresh = await loader.load(enable_tools=["patch.edit"])
        assert fresh.toolset is not None
        fresh_host = fresh.toolset.sdk_host()
        attached = await fresh_host.retransmit_json(
            OperationType.EDIT,
            arguments,
            pending.request_id,
            pending.correlation_id,
        )
        assert attached == pending
        assert isinstance(attached, PatchPending)
        terminal = create_task(fresh_host.await_terminal(attached))
        store.release_effect.set()
        result = await terminal
        assert result.lifecycle is LifecyclePhase.REQUEST_COMPLETED
        assert note.read_text(encoding="utf-8") == "after\n"
        await bundle.toolset.__aexit__(None, None, None)
        assert binder.runtime._process._process is None

    run(exercise())


def test_patch_phase_10_fresh_runtime_reaps_and_settles_lost_worker(
    tmp_path: Path,
) -> None:
    """Replace an expired reaped owner with durable indeterminate truth."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    note = root / "note.txt"
    note.write_text("before\n", encoding="utf-8")
    clock = _RuntimeClock()
    approvals = ApprovalService(_RuntimeBroker(), clock, RuntimeGrantStore())
    authority = HmacDurableApprovalAuthority.random()
    backend = InMemoryDurablePatchBackend(approval_verifier=authority)
    store = _BlockingFenceStore(backend)
    configuration = SandboxPatchServiceConfiguration(
        _runtime_subject(),
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, authority),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )

    def create_binder() -> SandboxPatchRuntimeBinder:
        """Create a fresh process owner over the shared durable backend."""
        return SandboxPatchRuntimeBinder.from_settings(
            _settings(root, namespace),
            configuration,
            _runtime_policy(),
            PatchApprovalBinding(True),
            PatchCoordinatorBinding(True, store),
            PatchPersistenceBinding(True, store),
        )

    async def exercise() -> None:
        """Lose the first process, then recover through a fresh service."""
        assert await _native_probe()
        original_binder = create_binder()
        original_bundle = await PatchToolLoader(
            original_binder,
            PatchTestHostProfile(enabled=True, authenticated=True),
        ).load(enable_tools=["patch.edit"])
        assert original_bundle.toolset is not None
        await original_bundle.toolset.__aenter__()
        arguments: dict[str, object] = {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        }
        invocation = create_task(
            original_bundle.toolset.sdk_host().invoke_json(
                OperationType.EDIT,
                arguments,
            )
        )
        await store.effect_reached.wait()
        invocation.cancel()
        pending = await invocation
        assert isinstance(pending, PatchPending)
        service = original_bundle.toolset._service
        assert isinstance(service, SandboxPatchSdkService)
        request_access = service._requests[pending.request_id].access
        process = original_binder.runtime._process._process
        assert process is not None
        process.terminate()
        await process.wait()
        store.release_effect.set()
        await original_bundle.toolset.__aexit__(None, None, None)
        snapshot = await store.inspect(request_access)
        assert snapshot.worker_bound
        assert snapshot.worker_reaped
        assert snapshot.terminal is None
        clock.advance(20)

        fresh_binder = create_binder()
        fresh_bundle = await PatchToolLoader(
            fresh_binder,
            PatchTestHostProfile(enabled=True, authenticated=True),
        ).load(enable_tools=["patch.edit"])
        assert fresh_bundle.toolset is not None
        await fresh_bundle.toolset.__aenter__()
        result = await fresh_bundle.toolset.sdk_host().retransmit_json(
            OperationType.EDIT,
            arguments,
            pending.request_id,
            pending.correlation_id,
        )
        assert isinstance(result, PatchResult)
        assert result.status is PatchStatus.INDETERMINATE
        assert result.lifecycle is LifecyclePhase.REQUEST_COMPLETED
        assert note.read_text(encoding="utf-8") == "before\n"
        terminal = await store.inspect(request_access)
        assert terminal.terminal is not None
        assert terminal.terminal.result == result
        await fresh_bundle.toolset.__aexit__(None, None, None)

    run(exercise())


@pytest.mark.parametrize("failure_point", ("worker_binding", "bind_worker"))
def test_patch_phase_10_recovers_every_post_claim_binding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """Settle no-child post-claim failures through a fresh service."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    note = root / "note.txt"
    note.write_text("before\n", encoding="utf-8")
    clock = _RuntimeClock()
    approvals = ApprovalService(_RuntimeBroker(), clock, RuntimeGrantStore())
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    configuration = SandboxPatchServiceConfiguration(
        _runtime_subject(),
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, authority),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )

    def create_binder() -> SandboxPatchRuntimeBinder:
        """Create one service over the shared durable request domain."""
        return SandboxPatchRuntimeBinder.from_settings(
            _settings(root, namespace),
            configuration,
            _runtime_policy(),
            PatchApprovalBinding(True),
            PatchCoordinatorBinding(True, store),
            PatchPersistenceBinding(True, store),
        )

    original_worker_binding = SandboxPatchSdkService._worker_binding

    async def worker_binding_failure(
        service: SandboxPatchSdkService,
    ) -> DurableWorkerBinding:
        """Fail after the runtime has issued and started its child."""
        await original_worker_binding(service)
        raise RuntimeError("injected worker binding failure")

    async def bind_worker_failure(*arguments: object) -> None:
        """Fail before the derived binding becomes durable."""
        del arguments
        raise RuntimeError("injected durable bind failure")

    async def exercise() -> None:
        """Attach after expiry and settle journal-derived unknown truth."""
        assert await _native_probe()
        original_binder = create_binder()
        original_bundle = await PatchToolLoader(
            original_binder,
            PatchTestHostProfile(enabled=True, authenticated=True),
        ).load(enable_tools=["patch.edit"])
        assert original_bundle.toolset is not None
        await original_bundle.toolset.__aenter__()
        arguments: dict[str, object] = {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        }
        with monkeypatch.context() as patcher:
            if failure_point == "worker_binding":
                patcher.setattr(
                    SandboxPatchSdkService,
                    "_worker_binding",
                    worker_binding_failure,
                )
            else:
                patcher.setattr(store, "bind_worker", bind_worker_failure)
            pending = await original_bundle.toolset.sdk_host().invoke_json(
                OperationType.EDIT,
                arguments,
            )
        assert isinstance(pending, PatchPending)
        service = original_bundle.toolset._service
        assert isinstance(service, SandboxPatchSdkService)
        access = service._requests[pending.request_id].access
        snapshot = await store.inspect(access)
        assert not snapshot.worker_bound
        assert snapshot.worker_reaped
        assert snapshot.pending is not None
        assert original_binder.runtime._process._process is None
        await original_bundle.toolset.__aexit__(None, None, None)
        clock.advance(20)

        fresh_binder = create_binder()
        fresh_bundle = await PatchToolLoader(
            fresh_binder,
            PatchTestHostProfile(enabled=True, authenticated=True),
        ).load(enable_tools=["patch.edit"])
        assert fresh_bundle.toolset is not None
        await fresh_bundle.toolset.__aenter__()
        recovered = await fresh_bundle.toolset.sdk_host().retransmit_json(
            OperationType.EDIT,
            arguments,
            pending.request_id,
            pending.correlation_id,
        )
        assert isinstance(recovered, PatchResult)
        assert recovered.status is PatchStatus.INDETERMINATE
        assert recovered.lifecycle is LifecyclePhase.REQUEST_COMPLETED
        terminal = await store.inspect(access)
        assert terminal.terminal is not None
        assert terminal.terminal.result == recovered
        assert note.read_text(encoding="utf-8") == "before\n"
        await fresh_bundle.toolset.__aexit__(None, None, None)

    run(exercise())


def test_patch_phase_10_recovers_pending_factory_failure_after_claim(
    tmp_path: Path,
) -> None:
    """Reap and recover when pending identity construction fails."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    note = root / "note.txt"
    note.write_text("before\n", encoding="utf-8")
    clock = _RuntimeClock()
    approvals = ApprovalService(_RuntimeBroker(), clock, RuntimeGrantStore())
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    configuration = SandboxPatchServiceConfiguration(
        _runtime_subject(),
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, authority),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )

    def pending_failure(
        correlation_id: PatchObserverCorrelationId,
        duration: DurationTicks,
    ) -> DurablePendingRequest:
        """Inject a failure at the validated pending-identity seam."""
        del correlation_id, duration
        raise RuntimeError("injected pending identity failure")

    def create_binder(
        selected: SandboxPatchServiceConfiguration,
    ) -> SandboxPatchRuntimeBinder:
        """Create one process owner over the shared durable domain."""
        return SandboxPatchRuntimeBinder.from_settings(
            _settings(root, namespace),
            selected,
            _runtime_policy(),
            PatchApprovalBinding(True),
            PatchCoordinatorBinding(True, store),
            PatchPersistenceBinding(True, store),
        )

    async def exercise() -> None:
        """Fail after claim, then settle through a fresh service."""
        assert await _native_probe()
        failed_binder = create_binder(
            replace(configuration, pending_factory=pending_failure)
        )
        failed_bundle = await PatchToolLoader(
            failed_binder,
            PatchTestHostProfile(enabled=True, authenticated=True),
        ).load(enable_tools=["patch.edit"])
        assert failed_bundle.toolset is not None
        await failed_bundle.toolset.__aenter__()
        arguments: dict[str, object] = {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        }
        with pytest.raises(PatchToolError, match="host reconciliation"):
            await failed_bundle.toolset.sdk_host().invoke_json(
                OperationType.EDIT, arguments
            )
        service = failed_bundle.toolset._service
        assert isinstance(service, SandboxPatchSdkService)
        assert len(service._requests) == 1
        request_id, request = next(iter(service._requests.items()))
        snapshot = await store.inspect(request.access)
        assert not snapshot.worker_bound
        assert snapshot.worker_reaped
        assert snapshot.pending is None
        assert snapshot.terminal is None
        assert failed_binder.runtime._process._process is None
        await failed_bundle.toolset.__aexit__(None, None, None)
        clock.advance(20)

        fresh_binder = create_binder(configuration)
        fresh_bundle = await PatchToolLoader(
            fresh_binder,
            PatchTestHostProfile(enabled=True, authenticated=True),
        ).load(enable_tools=["patch.edit"])
        assert fresh_bundle.toolset is not None
        await fresh_bundle.toolset.__aenter__()
        recovered = await fresh_bundle.toolset.sdk_host().retransmit_json(
            OperationType.EDIT,
            arguments,
            request_id,
            request.correlation_id,
        )
        assert isinstance(recovered, PatchResult)
        assert recovered.status is PatchStatus.INDETERMINATE
        assert note.read_text(encoding="utf-8") == "before\n"
        await fresh_bundle.toolset.__aexit__(None, None, None)

    run(exercise())


def test_patch_phase_10_retransmission_recovers_abrupt_claim_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settle an expired claim that never durably bound a worker."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    note = root / "note.txt"
    note.write_text("before\n", encoding="utf-8")
    clock = _RuntimeClock()
    approvals = ApprovalService(_RuntimeBroker(), clock, RuntimeGrantStore())
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    configuration = SandboxPatchServiceConfiguration(
        _runtime_subject(),
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, authority),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )

    def pending_failure(
        correlation_id: PatchObserverCorrelationId,
        duration: DurationTicks,
    ) -> DurablePendingRequest:
        """Stop immediately after the durable claim has completed."""
        del correlation_id, duration
        raise RuntimeError("injected abrupt post-claim crash")

    def create_binder(
        selected: SandboxPatchServiceConfiguration,
    ) -> SandboxPatchRuntimeBinder:
        """Create one runtime owner over the shared durable backend."""
        return SandboxPatchRuntimeBinder.from_settings(
            _settings(root, namespace),
            selected,
            _runtime_policy(),
            PatchApprovalBinding(True),
            PatchCoordinatorBinding(True, store),
            PatchPersistenceBinding(True, store),
        )

    async def abandon_recovery(
        service: SandboxPatchSdkService,
        lease: DurableCommitLease,
        binding: DurableWorkerBinding | None,
        durably_bound: bool,
    ) -> None:
        """Model host death before in-process no-worker cleanup can run."""
        del service, lease, binding, durably_bound

    async def exercise() -> None:
        """Retransmit after the claimed host disappears."""
        assert await _native_probe()
        failed_binder = create_binder(
            replace(configuration, pending_factory=pending_failure)
        )
        failed_bundle = await PatchToolLoader(
            failed_binder,
            PatchTestHostProfile(enabled=True, authenticated=True),
        ).load(enable_tools=["patch.edit"])
        assert failed_bundle.toolset is not None
        await failed_bundle.toolset.__aenter__()
        arguments: dict[str, object] = {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        }
        with monkeypatch.context() as patcher:
            patcher.setattr(
                SandboxPatchSdkService,
                "_reap_bound_worker",
                abandon_recovery,
            )
            with pytest.raises(PatchToolError, match="host reconciliation"):
                await failed_bundle.toolset.sdk_host().invoke_json(
                    OperationType.EDIT, arguments
                )
        service = failed_bundle.toolset._service
        assert isinstance(service, SandboxPatchSdkService)
        request_id, request = next(iter(service._requests.items()))
        snapshot = await store.inspect(request.access)
        assert snapshot.lifecycle is LifecyclePhase.COMMIT_STARTED
        assert snapshot.pending is None
        assert not snapshot.worker_bound
        assert not snapshot.worker_reaped
        assert snapshot.terminal is None
        await failed_bundle.toolset.__aexit__(None, None, None)
        clock.advance(20)

        fresh_binder = create_binder(configuration)
        fresh_bundle = await PatchToolLoader(
            fresh_binder,
            PatchTestHostProfile(enabled=True, authenticated=True),
        ).load(enable_tools=["patch.edit"])
        assert fresh_bundle.toolset is not None
        await fresh_bundle.toolset.__aenter__()
        recovered = await fresh_bundle.toolset.sdk_host().retransmit_json(
            OperationType.EDIT,
            arguments,
            request_id,
            request.correlation_id,
        )
        assert isinstance(recovered, PatchResult)
        assert recovered.status is PatchStatus.INDETERMINATE
        assert recovered.lifecycle is LifecyclePhase.REQUEST_COMPLETED
        terminal = await store.inspect(request.access)
        assert terminal.terminal is not None
        assert terminal.terminal.result == recovered
        assert note.read_text(encoding="utf-8") == "before\n"
        await fresh_bundle.toolset.__aexit__(None, None, None)

    run(exercise())


def test_patch_phase_10_reaps_every_post_bind_issuance_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durably reap a child when authority issuance fails after binding."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    (root / "note.txt").write_text("before\n", encoding="utf-8")
    clock = _RuntimeClock()
    approvals = ApprovalService(_RuntimeBroker(), clock, RuntimeGrantStore())
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    configuration = SandboxPatchServiceConfiguration(
        _runtime_subject(),
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, authority),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )
    binder = SandboxPatchRuntimeBinder.from_settings(
        _settings(root, namespace),
        configuration,
        _runtime_policy(),
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True, store),
        PatchPersistenceBinding(True, store),
    )

    async def issuance_failure(*arguments: object) -> None:
        """Fail only after the durable worker binding exists."""
        del arguments
        raise RuntimeError("injected post-bind authority failure")

    monkeypatch.setattr(
        sandbox_commit_module,
        "_issue_rooted_command_authority_for_validator",
        issuance_failure,
    )

    async def exercise() -> None:
        """Observe pending truth and exact reaping after the injected fault."""
        assert await _native_probe()
        bundle = await PatchToolLoader(
            binder,
            PatchTestHostProfile(enabled=True, authenticated=True),
        ).load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        await bundle.toolset.__aenter__()
        outcome = await bundle.toolset.sdk_host().invoke_json(
            OperationType.EDIT,
            {
                "path": "note.txt",
                "edits": [{"old_text": "before", "new_text": "after"}],
            },
        )
        assert isinstance(outcome, PatchPending)
        service = bundle.toolset._service
        assert isinstance(service, SandboxPatchSdkService)
        access = service._requests[outcome.request_id].access
        snapshot = await store.inspect(access)
        assert snapshot.worker_bound
        assert snapshot.worker_reaped
        assert snapshot.pending is not None
        assert binder.runtime._process._process is None
        assert (root / "note.txt").read_text(encoding="utf-8") == "before\n"
        await bundle.toolset.__aexit__(None, None, None)

    run(exercise())


def test_patch_phase_10_rejects_ordinary_sandbox_write_roots(
    tmp_path: Path,
) -> None:
    """Keep ordinary shell and code profiles read-only before patch startup."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    plan = _plan(
        root,
        namespace,
        ordinary_write_roots=[str(root)],
    )
    descriptor = _open_directory(root)
    try:
        status = fstat(descriptor)
        identity = TargetIdentity(
            PatchContextId("context_" + "b" * 16),
            PatchWorkspaceId("workspace_" + "b" * 16),
            PatchDomainId("domain_" + "b" * 16),
            PatchTargetId("target_" + "b" * 16),
            sandbox_protocol_id(
                SandboxWorkerProtocolVersion("sandbox-patch-runtime-v2")
            ),
            _filesystem_id(descriptor),
            _root_mount_id(descriptor, status),
            "policy-v2",
            "persistent-lease-v2",
            PatchApprovalId("approval_" + "b" * 16),
            SandboxWorkerImplementationId("seatbelt-runtime-v2"),
        )
    finally:
        close(descriptor)
    context = SandboxPatchRuntimeContext(
        identity,
        _limits(),
        ByteSize(65_536),
        None,
        SandboxChannelId("seatbelt-patch-channel-v2"),
        SandboxContextLifetimeId("seatbelt-patch-context-v2"),
        SandboxWorkerImplementationId("seatbelt-runtime-v2"),
    )
    with pytest.raises(TargetInspectionError) as denied:
        SandboxPatchRuntimeSettings(plan, context)
    assert denied.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE


def test_patch_phase_10_bubblewrap_uses_a_distinct_runtime_view(
    tmp_path: Path,
) -> None:
    """Derive Linux mounts from the selected immutable execution plan."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    settings = _settings(root, namespace, backend_name="bubblewrap")
    runtime = settings.create_runtime()
    profile = runtime.profile
    assert settings.execution_plan.settings.profile.write_roots == ()
    assert profile.workspace_view_root == "/workspace"
    assert profile.private_view_root == "/private"
    assert profile._mount_map == (
        ("/workspace", str(root)),
        ("/private", str(namespace)),
    )


def test_patch_phase_10_requires_worker_reaping_before_domain_reuse(
    tmp_path: Path,
) -> None:
    """Refuse a new owner while an expired write-capable child remains."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)

    async def exercise() -> None:
        """Use independent clients over one durable backing domain."""
        signer = HmacDurableApprovalAuthority.random()
        backend = InMemoryDurablePatchBackend(approval_verifier=signer)
        first = InMemoryDurablePatchStore(backend)
        second = InMemoryDurablePatchStore(backend)
        lease = await _claim_domain_owner(
            first,
            signer,
            runtime.profile.identity,
            "b" * 16,
            ExpiryTick(1),
        )
        binding = DurableWorkerBinding(
            "session_" + "b" * 16,
            "channel_" + "b" * 16,
            "implementation_" + "b" * 16,
            AlgorithmDigest.from_bytes(b"implementation"),
            AlgorithmDigest.from_bytes(b"root"),
        )
        await first.bind_worker(lease, binding, ExpiryTick(1))
        with pytest.raises(DurableStoreError) as live_worker:
            await _claim_domain_owner(
                second,
                signer,
                runtime.profile.identity,
                "c" * 16,
                ExpiryTick(11),
            )
        assert live_worker.value.code is DurableStoreErrorCode.FENCED
        await first.mark_worker_reaped(lease, binding)
        with pytest.raises(DurableStoreError) as recovering_owner:
            await _claim_domain_owner(
                second,
                signer,
                runtime.profile.identity,
                "c" * 16,
                ExpiryTick(11),
            )
        assert recovering_owner.value.code is DurableStoreErrorCode.FENCED
        recovery = await _settle_expired_domain_owner(
            first,
            backend,
            lease,
            "b" * 16,
            ExpiryTick(11),
        )
        replacement = await _claim_domain_owner(
            second,
            signer,
            runtime.profile.identity,
            "c" * 16,
            ExpiryTick(11),
        )
        assert replacement.fence.value == recovery.fence.value + 1

    run(exercise())


def test_patch_phase_10_pgsql_factory_shares_one_loader_owned_store(
    tmp_path: Path,
) -> None:
    """Bind local and sandbox coordination to one typed durable resource."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    authority = HmacDurableApprovalAuthority.random()
    binding = PgsqlDurablePatchStoreFactory(
        PgsqlDurablePatchStoreSettings(
            "postgresql://patch.invalid/avalan",
            pool_minimum=1,
            pool_maximum=2,
        ),
        approval_verifier=authority,
    ).bind()
    assert type(binding.store) is PgsqlDurablePatchStore
    assert binding.store is binding.resource
    clock = _RuntimeClock()
    approvals = ApprovalService(_RuntimeBroker(), clock, RuntimeGrantStore())
    configuration = SandboxPatchServiceConfiguration(
        _runtime_subject(),
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, authority),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )
    binder = SandboxPatchRuntimeBinder.from_shared_store(
        _settings(root, namespace),
        configuration,
        _runtime_policy(),
        PatchApprovalBinding(True),
        binding,
    )
    assert binder.coordinator.durable_store is binding.store
    assert binder.persistence.durable_store is binding.store
    assert binder.durable_resource is binding.store


def test_patch_phase_10_serializes_local_and_sandbox_durable_domains(
    tmp_path: Path,
) -> None:
    """Fence overlapping local and sandbox identities in one durable domain."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)

    async def exercise() -> None:
        """Use independent durable clients over the same real backend state."""
        assert await _native_probe()
        scope = await runtime.resolve(ScopeSelection(ContextKind.SANDBOX))
        signer = HmacDurableApprovalAuthority.random()
        backend = InMemoryDurablePatchBackend(approval_verifier=signer)
        sandbox_store = InMemoryDurablePatchStore(backend)
        local_store = InMemoryDurablePatchStore(backend)
        sandbox_lease = await _claim_domain_owner(
            sandbox_store,
            signer,
            runtime.profile.identity,
            "b" * 16,
            ExpiryTick(1),
        )
        local_identity = replace(
            runtime.profile.identity,
            context_id=PatchContextId("context_" + "c" * 16),
            target_id=PatchTargetId("target_" + "c" * 16),
            implementation_id="local-runtime-vary",
        )
        with pytest.raises(DurableStoreError) as serialized:
            await _claim_domain_owner(
                local_store,
                signer,
                local_identity,
                "c" * 16,
                ExpiryTick(1),
            )
        assert serialized.value.code is DurableStoreErrorCode.FENCED
        validator = _SandboxDurableCommandAuthority(
            runtime,
            scope,
            sandbox_lease,
            sandbox_store,
            _RuntimeClock(),
        )
        assert await validator.is_rooted_command_current(
            _unissued_command(
                runtime.profile.identity,
                sandbox_lease.fence.value,
                scope.cwd,
                sandbox_lease.request_id,
            )
        )
        await sandbox_store.mark_worker_absent(sandbox_lease)
        with pytest.raises(DurableStoreError) as recovering_owner:
            await _claim_domain_owner(
                local_store,
                signer,
                local_identity,
                "c" * 16,
                ExpiryTick(11),
            )
        assert recovering_owner.value.code is DurableStoreErrorCode.FENCED
        recovery = await _settle_expired_domain_owner(
            sandbox_store,
            backend,
            sandbox_lease,
            "b" * 16,
            ExpiryTick(11),
        )
        local_lease = await _claim_domain_owner(
            local_store,
            signer,
            local_identity,
            "c" * 16,
            ExpiryTick(11),
        )
        assert local_lease.fence.value == recovery.fence.value + 1
        assert not await validator.is_rooted_command_current(
            _unissued_command(
                runtime.profile.identity,
                sandbox_lease.fence.value,
                scope.cwd,
                sandbox_lease.request_id,
            )
        )
        await runtime.close()

    run(exercise())


def test_patch_phase_10_rejects_replayed_runtime_message(
    tmp_path: Path,
) -> None:
    """Require the child response to echo the exact authenticated sequence."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)
    token = b"x" * 32
    identity = _identity_payload(runtime.profile.identity)
    request: dict[str, object] = {
        "version": _MESSAGE_VERSION,
        "sequence": 2,
        "receipt": "receipt",
        "identity": identity,
        "channel_id": runtime.profile.channel_id,
        "implementation_id": runtime.profile.implementation_id,
    }
    payload = {**request, "body": {"state": "ok"}, "error": None}
    raw = dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    line = dumps(
        {"payload": payload, "mac": digest(token, raw, "sha256").hex()},
        separators=(",", ":"),
    ).encode()
    assert _response_payload(line, token, request) == {"state": "ok"}
    replay = {**payload, "sequence": 1}
    replay_raw = dumps(replay, separators=(",", ":"), sort_keys=True).encode()
    replay_line = dumps(
        {"payload": replay, "mac": digest(token, replay_raw, "sha256").hex()},
        separators=(",", ":"),
    ).encode()
    with pytest.raises(TargetInspectionError) as rejected:
        _response_payload(replay_line, token, request)
    assert rejected.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    wrong_identity = {
        **payload,
        "identity": {**identity, "target": "wrong-target"},
    }
    wrong_channel = {**payload, "channel_id": "wrong-channel"}
    out_of_order = {**payload, "sequence": 3}
    for invalid in (wrong_identity, wrong_channel, out_of_order):
        invalid_raw = dumps(
            invalid, separators=(",", ":"), sort_keys=True
        ).encode()
        invalid_line = dumps(
            {
                "payload": invalid,
                "mac": digest(token, invalid_raw, "sha256").hex(),
            },
            separators=(",", ":"),
        ).encode()
        with pytest.raises(TargetInspectionError) as mismatched:
            _response_payload(invalid_line, token, request)
        assert mismatched.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    forged_line = dumps(
        {"payload": payload, "mac": "0" * 64},
        separators=(",", ":"),
    ).encode()
    with pytest.raises(TargetInspectionError) as forged:
        _response_payload(forged_line, token, request)
    assert forged.value.code is TargetErrorCode.WORKER_UNAVAILABLE


def test_patch_phase_10_worker_rejects_forged_runtime_request() -> None:
    """Authenticate the request before the child accepts its bound identity."""
    token = b"y" * 32
    identity = {"context": "context", "target": "target"}
    config = cast(
        _WorkerChildConfig,
        {
            "receipt": "receipt",
            "identity": identity,
            "channel_id": "channel",
            "implementation_id": "implementation",
        },
    )
    request: dict[str, object] = {
        "version": _MESSAGE_VERSION,
        "sequence": 1,
        "kind": "probe",
        "receipt": "receipt",
        "identity": identity,
        "channel_id": "channel",
        "implementation_id": "implementation",
        "body": {},
    }

    def envelope(
        payload: dict[str, object], mac_token: bytes = token
    ) -> bytes:
        """Seal one canonical test request with the selected token."""
        raw = dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return dumps(
            {
                "payload": payload,
                "mac": digest(mac_token, raw, "sha256").hex(),
            },
            separators=(",", ":"),
        ).encode()

    accepted = _child_request(envelope(request), token, config, 1)
    assert accepted["sequence"] == 1
    assert accepted["identity"] == identity
    invalid_requests = (
        {**request, "sequence": 0},
        {**request, "sequence": 2},
        {**request, "channel_id": "wrong-channel"},
        {
            **request,
            "identity": {**identity, "target": "wrong-target"},
        },
    )
    for invalid in invalid_requests:
        with pytest.raises(ValueError):
            _child_request(envelope(invalid), token, config, 1)
    with pytest.raises(ValueError):
        _child_request(envelope(request, b"z" * 32), token, config, 1)


def test_patch_phase_10_deterministic_model_rejects_invalid_scripts() -> None:
    """Reject invalid deterministic plans before agent orchestration starts."""
    logger = getLogger(__name__)
    with pytest.raises(ValueError, match="deterministic tool plan is invalid"):
        DeterministicToolPlan("", {})
    with pytest.raises(ValueError, match="deterministic tool plan is invalid"):
        DeterministicToolPlan(
            "patch.edit", cast(Mapping[str, object], object())
        )
    with pytest.raises(
        ValueError, match="deterministic model script is empty"
    ):
        DeterministicModelManager([], logger)
    manager = DeterministicModelManager(["complete"], logger)

    async def exercise() -> None:
        """Exhaust the configured provider script without a fallback."""
        await manager(cast(ModelCall, object()))
        with pytest.raises(
            RuntimeError, match="deterministic model script exhausted"
        ):
            await manager(cast(ModelCall, object()))

    run(exercise())


def test_patch_phase_10_rooted_worker_wrappers_retain_target_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use rooted wrappers without accepting unissued worker capabilities."""
    root = tmp_path / "root"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    note = root / "note.txt"
    note.write_text("inside\n", encoding="utf-8")
    witness = rooted_worker_module.capture_rooted_root(root)
    snapshots = rooted_worker_module.inspect_rooted(
        rooted_worker_module.RootedInspectionProfile(root, None, 1024, 1024),
        (LogicalPath("note.txt"),),
        witness,
    )
    assert len(snapshots) == 1
    assert (
        rooted_worker_module.rooted_snapshot_payload(snapshots[0])["path"]
        == "note.txt"
    )
    with pytest.raises(TargetInspectionError) as invalid_command:
        rooted_worker_module.RootedMutationCommand(
            PatchPlanId("plan_" + "a" * 16), (), frozenset()
        )
    assert invalid_command.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
    monkeypatch.setattr(rooted_worker_module, "sys_platform", "linux")
    monkeypatch.setattr(
        rooted_worker_module, "readlink", lambda _path: "/selected/root"
    )
    assert rooted_worker_module._descriptor_path(10) == Path("/selected/root")
    monkeypatch.setattr(
        rooted_worker_module, "readlink", lambda _path: "relative/path"
    )
    with pytest.raises(TargetInspectionError) as stale_descriptor:
        rooted_worker_module._descriptor_path(10)
    assert stale_descriptor.value.code is TargetErrorCode.WITNESS_STALE
    monkeypatch.setattr(rooted_worker_module, "sys_platform", "freebsd")
    with pytest.raises(TargetInspectionError) as unsupported_platform:
        rooted_worker_module._descriptor_path(10)
    assert (
        unsupported_platform.value.code
        is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )

    def fence_failure() -> None:
        """Model the final ownership fence replacement before an effect."""
        raise CoordinatorError(CoordinatorErrorCode.FENCED)

    with pytest.raises(CoordinatorError) as fenced_effect:
        rooted_worker_module._require_effect_fence(fence_failure)
    assert fenced_effect.value.code is CoordinatorErrorCode.FENCED

    async def exercise() -> None:
        """Reject reconciliation and worker construction without issuance."""
        worker = RootedSandboxCommitWorker()
        with pytest.raises(CoordinatorError) as unissued_worker:
            await worker._reconcile_for_owner(
                PatchRequestId("request_" + "a" * 16)
            )
        assert unissued_worker.value.code is CoordinatorErrorCode.FENCED
        with pytest.raises(CoordinatorError) as unissued_endpoint:
            _sandbox_worker_for_endpoint(
                cast(coordinator_module._RootedSandboxEndpoint, object())
            )
        assert unissued_endpoint.value.code is CoordinatorErrorCode.FENCED
        endpoint = coordinator_module._RootedSandboxEndpoint(
            cast(RootedSandboxCommitChannel, object())
        )
        command = _unissued_command(_runtime(root, namespace).profile.identity)
        with pytest.raises(CoordinatorError) as unbound_commit:
            await endpoint.commit_sandbox(
                command, cast(RootedCommandAuthorityValidator, object())
            )
        assert unbound_commit.value.code is CoordinatorErrorCode.FENCED
        with pytest.raises(CoordinatorError) as unbound_reconcile:
            await endpoint.reconcile_sandbox(command.lease.request_id)
        assert unbound_reconcile.value.code is CoordinatorErrorCode.FENCED

        class CurrentAuthority(RootedCommandAuthorityValidator):
            """Expose a current validator for a corrupted authority record."""

            async def is_rooted_command_current(
                self, issued: SealedCommitCommand
            ) -> bool:
                """Return true only for the test's sealed command."""
                return issued is command

        coordinator_module._ROOTED_COMMAND_AUTHORITIES[command] = (
            coordinator_module._RootedCommandAuthority(
                CurrentAuthority(), None
            )
        )
        assert (
            await coordinator_module._consume_rooted_command_authority(command)
        ) is None

    run(exercise())


def test_patch_phase_10_durable_worker_identity_and_reaping_are_closed(
    tmp_path: Path,
) -> None:
    """Fence conflicting worker identities and unreaped expired children."""
    root = tmp_path / "root"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)
    binding = DurableWorkerBinding(
        "session_" + "d" * 16,
        "channel_" + "d" * 16,
        "implementation_" + "d" * 16,
        AlgorithmDigest.from_bytes(b"implementation"),
        AlgorithmDigest.from_bytes(b"root"),
    )
    assert len(binding.fingerprint()) == 64
    with pytest.raises(DurableStoreError) as invalid_binding:
        DurableWorkerBinding(
            "",
            binding.channel_id,
            binding.implementation_id,
            binding.implementation_digest,
            binding.root_digest,
        )
    assert invalid_binding.value.code is DurableStoreErrorCode.FENCED

    async def exercise() -> None:
        """Use one persistent durable record through fencing failures."""
        signer = HmacDurableApprovalAuthority.random()
        backend = InMemoryDurablePatchBackend(approval_verifier=signer)
        store = InMemoryDurablePatchStore(backend)
        with pytest.raises(DurableStoreError) as invalid_resource:
            DurablePatchStoreBinding(
                store, cast(AbstractAsyncContextManager[object], object())
            )
        assert (
            invalid_resource.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        lease = await _claim_domain_owner(
            store, signer, runtime.profile.identity, "d" * 16, ExpiryTick(1)
        )
        await store.bind_worker(lease, binding, ExpiryTick(1))
        replacement_binding = replace(
            binding, session_id="session_" + "e" * 16
        )
        with pytest.raises(DurableStoreError) as conflicting_binding:
            await store.bind_worker(lease, replacement_binding, ExpiryTick(1))
        assert conflicting_binding.value.code is DurableStoreErrorCode.FENCED
        with pytest.raises(DurableStoreError) as mismatched_reap:
            await store.mark_worker_reaped(lease, replacement_binding)
        assert mismatched_reap.value.code is DurableStoreErrorCode.FENCED
        wrong_lease = replace(lease, fence=SequenceNumber(2))
        with pytest.raises(DurableStoreError) as mismatched_absence:
            await store.mark_worker_absent(wrong_lease)
        assert mismatched_absence.value.code is DurableStoreErrorCode.FENCED
        record = backend.by_request[lease.request_id]
        with pytest.raises(DurableStoreError) as unreaped_expiry:
            await store.replace_expired_owner(
                record.reservation,
                lease,
                PatchCommitOwnerId("owner_" + "f" * 16),
                ExpiryTick(11),
                DurationTicks(10),
            )
        assert unreaped_expiry.value.code is DurableStoreErrorCode.FENCED
        with pytest.raises(DurableStoreError) as bound_worker_absence:
            await store.mark_worker_absent(lease)
        assert bound_worker_absence.value.code is DurableStoreErrorCode.FENCED
        await store.mark_worker_reaped(lease, binding)
        recovery = await store.replace_expired_owner(
            record.reservation,
            lease,
            PatchCommitOwnerId("owner_" + "f" * 16),
            ExpiryTick(11),
            DurationTicks(10),
        )
        await store.mark_worker_absent(recovery)
        never_bound_recovery = await store.replace_expired_owner(
            record.reservation,
            recovery,
            PatchCommitOwnerId("owner_" + "e" * 16),
            ExpiryTick(21),
            DurationTicks(10),
        )
        backend.active_leases[never_bound_recovery.domain_id] = wrong_lease
        with pytest.raises(DurableStoreError) as stale_active_lease:
            await store.bind_worker(
                never_bound_recovery, binding, ExpiryTick(21)
            )
        assert stale_active_lease.value.code is DurableStoreErrorCode.FENCED
        backend.by_request.pop(never_bound_recovery.request_id)
        store._require_unclaimed_domain(
            never_bound_recovery.domain_id, ExpiryTick(21)
        )
        assert never_bound_recovery.domain_id not in backend.active_leases

    run(exercise())


def test_patch_phase_10_rooted_publication_rethrows_unobservable_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface cleanup failures instead of claiming clean artifacts."""
    root = tmp_path / "root"
    root.mkdir()
    stage = ".stage"
    (root / stage).write_bytes(b"staged")
    target = root / "note.txt"
    target.write_bytes(b"before")
    parent_fd = _open_directory(root)
    target_fd = open_fd(target, 0)
    path = LogicalPath("note.txt")
    monkeypatch.setattr(
        rooted_worker_module,
        "_stage",
        lambda *_arguments, **_keywords: stage,
    )
    new_attempts = 0

    def publish_then_cleanup_fails(
        *_arguments: object, **_keywords: object
    ) -> None:
        """Succeed once, then model an unobservable cleanup failure."""
        nonlocal new_attempts
        new_attempts += 1
        if new_attempts == 2:
            raise RuntimeError("new artifact cleanup failed")

    new_artifacts = [ArtifactState.ABSENT]
    try:
        monkeypatch.setattr(
            rooted_worker_module,
            "_namespace_effect",
            publish_then_cleanup_fails,
        )
        with pytest.raises(RuntimeError, match="new artifact cleanup"):
            rooted_worker_module._publish_new(
                parent_fd,
                path,
                "new.txt",
                b"after",
                0o644,
                new_artifacts,
                0,
            )
        assert new_artifacts == [ArtifactState.LEAKED]

        def publish_and_cleanup_fail(
            *_arguments: object, **_keywords: object
        ) -> None:
            """Model a failed update followed by an unobservable cleanup."""
            raise RuntimeError("update cleanup failed")

        update_artifacts = [ArtifactState.ABSENT]
        monkeypatch.setattr(
            rooted_worker_module, "_namespace_effect", publish_and_cleanup_fail
        )
        with pytest.raises(RuntimeError, match="update cleanup"):
            rooted_worker_module._publish_update(
                parent_fd,
                path,
                "note.txt",
                b"after",
                0o644,
                cast(_ProtectedMetadata, object()),
                update_artifacts,
                0,
                target_fd,
            )
        assert update_artifacts == [ArtifactState.LEAKED]
    finally:
        close(target_fd)
        close(parent_fd)


def test_patch_phase_10_sandbox_wire_rejects_invalid_runtime_shapes(
    tmp_path: Path,
) -> None:
    """Reject malformed wire and sandbox facts before effects."""
    root = tmp_path / "root"
    namespace = tmp_path / "namespace"
    source_root = tmp_path / "source"
    root.mkdir()
    namespace.mkdir()
    source_root.mkdir()
    seatbelt_profile = _runtime(root, namespace).profile
    bubblewrap_profile = (
        _settings(root, namespace, backend_name="bubblewrap")
        .create_runtime()
        .profile
    )
    worker_argv = ("/usr/bin/python3", "-I", "-c", "pass")
    seatbelt_command = sandbox_commit_module._runtime_child_command(
        seatbelt_profile,
        "sandbox-exec",
        source_root,
        worker_argv,
        "encoded-config",
    )
    assert seatbelt_command[:2] == ("sandbox-exec", "-p")
    bubblewrap_command = sandbox_commit_module._runtime_child_command(
        bubblewrap_profile,
        "bwrap",
        source_root,
        worker_argv,
        "encoded-config",
    )
    assert ("--unshare-net", "--clearenv") == bubblewrap_command[11:13]
    assert "--bind" in bubblewrap_command
    assert sandbox_commit_module._backend_policy_digest(
        SandboxBackend.SEATBELT, seatbelt_command
    )
    assert sandbox_commit_module._backend_policy_digest(
        SandboxBackend.BUBBLEWRAP, bubblewrap_command
    )
    with pytest.raises(TargetInspectionError) as malformed_policy:
        sandbox_commit_module._backend_policy_digest(
            SandboxBackend.SEATBELT, ("sandbox-exec",)
        )
    assert (
        malformed_policy.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )
    with pytest.raises(TargetInspectionError) as relative_mount:
        sandbox_commit_module._bubblewrap_parent_directories(("relative",))
    assert relative_mount.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
    with pytest.raises(TargetInspectionError) as unknown_backend:
        run(
            sandbox_commit_module._runtime_backend_probe(
                cast(SandboxBackend, object())
            )
        )
    assert unknown_backend.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
    with pytest.raises(TargetInspectionError) as malformed_root:
        sandbox_commit_module._root_from_payload({"device": "wrong"})
    assert malformed_root.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    with pytest.raises(TargetInspectionError) as malformed_report:
        sandbox_commit_module._report_from_payload(
            _unissued_command(seatbelt_profile.identity), {}
        )
    assert malformed_report.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    with pytest.raises(TargetInspectionError) as malformed_response:
        sandbox_commit_module._response_payload(b"{}", b"x" * 32, {})
    assert malformed_response.value.code is TargetErrorCode.WORKER_UNAVAILABLE


def test_patch_phase_10_sdk_rejects_invalid_retransmission_ingress() -> None:
    """Reject noncanonical replay inputs before an SDK service is consulted."""
    request_id = PatchRequestId.new()
    correlation_id = PatchObserverCorrelationId.new()

    async def exercise() -> None:
        """Check replay serialization and identity gates in isolation."""
        with pytest.raises(PatchToolError, match="patch SDK arguments"):
            await PatchSdkHost.retransmit_json(
                cast(PatchSdkHost, object()),
                OperationType.EDIT,
                {"invalid": object()},
                request_id,
                correlation_id,
            )
        with pytest.raises(PatchToolError, match="retransmission identity"):
            await PatchSdkHost.retransmit_raw(
                cast(PatchSdkHost, object()),
                OperationType.EDIT,
                b"{}",
                cast(PatchRequestId, object()),
                correlation_id,
            )
        active = cast(
            PatchSdkHost,
            SimpleNamespace(
                _snapshot=SimpleNamespace(permits=lambda _operation: True),
                _is_active=lambda: True,
            ),
        )
        with pytest.raises(PatchToolError, match="patch operation"):
            await PatchSdkHost._invoke_raw_with_identity(
                active,
                cast(OperationType, object()),
                b"{}",
                request_id,
                correlation_id,
            )
        with pytest.raises(PatchToolError, match="patch SDK request"):
            await PatchSdkHost._invoke_raw_with_identity(
                active,
                OperationType.EDIT,
                cast(bytes, "not-bytes"),
                request_id,
                correlation_id,
            )

    run(exercise())


def _worker_child_config(root: Path) -> _WorkerChildConfig:
    """Return an authenticated worker configuration for a unit path."""
    return {
        "root": str(root),
        "namespace": str(root),
        "cwd": None,
        "maximum": 1024,
        "aggregate_maximum": 4096,
        "token": "a" * 64,
        "receipt": "receipt",
        "identity": {"context": "context", "target": "target"},
        "channel_id": "channel",
        "implementation_id": "implementation",
        "implementation_digest": "implementation-digest",
        "source_digest": "source-digest",
        "implementation_root": str(root),
        "read_canary": str(root / "outside-canary"),
        "session_id": "session",
        "execution_plan": sandbox_worker_module._ExecutionPlanFingerprint(
            "execution-plan"
        ),
        "backend": "seatbelt",
        "workspace_view": "/workspace",
        "private_view": "/private",
        "context_lifetime": "lifetime",
        "protocol": "protocol",
        "persistent_lease": "persistent-lease",
        "filesystem": "filesystem",
        "mount": "mount",
    }


def _worker_request_line(
    config: _WorkerChildConfig,
    token: bytes,
    kind: str,
    body: Mapping[str, object],
    *,
    sequence: int = 1,
) -> bytes:
    """Return one authenticated child-protocol request line."""
    payload: dict[str, object] = {
        "version": _MESSAGE_VERSION,
        "sequence": sequence,
        "kind": kind,
        "receipt": config["receipt"],
        "identity": config["identity"],
        "channel_id": config["channel_id"],
        "implementation_id": config["implementation_id"],
        "body": body,
    }
    raw = dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return (
        dumps(
            {"payload": payload, "mac": digest(token, raw, "sha256").hex()},
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def test_patch_phase_10_worker_main_authenticates_close_and_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serve authenticated close and rejected-request protocol terminals."""
    config = _worker_child_config(tmp_path)
    token = b"a" * 32
    root = rooted_worker_module.RootWitness(
        FileIdentity(1, 2), "mount", "filesystem"
    )
    monkeypatch.setattr(
        sandbox_worker_module, "_child_config", lambda _value: config
    )
    monkeypatch.setattr(
        sandbox_worker_module,
        "_implementation_digest",
        lambda _root: config["implementation_digest"],
    )
    monkeypatch.setattr(
        sandbox_worker_module,
        "_worker_source_digest",
        lambda _root: config["source_digest"],
    )
    monkeypatch.setattr(
        sandbox_worker_module,
        "capture_rooted_root",
        lambda _root: root,
    )
    monkeypatch.setenv(
        "AVALAN_SANDBOX_PATCH_SESSION",
        b64encode(b"{}").decode(),
    )
    request = cast(
        sandbox_worker_module._RuntimeRequestPayload,
        {
            "version": _MESSAGE_VERSION,
            "sequence": 1,
            "kind": "close",
            "receipt": config["receipt"],
            "identity": config["identity"],
            "channel_id": config["channel_id"],
            "implementation_id": config["implementation_id"],
            "body": {},
        },
    )
    monkeypatch.setattr(
        sandbox_worker_module,
        "_child_request",
        lambda *_arguments: request,
    )

    def invoke(kind: str, body: Mapping[str, object]) -> Mapping[str, object]:
        """Run one complete worker session and return its response envelope."""
        nonlocal request
        request = {
            "version": _MESSAGE_VERSION,
            "sequence": 1,
            "kind": kind,
            "receipt": config["receipt"],
            "identity": config["identity"],
            "channel_id": config["channel_id"],
            "implementation_id": config["implementation_id"],
            "body": body,
        }
        output = BytesIO()
        monkeypatch.setattr(
            sandbox_worker_module,
            "stdin",
            SimpleNamespace(
                buffer=BytesIO(_worker_request_line(config, token, kind, body))
            ),
        )
        monkeypatch.setattr(
            sandbox_worker_module,
            "stdout",
            SimpleNamespace(buffer=output),
        )
        assert sandbox_worker_module.main() == 0
        return cast(Mapping[str, object], loads(output.getvalue()))

    closed = invoke("close", {})
    assert closed["payload"] == {
        "version": _MESSAGE_VERSION,
        "sequence": 1,
        "receipt": "receipt",
        "identity": config["identity"],
        "channel_id": "channel",
        "implementation_id": "implementation",
        "body": {},
        "error": None,
    }
    rejected = invoke("witness", {"unexpected": True})
    assert cast(Mapping[str, object], rejected["payload"])["error"] == (
        TargetErrorCode.WORKER_UNAVAILABLE.value
    )


def test_patch_phase_10_worker_main_rejects_boot_and_protocol_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed before serving an unverifiable worker session."""
    monkeypatch.delenv("AVALAN_SANDBOX_PATCH_SESSION", raising=False)
    assert sandbox_worker_module.main() == 2

    monkeypatch.setenv("AVALAN_SANDBOX_PATCH_SESSION", "not-base64")
    assert sandbox_worker_module.main() == 2

    config = _worker_child_config(tmp_path)
    monkeypatch.setenv(
        "AVALAN_SANDBOX_PATCH_SESSION", b64encode(b"{}").decode()
    )
    monkeypatch.setattr(
        sandbox_worker_module, "_child_config", lambda _value: config
    )
    monkeypatch.setattr(
        sandbox_worker_module,
        "_implementation_digest",
        lambda _root: "wrong-digest",
    )
    monkeypatch.setattr(
        sandbox_worker_module,
        "_worker_source_digest",
        lambda _root: config["source_digest"],
    )
    assert sandbox_worker_module.main() == 2

    root = rooted_worker_module.RootWitness(
        FileIdentity(1, 2), "mount", "filesystem"
    )
    monkeypatch.setattr(
        sandbox_worker_module,
        "_implementation_digest",
        lambda _root: config["implementation_digest"],
    )
    monkeypatch.setattr(
        sandbox_worker_module,
        "capture_rooted_root",
        lambda _root: root,
    )
    monkeypatch.setattr(
        sandbox_worker_module,
        "stdin",
        SimpleNamespace(buffer=BytesIO()),
    )
    assert sandbox_worker_module.main() == 2

    monkeypatch.setattr(
        sandbox_worker_module,
        "stdin",
        SimpleNamespace(buffer=BytesIO(b"{}\n")),
    )

    def reject_message(*_arguments: object) -> object:
        """Model a malformed authenticated child message."""
        raise ValueError

    monkeypatch.setattr(
        sandbox_worker_module, "_child_request", reject_message
    )
    assert sandbox_worker_module.main() == 2


def test_patch_phase_10_worker_protocol_primitives_are_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate worker digests, decoding, dispatch, and fence permits."""
    config = _worker_child_config(tmp_path)
    token = b"a" * 32
    implementation = tmp_path / "implementation"
    implementation.mkdir()
    (implementation / "worker.py").write_text("worker\n", encoding="utf-8")
    source = implementation / "avalan"
    source.mkdir()
    (source / "worker.py").write_text("source\n", encoding="utf-8")
    assert (
        len(sandbox_worker_module._implementation_digest(implementation)) == 64
    )
    assert len(sandbox_worker_module._worker_source_digest(source)) == 64
    with pytest.raises(ValueError):
        sandbox_worker_module._implementation_digest(tmp_path / "missing")
    with pytest.raises(ValueError):
        sandbox_worker_module._worker_source_digest(tmp_path / "missing")
    assert sandbox_worker_module._child_config(config) == config
    with pytest.raises(ValueError):
        sandbox_worker_module._child_config({"root": str(tmp_path)})
    invalid_cwd = dict(config)
    invalid_cwd["cwd"] = 1
    with pytest.raises(ValueError):
        sandbox_worker_module._child_config(invalid_cwd)

    primitive_rejections: tuple[Callable[[], object], ...] = (
        lambda: sandbox_worker_module._mapping([], set()),
        lambda: sandbox_worker_module._string_mapping({"key": 1}),
        lambda: sandbox_worker_module._string_list(["value", 1]),
        lambda: sandbox_worker_module._string(""),
        lambda: sandbox_worker_module._integer(-1),
        lambda: sandbox_worker_module._b64("%%%"),
    )
    for primitive in primitive_rejections:
        with pytest.raises(TargetInspectionError) as rejected_primitive:
            primitive()
        assert (
            rejected_primitive.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        )

    root = rooted_worker_module.RootWitness(
        FileIdentity(1, 2), "mount", "filesystem"
    )
    root_payload = sandbox_worker_module._root_payload(root)
    assert sandbox_worker_module._root_from_payload(root_payload) == root
    with pytest.raises(TargetInspectionError) as malformed_root:
        sandbox_worker_module._root_from_payload({"device": "wrong"})
    assert malformed_root.value.code is TargetErrorCode.WORKER_UNAVAILABLE

    request = cast(
        sandbox_worker_module._RuntimeRequestPayload,
        {
            "version": _MESSAGE_VERSION,
            "sequence": 1,
            "kind": "commit",
            "receipt": config["receipt"],
            "identity": config["identity"],
            "channel_id": config["channel_id"],
            "implementation_id": config["implementation_id"],
            "body": {},
        },
    )
    monkeypatch.setattr(sandbox_worker_module, "getpid", lambda: 456)
    monkeypatch.setattr(
        sandbox_worker_module, "inspect_rooted", lambda *_args: ()
    )
    witness, witness_closed = sandbox_worker_module._child_dispatch(
        "witness", {}, config, root, request, token
    )
    assert witness == {"root": root_payload}
    assert not witness_closed
    canary, canary_closed = sandbox_worker_module._child_dispatch(
        "canary", {}, config, root, request, token
    )
    assert canary == {"pid": 456, "outside_read_denied": True}
    assert not canary_closed
    inspected, inspected_closed = sandbox_worker_module._child_dispatch(
        "inspect",
        {"paths": ["note.txt"], "root": root_payload},
        config,
        root,
        request,
        token,
    )
    assert inspected == {"snapshots": []}
    assert not inspected_closed

    permit = _worker_request_line(
        config,
        token,
        "fence_permit",
        {"effect": 1, "allowed": True},
    )
    output = BytesIO()
    monkeypatch.setattr(
        sandbox_worker_module,
        "stdin",
        SimpleNamespace(buffer=BytesIO(permit)),
    )
    monkeypatch.setattr(
        sandbox_worker_module,
        "stdout",
        SimpleNamespace(buffer=output),
    )
    sandbox_worker_module._FenceChecker(request, config, token).check()
    control = cast(Mapping[str, object], loads(output.getvalue()))
    assert cast(Mapping[str, object], control["payload"])["body"] == {
        "control": "fence",
        "effect": 1,
    }

    with pytest.raises(ValueError):
        sandbox_worker_module._child_request(b"{}", token, config, 1)
    with pytest.raises(ValueError):
        sandbox_worker_module._runtime_request_from_payload({})
    invalid_body = {
        "version": _MESSAGE_VERSION,
        "sequence": 1,
        "kind": "close",
        "receipt": config["receipt"],
        "identity": config["identity"],
        "channel_id": config["channel_id"],
        "implementation_id": config["implementation_id"],
        "body": [],
    }
    with pytest.raises(ValueError):
        sandbox_worker_module._runtime_request_from_payload(invalid_body)
    invalid_version = {**invalid_body, "body": {}, "version": -1}
    with pytest.raises(ValueError):
        sandbox_worker_module._runtime_request_from_payload(invalid_version)


def test_patch_phase_10_rejects_forged_worker_channel() -> None:
    """Keep the public worker factory closed to arbitrary protocol objects."""

    class ForgedChannel:
        """Mimic the old structural channel without runtime provenance."""

        async def commit_sandbox(
            self,
            command: SealedCommitCommand,
            validator: RootedCommandAuthorityValidator,
        ) -> WorkerReport:
            """Expose an invalid structural commit shape."""
            del command, validator
            raise AssertionError("forged channel must not run")

        async def reconcile_sandbox(
            self, request_id: PatchRequestId
        ) -> WorkerReport:
            """Expose an invalid structural reconciliation shape."""
            del request_id
            raise AssertionError("forged channel must not run")

    ForgedChannel.__module__ = "avalan.patch.sandbox_commit"
    ForgedChannel.__qualname__ = "_SandboxEndpoint"
    channel = ForgedChannel()
    assert isinstance(channel, RootedSandboxCommitChannel)
    with pytest.raises(CoordinatorError) as rejected:
        _rooted_sandbox_endpoint(channel)
    assert rejected.value.code is CoordinatorErrorCode.FENCED
    forged_endpoint = object.__new__(sandbox_commit_module._SandboxEndpoint)
    with pytest.raises(CoordinatorError) as unissued:
        _rooted_sandbox_endpoint(forged_endpoint)
    assert unissued.value.code is CoordinatorErrorCode.FENCED
    assert not hasattr(sandbox_commit_module, "_issue_sandbox_endpoint")
    assert not hasattr(sandbox_commit_module, "_is_issued_sandbox_endpoint")


def test_patch_phase_10_fences_immediately_before_namespace_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recheck durable ownership after validation and before the syscall."""
    order: list[str] = []

    def validate(*arguments: object, **keywords: object) -> None:
        """Record both final rooted validations."""
        del arguments, keywords
        order.append("validate")

    def fence() -> None:
        """Record the final durable owner check."""
        order.append("fence")

    def effect() -> None:
        """Record the requested namespace syscall."""
        order.append("effect")

    monkeypatch.setattr(
        rooted_worker_module,
        "_validate_namespace_context",
        validate,
    )
    rooted_worker_module._namespace_effect(
        -1,
        LogicalPath("note.txt"),
        effect,
        fence_check=fence,
    )
    assert order == ["validate", "validate", "fence", "effect"]

    order.clear()

    def replaced_fence() -> None:
        """Model ownership replacement at the last possible boundary."""
        order.append("fence")
        raise CoordinatorError(CoordinatorErrorCode.FENCED)

    with pytest.raises(CoordinatorError) as stale:
        rooted_worker_module._namespace_effect(
            -1,
            LogicalPath("note.txt"),
            effect,
            fence_check=replaced_fence,
        )
    assert stale.value.code is CoordinatorErrorCode.FENCED
    assert order == ["validate", "validate", "fence"]


@pytest.mark.parametrize("replacement", ("root", "ancestor"))
def test_patch_phase_10_stage_revalidates_current_root_before_create(
    tmp_path: Path, replacement: str
) -> None:
    """Create no private artifact after the selected root path is replaced."""
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    witness = rooted_worker_module.capture_rooted_root(root)
    root_fd = _open_directory(root)
    cwd_fd = _open_directory(root)
    parent_fd = _open_directory(nested)
    cwd_status = fstat(cwd_fd)
    parent_status = fstat(parent_fd)
    root_token = rooted_worker_module._ROOT_DESCRIPTOR.set(root_fd)
    parents_token = rooted_worker_module._PARENT_IDENTITIES.set(
        {
            LogicalPath("nested"): FileIdentity(
                parent_status.st_dev, parent_status.st_ino
            )
        }
    )
    context_token = rooted_worker_module._COMMIT_CONTEXT.set(
        rooted_worker_module._CommitContext(
            root_fd,
            cwd_fd,
            FileIdentity(cwd_status.st_dev, cwd_status.st_ino),
            witness,
            root,
        )
    )
    replaced = False

    def replace_selected_path(stage: str) -> None:
        """Replace the selected root or ancestor at the staging barrier."""
        nonlocal replaced
        if stage != "target.stage_artifact" or replaced:
            return
        replaced = True
        if replacement == "root":
            root.rename(tmp_path / "parked-root")
            root.mkdir()
        else:
            nested.rename(root / "parked-nested")
            nested.mkdir()

    barrier_token = rooted_worker_module._ROOTED_BARRIER.set(
        replace_selected_path
    )
    try:
        with pytest.raises(TargetInspectionError) as stale:
            rooted_worker_module._stage(
                parent_fd,
                b"value",
                0o600,
                path=LogicalPath("nested/note.txt"),
            )
        assert stale.value.code is TargetErrorCode.WITNESS_STALE
        assert replaced
        assert not tuple(tmp_path.rglob(".avalan-patch-*"))
    finally:
        rooted_worker_module._ROOTED_BARRIER.reset(barrier_token)
        rooted_worker_module._COMMIT_CONTEXT.reset(context_token)
        rooted_worker_module._PARENT_IDENTITIES.reset(parents_token)
        rooted_worker_module._ROOT_DESCRIPTOR.reset(root_token)
        close(parent_fd)
        close(cwd_fd)
        close(root_fd)


def test_patch_phase_10_stage_checks_fence_immediately_before_create(
    tmp_path: Path,
) -> None:
    """Create no private artifact after ownership changes at staging."""
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    witness = rooted_worker_module.capture_rooted_root(root)
    root_fd = _open_directory(root)
    cwd_fd = _open_directory(root)
    parent_fd = _open_directory(nested)
    cwd_status = fstat(cwd_fd)
    parent_status = fstat(parent_fd)
    root_token = rooted_worker_module._ROOT_DESCRIPTOR.set(root_fd)
    parents_token = rooted_worker_module._PARENT_IDENTITIES.set(
        {
            LogicalPath("nested"): FileIdentity(
                parent_status.st_dev, parent_status.st_ino
            )
        }
    )
    context_token = rooted_worker_module._COMMIT_CONTEXT.set(
        rooted_worker_module._CommitContext(
            root_fd,
            cwd_fd,
            FileIdentity(cwd_status.st_dev, cwd_status.st_ino),
            witness,
            root,
        )
    )
    fence_checks = 0

    def replaced_fence() -> None:
        """Model durable ownership replacement at the creation boundary."""
        nonlocal fence_checks
        fence_checks += 1
        raise CoordinatorError(CoordinatorErrorCode.FENCED)

    try:
        with pytest.raises(CoordinatorError) as fenced:
            rooted_worker_module._stage(
                parent_fd,
                b"value",
                0o600,
                path=LogicalPath("nested/note.txt"),
                fence_check=replaced_fence,
            )
        assert fenced.value.code is CoordinatorErrorCode.FENCED
        assert fence_checks == 1
        assert not tuple(root.rglob(".avalan-patch-*"))
    finally:
        rooted_worker_module._COMMIT_CONTEXT.reset(context_token)
        rooted_worker_module._PARENT_IDENTITIES.reset(parents_token)
        rooted_worker_module._ROOT_DESCRIPTOR.reset(root_token)
        close(parent_fd)
        close(cwd_fd)
        close(root_fd)


def test_patch_phase_10_fences_every_private_artifact_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain private artifacts when ownership changes before cleanup."""
    parent_fd = _open_directory(tmp_path)
    path = LogicalPath("note.txt")
    try:

        def stalled_write(descriptor: int, value: bytes) -> int:
            """Force the failed-stage cleanup branch."""
            del descriptor, value
            return 0

        def replaced() -> None:
            """Replace the owner at the exact cleanup boundary."""
            raise CoordinatorError(CoordinatorErrorCode.FENCED)

        cleanup_checks: list[Callable[[], None] | None] = []

        def fenced_cleanup(
            parent: int,
            logical_path: LogicalPath,
            effect: Callable[[], None],
            **keywords: object,
        ) -> None:
            """Run only after proving the cleanup callback was forwarded."""
            del parent, logical_path, effect
            check = cast(
                Callable[[], None] | None, keywords.get("fence_check")
            )
            cleanup_checks.append(check)
            assert check is not None
            check()

        with monkeypatch.context() as patcher:
            patcher.setattr(rooted_worker_module, "write_fd", stalled_write)
            patcher.setattr(
                rooted_worker_module,
                "_before_namespace_effect",
                lambda *arguments, **keywords: None,
            )
            patcher.setattr(
                rooted_worker_module, "_namespace_effect", fenced_cleanup
            )
            with pytest.raises(rooted_worker_module._ArtifactUncertainError):
                rooted_worker_module._stage(
                    parent_fd,
                    b"value",
                    0o600,
                    path=path,
                    fence_check=replaced,
                )
        assert cleanup_checks == [replaced]
        assert tuple(tmp_path.glob(".avalan-patch-*"))

        for private_path in tmp_path.glob(".avalan-patch-*"):
            private_path.unlink()
        stage_new = ".stage-new"
        (tmp_path / stage_new).write_bytes(b"new")
        artifact_states = [ArtifactState.STAGED]
        effects = 0

        def staged_new(*arguments: object, **keywords: object) -> str:
            """Return one known private artifact for publication."""
            del arguments, keywords
            return stage_new

        def publish_then_fence(
            parent: int,
            logical_path: LogicalPath,
            effect: Callable[[], None],
            **keywords: object,
        ) -> None:
            """Replace ownership only at the publication cleanup call."""
            nonlocal effects
            del parent, logical_path
            effects += 1
            if effects == 1:
                effect()
                return
            check = cast(
                Callable[[], None] | None, keywords.get("fence_check")
            )
            assert check is replaced
            check()

        with monkeypatch.context() as patcher:
            patcher.setattr(rooted_worker_module, "_stage", staged_new)
            patcher.setattr(
                rooted_worker_module,
                "_namespace_effect",
                publish_then_fence,
            )
            with pytest.raises(CoordinatorError):
                rooted_worker_module._publish_new(
                    parent_fd,
                    path,
                    "note.txt",
                    b"new",
                    0o600,
                    artifact_states,
                    0,
                    replaced,
                )
        assert artifact_states == [ArtifactState.LEAKED]
        assert (tmp_path / stage_new).is_file()

        (tmp_path / stage_new).unlink()
        stage_update = ".stage-update"
        (tmp_path / stage_update).write_bytes(b"update")
        artifact_states = [ArtifactState.STAGED]
        effects = 0

        def staged_update(*arguments: object, **keywords: object) -> str:
            """Return one known private artifact for replacement."""
            del arguments, keywords
            return stage_update

        def fail_then_fence(
            parent: int,
            logical_path: LogicalPath,
            effect: Callable[[], None],
            **keywords: object,
        ) -> None:
            """Fail publication, then replace ownership before cleanup."""
            nonlocal effects
            del parent, logical_path, effect
            effects += 1
            if effects == 1:
                raise OSError("publication unavailable")
            check = cast(
                Callable[[], None] | None, keywords.get("fence_check")
            )
            assert check is replaced
            check()

        with monkeypatch.context() as patcher:
            patcher.setattr(rooted_worker_module, "_stage", staged_update)
            patcher.setattr(
                rooted_worker_module,
                "_namespace_effect",
                fail_then_fence,
            )
            with pytest.raises(CoordinatorError):
                rooted_worker_module._publish_update(
                    parent_fd,
                    path,
                    "note.txt",
                    b"update",
                    0o600,
                    _ProtectedMetadata((), 0, None),
                    artifact_states,
                    0,
                    parent_fd,
                    replaced,
                )
        assert artifact_states == [ArtifactState.LEAKED]
        assert (tmp_path / stage_update).is_file()
    finally:
        close(parent_fd)


def test_patch_phase_10_rejects_unpinned_worker_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject installed worker drift before copying an executable bundle."""
    monkeypatch.setattr(
        sandbox_commit_module,
        "_PINNED_WORKER_SOURCE_DIGEST",
        "0" * 64,
    )
    with pytest.raises(TargetInspectionError) as drifted:
        sandbox_commit_module._ImplementationBundle.create(tmp_path)
    assert drifted.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE


def test_patch_phase_10_sandbox_commit_requires_pycparser_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when a worker namespace lacks its parser dependency."""
    original_find_spec = importlib_util.find_spec

    def without_pycparser(
        name: str,
        package: str | None = None,
    ) -> object:
        """Hide only the required worker parser from a fresh module load."""
        if name == "pycparser":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib_util, "find_spec", without_pycparser)
    with pytest.raises(RuntimeError, match="pycparser package is unavailable"):
        run_path(str(Path("src/avalan/patch/sandbox_commit.py").resolve()))


def test_patch_phase_10_sandbox_result_and_owned_resource_branches() -> None:
    """Project journal facts and close owned resources on every outcome."""
    request_id = PatchRequestId("request_" + "a" * 16)
    plan_id = PatchPlanId("plan_" + "a" * 16)
    step = JournalStep(
        PatchStepId("step_" + "a" * 16),
        PatchLineageId("lineage_" + "a" * 16),
        CommitStepState.COMMITTED,
    )

    def journal(*artifacts: ArtifactJournal) -> SettlementJournal:
        """Build one settled report with controlled target artifact facts."""
        return SettlementJournal(
            (step,), artifacts, PostconditionState.ESTABLISHED
        )

    assert (
        sandbox_commit_module._durable_artifact_state(
            ArtifactState.ABSENT
        ).value
        == "not_created"
    )
    assert (
        sandbox_commit_module._durable_artifact_state(
            ArtifactState.CLEANED
        ).value
        == "removed"
    )
    assert (
        sandbox_commit_module._durable_artifact_state(
            ArtifactState.LEAKED
        ).value
        == "leaked"
    )
    assert (
        sandbox_commit_module._durable_artifact_state(
            ArtifactState.UNKNOWN
        ).value
        == "unknown"
    )
    assert (
        sandbox_commit_module._durable_artifact_state(
            ArtifactState.STAGED
        ).value
        == "present"
    )
    assert (
        sandbox_commit_module._artifact_state(journal())
        is ArtifactState.ABSENT
    )
    assert (
        sandbox_commit_module._artifact_state(
            journal(
                ArtifactJournal("artifact_" + "a" * 16, ArtifactState.UNKNOWN)
            )
        )
        is ArtifactState.UNKNOWN
    )
    assert (
        sandbox_commit_module._artifact_state(
            journal(
                ArtifactJournal("artifact_" + "a" * 16, ArtifactState.LEAKED)
            )
        )
        is ArtifactState.LEAKED
    )
    assert (
        sandbox_commit_module._artifact_state(
            journal(
                ArtifactJournal("artifact_" + "a" * 16, ArtifactState.CLEANED)
            )
        )
        is ArtifactState.CLEANED
    )
    assert (
        sandbox_commit_module._worker_result(
            request_id, plan_id, WorkerReport(WorkerState.LIVE, None)
        ).status
        is PatchStatus.INDETERMINATE
    )
    assert (
        sandbox_commit_module._result(
            request_id,
            plan_id,
            (),
            ArtifactState.ABSENT,
            PostconditionState.ESTABLISHED,
            None,
            None,
        ).status
        is PatchStatus.COMMIT_FAILED
    )
    assert (
        sandbox_commit_module._result(
            request_id,
            plan_id,
            (CommitStepState.UNKNOWN, CommitStepState.COMMITTED),
            ArtifactState.UNKNOWN,
            PostconditionState.ESTABLISHED,
            None,
            None,
        ).status
        is PatchStatus.INDETERMINATE
    )
    assert (
        sandbox_commit_module._result(
            request_id,
            plan_id,
            (CommitStepState.COMMITTED, CommitStepState.NOT_COMMITTED),
            ArtifactState.CLEANED,
            PostconditionState.ESTABLISHED,
            None,
            None,
        ).status
        is PatchStatus.PARTIAL
    )

    events: list[str] = []

    class Durable:
        """Record durable resource ownership ordering."""

        async def __aenter__(self) -> object:
            """Record durable activation before service construction."""
            events.append("durable.enter")
            return self

        async def __aexit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_value: BaseException | None,
            _traceback: object,
        ) -> bool:
            """Record durable closure after service closure."""
            events.append("durable.exit")
            return False

    class Runtime:
        """Record worker reaping when resource entry fails."""

        async def close(self) -> None:
            """Record the runtime cleanup operation."""
            events.append("runtime.close")

    class Service:
        """Record service context entry and exit behavior."""

        def __init__(self, fail: bool = False) -> None:
            """Choose whether entry fails after durable activation."""
            self.fail = fail

        async def __aenter__(self) -> object:
            """Enter or fail after the durable resource is open."""
            events.append("service.enter")
            if self.fail:
                raise RuntimeError("service entry failed")
            return self

        async def __aexit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_value: BaseException | None,
            _traceback: object,
        ) -> bool:
            """Record service closure before durable release."""
            events.append("service.exit")
            return False

    async def exercise_resources() -> None:
        """Prove normal and failing ownership sequences are both closed."""
        durable = Durable()
        normal = sandbox_commit_module._SandboxPatchOwnedResources(
            cast(SandboxPatchRuntime, Runtime()),
            cast(SandboxPatchSdkService, Service()),
            durable,
        )
        assert await normal.__aenter__() is normal
        assert not await normal.__aexit__(None, None, None)
        failed = sandbox_commit_module._SandboxPatchOwnedResources(
            cast(SandboxPatchRuntime, Runtime()),
            cast(SandboxPatchSdkService, Service(fail=True)),
            durable,
        )
        with pytest.raises(RuntimeError, match="service entry failed"):
            await failed.__aenter__()

    run(exercise_resources())
    assert events == [
        "durable.enter",
        "service.enter",
        "service.exit",
        "durable.exit",
        "durable.enter",
        "service.enter",
        "runtime.close",
        "durable.exit",
    ]


def test_patch_phase_10_runtime_process_protocol_and_reap_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exchange authenticated replies and clear idle runtime resources."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    owner = _runtime(root, namespace)._process
    token = b"p" * 32
    receipt = sandbox_commit_module.SandboxProfileReceipt("receipt")

    class Writer:
        """Record authenticated child requests without native IO."""

        def __init__(self) -> None:
            """Start with no request envelopes."""
            self.values: list[bytes] = []

        def write(self, value: bytes) -> None:
            """Store one complete protocol line."""
            self.values.append(value)

        async def drain(self) -> None:
            """Complete each bounded protocol write immediately."""

    def response_line(
        request: Mapping[str, object], body: Mapping[str, object]
    ) -> bytes:
        """Authenticate one exact child response for the recorded request."""
        payload = {
            **request,
            "body": body,
            "error": None,
        }
        raw = dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return (
            dumps(
                {
                    "payload": payload,
                    "mac": digest(token, raw, "sha256").hex(),
                },
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )

    class Reader:
        """Return one normal reply or a fence-control/reply pair."""

        def __init__(self, writer: Writer, fenced: bool) -> None:
            """Bind the writer and choose whether a permit is required."""
            self.writer = writer
            self.fenced = fenced
            self.calls = 0

        async def readline(self) -> bytes:
            """Respond only after the matching host envelope was recorded."""
            request_value = loads(self.writer.values[-1])
            assert isinstance(request_value, dict)
            request = request_value["payload"]
            assert isinstance(request, dict)
            self.calls += 1
            if self.fenced and self.calls == 1:
                return response_line(
                    request, {"control": "fence", "effect": 1}
                )
            return response_line(request, {"completed": True})

    class Process:
        """Expose only the process operations used by the protocol owner."""

        def __init__(self, fenced: bool) -> None:
            """Create writable and readable protocol endpoints."""
            self.stdin = Writer()
            self.stdout = Reader(self.stdin, fenced)
            self.returncode: int | None = None
            self.pid = 999
            self.terminated = False

        def terminate(self) -> None:
            """Record bounded process termination."""
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            """Record forced termination after a reaping timeout."""
            self.returncode = -9

        async def wait(self) -> int:
            """Return the recorded terminal process code."""
            assert self.returncode is not None
            return self.returncode

    async def current(_command: object) -> bool:
        """Approve the one authenticated fence request."""
        return True

    async def exercise() -> None:
        """Serve normal and fenced requests without a native child."""
        normal = Process(fenced=False)
        monkeypatch.setattr(owner, "_process", normal)
        monkeypatch.setattr(owner, "_token", token)
        monkeypatch.setattr(owner, "_receipt", receipt)
        assert await owner._request_locked("witness", {}) == {
            "completed": True
        }
        assert owner._sequence == 1

        fenced = Process(fenced=True)
        monkeypatch.setattr(owner, "_process", fenced)
        assert await owner._request_locked(
            "commit",
            {"sealed": True},
            command=_unissued_command(owner.profile.identity),
            validator=cast(
                RootedCommandAuthorityValidator,
                SimpleNamespace(is_rooted_command_current=current),
            ),
        ) == {"completed": True}
        assert len(fenced.stdin.values) == 2

        cleaned: list[str] = []
        canary = tmp_path / "canary"
        canary.mkdir()
        monkeypatch.setattr(owner, "_process", None)
        monkeypatch.setattr(
            owner,
            "_bundle",
            SimpleNamespace(close=lambda: cleaned.append("bundle")),
        )
        monkeypatch.setattr(owner, "_canary_root", canary)
        await owner._reap()
        assert cleaned == ["bundle"]
        assert not canary.exists()

    run(exercise())


def test_patch_phase_10_force_reaps_a_silent_child_without_protocol_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terminate, kill, and await a child whose response pipe is silent."""
    root = tmp_path / "sandbox-view"
    namespace = tmp_path / "sandbox-private"
    root.mkdir()
    namespace.mkdir()
    owner = _runtime(root, namespace)._process

    class SilentReader:
        """Hold one response read until the fake process is killed."""

        def __init__(self) -> None:
            """Create entry and release synchronization events."""
            self.entered = Event()
            self.release = Event()

        async def readline(self) -> bytes:
            """Wait silently until forced process termination."""
            self.entered.set()
            await self.release.wait()
            return b""

    class WritablePipe:
        """Accept protocol bytes without blocking the initial drain."""

        def write(self, value: bytes) -> None:
            """Accept one bounded request line."""
            assert value.endswith(b"\n")

        async def drain(self) -> None:
            """Complete the write side immediately."""

    class SilentProcess:
        """Ignore terminate and exit only after the kill fallback."""

        def __init__(self) -> None:
            """Create a silent process-shaped protocol object."""
            self.stdin = WritablePipe()
            self.stdout = SilentReader()
            self.returncode: int | None = None
            self.pid = 4242
            self.terminated = False
            self.killed = False
            self.reaped = Event()

        def terminate(self) -> None:
            """Record the bounded graceful termination attempt."""
            self.terminated = True

        def kill(self) -> None:
            """Release the blocked pipe and process wait."""
            self.killed = True
            self.returncode = -9
            self.stdout.release.set()
            self.reaped.set()

        async def wait(self) -> int:
            """Wait until the kill fallback reaps this process."""
            await self.reaped.wait()
            assert self.returncode is not None
            return self.returncode

    process = SilentProcess()
    monkeypatch.setattr(owner, "_process", process)
    monkeypatch.setattr(owner, "_token", b"s" * 32)
    monkeypatch.setattr(owner, "_receipt", "runtime-receipt")
    monkeypatch.setattr(sandbox_commit_module, "_PROCESS_CLOSE_SECONDS", 0.01)
    monkeypatch.setattr(sandbox_commit_module, "_PROCESS_REAP_SECONDS", 0.01)
    monkeypatch.setattr(sandbox_commit_module, "_PROCESS_IO_SECONDS", 1.0)

    async def exercise() -> None:
        """Close without waiting for the held protocol lock."""
        blocked = create_task(owner._request("witness", {}))
        await process.stdout.entered.wait()
        await owner.close()
        with pytest.raises(TargetInspectionError) as unavailable:
            await blocked
        assert unavailable.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        assert process.terminated
        assert process.killed
        assert owner._process is None

    run(exercise())


def test_patch_phase_10_preserves_public_request_id_in_durable_store() -> None:
    """Bind the public tool request to one durable coordination row."""
    store = InMemoryDurablePatchStore(InMemoryDurablePatchBackend())
    request_id = PatchRequestId("request_" + "c" * 16)
    identity = DurableRequestIdentity(
        PatchTenantId("tenant-ten"),
        PatchPrincipalId("principal-user"),
        PatchExecutionId("execution_" + "c" * 16),
        PolicyRouteId("sandbox-runtime-route"),
        RetransmissionKey("sandbox-runtime-retry"),
    )

    async def reserve() -> None:
        """Reserve the public request identifier through durable storage."""
        reservation = await store.reserve(
            identity,
            AlgorithmDigest.from_bytes(b"sandbox-runtime"),
            request_id,
        )
        assert reservation.request_id == request_id

    run(reserve())


def test_patch_phase_10_worker_direct_protocol_failure_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed child operations without starting a sandbox process."""
    config = _worker_child_config(tmp_path)
    token = b"a" * 32
    root = rooted_worker_module.RootWitness(
        FileIdentity(1, 2), "mount", "filesystem"
    )
    request = cast(
        sandbox_worker_module._RuntimeRequestPayload,
        {
            "version": _MESSAGE_VERSION,
            "sequence": 1,
            "kind": "witness",
            "receipt": config["receipt"],
            "identity": config["identity"],
            "channel_id": config["channel_id"],
            "implementation_id": config["implementation_id"],
            "body": {},
        },
    )
    empty_implementation = tmp_path / "empty-implementation"
    empty_implementation.mkdir()
    empty_source = tmp_path / "empty-source"
    empty_source.mkdir()
    with pytest.raises(ValueError):
        sandbox_worker_module._implementation_digest(empty_implementation)
    with pytest.raises(ValueError):
        sandbox_worker_module._worker_source_digest(empty_source)
    with pytest.raises(TargetInspectionError) as malformed_root:
        sandbox_worker_module._root_from_payload(
            {
                "device": 1,
                "inode": "wrong",
                "mount": "m",
                "filesystem": "f",
            }
        )
    assert malformed_root.value.code is TargetErrorCode.WORKER_UNAVAILABLE

    Path(config["read_canary"]).write_bytes(b"visible only in this unit view")
    monkeypatch.setattr(sandbox_worker_module, "getpid", lambda: 789)
    canary, closed = sandbox_worker_module._child_dispatch(
        "canary", {}, config, root, request, token
    )
    assert canary == {"pid": 789, "outside_read_denied": False}
    assert not closed

    invalid_dispatches: tuple[tuple[str, Mapping[str, object]], ...] = (
        ("witness", {"unexpected": True}),
        ("canary", {"unexpected": True}),
        ("inspect", {}),
        (
            "inspect",
            {"paths": [], "root": sandbox_worker_module._root_payload(root)},
        ),
        ("commit", {}),
        ("close", {"unexpected": True}),
        ("unknown", {}),
    )
    for kind, body in invalid_dispatches:
        with pytest.raises(TargetInspectionError):
            sandbox_worker_module._child_dispatch(
                kind, body, config, root, request, token
            )

    stale_root = rooted_worker_module.RootWitness(
        FileIdentity(3, 4), "other-mount", "other-filesystem"
    )
    with pytest.raises(TargetInspectionError) as stale_inspection:
        sandbox_worker_module._child_dispatch(
            "inspect",
            {
                "paths": [],
                "root": sandbox_worker_module._root_payload(stale_root),
            },
            config,
            root,
            request,
            token,
        )
    assert stale_inspection.value.code is TargetErrorCode.WITNESS_STALE

    absent_file = sandbox_worker_module._planned_file(
        {
            "path": "missing.txt",
            "present": False,
            "content_kind": None,
            "content": None,
            "metadata": None,
            "digest": None,
            "size": 0,
            "identity": None,
            "protected_metadata": None,
        }
    )
    assert not absent_file.present
    assert absent_file.bytes_value is None
    assert absent_file.metadata is None
    assert absent_file.digest is None

    def fake_mutation(*_arguments: object) -> object:
        """Model a command already authenticated by the outer worker wire."""
        return object()

    def fake_commit(
        _command: object,
        _profile: object,
        _root: object,
        _fence: Callable[[], None],
    ) -> object:
        """Return a minimal settled worker journal through the native seam."""
        return SimpleNamespace(
            journal=SimpleNamespace(
                steps=(
                    SimpleNamespace(
                        identifier=PatchStepId("step_" + "a" * 16),
                        lineage=PatchLineageId("lineage_" + "a" * 16),
                        state=CommitStepState.COMMITTED,
                    ),
                ),
                artifacts=(
                    SimpleNamespace(
                        identifier="artifact_" + "a" * 16,
                        state=ArtifactState.CLEANED,
                    ),
                ),
                postcondition=PostconditionState.ESTABLISHED,
            )
        )

    monkeypatch.setattr(
        sandbox_worker_module, "_mutation_command", fake_mutation
    )
    monkeypatch.setattr(sandbox_worker_module, "_commit_rooted", fake_commit)
    committed, closed = sandbox_worker_module._child_dispatch(
        "commit", {"already": "authenticated"}, config, root, request, token
    )
    assert not closed
    assert committed == {
        "steps": [
            {
                "id": "step_" + "a" * 16,
                "lineage": "lineage_" + "a" * 16,
                "state": CommitStepState.COMMITTED.value,
            }
        ],
        "artifacts": [
            {
                "id": "artifact_" + "a" * 16,
                "state": ArtifactState.CLEANED.value,
            }
        ],
        "postcondition": PostconditionState.ESTABLISHED.value,
    }

    monkeypatch.setattr(
        sandbox_worker_module,
        "_commit_rooted",
        lambda *_arguments: SimpleNamespace(journal=None),
    )
    with pytest.raises(TargetInspectionError) as missing_journal:
        sandbox_worker_module._child_dispatch(
            "commit",
            {"already": "authenticated"},
            config,
            root,
            request,
            token,
        )
    assert missing_journal.value.code is TargetErrorCode.WORKER_UNAVAILABLE

    config_error = dict(config)
    config_error["root"] = 1
    with pytest.raises(ValueError):
        sandbox_worker_module._child_config(config_error)
    with pytest.raises(ValueError):
        sandbox_worker_module._child_response_from_line(
            b"{}", TargetErrorCode.WORKER_UNAVAILABLE, token
        )

    output = BytesIO()
    monkeypatch.setattr(
        sandbox_worker_module, "stdout", SimpleNamespace(buffer=output)
    )
    monkeypatch.setattr(
        sandbox_worker_module, "stdin", SimpleNamespace(buffer=BytesIO())
    )
    with pytest.raises(TargetInspectionError) as missing_fence:
        sandbox_worker_module._FenceChecker(request, config, token).check()
    assert missing_fence.value.code is TargetErrorCode.WORKER_UNAVAILABLE

    denied_permit = _worker_request_line(
        config, token, "fence_permit", {"effect": 1, "allowed": False}
    )
    monkeypatch.setattr(
        sandbox_worker_module,
        "stdin",
        SimpleNamespace(buffer=BytesIO(denied_permit)),
    )
    with pytest.raises(TargetInspectionError) as denied_fence:
        sandbox_worker_module._FenceChecker(request, config, token).check()
    assert denied_fence.value.code is TargetErrorCode.WITNESS_STALE


def test_patch_phase_10_pgsql_worker_cas_contracts_without_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise worker ownership SQL CAS without bypassing markers."""

    class Cursor:
        """Record parameterized statements and return configured CAS rows."""

        def __init__(self, rows: tuple[dict[str, object] | None, ...]) -> None:
            """Bind finite database outcomes for one fake transaction."""
            self.rows = iter(rows)
            self.statements: list[tuple[str, object]] = []

        async def execute(self, statement: str, parameters: object) -> None:
            """Record one closed statement and its bound parameters."""
            self.statements.append((statement, parameters))

        async def fetchone(self) -> dict[str, object] | None:
            """Return one configured compare-and-swap outcome."""
            return next(self.rows)

    identity = DurableRequestIdentity(
        PatchTenantId("tenant-phase-ten"),
        PatchPrincipalId("principal-phase-ten"),
        PatchExecutionId("execution_" + "a" * 16),
        PolicyRouteId("route-phase-ten"),
        RetransmissionKey("retry-phase-ten"),
    )
    request_id = PatchRequestId("request_" + "a" * 16)
    lease = DurableCommitLease(
        request_id,
        PatchDomainId("domain_" + "a" * 16),
        PatchCommitOwnerId("owner_" + "a" * 16),
        SequenceNumber(1),
        ExpiryTick(20),
    )
    binding = DurableWorkerBinding(
        "session_" + "a" * 16,
        "channel_" + "a" * 16,
        "implementation_" + "a" * 16,
        AlgorithmDigest.from_bytes(b"implementation"),
        AlgorithmDigest.from_bytes(b"root"),
    )
    store = PgsqlDurablePatchStore(
        type("Pool", (), {"connection": lambda self: None})()
    )
    cursor = Cursor(({"request_id": request_id.value},))

    async def transaction(
        operation: str, callback: Callable[[object], Awaitable[object]]
    ) -> object:
        """Run one public store operation against a controlled cursor."""
        del operation
        return await callback(cursor)

    async def selected_lease(*_arguments: object) -> dict[str, object]:
        """Model a current lease selected under the worker CAS lock."""
        return {}

    monkeypatch.setattr(store, "_transaction", transaction)
    monkeypatch.setattr(
        pgsql_store_module, "_select_lease_for_update", selected_lease
    )

    async def exercise() -> None:
        """Cover explicit identity and worker ownership transitions."""
        nonlocal cursor
        reservation = await store.reserve(
            identity, AlgorithmDigest.from_bytes(b"canonical"), request_id
        )
        assert reservation.request_id == request_id
        parameters = cursor.statements[0][1]
        assert isinstance(parameters, tuple)
        assert parameters[0] == request_id.value

        for method, arguments in (
            (store.bind_worker, (lease, binding, ExpiryTick(10))),
            (store.mark_worker_reaped, (lease, binding)),
            (store.mark_worker_absent, (lease,)),
        ):
            cursor = Cursor(({"request_id": request_id.value},))
            await method(*arguments)
            assert cursor.statements
            cursor = Cursor((None,))
            with pytest.raises(DurableStoreError) as fenced:
                await method(*arguments)
            assert fenced.value.code is DurableStoreErrorCode.FENCED

        cursor = Cursor((None,))
        await pgsql_store_module._require_unclaimed_domain(
            cast(PgsqlCursor, cursor),
            lease.domain_id,
            ExpiryTick(10),
        )
        cursor = Cursor(({"request_id": request_id.value},))
        with pytest.raises(DurableStoreError) as owned:
            await pgsql_store_module._require_unclaimed_domain(
                cast(PgsqlCursor, cursor),
                lease.domain_id,
                ExpiryTick(10),
            )
        assert owned.value.code is DurableStoreErrorCode.FENCED

        snapshot_row: dict[str, object] = {
            "lifecycle": LifecyclePhase.SETTLEMENT_PENDING.value,
            "journal_revision": 0,
            "worker_binding_digest": binding.fingerprint(),
            "worker_reaped": True,
            "cancellation_requested": False,
            "event_cursor": 0,
        }
        snapshot_reservation = DurableReservation(
            request_id,
            identity,
            AlgorithmDigest.from_bytes(b"canonical"),
            False,
        )
        snapshot_journal = DurableJournal(
            DurableJournalCursor(request_id, SequenceNumber(0)), (), ()
        )

        async def snapshot_journal_reader(
            _cursor: object,
            _request_id: PatchRequestId,
            _revision: SequenceNumber,
        ) -> DurableJournal:
            """Return a journal already held under the synthetic row lock."""
            return snapshot_journal

        monkeypatch.setattr(
            pgsql_store_module,
            "_reservation_from_row",
            lambda _row: snapshot_reservation,
        )
        monkeypatch.setattr(
            pgsql_store_module, "_plan_from_row", lambda _row: None
        )
        monkeypatch.setattr(
            pgsql_store_module, "_lease_from_row", lambda _row: lease
        )
        monkeypatch.setattr(
            pgsql_store_module, "_journal", snapshot_journal_reader
        )
        monkeypatch.setattr(
            pgsql_store_module, "_pending_from_row", lambda _row: None
        )
        snapshot = await pgsql_store_module._snapshot(
            cast(PgsqlCursor, cursor), cast(PgsqlRow, snapshot_row)
        )
        assert snapshot.worker_bound
        assert snapshot.worker_reaped
        assert not snapshot.cancellation_requested

        plan = DurablePlanReference(
            PatchPlanId("plan_" + "a" * 16),
            AlgorithmDigest.from_bytes(b"canonical"),
            AlgorithmDigest.from_bytes(b"fingerprint"),
            AlgorithmDigest.from_bytes(b"review"),
            PatchContextId("context_" + "a" * 16),
            PatchWorkspaceId("workspace_" + "a" * 16),
            lease.domain_id,
            (
                DurableStepBinding(
                    PatchStepId("step_" + "a" * 16),
                    PatchLineageId("lineage_" + "a" * 16),
                ),
            ),
        )
        approval = DurableApproval(
            PatchGrantId("grant_" + "a" * 16),
            PatchApprovalId("approval_" + "a" * 16),
            identity,
            plan.canonical_digest,
            plan.plan_id,
            plan.fingerprint_digest,
            plan.review_digest,
            plan.context_id,
            plan.workspace_id,
            plan.domain_id,
            "policy-phase-ten",
            PolicyBrokerId("broker-phase-ten"),
            PolicyReviewerRole("reviewer-phase-ten"),
            (identity.principal_id,),
            ExpiryTick(100),
            b"a" * 32,
        )

        class Approvals:
            """Accept the already-validated synthetic approval record."""

            def verify(self, received: DurableApproval) -> None:
                """Require the same exact approval passed to claim_commit."""
                assert received is approval

        async def lock_domain(_cursor: object, _domain: PatchDomainId) -> None:
            """Model the domain lock held by this controlled transaction."""

        async def select_reservation(
            _cursor: object, _reservation: DurableReservation
        ) -> dict[str, object]:
            """Return the planned, unowned row required before owner CAS."""
            return {
                "lifecycle": LifecyclePhase.PLANNED.value,
                "owner_id": None,
            }

        async def unclaimed_domain(
            _cursor: object, _domain: PatchDomainId, _now: ExpiryTick
        ) -> None:
            """Model a domain whose only owner is this transaction."""

        async def advanced_fence(
            _cursor: object, _domain: PatchDomainId
        ) -> SequenceNumber:
            """Return the next fenced epoch after the held domain lock."""
            return SequenceNumber(1)

        monkeypatch.setattr(store, "_approval_verifier", Approvals())
        monkeypatch.setattr(pgsql_store_module, "_lock_domain", lock_domain)
        monkeypatch.setattr(
            pgsql_store_module,
            "_select_reservation_for_update",
            select_reservation,
        )
        monkeypatch.setattr(
            pgsql_store_module, "_plan_from_row", lambda _row: plan
        )
        monkeypatch.setattr(
            pgsql_store_module, "_validate_approval", lambda *_arguments: None
        )
        monkeypatch.setattr(
            pgsql_store_module, "_require_unclaimed_domain", unclaimed_domain
        )
        monkeypatch.setattr(
            pgsql_store_module, "_advance_domain_fence", advanced_fence
        )
        cursor = Cursor(({"grant_id": approval.grant_id.value}, None))
        with pytest.raises(DurableStoreError) as fenced_claim:
            await store.claim_commit(
                snapshot_reservation,
                plan,
                approval,
                lease.owner_id,
                ExpiryTick(10),
                DurationTicks(10),
                (),
            )
        assert fenced_claim.value.code is DurableStoreErrorCode.FENCED

    run(exercise())

    with pytest.raises(DurableStoreError) as invalid_factory:
        PgsqlDurablePatchStoreFactory(
            cast(PgsqlDurablePatchStoreSettings, object())
        )
    assert (
        invalid_factory.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    )
    settings = PgsqlDurablePatchStoreSettings(
        "postgresql://patch.invalid/avalan", pool_minimum=1, pool_maximum=2
    )

    def assert_invalid_adapter(factory: Callable[[], object]) -> None:
        """Require the factory to reject one incomplete authority adapter."""
        with pytest.raises(DurableStoreError) as invalid_adapter:
            factory()
        assert (
            invalid_adapter.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )

    assert_invalid_adapter(
        lambda: PgsqlDurablePatchStoreFactory(
            settings,
            approval_verifier=cast(DurableApprovalVerifier, object()),
        )
    )
    assert_invalid_adapter(
        lambda: PgsqlDurablePatchStoreFactory(
            settings,
            retention_authorizer=cast(DurableRetentionAuthorizer, object()),
        )
    )
    assert_invalid_adapter(
        lambda: PgsqlDurablePatchStoreFactory(
            settings,
            retention_validator=cast(
                DurableRetentionEnvelopeValidator, object()
            ),
        )
    )

    shared = PgsqlDurablePatchStore(
        type("Pool", (), {"connection": lambda self: None})()
    )
    constructed: list[object] = []

    def from_settings(
        cls: type[PgsqlDurablePatchStore],
        /,
        configured: PgsqlDurablePatchStoreSettings,
        **adapters: object,
    ) -> PgsqlDurablePatchStore:
        """Capture factory ownership without attempting a network pool."""
        assert cls is PgsqlDurablePatchStore
        assert configured is settings
        assert set(adapters) == {
            "approval_verifier",
            "retention_authorizer",
            "retention_validator",
        }
        constructed.append(adapters)
        return shared

    monkeypatch.setattr(
        PgsqlDurablePatchStore, "from_settings", classmethod(from_settings)
    )
    shared_binding = PgsqlDurablePatchStoreFactory(settings).bind()
    assert shared_binding.store is shared
    assert shared_binding.resource is shared
    assert len(constructed) == 1


def test_patch_phase_10_sandbox_commit_rejects_invalid_private_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed private bundles, wire facts, and backend selections."""
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()
    for function, path in (
        (sandbox_commit_module._implementation_digest, missing),
        (sandbox_commit_module._implementation_digest, empty),
        (sandbox_commit_module._worker_source_digest, missing),
        (sandbox_commit_module._worker_source_digest, empty),
    ):
        with pytest.raises(TargetInspectionError) as unavailable:
            function(path)
        assert unavailable.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE

    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "target").write_text("target", encoding="utf-8")
    (linked / "link").symlink_to(linked / "target")
    with pytest.raises(TargetInspectionError) as symlink:
        sandbox_commit_module._lock_implementation_tree(linked)
    assert symlink.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE

    special = tmp_path / "special"
    special.mkdir()
    mkfifo(special / "fifo")
    with pytest.raises(TargetInspectionError) as unsupported_file:
        sandbox_commit_module._lock_implementation_tree(special)
    assert (
        unsupported_file.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )

    profile_root = tmp_path / "view"
    profile_namespace = tmp_path / "namespace"
    profile_root.mkdir()
    profile_namespace.mkdir()
    profile = _runtime(profile_root, profile_namespace).profile
    unavailable_probe = cast(
        SandboxBackendProbeResult,
        SimpleNamespace(ok=False, capabilities=None),
    )
    attestation = sandbox_commit_module._RuntimeAttestation(
        "runtime", "policy", "child", "canary"
    )
    with pytest.raises(TargetInspectionError) as unavailable_probe_error:
        sandbox_commit_module._primitive_receipts(
            sandbox_commit_module.SandboxProfileReceipt("receipt"),
            unavailable_probe,
            attestation,
        )
    assert (
        unavailable_probe_error.value.code
        is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )

    assert (
        run(
            sandbox_commit_module._runtime_backend_probe(
                SandboxBackend.BUBBLEWRAP
            )
        ).backend
        is SandboxBackend.BUBBLEWRAP
    )
    original_backend = profile.execution_plan.settings.backend
    object.__setattr__(profile.execution_plan.settings, "backend", object())
    try:
        with pytest.raises(TargetInspectionError) as unknown_command_backend:
            sandbox_commit_module._runtime_child_command(
                profile, "sandbox", profile_root, ("worker",), "config"
            )
    finally:
        object.__setattr__(
            profile.execution_plan.settings, "backend", original_backend
        )
    assert (
        unknown_command_backend.value.code
        is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )
    with pytest.raises(TargetInspectionError) as unknown_policy_backend:
        sandbox_commit_module._backend_policy_digest(
            cast(SandboxBackend, object()), ()
        )
    assert (
        unknown_policy_backend.value.code
        is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )
    monkeypatch.setattr(Path, "is_dir", lambda _path: False)
    with pytest.raises(TargetInspectionError) as no_read_roots:
        sandbox_commit_module._bubblewrap_read_roots(profile, profile_root)
    assert no_read_roots.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE


def test_patch_phase_10_sandbox_commit_rejects_wire_and_join_failures(
    tmp_path: Path,
) -> None:
    """Keep child result decoding and bounded teardown fail-closed."""
    with pytest.raises(TargetInspectionError) as malformed_root:
        sandbox_commit_module._root_from_payload(
            {
                "device": "bad",
                "inode": 1,
                "mount": "m",
                "filesystem": "f",
            }
        )
    assert malformed_root.value.code is TargetErrorCode.WORKER_UNAVAILABLE

    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    command = _unissued_command(_runtime(root, namespace).profile.identity)
    malformed_report = {
        "steps": [{}],
        "artifacts": [],
        "postcondition": PostconditionState.UNKNOWN.value,
    }
    with pytest.raises(TargetInspectionError) as invalid_report:
        sandbox_commit_module._report_from_payload(command, malformed_report)
    assert invalid_report.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    step_id = PatchStepId("step_" + "d" * 16)
    lineage_id = PatchLineageId("lineage_" + "d" * 16)
    mismatched_report = {
        "steps": [
            {
                "id": step_id.value,
                "lineage": lineage_id.value,
                "state": CommitStepState.COMMITTED.value,
            }
        ],
        "artifacts": [],
        "postcondition": PostconditionState.UNKNOWN.value,
    }
    report_patches = pytest.MonkeyPatch()
    report_patches.setattr(
        sandbox_commit_module, "_steps", lambda _command: ()
    )
    report_patches.setattr(
        sandbox_commit_module, "_artifacts", lambda _command: ()
    )
    try:
        with pytest.raises(TargetInspectionError) as mismatched_report_error:
            sandbox_commit_module._report_from_payload(
                command, mismatched_report
            )
    finally:
        report_patches.undo()
    assert (
        mismatched_report_error.value.code
        is TargetErrorCode.WORKER_UNAVAILABLE
    )

    with pytest.raises(TargetInspectionError) as malformed_envelope:
        sandbox_commit_module._response_payload(
            b'{"payload": [], "mac": "x"}', b"token", {}
        )
    assert malformed_envelope.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    invalid_body: dict[str, object] = {"body": [], "error": None}
    invalid_body_raw = dumps(
        invalid_body, separators=(",", ":"), sort_keys=True
    ).encode()
    invalid_body_line = dumps(
        {
            "payload": invalid_body,
            "mac": digest(b"token", invalid_body_raw, "sha256").hex(),
        },
        separators=(",", ":"),
    ).encode()
    with pytest.raises(TargetInspectionError) as malformed_body:
        sandbox_commit_module._response_payload(
            invalid_body_line, b"token", {}
        )
    assert malformed_body.value.code is TargetErrorCode.WORKER_UNAVAILABLE

    request_id = PatchRequestId("request_" + "d" * 16)
    plan_id = PatchPlanId("plan_" + "d" * 16)
    assert (
        sandbox_commit_module._approval_result(
            request_id, plan_id, ApprovalDecisionState.UNAVAILABLE
        ).status
        is PatchStatus.APPROVAL_UNAVAILABLE
    )
    assert (
        sandbox_commit_module._approval_result(
            request_id, plan_id, ApprovalDecisionState.DENIED
        ).status
        is PatchStatus.APPROVAL_DENIED
    )
    staged = SettlementJournal(
        (JournalStep(step_id, lineage_id, CommitStepState.COMMITTED),),
        (ArtifactJournal("artifact", ArtifactState.STAGED),),
        PostconditionState.UNKNOWN,
    )
    assert (
        sandbox_commit_module._artifact_state(staged) is ArtifactState.UNKNOWN
    )

    async def exercise_join_timeout() -> None:
        """Simulate a process teardown deadline and cancellation race."""
        gate = Event()
        task = create_task(gate.wait())
        calls = 0

        async def timeout_then_cancel(
            _awaitable: object, *, timeout: object
        ) -> object:
            """Model the two bounded waits without a wall-clock delay."""
            del timeout
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError
            raise CancelledError

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            sandbox_commit_module, "wait_for", timeout_then_cancel
        )
        try:
            await sandbox_commit_module._bounded_task_join(task)
        finally:
            monkeypatch.undo()
        await sleep(0)
        assert task.cancelled()

    run(exercise_join_timeout())


def test_patch_phase_10_toolset_private_authority_guards_are_fail_closed() -> (
    None
):
    """Reject direct closure extraction before it can mint patch authority."""
    loader_closure = getclosurevars(PatchToolLoader.load).nonlocals
    toolset_closure = getclosurevars(PatchToolSet.__init__).nonlocals
    for function, arguments in (
        (loader_closure["reserve"], (object(), object())),
        (loader_closure["discard"], (object(),)),
        (toolset_closure["claim"], (object(), object(), object())),
        (toolset_closure["bind_capability"], (object(), object(), object())),
        (toolset_closure["revoke"], (object(), object())),
        (toolset_closure["register"], (object(), object())),
    ):
        with pytest.raises(PatchToolError):
            function(*arguments)
    assert (
        patch_toolset_module._bound_invocation_subscription_access(
            object(), object()
        )
        is None
    )

    async def exercise() -> None:
        """Reject forged loaders and unavailable SDK operations at ingress."""
        with pytest.raises(PatchToolError, match="loader is invalid"):
            await PatchToolLoader.load(
                cast(PatchToolLoader, object()), enable_tools=None
            )
        unavailable_host = cast(
            PatchSdkHost,
            SimpleNamespace(
                _snapshot=SimpleNamespace(permits=lambda _operation: False),
                _is_active=lambda: True,
            ),
        )
        with pytest.raises(PatchToolError, match="operation is unavailable"):
            await PatchSdkHost._invoke_raw_with_identity(
                unavailable_host,
                OperationType.EDIT,
                b"{}",
                PatchRequestId("request_" + "e" * 16),
                PatchObserverCorrelationId("correlation_" + "e" * 16),
            )

    run(exercise())


def test_patch_phase_10_toolset_requires_sandbox_endpoint_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a selected sandbox service that lacks the sealed endpoint."""
    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    identity = _runtime(root, namespace).profile.identity
    primitives = frozenset(TargetPrimitive)
    scope = ResolvedMutationScope(
        ContextKind.SANDBOX,
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
    settlement = SimpleNamespace(
        inspect=lambda _handle: None,
        await_terminal=lambda _handle, _pending: None,
    )

    def no_events(_handle: object) -> AsyncIterator[object]:
        """Return a closed event stream; construction must fail before use."""

        async def empty() -> AsyncIterator[object]:
            """Yield no events from the deliberately incomplete host."""
            if False:
                yield object()

        return empty()

    service = cast(
        PatchSdkService,
        SimpleNamespace(
            settlement=settlement,
            invoke=lambda *_arguments: None,
            review=lambda _handle: None,
            approve=lambda _handle: None,
            subscribe=no_events,
        ),
    )
    binding = PatchRuntimeBinding(
        scope,
        handshake,
        _runtime_policy(),
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True),
        PatchPersistenceBinding(True),
        service,
    )
    binder = object.__new__(SandboxPatchRuntimeBinder)

    async def bind(_self: SandboxPatchRuntimeBinder) -> PatchRuntimeBinding:
        """Return the selected sandbox binding with no endpoint capability."""
        return binding

    monkeypatch.setattr(SandboxPatchRuntimeBinder, "bind", bind)

    async def exercise() -> None:
        """Enter the real loader and fail only at endpoint issuance."""

        class WrongBinder:
            """Return a sandbox binding without the selected runtime type."""

            async def bind(self) -> PatchRuntimeBinding:
                """Expose the otherwise complete synthetic sandbox binding."""
                return binding

        wrong_loader = PatchToolLoader(
            WrongBinder(),
            PatchTestHostProfile(enabled=True, authenticated=True),
        )
        with pytest.raises(PatchToolError, match="selected runtime"):
            await wrong_loader.load(enable_tools=["patch.edit"])
        loader = PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        )
        with pytest.raises(
            PatchToolError, match="sandbox endpoint is unavailable"
        ):
            await loader.load(enable_tools=["patch.edit"])

    run(exercise())


def test_patch_phase_10_runtime_process_rejects_reuse_and_startup_faults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject stale runtime reuse, backend mismatch, and malformed canaries."""
    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)
    owner = runtime._process

    async def exercise() -> None:
        """Exercise process-local failure branches without a native child."""
        owner._closed = True
        with pytest.raises(TargetInspectionError) as closed:
            await owner.start()
        assert closed.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        owner._closed = False

        witness = rooted_worker_module.RootWitness(
            FileIdentity(1, 2),
            runtime.profile.identity.mount_id,
            runtime.profile.identity.filesystem_id,
        )

        async def cached_witness(
            _owner: sandbox_commit_module._SandboxRuntimeProcess,
        ) -> rooted_worker_module.RootWitness:
            """Return the exact cached witness without inspecting a child."""
            return witness

        owner._process = cast(Process, SimpleNamespace(returncode=None))
        owner._receipt = sandbox_commit_module.SandboxProfileReceipt("receipt")
        owner._token = b"x" * 32
        owner._primitive_receipts = {}
        owner._attestation = sandbox_commit_module._RuntimeAttestation(
            "runtime", "policy", "child", "canary"
        )
        monkeypatch.setattr(
            sandbox_commit_module._SandboxRuntimeProcess,
            "_witness_locked",
            cached_witness,
        )
        cached = await owner.start()
        assert cached[0] == witness
        owner._process = None

        original_backend = runtime.profile.execution_plan.settings.backend
        object.__setattr__(
            runtime.profile.execution_plan.settings, "backend", object()
        )
        try:
            with pytest.raises(TargetInspectionError) as invalid_backend:
                await owner.start()
        finally:
            object.__setattr__(
                runtime.profile.execution_plan.settings,
                "backend",
                original_backend,
            )
        assert (
            invalid_backend.value.code
            is TargetErrorCode.CAPABILITY_UNAVAILABLE
        )

        unavailable = cast(
            SandboxBackendProbeResult,
            SimpleNamespace(ok=False, capabilities=None),
        )

        async def unavailable_probe(
            _backend: SandboxBackend,
        ) -> SandboxBackendProbeResult:
            """Report a selected backend with no usable capability proof."""
            return unavailable

        monkeypatch.setattr(
            sandbox_commit_module, "_runtime_backend_probe", unavailable_probe
        )
        with pytest.raises(TargetInspectionError) as unsupported_backend:
            await owner.start()
        assert (
            unsupported_backend.value.code
            is TargetErrorCode.CAPABILITY_UNAVAILABLE
        )

    run(exercise())


def test_patch_phase_10_runtime_and_endpoint_reject_stale_or_reused_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fence stale runtime identity and retain endpoint settlement truth."""
    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)

    async def closed_process(
        _process: sandbox_commit_module._SandboxRuntimeProcess,
    ) -> None:
        """Model a reaped child without invoking a native backend."""

    monkeypatch.setattr(
        sandbox_commit_module._SandboxRuntimeProcess, "close", closed_process
    )

    async def exercise() -> None:
        """Exercise runtime guards and endpoint reconciliation directly."""
        assert await runtime.__aenter__() is runtime
        await runtime.__aexit__(None, None, None)
        with pytest.raises(TargetInspectionError) as local_selection:
            await runtime.resolve(ScopeSelection(ContextKind.LOCAL))
        assert (
            local_selection.value.code
            is TargetErrorCode.CAPABILITY_UNAVAILABLE
        )

        original_profile = runtime._profile_guard
        runtime._profile_guard = replace(
            original_profile,
            channel_id=SandboxChannelId("different-channel"),
        )
        try:
            with pytest.raises(TargetInspectionError) as changed_profile:
                await runtime.resolve(ScopeSelection(ContextKind.SANDBOX))
        finally:
            runtime._profile_guard = original_profile
        assert changed_profile.value.code is TargetErrorCode.WITNESS_STALE

        mismatched = rooted_worker_module.RootWitness(
            FileIdentity(1, 2), "other-mount", "other-filesystem"
        )
        receipt = sandbox_commit_module.SandboxProfileReceipt("receipt")
        session = SandboxSessionId("session-" + "a" * 16)
        attestation = sandbox_commit_module._RuntimeAttestation(
            "runtime", "policy", "child", "canary"
        )

        async def mismatched_start(
            _process: sandbox_commit_module._SandboxRuntimeProcess,
        ) -> tuple[
            rooted_worker_module.RootWitness,
            sandbox_commit_module.SandboxProfileReceipt,
            SandboxSessionId,
            Mapping[TargetPrimitive, str],
            sandbox_commit_module._RuntimeAttestation,
        ]:
            """Return a child witness from a deliberately different mount."""
            return mismatched, receipt, session, {}, attestation

        monkeypatch.setattr(
            sandbox_commit_module._SandboxRuntimeProcess,
            "start",
            mismatched_start,
        )
        with pytest.raises(TargetInspectionError) as denied_mount:
            await runtime.resolve(ScopeSelection(ContextKind.SANDBOX))
        assert denied_mount.value.code is TargetErrorCode.MOUNT_DENIED

        scope = cast(ResolvedMutationScope, object())

        async def valid_scope(
            _runtime: SandboxPatchRuntime,
            expected: ResolvedMutationScope,
        ) -> object:
            """Accept only the endpoint's test-owned scope identity."""
            assert expected is scope
            return object()

        monkeypatch.setattr(SandboxPatchRuntime, "_require_scope", valid_scope)
        endpoint = sandbox_commit_module._SandboxEndpoint(runtime, scope)
        request_id = PatchRequestId("request_" + "b" * 16)
        settled = WorkerReport(WorkerState.LIVE, None)
        endpoint._settlements[request_id] = settled
        assert await endpoint.reconcile_sandbox(request_id) is settled
        other_request = PatchRequestId("request_" + "c" * 16)
        assert (
            await endpoint.reconcile_sandbox(other_request)
        ).state is WorkerState.LIVE

        delayed = create_task(Event().wait())
        endpoint._active_request = other_request
        endpoint._active_task = cast(Task[WorkerReport], delayed)
        assert (
            await endpoint.reconcile_sandbox(other_request)
        ).state is WorkerState.LIVE
        delayed.cancel()
        with pytest.raises(CancelledError):
            await delayed
        assert (
            await endpoint.reconcile_sandbox(other_request)
        ).state is WorkerState.LIVE

        async def raise_worker() -> WorkerReport:
            """Expose a failed worker task to the reconciliation path."""
            raise RuntimeError("worker lost")

        failed = create_task(raise_worker())
        await sleep(0)
        endpoint._active_request = other_request
        endpoint._active_task = failed
        assert (
            await endpoint.reconcile_sandbox(other_request)
        ).state is WorkerState.LIVE

    run(exercise())


def test_patch_phase_10_sandbox_service_reaps_failures_and_fences_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve private cleanup and helper fail-closed behavior."""
    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)
    service = object.__new__(SandboxPatchSdkService)
    request_id = PatchRequestId("request_" + "f" * 16)
    plan_id = PatchPlanId("plan_" + "f" * 16)
    result = sandbox_commit_module._approval_result(
        request_id, plan_id, ApprovalDecisionState.DENIED
    )

    class FailingRuntime:
        """Make teardown retain its first runtime-close failure."""

        async def close(self) -> None:
            """Fail after the service begins orderly worker cleanup."""
            raise RuntimeError("runtime close failed")

    service.runtime = cast(SandboxPatchRuntime, FailingRuntime())
    service._worker_tasks = {}
    service._workers = {}
    service._reconciliation_tasks = set()
    service._reader_tasks = set()
    service._latest = result

    async def exercise() -> None:
        """Check service-owned failure cleanup and helper accessors."""
        assert await service.review(cast(PatchInvocationHandle, object())) == {
            "kind": "sandbox_patch_review"
        }
        assert (
            await service.approve(cast(PatchInvocationHandle, object()))
            is result
        )
        with pytest.raises(RuntimeError, match="runtime close failed"):
            await service.__aexit__(None, None, None)

        service.runtime = runtime
        service.scope = cast(ResolvedMutationScope, object())
        runtime._process._implementation_digest_value = None

        async def live_receipt(
            _runtime: SandboxPatchRuntime,
            _scope: ResolvedMutationScope,
        ) -> object:
            """Return the sole receipt fact needed before digest validation."""
            return SimpleNamespace(
                root=rooted_worker_module.RootWitness(
                    FileIdentity(1, 2),
                    runtime.profile.identity.mount_id,
                    runtime.profile.identity.filesystem_id,
                ),
                session_id=SandboxSessionId("session-" + "f" * 16),
            )

        monkeypatch.setattr(
            SandboxPatchRuntime, "_require_scope", live_receipt
        )
        with pytest.raises(TargetInspectionError) as unavailable_digest:
            await service._worker_binding()
        assert (
            unavailable_digest.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        )

    run(exercise())

    with pytest.raises(TargetInspectionError) as invalid_binder:
        SandboxPatchRuntimeBinder(
            cast(SandboxPatchRuntime, object()),
            cast(SandboxPatchServiceFactory, object()),
            cast(TrustedPatchPolicy, object()),
            PatchApprovalBinding(True),
            PatchCoordinatorBinding(True),
            PatchPersistenceBinding(True),
        )
    assert invalid_binder.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
    with pytest.raises(TargetInspectionError) as invalid_settings:
        SandboxPatchRuntimeBinder.from_settings(
            cast(SandboxPatchRuntimeSettings, object()),
            cast(SandboxPatchServiceConfiguration, object()),
            cast(TrustedPatchPolicy, object()),
            PatchApprovalBinding(True),
            PatchCoordinatorBinding(True),
            PatchPersistenceBinding(True),
        )
    assert (
        invalid_settings.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )
    with pytest.raises(TargetInspectionError) as invalid_shared:
        SandboxPatchRuntimeBinder.from_shared_store(
            cast(SandboxPatchRuntimeSettings, object()),
            cast(SandboxPatchServiceConfiguration, object()),
            cast(TrustedPatchPolicy, object()),
            PatchApprovalBinding(True),
            cast(DurablePatchStoreBinding, object()),
        )
    assert invalid_shared.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE


def test_patch_phase_10_endpoint_serializes_and_fences_worker_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain completed reports while fencing wrong endpoint work."""
    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)
    identity = runtime.profile.identity
    root_witness = rooted_worker_module.RootWitness(
        FileIdentity(1, 2), identity.mount_id, identity.filesystem_id
    )
    worker_witness = EphemeralWorkerWitness(
        runtime.profile.channel_id, "worker-phase-ten", "fence-phase-ten"
    )
    scope = ResolvedMutationScope(
        ContextKind.SANDBOX,
        identity,
        None,
        _limits(),
        frozenset(Capability),
        frozenset(TargetPrimitive),
        root_witness,
        worker_witness,
    )
    request_id = PatchRequestId("request_" + "e" * 16)
    command = _unissued_command(identity, 1, None, request_id)
    endpoint = sandbox_commit_module._SandboxEndpoint(runtime, scope)
    report = WorkerReport(WorkerState.LIVE, None)

    async def accepted_scope(
        _runtime: SandboxPatchRuntime,
        expected: ResolvedMutationScope,
    ) -> object:
        """Accept only the endpoint's immutable scope object."""
        assert expected is scope
        return object()

    async def committed(
        _process: sandbox_commit_module._SandboxRuntimeProcess,
        received: SealedCommitCommand,
        _validator: RootedCommandAuthorityValidator,
    ) -> WorkerReport:
        """Return a fixed worker observation for the exact sealed command."""
        assert received is command
        return report

    monkeypatch.setattr(SandboxPatchRuntime, "_require_scope", accepted_scope)
    monkeypatch.setattr(
        sandbox_commit_module._SandboxRuntimeProcess, "commit", committed
    )

    async def exercise() -> None:
        """Exercise endpoint mismatch, serialization, and memoization."""
        wrong = cast(
            SealedCommitCommand,
            SimpleNamespace(
                plan=SimpleNamespace(
                    binding=SimpleNamespace(
                        context_kind=ContextKind.LOCAL,
                        target=identity,
                        cwd=None,
                    )
                ),
                lease=SimpleNamespace(request_id=request_id),
            ),
        )
        with pytest.raises(TargetInspectionError) as stale_endpoint:
            await endpoint.commit_sandbox(
                wrong, cast(RootedCommandAuthorityValidator, object())
            )
        assert stale_endpoint.value.code is TargetErrorCode.WITNESS_STALE

        blocked = create_task(Event().wait())
        endpoint._active_task = cast(Task[WorkerReport], blocked)
        with pytest.raises(TargetInspectionError) as active_worker:
            await endpoint.commit_sandbox(
                command, cast(RootedCommandAuthorityValidator, object())
            )
        assert active_worker.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        blocked.cancel()
        with pytest.raises(CancelledError):
            await blocked
        endpoint._active_task = None
        assert (
            await endpoint.commit_sandbox(
                command, cast(RootedCommandAuthorityValidator, object())
            )
            is report
        )
        assert (
            await endpoint.commit_sandbox(
                command, cast(RootedCommandAuthorityValidator, object())
            )
            is report
        )

    run(exercise())

    lease = DurableCommitLease(
        request_id,
        identity.domain_id,
        PatchCommitOwnerId("owner_" + "e" * 16),
        SequenceNumber(1),
        ExpiryTick(20),
    )
    authority = sandbox_commit_module._SandboxDurableCommandAuthority(
        runtime,
        scope,
        lease,
        cast(InMemoryDurablePatchStore, object()),
        cast(ApprovalClock, object()),
    )

    async def incomplete_receipt(
        _runtime: SandboxPatchRuntime,
        _scope: ResolvedMutationScope,
    ) -> object:
        """Return a receipt whose empty session cannot authorize a command."""
        return SimpleNamespace(
            root=root_witness, worker=worker_witness, session_id=""
        )

    authority_patch = pytest.MonkeyPatch()
    authority_patch.setattr(
        SandboxPatchRuntime, "_require_scope", incomplete_receipt
    )
    try:
        assert not run(authority.is_rooted_command_current(command))
    finally:
        authority_patch.undo()


def test_patch_phase_10_subscription_rejects_unissued_or_incoherent_outbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fence subscription handles and outbox rows before exposing events."""
    service = object.__new__(SandboxPatchSdkService)
    request_id = PatchRequestId("request_" + "9" * 16)
    correlation = PatchObserverCorrelationId("correlation_" + "9" * 16)
    handle = cast(PatchInvocationHandle, object())

    async def no_records(
        _access: DurableRequestAccess,
        _cursor: SequenceNumber,
        _limit: int,
    ) -> tuple[object, ...]:
        """Return no outbox records from the controlled durable store."""
        return ()

    async def no_pending(_access: DurableRequestAccess) -> object:
        """Return a nonterminal snapshot that lacks a resumable pending row."""
        return SimpleNamespace(terminal=None, pending=None)

    service.store = SimpleNamespace(outbox=no_records, inspect=no_pending)
    service._requests = {}
    bound_patch = pytest.MonkeyPatch()
    try:
        bound_patch.setattr(
            sandbox_commit_module,
            "_bound_invocation_subscription_access",
            lambda _handle, _service: None,
        )

        async def exercise() -> None:
            """Require valid authority, rows, and pending state."""
            with pytest.raises(TargetInspectionError) as unissued:
                await anext(service.subscribe(handle))
            assert unissued.value.code is TargetErrorCode.WITNESS_STALE

            bound_patch.setattr(
                sandbox_commit_module,
                "_bound_invocation_subscription_access",
                lambda _handle, _service: (request_id, correlation),
            )
            with pytest.raises(TargetInspectionError) as unknown_request:
                await anext(service.subscribe(handle))
            assert unknown_request.value.code is TargetErrorCode.WITNESS_STALE

            identity = DurableRequestIdentity(
                PatchTenantId("tenant-subscription"),
                PatchPrincipalId("principal-subscription"),
                PatchExecutionId("execution_" + "9" * 16),
                PolicyRouteId("route-subscription"),
                RetransmissionKey("retry-subscription"),
            )
            service._requests[request_id] = (
                sandbox_commit_module._SandboxRequestAccess(
                    DurableRequestAccess(request_id, identity), correlation
                )
            )
            with pytest.raises(TargetInspectionError) as missing_pending:
                await anext(service.subscribe(handle))
            assert (
                missing_pending.value.code
                is TargetErrorCode.WORKER_UNAVAILABLE
            )

            async def mismatched_record(
                _access: DurableRequestAccess,
                _cursor: SequenceNumber,
                _limit: int,
            ) -> tuple[object, ...]:
                """Return one outbox row belonging to a different request."""
                return (
                    SimpleNamespace(
                        request_id=PatchRequestId("request_" + "a" * 16),
                        correlation_id=correlation,
                        sequence=SequenceNumber(1),
                    ),
                )

            service.store = SimpleNamespace(
                outbox=mismatched_record,
                inspect=no_pending,
            )
            with pytest.raises(TargetInspectionError) as mismatched_outbox:
                await anext(service.subscribe(handle))
            assert (
                mismatched_outbox.value.code is TargetErrorCode.WITNESS_STALE
            )

            inspections = iter(
                (
                    SimpleNamespace(
                        terminal=SimpleNamespace(
                            outbox=SimpleNamespace(sequence=SequenceNumber(1))
                        )
                    ),
                    SimpleNamespace(
                        terminal=SimpleNamespace(
                            outbox=SimpleNamespace(sequence=SequenceNumber(0))
                        )
                    ),
                )
            )

            async def terminal_after_retry(
                _access: DurableRequestAccess,
            ) -> object:
                """Require an extra poll before accepting terminal truth."""
                return next(inspections)

            service.store = SimpleNamespace(
                outbox=no_records,
                inspect=terminal_after_retry,
            )
            with pytest.raises(StopAsyncIteration):
                await anext(service.subscribe(handle))

            pending = SimpleNamespace(
                request_id=request_id,
                pending_operation_id=PatchPendingOperationId.new(),
                correlation_id=correlation,
            )

            async def pending_snapshot(
                _access: DurableRequestAccess,
            ) -> object:
                """Expose one coherent durable pending continuation."""
                return SimpleNamespace(terminal=None, pending=pending)

            async def interrupted_wait(_access: object) -> object:
                """Model loss while awaiting the exact durable continuation."""
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)

            service.store = SimpleNamespace(
                outbox=no_records,
                inspect=pending_snapshot,
                await_terminal=interrupted_wait,
            )
            with pytest.raises(TargetInspectionError) as interrupted_pending:
                await anext(service.subscribe(handle))
            assert (
                interrupted_pending.value.code
                is TargetErrorCode.WORKER_UNAVAILABLE
            )

        run(exercise())
    finally:
        bound_patch.undo()


def test_patch_phase_10_attached_outcome_durable_truth() -> None:
    """Reject stale attachment state before worker recovery can begin."""
    service = object.__new__(SandboxPatchSdkService)
    request_id = PatchRequestId("request_" + "8" * 16)
    correlation = PatchObserverCorrelationId("correlation_" + "8" * 16)
    identity = DurableRequestIdentity(
        PatchTenantId("tenant-attached"),
        PatchPrincipalId("principal-attached"),
        PatchExecutionId("execution_" + "8" * 16),
        PolicyRouteId("route-attached"),
        RetransmissionKey("retry-attached"),
    )
    result = sandbox_commit_module._approval_result(
        request_id,
        PatchPlanId("plan_" + "8" * 16),
        ApprovalDecisionState.DENIED,
    )
    snapshots = iter(
        (
            SimpleNamespace(terminal=SimpleNamespace(result=result)),
            SimpleNamespace(
                terminal=None, pending=SimpleNamespace(correlation_id=object())
            ),
            SimpleNamespace(
                terminal=None,
                pending=None,
                worker_bound=False,
                worker_reaped=False,
                lease=None,
                plan=None,
            ),
            SimpleNamespace(
                terminal=None,
                pending=None,
                worker_bound=True,
                worker_reaped=False,
            ),
        )
    )

    async def inspect(_access: DurableRequestAccess) -> object:
        """Return one controlled stored state for each attachment attempt."""
        return next(snapshots)

    service.store = SimpleNamespace(inspect=inspect)
    service._pending = {}

    async def exercise() -> None:
        """Read terminal truth and fence incoherent nonterminal branches."""
        assert (
            await service._attached_outcome(request_id, identity, correlation)
            is result
        )
        with pytest.raises(TargetInspectionError) as stale_correlation:
            await service._attached_outcome(request_id, identity, correlation)
        assert stale_correlation.value.code is TargetErrorCode.WITNESS_STALE
        with pytest.raises(TargetInspectionError) as missing_recovery:
            await service._attached_outcome(request_id, identity, correlation)
        assert (
            missing_recovery.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        )
        with pytest.raises(TargetInspectionError) as missing_pending:
            await service._attached_outcome(request_id, identity, correlation)
        assert missing_pending.value.code is TargetErrorCode.WORKER_UNAVAILABLE

    run(exercise())


def test_patch_phase_10_private_bundle_rejects_overlap_and_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove a private worker bundle whenever copying loses its boundary."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        sandbox_commit_module, "mkdtemp", lambda **_kwargs: str(workspace)
    )
    with pytest.raises(TargetInspectionError) as overlapping_root:
        sandbox_commit_module._ImplementationBundle.create(workspace)
    assert (
        overlapping_root.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )
    assert not workspace.exists()

    def copied_with_forbidden_worker(
        _source: Path,
        destination: Path,
        **_kwargs: object,
    ) -> Path:
        """Model a copied bundle that accidentally retains host commit code."""
        destination.mkdir(parents=True, exist_ok=True)
        patch = destination / "patch"
        patch.mkdir(exist_ok=True)
        (patch / "sandbox_commit.py").write_text("forbidden", encoding="utf-8")
        return destination

    forbidden_root = tmp_path / "forbidden"
    monkeypatch.setattr(
        sandbox_commit_module, "mkdtemp", lambda **_kwargs: str(forbidden_root)
    )
    monkeypatch.setattr(
        sandbox_commit_module, "copytree", copied_with_forbidden_worker
    )
    with pytest.raises(TargetInspectionError) as retained_commit_source:
        sandbox_commit_module._ImplementationBundle.create(tmp_path / "other")
    assert (
        retained_commit_source.value.code
        is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )
    assert not forbidden_root.exists()

    def copied_minimal_bundle(
        _source: Path,
        destination: Path,
        **_kwargs: object,
    ) -> Path:
        """Create regular files sufficient for the final source-hash check."""
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "module.py").write_text("safe", encoding="utf-8")
        return destination

    digest_root = tmp_path / "digest"
    source_calls = 0

    def inconsistent_source_digest(_source: Path) -> str:
        """Model source changing after the private bundle was copied."""
        nonlocal source_calls
        source_calls += 1
        return (
            sandbox_commit_module._PINNED_WORKER_SOURCE_DIGEST
            if source_calls == 1
            else "changed-source"
        )

    monkeypatch.setattr(
        sandbox_commit_module, "mkdtemp", lambda **_kwargs: str(digest_root)
    )
    monkeypatch.setattr(
        sandbox_commit_module, "copytree", copied_minimal_bundle
    )
    monkeypatch.setattr(
        sandbox_commit_module,
        "_worker_source_digest",
        inconsistent_source_digest,
    )
    with pytest.raises(TargetInspectionError) as changed_source:
        sandbox_commit_module._ImplementationBundle.create(tmp_path / "third")
    assert changed_source.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE


def test_patch_phase_10_runtime_values_and_adapter_calls_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed runtime records and delegate scope calls."""
    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)
    profile = runtime.profile
    with pytest.raises(TargetInspectionError) as invalid_profile:
        replace(profile, channel_id=SandboxChannelId(""))
    assert invalid_profile.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE

    context = _settings(root, namespace).context
    with pytest.raises(TargetInspectionError) as invalid_context:
        replace(
            context, implementation_id=SandboxWorkerImplementationId("other")
        )
    assert invalid_context.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE

    root_witness = rooted_worker_module.RootWitness(
        FileIdentity(1, 2),
        profile.identity.mount_id,
        profile.identity.filesystem_id,
    )
    worker = EphemeralWorkerWitness(
        profile.channel_id, "worker-phase-ten", "fence-phase-ten"
    )
    primitives = {item: "receipt" for item in TargetPrimitive}
    with pytest.raises(TargetInspectionError) as invalid_receipt:
        sandbox_commit_module.SandboxRuntimeReceipt(
            SandboxSessionId(""),
            sandbox_commit_module.SandboxProfileReceipt("receipt"),
            root_witness,
            worker,
            primitives,
            "runtime",
            "policy",
            "child",
            "canary",
        )
    assert invalid_receipt.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE

    scope = cast(ResolvedMutationScope, object())
    handshake = cast(TargetHandshake, object())
    worker_result = cast(RootedSandboxCommitWorker, object())

    async def resolved(
        _runtime: SandboxPatchRuntime, selected: ScopeSelection
    ) -> ResolvedMutationScope:
        """Return one exact scope through the resolver adapter."""
        assert selected.context_kind is ContextKind.SANDBOX
        return scope

    async def handshaken(
        _runtime: SandboxPatchRuntime, selected: ResolvedMutationScope
    ) -> TargetHandshake:
        """Return one exact handshake through the inspection adapter."""
        assert selected is scope
        return handshake

    async def worked(
        _runtime: SandboxPatchRuntime, selected: ResolvedMutationScope
    ) -> RootedSandboxCommitWorker:
        """Return one exact worker through the commit adapter."""
        assert selected is scope
        return worker_result

    monkeypatch.setattr(SandboxPatchRuntime, "resolve", resolved)
    monkeypatch.setattr(SandboxPatchRuntime, "handshake", handshaken)
    monkeypatch.setattr(SandboxPatchRuntime, "worker", worked)

    async def exercise() -> None:
        """Use the public adapters without widening their capabilities."""
        resolver = sandbox_commit_module.SandboxScopeResolver(runtime)
        inspection = SandboxInspectionTarget(runtime)
        target = SandboxCommitTarget(runtime)
        assert (
            await resolver.resolve(ScopeSelection(ContextKind.SANDBOX))
            is scope
        )
        assert await inspection.handshake(scope) is handshake
        assert await target.worker(scope) is worker_result

    run(exercise())


def test_patch_phase_10_runtime_process_fences_bad_child_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reap a child that cannot carry an authenticated complete response."""
    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    owner = _runtime(root, namespace)._process
    owner._token = b"x" * 32
    owner._receipt = sandbox_commit_module.SandboxProfileReceipt("receipt")
    native_wait_for = wait_for
    reaped: list[str] = []

    async def reap(_owner: object) -> None:
        """Record each failed exchange without creating a process."""
        reaped.append("reaped")

    monkeypatch.setattr(
        sandbox_commit_module._SandboxRuntimeProcess, "_reap", reap
    )

    class Input:
        """Record message writes and complete host-side drains."""

        def __init__(self) -> None:
            self.writes: list[bytes] = []
            self.drains = 0
            self.fail_on_drain: int | None = None

        def write(self, value: bytes) -> None:
            """Retain one runtime wire message."""
            self.writes.append(value)

        async def drain(self) -> None:
            """Finish one local pipe flush."""
            self.drains += 1
            if self.drains == self.fail_on_drain:
                raise TimeoutError

    class Output:
        """Return one signed response generated from the received request."""

        def __init__(
            self, input_stream: Input, body: Mapping[str, object]
        ) -> None:
            self.input_stream = input_stream
            self.body = body

        async def readline(self) -> bytes:
            """Echo the request binding with the configured child body."""
            request = loads(self.input_stream.writes[-1])["payload"]
            assert isinstance(request, dict)
            payload = {
                field: request[field]
                for field in (
                    "version",
                    "sequence",
                    "receipt",
                    "identity",
                    "channel_id",
                    "implementation_id",
                )
            }
            payload.update(body=self.body, error=None)
            raw = dumps(
                payload, separators=(",", ":"), sort_keys=True
            ).encode()
            return (
                dumps(
                    {
                        "payload": payload,
                        "mac": (
                            digest(owner._token or b"", raw, "sha256").hex()
                        ),
                    },
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            )

    async def exercise() -> None:
        """Fence absent streams, oversize requests, and invalid controls."""
        owner._process = cast(
            Process, SimpleNamespace(returncode=None, stdin=None, stdout=None)
        )
        with pytest.raises(TargetInspectionError) as absent_streams:
            await owner._request_locked("witness", {})
        assert absent_streams.value.code is TargetErrorCode.WORKER_UNAVAILABLE

        input_stream = Input()
        output_stream = Output(input_stream, {"result": "ok"})
        owner._process = cast(
            Process,
            SimpleNamespace(
                returncode=None, stdin=input_stream, stdout=output_stream
            ),
        )
        with pytest.raises(TargetInspectionError) as oversize:
            await owner._request_locked(
                "inspect", {"payload": "x" * 1_100_000}
            )
        assert oversize.value.code is TargetErrorCode.LIMIT_EXCEEDED

        output_stream.body = {"control": "invalid", "effect": 1}
        with pytest.raises(TargetInspectionError) as invalid_control:
            await owner._request_locked("commit", {})
        assert invalid_control.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        assert reaped == ["reaped"]

        owner._process = cast(
            Process,
            SimpleNamespace(
                returncode=None, stdin=input_stream, stdout=output_stream
            ),
        )
        with pytest.raises(TargetInspectionError) as missing_commit_facts:
            await owner.commit(
                cast(SealedCommitCommand, object()),
                cast(RootedCommandAuthorityValidator, object()),
            )
        assert (
            missing_commit_facts.value.code
            is TargetErrorCode.WORKER_UNAVAILABLE
        )

        async def malformed_inspection(
            _owner: sandbox_commit_module._SandboxRuntimeProcess,
            _kind: str,
            _body: Mapping[str, object],
        ) -> Mapping[str, object]:
            """Return a worker body without the required snapshots list."""
            return {"snapshots": object()}

        monkeypatch.setattr(
            sandbox_commit_module._SandboxRuntimeProcess,
            "_request",
            malformed_inspection,
        )
        with pytest.raises(TargetInspectionError) as malformed_snapshots:
            await owner.inspect((), root_witness)
        assert (
            malformed_snapshots.value.code
            is TargetErrorCode.WORKER_UNAVAILABLE
        )

        async def timeout_after_flush(
            awaitable: Awaitable[object], *, timeout: float
        ) -> object:
            """Complete the local awaitable, then model an I/O deadline."""
            del timeout
            await awaitable
            raise TimeoutError

        monkeypatch.setattr(
            sandbox_commit_module, "wait_for", timeout_after_flush
        )
        owner._process = cast(
            Process,
            SimpleNamespace(
                returncode=None, stdin=input_stream, stdout=output_stream
            ),
        )
        with pytest.raises(TargetInspectionError) as stalled_drain:
            await owner._request_locked("witness", {})
        assert stalled_drain.value.code is TargetErrorCode.WORKER_UNAVAILABLE

        calls = 0

        async def timeout_read_only(
            awaitable: Awaitable[object], *, timeout: float
        ) -> object:
            """Permit the write then model a child that never replies."""
            del timeout
            nonlocal calls
            calls += 1
            await awaitable
            if calls == 2:
                raise TimeoutError
            return None

        monkeypatch.setattr(
            sandbox_commit_module, "wait_for", timeout_read_only
        )
        owner._process = cast(
            Process,
            SimpleNamespace(
                returncode=None, stdin=input_stream, stdout=output_stream
            ),
        )
        with pytest.raises(TargetInspectionError) as stalled_read:
            await owner._request_locked("witness", {})
        assert stalled_read.value.code is TargetErrorCode.WORKER_UNAVAILABLE

        output_stream.body = {"control": "fence", "effect": 1}
        calls = 0

        async def timeout_permit(
            awaitable: Awaitable[object], *, timeout: float
        ) -> object:
            """Fail the durable permit flush after one fence prompt."""
            del timeout
            nonlocal calls
            calls += 1
            await awaitable
            if calls == 3:
                raise TimeoutError
            return None

        class CurrentAuthority:
            """Authorize the one generated effect number for the fake child."""

            async def is_rooted_command_current(
                self, _command: SealedCommitCommand
            ) -> bool:
                """Confirm the synthetic command remains owned."""
                return True

        monkeypatch.setattr(sandbox_commit_module, "wait_for", timeout_permit)
        owner._process = cast(
            Process,
            SimpleNamespace(
                returncode=None, stdin=input_stream, stdout=output_stream
            ),
        )
        with pytest.raises(TargetInspectionError) as stalled_permit:
            await owner._request_locked(
                "commit",
                {},
                command=cast(SealedCommitCommand, object()),
                validator=cast(
                    RootedCommandAuthorityValidator, CurrentAuthority()
                ),
            )
        assert stalled_permit.value.code is TargetErrorCode.WORKER_UNAVAILABLE

        permit_input = Input()
        permit_input.fail_on_drain = 2
        permit_output = Output(permit_input, {"control": "fence", "effect": 1})
        monkeypatch.setattr(sandbox_commit_module, "wait_for", native_wait_for)
        owner._process = cast(
            Process,
            SimpleNamespace(
                returncode=None, stdin=permit_input, stdout=permit_output
            ),
        )
        with pytest.raises(TargetInspectionError) as delayed_permit:
            await owner._request_locked(
                "commit",
                {},
                command=cast(SealedCommitCommand, object()),
                validator=cast(
                    RootedCommandAuthorityValidator, CurrentAuthority()
                ),
            )
        assert delayed_permit.value.code is TargetErrorCode.WORKER_UNAVAILABLE

    root_witness = rooted_worker_module.RootWitness(
        FileIdentity(1, 2),
        owner.profile.identity.mount_id,
        owner.profile.identity.filesystem_id,
    )
    run(exercise())


def test_patch_phase_10_runtime_start_reaps_canary_and_launch_faults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release private worker resources when startup proof cannot complete."""
    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    bundle_root = tmp_path / "bundle"
    root.mkdir()
    namespace.mkdir()
    bundle_root.mkdir()
    backend = _runtime(root, namespace).profile.execution_plan.settings.backend
    closed: list[str] = []

    class Bundle:
        """Provide immutable bundle facts after a separately tested copy."""

        root = bundle_root
        digest = "implementation-digest"
        source_digest = sandbox_commit_module._PINNED_WORKER_SOURCE_DIGEST

        def close(self) -> None:
            """Record release of the temporary private bundle."""
            closed.append("bundle")

    async def available_probe(
        _backend: SandboxBackend,
    ) -> SandboxBackendProbeResult:
        """Return selected backend capability before child launch."""
        return cast(
            SandboxBackendProbeResult,
            SimpleNamespace(
                ok=True,
                capabilities=SimpleNamespace(
                    backend=backend,
                    runtime_name="test-runtime",
                    sandbox_executable="test-sandbox",
                ),
            ),
        )

    def private_command(*_arguments: object) -> tuple[str, ...]:
        """Keep launch failure tests independent of native command assembly."""
        return ("test-sandbox", "worker")

    monkeypatch.setattr(
        sandbox_commit_module,
        "_ImplementationBundle",
        SimpleNamespace(create=lambda _workspace: Bundle()),
    )
    monkeypatch.setattr(
        sandbox_commit_module, "_runtime_backend_probe", available_probe
    )
    monkeypatch.setattr(
        sandbox_commit_module, "_runtime_child_command", private_command
    )

    async def exercise() -> None:
        """Force each pre-attestation startup failure in isolation."""
        with monkeypatch.context() as patches:
            owner = _runtime(root, namespace)._process

            def cannot_write(_path: Path, _value: bytes) -> int:
                """Model an inaccessible host canary directory."""
                raise OSError("canary unavailable")

            patches.setattr(Path, "write_bytes", cannot_write)
            with pytest.raises(TargetInspectionError) as canary_error:
                await owner.start()
            assert (
                canary_error.value.code
                is TargetErrorCode.CAPABILITY_UNAVAILABLE
            )

        with monkeypatch.context() as patches:
            owner = _runtime(root, namespace)._process

            def cancelled_write(_path: Path, _value: bytes) -> int:
                """Propagate caller cancellation while releasing the bundle."""
                raise CancelledError

            patches.setattr(Path, "write_bytes", cancelled_write)
            with pytest.raises(CancelledError):
                await owner.start()

        with monkeypatch.context() as patches:
            owner = _runtime(root, namespace)._process

            async def launch_fails(
                *_args: object, **_kwargs: object
            ) -> object:
                """Model operating-system refusal before a child is created."""
                raise OSError("launch unavailable")

            patches.setattr(
                sandbox_commit_module, "create_subprocess_exec", launch_fails
            )
            with pytest.raises(TargetInspectionError) as launch_error:
                await owner.start()
            assert (
                launch_error.value.code
                is TargetErrorCode.CAPABILITY_UNAVAILABLE
            )

        with monkeypatch.context() as patches:
            owner = _runtime(root, namespace)._process

            async def inspection_error(
                *_args: object, **_kwargs: object
            ) -> object:
                """Propagate an already classified child launch failure."""
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)

            patches.setattr(
                sandbox_commit_module,
                "create_subprocess_exec",
                inspection_error,
            )
            with pytest.raises(
                TargetInspectionError
            ) as classified_launch_error:
                await owner.start()
            assert (
                classified_launch_error.value.code
                is TargetErrorCode.WORKER_UNAVAILABLE
            )

        with monkeypatch.context() as patches:
            owner = _runtime(root, namespace)._process

            async def launched(*_args: object, **_kwargs: object) -> object:
                """Return a child whose canary attestation is malformed."""
                return SimpleNamespace(pid=42)

            async def malformed_canary(
                _owner: sandbox_commit_module._SandboxRuntimeProcess,
                _kind: str,
                _body: Mapping[str, object],
                **_kwargs: object,
            ) -> Mapping[str, object]:
                """Expose a child response that fails canary attestation."""
                return {"pid": 42, "outside_read_denied": False}

            reaped: list[str] = []

            async def reap(
                _owner: sandbox_commit_module._SandboxRuntimeProcess,
            ) -> None:
                """Record session teardown after failed child attestation."""
                reaped.append("process")

            patches.setattr(
                sandbox_commit_module, "create_subprocess_exec", launched
            )
            patches.setattr(
                sandbox_commit_module._SandboxRuntimeProcess,
                "_request_locked",
                malformed_canary,
            )
            patches.setattr(
                sandbox_commit_module._SandboxRuntimeProcess, "_reap", reap
            )
            with pytest.raises(TargetInspectionError) as malformed:
                await owner.start()
            assert (
                malformed.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
            )
            assert reaped == ["process"]

    run(exercise())
    assert closed == ["bundle", "bundle", "bundle", "bundle"]


def test_patch_phase_10_service_reuses_durable_terminal_claim_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return prior durable outcomes without replaying planning or a worker."""
    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)
    request_id = PatchRequestId("request_" + "a" * 16)
    correlation = PatchObserverCorrelationId("correlation_" + "a" * 16)
    other_correlation = PatchObserverCorrelationId("correlation_" + "b" * 16)
    result = sandbox_commit_module._approval_result(
        request_id,
        PatchPlanId("plan_" + "a" * 16),
        ApprovalDecisionState.DENIED,
    )

    class Store:
        """Expose controlled durable branches before planning starts."""

        existing: object
        claim: object

        async def reserve(self, *_arguments: object) -> object:
            """Return one opaque reservation for the request key."""
            return object()

        async def inspect(self, _access: DurableRequestAccess) -> object:
            """Return the next durable state controlled by this test."""
            return self.existing

        async def persist_plan(self, *_arguments: object) -> None:
            """Accept the fake plan before approval is simulated."""

        async def claim_commit(self, *_arguments: object) -> object:
            """Return the preselected ownership/attachment branch."""
            return self.claim

    store = Store()
    service = object.__new__(SandboxPatchSdkService)
    service.runtime = runtime
    service.scope = cast(
        ResolvedMutationScope, SimpleNamespace(limits=_limits())
    )
    service.inspection = cast(
        SandboxInspectionTarget,
        SimpleNamespace(inspect=lambda _request: None),
    )
    service.store = cast(DurablePatchStore, store)
    service.policy = _runtime_policy()
    service._latest = None
    service._pending = {}
    service._requests = {}
    service._workers = {}
    service._worker_tasks = {}
    service._reconciliation_tasks = set()
    service._reader_tasks = set()

    async def now() -> ExpiryTick:
        """Return a stable planning tick for fake approval branches."""
        return ExpiryTick(1)

    service.configuration = cast(
        SandboxPatchServiceConfiguration,
        SimpleNamespace(
            subject=_runtime_subject(),
            input_limits=object(),
            planner=SimpleNamespace(plan=lambda *_arguments: None),
            approvals=SimpleNamespace(),
            approval_issuer=SimpleNamespace(),
            clock=SimpleNamespace(now=now),
            review_duration=DurationTicks(10),
            lease_duration=DurationTicks(10),
            pending_factory=lambda *_arguments: object(),
        ),
    )
    capability = cast(PatchInvocationCapability, object())
    monkeypatch.setattr(
        sandbox_commit_module,
        "_canonical_request",
        lambda *_arguments: SimpleNamespace(digest="request-digest"),
    )

    async def attached(
        _service: SandboxPatchSdkService,
        _request_id: PatchRequestId,
        _identity: DurableRequestIdentity,
        _correlation: PatchObserverCorrelationId,
    ) -> object:
        """Return one retained pending outcome without a duplicate worker."""
        return result

    monkeypatch.setattr(SandboxPatchSdkService, "_attached_outcome", attached)

    async def exercise() -> None:
        """Exercise correlation fencing and terminal/attached claim reuse."""
        store.existing = SimpleNamespace(
            terminal=SimpleNamespace(
                outbox=SimpleNamespace(correlation_id=other_correlation),
                result=result,
            ),
            plan=None,
        )
        with pytest.raises(TargetInspectionError) as stale_terminal:
            await service.invoke(
                OperationType.EDIT,
                b"{}",
                capability,
                request_id,
                correlation,
            )
        assert stale_terminal.value.code is TargetErrorCode.WITNESS_STALE

        store.existing = SimpleNamespace(
            terminal=SimpleNamespace(
                outbox=SimpleNamespace(correlation_id=correlation),
                result=result,
            ),
            plan=None,
        )
        assert (
            await service.invoke(
                OperationType.EDIT,
                b"{}",
                capability,
                request_id,
                correlation,
            )
            is result
        )

        store.existing = SimpleNamespace(terminal=None, plan=object())
        assert (
            await service.invoke(
                OperationType.EDIT,
                b"{}",
                capability,
                request_id,
                correlation,
            )
            is result
        )

    run(exercise())


def test_patch_phase_10_service_preserves_review_and_claim_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist planning once and honor durable review and claim decisions."""
    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)
    request_id = PatchRequestId("request_" + "c" * 16)
    correlation = PatchObserverCorrelationId("correlation_" + "c" * 16)
    plan_id = PatchPlanId("plan_" + "c" * 16)
    result = sandbox_commit_module._approval_result(
        request_id, plan_id, ApprovalDecisionState.DENIED
    )

    class Store:
        """Record planning and return one controlled durable claim result."""

        review: object
        claim: object

        async def reserve(self, *_arguments: object) -> object:
            """Reserve the single synthetic request identity."""
            return object()

        async def inspect(self, _access: DurableRequestAccess) -> object:
            """Report an unplanned request so the service performs planning."""
            return SimpleNamespace(terminal=None, plan=None)

        async def persist_plan(self, *_arguments: object) -> None:
            """Accept the fake durable plan before the review decision."""

        async def claim_commit(self, *_arguments: object) -> object:
            """Return the desired terminal, attachment, or owner state."""
            return self.claim

        async def request_cancellation(
            self, _access: DurableRequestAccess
        ) -> None:
            """Record cancellation before reaping the bound worker."""

        async def bind_worker(self, *_arguments: object) -> None:
            """Accept the synthetic worker binding before failure."""

    store = Store()
    service = object.__new__(SandboxPatchSdkService)
    service.runtime = runtime
    service.scope = ResolvedMutationScope(
        ContextKind.SANDBOX,
        runtime.profile.identity,
        None,
        _limits(),
        frozenset(
            (
                Capability.READ_FOR_MUTATION,
                Capability.OBSERVE_MUTATION_PRECONDITIONS,
            )
        ),
        frozenset(),
    )

    async def inspected(_request: InspectionRequest) -> object:
        """Return the planner workspace selected from the sandbox scope."""
        return SimpleNamespace(planner_workspace=lambda: object())

    async def planned(*_arguments: object) -> object:
        """Return an opaque candidate consumed only by fake sealing helpers."""
        return SimpleNamespace(request_digest="candidate-digest")

    async def review(_request: object) -> object:
        """Return the currently selected broker decision."""
        return store.review

    async def issue(*_arguments: object) -> object:
        """Return a sealed approval only after the broker approves."""
        return object()

    async def now() -> ExpiryTick:
        """Return one deterministic durable clock tick."""
        return ExpiryTick(1)

    service.inspection = cast(
        SandboxInspectionTarget, SimpleNamespace(inspect=inspected)
    )
    service.store = cast(DurablePatchStore, store)
    service.policy = _runtime_policy()
    service.handshake = cast(TargetHandshake, object())
    service.configuration = cast(
        SandboxPatchServiceConfiguration,
        SimpleNamespace(
            subject=_runtime_subject(),
            input_limits=object(),
            planner=SimpleNamespace(plan=planned),
            approvals=SimpleNamespace(await_review=review),
            approval_issuer=SimpleNamespace(issue=issue),
            clock=SimpleNamespace(now=now),
            review_duration=DurationTicks(10),
            lease_duration=DurationTicks(10),
            pending_factory=lambda *_arguments: object(),
        ),
    )
    capability = cast(PatchInvocationCapability, object())
    service._latest = None
    service._pending = {}
    service._requests = {}
    service._workers = {}
    service._worker_tasks = {}
    service._reconciliation_tasks = set()
    service._reader_tasks = set()

    class Authorizer:
        """Allow the service to reach its durable review boundary."""

        def __init__(self, _policy: TrustedPatchPolicy) -> None:
            """Retain no mutable policy state in the fake authorizer."""

        async def authorize_preinspection(self, _request: object) -> object:
            """Return a synthetic preflight proof."""
            return object()

        async def authorize_final(
            self, _preflight: object, _candidate: object, _handshake: object
        ) -> object:
            """Return a synthetic final policy proof."""
            return SimpleNamespace(approval=object())

    monkeypatch.setattr(sandbox_commit_module, "PolicyAuthorizer", Authorizer)
    monkeypatch.setattr(
        sandbox_commit_module,
        "_canonical_request",
        lambda *_arguments: SimpleNamespace(digest="request-digest"),
    )
    monkeypatch.setattr(
        sandbox_commit_module,
        "_semantic_paths",
        lambda _canonical: ((LogicalPath("note.txt"),), frozenset()),
    )
    monkeypatch.setattr(
        sandbox_commit_module, "PreflightRequest", lambda *_arguments: object()
    )
    monkeypatch.setattr(
        sandbox_commit_module, "_patch_request", lambda *_arguments: object()
    )
    monkeypatch.setattr(
        sandbox_commit_module, "PlanBinding", lambda *_arguments: object()
    )
    monkeypatch.setattr(
        sandbox_commit_module,
        "seal_plan",
        lambda selected_plan_id, *_arguments: SimpleNamespace(
            plan_id=selected_plan_id
        ),
    )
    monkeypatch.setattr(
        sandbox_commit_module, "_durable_plan", lambda _plan: object()
    )
    monkeypatch.setattr(
        sandbox_commit_module, "_durable_artifacts", lambda _plan: ()
    )
    monkeypatch.setattr(
        sandbox_commit_module,
        "PlanReviewRequest",
        lambda *_arguments: object(),
    )

    async def attached(
        _service: SandboxPatchSdkService,
        _request_id: PatchRequestId,
        _identity: DurableRequestIdentity,
        _correlation: PatchObserverCorrelationId,
    ) -> object:
        """Return the exact durable attachment instead of minting a worker."""
        return result

    async def reap_bound(
        _service: SandboxPatchSdkService, *_arguments: object
    ) -> None:
        """Fence an invalid owner branch without requiring a native runtime."""

    async def suspended(
        _service: SandboxPatchSdkService, *_arguments: object
    ) -> object:
        """Return the durable recovery projection after local setup fails."""
        return result

    monkeypatch.setattr(SandboxPatchSdkService, "_attached_outcome", attached)
    monkeypatch.setattr(
        SandboxPatchSdkService, "_reap_bound_worker", reap_bound
    )
    monkeypatch.setattr(SandboxPatchSdkService, "_suspend_worker", suspended)

    async def exercise() -> None:
        """Exercise denied, terminal, attached, and malformed outcomes."""
        store.review = SimpleNamespace(
            state=ApprovalDecisionState.DENIED, grant=None
        )
        denied = await service.invoke(
            OperationType.EDIT,
            b"{}",
            capability,
            request_id,
            correlation,
        )
        assert isinstance(denied, PatchResult)
        assert denied.status is PatchStatus.APPROVAL_DENIED

        store.review = SimpleNamespace(
            state=ApprovalDecisionState.APPROVED, grant=object()
        )
        store.claim = SimpleNamespace(
            state=DurableCommitClaimState.TERMINAL,
            terminal=SimpleNamespace(result=result),
        )
        assert (
            await service.invoke(
                OperationType.EDIT,
                b"{}",
                capability,
                request_id,
                correlation,
            )
            is result
        )

        store.claim = SimpleNamespace(
            state=DurableCommitClaimState.ATTACHED, terminal=None
        )
        assert (
            await service.invoke(
                OperationType.EDIT,
                b"{}",
                capability,
                request_id,
                correlation,
            )
            is result
        )

        store.claim = SimpleNamespace(
            state=DurableCommitClaimState.OWNER,
            lease=SimpleNamespace(),
            terminal=None,
        )
        outcome = await service.invoke(
            OperationType.EDIT,
            b"{}",
            capability,
            request_id,
            correlation,
        )
        assert outcome is result

        pending = DurablePendingRequest(
            PatchPendingOperationId.new(),
            correlation,
            DurationTicks(10),
        )
        object.__setattr__(
            service.configuration,
            "pending_factory",
            lambda *_arguments: pending,
        )
        owner_lease = DurableCommitLease(
            request_id,
            runtime.profile.identity.domain_id,
            PatchCommitOwnerId("owner_" + "c" * 16),
            SequenceNumber(1),
            ExpiryTick(20),
        )
        store.claim = SimpleNamespace(
            state=DurableCommitClaimState.OWNER,
            lease=owner_lease,
            terminal=None,
        )

        async def cancelled_binding(
            _service: SandboxPatchSdkService,
        ) -> DurableWorkerBinding:
            """Model cancellation before a worker task can be constructed."""
            raise CancelledError

        with monkeypatch.context() as patches:
            patches.setattr(
                SandboxPatchSdkService, "_worker_binding", cancelled_binding
            )
            assert (
                await service.invoke(
                    OperationType.EDIT,
                    b"{}",
                    capability,
                    request_id,
                    correlation,
                )
                is result
            )

        class FailedWorker:
            """Expose a started worker task that fails before settlement."""

            async def commit(self, _command: object) -> WorkerReport:
                """Fail after ownership has been durably bound."""
                raise RuntimeError("worker failed")

        async def bound_worker(
            _service: SandboxPatchSdkService,
        ) -> DurableWorkerBinding:
            """Return the opaque binding accepted by the fake store."""
            return cast(DurableWorkerBinding, object())

        async def worker(
            _runtime: SandboxPatchRuntime,
            _scope: ResolvedMutationScope,
        ) -> RootedSandboxCommitWorker:
            """Return the worker that fails after task registration."""
            return cast(RootedSandboxCommitWorker, FailedWorker())

        async def issued(*_arguments: object) -> None:
            """Accept authority issuance before the worker failure."""

        with monkeypatch.context() as patches:
            patches.setattr(
                SandboxPatchSdkService, "_worker_binding", bound_worker
            )
            patches.setattr(SandboxPatchRuntime, "worker", worker)
            patches.setattr(
                sandbox_commit_module,
                "_issue_rooted_command_authority_for_validator",
                issued,
            )
            patches.setattr(
                sandbox_commit_module, "footprint_for", lambda _plan: object()
            )
            patches.setattr(
                sandbox_commit_module,
                "SealedCommitCommand",
                lambda *_arguments: cast(SealedCommitCommand, object()),
            )
            assert (
                await service.invoke(
                    OperationType.EDIT,
                    b"{}",
                    capability,
                    request_id,
                    correlation,
                )
                is result
            )

    run(exercise())


def test_patch_phase_10_service_reaps_worker_failure_and_future_errors() -> (
    None
):
    """Retain durable cleanup and propagate only the exact reader failure."""
    request_id = PatchRequestId("request_" + "d" * 16)
    correlation = PatchObserverCorrelationId("correlation_" + "d" * 16)
    identity = DurableRequestIdentity(
        PatchTenantId("tenant-finish"),
        PatchPrincipalId("principal-finish"),
        PatchExecutionId("execution_" + "d" * 16),
        PolicyRouteId("route-finish"),
        RetransmissionKey("retry-finish"),
    )
    service = object.__new__(SandboxPatchSdkService)
    closed: list[str] = []
    reaped: list[str] = []

    class Runtime:
        """Expose the close operation required by failure reconciliation."""

        async def close(self) -> None:
            """Record worker shutdown before durable release."""
            closed.append("runtime")

    class Store:
        """Record reaping and reject the synthetic terminal reader access."""

        async def mark_worker_reaped(
            self, _lease: object, _binding: object
        ) -> None:
            """Persist release of the exact failed child binding."""
            reaped.append("worker")

    service.runtime = cast(SandboxPatchRuntime, Runtime())
    service.store = cast(DurablePatchStore, Store())
    service._worker_tasks = {}
    service._reader_tasks = set()
    service._pending = {}

    pending = PatchPending(
        1,
        PatchPendingOperationId.new(),
        request_id,
        correlation,
        LifecyclePhase.SETTLEMENT_PENDING,
    )

    async def fails() -> WorkerReport:
        """Make the shielded worker fail after the background task starts."""
        raise RuntimeError("worker failed")

    async def exercise() -> None:
        """Observe worker failure and reader error through service futures."""
        task = create_task(fails())
        await sleep(0)
        await service._finish_worker(
            task,
            request_id,
            identity,
            PatchPlanId("plan_" + "d" * 16),
            cast(DurableCommitLease, object()),
            correlation,
            (),
            cast(DurableWorkerBinding, object()),
        )
        assert closed == ["runtime"]
        assert reaped == ["worker"]

        future = service._terminal_future(pending)
        with pytest.raises(TargetInspectionError) as stale_pending:
            await future
        assert stale_pending.value.code is TargetErrorCode.WITNESS_STALE

    run(exercise())


def test_patch_phase_10_runtime_endpoint_and_authority_fence_stale_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unbound endpoints and unreadable durable fence checks."""
    root = tmp_path / "view"
    namespace = tmp_path / "namespace"
    root.mkdir()
    namespace.mkdir()
    runtime = _runtime(root, namespace)
    identity = runtime.profile.identity
    root_witness = rooted_worker_module.RootWitness(
        FileIdentity(1, 2), identity.mount_id, identity.filesystem_id
    )
    worker = EphemeralWorkerWitness(
        runtime.profile.channel_id, "worker-phase-ten", "fence-phase-ten"
    )
    scope = ResolvedMutationScope(
        ContextKind.SANDBOX,
        identity,
        None,
        _limits(),
        frozenset(
            (
                Capability.READ_FOR_MUTATION,
                Capability.OBSERVE_MUTATION_PRECONDITIONS,
            )
        ),
        frozenset(),
        root_witness,
        worker,
    )
    request_id = PatchRequestId("request_" + "e" * 16)
    command = _unissued_command(identity, 1, None, request_id)
    lease = DurableCommitLease(
        request_id,
        identity.domain_id,
        PatchCommitOwnerId("owner_" + "e" * 16),
        SequenceNumber(1),
        ExpiryTick(20),
    )
    runtime_primitives = (
        TargetPrimitive.METADATA_PRESERVATION,
        TargetPrimitive.BOUNDED_WRITE,
        TargetPrimitive.REPLACE_PUBLICATION,
        TargetPrimitive.NOREPLACE_CREATE_MOVE,
        TargetPrimitive.DIRECTORY_ENTRY_DELETE,
        TargetPrimitive.SAME_FILESYSTEM_MOVE,
        TargetPrimitive.STAGING,
        TargetPrimitive.STRUCTURAL_VERIFICATION,
        TargetPrimitive.PERSISTENCE,
        TargetPrimitive.CANCELLATION_SETTLEMENT,
        TargetPrimitive.JOURNAL_DELIVERY,
        TargetPrimitive.APPROVAL,
        TargetPrimitive.DURABLE_FENCING,
    )
    primitive_receipts = {item: "receipt" for item in runtime_primitives}
    receipt = sandbox_commit_module.SandboxRuntimeReceipt(
        SandboxSessionId("session-" + "e" * 16),
        sandbox_commit_module.SandboxProfileReceipt("receipt"),
        root_witness,
        worker,
        primitive_receipts,
        "runtime",
        "policy",
        "child",
        "canary",
    )
    runtime._scope = scope
    runtime._receipt = receipt
    runtime._receipt_guard = receipt

    async def exercise() -> None:
        """Exercise unavailable workers, stale binding, and recovery."""
        with pytest.raises(TargetInspectionError) as missing_endpoint:
            await runtime.worker(scope)
        assert (
            missing_endpoint.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        )

        runtime._scope = None
        with pytest.raises(TargetInspectionError) as stale_scope:
            runtime._bind_sandbox_endpoint(scope)
        assert stale_scope.value.code is TargetErrorCode.WITNESS_STALE
        runtime._scope = scope
        runtime._endpoint = object()
        with pytest.raises(TargetInspectionError) as invalid_endpoint:
            runtime._bind_sandbox_endpoint(scope)
        assert (
            invalid_endpoint.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        )

        endpoint = sandbox_commit_module._SandboxEndpoint(runtime, scope)
        report = WorkerReport(WorkerState.LIVE, None)

        async def completed_report() -> WorkerReport:
            """Return a report that reconciliation can durably retain."""
            return report

        task = create_task(completed_report())
        await sleep(0)
        endpoint._active_request = request_id
        endpoint._active_task = task
        assert await endpoint.reconcile_sandbox(request_id) is report
        assert endpoint._settlements[request_id] is report
        assert endpoint._active_task is None
        assert endpoint._active_request is None

    async def current_scope(
        _runtime: SandboxPatchRuntime, selected: ResolvedMutationScope
    ) -> sandbox_commit_module.SandboxRuntimeReceipt:
        """Return the exact issued receipt for a matching test scope."""
        assert selected is scope
        return receipt

    class FencedStore:
        """Model a durable store that cannot confirm the owner fence."""

        async def is_current_fence(
            self, _lease: DurableCommitLease, _now: ExpiryTick
        ) -> bool:
            """Raise a fence error rather than authorize a stale command."""
            raise DurableStoreError(DurableStoreErrorCode.FENCED)

    class Clock:
        """Provide a current durable tick for the authorization query."""

        async def now(self) -> ExpiryTick:
            """Return the bounded policy clock value."""
            return ExpiryTick(1)

    monkeypatch.setattr(SandboxPatchRuntime, "_require_scope", current_scope)
    authority = _SandboxDurableCommandAuthority(
        runtime,
        scope,
        lease,
        cast(InMemoryDurablePatchStore, FencedStore()),
        cast(ApprovalClock, Clock()),
    )
    run(exercise())
    assert not run(authority.is_rooted_command_current(command))

    with pytest.raises(TargetInspectionError) as invalid_configuration:
        SandboxPatchServiceConfiguration(
            cast(ExecutionSubject, object()),
            cast(PlannerFacade, object()),
            cast(ApprovalService, object()),
            cast(PhaseFiveDurableApprovalIssuer, object()),
            cast(ApprovalClock, object()),
            DurationTicks(10),
            DurationTicks(10),
        )
    assert (
        invalid_configuration.value.code
        is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )
