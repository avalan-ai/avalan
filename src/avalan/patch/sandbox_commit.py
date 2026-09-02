"""Bind patch mutation to one selected, persistent sandbox runtime.

The normal shell and code sandbox profiles remain read-only.  This module is
the separate, runtime-owned mutation endpoint: it starts one Seatbelt child
for a selected sandbox session and sends only authenticated sealed commands
to that child.  The child owns the selected view for its whole lifetime and
uses rooted primitives there; it never receives a local target object.
"""

from asyncio import (
    CancelledError,
    Future,
    Lock,
    Task,
    create_subprocess_exec,
    create_task,
    get_running_loop,
    shield,
    wait_for,
)
from asyncio.subprocess import DEVNULL, PIPE, Process
from base64 import b64encode
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from hashlib import sha256
from hmac import compare_digest, digest
from importlib.util import find_spec
from json import dumps, loads
from os import (
    O_DIRECTORY,
    O_NOFOLLOW,
    O_RDONLY,
    fchmod,
    fstat,
    listdir,
    lstat,
    rmdir,
    unlink,
)
from os import (
    close as close_file_descriptor,
)
from os import (
    open as open_file_descriptor,
)
from pathlib import Path
from secrets import token_bytes
from shutil import copytree, rmtree
from stat import S_IEXEC, S_IREAD, S_ISDIR, S_IWRITE
from sys import executable
from tempfile import mkdtemp
from types import MappingProxyType, TracebackType
from typing import NewType, Protocol, TypedDict, runtime_checkable

from cffi import __file__ as cffi_file
from cryptography import __file__ as cryptography_file

from avalan.isolation import SandboxBackend
from avalan.patch.coordinator import (
    ArtifactJournal,
    CommitLease,
    JournalStep,
    RetransmissionKey,
    RootedCommandAuthorityValidator,
    RootedSandboxCommitWorker,
    SealedCommitCommand,
    SettlementJournal,
    WorkerReport,
    WorkerState,
    _issue_rooted_command_authority_for_validator,
    _rooted_sandbox_endpoint,
    _sandbox_worker_for_endpoint,
    footprint_for,
)
from avalan.patch.domain import (
    AlgorithmDigest,
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
    PatchArtifactId,
    PatchCommitOwnerId,
    PatchDiagnostic,
    PatchErrorCode,
    PatchExecutionId,
    PatchInput,
    PatchInvocationOutcome,
    PatchLifecycleEvent,
    PatchLimits,
    PatchLineageId,
    PatchObserverCorrelationId,
    PatchObserverId,
    PatchPending,
    PatchPendingOperationId,
    PatchPlanId,
    PatchProtocolId,
    PatchRequest,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PatchStepId,
    PostconditionState,
    RequestedEffectOccurrence,
    Retryability,
    SequenceNumber,
    SourceBytes,
    WorkspaceChange,
)
from avalan.patch.durable_approval import PhaseFiveDurableApprovalIssuer
from avalan.patch.durable_coordinator import (
    DurableArtifactObservation,
    DurablePatchReconciler,
)
from avalan.patch.durable_store import (
    DurableArtifactState,
    DurableCommitClaimState,
    DurableCommitLease,
    DurablePatchStore,
    DurablePatchStoreBinding,
    DurablePendingAccess,
    DurablePendingRequest,
    DurablePlanReference,
    DurableProtocolOrigin,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableRequestSnapshot,
    DurableReservation,
    DurableStepBinding,
    DurableStoreError,
    DurableWorkerBinding,
)
from avalan.patch.parser import (
    AddDeclarationSyntax,
    CanonicalPatchRequest,
    DeleteDeclarationSyntax,
    PatchDocumentSyntax,
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderProfile,
    RawToolCallId,
    StructuredEditSyntax,
    UpdateDeclarationSyntax,
)
from avalan.patch.planner import (
    PlannedFile,
    PlannedLineage,
    PlannerFacade,
)
from avalan.patch.policy import (
    ApprovalClock,
    ApprovalDecisionState,
    ApprovalRequirements,
    ApprovalService,
    ExecutionSubject,
    PlanBinding,
    PlanReviewRequest,
    PolicyAuthorizer,
    PreflightRequest,
    SealedPlan,
    TrustedPatchPolicy,
    _canonical_fingerprint_bytes,
    _validate_sealed_plan,
    compose_limits,
    seal_plan,
)
from avalan.patch.rooted_worker import (
    _artifacts,
    _steps,
    capture_rooted_root_binding,
    validate_rooted_root_binding,
)
from avalan.patch.sandbox_wire import canonical_sandbox_plan_bytes
from avalan.patch.target import (
    _FUTURE_MUTATION_PRIMITIVES,
    EphemeralWorkerWitness,
    FileIdentity,
    InspectionBatch,
    InspectionRequest,
    LocalPlatformProfile,
    PrimitiveProbe,
    ProbeState,
    ResolvedMutationScope,
    RootWitness,
    ScopeSelection,
    TargetErrorCode,
    TargetHandshake,
    TargetIdentity,
    TargetInspectionError,
    TargetPrimitive,
    TargetSnapshot,
    _seatbelt_string,
    _snapshot_from_worker,
)
from avalan.patch.toolset import (
    PatchApprovalBinding,
    PatchCoordinatorBinding,
    PatchInvocationCapability,
    PatchInvocationHandle,
    PatchPersistenceBinding,
    PatchRuntimeBinding,
    PatchSettlementPort,
    RemotePatchRuntimeWitness,
    _bound_invocation_subscription_access,
)
from avalan.sandbox import (
    BubblewrapSandboxBackend,
    SandboxBackendProbeResult,
    SeatbeltSandboxBackend,
)
from avalan.sandbox.planning import SandboxExecutionPlan

SandboxChannelId = NewType("SandboxChannelId", str)
SandboxContextLifetimeId = NewType("SandboxContextLifetimeId", str)
SandboxWorkerImplementationId = NewType("SandboxWorkerImplementationId", str)
SandboxWorkerProtocolVersion = NewType("SandboxWorkerProtocolVersion", str)
SandboxSessionId = NewType("SandboxSessionId", str)
SandboxProfileReceipt = NewType("SandboxProfileReceipt", str)
SandboxExecutionPlanFingerprint = NewType(
    "SandboxExecutionPlanFingerprint", str
)

_PROTOCOL_VERSION = SandboxWorkerProtocolVersion("sandbox-patch-runtime-v2")
_MESSAGE_VERSION = 2
_MAX_MESSAGE_BYTES = 1_048_576
_PROCESS_CLOSE_SECONDS = 0.25
_PROCESS_IO_SECONDS = 2.0
_PROCESS_REAP_SECONDS = 2.0
_PROCESS_STARTUP_IO_SECONDS = 10.0
_PINNED_WORKER_SOURCE_DIGEST = (
    "dbcb7a365200c2f437bf5563e66dd55241e054e9aa7c0c04d53c2146a16ab29c"
)


def _worker_interpreter_path(current_executable: str) -> str:
    """Return the resolved interpreter that owns the running runtime."""
    interpreter = Path(current_executable).resolve()
    if not interpreter.is_file():
        raise RuntimeError("Python interpreter is unavailable")
    return str(interpreter)


base_executable = _worker_interpreter_path(executable)
_pycparser_spec = find_spec("pycparser")
if _pycparser_spec is None or _pycparser_spec.origin is None:
    raise RuntimeError("pycparser package is unavailable")
pycparser_file = _pycparser_spec.origin


def _is_sha256_digest(value: str) -> bool:
    """Return whether one attestation field has canonical SHA-256 form."""
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


_SANDBOX_PRIMITIVES = frozenset(
    (
        TargetPrimitive.PERSISTENCE,
        TargetPrimitive.CANCELLATION_SETTLEMENT,
        TargetPrimitive.JOURNAL_DELIVERY,
        TargetPrimitive.APPROVAL,
        TargetPrimitive.DURABLE_FENCING,
    )
)


@dataclass(frozen=True, slots=True, repr=False)
class _ImplementationBundle:
    """Keep verified worker code outside every mutable workspace view."""

    root: Path
    digest: str
    source_digest: str

    @classmethod
    def create(
        cls, forbidden_root: Path, *, include_dependencies: bool = True
    ) -> "_ImplementationBundle":
        """Copy the installed implementation into a read-only private root."""
        source_root = Path(__file__).resolve().parents[2]
        source_package = source_root / "avalan"
        source_digest = _worker_source_digest(source_package)
        dependency_roots = (
            Path(cffi_file).parent.resolve(),
            Path(cryptography_file).parent.resolve(),
            Path(pycparser_file).parent.resolve(),
        )
        workspace = forbidden_root.resolve()
        if (
            source_digest != _PINNED_WORKER_SOURCE_DIGEST
            or not source_package.is_dir()
            or any(
                _paths_overlap(workspace, path.resolve())
                for path in (source_package, *dependency_roots)
            )
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        root = Path(mkdtemp(prefix="avalan-sandbox-worker-")).resolve()
        try:
            if _paths_overlap(workspace, root):
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                )
            copytree(
                source_package,
                root / "avalan",
                ignore=lambda _path, names: {
                    name
                    for name in names
                    if name
                    in {"__pycache__", "local_commit.py", "sandbox_commit.py"}
                    or name.endswith(".pyc")
                },
            )
            if any(
                (root / relative).exists()
                for relative in (
                    "avalan/patch/local_commit.py",
                    "avalan/patch/sandbox_commit.py",
                )
            ):
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                )
            if include_dependencies:
                copytree(Path(cffi_file).parent, root / "cffi")
                copytree(Path(cryptography_file).parent, root / "cryptography")
                copytree(Path(pycparser_file).parent, root / "pycparser")
                for extension in Path(cffi_file).parent.parent.glob(
                    "_cffi_backend*.so"
                ):
                    target = root / extension.name
                    target.write_bytes(extension.read_bytes())
            _lock_implementation_tree(root)
            if _worker_source_digest(root / "avalan") != source_digest:
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                )
            return cls(root, _implementation_digest(root), source_digest)
        except BaseException as primary:
            try:
                _remove_owned_bundle(root)
            except BaseException as cleanup:
                raise primary from cleanup
            raise

    def close(self) -> None:
        """Remove the private immutable bundle after the child has reaped."""
        _remove_owned_bundle(self.root)


def _remove_owned_bundle(root: Path) -> None:
    """Remove one owned private bundle without traversing links."""
    if not root.is_absolute():
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    expected = lstat(root)
    if not S_ISDIR(expected.st_mode):
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    descriptor = open_file_descriptor(
        root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW
    )
    try:
        opened = fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        _remove_owned_bundle_contents(descriptor)
        # POSIX can remove a directory only by pathname, not by its already
        # verified descriptor. The bundle's private-name boundary therefore
        # excludes an uncooperative same-UID replacement after this immediate
        # check; any replacement observable before rmdir is left in place.
        current = lstat(root)
        if (current.st_dev, current.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        rmdir(root)
    finally:
        close_file_descriptor(descriptor)


def _remove_owned_bundle_contents(descriptor: int) -> None:
    """Delete entries through one trusted owned-directory descriptor."""
    fchmod(descriptor, S_IREAD | S_IWRITE | S_IEXEC)
    for name in listdir(descriptor):
        expected = lstat(name, dir_fd=descriptor)
        if not S_ISDIR(expected.st_mode):
            unlink(name, dir_fd=descriptor)
            continue
        child = open_file_descriptor(
            name,
            O_RDONLY | O_DIRECTORY | O_NOFOLLOW,
            dir_fd=descriptor,
        )
        try:
            opened = fstat(child)
            if (opened.st_dev, opened.st_ino) != (
                expected.st_dev,
                expected.st_ino,
            ):
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                )
            _remove_owned_bundle_contents(child)
            # See _remove_owned_bundle: verify at the last portable point
            # before pathname removal and fail closed on observed replacement.
            current = lstat(name, dir_fd=descriptor)
            if (current.st_dev, current.st_ino) != (
                expected.st_dev,
                expected.st_ino,
            ):
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                )
            rmdir(name, dir_fd=descriptor)
        finally:
            close_file_descriptor(child)


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either absolute path contains the other."""
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _implementation_digest(root: Path) -> str:
    """Return a stable digest over the exact worker import bundle."""
    if not root.is_dir():
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    digest_value = sha256()
    files = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    )
    if not files:
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest_value.update(len(relative).to_bytes(4, "big"))
        digest_value.update(relative)
        digest_value.update(len(payload).to_bytes(8, "big"))
        digest_value.update(payload)
    return digest_value.hexdigest()


def _worker_source_digest(source_package: Path) -> str:
    """Hash the exact tracked worker source accepted before private copying."""
    if not source_package.is_dir():
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    excluded = {
        Path("patch/local_commit.py"),
        Path("patch/sandbox_commit.py"),
    }
    files = tuple(
        path
        for path in sorted(source_package.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and not path.name.endswith(".pyc")
        and path.relative_to(source_package) not in excluded
    )
    if not files:
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    digest_value = sha256()
    for path in files:
        relative = (
            Path("avalan") / path.relative_to(source_package)
        ).as_posix()
        relative_bytes = relative.encode()
        payload = path.read_bytes()
        digest_value.update(len(relative_bytes).to_bytes(4, "big"))
        digest_value.update(relative_bytes)
        digest_value.update(len(payload).to_bytes(8, "big"))
        digest_value.update(payload)
    return digest_value.hexdigest()


def _lock_implementation_tree(root: Path) -> None:
    """Remove worker-side write permission from every bundle entry."""
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        if path.is_dir():
            path.chmod(S_IREAD | S_IEXEC)
        elif path.is_file():
            path.chmod(S_IREAD)
        else:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    root.chmod(S_IREAD | S_IEXEC)


def sandbox_protocol_id(
    version: SandboxWorkerProtocolVersion,
) -> PatchProtocolId:
    """Return the immutable target protocol identity for one runtime wire."""
    return PatchProtocolId(
        "protocol_" + sha256(version.encode()).hexdigest()[:16]
    )


@dataclass(frozen=True, slots=True, repr=False)
class SandboxRuntimeProfile:
    """Describe the runtime-selected sandbox view and private lease.

    ``execution_plan`` is the same trusted plan chosen for the ordinary
    sandbox context.  The two private paths are runtime configuration, never
    model, tool, SDK, or worker-message input.  The mutation child grants
    write access only to that selected view and its private staging namespace.
    """

    execution_plan: SandboxExecutionPlan
    identity: TargetIdentity
    limits: PatchLimits
    max_snapshot_bytes: ByteSize
    workspace_view_root: str
    private_view_root: str
    cwd: LogicalPath | None
    channel_id: SandboxChannelId
    context_lifetime_id: SandboxContextLifetimeId
    implementation_id: SandboxWorkerImplementationId
    _workspace_host_root: Path = field(repr=False)
    _private_namespace: Path = field(repr=False)
    _mount_map: tuple[tuple[str, str], ...] = field(repr=False)
    protocol_version: SandboxWorkerProtocolVersion = _PROTOCOL_VERSION

    def __post_init__(self) -> None:
        """Reject a profile that is not the selected Seatbelt context."""
        plan = self.execution_plan
        if (
            plan.settings.backend
            not in {SandboxBackend.SEATBELT, SandboxBackend.BUBBLEWRAP}
            or not self.workspace_view_root.startswith("/")
            or not self.private_view_root.startswith("/")
            or not self._workspace_host_root.is_absolute()
            or not self._private_namespace.is_absolute()
            or _paths_overlap(
                self._workspace_host_root.resolve(),
                self._private_namespace.resolve(),
            )
            or not self.channel_id
            or not self.context_lifetime_id
            or not self.implementation_id
            or self.protocol_version != _PROTOCOL_VERSION
            or self.identity.protocol_id
            != sandbox_protocol_id(self.protocol_version)
            or self.identity.implementation_id != self.implementation_id
            or self._mount_map
            != (
                (self.workspace_view_root, str(self._workspace_host_root)),
                (self.private_view_root, str(self._private_namespace)),
            )
            or (
                plan.settings.backend is SandboxBackend.SEATBELT
                and plan.request.cwd != self.workspace_view_root
            )
            or str(self._workspace_host_root)
            not in plan.settings.profile.read_roots
            or str(self._private_namespace)
            not in plan.settings.profile.scratch_roots
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


@dataclass(frozen=True, slots=True, repr=False)
class SandboxPatchRuntimeContext:
    """Carry only host-issued identities for one sandbox agent context.

    The agent runtime creates this value after it has selected the sandbox
    plan.  It cannot be decoded from tool arguments, model output, or a
    worker request.  The execution plan supplies the context-visible
    workspace and private staging root; this value supplies the identity
    facts which bind it to the patch policy and coordinator domain.
    """

    identity: TargetIdentity
    limits: PatchLimits
    max_snapshot_bytes: ByteSize
    cwd: LogicalPath | None
    channel_id: SandboxChannelId
    context_lifetime_id: SandboxContextLifetimeId
    implementation_id: SandboxWorkerImplementationId

    def __post_init__(self) -> None:
        """Require complete immutable host-issued context identities."""
        if (
            type(self.identity) is not TargetIdentity
            or type(self.limits) is not PatchLimits
            or type(self.max_snapshot_bytes) is not ByteSize
            or self.cwd is not None
            and type(self.cwd) is not LogicalPath
            or not self.channel_id
            or not self.context_lifetime_id
            or not self.implementation_id
            or self.identity.implementation_id != self.implementation_id
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


@dataclass(frozen=True, slots=True, repr=False)
class SandboxPatchRuntimeSettings:
    """Construct one mutation runtime from the selected trusted plan.

    This is the production construction boundary used by a loader after the
    normal sandbox session has been selected.  It deliberately derives all
    paths from the immutable execution plan and refuses an ordinary profile
    that already grants a write root.
    """

    execution_plan: SandboxExecutionPlan
    context: SandboxPatchRuntimeContext

    def __post_init__(self) -> None:
        """Require the one read-only Seatbelt plan selected for the context."""
        plan = self.execution_plan
        if (
            type(plan) is not SandboxExecutionPlan
            or type(self.context) is not SandboxPatchRuntimeContext
            or plan.settings.backend
            not in {SandboxBackend.SEATBELT, SandboxBackend.BUBBLEWRAP}
            or bool(plan.settings.profile.write_roots)
            or len(plan.settings.profile.scratch_roots) != 1
            or plan.request.cwd not in plan.settings.profile.read_roots
            or self.context.cwd is not None
            and self.context.cwd.value != plan.request.cwd
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)

    def create_runtime(self) -> "SandboxPatchRuntime":
        """Create the persistent runtime for this exact selected context."""
        plan = self.execution_plan
        backend = plan.settings.backend
        workspace_view_root = (
            plan.request.cwd
            if backend is SandboxBackend.SEATBELT
            else "/workspace"
        )
        private_view_root = (
            plan.settings.profile.scratch_roots[0]
            if backend is SandboxBackend.SEATBELT
            else "/private"
        )
        profile = SandboxRuntimeProfile(
            execution_plan=plan,
            identity=self.context.identity,
            limits=self.context.limits,
            max_snapshot_bytes=self.context.max_snapshot_bytes,
            workspace_view_root=workspace_view_root,
            private_view_root=private_view_root,
            cwd=self.context.cwd,
            channel_id=self.context.channel_id,
            context_lifetime_id=self.context.context_lifetime_id,
            implementation_id=self.context.implementation_id,
            _workspace_host_root=Path(plan.request.cwd),
            _private_namespace=Path(plan.settings.profile.scratch_roots[0]),
            _mount_map=(
                (workspace_view_root, plan.request.cwd),
                (private_view_root, plan.settings.profile.scratch_roots[0]),
            ),
        )
        return SandboxPatchRuntime(profile)


@dataclass(frozen=True, slots=True, repr=False)
class SandboxRuntimeReceipt:
    """Bind a live runtime probe, view, mount, and profile to one session."""

    session_id: SandboxSessionId
    profile_receipt: SandboxProfileReceipt
    root: RootWitness
    worker: EphemeralWorkerWitness
    primitive_receipts: Mapping[TargetPrimitive, str]
    runtime_command_digest: str
    backend_policy_digest: str
    child_process_identity: str
    canary_receipt: str

    def __post_init__(self) -> None:
        """Keep opaque runtime facts complete and typed."""
        if (
            not self.session_id
            or not self.profile_receipt
            or type(self.root) is not RootWitness
            or type(self.worker) is not EphemeralWorkerWitness
            or not isinstance(self.primitive_receipts, Mapping)
            or set(self.primitive_receipts)
            != _FUTURE_MUTATION_PRIMITIVES | _SANDBOX_PRIMITIVES
            or any(
                type(key) is not TargetPrimitive
                or not isinstance(value, str)
                or not value
                for key, value in self.primitive_receipts.items()
            )
            or any(
                not isinstance(value, str) or not value
                for value in (
                    self.runtime_command_digest,
                    self.backend_policy_digest,
                    self.child_process_identity,
                    self.canary_receipt,
                )
            )
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        object.__setattr__(
            self,
            "primitive_receipts",
            MappingProxyType(dict(self.primitive_receipts)),
        )


class _RuntimeMessage(TypedDict):
    """Encode one authenticated host-to-runtime command envelope."""

    payload: Mapping[str, object]
    mac: str


class _RuntimeRequestPayload(TypedDict):
    """Store the validated immutable fields of one child request."""

    version: int
    sequence: int
    kind: str
    receipt: str
    identity: dict[str, str]
    channel_id: str
    implementation_id: str
    body: Mapping[str, object]


class _RuntimeResponsePayload(TypedDict):
    """Store the immutable request echo and one closed worker response."""

    version: int
    sequence: int
    receipt: str
    identity: dict[str, str]
    channel_id: str
    implementation_id: str
    body: Mapping[str, object]
    error: str | None


class _RuntimeChildConfig(TypedDict):
    """Keep static selected-view data in the child process environment."""

    root: str
    namespace: str
    cwd: str | None
    maximum: int
    aggregate_maximum: int
    token: str
    receipt: str
    identity: dict[str, str]
    channel_id: str
    implementation_id: str
    implementation_digest: str
    source_digest: str
    implementation_root: str
    read_canary: str
    session_id: str
    execution_plan: SandboxExecutionPlanFingerprint
    backend: str
    workspace_view: str
    private_view: str
    context_lifetime: str
    protocol: str
    persistent_lease: str
    filesystem: str
    mount: str


def _runtime_profile_receipt(
    profile: SandboxRuntimeProfile,
    runtime_name: str,
    session_id: SandboxSessionId,
) -> SandboxProfileReceipt:
    """Return a non-disclosing receipt for the actual selected profile."""
    payload = "\x00".join(
        (
            runtime_name,
            session_id,
            profile.execution_plan.plan_fingerprint,
            profile.workspace_view_root,
            profile.private_view_root,
            profile.identity.context_id.value,
            profile.identity.workspace_id.value,
            profile.identity.domain_id.value,
            profile.identity.filesystem_id,
            profile.identity.mount_id,
            profile.identity.persistent_lease_id,
            profile.identity.implementation_id,
            profile.channel_id,
            profile.context_lifetime_id,
            *(view + "=" + host for view, host in profile._mount_map),
        )
    )
    return SandboxProfileReceipt(sha256(payload.encode()).hexdigest())


def _primitive_receipts(
    profile_receipt: SandboxProfileReceipt,
    probe: SandboxBackendProbeResult,
    attestation: "_RuntimeAttestation",
) -> Mapping[TargetPrimitive, str]:
    """Bind each advertised primitive to its actual backend probe fact."""
    if not probe.ok or probe.capabilities is None:
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    capability_payload = dumps(
        probe.capabilities.to_dict(), sort_keys=True, separators=(",", ":")
    )
    attestation_payload = dumps(
        {
            "runtime_command": attestation.runtime_command_digest,
            "backend_policy": attestation.backend_policy_digest,
            "child_process": attestation.child_process_identity,
            "canary": attestation.canary_receipt,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return MappingProxyType(
        {
            primitive: (
                sha256(
                    (
                        profile_receipt
                        + "\x00"
                        + primitive.value
                        + "\x00"
                        + capability_payload
                        + "\x00"
                        + attestation_payload
                    ).encode()
                ).hexdigest()
            )
            for primitive in _FUTURE_MUTATION_PRIMITIVES | _SANDBOX_PRIMITIVES
        }
    )


async def _runtime_backend_probe(
    backend: SandboxBackend,
) -> SandboxBackendProbeResult:
    """Probe only the native backend selected in the trusted plan."""
    match backend:
        case SandboxBackend.SEATBELT:
            return await SeatbeltSandboxBackend().probe()
        case SandboxBackend.BUBBLEWRAP:
            return await BubblewrapSandboxBackend().probe()
        case _:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


def _runtime_child_command(
    profile: SandboxRuntimeProfile,
    sandbox_executable: str,
    source_root: Path,
    worker_argv: tuple[str, ...],
    encoded_config: str,
    *,
    resolved_interpreter: Path | None = None,
) -> tuple[str, ...]:
    """Build the sole native child command for the selected runtime view."""
    match profile.execution_plan.settings.backend:
        case SandboxBackend.SEATBELT:
            return (
                sandbox_executable,
                "-p",
                _seatbelt_runtime_profile(profile, source_root),
                "--",
                *worker_argv,
            )
        case SandboxBackend.BUBBLEWRAP:
            return _bubblewrap_runtime_command(
                profile,
                sandbox_executable,
                source_root,
                worker_argv,
                encoded_config,
                resolved_interpreter=resolved_interpreter,
            )
        case _:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


def _runtime_command_digest(command: tuple[str, ...]) -> str:
    """Hash the exact argv emitted to the selected native sandbox."""
    return sha256(
        dumps(
            list(command), separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _backend_policy_digest(
    backend: SandboxBackend,
    command: tuple[str, ...],
) -> str:
    """Hash the exact Seatbelt policy or Bubblewrap argv policy."""
    if backend is SandboxBackend.SEATBELT:
        if len(command) < 4 or command[1] != "-p" or command[3] != "--":
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        payload = command[2].encode()
    elif backend is SandboxBackend.BUBBLEWRAP:
        payload = dumps(
            list(command), separators=(",", ":"), ensure_ascii=False
        ).encode()
    else:
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    return sha256(payload).hexdigest()


def _bubblewrap_runtime_command(
    profile: SandboxRuntimeProfile,
    sandbox_executable: str,
    source_root: Path,
    worker_argv: tuple[str, ...],
    encoded_config: str,
    *,
    resolved_interpreter: Path | None = None,
) -> tuple[str, ...]:
    """Mount the selected view without exposing host workspace paths."""
    roots = _bubblewrap_read_roots(profile, source_root, resolved_interpreter)
    directories = _bubblewrap_parent_directories(
        (
            *roots,
            profile.workspace_view_root,
            profile.private_view_root,
        )
    )
    command: list[str] = [
        sandbox_executable,
        "--die-with-parent",
        "--unshare-user",
        "--uid",
        "0",
        "--gid",
        "0",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--new-session",
        "--unshare-net",
        "--clearenv",
        "--setenv",
        "AVALAN_SANDBOX_PATCH_SESSION",
        encoded_config,
    ]
    for directory in directories:
        command.extend(("--dir", directory))
    for root in roots:
        command.extend(("--ro-bind", root, root))
    command.extend(
        (
            "--bind",
            str(profile._workspace_host_root),
            profile.workspace_view_root,
            "--bind",
            str(profile._private_namespace),
            profile.private_view_root,
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            "/",
            "--",
            *worker_argv,
        )
    )
    return tuple(command)


def _bubblewrap_read_roots(
    profile: SandboxRuntimeProfile,
    source_root: Path,
    resolved_interpreter: Path | None = None,
) -> tuple[str, ...]:
    """Return only interpreter and runtime-code roots needed by the child."""
    interpreter = resolved_interpreter or Path(base_executable).resolve()
    values = (
        Path("/lib"),
        Path("/lib64"),
        Path("/usr/lib"),
        interpreter.parent,
        interpreter.parent.parent,
        source_root,
    )
    roots: list[str] = []
    for value in values:
        root = str(value)
        if value.is_dir() and root not in roots:
            roots.append(root)
    if not roots:
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    return tuple(roots)


def _bubblewrap_parent_directories(
    paths: tuple[str, ...],
) -> tuple[str, ...]:
    """Create exact mount destinations before applying the native binds."""
    directories: set[str] = set()
    for path in paths:
        current = Path(path)
        if not current.is_absolute():
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        while current != current.parent:
            directories.add(str(current))
            current = current.parent
    return tuple(sorted(directories, key=lambda item: (item.count("/"), item)))


def _canary_child_process_identity(
    backend: SandboxBackend,
    session_id: SandboxSessionId,
    host_process_pid: int,
    canary_pid: object,
) -> str:
    """Bind an authenticated canary PID in its selected PID namespace."""
    if (
        type(host_process_pid) is not int
        or host_process_pid <= 0
        or type(canary_pid) is not int
        or canary_pid <= 0
    ):
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    match backend:
        case SandboxBackend.SEATBELT:
            if canary_pid != host_process_pid:
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                )
            payload = session_id + "\x00" + str(host_process_pid)
        case SandboxBackend.BUBBLEWRAP:
            # Bubblewrap retains PID 1 as the namespace reaper for its worker.
            if canary_pid != 2:
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                )
            payload = (
                session_id
                + "\x00"
                + str(host_process_pid)
                + "\x00"
                + str(canary_pid)
            )
        case _:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class _RuntimeAttestation:
    """Bind the exact emitted sandbox and live child canary evidence."""

    runtime_command_digest: str
    backend_policy_digest: str
    child_process_identity: str
    canary_receipt: str


@dataclass(slots=True, repr=False)
class _SandboxRuntimeProcess:
    """Own one persistent authenticated Seatbelt child and its sequence."""

    profile: SandboxRuntimeProfile
    _process: Process | None = field(default=None, init=False)
    _token: bytes | None = field(default=None, init=False)
    _receipt: SandboxProfileReceipt | None = field(default=None, init=False)
    _primitive_receipts: Mapping[TargetPrimitive, str] | None = field(
        default=None, init=False, repr=False
    )
    _root: RootWitness | None = field(default=None, init=False, repr=False)
    _session_id: SandboxSessionId | None = field(
        default=None, init=False, repr=False
    )
    _implementation_digest_value: str | None = field(
        default=None, init=False, repr=False
    )
    _bundle: _ImplementationBundle | None = field(
        default=None, init=False, repr=False
    )
    _canary_root: Path | None = field(default=None, init=False, repr=False)
    _attestation: _RuntimeAttestation | None = field(
        default=None, init=False, repr=False
    )
    _sequence: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    _reap_lock: Lock = field(default_factory=Lock, init=False)
    _closed: bool = field(default=False, init=False)

    async def start(
        self,
    ) -> tuple[
        RootWitness,
        SandboxProfileReceipt,
        SandboxSessionId,
        Mapping[TargetPrimitive, str],
        _RuntimeAttestation,
    ]:
        """Probe and start the one real child for this context lifetime."""
        async with self._lock:
            if self._closed:
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            if self._process is not None:
                root = await self._witness_locked()
                assert self._receipt is not None
                assert self._token is not None
                assert self._primitive_receipts is not None
                assert self._attestation is not None
                return (
                    root,
                    self._receipt,
                    _session_id(self.profile, self._token),
                    self._primitive_receipts,
                    self._attestation,
                )
            backend = self.profile.execution_plan.settings.backend
            if type(backend) is not SandboxBackend:
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                )
            probe = await _runtime_backend_probe(backend)
            if (
                not probe.ok
                or probe.capabilities is None
                or probe.capabilities.backend is not backend
            ):
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                )
            token = token_bytes(32)
            session_id = _session_id(self.profile, token)
            receipt = _runtime_profile_receipt(
                self.profile,
                probe.capabilities.runtime_name,
                session_id,
            )
            bundle = _ImplementationBundle.create(
                self.profile._workspace_host_root
            )
            canary_root = Path(
                mkdtemp(prefix="avalan-sandbox-read-canary-")
            ).resolve()
            canary_path = canary_root / "outside-workspace"
            try:
                canary_path.write_bytes(token_bytes(32))
            except BaseException as exc:
                rmtree(canary_root, ignore_errors=True)
                bundle.close()
                if isinstance(exc, CancelledError):
                    raise
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                ) from exc
            config: _RuntimeChildConfig = {
                "root": self.profile.workspace_view_root,
                "namespace": self.profile.private_view_root,
                "cwd": (
                    self.profile.cwd.value
                    if self.profile.cwd is not None
                    else None
                ),
                "maximum": self.profile.max_snapshot_bytes.value,
                "aggregate_maximum": self.profile.limits.snapshot_bytes.value,
                "token": token.hex(),
                "receipt": receipt,
                "identity": _identity_payload(self.profile.identity),
                "channel_id": self.profile.channel_id,
                "implementation_id": self.profile.implementation_id,
                "implementation_digest": bundle.digest,
                "source_digest": bundle.source_digest,
                "implementation_root": str(bundle.root),
                "read_canary": str(canary_path),
                "session_id": session_id,
                "execution_plan": SandboxExecutionPlanFingerprint(
                    self.profile.execution_plan.plan_fingerprint
                ),
                "backend": backend.value,
                "workspace_view": self.profile.workspace_view_root,
                "private_view": self.profile.private_view_root,
                "context_lifetime": self.profile.context_lifetime_id,
                "protocol": self.profile.identity.protocol_id.value,
                "persistent_lease": self.profile.identity.persistent_lease_id,
                "filesystem": self.profile.identity.filesystem_id,
                "mount": self.profile.identity.mount_id,
            }
            encoded_config = b64encode(
                dumps(config, separators=(",", ":")).encode()
            ).decode()
            source_root = bundle.root
            resolved_interpreter = (
                Path(base_executable).resolve()
                if backend is SandboxBackend.BUBBLEWRAP
                else None
            )
            worker_argv = (
                (
                    str(resolved_interpreter)
                    if resolved_interpreter is not None
                    else base_executable
                ),
                "-I",
                "-c",
                (
                    "import sys, types; "
                    "root=sys.argv[1]; "
                    "sys.path.insert(0, root); "
                    "package=types.ModuleType('avalan'); "
                    "package.__path__=[root + '/avalan']; "
                    "sys.modules['avalan']=package; "
                    "patch=types.ModuleType('avalan.patch'); "
                    "patch.__path__=[root + '/avalan/patch']; "
                    "sys.modules['avalan.patch']=patch; "
                    "model=types.ModuleType('avalan.model'); "
                    "model.__path__=[root + '/avalan/model']; "
                    "sys.modules['avalan.model']=model; "
                    "from avalan.patch.sandbox_worker import main; "
                    "raise SystemExit(main())"
                ),
                str(source_root),
            )
            try:
                command = _runtime_child_command(
                    self.profile,
                    probe.capabilities.sandbox_executable,
                    source_root,
                    worker_argv,
                    encoded_config,
                    resolved_interpreter=resolved_interpreter,
                )
                process = await create_subprocess_exec(
                    *command,
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=DEVNULL,
                    cwd=str(source_root),
                    env={"AVALAN_SANDBOX_PATCH_SESSION": encoded_config},
                    close_fds=True,
                )
            except BaseException as exc:
                rmtree(canary_root, ignore_errors=True)
                bundle.close()
                if isinstance(exc, (CancelledError, TargetInspectionError)):
                    raise
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                ) from exc
            self._process = process
            self._token = token
            self._receipt = receipt
            self._primitive_receipts = None
            self._session_id = session_id
            self._implementation_digest_value = bundle.digest
            self._bundle = bundle
            self._canary_root = canary_root
            self._sequence = 0
            try:
                canary = await self._request_locked(
                    "canary", {}, response_timeout=_PROCESS_STARTUP_IO_SECONDS
                )
                if (
                    set(canary)
                    != {"pid", "outside_read_denied", "metadata_probe"}
                    or canary["outside_read_denied"] is not True
                    or not isinstance(canary["metadata_probe"], str)
                    or not _is_sha256_digest(canary["metadata_probe"])
                ):
                    raise TargetInspectionError(
                        TargetErrorCode.CAPABILITY_UNAVAILABLE
                    )
                child_process_identity = _canary_child_process_identity(
                    backend, session_id, process.pid, canary["pid"]
                )
                attestation = _RuntimeAttestation(
                    _runtime_command_digest(command),
                    _backend_policy_digest(backend, command),
                    child_process_identity,
                    sha256(
                        dumps(
                            canary, separators=(",", ":"), sort_keys=True
                        ).encode()
                    ).hexdigest(),
                )
                primitive_receipts = _primitive_receipts(
                    receipt, probe, attestation
                )
                root = await self._witness_locked(
                    response_timeout=_PROCESS_STARTUP_IO_SECONDS
                )
            except BaseException:
                await self._reap()
                raise
            self._attestation = attestation
            self._primitive_receipts = primitive_receipts
            self._root = root
            return root, receipt, session_id, primitive_receipts, attestation

    async def inspect(
        self, paths: tuple[LogicalPath, ...], expected_root: RootWitness
    ) -> tuple[TargetSnapshot, ...]:
        """Inspect through the already-selected runtime view."""
        response = await self._request(
            "inspect",
            {
                "paths": [path.value for path in paths],
                "root": _root_payload(expected_root),
            },
        )
        snapshots = response.get("snapshots")
        if not isinstance(snapshots, list):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        return tuple(_snapshot_from_worker(item) for item in snapshots)

    async def commit(
        self,
        command: SealedCommitCommand,
        validator: RootedCommandAuthorityValidator,
    ) -> WorkerReport:
        """Commit one sealed command through the selected child only."""
        root = self._root
        session_id = self._session_id
        implementation_digest = self._implementation_digest_value
        receipt = self._receipt
        if (
            root is None
            or session_id is None
            or implementation_digest is None
            or receipt is None
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        response = await self._request(
            "commit",
            _command_payload(
                command,
                self.profile,
                receipt,
                session_id,
                root,
                implementation_digest,
            ),
            command=command,
            validator=validator,
        )
        return _report_from_payload(command, response)

    async def close(self) -> None:
        """Terminate and await the child before ending the runtime lifetime."""
        self._closed = True
        acquired = False
        try:
            await wait_for(
                self._lock.acquire(), timeout=_PROCESS_CLOSE_SECONDS
            )
            acquired = True
            try:
                if self._process is not None:
                    await self._request_locked("close", {})
            except (OSError, TargetInspectionError):
                pass
        except TimeoutError:
            pass
        finally:
            if acquired:
                self._lock.release()
            await self._reap()

    async def _witness_locked(
        self, *, response_timeout: float = _PROCESS_IO_SECONDS
    ) -> RootWitness:
        """Read one rooted witness from the selected child process."""
        response = await self._request_locked(
            "witness", {}, response_timeout=response_timeout
        )
        root = response.get("root")
        return _root_from_payload(root)

    async def _request(
        self,
        kind: str,
        body: Mapping[str, object],
        *,
        command: SealedCommitCommand | None = None,
        validator: RootedCommandAuthorityValidator | None = None,
    ) -> Mapping[str, object]:
        """Serialize one authenticated child request in session order."""
        async with self._lock:
            return await self._request_locked(
                kind, body, command=command, validator=validator
            )

    async def _request_locked(
        self,
        kind: str,
        body: Mapping[str, object],
        *,
        command: SealedCommitCommand | None = None,
        validator: RootedCommandAuthorityValidator | None = None,
        response_timeout: float = _PROCESS_IO_SECONDS,
    ) -> Mapping[str, object]:
        """Exchange exactly one replay-resistant message with the child."""
        if response_timeout <= 0:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        process = self._process
        token = self._token
        receipt = self._receipt
        if (
            process is None
            or token is None
            or receipt is None
            or process.returncode is not None
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        stdin = process.stdin
        stdout = process.stdout
        if stdin is None or stdout is None:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        self._sequence += 1
        payload: _RuntimeRequestPayload = {
            "version": _MESSAGE_VERSION,
            "sequence": self._sequence,
            "kind": kind,
            "receipt": receipt,
            "identity": _identity_payload(self.profile.identity),
            "channel_id": self.profile.channel_id,
            "implementation_id": self.profile.implementation_id,
            "body": body,
        }
        raw = dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        message: _RuntimeMessage = {
            "payload": payload,
            "mac": digest(token, raw, "sha256").hex(),
        }
        request_line = dumps(message, separators=(",", ":")).encode() + b"\n"
        if len(request_line) > _MAX_MESSAGE_BYTES:
            raise TargetInspectionError(TargetErrorCode.LIMIT_EXCEEDED)
        stdin.write(request_line)
        try:
            await wait_for(stdin.drain(), timeout=response_timeout)
        except TimeoutError as exc:
            await self._reap()
            raise TargetInspectionError(
                TargetErrorCode.WORKER_UNAVAILABLE
            ) from exc
        while True:
            try:
                response_line = await wait_for(
                    stdout.readline(), timeout=response_timeout
                )
            except TimeoutError as exc:
                await self._reap()
                raise TargetInspectionError(
                    TargetErrorCode.WORKER_UNAVAILABLE
                ) from exc
            if not response_line or len(response_line) > _MAX_MESSAGE_BYTES:
                await self._reap()
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            response = _response_payload(response_line, token, payload)
            if set(response) != {"control", "effect"}:
                return response
            effect = response["effect"]
            if (
                response["control"] != "fence"
                or type(effect) is not int
                or effect <= 0
                or command is None
                or validator is None
            ):
                await self._reap()
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            allowed = await validator.is_rooted_command_current(command)
            permit_payload: _RuntimeRequestPayload = {
                **payload,
                "kind": "fence_permit",
                "body": {"effect": effect, "allowed": allowed},
            }
            permit_raw = dumps(
                permit_payload, separators=(",", ":"), sort_keys=True
            ).encode()
            permit: _RuntimeMessage = {
                "payload": permit_payload,
                "mac": digest(token, permit_raw, "sha256").hex(),
            }
            stdin.write(dumps(permit, separators=(",", ":")).encode() + b"\n")
            try:
                await wait_for(stdin.drain(), timeout=response_timeout)
            except TimeoutError as exc:
                await self._reap()
                raise TargetInspectionError(
                    TargetErrorCode.WORKER_UNAVAILABLE
                ) from exc

    async def _reap(self) -> None:
        """Cancel the child and clear every unusable session credential."""
        async with self._reap_lock:
            process = self._process
            self._process = None
            self._token = None
            self._receipt = None
            self._primitive_receipts = None
            self._root = None
            self._session_id = None
            self._implementation_digest_value = None
            self._attestation = None
            bundle = self._bundle
            self._bundle = None
            canary_root = self._canary_root
            self._canary_root = None
            self._sequence = 0
            if process is None:
                if bundle is not None:
                    bundle.close()
                if canary_root is not None:
                    rmtree(canary_root, ignore_errors=True)
                return
            try:
                if process.returncode is None:
                    process.terminate()
                try:
                    await wait_for(
                        process.wait(), timeout=_PROCESS_REAP_SECONDS
                    )
                except TimeoutError:
                    if process.returncode is None:
                        process.kill()
                    await wait_for(
                        process.wait(), timeout=_PROCESS_REAP_SECONDS
                    )
            finally:
                if bundle is not None:
                    bundle.close()
                if canary_root is not None:
                    rmtree(canary_root, ignore_errors=True)


def _session_id(
    profile: SandboxRuntimeProfile, token: bytes
) -> SandboxSessionId:
    """Derive a non-exportable session identity from trusted runtime facts."""
    return SandboxSessionId(
        "session_"
        + sha256(
            token
            + profile.context_lifetime_id.encode()
            + profile.identity.persistent_lease_id.encode()
        ).hexdigest()[:32]
    )


def _identity_payload(identity: TargetIdentity) -> dict[str, str]:
    """Return every plan-bound identity for the child authentication check."""
    return {
        "context": identity.context_id.value,
        "workspace": identity.workspace_id.value,
        "domain": identity.domain_id.value,
        "target": identity.target_id.value,
        "protocol": identity.protocol_id.value,
        "filesystem": identity.filesystem_id,
        "mount": identity.mount_id,
        "policy": identity.policy_revision,
        "persistent_lease": identity.persistent_lease_id,
        "approval": identity.approval_channel_id.value,
        "implementation": identity.implementation_id,
    }


def _root_payload(root: RootWitness) -> Mapping[str, object]:
    """Return the full non-path root witness for authenticated comparison."""
    return {
        "device": root.identity.device,
        "inode": root.identity.inode,
        "mount": root.mount_id,
        "filesystem": root.filesystem_id,
    }


def _root_from_payload(value: object) -> RootWitness:
    """Decode an exact rooted witness returned by the sandbox process."""
    if not isinstance(value, dict) or set(value) != {
        "device",
        "inode",
        "mount",
        "filesystem",
    }:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    device = value["device"]
    inode = value["inode"]
    mount = value["mount"]
    filesystem = value["filesystem"]
    if (
        type(device) is not int
        or type(inode) is not int
        or not isinstance(mount, str)
        or not isinstance(filesystem, str)
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return RootWitness(FileIdentity(device, inode), mount, filesystem)


def _command_payload(
    command: SealedCommitCommand,
    profile: SandboxRuntimeProfile,
    receipt: SandboxProfileReceipt,
    session_id: SandboxSessionId,
    root: RootWitness,
    implementation_digest: str,
) -> Mapping[str, object]:
    """Encode the complete sealed transaction as closed canonical JSON."""
    backend = profile.execution_plan.settings.backend
    if type(backend) is not SandboxBackend:
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    return _sealed_command_payload(
        command,
        profile.identity,
        profile.cwd,
        root,
        {
            "backend": backend.value,
            "execution_plan": profile.execution_plan.plan_fingerprint,
            "workspace_view": profile.workspace_view_root,
            "private_view": profile.private_view_root,
            "channel": profile.channel_id,
            "protocol": profile.identity.protocol_id.value,
            "implementation": profile.implementation_id,
            "implementation_digest": implementation_digest,
            "receipt": receipt,
            "session": session_id,
            "context_lifetime": profile.context_lifetime_id,
            "persistent_lease": profile.identity.persistent_lease_id,
            "filesystem": profile.identity.filesystem_id,
            "mount": profile.identity.mount_id,
        },
    )


def _sealed_command_payload(
    command: SealedCommitCommand,
    identity: TargetIdentity,
    cwd: LogicalPath | None,
    root: RootWitness,
    runtime: Mapping[str, object],
) -> Mapping[str, object]:
    """Encode one sealed command for an already-selected runtime profile."""
    _validate_sealed_plan(command.plan)
    plan = command.plan
    canonical = _canonical_fingerprint_bytes(
        plan.binding, plan.candidate, plan.review.expiry
    )
    if sha256(canonical).digest() != plan.fingerprint._value:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    binding = plan.binding
    request = binding.request
    candidate = plan.candidate
    plan_fields = {
        "id": plan.plan_id.value,
        "sealed_fingerprint": b64encode(plan.fingerprint._value).decode(),
        "sealed_canonical": b64encode(canonical).decode(),
        "request": {
            "schema": request.schema_version,
            "id": request.request_id.value,
            "execution": request.execution_id.value,
            "operation": request.operation.value,
            "input_digest": request.input_bytes.digest().value,
            "paths": [item.value for item in request.logical_paths],
        },
        "subject": {
            "principal": binding.subject.principal.value,
            "tenant": binding.subject.tenant.value,
            "run": binding.subject.run.value,
            "session": binding.subject.session.value,
            "task": binding.subject.task.value,
            "agent": binding.subject.agent.value,
        },
        "context": binding.context_kind.value,
        "target": _identity_payload(binding.target),
        "cwd": None if binding.cwd is None else binding.cwd.value,
        "request_digest": candidate.request_digest.value,
        "authorized_effects": sorted(
            item.value for item in binding.final.effects
        ),
        "lineages": [_lineage_payload(item) for item in candidate.lineages],
        "final_files": [_file_payload(item) for item in candidate.final_files],
        "diff": {
            "entries": [
                b64encode(item).decode() for item in candidate.diff.entries
            ],
            "rendered": b64encode(candidate.diff.rendered).decode(),
            "digest": candidate.diff.digest.value,
        },
        "review": {
            "expiry": plan.review.expiry.value,
            "diff_digest": plan.review.diff.digest.value,
        },
    }
    wire_canonical = canonical_sandbox_plan_bytes(plan_fields)
    plan_payload = {
        **plan_fields,
        "fingerprint": b64encode(sha256(wire_canonical).digest()).decode(),
        "canonical": b64encode(wire_canonical).decode(),
    }
    return {
        "schema": binding.context_kind.value + "-patch-command-v1",
        "plan": plan_payload,
        "command": {
            "domain": command.lease.domain_id.value,
            "request": command.lease.request_id.value,
            "fence": command.lease.fence,
            "footprint": list(command.footprint.keys),
        },
        "scope": {
            "target": _identity_payload(identity),
            "cwd": None if cwd is None else cwd.value,
            "root": _root_payload(root),
        },
        "runtime": dict(runtime),
    }


def _lineage_payload(lineage: PlannedLineage) -> Mapping[str, object]:
    """Encode every immutable target lineage field without Python objects."""
    return {
        "id": lineage.lineage_id.value,
        "initial": _file_payload(lineage.initial),
        "final": _file_payload(lineage.final),
        "source": (
            None if lineage.source_path is None else lineage.source_path.value
        ),
        "destination": (
            None
            if lineage.destination_path is None
            else lineage.destination_path.value
        ),
        "capabilities": sorted(item.value for item in lineage.capabilities),
        "matches": [
            {
                "kind": item.kind.value,
                "logical_start": item.span.logical_start,
                "logical_end": item.span.logical_end,
                "byte_start": item.span.byte_start,
                "byte_end": item.span.byte_end,
            }
            for item in lineage.matches
        ],
        "parents": [item.value for item in lineage.parent_paths],
        "mounts": list(lineage.mount_ids),
        "locks": [item.value for item in lineage.lock_footprint],
        "atomicity": lineage.atomicity_class,
        "steps": list(lineage.step_graph),
        "staging": lineage.staging_class,
        "diff": b64encode(lineage.diff_contribution).decode(),
        "parent_identities": [
            [
                None if path is None else path.value,
                identity[0],
                identity[1],
            ]
            for path, identity in lineage.parent_identities
        ],
    }


def _file_payload(value: PlannedFile) -> Mapping[str, object]:
    """Encode one complete expected or final regular-file fact."""
    content = value.bytes_value
    metadata = value.metadata
    return {
        "path": value.path.value,
        "present": value.present,
        "content_kind": (
            None
            if content is None
            else "source" if type(content) is SourceBytes else "proposed"
        ),
        "content": (
            None if content is None else b64encode(content._value).decode()
        ),
        "metadata": (
            None
            if metadata is None
            else {
                "mode": metadata.mode.value,
                "bom": metadata.has_utf8_bom,
                "newline": metadata.newline,
            }
        ),
        "digest": None if value.digest is None else value.digest.value,
        "size": value.size.value,
        "identity": None if value.identity is None else list(value.identity),
        "protected_metadata": (
            None
            if value.protected_metadata is None
            else value.protected_metadata.value
        ),
    }


def _report_from_payload(
    command: SealedCommitCommand, value: Mapping[str, object]
) -> WorkerReport:
    """Decode a child report while retaining exact command-derived IDs."""
    steps = value.get("steps")
    artifacts = value.get("artifacts")
    postcondition = value.get("postcondition")
    if (
        not isinstance(steps, list)
        or not isinstance(artifacts, list)
        or not isinstance(postcondition, str)
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    try:
        journal = SettlementJournal(
            tuple(
                JournalStep(
                    PatchStepId(item["id"]),
                    PatchLineageId(item["lineage"]),
                    CommitStepState(item["state"]),
                )
                for item in steps
                if isinstance(item, dict)
            ),
            tuple(
                ArtifactJournal(item["id"], ArtifactState(item["state"]))
                for item in artifacts
                if isinstance(item, dict)
            ),
            PostconditionState(postcondition),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TargetInspectionError(
            TargetErrorCode.WORKER_UNAVAILABLE
        ) from exc
    if tuple(item.identifier for item in journal.steps) != tuple(
        identifier for identifier, _ in _steps(command)
    ) or tuple(item.identifier for item in journal.artifacts) != _artifacts(
        command
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return WorkerReport(WorkerState.SETTLED, journal)


def _response_payload(
    line: bytes, token: bytes, request: Mapping[str, object]
) -> Mapping[str, object]:
    """Validate response MAC, request sequence, and complete identity echo."""
    try:
        envelope = loads(line)
        if not isinstance(envelope, dict) or set(envelope) != {
            "payload",
            "mac",
        }:
            raise ValueError
        payload = envelope["payload"]
        mac = envelope["mac"]
        if not isinstance(payload, dict) or not isinstance(mac, str):
            raise ValueError
        raw = dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        if not compare_digest(mac, digest(token, raw, "sha256").hex()):
            raise ValueError
        for field in (
            "version",
            "sequence",
            "receipt",
            "identity",
            "channel_id",
            "implementation_id",
        ):
            if payload.get(field) != request.get(field):
                raise ValueError
        error = payload.get("error")
        body = payload.get("body")
        if error is not None:
            raise TargetInspectionError(TargetErrorCode(error))
        if not isinstance(body, dict):
            raise ValueError
        return body
    except (TypeError, ValueError) as exc:
        raise TargetInspectionError(
            TargetErrorCode.WORKER_UNAVAILABLE
        ) from exc


def _seatbelt_runtime_profile(
    profile: SandboxRuntimeProfile, source_root: Path
) -> str:
    """Grant runtime code, one selected view, and private staging writes."""
    reads = (
        "/",
        "/System/Library/dyld",
        "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld",
        "/private/var/db/dyld",
        "/usr/lib/dyld",
        str(Path(base_executable)),
        str(Path(base_executable).resolve()),
        str(Path(base_executable).parent),
        str(Path(base_executable).resolve().parent.parent),
        str(source_root),
        "/opt/homebrew/opt/openssl@3/lib",
        str(profile._workspace_host_root),
        str(profile._private_namespace),
    )
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process*)",
        "(allow sysctl-read)",
        "(allow file-read-metadata)",
    ]
    for path in reads:
        candidate = Path(path)
        if not candidate.exists():
            continue
        escaped = _seatbelt_string(path)
        if path == "/":
            # macOS dyld requires a data read of the root directory before the
            # interpreter starts.  This is not a descendant grant; the live
            # child canary below proves arbitrary outside files remain denied.
            lines.append("(allow file-read-data (literal " + escaped + "))")
            continue
        lines.append("(allow file-read* (literal " + escaped + "))")
        if candidate.is_dir():
            lines.append("(allow file-read* (subpath " + escaped + "))")
    for path in (
        str(profile._workspace_host_root),
        str(profile._private_namespace),
    ):
        lines.append(
            "(allow file-write* (subpath " + _seatbelt_string(path) + "))"
        )
    lines.extend(("(deny network*)", "(deny process-fork)"))
    return "\n".join(lines) + "\n"


@dataclass(slots=True, repr=False)
class SandboxPatchRuntime:
    """Own one selected mutable sandbox session and all runtime handles."""

    profile: SandboxRuntimeProfile
    _profile_guard: SandboxRuntimeProfile = field(init=False, repr=False)
    _host_root: RootWitness = field(init=False, repr=False)
    _host_mount_binding: str = field(init=False, repr=False)
    _process: _SandboxRuntimeProcess = field(init=False, repr=False)
    _scope: ResolvedMutationScope | None = field(
        default=None, init=False, repr=False
    )
    _receipt: SandboxRuntimeReceipt | None = field(
        default=None, init=False, repr=False
    )
    _receipt_guard: SandboxRuntimeReceipt | None = field(
        default=None, init=False, repr=False
    )
    _endpoint: object | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the process owner without exposing a child channel."""
        self._profile_guard = replace(self.profile)
        self._host_root, self._host_mount_binding = (
            capture_rooted_root_binding(self.profile._workspace_host_root)
        )
        self._process = _SandboxRuntimeProcess(self.profile)

    def _validate_host_root_binding(self) -> None:
        """Reject a replaced host workspace before using its child view."""
        validate_rooted_root_binding(
            self.profile._workspace_host_root,
            self._host_root,
            self._host_mount_binding,
        )

    async def __aenter__(self) -> "SandboxPatchRuntime":
        """Retain this already-selected runtime for its loader lifetime."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Reap the worker and revoke all runtime handles on loader exit."""
        await self.close()

    async def resolve(
        self, selection: ScopeSelection
    ) -> ResolvedMutationScope:
        """Start and bind only the selected sandbox context."""
        if selection.context_kind is not ContextKind.SANDBOX:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        async with self._lock:
            if (
                self.profile != self._profile_guard
                or self._process.profile != self._profile_guard
            ):
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            self._validate_host_root_binding()
            if self._scope is not None:
                return self._scope
            (
                root,
                profile_receipt,
                session_id,
                primitive_receipts,
                attestation,
            ) = await self._process.start()
            try:
                self._validate_host_root_binding()
            except BaseException as stale_error:
                try:
                    await self._process.close()
                except CancelledError:
                    raise
                except BaseException as cleanup_error:
                    stale_error.add_note(
                        "runtime cleanup after stale host binding failed: "
                        + type(cleanup_error).__name__
                    )
                raise
            if (
                root.filesystem_id != self.profile.identity.filesystem_id
                or root.mount_id != self.profile.identity.mount_id
            ):
                await self._process.close()
                raise TargetInspectionError(TargetErrorCode.MOUNT_DENIED)
            worker = EphemeralWorkerWitness(
                self.profile.channel_id,
                self.profile.implementation_id
                + "-"
                + sha256(
                    (
                        profile_receipt
                        + attestation.runtime_command_digest
                        + attestation.child_process_identity
                    ).encode()
                ).hexdigest()[:16],
                "fence-"
                + sha256(
                    (
                        profile_receipt
                        + self.profile.identity.persistent_lease_id
                        + attestation.canary_receipt
                    ).encode()
                ).hexdigest()[:32],
            )
            scope = ResolvedMutationScope(
                ContextKind.SANDBOX,
                self.profile.identity,
                self.profile.cwd,
                self.profile.limits,
                frozenset(
                    (
                        Capability.READ_FOR_MUTATION,
                        Capability.OBSERVE_MUTATION_PRECONDITIONS,
                    )
                ),
                frozenset(
                    (
                        TargetPrimitive.ROOTED_CONTAINMENT,
                        TargetPrimitive.NOFOLLOW_INSPECTION,
                        TargetPrimitive.REGULAR_FILE_IDENTITY,
                        TargetPrimitive.BOUNDED_READ,
                    )
                ),
                root,
                worker,
                None,
                (),
            )
            self._scope = scope
            self._receipt = SandboxRuntimeReceipt(
                session_id,
                profile_receipt,
                root,
                worker,
                primitive_receipts,
                attestation.runtime_command_digest,
                attestation.backend_policy_digest,
                attestation.child_process_identity,
                attestation.canary_receipt,
            )
            self._receipt_guard = replace(self._receipt)
            return scope

    async def handshake(self, scope: ResolvedMutationScope) -> TargetHandshake:
        """Advertise only primitives proven by this selected runtime."""
        receipt = await self._require_scope(scope)
        primitives = (
            scope.primitives
            | _FUTURE_MUTATION_PRIMITIVES
            | _SANDBOX_PRIMITIVES
        )
        probes = tuple(
            PrimitiveProbe(
                item,
                ProbeState.AVAILABLE,
                receipt.primitive_receipts[item],
            )
            for item in sorted(
                _FUTURE_MUTATION_PRIMITIVES | _SANDBOX_PRIMITIVES,
                key=lambda item: item.value,
            )
        )
        return TargetHandshake(
            self.profile.identity,
            primitives,
            (),
            probes,
            (
                LocalPlatformProfile.DARWIN
                if self.profile.execution_plan.settings.backend
                is SandboxBackend.SEATBELT
                else LocalPlatformProfile.LINUX
            ),
            worker=scope.worker,
        )

    async def inspect(self, request: InspectionRequest) -> InspectionBatch:
        """Read only from the established runtime view."""
        await self._require_scope(request.scope)
        assert request.scope.root_witness is not None
        snapshots = await self._process.inspect(
            request.paths, request.scope.root_witness
        )
        return InspectionBatch(snapshots)

    async def worker(
        self, scope: ResolvedMutationScope
    ) -> RootedSandboxCommitWorker:
        """Mint an opaque worker that routes to this context-owned runtime."""
        await self.handshake(scope)
        endpoint = self._endpoint
        if type(endpoint) is not _SandboxEndpoint:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        return _sandbox_worker_for_endpoint(_rooted_sandbox_endpoint(endpoint))

    def _bind_sandbox_endpoint(
        self, scope: ResolvedMutationScope
    ) -> "_SandboxEndpoint":
        """Return the endpoint later sealed by issued SDK authority."""
        if (
            scope is not self._scope
            or self._receipt is None
            or self._receipt != self._receipt_guard
            or self.profile != self._profile_guard
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        endpoint = self._endpoint
        if endpoint is None:
            endpoint = _SandboxEndpoint(self, scope)
            self._endpoint = endpoint
        if type(endpoint) is not _SandboxEndpoint:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        return endpoint

    async def close(self) -> None:
        """Cancel/reap the child and fence all existing context handles."""
        async with self._lock:
            await self._process.close()
            self._scope = None
            self._receipt = None
            self._receipt_guard = None
            self._endpoint = None

    async def _require_scope(
        self, scope: ResolvedMutationScope
    ) -> SandboxRuntimeReceipt:
        """Reject copied, stale, retargeted, or post-close scope values."""
        async with self._lock:
            receipt = self._receipt
            if (
                self.profile != self._profile_guard
                or self._process.profile != self._profile_guard
                or scope is not self._scope
                or receipt is None
                or receipt != self._receipt_guard
                or scope.worker is not receipt.worker
                or scope.root_witness != receipt.root
            ):
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            self._validate_host_root_binding()
            return receipt


class _SelectedPatchRuntimeProfile(Protocol):
    """Expose the identities shared by selected mutation runtimes."""

    @property
    def identity(self) -> TargetIdentity:
        """Return the immutable target identity."""

    @property
    def channel_id(self) -> SandboxChannelId:
        """Return the authenticated worker channel identity."""

    @property
    def implementation_id(self) -> SandboxWorkerImplementationId:
        """Return the immutable worker implementation identity."""


class _SelectedPatchRuntimeReceipt(Protocol):
    """Expose the live witnesses needed for fenced settlement."""

    @property
    def session_id(self) -> SandboxSessionId:
        """Return the live worker session identity."""

    @property
    def root(self) -> RootWitness:
        """Return the rooted filesystem witness."""

    @property
    def worker(self) -> EphemeralWorkerWitness:
        """Return the ephemeral fenced worker witness."""


class _SelectedPatchWorkerProcess(Protocol):
    """Expose only the sealed worker operations shared by runtimes."""

    @property
    def _implementation_digest_value(self) -> str | None:
        """Return the live immutable implementation digest."""

    async def commit(
        self,
        command: SealedCommitCommand,
        validator: RootedCommandAuthorityValidator,
    ) -> WorkerReport:
        """Commit one already-authorized command."""


class _SelectedPatchRuntime(Protocol):
    """Describe the opaque runtime contract consumed by durable patching."""

    @property
    def profile(self) -> _SelectedPatchRuntimeProfile:
        """Return the immutable selected runtime profile."""

    @property
    def _process(self) -> _SelectedPatchWorkerProcess:
        """Return the private sealed worker process."""

    async def resolve(
        self, selection: ScopeSelection
    ) -> ResolvedMutationScope:
        """Resolve the selected mutation scope."""

    async def worker(
        self, scope: ResolvedMutationScope
    ) -> RootedSandboxCommitWorker:
        """Return the sealed commit worker."""

    async def close(self) -> None:
        """Fence the active runtime."""

    async def _require_scope(
        self, scope: ResolvedMutationScope
    ) -> _SelectedPatchRuntimeReceipt:
        """Return the live scope receipt."""

    def _bind_sandbox_endpoint(
        self, scope: ResolvedMutationScope
    ) -> "_SandboxEndpoint":
        """Return the issued private endpoint."""


class _SelectedPatchInspectionTarget(Protocol):
    """Inspect the selected runtime view without commit authority."""

    async def inspect(self, request: InspectionRequest) -> InspectionBatch:
        """Inspect the bounded request paths."""


@dataclass(frozen=True, slots=True, repr=False)
class SandboxScopeResolver:
    """Resolve only a runtime-owned sandbox scope."""

    runtime: SandboxPatchRuntime

    async def resolve(
        self, selection: ScopeSelection
    ) -> ResolvedMutationScope:
        """Return the selected immutable sandbox scope."""
        return await self.runtime.resolve(selection)


@dataclass(frozen=True, slots=True, repr=False)
class SandboxInspectionTarget:
    """Expose rooted inspection through the selected runtime view."""

    runtime: SandboxPatchRuntime

    async def handshake(self, scope: ResolvedMutationScope) -> TargetHandshake:
        """Return the selected runtime's live handshake."""
        return await self.runtime.handshake(scope)

    async def inspect(self, request: InspectionRequest) -> InspectionBatch:
        """Inspect only through the selected runtime view."""
        return await self.runtime.inspect(request)


@dataclass(frozen=True, slots=True, repr=False)
class SandboxCommitTarget:
    """Expose an opaque commit worker owned by the sandbox runtime."""

    runtime: SandboxPatchRuntime

    async def handshake(self, scope: ResolvedMutationScope) -> TargetHandshake:
        """Return the selected runtime's live mutation handshake."""
        return await self.runtime.handshake(scope)

    async def worker(
        self, scope: ResolvedMutationScope
    ) -> RootedSandboxCommitWorker:
        """Return a context-owned opaque worker capability."""
        return await self.runtime.worker(scope)


@dataclass(eq=False, slots=True, repr=False, weakref_slot=True)
class _SandboxEndpoint:
    """Keep the runtime and scope behind the coordinator-private endpoint."""

    runtime: _SelectedPatchRuntime
    scope: ResolvedMutationScope
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _active_request: PatchRequestId | None = field(
        default=None, init=False, repr=False
    )
    _active_task: Task[WorkerReport] | None = field(
        default=None, init=False, repr=False
    )
    _settlements: dict[PatchRequestId, WorkerReport] = field(
        default_factory=dict, init=False, repr=False
    )

    async def commit_sandbox(
        self,
        command: SealedCommitCommand,
        validator: RootedCommandAuthorityValidator,
    ) -> WorkerReport:
        """Execute one coordinator-owned sealed command."""
        await self.runtime._require_scope(self.scope)
        if (
            command.plan.binding.context_kind is not self.scope.context_kind
            or command.plan.binding.target != self.runtime.profile.identity
            or command.plan.binding.cwd != self.scope.cwd
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        async with self._lock:
            prior = self._settlements.get(command.lease.request_id)
            if prior is not None:
                return prior
            if self._active_task is not None:
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            self._active_request = command.lease.request_id
            task = create_task(
                self.runtime._process.commit(command, validator)
            )
            self._active_task = task
        try:
            report = await shield(task)
            async with self._lock:
                self._settlements[command.lease.request_id] = report
            return report
        finally:
            async with self._lock:
                if self._active_task is task and task.done():
                    self._active_request = None
                    self._active_task = None

    async def reconcile_sandbox(
        self, request_id: PatchRequestId
    ) -> WorkerReport:
        """Read retained settlement truth without command replay."""
        await self.runtime._require_scope(self.scope)
        async with self._lock:
            settled = self._settlements.get(request_id)
            task = self._active_task
            if settled is not None:
                return settled
            if self._active_request != request_id or task is None:
                return WorkerReport(WorkerState.LIVE, None)
        if not task.done():
            return WorkerReport(WorkerState.LIVE, None)
        try:
            report = task.result()
        except (CancelledError, Exception):
            return WorkerReport(WorkerState.LIVE, None)
        async with self._lock:
            self._settlements[request_id] = report
            if self._active_task is task:
                self._active_request = None
                self._active_task = None
        return report


@dataclass(frozen=True, slots=True, repr=False)
class _SandboxDurableCommandAuthority(RootedCommandAuthorityValidator):
    """Bind one fieldless worker dispatch to the live durable owner record."""

    runtime: _SelectedPatchRuntime
    scope: ResolvedMutationScope
    lease: DurableCommitLease
    store: DurablePatchStore
    clock: ApprovalClock

    async def is_rooted_command_current(
        self, command: SealedCommitCommand
    ) -> bool:
        """Require matching runtime identity and an unexpired durable fence."""
        profile = self.runtime.profile
        if (
            command.lease.request_id != self.lease.request_id
            or command.lease.domain_id != self.lease.domain_id
            or command.lease.fence != self.lease.fence.value
            or command.plan.binding.context_kind is not self.scope.context_kind
            or command.plan.binding.target != profile.identity
            or command.plan.binding.cwd != self.scope.cwd
            or self.scope.identity != profile.identity
            or self.scope.root_witness is None
            or self.scope.worker is None
        ):
            return False
        try:
            receipt = await self.runtime._require_scope(self.scope)
        except TargetInspectionError:
            return False
        if (
            receipt.root != self.scope.root_witness
            or receipt.worker is not self.scope.worker
            or not receipt.session_id
            or not profile.channel_id
            or not profile.implementation_id
            or profile.identity.filesystem_id == ""
            or profile.identity.mount_id == ""
            or profile.identity.persistent_lease_id == ""
        ):
            return False
        try:
            return await self.store.is_current_fence(
                self.lease, await self.clock.now()
            )
        except DurableStoreError:
            return False


def _pending_request(
    correlation_id: PatchObserverCorrelationId,
    duration: DurationTicks,
) -> DurablePendingRequest:
    """Create one validated host-pending identity before worker startup."""
    return DurablePendingRequest(
        PatchPendingOperationId.new(), correlation_id, duration
    )


@dataclass(frozen=True, slots=True, repr=False)
class SandboxPatchServiceConfiguration:
    """Bind host-owned policy, planning, review, and lease services."""

    subject: ExecutionSubject
    planner: PlannerFacade
    approvals: ApprovalService
    approval_issuer: PhaseFiveDurableApprovalIssuer
    clock: ApprovalClock
    review_duration: DurationTicks
    lease_duration: DurationTicks
    execution_id: PatchExecutionId | None = None
    input_limits: PatchInputLimits = PatchInputLimits()
    pending_factory: Callable[
        [PatchObserverCorrelationId, DurationTicks], DurablePendingRequest
    ] = _pending_request

    def __post_init__(self) -> None:
        """Require concrete trusted services rather than test callbacks."""
        if (
            type(self.subject) is not ExecutionSubject
            or type(self.planner) is not PlannerFacade
            or type(self.approvals) is not ApprovalService
            or type(self.approval_issuer) is not PhaseFiveDurableApprovalIssuer
            or not callable(getattr(self.clock, "now", None))
            or type(self.review_duration) is not DurationTicks
            or type(self.lease_duration) is not DurationTicks
            or self.execution_id is not None
            and type(self.execution_id) is not PatchExecutionId
            or type(self.input_limits) is not PatchInputLimits
            or not callable(self.pending_factory)
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


@dataclass(frozen=True, slots=True, repr=False)
class SandboxPatchServiceFactory:
    """Construct the production sandbox SDK service from durable bindings."""

    configuration: SandboxPatchServiceConfiguration

    async def bind_sandbox(
        self,
        runtime: SandboxPatchRuntime,
        scope: ResolvedMutationScope,
        handshake: TargetHandshake,
        target: SandboxCommitTarget,
        inspection: SandboxInspectionTarget,
        store: DurablePatchStore,
        policy: TrustedPatchPolicy,
    ) -> "SandboxPatchSdkService":
        """Return the complete runtime/policy/durable-store service."""
        del target
        return SandboxPatchSdkService(
            runtime,
            scope,
            handshake,
            inspection,
            store,
            policy,
            self.configuration,
        )


@dataclass(slots=True, repr=False)
class _SandboxSettlementPort:
    """Read only durable outcomes retained by one SDK service."""

    service: "SandboxPatchSdkService"

    def inspect(
        self, handle: PatchInvocationHandle
    ) -> Future[PatchInvocationOutcome]:
        """Return only the exact durable observation for one issued handle."""
        bound = _bound_invocation_subscription_access(handle, self.service)
        if (
            not isinstance(bound, tuple)
            or len(bound) != 2
            or type(bound[0]) is not PatchRequestId
            or type(bound[1]) is not PatchObserverCorrelationId
        ):
            return self.service._inspection_error_future()
        return self.service._inspect_request_future(bound[0], bound[1])

    def await_terminal(
        self,
        handle: PatchInvocationHandle,
        pending: PatchPending,
    ) -> Future[PatchResult]:
        """Await exactly the persisted pending branch."""
        del handle
        return self.service._terminal_future(pending)


@runtime_checkable
class PatchActivationObserver(Protocol):
    """Observe exact durable ownership transitions for one active host."""

    async def bind_durable_commit(self, lease: DurableCommitLease) -> None:
        """Bind one coordinator-issued owner before an effect."""

    async def retain_durable_commit(self, lease: DurableCommitLease) -> None:
        """Retain one pending or partial coordinator owner."""

    async def release_durable_commit(self, lease: DurableCommitLease) -> None:
        """Release one terminal coordinator owner."""


@dataclass(frozen=True, slots=True, repr=False)
class _SandboxRequestAccess:
    """Keep durable read authority behind the service's issued SDK handle."""

    access: DurableRequestAccess
    correlation_id: PatchObserverCorrelationId


async def _bounded_task_join(
    task: Task[object], *, cancel: bool = False
) -> None:
    """Observe one task within the runtime teardown deadline."""
    if cancel and not task.done():
        task.cancel()
    try:
        await wait_for(shield(task), timeout=_PROCESS_REAP_SECONDS)
    except TimeoutError:
        if not task.done():
            task.cancel()
        try:
            await wait_for(shield(task), timeout=_PROCESS_REAP_SECONDS)
        except (CancelledError, TimeoutError):
            pass
    except (CancelledError, Exception):
        pass


@dataclass(slots=True, repr=False)
class SandboxPatchSdkService:
    """Execute the patch lifecycle through one sandbox durable domain."""

    runtime: _SelectedPatchRuntime
    scope: ResolvedMutationScope
    handshake: TargetHandshake
    inspection: _SelectedPatchInspectionTarget
    store: DurablePatchStore
    policy: TrustedPatchPolicy
    configuration: SandboxPatchServiceConfiguration
    _settlement: _SandboxSettlementPort = field(init=False, repr=False)
    _latest: PatchInvocationOutcome | None = field(
        default=None, init=False, repr=False
    )
    _pending: dict[PatchPendingOperationId, DurablePendingAccess] = field(
        default_factory=dict, init=False, repr=False
    )
    _requests: dict[PatchRequestId, _SandboxRequestAccess] = field(
        default_factory=dict, init=False, repr=False
    )
    _workers: dict[
        PatchRequestId, tuple[DurableCommitLease, DurableWorkerBinding]
    ] = field(default_factory=dict, init=False, repr=False)
    _worker_tasks: dict[PatchRequestId, Task[WorkerReport]] = field(
        default_factory=dict, init=False, repr=False
    )
    _reconciliation_tasks: set[Task[None]] = field(
        default_factory=set, init=False, repr=False
    )
    _reader_tasks: set[Task[None]] = field(
        default_factory=set, init=False, repr=False
    )
    _protocol_claimed: set[PatchRequestId] = field(
        default_factory=set, init=False, repr=False
    )
    _protocol_claim_waiters: dict[PatchRequestId, Future[None]] = field(
        default_factory=dict, init=False, repr=False
    )
    _activation_observer: PatchActivationObserver | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Create the service-owned durable settlement port."""
        self._settlement = _SandboxSettlementPort(self)

    async def _await_protocol_claim(self, request_id: PatchRequestId) -> None:
        """Await protocol-owned durable claim before any target effect."""
        if type(request_id) is not PatchRequestId:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        if request_id in self._protocol_claimed:
            return
        waiter = self._protocol_claim_waiters.get(request_id)
        if waiter is None:
            waiter = get_running_loop().create_future()
            self._protocol_claim_waiters[request_id] = waiter
        await shield(waiter)

    def _signal_protocol_claim(self, request_id: PatchRequestId) -> None:
        """Release protocol staging after owner and fence are durable."""
        self._protocol_claimed.add(request_id)
        waiter = self._protocol_claim_waiters.get(request_id)
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    def _fail_protocol_claim(self, request_id: PatchRequestId) -> None:
        """Release protocol staging when no durable claim can be observed."""
        waiter = self._protocol_claim_waiters.get(request_id)
        if waiter is not None and not waiter.done():
            waiter.set_exception(
                TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            )

    def _patch_sandbox_endpoint(self) -> object:
        """Return the runtime endpoint for sealed SDK authority issuance."""
        if self.scope.context_kind is not ContextKind.SANDBOX:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        return self.runtime._bind_sandbox_endpoint(self.scope)

    def _patch_container_endpoint(self) -> object:
        """Return the container endpoint for sealed SDK authority issuance."""
        if self.scope.context_kind is not ContextKind.CONTAINER:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        return self.runtime._bind_sandbox_endpoint(self.scope)

    def set_activation_observer(
        self, observer: PatchActivationObserver
    ) -> None:
        """Attach one loader-owned durable activation observer to this host."""
        if self._activation_observer is not None or not isinstance(
            observer, PatchActivationObserver
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        self._activation_observer = observer

    async def _bind_activation_commit(self, lease: DurableCommitLease) -> None:
        """Bind a real durable owner before starting its worker."""
        observer = self._activation_observer
        if observer is None:
            return
        await observer.bind_durable_commit(lease)

    async def _retain_activation_commit(
        self, lease: DurableCommitLease
    ) -> None:
        """Retain one actual pending durable owner after host suspension."""
        observer = self._activation_observer
        if observer is None:
            return
        await observer.retain_durable_commit(lease)

    async def _release_activation_commit(
        self, lease: DurableCommitLease
    ) -> None:
        """Release one actual terminal durable owner after settlement."""
        observer = self._activation_observer
        if observer is None:
            return
        await observer.release_durable_commit(lease)

    async def __aenter__(self) -> "SandboxPatchSdkService":
        """Own the selected runtime for the loaded toolset lifetime."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Reap live work before durably releasing replacement fencing."""
        del exc_type, exc_value, traceback
        close_error: BaseException | None = None
        try:
            await self.runtime.close()
        except BaseException as error:
            close_error = error
        for task in tuple(self._worker_tasks.values()):
            await _bounded_task_join(task)
        if close_error is None:
            for lease, binding in tuple(self._workers.values()):
                await self.store.mark_worker_reaped(lease, binding)
        for reader_task in (
            *tuple(self._reconciliation_tasks),
            *tuple(self._reader_tasks),
        ):
            await _bounded_task_join(reader_task, cancel=True)
        if close_error is not None:
            raise close_error

    @property
    def settlement(self) -> PatchSettlementPort:
        """Return the durable settlement port used by public tool callers."""
        return self._settlement

    async def invoke(
        self,
        operation: OperationType,
        raw_arguments: bytes,
        capability: PatchInvocationCapability,
        request_id: PatchRequestId,
        correlation_id: "PatchObserverCorrelationId",
        *,
        identity: DurableRequestIdentity | None = None,
        origin: DurableProtocolOrigin | None = None,
    ) -> PatchInvocationOutcome:
        """Parse, plan, approve, and durably execute one selected request."""
        del capability
        canonical = _canonical_request(
            operation,
            raw_arguments,
            correlation_id,
            self.configuration.input_limits,
        )
        expected_identity = DurableRequestIdentity(
            self.configuration.subject.tenant,
            self.configuration.subject.principal,
            _execution_id_for_request(request_id),
            self.policy.approval.route,
            RetransmissionKey("sandbox-" + request_id.value),
        )
        if identity is None:
            identity = expected_identity
        elif (
            type(identity) is not DurableRequestIdentity
            or identity.tenant_id != expected_identity.tenant_id
            or identity.principal_id != expected_identity.principal_id
            or identity.route_id != expected_identity.route_id
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        if origin is not None:
            self._validate_protocol_origin(origin, identity)
        execution_id = identity.execution_id
        access = DurableRequestAccess(request_id, identity)
        reservation = await self.store.reserve(
            identity, canonical.digest, request_id
        )
        existing = await self.store.inspect(access)
        if (
            origin is not None
            and existing.plan is not None
            and existing.plan.origin != origin
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        if existing.terminal is not None:
            terminal = existing.terminal
            if terminal.outbox.correlation_id != correlation_id:
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            self._requests[request_id] = _SandboxRequestAccess(
                access, terminal.outbox.correlation_id
            )
            self._latest = terminal.result
            return terminal.result
        if getattr(existing, "lifecycle", None) is LifecyclePhase.PLANNED:
            stored_plan = existing.plan
            if type(stored_plan) is not DurablePlanReference or origin is None:
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            self._requests[request_id] = _SandboxRequestAccess(
                access, correlation_id
            )
            try:
                restored = (
                    self.configuration.approval_issuer.open_plan_material(
                        identity,
                        origin,
                        request_id,
                        stored_plan,
                    )
                )
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise TargetInspectionError(
                    TargetErrorCode.WITNESS_STALE
                ) from None
            return await self._review_and_commit(
                reservation,
                request_id,
                restored,
                stored_plan,
                identity,
                correlation_id,
                restored.binding.final.approval,
            )
        if existing.plan is not None:
            self._requests[request_id] = _SandboxRequestAccess(
                access, correlation_id
            )
            outcome = await self._attached_outcome(
                request_id,
                identity,
                correlation_id,
            )
            self._latest = outcome
            return outcome
        self._requests[request_id] = _SandboxRequestAccess(
            access, correlation_id
        )
        paths, effects = _semantic_paths(canonical)
        authorizer = PolicyAuthorizer(self.policy)
        preflight = await authorizer.authorize_preinspection(
            PreflightRequest(
                operation,
                paths,
                effects,
                frozenset(paths),
                compose_limits(
                    self.scope.limits,
                    self.scope.limits,
                    self.scope.limits,
                    self.scope.limits,
                    self.scope.limits,
                ),
            )
        )
        observed = await self.inspection.inspect(
            InspectionRequest(self.scope, paths)
        )
        candidate = await self.configuration.planner.plan(
            canonical, observed.planner_workspace()
        )
        final = await authorizer.authorize_final(
            preflight, candidate, self.handshake
        )
        planning_now = await self.configuration.clock.now()
        request = _patch_request(
            request_id, execution_id, operation, raw_arguments, paths
        )
        sealed_plan = seal_plan(
            _plan_id_for_request(request_id),
            PlanBinding(
                request,
                candidate.request_digest,
                self.configuration.subject,
                self.scope.context_kind,
                self.runtime.profile.identity,
                self.scope.cwd,
                preflight,
                final,
            ),
            candidate,
            ExpiryTick(
                planning_now.value + self.configuration.review_duration.value
            ),
        )
        rehydration = b""
        if origin is not None:
            rehydration = (
                self.configuration.approval_issuer.seal_plan_material(
                    identity, origin, sealed_plan
                )
            )
        durable_plan = (
            _durable_plan(sealed_plan)
            if origin is None
            else _durable_plan(sealed_plan, origin, rehydration)
        )
        assert existing.plan is None
        await self.store.persist_plan(reservation, durable_plan)
        return await self._review_and_commit(
            reservation,
            request_id,
            sealed_plan,
            durable_plan,
            identity,
            correlation_id,
            final.approval,
        )

    async def _review_and_commit(
        self,
        reservation: DurableReservation,
        request_id: PatchRequestId,
        plan: SealedPlan,
        durable_plan: DurablePlanReference,
        identity: DurableRequestIdentity,
        correlation_id: PatchObserverCorrelationId,
        requirements: ApprovalRequirements,
    ) -> PatchInvocationOutcome:
        """Review one persisted sealed plan and claim its lone effect owner."""
        access = DurableRequestAccess(request_id, identity)
        reviewed = await self.configuration.approvals.await_review(
            PlanReviewRequest(
                plan,
                self.configuration.subject,
                requirements,
            )
        )
        if reviewed.state is not ApprovalDecisionState.APPROVED:
            result = _approval_result(request_id, plan.plan_id, reviewed.state)
            self._latest = result
            return result
        assert reviewed.grant is not None
        approval = await self.configuration.approval_issuer.issue(
            identity,
            durable_plan,
            reviewed.grant,
            plan,
            self.configuration.subject,
        )
        artifacts = _durable_artifacts(plan)
        now = await self.configuration.clock.now()
        claim = await self.store.claim_commit(
            reservation,
            durable_plan,
            approval,
            PatchCommitOwnerId.new(),
            now,
            self.configuration.lease_duration,
            tuple(item[1] for item in artifacts),
        )
        if claim.state is DurableCommitClaimState.TERMINAL:
            assert claim.terminal is not None
            self._signal_protocol_claim(request_id)
            self._latest = claim.terminal.result
            return claim.terminal.result
        if claim.state is DurableCommitClaimState.ATTACHED:
            self._signal_protocol_claim(request_id)
            outcome = await self._attached_outcome(
                request_id, identity, correlation_id
            )
            self._latest = outcome
            return outcome
        assert claim.lease is not None
        pending_request: DurablePendingRequest | None = None
        worker_binding: DurableWorkerBinding | None = None
        worker_bound = False
        activation_bound = False
        task: Task[WorkerReport] | None = None
        try:
            await self._bind_activation_commit(claim.lease)
            activation_bound = True
            candidate_pending = self.configuration.pending_factory(
                correlation_id,
                self.configuration.lease_duration,
            )
            if type(candidate_pending) is not DurablePendingRequest:
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            pending_request = candidate_pending
            worker_binding = await self._worker_binding()
            await self.store.bind_worker(claim.lease, worker_binding, now)
            worker_bound = True
            self._workers[request_id] = (claim.lease, worker_binding)
            command = SealedCommitCommand(
                plan,
                CommitLease(
                    claim.lease.domain_id,
                    request_id,
                    claim.lease.fence.value,
                ),
                footprint_for(plan),
            )
            worker = await self.runtime.worker(self.scope)
            await _issue_rooted_command_authority_for_validator(
                command,
                _SandboxDurableCommandAuthority(
                    self.runtime,
                    self.scope,
                    claim.lease,
                    self.store,
                    self.configuration.clock,
                ),
            )
            task = create_task(worker.commit(command))
            self._worker_tasks[request_id] = task
            self._signal_protocol_claim(request_id)
            report = await shield(task)
        except CancelledError:
            assert pending_request is not None
            await self.store.request_cancellation(access)
            if task is None:
                await self._reap_bound_worker(
                    claim.lease, worker_binding, worker_bound
                )
            outcome = await self._suspend_worker(
                request_id,
                identity,
                plan.plan_id,
                claim.lease,
                correlation_id,
                now,
                pending_request,
            )
            if task is not None and worker_binding is not None:
                background = create_task(
                    self._finish_worker(
                        task,
                        request_id,
                        identity,
                        plan.plan_id,
                        claim.lease,
                        correlation_id,
                        artifacts,
                        worker_binding,
                    )
                )
                self._reconciliation_tasks.add(background)
                background.add_done_callback(
                    self._reconciliation_tasks.discard
                )
        except BaseException:
            self._fail_protocol_claim(request_id)
            await self._reap_bound_worker(
                claim.lease, worker_binding, worker_bound
            )
            if pending_request is None:
                if activation_bound:
                    await self._release_activation_commit(claim.lease)
                raise
            outcome = await self._suspend_worker(
                request_id,
                identity,
                plan.plan_id,
                claim.lease,
                correlation_id,
                now,
                pending_request,
            )
            if task is not None and task.done():
                self._worker_tasks.pop(request_id, None)
        else:
            outcome = await self._reconcile_worker(
                request_id,
                identity,
                plan.plan_id,
                claim.lease,
                report,
                correlation_id,
                now,
                pending_request,
                artifacts,
            )
            self._worker_tasks.pop(request_id, None)
        if isinstance(outcome, PatchPending):
            self._pending[outcome.pending_operation_id] = DurablePendingAccess(
                DurableRequestAccess(request_id, identity),
                outcome.pending_operation_id,
                correlation_id,
            )
            await self._retain_activation_commit(claim.lease)
        elif activation_bound:
            await self._release_activation_commit(claim.lease)
        self._latest = outcome
        return outcome

    async def invoke_remote(
        self,
        operation: OperationType,
        raw_arguments: bytes,
        capability: PatchInvocationCapability,
        request_id: PatchRequestId,
        correlation_id: "PatchObserverCorrelationId",
        identity: DurableRequestIdentity,
        origin: DurableProtocolOrigin | None = None,
    ) -> PatchInvocationOutcome:
        """Invoke only a server-derived retransmission identity.

        The normal ``invoke`` path derives its local SDK retransmission key.
        This transport-specific path is used exclusively by the authenticated
        loopback test server after it has reserved the exact durable tuple.
        It still verifies tenant, principal, execution, and route against the
        service's trusted runtime configuration before any target inspection.
        """
        if type(identity) is not DurableRequestIdentity:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        return await self.invoke(
            operation,
            raw_arguments,
            capability,
            request_id,
            correlation_id,
            identity=identity,
            origin=origin,
        )

    def _validate_protocol_origin(
        self,
        origin: DurableProtocolOrigin,
        identity: DurableRequestIdentity,
    ) -> None:
        """Reject an origin that differs from this fixed service subject."""
        subject = self.configuration.subject
        target = self.runtime.profile.identity
        if (
            type(origin) is not DurableProtocolOrigin
            or type(identity) is not DurableRequestIdentity
            or not origin.matches(identity)
            or origin.tenant_id != subject.tenant
            or origin.principal_id != subject.principal
            or self.configuration.execution_id != origin.execution_id
            or origin.run_id != subject.run
            or origin.session_id != subject.session
            or origin.task_id != subject.task
            or origin.agent_id != subject.agent
            or origin.route_id != self.policy.approval.route
            or origin.context_id != target.context_id
            or origin.workspace_id != target.workspace_id
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)

    async def _reap_bound_worker(
        self,
        lease: DurableCommitLease,
        binding: DurableWorkerBinding | None,
        durably_bound: bool,
    ) -> None:
        """Reap and durably release one exact bound child after failure."""
        await self.runtime.close()
        if durably_bound:
            assert binding is not None
            await self.store.mark_worker_reaped(lease, binding)
        else:
            await self.store.mark_worker_absent(lease)

    async def _worker_binding(self) -> DurableWorkerBinding:
        """Derive durable non-secret child identity from the live receipt."""
        receipt = await self.runtime._require_scope(self.scope)
        implementation_digest = (
            self.runtime._process._implementation_digest_value
        )
        if implementation_digest is None:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        root_bytes = dumps(
            _root_payload(receipt.root), separators=(",", ":"), sort_keys=True
        ).encode()
        return DurableWorkerBinding(
            receipt.session_id,
            self.runtime.profile.channel_id,
            self.runtime.profile.implementation_id,
            AlgorithmDigest("sha256", implementation_digest),
            AlgorithmDigest.from_bytes(root_bytes),
        )

    async def _suspend_worker(
        self,
        request_id: PatchRequestId,
        identity: DurableRequestIdentity,
        plan_id: PatchPlanId,
        lease: DurableCommitLease,
        correlation_id: PatchObserverCorrelationId,
        now: ExpiryTick,
        pending: DurablePendingRequest,
    ) -> PatchPending:
        """Persist worker loss or cancellation without guessing truth."""
        report = WorkerReport(WorkerState.LIVE, None)
        outcome = await self._reconcile_worker(
            request_id,
            identity,
            plan_id,
            lease,
            report,
            correlation_id,
            now,
            pending,
            (),
        )
        assert isinstance(outcome, PatchPending)
        return outcome

    async def _finish_worker(
        self,
        task: Task[WorkerReport],
        request_id: PatchRequestId,
        identity: DurableRequestIdentity,
        plan_id: PatchPlanId,
        lease: DurableCommitLease,
        correlation_id: PatchObserverCorrelationId,
        artifacts: tuple[tuple[str, PatchArtifactId], ...],
        binding: DurableWorkerBinding,
    ) -> None:
        """Reconcile a shielded worker after its caller has disconnected."""
        try:
            try:
                report = await task
            except BaseException:
                await self.runtime.close()
                await self.store.mark_worker_reaped(lease, binding)
                return
            now = await self.configuration.clock.now()
            outcome = await self._reconcile_worker(
                request_id,
                identity,
                plan_id,
                lease,
                report,
                correlation_id,
                now,
                None,
                artifacts,
            )
            if not isinstance(outcome, PatchPending):
                await self._release_activation_commit(lease)
        finally:
            self._worker_tasks.pop(request_id, None)

    async def _reconcile_worker(
        self,
        request_id: PatchRequestId,
        identity: DurableRequestIdentity,
        plan_id: PatchPlanId,
        lease: DurableCommitLease,
        report: WorkerReport,
        correlation_id: PatchObserverCorrelationId,
        now: ExpiryTick,
        pending: DurablePendingRequest | None,
        artifacts: tuple[tuple[str, PatchArtifactId], ...],
    ) -> PatchInvocationOutcome:
        """Persist exact worker journal or one durable pending branch."""
        result = _worker_result(request_id, plan_id, report)
        return await DurablePatchReconciler(self.store).reconcile(
            DurableRequestAccess(request_id, identity),
            lease,
            report,
            result,
            correlation_id,
            now,
            pending=pending,
            artifacts=_durable_observations(report, artifacts),
        )

    async def review(
        self, handle: PatchInvocationHandle
    ) -> Mapping[str, object]:
        """Return a bounded host review projection for the active request."""
        del handle
        return {"kind": "sandbox_patch_review"}

    async def approve(
        self, handle: PatchInvocationHandle
    ) -> PatchInvocationOutcome:
        """Return the durable outcome after broker approval has completed."""
        del handle
        return await self._inspect_latest()

    async def subscribe(
        self, handle: PatchInvocationHandle
    ) -> AsyncIterator[PatchLifecycleEvent]:
        """Yield exact durable-outbox records for one sealed SDK request."""
        bound = _bound_invocation_subscription_access(handle, self)
        if (
            not isinstance(bound, tuple)
            or len(bound) != 2
            or type(bound[0]) is not PatchRequestId
            or type(bound[1]) is not PatchObserverCorrelationId
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        request_id, correlation_id = bound
        request = self._requests.get(request_id)
        if request is None or request.correlation_id is not correlation_id:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        cursor = SequenceNumber(0)
        observer_id = PatchObserverId.new()
        while True:
            records = await self.store.outbox(request.access, cursor, 1024)
            for record in records:
                if (
                    record.request_id != request_id
                    or record.correlation_id is not correlation_id
                    or record.sequence.value <= cursor.value
                ):
                    raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
                cursor = record.sequence
                yield PatchLifecycleEvent(
                    1,
                    record.event_id,
                    observer_id,
                    record.correlation_id,
                    record.request_id,
                    record.sequence,
                    record.lifecycle,
                )
            snapshot = await self.store.inspect(request.access)
            terminal = snapshot.terminal
            if terminal is not None:
                if terminal.outbox.sequence.value <= cursor.value:
                    return
                continue
            pending = snapshot.pending
            if (
                pending is None
                or pending.request_id != request_id
                or pending.correlation_id is not correlation_id
            ):
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            await self.store.await_terminal(
                DurablePendingAccess(
                    request.access,
                    pending.pending_operation_id,
                    pending.correlation_id,
                )
            )

    async def _inspect_latest(self) -> PatchInvocationOutcome:
        """Return known durable truth without a new mutation dispatch."""
        if self._latest is None:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        return self._latest

    def _inspection_error_future(self) -> Future[PatchInvocationOutcome]:
        """Return one failed future for an unissued request observation."""
        future: Future[PatchInvocationOutcome] = (
            get_running_loop().create_future()
        )
        future.set_exception(
            TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        )
        return future

    def _inspect_request_future(
        self,
        request_id: PatchRequestId,
        correlation_id: PatchObserverCorrelationId,
    ) -> Future[PatchInvocationOutcome]:
        """Return one future for the issued request's durable observation."""
        future: Future[PatchInvocationOutcome] = (
            get_running_loop().create_future()
        )

        async def resolve() -> None:
            """Settle the exact request future without exposing the reader."""
            try:
                access = self._requests.get(request_id)
                if (
                    access is None
                    or access.correlation_id is not correlation_id
                ):
                    raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
                value = await self._attached_outcome(
                    request_id, access.access.identity, correlation_id
                )
                if not future.done():
                    future.set_result(value)
            except BaseException as error:
                if not future.done():
                    future.set_exception(error)

        task = create_task(resolve())
        self._reader_tasks.add(task)
        task.add_done_callback(self._reader_tasks.discard)
        return future

    def _terminal_future(self, pending: PatchPending) -> Future[PatchResult]:
        """Return one exact durable terminal wait through a loop future."""
        future: Future[PatchResult] = get_running_loop().create_future()

        async def resolve() -> None:
            """Settle the host future without exposing the reader task."""
            try:
                value = await self._await_terminal(pending)
                if not future.done():
                    future.set_result(value)
            except BaseException as error:
                if not future.done():
                    future.set_exception(error)

        task = create_task(resolve())
        self._reader_tasks.add(task)
        task.add_done_callback(self._reader_tasks.discard)
        return future

    async def _await_terminal(self, pending: PatchPending) -> PatchResult:
        """Await the exact durable continuation stored by this service."""
        access = self._pending.get(pending.pending_operation_id)
        if access is None or pending.correlation_id != access.correlation_id:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        terminal = await self.store.await_terminal(access)
        snapshot = await self.store.inspect(access.request)
        if snapshot.lease is not None:
            await self._release_activation_commit(snapshot.lease)
        self._latest = terminal.result
        return terminal.result

    async def _attached_outcome(
        self,
        request_id: PatchRequestId,
        identity: DurableRequestIdentity,
        correlation_id: PatchObserverCorrelationId,
    ) -> PatchInvocationOutcome:
        """Read an existing durable pending or terminal branch."""
        snapshot = await self.store.inspect(
            DurableRequestAccess(request_id, identity)
        )
        if snapshot.terminal is not None:
            return snapshot.terminal.result
        pending = snapshot.pending
        if pending is not None and pending.correlation_id != correlation_id:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        if not snapshot.worker_bound or snapshot.worker_reaped:
            lease = snapshot.lease
            plan = snapshot.plan
            if lease is None or plan is None:
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            now = await self.configuration.clock.now()
            if now.value >= lease.expires_at.value:
                replacement = await DurablePatchReconciler(
                    self.store
                ).replace_expired_owner(
                    DurableRequestAccess(request_id, identity),
                    PatchCommitOwnerId.new(),
                    now,
                    self.configuration.lease_duration,
                )
                artifact_ids = tuple(
                    dict.fromkeys(
                        item.artifact_id for item in snapshot.journal.artifacts
                    )
                )
                artifacts = tuple(
                    ("recovery:" + item.value, item) for item in artifact_ids
                )
                report = _recovery_report(snapshot, plan, artifacts)
                return await self._reconcile_worker(
                    request_id,
                    identity,
                    plan.plan_id,
                    replacement,
                    report,
                    correlation_id,
                    now,
                    None,
                    artifacts,
                )
        if pending is None or pending.correlation_id != correlation_id:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        outcome = PatchPending(
            1,
            pending.pending_operation_id,
            pending.request_id,
            pending.correlation_id,
            LifecyclePhase.SETTLEMENT_PENDING,
        )
        self._pending[outcome.pending_operation_id] = DurablePendingAccess(
            DurableRequestAccess(outcome.request_id, identity),
            outcome.pending_operation_id,
            outcome.correlation_id,
        )
        return outcome


def _canonical_request(
    operation: OperationType,
    raw_arguments: bytes,
    correlation_id: PatchObserverCorrelationId,
    limits: PatchInputLimits,
) -> CanonicalPatchRequest:
    """Parse exactly one public request before any sandbox inspection."""
    kind = (
        RawPatchInputKind.EDIT_JSON
        if operation is OperationType.EDIT
        else RawPatchInputKind.APPLY_JSON
    )
    return PatchRequestParser(limits).parse(
        RawPatchIngress(
            RawProviderProfile("sandbox-patch-runtime"),
            RawToolCallId(correlation_id.value),
            kind,
            RawPatchInputState.COMPLETE,
            raw_arguments,
        )
    )


def _execution_id_for_request(request_id: PatchRequestId) -> PatchExecutionId:
    """Derive a stable durable execution identity from one public request."""
    return PatchExecutionId(
        "execution_" + sha256(request_id.value.encode()).hexdigest()[:32]
    )


def _plan_id_for_request(request_id: PatchRequestId) -> PatchPlanId:
    """Derive a stable sealed-plan identity for durable retransmission."""
    return PatchPlanId(
        "plan_" + sha256(request_id.value.encode()).hexdigest()[:32]
    )


def _semantic_paths(
    request: CanonicalPatchRequest,
) -> tuple[tuple[LogicalPath, ...], frozenset[Capability]]:
    """Derive conservative paths and effects from parsed syntax only."""
    syntax = request.syntax
    if type(syntax) is StructuredEditSyntax:
        return (syntax.path,), frozenset((Capability.UPDATE,))
    assert type(syntax) is PatchDocumentSyntax
    paths: set[LogicalPath] = set()
    effects: set[Capability] = set()
    for declaration in syntax.declarations:
        match declaration:
            case AddDeclarationSyntax(path=path):
                paths.add(path)
                effects.add(Capability.CREATE)
            case DeleteDeclarationSyntax(path=path):
                paths.add(path)
                effects.add(Capability.DELETE)
            case UpdateDeclarationSyntax(path=path, move_to=move_to):
                paths.add(path)
                if move_to is None:
                    effects.add(Capability.UPDATE)
                else:
                    paths.add(move_to)
                    effects.add(Capability.MOVE)
    return tuple(sorted(paths, key=lambda item: item.value)), frozenset(
        effects
    )


def _patch_request(
    request_id: PatchRequestId,
    execution_id: PatchExecutionId,
    operation: OperationType,
    raw_arguments: bytes,
    paths: tuple[LogicalPath, ...],
) -> PatchRequest:
    """Bind public request identity to the durable coordination row."""
    return PatchRequest(
        1,
        request_id,
        execution_id,
        operation,
        PatchInput(raw_arguments),
        paths,
    )


def _durable_plan(
    plan: "SealedPlan",
    origin: DurableProtocolOrigin | None = None,
    rehydration: bytes = b"",
) -> DurablePlanReference:
    """Project a sealed plan into the exact durable journal graph."""
    return DurablePlanReference(
        plan.plan_id,
        plan.binding.request_digest,
        AlgorithmDigest.from_bytes(plan.fingerprint._value),
        plan.review.diff.digest,
        plan.binding.target.context_id,
        plan.binding.target.workspace_id,
        plan.binding.target.domain_id,
        tuple(
            DurableStepBinding(identifier, lineage)
            for identifier, lineage in _steps(
                SealedCommitCommand(
                    plan,
                    CommitLease(
                        plan.binding.target.domain_id,
                        plan.binding.request.request_id,
                        1,
                    ),
                    footprint_for(plan),
                )
            )
        ),
        origin,
        rehydration,
    )


def _durable_artifacts(
    plan: "SealedPlan",
) -> tuple[tuple[str, PatchArtifactId], ...]:
    """Bind each private worker artifact identifier to durable storage."""
    command = SealedCommitCommand(
        plan,
        CommitLease(
            plan.binding.target.domain_id,
            plan.binding.request.request_id,
            1,
        ),
        footprint_for(plan),
    )
    return tuple(
        (
            identifier,
            PatchArtifactId(
                "artifact_" + sha256(identifier.encode()).hexdigest()[:32]
            ),
        )
        for identifier in _artifacts(command)
    )


def _durable_observations(
    report: WorkerReport,
    artifacts: tuple[tuple[str, PatchArtifactId], ...],
) -> tuple[DurableArtifactObservation, ...]:
    """Convert exact worker artifact facts to durable journal transitions."""
    if report.journal is None:
        return ()
    identifiers = dict(artifacts)
    return tuple(
        DurableArtifactObservation(
            item.identifier,
            identifiers[item.identifier],
            _durable_artifact_state(item.state),
        )
        for item in report.journal.artifacts
    )


def _durable_artifact_state(value: ArtifactState) -> DurableArtifactState:
    """Map a target artifact fact to one terminal durable transition."""
    match value:
        case ArtifactState.ABSENT:
            return DurableArtifactState.NOT_CREATED
        case ArtifactState.CLEANED:
            return DurableArtifactState.REMOVED
        case ArtifactState.LEAKED:
            return DurableArtifactState.LEAKED
        case ArtifactState.UNKNOWN:
            return DurableArtifactState.UNKNOWN
        case ArtifactState.STAGED:
            return DurableArtifactState.PRESENT


def _approval_result(
    request_id: PatchRequestId,
    plan_id: PatchPlanId,
    state: ApprovalDecisionState,
) -> PatchResult:
    """Return a no-write terminal result for a broker rejection or outage."""
    code = (
        PatchErrorCode.APPROVAL_UNAVAILABLE
        if state is ApprovalDecisionState.UNAVAILABLE
        else PatchErrorCode.APPROVAL_DENIED
    )
    return _result(
        request_id,
        plan_id,
        (),
        ArtifactState.ABSENT,
        PostconditionState.UNKNOWN,
        (
            PatchStatus.APPROVAL_UNAVAILABLE
            if state is ApprovalDecisionState.UNAVAILABLE
            else PatchStatus.APPROVAL_DENIED
        ),
        PatchDiagnostic(ErrorStage.APPROVAL, code, Retryability.NOT_RETRYABLE),
    )


def _recovery_report(
    snapshot: DurableRequestSnapshot,
    plan: DurablePlanReference,
    artifacts: tuple[tuple[str, PatchArtifactId], ...],
) -> WorkerReport:
    """Return terminal journal truth already durably observed before loss."""
    states = {item.step_id: item.state for item in snapshot.journal.steps}
    terminal_steps = tuple(
        JournalStep(
            item.step_id,
            item.lineage_id,
            states.get(item.step_id, CommitStepState.UNKNOWN),
        )
        for item in plan.steps
    )
    committed = bool(terminal_steps) and all(
        item.state is CommitStepState.COMMITTED for item in terminal_steps
    )
    return WorkerReport(
        WorkerState.SETTLED if committed else WorkerState.FENCED,
        SettlementJournal(
            terminal_steps,
            tuple(
                ArtifactJournal(identifier, ArtifactState.UNKNOWN)
                for identifier, _ in artifacts
            ),
            (
                PostconditionState.ESTABLISHED
                if committed
                else PostconditionState.UNKNOWN
            ),
        ),
    )


def _worker_result(
    request_id: PatchRequestId,
    plan_id: PatchPlanId,
    report: WorkerReport,
) -> PatchResult:
    """Derive terminal truth solely from a settled sandbox worker journal."""
    if report.journal is None:
        return _result(
            request_id,
            plan_id,
            (CommitStepState.UNKNOWN,),
            ArtifactState.UNKNOWN,
            PostconditionState.UNKNOWN,
            PatchStatus.INDETERMINATE,
            PatchDiagnostic(
                ErrorStage.SETTLEMENT,
                PatchErrorCode.INDETERMINATE,
                Retryability.NOT_RETRYABLE,
            ),
        )
    states = tuple(item.state for item in report.journal.steps)
    artifact = _artifact_state(report.journal)
    return _result(
        request_id,
        plan_id,
        states,
        artifact,
        report.journal.postcondition,
        None,
        None,
    )


def _result(
    request_id: PatchRequestId,
    plan_id: PatchPlanId,
    states: tuple[CommitStepState, ...],
    artifact: ArtifactState,
    postcondition: PostconditionState,
    status: PatchStatus | None,
    diagnostic: PatchDiagnostic | None,
) -> PatchResult:
    """Create one terminal result from exact journal facts only."""
    if not states or all(
        item is CommitStepState.NOT_COMMITTED for item in states
    ):
        mutation = MutationState.NOT_COMMITTED
        occurrence = RequestedEffectOccurrence.FALSE
        postcondition = PostconditionState.UNKNOWN
    elif CommitStepState.UNKNOWN in states:
        mutation = MutationState.INDETERMINATE
        occurrence = (
            RequestedEffectOccurrence.TRUE
            if CommitStepState.COMMITTED in states
            else RequestedEffectOccurrence.UNKNOWN
        )
        if occurrence is not RequestedEffectOccurrence.TRUE:
            postcondition = PostconditionState.UNKNOWN
    elif all(item is CommitStepState.COMMITTED for item in states):
        mutation = MutationState.COMMITTED
        occurrence = RequestedEffectOccurrence.TRUE
    else:
        mutation = MutationState.PARTIALLY_COMMITTED
        occurrence = RequestedEffectOccurrence.TRUE
    workspace = (
        WorkspaceChange.CHANGED
        if occurrence is RequestedEffectOccurrence.TRUE
        or artifact in {ArtifactState.STAGED, ArtifactState.LEAKED}
        else (
            WorkspaceChange.UNKNOWN
            if occurrence is RequestedEffectOccurrence.UNKNOWN
            or artifact is ArtifactState.UNKNOWN
            else WorkspaceChange.UNCHANGED
        )
    )
    resolved_status = (
        status
        or {
            MutationState.COMMITTED: PatchStatus.COMMITTED,
            MutationState.NOT_COMMITTED: PatchStatus.COMMIT_FAILED,
            MutationState.PARTIALLY_COMMITTED: PatchStatus.PARTIAL,
            MutationState.INDETERMINATE: PatchStatus.INDETERMINATE,
        }[mutation]
    )
    resolved_diagnostic = diagnostic
    if (
        resolved_diagnostic is None
        and resolved_status is not PatchStatus.COMMITTED
    ):
        resolved_diagnostic = PatchDiagnostic(
            ErrorStage.COMMIT,
            PatchErrorCode.COMMIT_FAILED,
            Retryability.NOT_RETRYABLE,
        )
    return PatchResult(
        1,
        request_id,
        plan_id,
        LifecyclePhase.REQUEST_COMPLETED,
        resolved_status,
        CommitTruth(
            mutation,
            LineageState(mutation.value),
            occurrence,
            artifact,
            workspace,
            mutation is not MutationState.INDETERMINATE,
            postcondition,
        ),
        resolved_diagnostic,
    )


def _artifact_state(journal: SettlementJournal) -> ArtifactState:
    """Aggregate target-owned artifact facts without reading the workspace."""
    states = tuple(item.state for item in journal.artifacts)
    if not states or all(item is ArtifactState.ABSENT for item in states):
        return ArtifactState.ABSENT
    if ArtifactState.UNKNOWN in states:
        return ArtifactState.UNKNOWN
    if ArtifactState.LEAKED in states:
        return ArtifactState.LEAKED
    if all(
        item in {ArtifactState.ABSENT, ArtifactState.CLEANED}
        for item in states
    ):
        return ArtifactState.CLEANED
    return ArtifactState.UNKNOWN


@dataclass(frozen=True, slots=True, repr=False)
class _SandboxPatchOwnedResources:
    """Order durable-store and worker-service ownership as one resource."""

    runtime: SandboxPatchRuntime
    service: SandboxPatchSdkService
    durable: AbstractAsyncContextManager[object]

    async def __aenter__(self) -> object:
        """Open durability before service use and reap on either failure."""
        durable_entered = False
        try:
            await self.durable.__aenter__()
            durable_entered = True
            await self.service.__aenter__()
        except BaseException:
            try:
                await self.runtime.close()
            finally:
                if durable_entered:
                    await self.durable.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Reap and journal the worker before closing durable storage."""
        try:
            await self.service.__aexit__(
                exc_type,
                exc_value,
                traceback,
            )
        finally:
            durable_result = await self.durable.__aexit__(
                exc_type,
                exc_value,
                traceback,
            )
        return durable_result


@dataclass(frozen=True, slots=True, repr=False)
class SandboxPatchRuntimeBinder:
    """Produce a public patch binding only from a selected sandbox runtime."""

    runtime: SandboxPatchRuntime
    service_factory: SandboxPatchServiceFactory
    policy: TrustedPatchPolicy
    approval: PatchApprovalBinding
    coordinator: PatchCoordinatorBinding
    persistence: PatchPersistenceBinding
    durable_resource: AbstractAsyncContextManager[object] | None = None

    def __post_init__(self) -> None:
        """Require one actual durable store shared with the local domain."""
        store = self.coordinator.durable_store
        if (
            type(self.runtime) is not SandboxPatchRuntime
            or type(self.service_factory) is not SandboxPatchServiceFactory
            or type(self.policy) is not TrustedPatchPolicy
            or type(self.approval) is not PatchApprovalBinding
            or type(self.coordinator) is not PatchCoordinatorBinding
            or type(self.persistence) is not PatchPersistenceBinding
            or store is None
            or self.persistence.durable_store is not store
            or self.durable_resource is not None
            and id(self.durable_resource) != id(store)
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)

    @classmethod
    def from_settings(
        cls,
        settings: SandboxPatchRuntimeSettings,
        configuration: SandboxPatchServiceConfiguration,
        policy: TrustedPatchPolicy,
        approval: PatchApprovalBinding,
        coordinator: PatchCoordinatorBinding,
        persistence: PatchPersistenceBinding,
    ) -> "SandboxPatchRuntimeBinder":
        """Bind a loader from the selected trusted sandbox plan only."""
        if (
            type(settings) is not SandboxPatchRuntimeSettings
            or type(configuration) is not SandboxPatchServiceConfiguration
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        return cls(
            settings.create_runtime(),
            SandboxPatchServiceFactory(configuration),
            policy,
            approval,
            coordinator,
            persistence,
        )

    @classmethod
    def from_shared_store(
        cls,
        settings: SandboxPatchRuntimeSettings,
        configuration: SandboxPatchServiceConfiguration,
        policy: TrustedPatchPolicy,
        approval: PatchApprovalBinding,
        binding: DurablePatchStoreBinding,
    ) -> "SandboxPatchRuntimeBinder":
        """Bind the sandbox to one loader-owned cross-context durable store."""
        if type(binding) is not DurablePatchStoreBinding:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        return cls(
            settings.create_runtime(),
            SandboxPatchServiceFactory(configuration),
            policy,
            approval,
            PatchCoordinatorBinding(True, binding.store),
            PatchPersistenceBinding(True, binding.store),
            binding.resource,
        )

    async def bind(self) -> PatchRuntimeBinding:
        """Start the selected runtime and bind its capability receipt."""
        scope = await self.runtime.resolve(ScopeSelection(ContextKind.SANDBOX))
        target = SandboxCommitTarget(self.runtime)
        inspection = SandboxInspectionTarget(self.runtime)
        handshake = await target.handshake(scope)
        store = self.coordinator.durable_store
        assert store is not None
        service = await self.service_factory.bind_sandbox(
            runtime=self.runtime,
            scope=scope,
            handshake=handshake,
            target=target,
            inspection=inspection,
            store=store,
            policy=self.policy,
        )
        return PatchRuntimeBinding(
            scope,
            handshake,
            self.policy,
            self.approval,
            self.coordinator,
            self.persistence,
            service,
            (
                (
                    _SandboxPatchOwnedResources(
                        self.runtime,
                        service,
                        self.durable_resource,
                    ),
                )
                if self.durable_resource is not None
                else (service,)
            ),
            RemotePatchRuntimeWitness(
                tenant=self.service_factory.configuration.subject.tenant,
                principal=self.service_factory.configuration.subject.principal,
                run=self.service_factory.configuration.subject.run,
                session=self.service_factory.configuration.subject.session,
                task=self.service_factory.configuration.subject.task,
                agent=self.service_factory.configuration.subject.agent,
                execution_scope=scope.identity.domain_id.value,
                route=self.policy.approval.route,
                context=scope.identity.context_id,
                workspace=scope.identity.workspace_id,
                policy_revision=self.policy.revision,
                disclosures=frozenset(
                    disclosure
                    for rule in self.policy.rules
                    for disclosure in rule.disclosures
                ),
                approval_route=self.policy.approval.route,
                capabilities=scope.capabilities,
            ),
        )
