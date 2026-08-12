"""Exercise test-profile local mutation through the sealed coordinator."""

from asyncio import (
    CancelledError,
    create_task,
    gather,
    run,
)
from asyncio import (
    Event as AsyncEvent,
)
from asyncio import sleep as async_sleep
from base64 import b64encode
from dataclasses import replace
from enum import Enum
from errno import ENOSYS, EXDEV
from hashlib import sha256
from hmac import digest
from io import BytesIO
from json import dumps, loads
from multiprocessing import get_context
from os import (
    O_NOFOLLOW,
    O_NONBLOCK,
    O_RDONLY,
    O_RDWR,
    chmod,
    close,
    fstat,
    link,
    mkfifo,
    symlink,
)
from os import open as open_fd
from pathlib import Path
from pickle import dumps as pickle_dumps
from runpy import run_path
from stat import S_IMODE, S_ISREG
from subprocess import run as run_process
from sys import executable
from threading import Event, Thread
from time import sleep
from typing import Callable, Protocol, cast

import pytest

import avalan.patch.coordinator as coordinator_module
import avalan.patch.local_commit as local_commit_module
import avalan.patch.parser as parser_module
import avalan.patch.policy as policy_module
import avalan.patch.rooted_worker as rooted_worker_module
import avalan.patch.target as target_module
from avalan.patch.coordinator import (
    CommitLease,
    CoordinatorError,
    CoordinatorErrorCode,
    InMemoryCoordinatorStore,
    InMemoryLeaseManager,
    InMemoryPatchCoordinator,
    RetransmissionKey,
    RevalidationFact,
    RevalidationField,
    RevalidationSnapshot,
    RootedLocalCommitWorker,
    RuntimeIdentity,
    ScriptedCommitWorker,
    ScriptedReconciler,
    SealedCommitCommand,
    WorkerReport,
    WorkerState,
    footprint_for,
)
from avalan.patch.domain import (
    AlgorithmDigest,
    ApprovalMode,
    ArtifactState,
    ByteSize,
    Capability,
    CommitStepState,
    ContextKind,
    DurationTicks,
    ExpiryTick,
    FileMode,
    LogicalPath,
    MetadataProfile,
    OperationType,
    PatchErrorCode,
    PatchExecutionId,
    PatchGrantId,
    PatchInput,
    PatchLimits,
    PatchPlanId,
    PatchRequest,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PostconditionState,
    SourceBytes,
)
from avalan.patch.local_commit import LocalCommitTarget
from avalan.patch.parser import (
    CanonicalPatchRequest,
    PatchInputError,
    PatchInputErrorCode,
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderProfile,
    RawToolCallId,
)
from avalan.patch.planner import (
    LogicalText,
    PlannedLineage,
    PlannerCandidate,
    PlannerError,
    PlannerErrorCode,
    PlannerFile,
    PlannerLimits,
    PlannerParentMount,
    PlannerWorkspace,
    plan,
)
from avalan.patch.policy import (
    ApprovalClock,
    ApprovalDecisionState,
    ApprovalRequirements,
    ApprovalService,
    BrokerDecision,
    CapabilityMode,
    ExecutionSubject,
    FinalAuthorization,
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PlanBinding,
    PlanBoundGrant,
    PlanReviewRequest,
    PolicyAuthorizer,
    PolicyBrokerId,
    PolicyError,
    PolicyErrorCode,
    PolicyPathSelector,
    PolicyReviewerRole,
    PolicyRevision,
    PolicyRouteId,
    PolicyRule,
    PreauthorizationClass,
    PreflightAuthorization,
    PreflightRequest,
    ReviewerDecision,
    RuntimeGrantStore,
    SealedPlan,
    TrustedPatchPolicy,
    compose_limits,
    seal_plan,
)
from avalan.patch.target import (
    _WORKER_TOKEN_ENV,
    ForeignWriterGuarantee,
    InspectionRequest,
    LocalInspectionTarget,
    LocalPlatformProfile,
    LocalScopeResolver,
    LocalTargetProfile,
    ResolvedMutationScope,
    ScopeSelection,
    TargetErrorCode,
    TargetHandshake,
    TargetInspectionError,
    TargetPrimitive,
)

_PHASE4 = run_path("tests/patch/phase_4_contract_test.py")
_PHASE5 = run_path("tests/patch/phase_5_contract_test.py")
_PHASE6 = run_path("tests/patch/phase_6_contract_test.py")

_SEMANTIC_PRECOMMIT_CATEGORIES = (
    "schema_grammar_parse",
    "lexical_path",
    "missing_match",
    "ambiguous_match",
    "duplicate_transition",
    "nested_match",
    "overlap",
    "prohibited_touch",
    "exact_first_no_fallback",
    "representation_compatible_ambiguity",
    "eof_final_newline",
    "no_op",
    "preinspection_policy",
    "final_policy",
    "approval",
    "stale",
    "limit",
    "cancellation",
    "timeout",
    "target_capability",
)


class _DirectoryUnlink(Protocol):
    """Describe one test-only directory-relative unlink hook."""

    def __call__(self, path: str, *, dir_fd: int) -> None:
        """Remove one path through an already-open directory."""


class _DirectoryLink(Protocol):
    """Describe one test-only directory-relative link hook."""

    def __call__(
        self,
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        """Link one path through already-open parent descriptors."""


class _FileWrite(Protocol):
    """Describe one test-only staging-write hook."""

    def __call__(self, descriptor: int, value: bytes) -> int:
        """Write bytes to one already-open staging descriptor."""


class _ThreadFunction(Protocol):
    """Describe one synchronous operation delegated through a test thread."""

    def __call__(self, *arguments: object) -> object:
        """Return one delegated result."""


class _PrecommitFault(BaseException):
    """Stop a reached precommit component before any workspace effect."""


class _E2EPrecommitBoundary(str, Enum):
    """Name every fixed local precommit boundary exercised by E2E-003."""

    LIFECYCLE_RECEIVED = "lifecycle.received"
    LIFECYCLE_SCOPE_BOUND = "lifecycle.scope_bound"
    LIFECYCLE_PREFLIGHT = "lifecycle.preinspection_authorized"
    LIFECYCLE_PLANNED = "lifecycle.planned"
    LIFECYCLE_AWAITING_APPROVAL = "lifecycle.awaiting_approval"
    LIFECYCLE_COMMIT_OWNER = "lifecycle.commit_owner_assigned"
    LIFECYCLE_COMPLETED = "lifecycle.request_completed"
    TARGET_NEGOTIATE = "target.negotiate_capabilities"
    TARGET_INSPECT = "target.inspect"
    TARGET_PRECONDITION = "target.observe_precondition"
    TARGET_OPEN = "target.open_handle"
    TARGET_CLOSE = "target.close_handle"
    TARGET_ACQUIRE_LOCK = "target.acquire_lock"
    TARGET_RELEASE_LOCK = "target.release_lock"
    TARGET_STAGE = "target.stage_artifact"
    REQUESTED_EFFECT = "requested_effect.step_before"
    STORE_RESERVE = "store.reserve_request"
    STORE_PERSIST = "store.persist_plan"
    STORE_CONSUME = "store.consume_grant"
    STORE_OWNER = "store.assign_commit_owner"
    APPROVAL_DECIDE = "approval.decide"
    APPROVAL_CONSUME = "approval.consume"
    APPROVAL_CONCURRENT = "approval.concurrent_consume"
    COMMIT_FENCE = "commit.intent_fence"
    ARTIFACT_STAGE = "artifact.stage"
    CANCELLATION = "cancellation.before_commit"
    TIMEOUT = "timeout.before_commit"
    DISCONNECT = "disconnect.before_commit"


_EXECUTOR_BOUNDARIES: dict[str, _E2EPrecommitBoundary] = {
    "lifecycle_received": _E2EPrecommitBoundary.LIFECYCLE_RECEIVED,
    "lifecycle_scope_bound": _E2EPrecommitBoundary.LIFECYCLE_SCOPE_BOUND,
    "lifecycle_preinspection_authorized": (
        _E2EPrecommitBoundary.LIFECYCLE_PREFLIGHT
    ),
    "lifecycle_planned": _E2EPrecommitBoundary.LIFECYCLE_PLANNED,
    "lifecycle_awaiting_approval": (
        _E2EPrecommitBoundary.LIFECYCLE_AWAITING_APPROVAL
    ),
    "lifecycle_commit_owner_assigned": (
        _E2EPrecommitBoundary.LIFECYCLE_COMMIT_OWNER
    ),
    "lifecycle_request_completed": _E2EPrecommitBoundary.LIFECYCLE_COMPLETED,
    "target_negotiate_capabilities": _E2EPrecommitBoundary.TARGET_NEGOTIATE,
    "target_inspect": _E2EPrecommitBoundary.TARGET_INSPECT,
    "target_observe_precondition": _E2EPrecommitBoundary.TARGET_PRECONDITION,
    "target_open_handle": _E2EPrecommitBoundary.TARGET_OPEN,
    "target_close_handle": _E2EPrecommitBoundary.TARGET_CLOSE,
    "target_acquire_lock": _E2EPrecommitBoundary.TARGET_ACQUIRE_LOCK,
    "target_release_lock": _E2EPrecommitBoundary.TARGET_RELEASE_LOCK,
    "target_stage_artifact": _E2EPrecommitBoundary.TARGET_STAGE,
    "requested_effect_step_before": _E2EPrecommitBoundary.REQUESTED_EFFECT,
    "store_reserve_request": _E2EPrecommitBoundary.STORE_RESERVE,
    "store_persist_plan": _E2EPrecommitBoundary.STORE_PERSIST,
    "store_consume_grant": _E2EPrecommitBoundary.STORE_CONSUME,
    "store_assign_commit_owner": _E2EPrecommitBoundary.STORE_OWNER,
    "approval_decide": _E2EPrecommitBoundary.APPROVAL_DECIDE,
    "approval_consume": _E2EPrecommitBoundary.APPROVAL_CONSUME,
    "approval_concurrent_consume": _E2EPrecommitBoundary.APPROVAL_CONCURRENT,
    "commit_intent_fence": _E2EPrecommitBoundary.COMMIT_FENCE,
    "artifact_stage": _E2EPrecommitBoundary.ARTIFACT_STAGE,
    "cancellation_before_commit": _E2EPrecommitBoundary.CANCELLATION,
    "timeout_before_commit": _E2EPrecommitBoundary.TIMEOUT,
    "disconnect_before_commit": _E2EPrecommitBoundary.DISCONNECT,
}


_STORE_BOUNDARIES: dict[
    coordinator_module._PrecommitBoundary, _E2EPrecommitBoundary
] = {
    coordinator_module._PrecommitBoundary.RESERVE_REQUEST: (
        _E2EPrecommitBoundary.STORE_RESERVE
    ),
    coordinator_module._PrecommitBoundary.PERSIST_PLAN: (
        _E2EPrecommitBoundary.STORE_PERSIST
    ),
    coordinator_module._PrecommitBoundary.LIFECYCLE_PLANNED: (
        _E2EPrecommitBoundary.LIFECYCLE_PLANNED
    ),
    coordinator_module._PrecommitBoundary.LIFECYCLE_AWAITING_APPROVAL: (
        _E2EPrecommitBoundary.LIFECYCLE_AWAITING_APPROVAL
    ),
    coordinator_module._PrecommitBoundary.CONSUME_GRANT: (
        _E2EPrecommitBoundary.STORE_CONSUME
    ),
    coordinator_module._PrecommitBoundary.ASSIGN_COMMIT_OWNER: (
        _E2EPrecommitBoundary.STORE_OWNER
    ),
    coordinator_module._PrecommitBoundary.LIFECYCLE_COMMIT_OWNER: (
        _E2EPrecommitBoundary.LIFECYCLE_COMMIT_OWNER
    ),
    coordinator_module._PrecommitBoundary.INTENT_FENCE: (
        _E2EPrecommitBoundary.COMMIT_FENCE
    ),
    coordinator_module._PrecommitBoundary.LIFECYCLE_COMPLETED: (
        _E2EPrecommitBoundary.LIFECYCLE_COMPLETED
    ),
    coordinator_module._PrecommitBoundary.ACQUIRE_LOCK: (
        _E2EPrecommitBoundary.TARGET_ACQUIRE_LOCK
    ),
    coordinator_module._PrecommitBoundary.RELEASE_LOCK: (
        _E2EPrecommitBoundary.TARGET_RELEASE_LOCK
    ),
    coordinator_module._PrecommitBoundary.CANCELLATION: (
        _E2EPrecommitBoundary.CANCELLATION
    ),
    coordinator_module._PrecommitBoundary.TIMEOUT: (
        _E2EPrecommitBoundary.TIMEOUT
    ),
    coordinator_module._PrecommitBoundary.DISCONNECT: (
        _E2EPrecommitBoundary.DISCONNECT
    ),
}


class _PrecommitFaults:
    """Select one fixed component-owned precommit failure for this test."""

    def __init__(self, failure: _E2EPrecommitBoundary) -> None:
        """Bind exactly one hard-coded executor boundary as the fault."""
        self._failure = failure
        self.observed: list[_E2EPrecommitBoundary] = []
        self._triggered = False

    async def checkpoint(
        self, boundary: coordinator_module._PrecommitBoundary
    ) -> None:
        """Observe a checkpoint emitted within the real coordinator store."""
        await self.reached(_STORE_BOUNDARIES[boundary])

    async def reached(self, boundary: _E2EPrecommitBoundary) -> None:
        """Record an async component boundary and stop only the selection."""
        self.reached_now(boundary)

    def reached_now(self, boundary: _E2EPrecommitBoundary) -> None:
        """Record a native component boundary before its write primitive."""
        if self._triggered:
            return
        self.observed.append(boundary)
        if boundary is self._failure:
            self._triggered = True
            raise _PrecommitFault()


class _E2EClock(ApprovalClock):
    """Return one fixed unexpired approval clock value."""

    async def now(self) -> ExpiryTick:
        """Return an unexpired monotonic test tick."""
        return ExpiryTick(1)


class _E2EApprovalBroker:
    """Run the real approval-service broker call under a fixed checkpoint."""

    def __init__(self, faults: _PrecommitFaults | None = None) -> None:
        """Bind an optional boundary emitted inside broker decision."""
        self._faults = faults

    async def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Approve the complete plan after emitting the broker boundary."""
        if self._faults is not None:
            await self._faults.reached(_E2EPrecommitBoundary.APPROVAL_DECIDE)
        return BrokerDecision(
            request.requirements.broker,
            (
                ReviewerDecision(
                    PatchPrincipalId("reviewer-seven"),
                    request.subject.tenant,
                    request.requirements.reviewer_role,
                    ApprovalDecisionState.APPROVED,
                ),
            ),
        )


class _E2EGrantStore(RuntimeGrantStore):
    """Expose one real grant lookup checkpoint inside approval validation."""

    def __init__(self, faults: _PrecommitFaults) -> None:
        """Bind the selected approval-consume observation hook."""
        super().__init__()
        self._faults = faults

    async def get(self, grant_id: PatchGrantId) -> PlanBoundGrant | None:
        """Read the private grant then emit the exact approval boundary."""
        result = await super().get(grant_id)
        await self._faults.reached(_E2EPrecommitBoundary.APPROVAL_CONSUME)
        return result


class _ConcurrentE2EGrantStore(RuntimeGrantStore):
    """Synchronize two real approval-grant lookups before one observation."""

    def __init__(self, faults: _PrecommitFaults) -> None:
        """Bind the selected concurrent-consume observation hook."""
        super().__init__()
        self._faults = faults
        self._arrived = AsyncEvent()
        self._calls = 0
        self._emitted = False

    async def get(self, grant_id: PatchGrantId) -> PlanBoundGrant | None:
        """Read one grant concurrently and fault only after both reached it."""
        self._calls += 1
        if self._calls == 2:
            self._arrived.set()
        await self._arrived.wait()
        result = await super().get(grant_id)
        if not self._emitted:
            self._emitted = True
            await self._faults.reached(
                _E2EPrecommitBoundary.APPROVAL_CONCURRENT
            )
        return result


def _profile(root: Path) -> LocalTargetProfile:
    """Return a signed local profile enabled only for this test."""
    target_module._RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES = _PHASE4[
        "_TEST_RUNTIME_AUTHORITY_VERIFIER_BYTES"
    ]
    target_module._WORKER_BOOTSTRAP = _PHASE4["_test_worker_bootstrap"]()
    profile = _PHASE4["_profile"](root, policy="policy-six")
    assert isinstance(profile, LocalTargetProfile)
    namespace = root.parent / (".avalan-patch-private-" + root.name)
    namespace.mkdir(mode=0o700, exist_ok=True)
    return replace(
        profile,
        platform=LocalPlatformProfile.DARWIN,
        mutation_test_profile=True,
        commit_namespace=namespace,
    )


def _limits() -> PatchLimits:
    """Return one finite target, policy, and planner limit matrix."""
    return PatchLimits(
        ByteSize(10_000),
        ByteSize(20),
        ByteSize(512),
        ByteSize(20),
        ByteSize(20),
        ByteSize(10_000),
        ByteSize(10_000),
        ByteSize(10_000),
        DurationTicks(100),
        DurationTicks(100),
        DurationTicks(100),
    )


def _e2e_inventory() -> dict[str, object]:
    """Return the immutable local-commit scenario inventory."""
    value = loads(
        Path("tests/fixtures/patch/local_commit_e2e.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    sealed = dict(value)
    digest = sealed.pop("inventory_sha256")
    assert isinstance(digest, str)
    assert (
        sha256(
            dumps(sealed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == digest
    )
    return value


def _phase_seven_snapshot() -> RevalidationSnapshot:
    """Return a complete revalidation witness without a test helper."""
    return RevalidationSnapshot(
        tuple(
            sorted(
                (
                    RevalidationFact(
                        field,
                        "key-" + field.value,
                        "value-" + field.value,
                    )
                    for field in RevalidationField
                ),
                key=lambda item: (item.field.value, item.key, item.value),
            )
        )
    )


def _tree_snapshot(root: Path) -> tuple[tuple[str, bytes, int, str], ...]:
    """Return regular entries with bytes, mode, and metadata digest."""
    entries: list[tuple[str, bytes, int, str]] = []
    for item in root.rglob("*"):
        if not item.is_file() or item.is_symlink():
            continue
        descriptor = open_fd(item, O_RDONLY)
        try:
            protected = (
                target_module._capture_protected_metadata(descriptor)
                .digest()
                .value
            )
        finally:
            close(descriptor)
        entries.append(
            (
                item.relative_to(root).as_posix(),
                item.read_bytes(),
                S_IMODE(item.stat(follow_symlinks=False).st_mode),
                protected,
            )
        )
    return tuple(sorted(entries))


def test_patch_phase_7_requirements(tmp_path: Path) -> None:
    """Bind mutation to one trusted local scope and complete receipt."""
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        handshake = await target.handshake(scope)
        worker = await target.worker(scope)
        assert handshake.identity == profile.identity
        assert handshake.advertised_operations() >= {
            Capability.CREATE,
            Capability.UPDATE,
            Capability.DELETE,
            Capability.MOVE,
        }
        assert (
            handshake.foreign_writer_guarantee
            is ForeignWriterGuarantee.REVALIDATE_BEFORE_COMMIT
        )
        assert TargetPrimitive.PERSISTENCE not in handshake.primitives
        assert all(
            probe.receipt is not None
            for probe in handshake.probes
            if probe.state.value == "available"
        )
        assert worker.__class__.__name__ == "RootedLocalCommitWorker"

    run(execute())
    assert profile.root._path.is_dir()


def test_patch_phase_7_live_mutation_receipts_use_private_namespace(
    tmp_path: Path,
) -> None:
    """Exercise and clean every Darwin mutation primitive before advertising.

    Ensure the target advertises only its live-tested mutation primitives.
    """
    (tmp_path / "note.txt").write_bytes(b"before\n")
    profile = _profile(tmp_path)
    before = _tree_snapshot(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        future = target_module._FUTURE_MUTATION_PRIMITIVES
        probes = {
            probe.primitive: probe
            for probe in scope.probes
            if probe.primitive in future
        }
        assert frozenset(probes) == future
        assert all(
            probe.state.value == "available" for probe in probes.values()
        )
        receipts = {probe.receipt for probe in probes.values()}
        assert len(receipts) == 1
        assert None not in receipts
        handshake = await LocalCommitTarget(profile).handshake(scope)
        assert future.issubset(handshake.primitives)

    run(execute())
    assert _tree_snapshot(tmp_path) == before
    assert profile.commit_namespace is not None
    assert not tuple(profile.commit_namespace.iterdir())


def test_patch_phase_7_failed_live_mutation_probe_withholds_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Withdraw every mutation capability when one real probe fails closed."""
    (tmp_path / "note.txt").write_bytes(b"before\n")
    profile = _profile(tmp_path)
    before = _tree_snapshot(tmp_path)

    def unavailable_metadata(descriptor: int) -> None:
        """Fail one live metadata probe before capability advertisement."""
        del descriptor
        raise OSError("injected metadata receipt failure")

    monkeypatch.setattr(
        target_module, "_probe_metadata_round_trip", unavailable_metadata
    )

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        future = target_module._FUTURE_MUTATION_PRIMITIVES
        probes = tuple(
            probe for probe in scope.probes if probe.primitive in future
        )
        assert len(probes) == len(future)
        assert all(probe.state.value == "unavailable" for probe in probes)
        assert all(probe.receipt is None for probe in probes)
        with pytest.raises(TargetInspectionError) as error:
            await LocalCommitTarget(profile).handshake(scope)
        assert error.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE

    run(execute())
    assert _tree_snapshot(tmp_path) == before
    assert profile.commit_namespace is not None
    assert not tuple(profile.commit_namespace.iterdir())


def test_patch_phase_7_live_acl_probe_sets_and_clears_empty_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require a real non-empty ACL set and exact empty-baseline clear."""
    path = tmp_path / "acl-probe.txt"
    path.write_bytes(b"probe\n")
    descriptor = open_fd(path, O_RDWR)
    observed: list[bytes | None] = []
    original_set_acl = target_module._set_acl

    def observe_set_acl(target_descriptor: int, acl: object) -> None:
        """Record each live ACL state after applying its native handle."""
        original_set_acl(target_descriptor, acl)
        observed.append(target_module._capture_acl(target_descriptor))

    try:
        assert target_module._capture_acl(descriptor) is None
        monkeypatch.setattr(target_module, "_set_acl", observe_set_acl)
        target_module._probe_metadata_round_trip(descriptor)
        assert observed[0] is not None
        assert observed[1] is None
        assert target_module._capture_acl(descriptor) is None
    finally:
        close(descriptor)


def test_patch_phase_7_failed_live_acl_probe_withholds_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Withhold every mutation receipt when an ACL probe cannot round trip."""
    profile = _profile(tmp_path)

    def unavailable_acl(descriptor: int, acl: object) -> None:
        """Reject the live ACL application without a capability fallback."""
        del descriptor, acl
        raise OSError("injected ACL receipt failure")

    monkeypatch.setattr(target_module, "_set_acl", unavailable_acl)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        future = target_module._FUTURE_MUTATION_PRIMITIVES
        probes = tuple(
            probe for probe in scope.probes if probe.primitive in future
        )
        assert len(probes) == len(future)
        assert all(probe.state.value == "unavailable" for probe in probes)
        assert all(probe.receipt is None for probe in probes)

    run(execute())
    assert profile.commit_namespace is not None
    assert not tuple(profile.commit_namespace.iterdir())


def test_patch_phase_7_native_probe_failure_paths_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject private-probe namespace, write, and verification faults."""
    profile = _profile(tmp_path)

    async def resolve() -> ResolvedMutationScope:
        """Resolve one live root witness before direct probe checks."""
        return await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )

    scope = run(resolve())
    assert scope.root_witness is not None
    with pytest.raises(OSError):
        target_module._probe_mutation_primitives(
            replace(profile, commit_namespace=None), scope.root_witness
        )
    wrong_witness = replace(
        scope.root_witness,
        identity=target_module.FileIdentity(
            scope.root_witness.identity.device + 1,
            scope.root_witness.identity.inode,
        ),
    )
    with pytest.raises(OSError):
        target_module._probe_mutation_primitives(profile, wrong_witness)

    def stalled_write(descriptor: int, value: bytes) -> int:
        """Report no forward progress for the first staged probe write."""
        del descriptor, value
        return 0

    with monkeypatch.context() as patcher:
        patcher.setattr(target_module, "write", stalled_write)
        with pytest.raises(OSError):
            target_module._probe_mutation_primitives(
                profile, scope.root_witness
            )

    calls = 0

    def replacement_stalled(descriptor: int, value: bytes) -> int:
        """Allow staging once, then stall the replacement probe write."""
        nonlocal calls
        del descriptor
        calls += 1
        return len(value) if calls == 1 else 0

    with monkeypatch.context() as patcher:
        patcher.setattr(target_module, "write", replacement_stalled)
        with pytest.raises(OSError):
            target_module._probe_mutation_primitives(
                profile, scope.root_witness
            )

    def wrong_replacement(descriptor: int, maximum: int) -> bytes:
        """Return distinct bytes after the private replacement publication."""
        del descriptor, maximum
        return b"different\n"

    with monkeypatch.context() as patcher:
        patcher.setattr(target_module, "_read_bounded", wrong_replacement)
        with pytest.raises(OSError):
            target_module._probe_mutation_primitives(
                profile, scope.root_witness
            )


def test_patch_phase_7_native_metadata_probe_rejects_each_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed for native metadata probe and ACL round-trip errors."""
    path = tmp_path / "metadata-probe.txt"
    path.write_bytes(b"probe\n")
    descriptor = open_fd(path, O_RDWR)
    baseline = target_module._ProtectedMetadata((), 0, None)

    class MetadataFailure:
        """Provide selected nonzero native metadata return values."""

        def __init__(
            self,
            set_result: int,
            remove_result: int,
            flag_results: tuple[int, ...],
        ) -> None:
            """Store the configured native failure sequence."""
            self._set_result = set_result
            self._remove_result = remove_result
            self._flag_results = list(flag_results)

        def fsetxattr(self, *arguments: object) -> int:
            """Return the configured extended-attribute set result."""
            del arguments
            return self._set_result

        def fremovexattr(self, *arguments: object) -> int:
            """Return the configured extended-attribute removal result."""
            del arguments
            return self._remove_result

        def fchflags(self, *arguments: object) -> int:
            """Return the next configured flags result."""
            del arguments
            return self._flag_results.pop(0)

    try:
        for native in (
            MetadataFailure(-1, 0, ()),
            MetadataFailure(0, -1, ()),
            MetadataFailure(0, 0, (-1,)),
            MetadataFailure(0, 0, (0, -1)),
        ):
            with monkeypatch.context() as patcher:
                patcher.setattr(
                    target_module,
                    "_capture_protected_metadata",
                    lambda _fd: baseline,
                )
                patcher.setattr(
                    target_module,
                    "_probe_acl_round_trip",
                    lambda _fd, _acl: None,
                )
                patcher.setattr(target_module, "_METADATA_LIBC", native)
                with pytest.raises(OSError):
                    target_module._probe_metadata_round_trip(descriptor)
    finally:
        close(descriptor)

    class AclRelease:
        """Release a synthetic ACL handle without a native allocation."""

        def acl_free(self, value: object) -> int:
            """Accept one test-only ACL value without side effects."""
            del value
            return 0

    with monkeypatch.context() as patcher:
        patcher.setattr(target_module, "_probe_acl", lambda: object())
        patcher.setattr(target_module, "_set_acl", lambda _fd, _acl: None)
        patcher.setattr(target_module, "_restore_acl", lambda _fd, _acl: None)
        patcher.setattr(target_module, "_capture_acl", lambda _fd: None)
        patcher.setattr(target_module, "_METADATA_LIBC", AclRelease())
        with pytest.raises(OSError):
            target_module._probe_acl_round_trip(0, None)

    captured = iter((b"set", b"different"))
    with monkeypatch.context() as patcher:
        patcher.setattr(target_module, "_probe_acl", lambda: object())
        patcher.setattr(target_module, "_set_acl", lambda _fd, _acl: None)
        patcher.setattr(target_module, "_restore_acl", lambda _fd, _acl: None)
        patcher.setattr(
            target_module, "_capture_acl", lambda _fd: next(captured)
        )
        patcher.setattr(target_module, "_METADATA_LIBC", AclRelease())
        with pytest.raises(OSError):
            target_module._probe_acl_round_trip(0, None)


def test_patch_phase_7_native_metadata_helpers_reject_malformed_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject malformed native metadata observations before publication."""
    with pytest.raises(TargetInspectionError):
        target_module.PrimitiveProbe(
            TargetPrimitive.PERSISTENCE,
            target_module.ProbeState.UNAVAILABLE,
            "receipt",
        )

    path = tmp_path / "leaf.txt"
    path.write_bytes(b"leaf\n")
    descriptor = open_fd(tmp_path, O_RDONLY)
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(
                target_module,
                "_capture_protected_metadata",
                lambda _fd: (_ for _ in ()).throw(OSError("metadata")),
            )
            status = fstat(descriptor)
            parent = target_module.ParentWitness(
                None,
                target_module.FileIdentity(status.st_dev, status.st_ino),
                "mount",
            )
            worker_profile = target_module._WorkerInspectionProfile(
                tmp_path, None, 128, 128
            )
            with pytest.raises(TargetInspectionError) as metadata:
                target_module._snapshot_leaf(
                    descriptor,
                    LogicalPath("leaf.txt"),
                    "leaf.txt",
                    parent,
                    worker_profile,
                )
            assert metadata.value.code is TargetErrorCode.METADATA_DENIED
    finally:
        close(descriptor)

    with monkeypatch.context() as patcher:
        patcher.setattr(target_module, "fstat", lambda _fd: object())
        with pytest.raises(OSError):
            target_module._capture_protected_metadata(0)

    class XattrResults:
        """Return a selected native extended-attribute observation sequence."""

        def __init__(
            self,
            list_results: tuple[int, ...],
            names: bytes = b"",
            get_results: tuple[int, ...] = (),
        ) -> None:
            """Store list and value result sequences for one test case."""
            self._list_results = list(list_results)
            self._names = names
            self._get_results = list(get_results)

        def flistxattr(
            self, descriptor: int, buffer: object, length: int, flags: int
        ) -> int:
            """Return a list result and fill the supplied native buffer."""
            del descriptor, flags
            result = self._list_results.pop(0)
            if buffer != target_module._METADATA_FFI.NULL and result == length:
                target_module._METADATA_FFI.buffer(buffer, length)[
                    :
                ] = self._names
            return result

        def fgetxattr(self, *arguments: object) -> int:
            """Return the next configured extended-attribute value result."""
            del arguments
            return self._get_results.pop(0)

    for native, rejected in (
        (XattrResults((-1,)), True),
        (XattrResults((0,)), False),
        (XattrResults((2, 1), b"a\x00"), True),
        (XattrResults((2, 2), b"ab"), True),
        (XattrResults((1, 1), b"\x00"), True),
        (XattrResults((2, 2), b"a\x00", (-1,)), True),
        (XattrResults((2, 2), b"a\x00", (1, 0)), True),
    ):
        with monkeypatch.context() as patcher:
            patcher.setattr(target_module, "_METADATA_LIBC", native)
            if rejected:
                with pytest.raises(OSError):
                    target_module._capture_xattrs(0)
            else:
                assert target_module._capture_xattrs(0) == ()

    class AclUnavailable:
        """Return unavailable ACL values."""

        def acl_get_fd(self, descriptor: int) -> object:
            """Report no ACL handle for one invalid descriptor."""
            del descriptor
            return target_module._METADATA_FFI.NULL

    with monkeypatch.context() as patcher:
        target_module._METADATA_FFI.errno = 1
        patcher.setattr(target_module, "_METADATA_LIBC", AclUnavailable())
        with pytest.raises(OSError):
            target_module._capture_acl(0)

    class AclTextUnavailable:
        """Return a non-null ACL with an unavailable text projection."""

        def acl_get_fd(self, descriptor: int) -> object:
            """Return a synthetic ACL handle for text conversion."""
            del descriptor
            return object()

        def acl_to_text(self, acl: object, length: object) -> object:
            """Report unavailable ACL text without changing its length."""
            del acl, length
            return target_module._METADATA_FFI.NULL

        def acl_free(self, value: object) -> int:
            """Release one synthetic ACL resource without side effects."""
            del value
            return 0

    with monkeypatch.context() as patcher:
        patcher.setattr(target_module, "_METADATA_LIBC", AclTextUnavailable())
        with pytest.raises(OSError):
            target_module._capture_acl(0)

    class AclSetFailure:
        """Return a failed ACL application result."""

        def acl_set_fd(self, descriptor: int, acl: object) -> int:
            """Reject one synthetic ACL application."""
            del descriptor, acl
            return -1

    with monkeypatch.context() as patcher:
        patcher.setattr(target_module, "_METADATA_LIBC", AclSetFailure())
        with pytest.raises(OSError):
            target_module._set_acl(0, object())

    with pytest.raises(TargetInspectionError):
        target_module._ProtectedMetadata(((b"", b"value"),), 0, None)
    parent = target_module.ParentWitness(
        None, target_module.FileIdentity(1, 1), "mount"
    )
    with pytest.raises(TargetInspectionError):
        target_module.TargetSnapshot(
            LogicalPath("absent.txt"),
            False,
            None,
            None,
            None,
            0,
            parent,
            protected_metadata=cast(AlgorithmDigest, object()),
        )

    class AclInitializationFailure:
        """Reject allocation and parsing of synthetic ACL values."""

        def acl_init(self, count: int) -> object:
            """Report unavailable empty ACL initialization."""
            del count
            return target_module._METADATA_FFI.NULL

        def acl_from_text(self, value: bytes) -> object:
            """Report unavailable native ACL parsing."""
            del value
            return target_module._METADATA_FFI.NULL

    with monkeypatch.context() as patcher:
        patcher.setattr(
            target_module, "_METADATA_LIBC", AclInitializationFailure()
        )
        with pytest.raises(OSError):
            target_module._probe_acl()
        with pytest.raises(OSError):
            target_module._restore_acl(0, None)
        with pytest.raises(OSError):
            target_module._restore_acl(0, b"invalid")

    class AclConstructionFailure:
        """Reject a non-null ACL whose first entry cannot be configured."""

        def acl_init(self, count: int) -> object:
            """Return a synthetic non-null ACL handle."""
            del count
            return target_module._METADATA_FFI.cast("void *", 1)

        def acl_create_entry(self, pointer: object, entry: object) -> int:
            """Fail the first ACL entry construction step."""
            del pointer, entry
            return -1

        def acl_free(self, value: object) -> int:
            """Release synthetic ACL data without a native free."""
            del value
            return 0

    with monkeypatch.context() as patcher:
        patcher.setattr(
            target_module, "_METADATA_LIBC", AclConstructionFailure()
        )
        with pytest.raises(OSError):
            target_module._probe_acl()

    current = target_module._ProtectedMetadata(((b"old", b"value"),), 0, None)
    desired = target_module._ProtectedMetadata((), 0, None)

    class MetadataRestoreFailure:
        """Return controlled native restore errors without touching a file."""

        def __init__(self, remove: int, set_value: int, flags: int) -> None:
            """Store native removal, set, and flag result values."""
            self._remove = remove
            self._set_value = set_value
            self._flags = flags

        def fremovexattr(self, *arguments: object) -> int:
            """Return the configured attribute-removal result."""
            del arguments
            return self._remove

        def fsetxattr(self, *arguments: object) -> int:
            """Return the configured attribute-set result."""
            del arguments
            return self._set_value

        def fchflags(self, *arguments: object) -> int:
            """Return the configured flags restore result."""
            del arguments
            return self._flags

    with monkeypatch.context() as patcher:
        patcher.setattr(
            target_module, "_capture_protected_metadata", lambda _fd: current
        )
        patcher.setattr(
            target_module, "_METADATA_LIBC", MetadataRestoreFailure(-1, 0, 0)
        )
        with pytest.raises(OSError):
            target_module._restore_protected_metadata(0, desired)

    desired_xattr = target_module._ProtectedMetadata(
        ((b"new", b"value"),), 0, None
    )
    with monkeypatch.context() as patcher:
        patcher.setattr(
            target_module,
            "_capture_protected_metadata",
            lambda _fd: desired,
        )
        patcher.setattr(
            target_module, "_METADATA_LIBC", MetadataRestoreFailure(0, -1, 0)
        )
        with pytest.raises(OSError):
            target_module._restore_protected_metadata(0, desired_xattr)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            target_module,
            "_capture_protected_metadata",
            lambda _fd: desired,
        )
        patcher.setattr(target_module, "_restore_acl", lambda _fd, _acl: None)
        patcher.setattr(
            target_module, "_METADATA_LIBC", MetadataRestoreFailure(0, 0, -1)
        )
        with pytest.raises(OSError):
            target_module._restore_protected_metadata(0, desired)

    observations = iter((desired, current))
    with monkeypatch.context() as patcher:
        patcher.setattr(
            target_module,
            "_capture_protected_metadata",
            lambda _fd: next(observations),
        )
        patcher.setattr(target_module, "_restore_acl", lambda _fd, _acl: None)
        patcher.setattr(
            target_module, "_METADATA_LIBC", MetadataRestoreFailure(0, 0, 0)
        )
        with pytest.raises(OSError):
            target_module._restore_protected_metadata(0, desired)


def test_patch_phase_7_barrier_helpers_reject_malformed_and_replayed_messages(
    tmp_path: Path,
) -> None:
    """Reject malformed HMAC envelopes and authenticated older sequences."""
    token = bytes(range(32))
    marker = tmp_path / "marker"
    release = tmp_path / "release"
    marker.write_text(
        dumps({"mac": "0" * 64, "value": "1:artifact.stage"}),
        encoding="utf-8",
    )
    with pytest.raises(TargetInspectionError) as malformed:
        local_commit_module._read_barrier_message(marker, token)
    assert malformed.value.code is TargetErrorCode.WITNESS_STALE

    async def replay() -> None:
        local_commit_module._write_barrier_message(
            marker, "1:artifact.stage", token
        )
        relay = create_task(
            local_commit_module._relay_seatbelt_barriers(
                marker, release, token
            )
        )
        for sequence, stage in (
            (1, "artifact.stage"),
            (2, "target.stage_artifact"),
        ):
            expected = str(sequence) + ":" + stage
            for _ in range(2_000):
                if (
                    local_commit_module._read_barrier_message(release, token)
                    == expected
                ):
                    break
                await async_sleep(0.001)
            assert (
                local_commit_module._read_barrier_message(release, token)
                == expected
            )
            if sequence == 1:
                local_commit_module._write_barrier_message(
                    marker, "2:target.stage_artifact", token
                )
        local_commit_module._write_barrier_message(
            marker, "1:artifact.stage", token
        )
        with pytest.raises(TargetInspectionError) as replayed:
            await relay
        assert replayed.value.code is TargetErrorCode.WITNESS_STALE

    run(replay())


def test_patch_phase_7_worker_barrier_authenticates_success_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise every authenticated child-barrier terminal outcome."""
    token = bytes(range(32))

    def configure(label: str) -> tuple[Path, Path]:
        """Bind one fresh barrier session with a fixed child token."""
        marker = tmp_path / (label + "-marker")
        release = tmp_path / (label + "-release")
        monkeypatch.setenv(
            local_commit_module._SEATBELT_BARRIER_ENV, str(marker)
        )
        monkeypatch.setenv(
            local_commit_module._SEATBELT_RELEASE_ENV, str(release)
        )
        monkeypatch.setenv(_WORKER_TOKEN_ENV, token.hex())
        monkeypatch.setattr(
            local_commit_module, "_SEATBELT_WORKER_SESSION", None
        )
        monkeypatch.setattr(
            local_commit_module, "_SEATBELT_WORKER_SEQUENCE", 0
        )
        return marker, release

    marker, release = configure("success")
    local_commit_module._write_barrier_message(
        release, "1:artifact.stage", token
    )
    local_commit_module._commit_barrier("artifact.stage")
    assert (
        local_commit_module._read_barrier_message(marker, token)
        == "1:artifact.stage"
    )
    local_commit_module._write_barrier_message(
        release, "2:target.stage_artifact", token
    )
    local_commit_module._commit_barrier("target.stage_artifact")

    outcomes: tuple[tuple[str, type[BaseException]], ...] = (
        (
            "failure:target:patch.metadata_denied:1:artifact.stage",
            TargetInspectionError,
        ),
        (
            "failure:artifact_unknown:0:1:artifact.stage",
            rooted_worker_module._ArtifactUncertainError,
        ),
        ("failure:os:0:1:artifact.stage", OSError),
        ("failure:os:17:1:artifact.stage", OSError),
        ("failure:invalid:0:1:artifact.stage", TargetInspectionError),
    )
    for index, (outcome, error_type) in enumerate(outcomes):
        _, release = configure("failure-" + str(index))
        local_commit_module._write_barrier_message(release, outcome, token)
        with pytest.raises(error_type):
            local_commit_module._commit_barrier("artifact.stage")

    _, release = configure("invalid-target")
    local_commit_module._write_barrier_message(
        release, "failure:target:not-a-code:1:artifact.stage", token
    )
    with pytest.raises(TargetInspectionError) as invalid_target:
        local_commit_module._commit_barrier("artifact.stage")
    assert invalid_target.value.code is TargetErrorCode.WITNESS_STALE

    configure("timeout")
    ticks = iter((0.0, 3.0))

    def elapsed() -> float:
        """Advance immediately beyond the bounded release deadline."""
        return next(ticks)

    monkeypatch.setattr(local_commit_module, "monotonic", elapsed)
    with pytest.raises(TargetInspectionError) as timeout:
        local_commit_module._commit_barrier("artifact.stage")
    assert timeout.value.code is TargetErrorCode.WITNESS_STALE

    monkeypatch.delenv(_WORKER_TOKEN_ENV)
    with pytest.raises(TargetInspectionError):
        local_commit_module._barrier_token()
    monkeypatch.setenv(_WORKER_TOKEN_ENV, "invalid")
    with pytest.raises(TargetInspectionError):
        local_commit_module._barrier_token()
    monkeypatch.setenv(_WORKER_TOKEN_ENV, "00")
    with pytest.raises(TargetInspectionError):
        local_commit_module._barrier_token()
    with pytest.raises(TargetInspectionError):
        local_commit_module._barrier_stage("0:artifact.stage")
    with pytest.raises(TargetInspectionError):
        local_commit_module._barrier_stage("1:not-a-stage")
    malformed = tmp_path / "malformed-marker"
    malformed.write_text("{}", encoding="utf-8")
    with pytest.raises(TargetInspectionError):
        local_commit_module._read_barrier_message(malformed, token)


def test_patch_phase_7_worker_barrier_rejects_invalid_continuations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject invalid stages, continuity, malformed release, and timeout."""
    token = bytes(range(32))

    def configure(label: str) -> tuple[Path, Path]:
        """Install one fresh signed worker barrier session."""
        marker = tmp_path / (label + "-marker")
        release = tmp_path / (label + "-release")
        monkeypatch.setenv(
            local_commit_module._SEATBELT_BARRIER_ENV, str(marker)
        )
        monkeypatch.setenv(
            local_commit_module._SEATBELT_RELEASE_ENV, str(release)
        )
        monkeypatch.setenv(_WORKER_TOKEN_ENV, token.hex())
        monkeypatch.setattr(
            local_commit_module, "_SEATBELT_WORKER_SESSION", None
        )
        monkeypatch.setattr(
            local_commit_module, "_SEATBELT_WORKER_SEQUENCE", 0
        )
        return marker, release

    configure("stage")
    with pytest.raises(TargetInspectionError):
        local_commit_module._commit_barrier("not-a-stage")

    marker, _ = configure("initial-replay")
    local_commit_module._write_barrier_message(
        marker, "1:artifact.stage", token
    )
    with pytest.raises(TargetInspectionError):
        local_commit_module._commit_barrier("artifact.stage")

    marker, _ = configure("missing-continuation")
    session = (
        str(marker),
        str(tmp_path / "missing-continuation-release"),
        token,
    )
    monkeypatch.setattr(
        local_commit_module, "_SEATBELT_WORKER_SESSION", session
    )
    monkeypatch.setattr(local_commit_module, "_SEATBELT_WORKER_SEQUENCE", 1)
    with pytest.raises(TargetInspectionError):
        local_commit_module._commit_barrier("artifact.stage")

    marker, release = configure("wrong-continuation")
    local_commit_module._write_barrier_message(
        marker, "2:artifact.stage", token
    )
    session = (str(marker), str(release), token)
    monkeypatch.setattr(
        local_commit_module, "_SEATBELT_WORKER_SESSION", session
    )
    monkeypatch.setattr(local_commit_module, "_SEATBELT_WORKER_SEQUENCE", 1)
    with pytest.raises(TargetInspectionError):
        local_commit_module._commit_barrier("artifact.stage")

    _, release = configure("malformed-release")
    local_commit_module._write_barrier_message(release, "malformed", token)
    ticks = iter((0.0, 0.0, 3.0))
    monkeypatch.setattr(local_commit_module, "monotonic", lambda: next(ticks))
    with pytest.raises(TargetInspectionError):
        local_commit_module._commit_barrier("artifact.stage")

    configure("timeout-sleep")
    ticks = iter((0.0, 0.0, 3.0))
    monkeypatch.setattr(local_commit_module, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        local_commit_module, "blocking_sleep", lambda _value: None
    )
    with pytest.raises(TargetInspectionError):
        local_commit_module._commit_barrier("artifact.stage")


def test_patch_phase_7_seatbelt_protocol_validates_bound_worker_journals(
    tmp_path: Path,
) -> None:
    """Authenticate the child payload and its complete rooted journal."""
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        command = await _seatbelt_command(profile, scope, "6")
        payload = _seatbelt_payload_value(command, profile, scope)
        checked_payload = local_commit_module._seatbelt_payload(payload)
        response = local_commit_module._seatbelt_worker_response(
            checked_payload
        )
        assert (tmp_path / "protocol.txt").read_bytes() == b"protocol\n"
        token = bytes(range(32))
        raw_response = dumps(response, separators=(",", ":")).encode()
        envelope = dumps(
            {
                "payload": response,
                "mac": digest(token, raw_response, "sha256").hex(),
            },
            separators=(",", ":"),
        ).encode()
        report = local_commit_module._decode_seatbelt_response(
            command, token, envelope
        )
        assert report.journal is not None
        assert report.journal.postcondition is PostconditionState.ESTABLISHED

        malformed = dict(response)
        malformed["state"] = WorkerState.LIVE.value
        bad_state = dumps(
            {
                "payload": malformed,
                "mac": (
                    digest(
                        token,
                        dumps(malformed, separators=(",", ":")).encode(),
                        "sha256",
                    ).hex()
                ),
            },
            separators=(",", ":"),
        ).encode()
        with pytest.raises(TargetInspectionError) as invalid_state:
            local_commit_module._decode_seatbelt_response(
                command, token, bad_state
            )
        assert (
            invalid_state.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
        )

    run(execute())
    with pytest.raises(ValueError):
        local_commit_module._seatbelt_response(None)
    with pytest.raises(ValueError):
        local_commit_module._seatbelt_response(
            {
                "artifacts": [],
                "postcondition": PostconditionState.ESTABLISHED.value,
                "state": WorkerState.SETTLED.value,
                "steps": ["invalid"],
            }
        )
    with pytest.raises(ValueError):
        local_commit_module._seatbelt_payload({"version": 1})


def test_patch_phase_7_seatbelt_worker_main_rejects_unbound_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject malformed child envelopes before a sealed command is decoded."""
    profile = _profile(tmp_path)

    async def command_and_payload() -> (
        tuple[SealedCommitCommand, dict[str, object]]
    ):
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        command = await _seatbelt_command(profile, scope, "7")
        return command, _seatbelt_payload_value(command, profile, scope)

    command, payload = run(command_and_payload())
    token = bytes(range(32))
    expected_response: dict[str, object] = {
        "artifacts": [],
        "postcondition": PostconditionState.ESTABLISHED.value,
        "state": WorkerState.SETTLED.value,
        "steps": [],
    }

    def response(value: object) -> dict[str, object]:
        """Return one bounded journal only after payload validation."""
        decoded = local_commit_module._seatbelt_payload(value)
        assert decoded["request_id"] == command.lease.request_id.value
        return expected_response

    monkeypatch.setenv(_WORKER_TOKEN_ENV, token.hex())
    monkeypatch.setattr(
        local_commit_module, "_seatbelt_worker_response", response
    )
    input_stream = _SeatbeltStream(_signed_seatbelt_message(token, payload))
    output_stream = _SeatbeltStream()
    monkeypatch.setattr(local_commit_module, "stdin", input_stream)
    monkeypatch.setattr(local_commit_module, "stdout", output_stream)
    assert local_commit_module._seatbelt_worker_main() == 0
    envelope = loads(output_stream.buffer.getvalue())
    assert isinstance(envelope, dict)
    assert envelope["payload"] == expected_response
    assert (
        envelope["mac"]
        == digest(
            token,
            dumps(expected_response, separators=(",", ":")).encode(),
            "sha256",
        ).hex()
    )

    monkeypatch.setattr(
        local_commit_module,
        "stdin",
        _SeatbeltStream(b'{"payload":{},"mac":"invalid"}'),
    )
    assert local_commit_module._seatbelt_worker_main() == 2
    monkeypatch.delenv(_WORKER_TOKEN_ENV)
    assert local_commit_module._seatbelt_worker_main() == 2


def test_patch_phase_7_local_commit_private_boundary_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settle private channel and Seatbelt setup failures without effects."""
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        command = await _seatbelt_command(profile, scope, "8")
        target = LocalCommitTarget(profile)
        channel = local_commit_module._Channel(target, scope)
        request_id = PatchRequestId("request_" + "d" * 16)
        assert (
            await channel.reconcile_local(request_id)
        ).state is WorkerState.LIVE

        async def pending() -> WorkerReport:
            """Remain pending beyond a zero-duration reconciliation check."""
            await async_sleep(1)
            return WorkerReport(WorkerState.LIVE, None)

        task = create_task(pending())
        channel._settlements[request_id] = task
        with monkeypatch.context() as patcher:
            patcher.setattr(
                local_commit_module, "_SEATBELT_BARRIER_TIMEOUT_SECONDS", 0.0
            )
            assert (
                await channel.reconcile_local(request_id)
            ).state is WorkerState.LIVE
        task.cancel()
        with pytest.raises(BaseException):
            await task

        stale_scope = replace(
            scope, identity=replace(scope.identity, mount_id="stale")
        )
        with pytest.raises(TargetInspectionError) as stale:
            target._require_scope(stale_scope)
        assert stale.value.code is TargetErrorCode.WITNESS_STALE

        mismatched_profile = replace(
            profile,
            identity=replace(profile.identity, mount_id="mismatch"),
        )
        mismatched_scope = replace(scope, identity=mismatched_profile.identity)
        failed = await LocalCommitTarget(mismatched_profile)._commit(
            mismatched_scope, command
        )
        assert failed.journal is not None
        assert all(
            item.state is CommitStepState.NOT_COMMITTED
            for item in failed.journal.steps
        )

        async def unavailable_subprocess(
            *arguments: object, **keywords: object
        ) -> object:
            """Reject child startup before a command reaches the filesystem."""
            del arguments, keywords
            raise OSError("Seatbelt unavailable")

        assert scope.root_witness is not None
        with monkeypatch.context() as patcher:
            patcher.setattr(
                local_commit_module,
                "create_subprocess_exec",
                unavailable_subprocess,
            )
            with pytest.raises(TargetInspectionError) as unavailable:
                await local_commit_module._commit_in_seatbelt(
                    command, profile, scope.root_witness
                )
        assert unavailable.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE

    run(execute())


def test_patch_phase_7_private_coordinator_authority_and_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject unowned rooted commands."""
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        command = await _seatbelt_command(profile, scope, "9")
        approvals = ApprovalService(
            _PHASE6["_Broker"](),
            _PHASE6["_Clock"](),
            _PHASE6["RuntimeGrantStore"](),
        )
        store = InMemoryCoordinatorStore(approvals)
        with pytest.raises(CoordinatorError) as rejected:
            await coordinator_module._issue_rooted_command_authority(
                command, store
            )
        assert rejected.value.code is CoordinatorErrorCode.INVARIANT

        coordinator = InMemoryPatchCoordinator(
            store,
            InMemoryLeaseManager(store),
            ScriptedReconciler(_PHASE6["_snapshot"]()),
        )
        reservation = await coordinator.reserve(
            RuntimeIdentity(
                command.plan.binding.subject,
                command.plan.binding.final.approval.route,
                RetransmissionKey("phase-seven-disconnect"),
            ),
            command.plan.binding.request_digest,
        )
        expected = cast(PatchResult, object())

        async def cancelled(
            self: InMemoryPatchCoordinator,
            observed: object,
            before_commit: bool,
        ) -> PatchResult:
            """Return a sentinel after the private disconnect checkpoint."""
            del self
            assert observed == reservation
            assert before_commit
            return expected

        monkeypatch.setattr(InMemoryPatchCoordinator, "cancel", cancelled)
        assert (
            await coordinator._disconnect_before_commit(reservation)
        ) is expected

    run(execute())


def test_patch_phase_7_local_commit_helper_rejections_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject malformed worker data and rooted helper race observations."""
    profile = _profile(tmp_path)

    async def command_and_scope() -> (
        tuple[SealedCommitCommand, ResolvedMutationScope]
    ):
        """Build one valid command whose private boundaries can be rejected."""
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        return await _seatbelt_command(profile, scope, "b"), scope

    command, scope = run(command_and_scope())
    assert scope.root_witness is not None
    with pytest.raises(TargetInspectionError):
        local_commit_module._commit_namespace(
            replace(profile, commit_namespace=None), scope.root_witness
        )
    with pytest.raises(TargetInspectionError):
        local_commit_module._commit_rooted(
            command,
            profile,
            replace(scope.root_witness, mount_id="stale-mount"),
        )
    with pytest.raises(TargetInspectionError):
        rooted_worker_module._descriptor_path(-1)

    token = bytes(range(32))
    for envelope in (b"[]", b"{}"):
        with pytest.raises(TargetInspectionError):
            local_commit_module._decode_seatbelt_response(
                command, token, envelope
            )
    valid_payload = _seatbelt_payload_value(command, profile, scope)
    bad_mac = dumps(
        {"payload": valid_payload, "mac": "0" * 64}, separators=(",", ":")
    ).encode()
    with pytest.raises(TargetInspectionError):
        local_commit_module._decode_seatbelt_response(command, token, bad_mac)

    malformed_command = dict(valid_payload)
    malformed_command["command"] = "%%%"
    with pytest.raises(TargetInspectionError):
        local_commit_module._seatbelt_worker_response(
            local_commit_module._seatbelt_payload(malformed_command)
        )
    wrong_fence = dict(valid_payload)
    wrong_fence["fence"] = 2
    with pytest.raises(TargetInspectionError):
        local_commit_module._seatbelt_worker_response(
            local_commit_module._seatbelt_payload(wrong_fence)
        )

    monkeypatch.setenv(_WORKER_TOKEN_ENV, token.hex())
    monkeypatch.setattr(local_commit_module, "stdout", _SeatbeltStream())
    monkeypatch.setattr(local_commit_module, "stdin", _SeatbeltStream(b"[]"))
    assert local_commit_module._seatbelt_worker_main() == 2
    monkeypatch.setattr(local_commit_module, "stdin", _SeatbeltStream(b"{}"))
    assert local_commit_module._seatbelt_worker_main() == 2

    parent_fd = open_fd(tmp_path, O_RDONLY)
    source = tmp_path / "short.txt"
    source.write_bytes(b"short")
    try:
        source_fd = open_fd(source, O_RDONLY)
        try:
            with pytest.raises(TargetInspectionError):
                rooted_worker_module._read_exact(source_fd, 6)
        finally:
            close(source_fd)
        with pytest.raises(TargetInspectionError):
            rooted_worker_module._absent(parent_fd, "short.txt")

        fixed_token = b"fixed-stage-token" * 2
        stage_name = ".avalan-patch-" + sha256(fixed_token).hexdigest()[:32]
        (tmp_path / stage_name).write_bytes(b"collision")
        monkeypatch.setattr(
            rooted_worker_module,
            "_validate_namespace_context",
            lambda *arguments, **keywords: None,
        )
        stage_path = LogicalPath("short.txt")
        with monkeypatch.context() as patcher:
            patcher.setattr(
                rooted_worker_module, "token_bytes", lambda _size: fixed_token
            )
            with pytest.raises(TargetInspectionError):
                rooted_worker_module._stage(
                    parent_fd, b"value", 0o600, path=stage_path
                )
        (tmp_path / stage_name).unlink()

        def stalled_write(descriptor: int, value: bytes) -> int:
            """Report no forward progress for one private staging write."""
            del descriptor, value
            return 0

        with monkeypatch.context() as patcher:
            patcher.setattr(rooted_worker_module, "write_fd", stalled_write)
            with pytest.raises(OSError):
                rooted_worker_module._stage(
                    parent_fd, b"value", 0o600, path=stage_path
                )

        def failed_cleanup(name: str, *, dir_fd: int) -> None:
            """Make a failed stage cleanup explicitly uncertain."""
            del name, dir_fd
            raise OSError("cleanup unavailable")

        with monkeypatch.context() as patcher:
            patcher.setattr(rooted_worker_module, "write_fd", stalled_write)
            patcher.setattr(rooted_worker_module, "unlink", failed_cleanup)
            with pytest.raises(rooted_worker_module._ArtifactUncertainError):
                rooted_worker_module._stage(
                    parent_fd, b"value", 0o600, path=stage_path
                )
    finally:
        close(parent_fd)


def test_patch_phase_7_local_worker_protocol_and_rooted_error_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settle private worker failures fail closed."""
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        command = await _seatbelt_command(profile, scope, "e")
        target = LocalCommitTarget(profile)

        async def limited_handshake(
            self: LocalCommitTarget, observed: ResolvedMutationScope
        ) -> TargetHandshake:
            """Advertise no mutations after a completed trusted scope check."""
            del self
            return TargetHandshake(
                profile.identity,
                frozenset(),
                (),
                (),
                LocalPlatformProfile.DARWIN,
                worker=observed.worker,
            )

        with monkeypatch.context() as patcher:
            patcher.setattr(LocalCommitTarget, "handshake", limited_handshake)
            with pytest.raises(TargetInspectionError):
                await target.worker(scope)

        changed_profile = replace(
            profile,
            identity=replace(profile.identity, mount_id="mismatched-target"),
        )
        changed_target = LocalCommitTarget(changed_profile)
        with monkeypatch.context() as patcher:
            patcher.setattr(
                LocalCommitTarget, "_require_scope", lambda _self, _scope: None
            )
            failed = await changed_target._commit(scope, command)
        assert failed.journal is not None
        assert all(
            item.state is CommitStepState.NOT_COMMITTED
            for item in failed.journal.steps
        )

        async def unavailable_commit(
            _command: SealedCommitCommand,
            _profile: LocalTargetProfile,
            _witness: target_module.RootWitness,
        ) -> WorkerReport:
            """Raise a native failure before a child report exists."""
            raise OSError("worker unavailable")

        with monkeypatch.context() as patcher:
            patcher.setattr(
                local_commit_module, "_commit_in_seatbelt", unavailable_commit
            )
            uncertain = await target._commit(scope, command)
        assert uncertain.journal is not None
        assert all(
            item.state is CommitStepState.UNKNOWN
            for item in uncertain.journal.steps
        )

        response = {
            "artifacts": [],
            "postcondition": PostconditionState.ESTABLISHED.value,
            "state": WorkerState.SETTLED.value,
            "steps": [],
        }
        token = bytes(range(32))
        envelope = dumps(
            {
                "payload": response,
                "mac": (
                    digest(
                        token,
                        dumps(response, separators=(",", ":")).encode(),
                        "sha256",
                    ).hex()
                ),
            },
            separators=(",", ":"),
        ).encode()
        with pytest.raises(TargetInspectionError):
            local_commit_module._decode_seatbelt_response(
                command, token, envelope
            )

        payload = _seatbelt_payload_value(command, profile, scope)
        invalid_payload = dict(payload)
        invalid_payload["version"] = 2
        with pytest.raises(ValueError):
            local_commit_module._seatbelt_payload(invalid_payload)

        output_stream = _SeatbeltStream()
        monkeypatch.setenv(_WORKER_TOKEN_ENV, token.hex())
        monkeypatch.setattr(local_commit_module, "stdout", output_stream)
        monkeypatch.setattr(
            local_commit_module,
            "stdin",
            _SeatbeltStream(_signed_seatbelt_message(token, payload)),
        )

        def failed_worker_response(value: object) -> dict[str, object]:
            """Raise a native worker error after HMAC validation."""
            del value
            raise OSError("response unavailable")

        monkeypatch.setattr(
            local_commit_module,
            "_seatbelt_worker_response",
            failed_worker_response,
        )
        assert local_commit_module._seatbelt_worker_main() == 2

        assert scope.root_witness is not None
        with monkeypatch.context() as patcher:

            def fail_lineage(*arguments: object, **keywords: object) -> None:
                """Raise one target validation failure."""
                del arguments, keywords
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)

            patcher.setattr(
                rooted_worker_module, "_commit_lineage", fail_lineage
            )
            rejected = local_commit_module._commit_rooted(
                command, profile, scope.root_witness
            )
        assert rejected.journal is not None
        assert rejected.journal.postcondition is PostconditionState.UNKNOWN

        with monkeypatch.context() as patcher:

            def fail_lineage_os(
                *arguments: object, **keywords: object
            ) -> None:
                """Raise one uncertain native effect."""
                del arguments, keywords
                raise OSError("effect unavailable")

            patcher.setattr(
                rooted_worker_module, "_commit_lineage", fail_lineage_os
            )
            uncertain = local_commit_module._commit_rooted(
                command, profile, scope.root_witness
            )
        assert uncertain.journal is not None
        assert uncertain.journal.steps[0].state is CommitStepState.UNKNOWN

        with monkeypatch.context() as patcher:

            def no_lineage(*arguments: object, **keywords: object) -> None:
                """Leave the synthetic mismatched step inventory unchanged."""
                del arguments, keywords

            patcher.setattr(
                rooted_worker_module, "_steps", lambda _command: ()
            )
            patcher.setattr(
                rooted_worker_module, "_commit_lineage", no_lineage
            )
            with pytest.raises(TargetInspectionError):
                local_commit_module._commit_rooted(
                    command, profile, scope.root_witness
                )

        with monkeypatch.context() as patcher:

            def verification_error(
                *arguments: object, **keywords: object
            ) -> PostconditionState:
                """Raise one native verification failure."""
                del arguments, keywords
                raise OSError("verification unavailable")

            patcher.setattr(
                rooted_worker_module, "_verify", verification_error
            )
            report = local_commit_module._commit_rooted(
                command, profile, scope.root_witness
            )
        assert report.journal is not None
        assert report.journal.postcondition is PostconditionState.UNKNOWN

    run(execute())


def test_patch_phase_7_rooted_parent_and_precondition_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject stale parents, expected files, and namespace observations."""
    (tmp_path / "sub").mkdir()
    profile = _profile(tmp_path)

    async def command_and_scope() -> (
        tuple[SealedCommitCommand, ResolvedMutationScope]
    ):
        """Build one sealed final file used by direct rooted helper checks."""
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        return await _seatbelt_command(profile, scope, "f"), scope

    command, scope = run(command_and_scope())
    assert scope.root_witness is not None
    witness = scope.root_witness
    planned = command.plan.candidate.lineages[0].final
    assert planned.bytes_value is not None
    (tmp_path / "protocol.txt").write_bytes(planned.bytes_value._value)
    root_fd = open_fd(tmp_path, O_RDONLY)
    status = fstat(root_fd)
    identity = target_module.FileIdentity(status.st_dev, status.st_ino)
    root_token = rooted_worker_module._ROOT_DESCRIPTOR.set(root_fd)
    parents_token = rooted_worker_module._PARENT_IDENTITIES.set(
        {
            None: identity,
            LogicalPath("sub"): target_module.FileIdentity(
                (tmp_path / "sub").stat().st_dev,
                (tmp_path / "sub").stat().st_ino,
            ),
        }
    )
    try:
        parent, leaf = rooted_worker_module._parent(
            root_fd, identity, witness, LogicalPath("sub/file.txt")
        )
        assert leaf == "file.txt"
        close(parent)
        with pytest.raises(TargetInspectionError):
            rooted_worker_module._parent(
                root_fd,
                target_module.FileIdentity(
                    identity.device + 1, identity.inode
                ),
                witness,
                LogicalPath("protocol.txt"),
            )
        mismatch = rooted_worker_module._PARENT_IDENTITIES.set(
            {
                None: target_module.FileIdentity(
                    identity.device + 1, identity.inode
                )
            }
        )
        try:
            with pytest.raises(TargetInspectionError):
                rooted_worker_module._parent(
                    root_fd, identity, witness, LogicalPath("protocol.txt")
                )
        finally:
            rooted_worker_module._PARENT_IDENTITIES.reset(mismatch)
        absent_root = rooted_worker_module._ROOT_DESCRIPTOR.set(None)
        try:
            with pytest.raises(TargetInspectionError):
                rooted_worker_module._parent(
                    root_fd, identity, witness, LogicalPath("protocol.txt")
                )
        finally:
            rooted_worker_module._ROOT_DESCRIPTOR.reset(absent_root)
        with monkeypatch.context() as patcher:
            patcher.setattr(
                rooted_worker_module, "_rebind_parent", lambda *_args: False
            )
            with pytest.raises(TargetInspectionError):
                rooted_worker_module._parent(
                    root_fd, identity, witness, LogicalPath("protocol.txt")
                )

        with pytest.raises(TargetInspectionError):
            rooted_worker_module._expected(
                root_fd,
                "protocol.txt",
                replace(
                    planned,
                    present=False,
                    bytes_value=None,
                    metadata=None,
                    digest=None,
                    size=ByteSize(0),
                ),
            )
        with pytest.raises(TargetInspectionError):
            rooted_worker_module._expected(root_fd, "missing.txt", planned)
        with pytest.raises(TargetInspectionError):
            rooted_worker_module._expected(
                root_fd,
                "protocol.txt",
                replace(planned, size=ByteSize(planned.size.value + 1)),
            )
        mismatched_bytes = replace(
            planned, bytes_value=SourceBytes(b"mismatched\n")
        )
        with pytest.raises(TargetInspectionError):
            rooted_worker_module._expected(
                root_fd, "protocol.txt", mismatched_bytes
            )

        with pytest.raises(TargetInspectionError):
            rooted_worker_module._validate_namespace_context(
                root_fd, LogicalPath("protocol.txt"), ()
            )
        context = rooted_worker_module._CommitContext(
            root_fd, root_fd, identity, witness, tmp_path
        )
        context_token = rooted_worker_module._COMMIT_CONTEXT.set(context)
        try:
            rooted_worker_module._validate_namespace_context(
                root_fd, LogicalPath("protocol.txt"), ()
            )
            missing_root = replace(context, root_path=tmp_path / "missing")
            changed_context = rooted_worker_module._COMMIT_CONTEXT.set(
                missing_root
            )
            try:
                with pytest.raises(TargetInspectionError):
                    rooted_worker_module._validate_namespace_context(
                        root_fd, LogicalPath("protocol.txt"), ()
                    )
            finally:
                rooted_worker_module._COMMIT_CONTEXT.reset(changed_context)
            stale_root = replace(
                context,
                root=replace(
                    witness,
                    identity=target_module.FileIdentity(
                        identity.device + 1, identity.inode
                    ),
                ),
            )
            changed_context = rooted_worker_module._COMMIT_CONTEXT.set(
                stale_root
            )
            try:
                with pytest.raises(TargetInspectionError):
                    rooted_worker_module._validate_namespace_context(
                        root_fd, LogicalPath("protocol.txt"), ()
                    )
            finally:
                rooted_worker_module._COMMIT_CONTEXT.reset(changed_context)
            with pytest.raises(TargetInspectionError):
                rooted_worker_module._validate_namespace_context(
                    root_fd,
                    LogicalPath("protocol.txt"),
                    (
                        (
                            root_fd,
                            LogicalPath("protocol.txt"),
                            "missing.txt",
                            root_fd,
                        ),
                    ),
                )
            missing_parent = rooted_worker_module._PARENT_IDENTITIES.set({})
            try:
                with pytest.raises(TargetInspectionError):
                    rooted_worker_module._validate_parent_context(
                        root_fd, LogicalPath("protocol.txt"), context
                    )
            finally:
                rooted_worker_module._PARENT_IDENTITIES.reset(missing_parent)
        finally:
            rooted_worker_module._COMMIT_CONTEXT.reset(context_token)
    finally:
        rooted_worker_module._PARENT_IDENTITIES.reset(parents_token)
        rooted_worker_module._ROOT_DESCRIPTOR.reset(root_token)
        close(root_fd)

    class EmptyDescriptorPath:
        """Return a successful F_GETPATH status without a path payload."""

        def fcntl(self, descriptor: int, command: int, buffer: object) -> int:
            """Leave an empty stale path."""
            del descriptor, command, buffer
            return 0

    with monkeypatch.context() as patcher:
        patcher.setattr(rooted_worker_module, "_LIBC", EmptyDescriptorPath())
        with pytest.raises(TargetInspectionError):
            rooted_worker_module._descriptor_path(0)


def test_patch_phase_7_private_publication_failures_preserve_journal_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Classify private staging and publication failures without retrying."""
    parent_fd = open_fd(tmp_path, O_RDONLY)
    path = LogicalPath("published.txt")
    metadata = target_module._ProtectedMetadata((), 0, None)
    try:

        def uncertain_stage(*arguments: object, **keywords: object) -> str:
            """Report an artifact whose cleanup outcome cannot be observed."""
            del arguments, keywords
            raise rooted_worker_module._ArtifactUncertainError("uncertain")

        artifact_states = [ArtifactState.ABSENT]
        with monkeypatch.context() as patcher:
            patcher.setattr(rooted_worker_module, "_stage", uncertain_stage)
            with pytest.raises(rooted_worker_module._ArtifactUncertainError):
                rooted_worker_module._publish_new(
                    parent_fd,
                    path,
                    "published.txt",
                    b"value\n",
                    0o600,
                    artifact_states,
                    0,
                )
        assert artifact_states[0] is ArtifactState.UNKNOWN

        artifact_states = [ArtifactState.ABSENT]
        with monkeypatch.context() as patcher:
            patcher.setattr(rooted_worker_module, "_stage", uncertain_stage)
            with pytest.raises(rooted_worker_module._ArtifactUncertainError):
                rooted_worker_module._publish_update(
                    parent_fd,
                    path,
                    "published.txt",
                    b"value\n",
                    0o600,
                    metadata,
                    artifact_states,
                    0,
                    parent_fd,
                )
        assert artifact_states[0] is ArtifactState.UNKNOWN

        def staged_file(name: str) -> Path:
            """Create one known private artifact for a publication branch."""
            artifact = tmp_path / name
            artifact.write_bytes(b"value\n")
            return artifact

        def direct_effect(
            _parent: int,
            _path: LogicalPath,
            effect: Callable[[], None],
            **_keywords: object,
        ) -> None:
            """Run one supplied namespace effect without revalidation setup."""
            effect()

        artifact = staged_file("file-exists-stage")

        def occupied_link(*arguments: object, **keywords: object) -> None:
            """Report a raced destination."""
            del arguments, keywords
            raise FileExistsError

        with monkeypatch.context() as patcher:
            patcher.setattr(
                rooted_worker_module,
                "_stage",
                lambda *_args, **_kwargs: artifact.name,
            )
            patcher.setattr(
                rooted_worker_module, "_namespace_effect", direct_effect
            )
            patcher.setattr(rooted_worker_module, "link", occupied_link)
            with pytest.raises(TargetInspectionError):
                rooted_worker_module._publish_new(
                    parent_fd,
                    path,
                    "published.txt",
                    b"value\n",
                    0o600,
                    [ArtifactState.ABSENT],
                    0,
                )
        assert not artifact.exists()

        artifact = staged_file("unsupported-stage")

        def unsupported_link(*arguments: object, **keywords: object) -> None:
            """Report a required native link primitive as unavailable."""
            del arguments, keywords
            raise OSError(ENOSYS, "link unavailable")

        with monkeypatch.context() as patcher:
            patcher.setattr(
                rooted_worker_module,
                "_stage",
                lambda *_args, **_kwargs: artifact.name,
            )
            patcher.setattr(
                rooted_worker_module, "_namespace_effect", direct_effect
            )
            patcher.setattr(rooted_worker_module, "link", unsupported_link)
            with pytest.raises(TargetInspectionError):
                rooted_worker_module._publish_new(
                    parent_fd,
                    path,
                    "published.txt",
                    b"value\n",
                    0o600,
                    [ArtifactState.ABSENT],
                    0,
                )
        assert not artifact.exists()

        artifact = staged_file("leaked-stage")
        effects = 0

        def leaked_cleanup(
            _parent: int,
            _path: LogicalPath,
            effect: Callable[[], None],
            **_keywords: object,
        ) -> None:
            """Apply publication and lose cleanup."""
            nonlocal effects
            effects += 1
            if effects == 1:
                effect()
                return
            raise OSError("cleanup unavailable")

        artifact_states = [ArtifactState.ABSENT]
        with monkeypatch.context() as patcher:
            patcher.setattr(
                rooted_worker_module,
                "_stage",
                lambda *_args, **_kwargs: artifact.name,
            )
            patcher.setattr(
                rooted_worker_module, "_namespace_effect", leaked_cleanup
            )
            rooted_worker_module._publish_new(
                parent_fd,
                path,
                "published.txt",
                b"value\n",
                0o600,
                artifact_states,
                0,
            )
        assert artifact_states[0] is ArtifactState.LEAKED
        assert artifact.exists()
        artifact.unlink()
        (tmp_path / "published.txt").unlink()

        artifact = staged_file("update-stage")
        expected = tmp_path / "published.txt"
        expected.write_bytes(b"before\n")
        expected_fd = open_fd(expected, O_RDONLY)
        try:
            artifact_states = [ArtifactState.ABSENT]

            def failed_update(
                _parent: int,
                _path: LogicalPath,
                _effect: Callable[[], None],
                **_keywords: object,
            ) -> None:
                """Lose both replacement and its staged cleanup observation."""
                raise OSError("replace unavailable")

            with monkeypatch.context() as patcher:
                patcher.setattr(
                    rooted_worker_module,
                    "_stage",
                    lambda *_args, **_kwargs: artifact.name,
                )
                patcher.setattr(
                    rooted_worker_module, "_namespace_effect", failed_update
                )
                with pytest.raises(OSError):
                    rooted_worker_module._publish_update(
                        parent_fd,
                        path,
                        "published.txt",
                        b"value\n",
                        0o600,
                        metadata,
                        artifact_states,
                        0,
                        expected_fd,
                    )
            assert artifact_states[0] is ArtifactState.LEAKED
        finally:
            close(expected_fd)
        artifact.unlink()
        expected.unlink()
    finally:
        close(parent_fd)


def test_patch_phase_7_local_worker_micro_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover the remaining local worker protocol and helper failures."""
    profile = _profile(tmp_path)

    async def command_and_scope() -> (
        tuple[SealedCommitCommand, ResolvedMutationScope]
    ):
        """Build one sealed command for child-protocol rejection paths."""
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        return await _seatbelt_command(profile, scope, "a"), scope

    command, scope = run(command_and_scope())
    token = bytes(range(32))

    class CancelledProcess:
        """Raise cancellation while a private child is still communicating."""

        returncode = 0

        async def communicate(self, value: bytes) -> tuple[bytes, bytes]:
            """Cancel before a worker response can be accepted."""
            del value
            raise CancelledError

    async def cancelled_subprocess(
        *arguments: object, **keywords: object
    ) -> object:
        """Return one cancelling Seatbelt process double."""
        del arguments, keywords
        return CancelledProcess()

    async def idle_relay(marker: Path, release: Path, value: bytes) -> None:
        """Wait until parent cleanup cancels the private barrier relay."""
        del marker, release, value
        await async_sleep(10)

    assert scope.root_witness is not None
    with monkeypatch.context() as patcher:
        patcher.setattr(
            local_commit_module, "create_subprocess_exec", cancelled_subprocess
        )
        patcher.setattr(
            local_commit_module, "_relay_seatbelt_barriers", idle_relay
        )
        with pytest.raises(CancelledError):
            run(
                local_commit_module._commit_in_seatbelt(
                    command, profile, scope.root_witness
                )
            )

    marker = tmp_path / "relay-marker"
    release = tmp_path / "relay-release"
    local_commit_module._write_barrier_message(
        marker, "1:artifact.stage", token
    )

    async def direct_thread(
        function: _ThreadFunction, *arguments: object
    ) -> object:
        """Run the relay's bounded synchronous helpers in this test task."""
        return function(*arguments)

    async def cancelled_sleep(value: float) -> None:
        """Stop the relay after its first authenticated failure response."""
        del value
        raise CancelledError

    def uncertain_barrier(stage: str) -> None:
        """Report one authenticated artifact-uncertain child barrier."""
        assert stage == "artifact.stage"
        raise rooted_worker_module._ArtifactUncertainError("uncertain")

    with monkeypatch.context() as patcher:
        patcher.setattr(local_commit_module, "to_thread", direct_thread)
        patcher.setattr(local_commit_module, "sleep", cancelled_sleep)
        patcher.setattr(
            local_commit_module, "_commit_barrier", uncertain_barrier
        )
        with pytest.raises(CancelledError):
            run(
                local_commit_module._relay_seatbelt_barriers(
                    marker, release, token
                )
            )
    assert (
        local_commit_module._read_barrier_message(release, token)
        == "failure:artifact_unknown:0:1:artifact.stage"
    )

    marker_path = tmp_path / "write-marker"
    temporary_path = marker_path.with_name(marker_path.name + ".next")

    def stalled_write(descriptor: int, value: bytes) -> int:
        """Report no authenticated marker-write progress."""
        del descriptor, value
        return 0

    def absent_cleanup(self: Path) -> None:
        """Simulate a concurrently removed temporary marker file."""
        del self
        raise FileNotFoundError

    with monkeypatch.context() as patcher:
        patcher.setattr(local_commit_module, "write_fd", stalled_write)
        patcher.setattr(Path, "unlink", absent_cleanup)
        with pytest.raises(OSError):
            local_commit_module._write_barrier_message(
                marker_path, "1:artifact.stage", token
            )
    assert temporary_path.exists()
    temporary_path.unlink()

    step_id, lineage_id = rooted_worker_module._steps(command)[0]
    mismatched_response = {
        "artifacts": [],
        "postcondition": PostconditionState.ESTABLISHED.value,
        "state": WorkerState.SETTLED.value,
        "steps": [
            {
                "id": step_id.value,
                "lineage": lineage_id.value,
                "state": CommitStepState.NOT_COMMITTED.value,
            }
        ],
    }
    envelope = dumps(
        {
            "payload": mismatched_response,
            "mac": (
                digest(
                    token,
                    dumps(mismatched_response, separators=(",", ":")).encode(),
                    "sha256",
                ).hex()
            ),
        },
        separators=(",", ":"),
    ).encode()
    with pytest.raises(TargetInspectionError):
        local_commit_module._decode_seatbelt_response(command, token, envelope)


def test_patch_phase_7_final_local_commit_failure_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject every remaining rooted worker defensive failure branch."""
    profile = _profile(tmp_path)

    async def commands() -> tuple[
        ResolvedMutationScope,
        SealedCommitCommand,
        SealedPlan,
        SealedPlan,
        SealedPlan,
        SealedPlan,
        SealedPlan,
    ]:
        """Seal one command for each isolated rooted rejection shape."""
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        command = await _seatbelt_command(profile, scope, "b")
        (tmp_path / "update.txt").write_bytes(b"before\n")
        (tmp_path / "executable.txt").write_bytes(b"before\n")
        chmod(tmp_path / "executable.txt", 0o744)
        (tmp_path / "move.txt").write_bytes(b"move\n")
        (tmp_path / "move-executable.txt").write_bytes(b"move\n")
        chmod(tmp_path / "move-executable.txt", 0o744)
        (tmp_path / "delete.txt").write_bytes(b"delete\n")
        update = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: update.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            ),
            {"update.txt": b"before\n"},
        )
        executable = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: executable.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            ),
            {"executable.txt": b"before\n"},
            {"executable.txt": FileMode(0o744)},
        )
        move = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: move.txt",
                    "*** Move to: moved.txt",
                    "*** End Patch",
                )
            ),
            {"move.txt": b"move\n"},
        )
        move_executable = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: move-executable.txt",
                    "*** Move to: moved-executable.txt",
                    "*** End Patch",
                )
            ),
            {"move-executable.txt": b"move\n"},
            {"move-executable.txt": FileMode(0o744)},
        )
        deletion = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Delete File: delete.txt",
                    "*** End Patch",
                )
            ),
            {"delete.txt": b"delete\n"},
        )
        return (
            scope,
            command,
            update,
            executable,
            move,
            move_executable,
            deletion,
        )

    (
        scope,
        command,
        update,
        executable,
        move,
        move_executable,
        deletion,
    ) = run(commands())
    assert scope.root_witness is not None
    witness = scope.root_witness

    def command_for(plan: SealedPlan) -> SealedCommitCommand:
        """Bind one replacement sealed candidate to the same local fence."""
        return SealedCommitCommand(plan, command.lease, footprint_for(plan))

    def replace_lineage(
        plan: SealedPlan, lineage: PlannedLineage
    ) -> SealedCommitCommand:
        """Return one command with a deliberately rejected lineage shape."""
        return command_for(
            replace(
                plan,
                candidate=replace(
                    plan.candidate,
                    lineages=(lineage,),
                ),
            )
        )

    async def direct_thread(
        function: _ThreadFunction, *arguments: object
    ) -> object:
        """Run the one bounded barrier reader without an executor thread."""
        return function(*arguments)

    token = bytes(range(32))
    skipped_marker = tmp_path / "skipped-marker"
    local_commit_module._write_barrier_message(
        skipped_marker, "2:artifact.stage", token
    )
    with monkeypatch.context() as patcher:
        patcher.setattr(local_commit_module, "to_thread", direct_thread)
        with pytest.raises(TargetInspectionError):
            run(
                local_commit_module._relay_seatbelt_barriers(
                    skipped_marker, tmp_path / "skipped-release", token
                )
            )

    add_lineage = command.plan.candidate.lineages[0]
    nested_add = replace(
        add_lineage, destination_path=LogicalPath("sub/protocol.txt")
    )
    nested_command = replace_lineage(command.plan, nested_add)
    with pytest.raises(TargetInspectionError):
        local_commit_module._commit_rooted(nested_command, profile, witness)

    update_lineage = update.candidate.lineages[0]
    no_protected_update = replace(
        update_lineage,
        initial=replace(update_lineage.initial, protected_metadata=None),
    )
    assert (
        local_commit_module._commit_rooted(
            replace_lineage(update, no_protected_update), profile, witness
        ).journal
        is not None
    )

    executable_command = command_for(
        replace(
            executable,
            binding=replace(
                executable.binding,
                final=replace(
                    executable.binding.final,
                    effects=frozenset((Capability.UPDATE,)),
                ),
            ),
        )
    )
    assert (
        local_commit_module._commit_rooted(
            executable_command, profile, witness
        ).journal
        is not None
    )

    malformed_update = replace(
        update_lineage, source_path=None, destination_path=None
    )
    malformed_command = replace_lineage(update, malformed_update)
    assert (
        local_commit_module._commit_rooted(
            malformed_command, profile, witness
        ).journal
        is not None
    )

    move_lineage = move.candidate.lineages[0]
    no_protected_move = replace(
        move_lineage,
        initial=replace(move_lineage.initial, protected_metadata=None),
    )
    assert (
        local_commit_module._commit_rooted(
            replace_lineage(move, no_protected_move), profile, witness
        ).journal
        is not None
    )

    move_executable_command = command_for(
        replace(
            move_executable,
            binding=replace(
                move_executable.binding,
                final=replace(
                    move_executable.binding.final,
                    effects=frozenset((Capability.UPDATE,)),
                ),
            ),
        )
    )
    assert (
        local_commit_module._commit_rooted(
            move_executable_command, profile, witness
        ).journal
        is not None
    )

    def unavailable_link(*arguments: object, **keywords: object) -> None:
        """Report an unsupported atomic move link primitive."""
        del arguments, keywords
        raise OSError(ENOSYS, "link unavailable")

    with monkeypatch.context() as patcher:
        patcher.setattr(rooted_worker_module, "link", unavailable_link)
        assert (
            local_commit_module._commit_rooted(
                command_for(move), profile, witness
            ).journal
            is not None
        )

    def generic_link(*arguments: object, **keywords: object) -> None:
        """Report one non-classified atomic move link failure."""
        del arguments, keywords
        raise OSError("link unavailable")

    with monkeypatch.context() as patcher:
        patcher.setattr(rooted_worker_module, "link", generic_link)
        assert (
            local_commit_module._commit_rooted(
                command_for(move), profile, witness
            ).journal
            is not None
        )

    parent_fd = open_fd(tmp_path, O_RDONLY)
    generic_stage = tmp_path / "generic-stage"
    generic_stage.write_bytes(b"value\n")

    def direct_effect(
        _parent: int,
        _path: LogicalPath,
        effect: Callable[[], None],
        **_keywords: object,
    ) -> None:
        """Run a namespace effect without unrelated rooted setup."""
        effect()

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(
                rooted_worker_module,
                "_stage",
                lambda *_args, **_kwargs: generic_stage.name,
            )
            patcher.setattr(
                rooted_worker_module, "_namespace_effect", direct_effect
            )
            patcher.setattr(rooted_worker_module, "link", generic_link)
            with pytest.raises(OSError):
                rooted_worker_module._publish_new(
                    parent_fd,
                    LogicalPath("generic.txt"),
                    "generic.txt",
                    b"value\n",
                    0o600,
                    [ArtifactState.ABSENT],
                    0,
                )
    finally:
        close(parent_fd)
    assert not generic_stage.exists()

    root_fd = open_fd(tmp_path, O_RDONLY)
    root_status = fstat(root_fd)
    root_identity = target_module.FileIdentity(
        root_status.st_dev, root_status.st_ino
    )
    root_token = rooted_worker_module._ROOT_DESCRIPTOR.set(root_fd)
    parents_token = rooted_worker_module._PARENT_IDENTITIES.set(
        {None: root_identity}
    )
    context_token = rooted_worker_module._COMMIT_CONTEXT.set(
        rooted_worker_module._CommitContext(
            root_fd, root_fd, root_identity, witness, tmp_path
        )
    )
    try:

        def failed_child(*arguments: object, **keywords: object) -> int:
            """Reject one retained-parent rebind before identity comparison."""
            del arguments, keywords
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)

        with monkeypatch.context() as patcher:
            patcher.setattr(
                rooted_worker_module, "_open_child_directory", failed_child
            )
            assert not rooted_worker_module._rebind_parent(
                root_fd,
                root_identity,
                witness,
                LogicalPath("sub/file.txt"),
                root_fd,
            )

        source = tmp_path / "source-validation.txt"
        source.write_bytes(b"source\n")
        with pytest.raises(TargetInspectionError):
            rooted_worker_module._validate_namespace_context(
                root_fd,
                LogicalPath("source-validation.txt"),
                (
                    (
                        root_fd,
                        LogicalPath("source-validation.txt"),
                        "source-validation.txt",
                        root_fd,
                    ),
                ),
            )

        assert (
            rooted_worker_module._verify(
                malformed_command, root_fd, root_identity, witness
            )
            is PostconditionState.UNKNOWN
        )
        assert (
            rooted_worker_module._verify(
                command_for(deletion), root_fd, root_identity, witness
            )
            is PostconditionState.SUPERSEDED
        )
        assert (
            rooted_worker_module._verify(
                command, root_fd, root_identity, witness
            )
            is PostconditionState.SUPERSEDED
        )
    finally:
        rooted_worker_module._COMMIT_CONTEXT.reset(context_token)
        rooted_worker_module._PARENT_IDENTITIES.reset(parents_token)
        rooted_worker_module._ROOT_DESCRIPTOR.reset(root_token)
        close(root_fd)


@pytest.mark.parametrize("failure", ("write", "fsync", "replace"))
def test_patch_phase_7_barrier_message_failure_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Remove partial authenticated markers for write publication faults."""
    marker = tmp_path / "marker"
    temporary = tmp_path / "marker.next"
    token = bytes(range(32))

    def fail_write(descriptor: int, value: bytes) -> int:
        """Fail the marker payload write before publication."""
        del descriptor, value
        raise OSError("injected marker write failure")

    def fail_fsync(descriptor: int) -> None:
        """Fail the marker durability sync before publication."""
        del descriptor
        raise OSError("injected marker fsync failure")

    def fail_replace(source: Path, destination: Path) -> None:
        """Fail the atomic marker publication after its temporary write."""
        del source, destination
        raise OSError("injected marker replace failure")

    match failure:
        case "write":
            monkeypatch.setattr(local_commit_module, "write_fd", fail_write)
        case "fsync":
            monkeypatch.setattr(local_commit_module, "fsync", fail_fsync)
        case "replace":
            monkeypatch.setattr(local_commit_module, "replace", fail_replace)
        case _:
            raise AssertionError(failure)
    with pytest.raises(OSError):
        local_commit_module._write_barrier_message(
            marker, "1:artifact.stage", token
        )
    assert not temporary.exists()
    assert not marker.exists()


def test_patch_phase_7_failed_relay_still_removes_child_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clean private marker paths even when the relay has already failed."""
    profile = _profile(tmp_path)

    class FailedRelayProcess:
        """Return from one child call after allowing the relay to fail."""

        returncode = 0

        async def communicate(self, message: bytes) -> tuple[bytes, bytes]:
            """Yield once so the scheduled relay can raise its failure."""
            del message
            await async_sleep(0)
            return b"", b""

    async def failed_relay(marker: Path, release: Path, token: bytes) -> None:
        """Raise one non-cancellation relay failure after child startup."""
        del marker, release, token
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)

    async def fake_subprocess(
        *arguments: object, **keywords: object
    ) -> object:
        """Return a bounded child-process double without a host mutation."""
        del arguments, keywords
        return FailedRelayProcess()

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: created.txt",
                    "+created",
                    "*** End Patch",
                )
            ),
            {},
        )
        assert scope.root_witness is not None
        command = SealedCommitCommand(
            sealed,
            CommitLease(
                profile.identity.domain_id,
                PatchRequestId("request_" + "f" * 16),
                1,
            ),
            footprint_for(sealed),
        )
        with pytest.raises(TargetInspectionError) as error:
            await local_commit_module._commit_in_seatbelt(
                command, profile, scope.root_witness
            )
        assert error.value.code is TargetErrorCode.WITNESS_STALE

    monkeypatch.setattr(
        local_commit_module, "create_subprocess_exec", fake_subprocess
    )
    monkeypatch.setattr(
        local_commit_module, "_relay_seatbelt_barriers", failed_relay
    )
    run(execute())
    assert profile.commit_namespace is not None
    assert not tuple(profile.commit_namespace.iterdir())


def test_patch_phase_7_generic_posix_profile_never_claims_darwin_commit(
    tmp_path: Path,
) -> None:
    """Keep a generic POSIX scope inspection-only without Darwin receipts."""
    profile = replace(
        _profile(tmp_path),
        platform=LocalPlatformProfile.POSIX,
    )

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        assert not (
            {Capability.CREATE, Capability.UPDATE, Capability.DELETE}
            & scope.capabilities
        )
        with pytest.raises(TargetInspectionError) as error:
            LocalCommitTarget(profile)
        assert error.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE

    run(execute())


def _metadata(value: bytes, mode: FileMode) -> MetadataProfile:
    """Return the planner metadata matching one exact source value."""
    text = LogicalText.from_bytes(value)
    representation = text.representation.value
    return MetadataProfile(
        mode,
        text.has_bom,
        representation if representation != "none" else "lf",
    )


def _protected_metadata(root: Path, path: str) -> AlgorithmDigest:
    """Capture regular metadata without opening an untrusted test source."""
    try:
        descriptor = open_fd(root / path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW)
    except OSError:
        return AlgorithmDigest.from_bytes(
            b"phase-seven-unopenable-source-v1:" + path.encode()
        )
    try:
        if not S_ISREG(fstat(descriptor).st_mode):
            return AlgorithmDigest.from_bytes(
                b"phase-seven-nonregular-source-v1:" + path.encode()
            )
        return target_module._capture_protected_metadata(descriptor).digest()
    finally:
        close(descriptor)


async def _sealed(
    profile: LocalTargetProfile,
    target: LocalCommitTarget,
    scope: ResolvedMutationScope,
    document: str,
    files: dict[str, bytes],
    modes: dict[str, FileMode] | None = None,
) -> SealedPlan:
    """Seal one test-profile candidate with trusted authorities."""
    canonical = PatchRequestParser(PatchInputLimits()).parse(
        RawPatchIngress(
            RawProviderProfile("phase-seven"),
            RawToolCallId("phase-seven"),
            RawPatchInputKind.APPLY_JSON,
            RawPatchInputState.COMPLETE,
            dumps({"patch": document}, separators=(",", ":")).encode(),
        )
    )
    planner_files = tuple(
        PlannerFile(
            LogicalPath(path),
            SourceBytes(value),
            _metadata(value, (modes or {}).get(path, FileMode(0o644))),
            LogicalPath(path.rsplit("/", 1)[0]) if "/" in path else None,
            profile.identity.mount_id,
            "identity-" + path,
            (
                (profile.root._path / path).lstat().st_dev,
                (profile.root._path / path).lstat().st_ino,
            ),
            (
                (
                    profile.root._path
                    / (path.rsplit("/", 1)[0] if "/" in path else "")
                )
                .stat()
                .st_dev,
                (
                    profile.root._path
                    / (path.rsplit("/", 1)[0] if "/" in path else "")
                )
                .stat()
                .st_ino,
            ),
            _protected_metadata(profile.root._path, path),
        )
        for path, value in sorted(files.items())
    )
    parents = frozenset(
        item.parent for item in planner_files if item.parent is not None
    )
    workspace = PlannerWorkspace(
        planner_files,
        parents,
        (
            PlannerParentMount(
                None,
                profile.identity.mount_id,
                (
                    profile.root._path.stat().st_dev,
                    profile.root._path.stat().st_ino,
                ),
            ),
        )
        + tuple(
            PlannerParentMount(
                path,
                profile.identity.mount_id,
                (
                    (profile.root._path / path.value).stat().st_dev,
                    (profile.root._path / path.value).stat().st_ino,
                ),
            )
            for path in sorted(parents, key=lambda item: item.value)
        ),
    )
    candidate = plan(canonical, workspace)
    paths = tuple(
        sorted(
            {
                path
                for lineage in candidate.lineages
                for path in (lineage.source_path, lineage.destination_path)
                if path is not None
            },
            key=lambda item: item.value,
        )
    )
    limits = _limits()
    reader = PreauthorizationClass("phase-seven-reader")
    rule = PolicyRule(
        PolicyPathSelector(None),
        tuple(
            CapabilityMode(item, ApprovalMode.PREAUTHORIZED, reader)
            for item in Capability
        ),
        atomicity_classes=frozenset(("single_step", "dependency_ordered")),
    )
    requirements = ApprovalRequirements(
        ApprovalMode.PREAUTHORIZED,
        PolicyRouteId("route-seven"),
        PolicyBrokerId("broker-seven"),
        PolicyReviewerRole("reviewer-seven"),
        1,
        reader,
    )
    authorizer = PolicyAuthorizer(
        TrustedPatchPolicy(
            PolicyRevision("policy-six"),
            frozenset((OperationType.APPLY,)),
            (rule,),
            limits,
            requirements,
        )
    )
    effects = frozenset(
        item for lineage in candidate.lineages for item in lineage.capabilities
    )
    if any(
        lineage.final.metadata is not None
        and lineage.final.metadata.mode.value & 0o111
        and Capability.UPDATE in lineage.capabilities
        for lineage in candidate.lineages
    ):
        effects = effects | frozenset((Capability.UPDATE_EXECUTABLE,))
    preflight = await authorizer.authorize_preinspection(
        PreflightRequest(
            OperationType.APPLY,
            paths,
            effects,
            frozenset(paths),
            compose_limits(limits, limits, limits, limits, limits),
        )
    )
    final = await authorizer.authorize_final(
        preflight, candidate, await target.handshake(scope)
    )
    return seal_plan(
        PatchPlanId("plan_" + "c" * 16),
        PlanBinding(
            PatchRequest(
                1,
                PatchRequestId("request_" + "c" * 16),
                PatchExecutionId("execution_" + "c" * 16),
                OperationType.APPLY,
                PatchInput(b"phase-seven"),
                paths,
            ),
            candidate.request_digest,
            ExecutionSubject(
                PatchPrincipalId("principal-seven"),
                PatchTenantId("tenant-seven"),
                PatchRunId("run-seven"),
                PatchSessionId("session-seven"),
                PatchTaskId("task-seven"),
                PatchAgentId("agent-seven"),
            ),
            ContextKind.LOCAL,
            profile.identity,
            None,
            preflight,
            final,
        ),
        candidate,
        ExpiryTick(100),
    )


async def _seatbelt_command(
    profile: LocalTargetProfile,
    scope: ResolvedMutationScope,
    token: str,
) -> SealedCommitCommand:
    """Return one sealed add command for private protocol tests."""
    sealed = await _sealed(
        profile,
        LocalCommitTarget(profile),
        scope,
        "\n".join(
            (
                "*** Begin Patch v1",
                "*** Add File: protocol.txt",
                "+protocol",
                "*** End Patch",
            )
        ),
        {},
    )
    return SealedCommitCommand(
        sealed,
        CommitLease(
            profile.identity.domain_id,
            PatchRequestId("request_" + token * 16),
            1,
        ),
        footprint_for(sealed),
    )


def _seatbelt_payload_value(
    command: SealedCommitCommand,
    profile: LocalTargetProfile,
    scope: ResolvedMutationScope,
) -> dict[str, object]:
    """Return the complete JSON-safe child payload for one sealed command."""
    assert scope.root_witness is not None
    witness = scope.root_witness
    return {
        "command": b64encode(pickle_dumps(command)).decode("ascii"),
        "cwd": profile.cwd.value if profile.cwd is not None else None,
        "fence": command.lease.fence,
        "namespace": str(profile.commit_namespace),
        "plan_id": command.plan.plan_id.value,
        "request_id": command.lease.request_id.value,
        "root": str(profile.root._path),
        "version": 1,
        "witness": {
            "device": witness.identity.device,
            "filesystem_id": witness.filesystem_id,
            "inode": witness.identity.inode,
            "mount_id": witness.mount_id,
        },
    }


def _signed_seatbelt_message(
    token: bytes, payload: dict[str, object]
) -> bytes:
    """Bind one private worker payload to its exact HMAC envelope."""
    raw_payload = dumps(payload, separators=(",", ":")).encode()
    return dumps(
        {
            "payload": payload,
            "mac": digest(token, raw_payload, "sha256").hex(),
        },
        separators=(",", ":"),
    ).encode()


class _SeatbeltStream:
    """Expose an in-memory binary stream through the child stdio shape."""

    def __init__(self, value: bytes = b"") -> None:
        """Initialize the exact bytes available to the worker stream."""
        self.buffer = BytesIO(value)


async def _test_commit(
    worker: RootedLocalCommitWorker, command: SealedCommitCommand
) -> WorkerReport:
    """Drive one local command through the same sealed coordinator path."""
    approvals = ApprovalService(
        _PHASE6["_Broker"](),
        _PHASE6["_Clock"](),
        _PHASE6["RuntimeGrantStore"](),
    )
    grant = await _PHASE6["_issue_grant"](command.plan, approvals)
    store = InMemoryCoordinatorStore(approvals)
    coordinator = InMemoryPatchCoordinator(
        store,
        InMemoryLeaseManager(store),
        ScriptedReconciler(_PHASE6["_snapshot"]()),
    )
    reservation = await coordinator.reserve(
        RuntimeIdentity(
            command.plan.binding.subject,
            command.plan.binding.final.approval.route,
            RetransmissionKey("phase-seven-test-" + str(command.lease.fence)),
        ),
        command.plan.binding.request_digest,
    )
    await coordinator.execute(
        reservation,
        command.plan,
        grant,
        _PHASE6["_snapshot"](),
        worker,
        "phase-seven-test-owner",
    )
    record = await store.record(reservation)
    assert record.journal is not None
    return WorkerReport(WorkerState.SETTLED, record.journal)


def test_patch_phase_7_rooted_update_and_coordinator_e2e(
    tmp_path: Path,
) -> None:
    """Publish complete bytes only after a sealed fence-bearing command."""
    (tmp_path / "note0.txt").write_bytes(b"before\n")
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        worker = await target.worker(scope)
        plan = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: note0.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            ),
            {"note0.txt": b"before\n"},
        )
        approvals = _PHASE6["ApprovalService"](
            _PHASE6["_Broker"](),
            _PHASE6["_Clock"](),
            _PHASE6["RuntimeGrantStore"](),
        )
        grant = await _PHASE6["_issue_grant"](plan, approvals)
        store = InMemoryCoordinatorStore(approvals)
        coordinator = InMemoryPatchCoordinator(
            store,
            InMemoryLeaseManager(store),
            ScriptedReconciler(_PHASE6["_snapshot"]()),
        )
        identity = RuntimeIdentity(
            plan.binding.subject,
            PolicyRouteId("route-six"),
            RetransmissionKey("phase-seven-local"),
        )
        reservation = await coordinator.reserve(
            identity, plan.binding.request_digest
        )
        result = await coordinator.execute(
            reservation,
            plan,
            grant,
            _PHASE6["_snapshot"](),
            worker,
            "local-controller",
        )
        assert isinstance(result, PatchResult)
        assert result.status is PatchStatus.COMMITTED
        assert (tmp_path / "note0.txt").read_bytes() == b"after\n"
        assert not tuple(tmp_path.glob(".avalan-patch-*"))

        later_scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        later = await LocalInspectionTarget(profile).inspect(
            InspectionRequest(later_scope, (LogicalPath("note0.txt"),))
        )
        snapshot = later.snapshots[0]
        assert snapshot.bytes_value is not None
        assert snapshot.bytes_value._value == b"after\n"
        assert snapshot.metadata is not None
        assert snapshot.metadata.mode == FileMode(0o644)

        command = SealedCommitCommand(
            plan,
            CommitLease(
                profile.identity.domain_id,
                PatchRequestId("request_" + "b" * 16),
                2,
            ),
            footprint_for(plan),
        )
        with pytest.raises(CoordinatorError):
            await worker.commit(command)

    run(execute())


def test_patch_phase_7_local_operation_matrix(tmp_path: Path) -> None:
    """Commit create, delete, move, and move-with-update in graph order."""
    sources = {
        "delete.txt": b"delete\n",
        "move.txt": b"move\n",
        "change.txt": b"old\n",
        "update.txt": b"old\n",
    }
    for path, value in sources.items():
        (tmp_path / path).write_bytes(value)
    profile = _profile(tmp_path)

    async def commit(
        worker: RootedLocalCommitWorker,
        scope: ResolvedMutationScope,
        document: str,
        files: dict[str, bytes],
        token: str,
    ) -> None:
        plan = await _sealed(
            profile, LocalCommitTarget(profile), scope, document, files
        )
        report = await _test_commit(
            worker,
            SealedCommitCommand(
                plan,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + token * 16),
                    1,
                ),
                footprint_for(plan),
            ),
        )
        assert report.journal is not None
        assert all(
            item.state.value == "committed" for item in report.journal.steps
        )
        assert report.journal.postcondition.value == "established"

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        worker = await target.worker(scope)
        await commit(
            worker,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: create.txt",
                    "+created",
                    "*** End Patch",
                )
            ),
            {},
            "d",
        )
        await commit(
            worker,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Delete File: delete.txt",
                    "*** End Patch",
                )
            ),
            {"delete.txt": b"delete\n"},
            "e",
        )
        await commit(
            worker,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: move.txt",
                    "*** Move to: moved.txt",
                    "*** End Patch",
                )
            ),
            {"move.txt": b"move\n"},
            "f",
        )
        await commit(
            worker,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: change.txt",
                    "*** Move to: changed.txt",
                    "@@",
                    "-old",
                    "+new",
                    "*** End Patch",
                )
            ),
            {"change.txt": b"old\n"},
            "a",
        )
        await commit(
            worker,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: update.txt",
                    "@@",
                    "-old",
                    "+new",
                    "*** End Patch",
                )
            ),
            {"update.txt": b"old\n"},
            "9",
        )

    run(execute())
    assert (tmp_path / "create.txt").read_bytes() == b"created\n"
    assert S_IMODE((tmp_path / "create.txt").stat().st_mode) == 0o644
    assert not (tmp_path / "delete.txt").exists()
    assert not (tmp_path / "move.txt").exists()
    assert (tmp_path / "moved.txt").read_bytes() == b"move\n"
    assert not (tmp_path / "change.txt").exists()
    assert (tmp_path / "changed.txt").read_bytes() == b"new\n"
    assert (tmp_path / "update.txt").read_bytes() == b"new\n"
    assert not tuple(tmp_path.glob(".avalan-patch-*"))


def test_patch_phase_7_rooted_commit_helpers_cover_operation_matrix(
    tmp_path: Path,
) -> None:
    """Exercise the isolated rooted implementation under test coverage."""
    sources = {
        "delete.txt": b"delete\n",
        "move.txt": b"move\n",
        "change.txt": b"old\n",
        "update.txt": b"old\n",
    }
    for path, value in sources.items():
        (tmp_path / path).write_bytes(value)
    profile = _profile(tmp_path)

    async def commit(
        scope: ResolvedMutationScope,
        document: str,
        files: dict[str, bytes],
        token: str,
    ) -> None:
        target = LocalCommitTarget(profile)
        sealed = await _sealed(profile, target, scope, document, files)
        assert scope.root_witness is not None
        report = local_commit_module._commit_rooted(
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + token * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
            profile,
            scope.root_witness,
        )
        assert report.journal is not None
        assert all(
            item.state is CommitStepState.COMMITTED
            for item in report.journal.steps
        )
        assert report.journal.postcondition is PostconditionState.ESTABLISHED

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        await commit(
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: create.txt",
                    "+created",
                    "*** End Patch",
                )
            ),
            {},
            "1",
        )
        await commit(
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Delete File: delete.txt",
                    "*** End Patch",
                )
            ),
            {"delete.txt": b"delete\n"},
            "2",
        )
        await commit(
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: move.txt",
                    "*** Move to: moved.txt",
                    "*** End Patch",
                )
            ),
            {"move.txt": b"move\n"},
            "3",
        )
        await commit(
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: change.txt",
                    "*** Move to: changed.txt",
                    "@@",
                    "-old",
                    "+new",
                    "*** End Patch",
                )
            ),
            {"change.txt": b"old\n"},
            "4",
        )
        await commit(
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: update.txt",
                    "@@",
                    "-old",
                    "+new",
                    "*** End Patch",
                )
            ),
            {"update.txt": b"old\n"},
            "5",
        )

    run(execute())
    assert (tmp_path / "create.txt").read_bytes() == b"created\n"
    assert not (tmp_path / "delete.txt").exists()
    assert not (tmp_path / "move.txt").exists()
    assert (tmp_path / "moved.txt").read_bytes() == b"move\n"
    assert not (tmp_path / "change.txt").exists()
    assert (tmp_path / "changed.txt").read_bytes() == b"new\n"
    assert (tmp_path / "update.txt").read_bytes() == b"new\n"


def test_patch_phase_7_preserves_native_protected_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-trip xattrs, flags, and ACL witness or fail before publication."""
    source = tmp_path / "metadata.txt"
    source.write_bytes(b"before\n")
    descriptor = open_fd(source, O_RDONLY)
    try:
        assert (
            target_module._METADATA_LIBC.fsetxattr(
                descriptor,
                b"user.avalan.phase7",
                target_module._METADATA_FFI.new("char[]", b"retained"),
                len(b"retained"),
                0,
                0,
            )
            == 0
        )
        before = target_module._capture_protected_metadata(descriptor)
    finally:
        close(descriptor)
    assert before.flags == source.stat().st_flags
    assert before.acl is None
    assert (b"user.avalan.phase7", b"retained") in before.xattrs
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: metadata.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            ),
            {"metadata.txt": b"before\n"},
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "7" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert report.journal.postcondition is PostconditionState.ESTABLISHED
        descriptor = open_fd(source, O_RDONLY)
        try:
            assert (
                target_module._capture_protected_metadata(descriptor) == before
            )
        finally:
            close(descriptor)

        source.write_bytes(b"before\n")
        descriptor = open_fd(source, O_RDONLY)
        try:
            assert (
                target_module._METADATA_LIBC.fsetxattr(
                    descriptor,
                    b"user.avalan.phase7",
                    target_module._METADATA_FFI.new("char[]", b"retained"),
                    len(b"retained"),
                    0,
                    0,
                )
                == 0
            )
        finally:
            close(descriptor)
        rejected = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: metadata.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            ),
            {"metadata.txt": b"before\n"},
        )
        with monkeypatch.context() as patcher:
            original_barrier = local_commit_module._commit_barrier

            def unavailable_metadata(stage: str) -> None:
                """Fail the authenticated stage after metadata preparation."""
                original_barrier(stage)
                if stage == "target.stage_artifact":
                    raise OSError("native metadata unavailable")

            patcher.setattr(
                local_commit_module,
                "_commit_barrier",
                unavailable_metadata,
            )
            failed = await _test_commit(
                await target.worker(scope),
                SealedCommitCommand(
                    rejected,
                    CommitLease(
                        profile.identity.domain_id,
                        PatchRequestId("request_" + "8" * 16),
                        1,
                    ),
                    footprint_for(rejected),
                ),
            )
        assert failed.journal is not None
        assert failed.journal.steps[0].state is CommitStepState.UNKNOWN

    run(execute())
    assert source.read_bytes() == b"before\n"


def test_patch_phase_7_rejects_stale_source_and_destination_races(
    tmp_path: Path,
) -> None:
    """Keep foreign writes intact when final rooted barriers changed."""
    (tmp_path / "update.txt").write_bytes(b"before\n")
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        worker = await target.worker(scope)
        create = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: created.txt",
                    "+planned",
                    "*** End Patch",
                )
            ),
            {},
        )
        update = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: update.txt",
                    "@@",
                    "-before",
                    "+planned",
                    "*** End Patch",
                )
            ),
            {"update.txt": b"before\n"},
        )
        (tmp_path / "created.txt").write_bytes(b"foreign-create\n")
        (tmp_path / "update.txt").write_bytes(b"foreign-update\n")
        for token, sealed in (("8", create), ("7", update)):
            report = await _test_commit(
                worker,
                SealedCommitCommand(
                    sealed,
                    CommitLease(
                        profile.identity.domain_id,
                        PatchRequestId("request_" + token * 16),
                        1,
                    ),
                    footprint_for(sealed),
                ),
            )
            assert report.journal is not None
            assert report.journal.steps[0].state.value == "not_committed"

    run(execute())
    assert (tmp_path / "created.txt").read_bytes() == b"foreign-create\n"
    assert (tmp_path / "update.txt").read_bytes() == b"foreign-update\n"
    assert not tuple(tmp_path.glob(".avalan-patch-*"))


def test_patch_phase_7_commits_one_deterministic_multilineage_graph(
    tmp_path: Path,
) -> None:
    """Settle every sealed lineage step without transaction atomicity."""
    sources = {
        "delete.txt": b"delete\n",
        "move.txt": b"move\n",
        "update.txt": b"before\n",
    }
    for path, value in sources.items():
        (tmp_path / path).write_bytes(value)
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: create.txt",
                    "+created",
                    "*** Delete File: delete.txt",
                    "*** Update File: move.txt",
                    "*** Move to: moved.txt",
                    "*** Update File: update.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            ),
            sources,
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "6" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        expected_lineages = tuple(
            lineage.lineage_id
            for lineage in sealed.candidate.lineages
            for _ in lineage.step_graph
        )
        assert (
            tuple(step.lineage for step in report.journal.steps)
            == expected_lineages
        )
        assert all(
            step.state.value == "committed" for step in report.journal.steps
        )
        assert report.journal.postcondition is PostconditionState.ESTABLISHED

    run(execute())
    assert (tmp_path / "create.txt").read_bytes() == b"created\n"
    assert not (tmp_path / "delete.txt").exists()
    assert not (tmp_path / "move.txt").exists()
    assert (tmp_path / "moved.txt").read_bytes() == b"move\n"
    assert (tmp_path / "update.txt").read_bytes() == b"after\n"


def test_patch_phase_7_preserves_approved_executables_and_rejects_links(
    tmp_path: Path,
) -> None:
    """Allow sealed executable replacement but reject linked files."""
    (tmp_path / "program.txt").write_bytes(b"before\n")
    (tmp_path / "linked.txt").write_bytes(b"before\n")
    (tmp_path / "privileged.txt").write_bytes(b"before\n")
    chmod(tmp_path / "program.txt", 0o755)
    chmod(tmp_path / "privileged.txt", 0o4755)
    link(tmp_path / "linked.txt", tmp_path / "linked-alias.txt")
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        worker = await target.worker(scope)
        plans = (
            (
                "6",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: program.txt",
                            "@@",
                            "-before",
                            "+after",
                            "*** End Patch",
                        )
                    ),
                    {"program.txt": b"before\n"},
                    {"program.txt": FileMode(0o755)},
                ),
                "committed",
            ),
            (
                "5",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: linked.txt",
                            "@@",
                            "-before",
                            "+after",
                            "*** End Patch",
                        )
                    ),
                    {"linked.txt": b"before\n"},
                ),
                "not_committed",
            ),
            (
                "4",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: privileged.txt",
                            "@@",
                            "-before",
                            "+after",
                            "*** End Patch",
                        )
                    ),
                    {"privileged.txt": b"before\n"},
                    {"privileged.txt": FileMode(0o755)},
                ),
                "not_committed",
            ),
        )
        for token, sealed, outcome in plans:
            report = await _test_commit(
                worker,
                SealedCommitCommand(
                    sealed,
                    CommitLease(
                        profile.identity.domain_id,
                        PatchRequestId("request_" + token * 16),
                        1,
                    ),
                    footprint_for(sealed),
                ),
            )
            assert report.journal is not None
            assert report.journal.steps[0].state.value == outcome

    run(execute())
    assert (tmp_path / "program.txt").read_bytes() == b"after\n"
    assert S_IMODE((tmp_path / "program.txt").stat().st_mode) == 0o755
    assert (tmp_path / "linked.txt").read_bytes() == b"before\n"
    assert (tmp_path / "privileged.txt").read_bytes() == b"before\n"


def test_patch_phase_7_preserves_exact_text_representation(
    tmp_path: Path,
) -> None:
    """Commit BOM/CRLF and no-final-newline replacements byte-for-byte."""
    (tmp_path / "bom.txt").write_bytes(b"\xef\xbb\xbfbefore\r\n")
    chmod(tmp_path / "bom.txt", 0o640)
    (tmp_path / "lf.txt").write_bytes(b"one\ntwo\nthree\n")
    (tmp_path / "none.txt").write_bytes(b"before")
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        worker = await target.worker(scope)
        plans = (
            (
                "2",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: bom.txt",
                            "@@",
                            "-before",
                            "+after",
                            "*** End Patch",
                        )
                    ),
                    {"bom.txt": b"\xef\xbb\xbfbefore\r\n"},
                    {"bom.txt": FileMode(0o640)},
                ),
            ),
            (
                "3",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: none.txt",
                            "@@",
                            "-before",
                            "\\ No newline at end of file",
                            "+after",
                            "\\ No newline at end of file",
                            "*** End of File",
                            "*** End Patch",
                        )
                    ),
                    {"none.txt": b"before"},
                ),
            ),
            (
                "4",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: lf.txt",
                            "@@",
                            "-one",
                            "+ONE",
                            "@@",
                            "-three",
                            "+THREE",
                            "*** End Patch",
                        )
                    ),
                    {"lf.txt": b"one\ntwo\nthree\n"},
                ),
            ),
        )
        for token, sealed in plans:
            report = await _test_commit(
                worker,
                SealedCommitCommand(
                    sealed,
                    CommitLease(
                        profile.identity.domain_id,
                        PatchRequestId("request_" + token * 16),
                        1,
                    ),
                    footprint_for(sealed),
                ),
            )
            assert report.journal is not None
            assert (
                report.journal.postcondition is PostconditionState.ESTABLISHED
            )

    run(execute())
    assert (tmp_path / "bom.txt").read_bytes() == b"\xef\xbb\xbfafter\r\n"
    assert S_IMODE((tmp_path / "bom.txt").stat().st_mode) == 0o640
    assert (tmp_path / "lf.txt").read_bytes() == b"ONE\ntwo\nTHREE\n"
    assert (tmp_path / "none.txt").read_bytes() == b"after"


def test_patch_phase_7_reader_never_observes_partial_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose only complete old or new bytes while staging a slow update."""
    path = tmp_path / "note.txt"
    before = b"before\n"
    after = b"after-" + b"x" * 512 + b"\n"
    path.write_bytes(before)
    profile = _profile(tmp_path)
    reader_started = Event()
    reader_stop = Event()
    observed: set[bytes] = set()

    def read_until_stopped() -> None:
        reader_started.set()
        while not reader_stop.is_set():
            observed.add(path.read_bytes())
        for _ in range(16):
            observed.add(path.read_bytes())

    thread = Thread(target=read_until_stopped)
    thread.start()
    assert reader_started.wait(1)
    original_write = cast(
        Callable[[int, bytes], int], getattr(local_commit_module, "write_fd")
    )

    def slow_stage_write(descriptor: int, value: bytes) -> int:
        count = original_write(descriptor, value[:1])
        sleep(0.0001)
        return int(count)

    monkeypatch.setattr(local_commit_module, "write_fd", slow_stage_write)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: note.txt",
                    "@@",
                    "-before",
                    "+" + after.decode().removesuffix("\n"),
                    "*** End Patch",
                )
            ),
            {"note.txt": before},
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "1" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert report.journal.postcondition is PostconditionState.ESTABLISHED

    try:
        run(execute())
    finally:
        reader_stop.set()
        thread.join()
    assert observed <= {before, after}
    assert before in observed
    assert after in observed


def test_patch_phase_7_reader_never_observes_partial_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose either absence or complete bytes while staging a slow create."""
    path = tmp_path / "created.txt"
    created = b"created-" + b"x" * 512 + b"\n"
    profile = _profile(tmp_path)
    reader_started = Event()
    reader_stop = Event()
    observed: set[bytes | None] = set()

    def read_until_stopped() -> None:
        reader_started.set()
        while not reader_stop.is_set():
            try:
                observed.add(path.read_bytes())
            except FileNotFoundError:
                observed.add(None)
        for _ in range(16):
            observed.add(path.read_bytes())

    thread = Thread(target=read_until_stopped)
    thread.start()
    assert reader_started.wait(1)
    original_write = cast(
        Callable[[int, bytes], int], getattr(local_commit_module, "write_fd")
    )

    def slow_stage_write(descriptor: int, value: bytes) -> int:
        count = original_write(descriptor, value[:1])
        sleep(0.0001)
        return int(count)

    monkeypatch.setattr(local_commit_module, "write_fd", slow_stage_write)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: created.txt",
                    "+" + created.decode().removesuffix("\n"),
                    "*** End Patch",
                )
            ),
            {},
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "e" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert report.journal.postcondition is PostconditionState.ESTABLISHED

    try:
        run(execute())
    finally:
        reader_stop.set()
        thread.join()
    assert observed <= {None, created}
    assert None in observed
    assert created in observed


def test_patch_phase_7_records_superseded_after_proven_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain occurrence when a foreign writer wins after publication."""
    path = tmp_path / "note.txt"
    path.write_bytes(b"before\n")
    profile = _profile(tmp_path)

    original_barrier = local_commit_module._commit_barrier

    def foreign_write_before_verify(stage: str) -> None:
        """Publish a real foreign replacement after the child commit."""
        original_barrier(stage)
        if stage == "verification.before":
            path.write_bytes(b"foreign\n")

    monkeypatch.setattr(
        local_commit_module, "_commit_barrier", foreign_write_before_verify
    )

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
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
            {"note.txt": b"before\n"},
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "0" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert report.journal.steps[0].state.value == "committed"
        assert report.journal.postcondition is PostconditionState.SUPERSEDED

    run(execute())
    assert path.read_bytes() == b"foreign\n"


def test_patch_phase_7_preserves_known_step_after_verification_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a committed effect distinct from unavailable verification."""
    path = tmp_path / "note.txt"
    path.write_bytes(b"before\n")
    profile = _profile(tmp_path)

    original_barrier = local_commit_module._commit_barrier

    def unavailable_verify(stage: str) -> None:
        """Fail child at its fixed post-publication verification barrier."""
        original_barrier(stage)
        if stage == "verification.before":
            raise OSError("verification unavailable")

    monkeypatch.setattr(
        local_commit_module, "_commit_barrier", unavailable_verify
    )

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
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
            {"note.txt": b"before\n"},
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "b" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert report.journal.steps[0].state.value == "committed"
        assert report.journal.postcondition is PostconditionState.UNKNOWN

    run(execute())
    assert path.read_bytes() == b"after\n"


def test_patch_phase_7_stops_after_move_update_source_remove_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report a known publish and unknown source removal without a retry."""
    (tmp_path / "source.txt").write_bytes(b"before\n")
    profile = _profile(tmp_path)

    original_barrier = local_commit_module._commit_barrier
    observed_stages: list[str] = []

    def fail_source_remove(stage: str) -> None:
        """Fail only the child source-removal syscall boundary."""
        original_barrier(stage)
        observed_stages.append(stage)
        if stage == "move.source_remove_before":
            raise OSError("source remove failed")

    monkeypatch.setattr(
        local_commit_module, "_commit_barrier", fail_source_remove
    )

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: source.txt",
                    "*** Move to: destination.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            ),
            {"source.txt": b"before\n"},
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "a" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert "move.source_remove_before" in observed_stages
        assert tuple(step.state.value for step in report.journal.steps) == (
            "committed",
            "unknown",
        )
        assert report.journal.postcondition is PostconditionState.UNKNOWN

    run(execute())
    assert (tmp_path / "source.txt").read_bytes() == b"before\n"
    assert (tmp_path / "destination.txt").read_bytes() == b"after\n"


def test_patch_phase_7_settles_each_move_update_failure_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep each move-update intermediate state explicit and non-retrying."""

    async def commit_case(case: str) -> tuple[str, str, tuple[str, ...]]:
        root = tmp_path / case
        root.mkdir()
        (root / "source.txt").write_bytes(b"before\n")
        profile = _profile(root)
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: source.txt",
                    "*** Move to: destination.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            ),
            {"source.txt": b"before\n"},
        )
        token = str(("stage", "publish", "source_remove").index(case) + 1)
        with monkeypatch.context() as patcher:
            original_barrier = local_commit_module._commit_barrier

            def fail_boundary(stage: str) -> None:
                """Inject one child failure at its selected fixed boundary."""
                original_barrier(stage)
                expected_stage = {
                    "stage": "target.stage_artifact",
                    "publish": "publication.before_link",
                    "source_remove": "move.source_remove_before",
                }[case]
                if stage == expected_stage:
                    raise OSError(case + " failed")

            if case == "stage":
                patcher.setattr(
                    local_commit_module, "_commit_barrier", fail_boundary
                )
            elif case == "publish":
                patcher.setattr(
                    local_commit_module, "_commit_barrier", fail_boundary
                )
            elif case == "source_remove":
                patcher.setattr(
                    local_commit_module, "_commit_barrier", fail_boundary
                )
            else:
                raise AssertionError(case)
            report = await _test_commit(
                await target.worker(scope),
                SealedCommitCommand(
                    sealed,
                    CommitLease(
                        profile.identity.domain_id,
                        PatchRequestId("request_" + token * 16),
                        1,
                    ),
                    footprint_for(sealed),
                ),
            )
        assert report.journal is not None
        destination = root / "destination.txt"
        return (
            (root / "source.txt").read_text(),
            destination.read_text() if destination.exists() else "",
            tuple(step.state.value for step in report.journal.steps),
        )

    async def execute() -> tuple[
        tuple[str, str, tuple[str, ...]],
        tuple[str, str, tuple[str, ...]],
        tuple[str, str, tuple[str, ...]],
    ]:
        return (
            await commit_case("stage"),
            await commit_case("publish"),
            await commit_case("source_remove"),
        )

    stage, publish, source_remove = run(execute())
    assert stage == ("before\n", "", ("unknown", "not_committed"))
    assert publish == ("before\n", "", ("unknown", "not_committed"))
    assert source_remove == (
        "before\n",
        "after\n",
        ("committed", "unknown"),
    )


def test_patch_phase_7_never_clobbers_raced_create_or_move_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject entries created at the atomic publication barrier."""
    (tmp_path / "source.txt").write_bytes(b"source\n")
    profile = _profile(tmp_path)
    raced: set[str] = set()
    original_barrier = local_commit_module._commit_barrier

    def race_destination(stage: str) -> None:
        """Create the selected foreign entry before the child link syscall."""
        original_barrier(stage)
        if stage != "publication.before_link":
            return
        destination = "created.txt" if not raced else "moved.txt"
        raced.add(destination)
        (tmp_path / destination).write_bytes(b"foreign\n")

    monkeypatch.setattr(
        local_commit_module, "_commit_barrier", race_destination
    )

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        worker = await target.worker(scope)
        plans = (
            (
                "b",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Add File: created.txt",
                            "+planned",
                            "*** End Patch",
                        )
                    ),
                    {},
                ),
            ),
            (
                "c",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: source.txt",
                            "*** Move to: moved.txt",
                            "*** End Patch",
                        )
                    ),
                    {"source.txt": b"source\n"},
                ),
            ),
        )
        for token, sealed in plans:
            report = await _test_commit(
                worker,
                SealedCommitCommand(
                    sealed,
                    CommitLease(
                        profile.identity.domain_id,
                        PatchRequestId("request_" + token * 16),
                        1,
                    ),
                    footprint_for(sealed),
                ),
            )
            assert report.journal is not None
            assert all(
                step.state.value != "committed"
                for step in report.journal.steps
            )

    run(execute())
    assert raced == {"created.txt", "moved.txt"}
    assert (tmp_path / "created.txt").read_bytes() == b"foreign\n"
    assert (tmp_path / "moved.txt").read_bytes() == b"foreign\n"
    assert (tmp_path / "source.txt").read_bytes() == b"source\n"


def test_patch_phase_7_keeps_leaked_staging_separate_from_effect_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report a leaked private artifact without rolling back its publish."""
    profile = _profile(tmp_path)
    original_barrier = local_commit_module._commit_barrier

    def lose_stage_cleanup(stage: str) -> None:
        """Lose cleanup only after the child publishes its requested effect."""
        original_barrier(stage)
        if stage == "artifact.cleanup_before":
            raise OSError("staging cleanup lost")

    monkeypatch.setattr(
        local_commit_module, "_commit_barrier", lose_stage_cleanup
    )

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: created.txt",
                    "+created",
                    "*** End Patch",
                )
            ),
            {},
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "f" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert report.journal.steps[0].state.value == "committed"
        assert report.journal.artifacts[0].state.value == "leaked"
        assert report.journal.postcondition is PostconditionState.SUPERSEDED

    run(execute())
    assert (tmp_path / "created.txt").read_bytes() == b"created\n"
    assert len(tuple(tmp_path.glob(".avalan-patch-*"))) == 1


def test_patch_phase_7_rejects_symlink_and_special_source_before_effect(
    tmp_path: Path,
) -> None:
    """Reject untrusted link and special-file source entries before opening."""
    outside = tmp_path.parent / "outside.txt"
    outside.write_bytes(b"outside\n")
    symlink(outside, tmp_path / "linked.txt")
    mkfifo(tmp_path / "special.txt")
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        worker = await target.worker(scope)
        plans = (
            (
                "8",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: linked.txt",
                            "@@",
                            "-outside",
                            "+after",
                            "*** End Patch",
                        )
                    ),
                    {"linked.txt": b"outside\n"},
                ),
            ),
            (
                "9",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: special.txt",
                            "@@",
                            "-before",
                            "+after",
                            "*** End Patch",
                        )
                    ),
                    {"special.txt": b"before\n"},
                ),
            ),
        )
        for token, sealed in plans:
            report = await _test_commit(
                worker,
                SealedCommitCommand(
                    sealed,
                    CommitLease(
                        profile.identity.domain_id,
                        PatchRequestId("request_" + token * 16),
                        1,
                    ),
                    footprint_for(sealed),
                ),
            )
            assert report.journal is not None
            assert report.journal.steps[0].state.value == "not_committed"

    run(execute())
    assert outside.read_bytes() == b"outside\n"
    assert (tmp_path / "linked.txt").is_symlink()


def test_patch_phase_7_rejects_ancestor_link_before_native_commit(
    tmp_path: Path,
) -> None:
    """Keep outside-root canaries intact when a parent becomes a link."""
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "note.txt").write_bytes(b"before\n")
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    canary = outside / "note.txt"
    canary.write_bytes(b"outside\n")
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: nested/note.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            ),
            {"nested/note.txt": b"before\n"},
        )
        parked = tmp_path / "parked"
        nested.rename(parked)
        symlink(outside, nested)
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "d" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert report.journal.steps[0].state.value == "not_committed"

    run(execute())
    assert canary.read_bytes() == b"outside\n"


def test_patch_phase_7_rejects_remounted_root_before_native_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a changed filesystem witness before attempting any update."""
    path = tmp_path / "note.txt"
    path.write_bytes(b"before\n")
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
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
            {"note.txt": b"before\n"},
        )
        monkeypatch.setattr(
            local_commit_module, "_filesystem_id", lambda descriptor: "other"
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "c" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert report.journal.steps[0].state.value == "not_committed"

    run(execute())
    assert path.read_bytes() == b"before\n"


def test_patch_phase_7_public_shell_surface_cannot_reach_rooted_worker() -> (
    None
):
    """Keep local commit authority out of the public patch package surface."""
    probe = run_process(
        (
            executable,
            "-c",
            (
                "import avalan.patch as patch; raise SystemExit(hasattr(patch,"
                " 'RootedLocalCommitWorker') or hasattr(patch,"
                " 'LocalCommitTarget'))"
            ),
        ),
        check=False,
    )
    assert probe.returncode == 0


def test_patch_phase_7_e2e_003_precommit_failures_leave_tree_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute every active precommit cell without writing its target tree."""
    path = tmp_path / "note.txt"
    path.write_bytes(b"before\n")
    profile = _profile(tmp_path)
    inventory = _e2e_inventory()
    precommit = inventory["precommit"]
    assert isinstance(precommit, dict)
    expected_ids = tuple(precommit["failure_ids"])
    assert all(isinstance(item, str) for item in expected_ids)
    raw_scenarios = precommit["scenarios"]
    assert isinstance(raw_scenarios, list)
    scenarios = tuple(
        (str(item["id"]), str(item["boundary"]), str(item["executor"]))
        for item in raw_scenarios
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and isinstance(item.get("boundary"), str)
        and isinstance(item.get("executor"), str)
    )
    assert len(scenarios) == len(raw_scenarios)
    assert tuple(item[0] for item in scenarios) == expected_ids
    assert len({item[2] for item in scenarios}) == len(scenarios)
    matrix = loads(
        Path("tests/fixtures/patch/failure_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(matrix, dict)
    cells = matrix["cells"]
    assert isinstance(cells, list)
    active = tuple(
        str(cell["id"])
        for cell in cells
        if isinstance(cell, dict)
        and cell["lifecycle"] == "active"
        and cell["active_from_phase"] <= 7
        and cell["commit_started"] is False
        and cell["expected_workspace_write_count"] == 0
    )
    assert expected_ids == active
    assert len(expected_ids) == len(set(expected_ids))
    boundary_by_id = {
        str(cell["id"]): str(cell["boundary"])
        for cell in cells
        if isinstance(cell, dict)
        and cell.get("id") in expected_ids
        and isinstance(cell.get("boundary"), str)
    }
    observed_boundaries = tuple(
        boundary_by_id[identifier] for identifier in expected_ids
    )
    assert observed_boundaries == tuple(item[1] for item in scenarios)
    before = _tree_snapshot(tmp_path)

    async def runtime(
        faults: _PrecommitFaults,
        *,
        grants: RuntimeGrantStore | None = None,
    ) -> tuple[
        InMemoryPatchCoordinator,
        InMemoryCoordinatorStore,
        InMemoryLeaseManager,
        RuntimeIdentity,
        SealedPlan,
        PlanBoundGrant,
    ]:
        """Build canonical local components without a test-function import."""
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
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
            {"note.txt": b"before\n"},
        )
        grant_store = grants or RuntimeGrantStore()
        approval = ApprovalService(
            _E2EApprovalBroker(), _E2EClock(), grant_store
        )
        approval_result = await approval.await_review(
            PlanReviewRequest(
                sealed, sealed.binding.subject, sealed.binding.final.approval
            )
        )
        assert approval_result.grant is not None
        store = InMemoryCoordinatorStore(
            approval, _precommit_checkpoint=faults
        )
        leases = InMemoryLeaseManager(store)
        coordinator = InMemoryPatchCoordinator(
            store, leases, ScriptedReconciler(_phase_seven_snapshot())
        )
        return (
            coordinator,
            store,
            leases,
            RuntimeIdentity(
                sealed.binding.subject,
                sealed.binding.final.approval.route,
                RetransmissionKey("phase-seven-e2e"),
            ),
            sealed,
            approval_result.grant,
        )

    async def native(
        faults: _PrecommitFaults,
    ) -> tuple[int, int]:
        """Reach fixed native worker barriers before a write is possible."""
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
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
            {"note.txt": b"before\n"},
        )
        barriers = {
            "target.open_handle": _E2EPrecommitBoundary.TARGET_OPEN,
            "target.close_handle": _E2EPrecommitBoundary.TARGET_CLOSE,
            "target.stage_artifact": _E2EPrecommitBoundary.TARGET_STAGE,
            "requested_effect.step_before": (
                _E2EPrecommitBoundary.REQUESTED_EFFECT
            ),
            "artifact.stage": _E2EPrecommitBoundary.ARTIFACT_STAGE,
        }
        original = local_commit_module._commit_barrier

        def observe(stage: str) -> None:
            """Emit only a native fixed stage reached by the real worker."""
            original(stage)
            boundary = barriers.get(stage)
            if boundary is not None:
                faults.reached_now(boundary)

        with monkeypatch.context() as patcher:
            patcher.setattr(local_commit_module, "_commit_barrier", observe)
            await _test_commit(
                await target.worker(scope),
                SealedCommitCommand(
                    sealed,
                    CommitLease(
                        profile.identity.domain_id,
                        PatchRequestId("request_" + "e" * 16),
                        1,
                    ),
                    footprint_for(sealed),
                ),
            )
        return 0, 0

    async def execute_scenario(
        executor: str,
    ) -> tuple[int, int, tuple[str, ...]]:
        """Drive one canonical operation until its fixed boundary fails."""
        faults = _PrecommitFaults(_EXECUTOR_BOUNDARIES[executor])
        active_faults[:] = [faults]
        inspections = 0
        commits = 0
        try:
            match executor:
                case "lifecycle_received":
                    parser = PatchRequestParser(PatchInputLimits())
                    active_faults.clear()
                    for payload in (
                        b'{"path":"../escape","edits":[]}',
                        b'{"patch":"*** Begin Patch v1\\n*** End Patch"}',
                    ):
                        with pytest.raises(BaseException):
                            parser.parse(
                                RawPatchIngress(
                                    RawProviderProfile("phase-seven"),
                                    RawToolCallId("phase-seven"),
                                    RawPatchInputKind.EDIT_JSON,
                                    RawPatchInputState.COMPLETE,
                                    payload,
                                )
                            )
                    active_faults[:] = [faults]
                    parser.parse(
                        RawPatchIngress(
                            RawProviderProfile("phase-seven"),
                            RawToolCallId("phase-seven"),
                            RawPatchInputKind.APPLY_JSON,
                            RawPatchInputState.COMPLETE,
                            b'{"patch":"*** Begin Patch v1\\n'
                            b"*** Delete File: note.txt\\n"
                            b'*** End Patch\\n"}',
                        )
                    )
                case "lifecycle_scope_bound":
                    await LocalScopeResolver(profile).resolve(
                        ScopeSelection(ContextKind.LOCAL)
                    )
                case "lifecycle_preinspection_authorized":
                    await runtime(faults)
                case "lifecycle_planned":
                    coordinator, _, _, identity, sealed, _ = await runtime(
                        faults
                    )
                    reservation = await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                    await coordinator.prepare(
                        reservation, sealed, approval_required=False
                    )
                case "lifecycle_awaiting_approval":
                    coordinator, _, _, identity, sealed, _ = await runtime(
                        faults
                    )
                    reservation = await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                    await coordinator.prepare(
                        reservation, sealed, approval_required=True
                    )
                case "lifecycle_commit_owner_assigned":
                    coordinator, _, _, identity, sealed, grant = await runtime(
                        faults
                    )
                    reservation = await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                    await coordinator.prepare(
                        reservation, sealed, approval_required=False
                    )
                    await coordinator.execute(
                        reservation,
                        sealed,
                        grant,
                        _phase_seven_snapshot(),
                        ScriptedCommitWorker(
                            WorkerReport(WorkerState.LIVE, None)
                        ),
                        "phase-seven",
                    )
                case "lifecycle_request_completed":
                    coordinator, _, _, identity, sealed, _ = await runtime(
                        faults
                    )
                    reservation = await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                    await coordinator.prepare(
                        reservation, sealed, approval_required=False
                    )
                    await coordinator.cancel(reservation, before_commit=True)
                case "target_negotiate_capabilities":
                    scope = await LocalScopeResolver(profile).resolve(
                        ScopeSelection(ContextKind.LOCAL)
                    )
                    await LocalCommitTarget(profile).handshake(scope)
                case "target_inspect" | "target_observe_precondition":
                    scope = await LocalScopeResolver(profile).resolve(
                        ScopeSelection(ContextKind.LOCAL)
                    )
                    inspections = 1
                    snapshot = await LocalInspectionTarget(profile).inspect(
                        InspectionRequest(scope, (LogicalPath("note.txt"),))
                    )
                    assert snapshot.snapshots[0].bytes_value == SourceBytes(
                        b"before\n"
                    )
                    inspections = 1
                case (
                    "target_open_handle"
                    | "target_close_handle"
                    | "target_stage_artifact"
                    | "requested_effect_step_before"
                    | "artifact_stage"
                ):
                    inspections, commits = await native(faults)
                case "target_acquire_lock" | "target_release_lock":
                    (
                        coordinator,
                        _,
                        leases,
                        identity,
                        sealed,
                        _,
                    ) = await runtime(faults)
                    reservation = await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                    lease = await leases.acquire(
                        footprint_for(sealed), reservation
                    )
                    await leases.release(lease)
                    del coordinator
                case "store_reserve_request":
                    coordinator, _, _, identity, sealed, _ = await runtime(
                        faults
                    )
                    await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                case "store_persist_plan":
                    coordinator, _, _, identity, sealed, _ = await runtime(
                        faults
                    )
                    reservation = await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                    await coordinator.prepare(
                        reservation, sealed, approval_required=False
                    )
                case "store_consume_grant" | "store_assign_commit_owner":
                    (
                        coordinator,
                        store,
                        leases,
                        identity,
                        sealed,
                        grant,
                    ) = await runtime(faults)
                    reservation = await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                    await coordinator.prepare(
                        reservation, sealed, approval_required=False
                    )
                    lease = await leases.acquire(
                        footprint_for(sealed), reservation
                    )
                    await store.begin_commit(reservation, sealed, grant, lease)
                case "approval_decide":
                    scope = await LocalScopeResolver(profile).resolve(
                        ScopeSelection(ContextKind.LOCAL)
                    )
                    sealed = await _sealed(
                        profile,
                        LocalCommitTarget(profile),
                        scope,
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
                        {"note.txt": b"before\n"},
                    )
                    await ApprovalService(
                        _E2EApprovalBroker(faults),
                        _E2EClock(),
                        RuntimeGrantStore(),
                    ).await_review(
                        PlanReviewRequest(
                            sealed,
                            sealed.binding.subject,
                            sealed.binding.final.approval,
                        )
                    )
                case "approval_consume":
                    grants = _E2EGrantStore(faults)
                    _, _, _, _, sealed, grant = await runtime(
                        faults, grants=grants
                    )
                    await ApprovalService(
                        _E2EApprovalBroker(), _E2EClock(), grants
                    ).validate_grant(grant, sealed, sealed.binding.subject)
                case "approval_concurrent_consume":
                    concurrent_grants = _ConcurrentE2EGrantStore(faults)
                    _, _, _, _, sealed, grant = await runtime(
                        faults, grants=concurrent_grants
                    )
                    service = ApprovalService(
                        _E2EApprovalBroker(), _E2EClock(), concurrent_grants
                    )
                    await gather(
                        service.validate_grant(
                            grant, sealed, sealed.binding.subject
                        ),
                        service.validate_grant(
                            grant, sealed, sealed.binding.subject
                        ),
                    )
                case "commit_intent_fence":
                    (
                        coordinator,
                        _,
                        leases,
                        identity,
                        sealed,
                        _,
                    ) = await runtime(faults)
                    reservation = await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                    await leases.acquire(footprint_for(sealed), reservation)
                case "cancellation_before_commit":
                    coordinator, _, _, identity, sealed, _ = await runtime(
                        faults
                    )
                    reservation = await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                    await coordinator.prepare(
                        reservation, sealed, approval_required=False
                    )
                    await coordinator.cancel(reservation, before_commit=True)
                case "timeout_before_commit":
                    coordinator, _, _, identity, sealed, _ = await runtime(
                        faults
                    )
                    reservation = await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                    await coordinator.prepare(
                        reservation, sealed, approval_required=False
                    )
                    await coordinator._timeout_before_commit(reservation)
                case "disconnect_before_commit":
                    coordinator, _, _, identity, sealed, _ = await runtime(
                        faults
                    )
                    reservation = await coordinator.reserve(
                        identity, sealed.binding.request_digest
                    )
                    await coordinator.prepare(
                        reservation, sealed, approval_required=False
                    )
                    await coordinator._disconnect_before_commit(reservation)
                case _:
                    raise AssertionError(executor)
        except _PrecommitFault:
            return (
                inspections,
                commits,
                tuple(item.value for item in faults.observed),
            )
        raise AssertionError("precommit fault did not fire")

    active_faults: list[_PrecommitFaults] = []

    def sync_component(stage: str) -> None:
        """Emit parser-local boundaries from the real parser method."""
        if active_faults and stage == "lifecycle.received":
            active_faults[0].reached_now(
                _E2EPrecommitBoundary.LIFECYCLE_RECEIVED
            )

    async def async_component(stage: str) -> None:
        """Emit policy and target boundaries from their real async methods."""
        if not active_faults:
            return
        boundary = {
            "lifecycle.preinspection_authorized": (
                _E2EPrecommitBoundary.LIFECYCLE_PREFLIGHT
            ),
            "lifecycle.scope_bound": (
                _E2EPrecommitBoundary.LIFECYCLE_SCOPE_BOUND
            ),
            "target.inspect": _E2EPrecommitBoundary.TARGET_INSPECT,
            "target.observe_precondition": (
                _E2EPrecommitBoundary.TARGET_PRECONDITION
            ),
        }.get(stage)
        if boundary is not None:
            await active_faults[0].reached(boundary)

    def local_component(stage: str) -> None:
        """Emit the local target handshake boundary from its native hook."""
        if active_faults and stage == "target.negotiate_capabilities":
            active_faults[0].reached_now(
                _E2EPrecommitBoundary.TARGET_NEGOTIATE
            )

    for identifier, boundary, executor in scenarios:
        cell = next(
            item
            for item in cells
            if isinstance(item, dict) and item["id"] == identifier
        )
        assert isinstance(cell, dict)
        with monkeypatch.context() as patcher:
            patcher.setattr(
                parser_module, "_test_precommit_checkpoint", sync_component
            )
            patcher.setattr(
                policy_module, "_test_precommit_checkpoint", async_component
            )
            patcher.setattr(
                target_module, "_test_precommit_checkpoint", async_component
            )
            patcher.setattr(
                local_commit_module, "_commit_barrier", local_component
            )
            active_faults[:] = []
            inspections, commits, observed = run(execute_scenario(executor))
        writes = int(_tree_snapshot(tmp_path) != before)
        assert _tree_snapshot(tmp_path) == before
        assert not tuple(tmp_path.glob(".avalan-patch-*"))
        assert inspections == cell["expected_inspection_count"]
        assert commits == cell["expected_commit_count"]
        assert writes == int(precommit["expected_workspace_write_count"])
        assert observed[-1:] == (cell["boundary"],)
        assert observed.count(cell["boundary"]) == 1


def test_patch_phase_7_e2e_003_runs_exhaustive_semantic_precommit_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise each semantic precommit category before any target write."""
    inventory = _e2e_inventory()
    precommit = inventory["precommit"]
    assert isinstance(precommit, dict)
    raw_categories = precommit["semantic_categories"]
    assert isinstance(raw_categories, list)
    categories = tuple(str(item) for item in raw_categories)
    assert categories == _SEMANTIC_PRECOMMIT_CATEGORIES
    assert len(categories) == len(set(categories))
    path = tmp_path / "note.txt"
    path.write_bytes(b"before\n")
    profile = _profile(tmp_path)
    before = _tree_snapshot(tmp_path)
    components: list[str] = []
    original_authorize_final = PolicyAuthorizer.authorize_final
    approval_runtime = run(
        _PHASE6["_runtime"](key="phase-seven-semantic-approval")
    )
    approved = approval_runtime[4]

    def parser_checkpoint(stage: str) -> None:
        """Record the parser boundary without changing parser semantics."""
        components.append(stage)

    async def policy_checkpoint(stage: str) -> None:
        """Record the policy boundary without preempting authorization."""
        components.append(stage)

    async def record_final_policy(
        authorizer: PolicyAuthorizer,
        preflight: PreflightAuthorization,
        candidate: PlannerCandidate,
        handshake: TargetHandshake,
    ) -> FinalAuthorization:
        """Record the final-policy call then preserve its actual result."""
        components.append("policy.final")
        return await original_authorize_final(
            authorizer,
            preflight,
            candidate,
            handshake,
        )

    def parse_edit(source_path: str, old_text: str, new_text: str) -> object:
        """Parse one real structured-edit request for pure planning."""
        return PatchRequestParser(PatchInputLimits()).parse(
            RawPatchIngress(
                RawProviderProfile("phase-seven"),
                RawToolCallId("semantic-edit"),
                RawPatchInputKind.EDIT_JSON,
                RawPatchInputState.COMPLETE,
                dumps(
                    {
                        "path": source_path,
                        "edits": [
                            {"old_text": old_text, "new_text": new_text}
                        ],
                    },
                    separators=(",", ":"),
                ).encode(),
            )
        )

    def parse_apply(*lines: str) -> object:
        """Parse one real apply document for transition planning."""
        return PatchRequestParser(PatchInputLimits()).parse(
            RawPatchIngress(
                RawProviderProfile("phase-seven"),
                RawToolCallId("semantic-apply"),
                RawPatchInputKind.APPLY_JSON,
                RawPatchInputState.COMPLETE,
                dumps(
                    {"patch": "\n".join(lines) + "\n"},
                    separators=(",", ":"),
                ).encode(),
            )
        )

    def workspace(source_path: str, value: bytes) -> PlannerWorkspace:
        """Build one exact immutable source fact for a pure plan."""
        view = LogicalText.from_bytes(value)
        return PlannerWorkspace(
            (
                PlannerFile(
                    LogicalPath(source_path),
                    SourceBytes(value),
                    MetadataProfile(
                        FileMode(0o644),
                        view.has_bom,
                        (
                            view.representation.value
                            if view.representation.value != "none"
                            else "lf"
                        ),
                    ),
                    None,
                    "semantic-mount",
                    "semantic-identity-" + source_path,
                ),
            ),
            frozenset(),
        )

    def planner_error(
        request: object,
        source: PlannerWorkspace,
        expected: PlannerErrorCode,
        limits: PlannerLimits = PlannerLimits(),
    ) -> None:
        """Assert an actual pure-plan rejection for one source fact."""
        with pytest.raises(PlannerError) as error:
            plan(cast("CanonicalPatchRequest", request), source, limits)
        assert error.value.code is expected

    async def exercise() -> None:
        """Run every immutable semantic category through its owner."""
        with pytest.raises(PatchInputError) as error:
            parse_apply("*** Begin Patch v1", "*** End Patch", "trailing")
        assert error.value.code is PatchInputErrorCode.GRAMMAR
        with pytest.raises(PatchInputError) as error:
            parse_edit("../escape", "before", "after")
        assert error.value.code is PatchInputErrorCode.PATH

        planner_error(
            parse_edit("missing.txt", "before", "after"),
            PlannerWorkspace((), frozenset()),
            PlannerErrorCode.SOURCE_MISSING,
        )
        planner_error(
            parse_edit("note.txt", "a", "b"),
            workspace("note.txt", b"a\na\n"),
            PlannerErrorCode.AMBIGUOUS_MATCH,
        )
        planner_error(
            parse_apply(
                "*** Begin Patch v1",
                "*** Add File: note.txt",
                "+again",
                "*** End Patch",
            ),
            workspace("note.txt", b"before\n"),
            PlannerErrorCode.DESTINATION_EXISTS,
        )
        nested = PatchRequestParser(PatchInputLimits()).parse(
            RawPatchIngress(
                RawProviderProfile("phase-seven"),
                RawToolCallId("semantic-nested"),
                RawPatchInputKind.EDIT_JSON,
                RawPatchInputState.COMPLETE,
                dumps(
                    {
                        "path": "note.txt",
                        "edits": [
                            {"old_text": "a\nb", "new_text": "x"},
                            {"old_text": "b", "new_text": "y"},
                        ],
                    },
                    separators=(",", ":"),
                ).encode(),
            )
        )
        planner_error(
            nested,
            workspace("note.txt", b"a\nb\n"),
            PlannerErrorCode.OVERLAPPING_EDITS,
        )
        overlap = PatchRequestParser(PatchInputLimits()).parse(
            RawPatchIngress(
                RawProviderProfile("phase-seven"),
                RawToolCallId("semantic-overlap"),
                RawPatchInputKind.EDIT_JSON,
                RawPatchInputState.COMPLETE,
                b'{"path":"note.txt","edits":['
                b'{"old_text":"a","new_text":"b"},'
                b'{"old_text":"a","new_text":"c"}]}',
            )
        )
        planner_error(
            overlap,
            workspace("note.txt", b"a\n"),
            PlannerErrorCode.OVERLAPPING_EDITS,
        )
        exact = plan(
            cast(
                "CanonicalPatchRequest",
                parse_edit("note.txt", "a\r\n", "b\r\n"),
            ),
            workspace("note.txt", b"a\r\n"),
        )
        exact_bytes = exact.lineages[0].final.bytes_value
        assert exact_bytes is not None
        assert exact_bytes._value == b"b\r\n"
        planner_error(
            parse_edit("note.txt", "a\n", "b\n"),
            workspace("note.txt", b"a\r\na\r\n"),
            PlannerErrorCode.AMBIGUOUS_MATCH,
        )
        planner_error(
            parse_edit("note.txt", "a", "a\n"),
            workspace("note.txt", b"a"),
            PlannerErrorCode.CONFLICT,
        )
        planner_error(
            parse_edit("note.txt", "a", "a"),
            workspace("note.txt", b"a\n"),
            PlannerErrorCode.NO_EFFECT,
        )
        planner_error(
            parse_edit("note.txt", "before", "after"),
            workspace("note.txt", b"before\n"),
            PlannerErrorCode.LIMIT,
            PlannerLimits(max_memory_bytes=1),
        )

        effective = _PHASE5["_effective_limits"]()
        forbidden = PreflightRequest(
            OperationType.EDIT,
            (LogicalPath(".git/config"),),
            frozenset((Capability.UPDATE,)),
            frozenset((LogicalPath(".git/config"),)),
            effective,
        )
        with pytest.raises(PolicyError) as policy_error:
            await PolicyAuthorizer(
                _PHASE5["_policy"]()
            ).authorize_preinspection(forbidden)
        assert policy_error.value.code is PolicyErrorCode.PATH_DENIED

        authorizer = PolicyAuthorizer(
            _PHASE5["_policy"](ApprovalMode.PREAUTHORIZED)
        )
        preflight = await authorizer.authorize_preinspection(
            PreflightRequest(
                OperationType.EDIT,
                (LogicalPath("note.txt"),),
                frozenset((Capability.UPDATE,)),
                frozenset((LogicalPath("note.txt"),)),
                effective,
            )
        )
        with pytest.raises(PolicyError) as final_policy_error:
            await authorizer.authorize_final(
                preflight,
                _PHASE5["_candidate"](),
                _PHASE5["_handshake"](effectful=False),
            )
        assert final_policy_error.value.code is PolicyErrorCode.DENIED

        denied = await ApprovalService(
            _PHASE5["_Broker"](ApprovalDecisionState.DENIED),
            _PHASE5["_Clock"](1),
            RuntimeGrantStore(),
        ).await_review(
            PlanReviewRequest(
                approved,
                approved.binding.subject,
                approved.binding.final.approval,
            )
        )
        assert denied.state is ApprovalDecisionState.DENIED

        (
            stale_coordinator,
            _,
            stale_identity,
            stale_digest,
            stale_plan,
            stale_grant,
        ) = await _PHASE6["_runtime"](
            key="phase-seven-semantic-stale",
            current=_PHASE6["_snapshot"](RevalidationField.WORKSPACE),
        )
        stale_reservation = await stale_coordinator.reserve(
            stale_identity, stale_digest
        )
        stale = await stale_coordinator.execute(
            stale_reservation,
            stale_plan,
            stale_grant,
            _PHASE6["_snapshot"](),
            ScriptedCommitWorker(WorkerReport(WorkerState.LIVE, None)),
            "phase-seven-semantic",
        )
        assert isinstance(stale, PatchResult)
        assert stale.status is PatchStatus.STALE

        for category, method in (
            ("cancellation", "cancel"),
            ("timeout", "_timeout_before_commit"),
        ):
            (
                coordinator,
                _,
                identity,
                request_digest,
                sealed,
                _,
            ) = await _PHASE6["_runtime"](
                key="phase-seven-semantic-" + category
            )
            reservation = await coordinator.reserve(identity, request_digest)
            await coordinator.prepare(
                reservation, sealed, approval_required=False
            )
            if method == "cancel":
                result = await coordinator.cancel(
                    reservation, before_commit=True
                )
            else:
                result = await coordinator._timeout_before_commit(reservation)
            assert isinstance(result, PatchResult)
            assert result.status is PatchStatus.CANCELLED

        with pytest.raises(TargetInspectionError) as target_error:
            LocalCommitTarget(replace(profile, mutation_test_profile=False))
        assert (
            target_error.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(
            parser_module, "_test_precommit_checkpoint", parser_checkpoint
        )
        patcher.setattr(
            policy_module, "_test_precommit_checkpoint", policy_checkpoint
        )
        patcher.setattr(
            PolicyAuthorizer, "authorize_final", record_final_policy
        )
        run(exercise())
    assert components.count("lifecycle.received") >= 11
    assert components.count("lifecycle.preinspection_authorized") >= 2
    assert components.count("policy.final") >= 1
    assert _tree_snapshot(tmp_path) == before
    assert profile.commit_namespace is not None
    assert not tuple(profile.commit_namespace.iterdir())


def test_patch_phase_7_e2e_004_uncertain_outcome_is_not_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain each local uncertain result without issuing a retry."""
    inventory = _e2e_inventory()
    raw_cases = inventory["uncertain_outcomes"]
    assert isinstance(raw_cases, list)
    expected = {
        str(raw["id"]): (
            tuple(raw["steps"]),
            str(raw["artifact"]),
            str(raw["postcondition"]),
        )
        for raw in raw_cases
        if isinstance(raw, dict)
    }
    assert tuple(expected) == (
        "partial",
        "indeterminate",
        "superseded",
        "leaked",
        "unknown_artifact",
    )
    expected_results = {
        "partial": (PatchStatus.PARTIAL, PatchErrorCode.PARTIAL_COMMIT),
        "indeterminate": (
            PatchStatus.INDETERMINATE,
            PatchErrorCode.INDETERMINATE,
        ),
        "superseded": (PatchStatus.COMMITTED, None),
        "leaked": (PatchStatus.COMMITTED, None),
        "unknown_artifact": (
            PatchStatus.INDETERMINATE,
            PatchErrorCode.INDETERMINATE,
        ),
    }

    async def commit_case(
        identifier: str,
    ) -> tuple[tuple[str, ...], str, str]:
        root = tmp_path / identifier
        root.mkdir()
        profile = _profile(root)
        if identifier in {"partial", "indeterminate"}:
            (root / "source.txt").write_bytes(b"before\n")
            document = "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: source.txt",
                    "*** Move to: destination.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            )
            files = {"source.txt": b"before\n"}
        else:
            document = "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: created.txt",
                    "+created",
                    "*** End Patch",
                )
            )
            files = {}
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(profile, target, scope, document, files)
        approvals = _PHASE6["ApprovalService"](
            _PHASE6["_Broker"](),
            _PHASE6["_Clock"](),
            _PHASE6["RuntimeGrantStore"](),
        )
        grant = await _PHASE6["_issue_grant"](sealed, approvals)
        store = InMemoryCoordinatorStore(approvals)
        coordinator = InMemoryPatchCoordinator(
            store,
            InMemoryLeaseManager(store),
            ScriptedReconciler(_PHASE6["_snapshot"]()),
        )
        reservation = await coordinator.reserve(
            RuntimeIdentity(
                sealed.binding.subject,
                PolicyRouteId("route-seven"),
                RetransmissionKey("phase-seven-uncertain-" + identifier),
            ),
            sealed.binding.request_digest,
        )
        original_barrier = local_commit_module._commit_barrier
        commits = 0

        def uncertain_boundary(stage: str) -> None:
            nonlocal commits
            original_barrier(stage)
            if stage == "artifact.stage":
                commits += 1
            if (
                identifier in {"partial", "indeterminate"}
                and stage == "move.source_remove_before"
            ):
                if identifier == "partial":
                    raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
                raise OSError("source removal outcome unavailable")
            if identifier == "superseded" and stage == "verification.before":
                (root / "created.txt").write_bytes(b"foreign\n")
            if identifier == "leaked":
                if stage == "artifact.cleanup_before":
                    raise OSError("staging cleanup unavailable")
                if stage == "verification.before":
                    (root / "created.txt").write_bytes(b"foreign\n")
            if identifier == "unknown_artifact":
                if stage == "artifact.stage_write_before":
                    raise OSError("staging write unavailable")
                if stage == "artifact.stage_cleanup_before":
                    raise OSError("staging cleanup outcome unavailable")

        with monkeypatch.context() as patcher:
            patcher.setattr(
                local_commit_module, "_commit_barrier", uncertain_boundary
            )
            worker = await target.worker(scope)
            result = await coordinator.execute(
                reservation,
                sealed,
                grant,
                _PHASE6["_snapshot"](),
                worker,
                "phase-seven-uncertain",
            )
            replay = await coordinator.execute(
                reservation,
                sealed,
                grant,
                _PHASE6["_snapshot"](),
                worker,
                "phase-seven-uncertain",
            )
        assert isinstance(result, PatchResult)
        assert replay is result
        assert commits == 1
        expected_status, expected_code = expected_results[identifier]
        assert result.status is expected_status
        if expected_code is None:
            assert result.diagnostic is None
        else:
            assert result.diagnostic is not None
            assert result.diagnostic.code is expected_code
        record = await store.record(reservation)
        assert record.journal is not None
        return (
            tuple(step.state.value for step in record.journal.steps),
            record.journal.artifacts[0].state.value,
            record.journal.postcondition.value,
        )

    observed = {
        identifier: run(commit_case(identifier)) for identifier in expected
    }
    assert observed == expected


def test_patch_phase_7_e2e_005_process_race_never_clobbers_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep rooted containment and no-replace boundaries race-safe."""
    inventory = _e2e_inventory()
    raw_races = inventory["process_races"]
    assert isinstance(raw_races, list)
    races = tuple(item for item in raw_races if isinstance(item, str))
    assert len(races) == len(raw_races)
    assert races == ("rooted_containment", "no_replace")
    context = get_context("fork")

    containment_root = tmp_path / "containment"
    containment_root.mkdir()
    note = containment_root / "note.txt"
    note.write_bytes(b"before\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "note.txt"
    canary.write_bytes(b"outside\n")
    containment_profile = _profile(containment_root)
    parent_started = context.Event()
    parent_replaced = context.Event()

    def replace_parent() -> None:
        assert parent_started.wait(2)
        containment_root.rename(tmp_path / "parked")
        symlink(outside, containment_root)
        parent_replaced.set()

    containment_process = context.Process(target=replace_parent)
    containment_process.start()
    original_barrier = local_commit_module._commit_barrier

    def swap_before_namespace_effect(stage: str) -> None:
        if stage == "target.namespace_before_effect":
            parent_started.set()
            assert parent_replaced.wait(2)

    async def containment_commit() -> None:
        scope = await LocalScopeResolver(containment_profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(containment_profile)
        sealed = await _sealed(
            containment_profile,
            target,
            scope,
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
            {"note.txt": b"before\n"},
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    containment_profile.identity.domain_id,
                    PatchRequestId("request_" + "3" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert report.journal.steps[0].state.value == "not_committed"

    try:
        monkeypatch.setattr(
            local_commit_module,
            "_commit_barrier",
            swap_before_namespace_effect,
        )
        run(containment_commit())
    finally:
        monkeypatch.setattr(
            local_commit_module, "_commit_barrier", original_barrier
        )
        containment_process.join(timeout=2)
    assert containment_process.exitcode == 0
    assert canary.read_bytes() == b"outside\n"
    assert (tmp_path / "parked" / "note.txt").read_bytes() == b"before\n"

    no_replace_root = tmp_path / "no-replace"
    no_replace_root.mkdir()
    profile = _profile(no_replace_root)
    publish_started = context.Event()
    foreign_written = context.Event()
    destination = no_replace_root / "created.txt"

    def foreign_writer() -> None:
        assert publish_started.wait(2)
        destination.write_bytes(b"foreign\n")
        foreign_written.set()

    process = context.Process(target=foreign_writer)
    process.start()

    def publish_after_final_check(stage: str) -> None:
        if stage == "target.namespace_after_final_check":
            publish_started.set()
            assert foreign_written.wait(2)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: created.txt",
                    "+planned",
                    "*** End Patch",
                )
            ),
            {},
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "3" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert report.journal.steps[0].state.value == "not_committed"

    try:
        monkeypatch.setattr(
            local_commit_module,
            "_commit_barrier",
            publish_after_final_check,
        )
        run(execute())
    finally:
        monkeypatch.setattr(
            local_commit_module, "_commit_barrier", original_barrier
        )
        process.join(timeout=2)
    assert process.exitcode == 0
    assert destination.read_bytes() == b"foreign\n"
    assert canary.read_bytes() == b"outside\n"


def test_patch_phase_7_revalidates_each_source_barrier_before_effect(
    tmp_path: Path,
) -> None:
    """Reject source replacement, disappearance, retyping, and mode drift."""
    cases = (
        "replace",
        "inode_swap",
        "disappear",
        "retype",
        "relink",
        "mode",
    )

    async def execute() -> None:
        for index, case in enumerate(cases):
            root = tmp_path / case
            root.mkdir()
            path = root / "note.txt"
            path.write_bytes(b"before\n")
            profile = _profile(root)
            scope = await LocalScopeResolver(profile).resolve(
                ScopeSelection(ContextKind.LOCAL)
            )
            target = LocalCommitTarget(profile)
            sealed = await _sealed(
                profile,
                target,
                scope,
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
                {"note.txt": b"before\n"},
            )
            match case:
                case "replace":
                    path.write_bytes(b"foreign\n")
                case "inode_swap":
                    replacement = root / "replacement.txt"
                    replacement.write_bytes(b"before\n")
                    chmod(replacement, 0o644)
                    replacement.replace(path)
                case "disappear":
                    path.unlink()
                case "retype":
                    path.unlink()
                    mkfifo(path)
                case "relink":
                    link(path, root / "alias.txt")
                case "mode":
                    chmod(path, 0o600)
                case _:
                    raise AssertionError(case)
            report = await _test_commit(
                await target.worker(scope),
                SealedCommitCommand(
                    sealed,
                    CommitLease(
                        profile.identity.domain_id,
                        PatchRequestId("request_" + str(index + 1) * 16),
                        1,
                    ),
                    footprint_for(sealed),
                ),
            )
            assert report.journal is not None
            assert report.journal.steps[0].state.value == "not_committed"

    run(execute())


def test_patch_phase_7_disables_unavailable_or_crossfs_link_primitives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed before create or move when POSIX link cannot publish."""
    (tmp_path / "source.txt").write_bytes(b"source\n")
    profile = _profile(tmp_path)
    original_barrier = local_commit_module._commit_barrier
    publications = 0

    def unavailable_link(stage: str) -> None:
        """Fail each child link publication with its native error class."""
        nonlocal publications
        original_barrier(stage)
        if stage != "publication.before_link":
            return
        publications += 1
        raise OSError(
            EXDEV if publications == 2 else ENOSYS, "link unavailable"
        )

    monkeypatch.setattr(
        local_commit_module, "_commit_barrier", unavailable_link
    )

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        worker = await target.worker(scope)
        plans = (
            (
                "a",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Add File: created.txt",
                            "+created",
                            "*** End Patch",
                        )
                    ),
                    {},
                ),
            ),
            (
                "b",
                await _sealed(
                    profile,
                    target,
                    scope,
                    "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: source.txt",
                            "*** Move to: moved.txt",
                            "*** End Patch",
                        )
                    ),
                    {"source.txt": b"source\n"},
                ),
            ),
        )
        for token, sealed in plans:
            report = await _test_commit(
                worker,
                SealedCommitCommand(
                    sealed,
                    CommitLease(
                        profile.identity.domain_id,
                        PatchRequestId("request_" + token * 16),
                        1,
                    ),
                    footprint_for(sealed),
                ),
            )
            assert report.journal is not None
            assert report.journal.steps[0].state.value == "not_committed"

    run(execute())
    assert not (tmp_path / "created.txt").exists()
    assert (tmp_path / "source.txt").read_bytes() == b"source\n"
    assert not (tmp_path / "moved.txt").exists()


def test_patch_phase_7_cancellation_never_exposes_partial_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow cancellation to detach without an in-place partial replacement."""
    path = tmp_path / "note.txt"
    path.write_bytes(b"before\n")
    profile = _profile(tmp_path)
    entered_stage = Event()
    release_stage = Event()
    original_barrier = local_commit_module._commit_barrier

    def paused_stage(stage: str) -> None:
        """Hold the authenticated child before its final update effect."""
        original_barrier(stage)
        if stage == "target.namespace_before_final_check":
            entered_stage.set()
            assert release_stage.wait(1)

    monkeypatch.setattr(local_commit_module, "_commit_barrier", paused_stage)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
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
            {"note.txt": b"before\n"},
        )
        approvals = _PHASE6["ApprovalService"](
            _PHASE6["_Broker"](),
            _PHASE6["_Clock"](),
            _PHASE6["RuntimeGrantStore"](),
        )
        grant = await _PHASE6["_issue_grant"](sealed, approvals)
        store = InMemoryCoordinatorStore(approvals)
        coordinator = InMemoryPatchCoordinator(
            store,
            InMemoryLeaseManager(store),
            ScriptedReconciler(_PHASE6["_snapshot"]()),
        )
        reservation = await coordinator.reserve(
            RuntimeIdentity(
                sealed.binding.subject,
                PolicyRouteId("route-seven"),
                RetransmissionKey("phase-seven-cancelled-update"),
            ),
            sealed.binding.request_digest,
        )
        worker = await target.worker(scope)
        task = create_task(
            coordinator.execute(
                reservation,
                sealed,
                grant,
                _PHASE6["_snapshot"](),
                worker,
                "phase-seven-cancelled-update",
            )
        )
        for _ in range(2_000):
            if entered_stage.is_set():
                break
            await async_sleep(0.001)
        assert entered_stage.is_set()
        task.cancel()
        pending = await task
        assert pending.__class__.__name__ == "_AttachedPending"
        assert path.read_bytes() == b"before\n"
        release_stage.set()
        settled = await coordinator.execute(
            reservation,
            sealed,
            grant,
            _PHASE6["_snapshot"](),
            worker,
            "phase-seven-cancelled-update",
        )
        assert isinstance(settled, PatchResult)
        assert settled.status is PatchStatus.COMMITTED
        assert path.read_bytes() == b"after\n"

    run(execute())


def test_patch_phase_7_cancelled_owner_reconciles_retained_local_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settle a cancelled owned local commit exactly once from its journal."""
    path = tmp_path / "note.txt"
    path.write_bytes(b"before\n")
    profile = _profile(tmp_path)
    entered_stage = Event()
    release_stage = Event()
    original_barrier = local_commit_module._commit_barrier

    def paused_stage(stage: str) -> None:
        """Hold the authenticated child before staging its replacement."""
        original_barrier(stage)
        if stage == "target.stage_artifact":
            entered_stage.set()
            assert release_stage.wait(1)

    monkeypatch.setattr(local_commit_module, "_commit_barrier", paused_stage)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        worker = await target.worker(scope)
        sealed = await _sealed(
            profile,
            target,
            scope,
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
            {"note.txt": b"before\n"},
        )
        approvals = _PHASE6["ApprovalService"](
            _PHASE6["_Broker"](),
            _PHASE6["_Clock"](),
            _PHASE6["RuntimeGrantStore"](),
        )
        grant = await _PHASE6["_issue_grant"](sealed, approvals)
        store = InMemoryCoordinatorStore(approvals)
        coordinator = InMemoryPatchCoordinator(
            store,
            InMemoryLeaseManager(store),
            ScriptedReconciler(_PHASE6["_snapshot"]()),
        )
        identity = RuntimeIdentity(
            sealed.binding.subject,
            PolicyRouteId("route-seven"),
            RetransmissionKey("phase-seven-cancelled-local"),
        )
        reservation = await coordinator.reserve(
            identity, sealed.binding.request_digest
        )
        task = create_task(
            coordinator.execute(
                reservation,
                sealed,
                grant,
                _PHASE6["_snapshot"](),
                worker,
                "phase-seven-owner",
            )
        )
        for _ in range(2_000):
            if entered_stage.is_set():
                break
            await async_sleep(0.001)
        assert entered_stage.is_set()
        task.cancel()
        pending = await task
        assert pending.__class__.__name__ == "_AttachedPending"
        assert path.read_bytes() == b"before\n"
        scripted_alternate = ScriptedCommitWorker(
            WorkerReport(WorkerState.LIVE, None)
        )
        with pytest.raises(CoordinatorError) as error:
            await coordinator.execute(
                reservation,
                sealed,
                grant,
                _PHASE6["_snapshot"](),
                scripted_alternate,
                "phase-seven-owner",
            )
        assert error.value.code is CoordinatorErrorCode.FENCED
        alternate_worker = await target.worker(scope)
        with pytest.raises(CoordinatorError) as error:
            await coordinator.execute(
                reservation,
                sealed,
                grant,
                _PHASE6["_snapshot"](),
                alternate_worker,
                "phase-seven-owner",
            )
        assert error.value.code is CoordinatorErrorCode.FENCED
        release_stage.set()
        settled = await coordinator.execute(
            reservation,
            sealed,
            grant,
            _PHASE6["_snapshot"](),
            worker,
            "phase-seven-owner",
        )
        assert isinstance(settled, PatchResult)
        assert settled.status is PatchStatus.COMMITTED
        assert path.read_bytes() == b"after\n"
        replay = await coordinator.execute(
            reservation,
            sealed,
            grant,
            _PHASE6["_snapshot"](),
            worker,
            "phase-seven-owner",
        )
        assert replay is settled

    run(execute())
    monkeypatch.setattr(
        local_commit_module, "_commit_barrier", original_barrier
    )
    assert path.read_bytes() == b"after\n"


def test_patch_phase_7_never_updates_git_index_or_head(tmp_path: Path) -> None:
    """Leave the repository index and HEAD untouched by every effect kind."""

    def git(*arguments: str) -> str:
        result = run_process(
            ("git", "-C", str(tmp_path), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    git("init", "-q")
    git("config", "user.email", "patch@example.test")
    git("config", "user.name", "Patch Test")
    sources = {
        "delete.txt": b"delete\n",
        "move.txt": b"move\n",
        "move-update.txt": b"before\n",
        "note.txt": b"before\n",
    }
    for path, value in sources.items():
        (tmp_path / path).write_bytes(value)
    git("add", *sources)
    git("commit", "-q", "-m", "baseline")
    head_before = git("rev-parse", "HEAD")
    index_before = git("diff", "--cached", "--binary")
    profile = _profile(tmp_path)

    async def execute() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await _sealed(
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: create.txt",
                    "+created",
                    "*** Delete File: delete.txt",
                    "*** Update File: move.txt",
                    "*** Move to: moved.txt",
                    "*** Update File: move-update.txt",
                    "*** Move to: moved-update.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** Update File: note.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            ),
            sources,
        )
        report = await _test_commit(
            await target.worker(scope),
            SealedCommitCommand(
                sealed,
                CommitLease(
                    profile.identity.domain_id,
                    PatchRequestId("request_" + "7" * 16),
                    1,
                ),
                footprint_for(sealed),
            ),
        )
        assert report.journal is not None
        assert report.journal.postcondition is PostconditionState.ESTABLISHED

    run(execute())
    assert git("rev-parse", "HEAD") == head_before
    assert git("diff", "--cached", "--binary") == index_before
    status = git("status", "--porcelain")
    for path in (
        "create.txt",
        "delete.txt",
        "move.txt",
        "move-update.txt",
        "moved.txt",
        "moved-update.txt",
        "note.txt",
    ):
        assert path in status
    assert "git" not in Path(local_commit_module.__file__).read_text(
        encoding="utf-8"
    )
