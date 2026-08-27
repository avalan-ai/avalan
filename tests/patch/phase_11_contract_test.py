"""Exercise the persistent narrow Docker patch service."""

from asyncio import Event, create_subprocess_exec, create_task, run, sleep
from asyncio.subprocess import DEVNULL, PIPE, Process
from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from pathlib import Path
from platform import machine
from runpy import run_path
from subprocess import run as run_process
from sys import argv, executable
from types import SimpleNamespace
from typing import Protocol, cast

import pytest
from context_contract_corpus import (
    SHARED_CONTEXT_CORPUS,
    ContextCorpusCase,
)

import avalan.patch.container_target as _CONTAINER
import avalan.patch.rooted_worker as _ROOTED
import avalan.patch.sandbox_commit as _SANDBOX
import avalan.patch.sandbox_worker as _WORKER
import avalan.patch.toolset as _TOOLSET
from avalan._patch_authority import _PatchAuthorityValidator
from avalan.patch.container_target import (
    ContainerInspectionTarget,
    ContainerPatchImage,
    ContainerPatchRuntimeBinder,
    ContainerPatchRuntimeContext,
    ContainerPatchRuntimeSettings,
    ContainerPatchTarget,
    ContainerPersistentLeaseAuthority,
    _docker_output,
    container_protocol_id,
)
from avalan.patch.domain import (
    ApprovalMode,
    ByteSize,
    Capability,
    ContextKind,
    DurationTicks,
    ExpiryTick,
    FileMode,
    LifecyclePhase,
    LogicalPath,
    MetadataProfile,
    OperationType,
    PatchApprovalId,
    PatchContextId,
    PatchDomainId,
    PatchLimits,
    PatchLineageId,
    PatchObserverCorrelationId,
    PatchPending,
    PatchPendingOperationId,
    PatchPlanId,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PatchTargetId,
    PatchWorkspaceId,
    ProposedBytes,
    SequenceNumber,
)
from avalan.patch.durable_approval import (
    HmacDurableApprovalAuthority,
    PhaseFiveDurableApprovalIssuer,
)
from avalan.patch.durable_store import (
    DurableCommitLease,
    DurableJournalCursor,
    DurablePatchStoreBinding,
    DurableStoreError,
    DurableStoreErrorCode,
    DurableTerminalRecord,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.planner import (
    BoundedPlannerWorker,
    PlannedFile,
    PlannedLineage,
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
    PolicyBrokerId,
    PolicyPathSelector,
    PolicyReviewerRole,
    PolicyRevision,
    PolicyRouteId,
    PolicyRule,
    PreauthorizationClass,
    ReviewerDecision,
    RuntimeGrantStore,
    TrustedPatchPolicy,
)
from avalan.patch.sandbox_commit import (
    SandboxChannelId,
    SandboxContextLifetimeId,
    SandboxExecutionPlanFingerprint,
    SandboxPatchSdkService,
    SandboxPatchServiceConfiguration,
    SandboxWorkerImplementationId,
)
from avalan.patch.target import (
    FileIdentity,
    InspectionRequest,
    ScopeSelection,
    TargetErrorCode,
    TargetIdentity,
    TargetInspectionError,
)
from avalan.patch.toolset import (
    PatchApprovalBinding,
    PatchCapabilitySnapshot,
    PatchCoordinatorBinding,
    PatchInvocationHandle,
    PatchPersistenceBinding,
    PatchRuntimeBinding,
    PatchSdkHost,
    PatchTestHostProfile,
    PatchToolError,
    PatchToolLoader,
    project_model_result,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "patch"
_TEST_WORKER_BASE_IMAGE = (
    "python:3.11-slim-bookworm@sha256:"
    "2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91"
)
_TEST_WORKER_WHEELHOUSE = _FIXTURES / "container_wheels"
_TEST_CFFI_ARM64_WHEEL = ".".join(
    (
        "cffi-2.0.0-cp311-cp311-manylinux2014_aarch64",
        "manylinux_2_17_aarch64.whl",
    )
)
_TEST_CRYPTOGRAPHY_ARM64_WHEEL = ".".join(
    (
        "cryptography-48.0.1-cp311-abi3-manylinux2014_aarch64",
        "manylinux_2_17_aarch64.whl",
    )
)
_TEST_CRYPTOGRAPHY_X86_64_WHEEL = ".".join(
    (
        "cryptography-48.0.1-cp311-abi3-manylinux2014_x86_64",
        "manylinux_2_17_x86_64.whl",
    )
)
_TEST_WORKER_WHEEL_SHA256 = {
    _TEST_CFFI_ARM64_WHEEL: (
        "730cacb21e1bdff3ce90babf007d0a0917cc3e6492f336c2f0134101e0944f93"
    ),
    "cffi-2.0.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": (
        "8941aaadaf67246224cee8c3803777eed332a19d909b47e29c9842ef1e79ac26"
    ),
    _TEST_CRYPTOGRAPHY_ARM64_WHEEL: (
        "32143b24adb918f078134e1e230f1eb8cc04886b92c28b5f0041aaf3e5699225"
    ),
    _TEST_CRYPTOGRAPHY_X86_64_WHEEL: (
        "f0d27a5696721ef7a672b8c810f6aded391058e0b9486e63e6d93baf765da691"
    ),
    "pycparser-2.23-py3-none-any.whl": (
        "e5c6e8d3fbad53479cab09ac03729e0a9faf2bee3db8208a550daf5af81a5934"
    ),
}
_TEST_LEASE_AUTHORITY = ContainerPersistentLeaseAuthority.from_bytes(
    b"phase11-container-lease-authorit"
)
_RESTART_PROCESS_BOOTSTRAP = (
    "import runpy,sys;"
    "sys.path.insert(0,sys.argv[1]);"
    "sys.argv=sys.argv[2:];"
    "runpy.run_path(sys.argv[0],run_name='__main__')"
)


class _DefaultContainerAuthority(Protocol):
    """Describe the closed default container authority hook."""

    @staticmethod
    def container_endpoint_is_issued(endpoint: object) -> bool:
        """Return whether a default container endpoint is issued."""


class _ContainerPipe:
    """Provide one bounded in-memory subprocess stream."""

    def __init__(
        self,
        *,
        drain_error: BaseException | None = None,
        lines: list[bytes] | None = None,
    ) -> None:
        """Create one stream with optional finite read or drain failures."""
        self.drain_error = drain_error
        self.lines = [] if lines is None else lines
        self.writes: list[bytes] = []

    def write(self, value: bytes) -> None:
        """Record one worker-channel message."""
        self.writes.append(value)

    async def drain(self) -> None:
        """Settle one written message or raise the configured failure."""
        if self.drain_error is not None:
            raise self.drain_error

    async def readline(self) -> bytes:
        """Return one configured response line."""
        if not self.lines:
            return b""
        value = self.lines.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class _ContainerProcess:
    """Expose only the subprocess operations used by the Docker authority."""

    def __init__(
        self,
        *,
        stdin: _ContainerPipe | None = None,
        stdout: _ContainerPipe | None = None,
        returncode: int | None = None,
        wait_error: BaseException | None = None,
    ) -> None:
        """Create one finite process response fixture."""
        self.stdin = stdin
        self.stdout = stdout
        self.returncode = returncode
        self.wait_error = wait_error
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def terminate(self) -> None:
        """Record a graceful termination request."""
        self.terminated = True

    def kill(self) -> None:
        """Record a forced termination request."""
        self.killed = True

    async def wait(self) -> int:
        """Settle the process or raise the configured deadline fault."""
        self.wait_calls += 1
        if self.wait_error is not None and self.wait_calls == 1:
            raise self.wait_error
        return 0

    async def communicate(self) -> tuple[bytes, bytes]:
        """Return one finite Docker CLI response."""
        return b"output", b""


class _DockerOutputProcess(_ContainerProcess):
    """Provide one Docker CLI output fixture with an exact return code."""

    def __init__(self, output: bytes, returncode: int = 0) -> None:
        """Create one deterministic Docker CLI output response."""
        super().__init__(returncode=returncode)
        self.output = output

    async def communicate(self) -> tuple[bytes, bytes]:
        """Return the configured raw Docker CLI output."""
        return self.output, b""


class _SlowDockerOutputProcess(_ContainerProcess):
    """Keep a Docker CLI output request pending until its deadline."""

    async def communicate(self) -> tuple[bytes, bytes]:
        """Exceed the caller-selected bounded command deadline."""
        await sleep(60.0)
        return b"", b""


class _UnreapableDockerOutputProcess(_SlowDockerOutputProcess):
    """Keep the Docker child unreaped after a forced termination request."""

    async def wait(self) -> int:
        """Exceed each bounded process-reap deadline."""
        raise TimeoutError()


class _PermitPipe(_ContainerPipe):
    """Raise only when the authority writes the second fenced message."""

    def __init__(self) -> None:
        """Create a channel whose permit write times out."""
        super().__init__()
        self.drains = 0

    async def drain(self) -> None:
        """Settle the request then fail the permit response."""
        self.drains += 1
        if self.drains == 2:
            raise TimeoutError()


class _Bundle:
    """Model the one implementation bundle cleanup boundary."""

    digest = "implementation-digest"
    source_digest = "source-digest"
    root = Path("/private/implementation")

    def __init__(self) -> None:
        """Create a bundle with observable cleanup."""
        self.closed = False

    def close(self) -> None:
        """Record bundle cleanup."""
        self.closed = True


def _mock_bundle_readable(root: Path) -> None:
    """Assert the fake bundle crossed the sealed-readability boundary."""
    assert root == _Bundle.root


class _BoundStore(InMemoryDurablePatchStore):
    """Give the in-memory store its required async owner lifetime."""

    async def __aenter__(self) -> "_BoundStore":
        """Return the one owned shared durable store."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Close no external resource for the finite in-memory fixture."""
        del exc_type, exc_value, traceback


def _limits() -> PatchLimits:
    """Return the finite limits for the exact Docker test profile."""
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
        DurationTicks(1_000),
    )


class _Clock(ApprovalClock):
    """Return a stable trusted time for the finite Docker E2E."""

    async def now(self) -> ExpiryTick:
        """Return the nonexpired test-profile tick."""
        return ExpiryTick(1)


class _BlockingFenceStore(InMemoryDurablePatchStore):
    """Pause the first Docker effect at the durable authority fence."""

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


class _SecondEffectFenceStore(InMemoryDurablePatchStore):
    """Pause a Docker second effect and retain its journal as pending."""

    def __init__(self, backend: InMemoryDurablePatchBackend) -> None:
        """Create the exact second-effect authority barrier."""
        super().__init__(backend)
        self.effect_reached = Event()
        self.release_effect = Event()
        self.checks = 0
        self.reject_terminal_once = True

    async def is_current_fence(
        self, lease: DurableCommitLease, now: ExpiryTick
    ) -> bool:
        """Pause only the second per-effect fence after the initial check."""
        self.checks += 1
        if self.checks == 5:
            self.effect_reached.set()
            await self.release_effect.wait()
        return await super().is_current_fence(lease, now)

    async def settle(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        result: PatchResult,
        correlation_id: PatchObserverCorrelationId,
        now: ExpiryTick,
    ) -> DurableTerminalRecord:
        """Model one post-effect terminal gap without losing its journal."""
        if self.reject_terminal_once:
            self.reject_terminal_once = False
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_INCOMPLETE)
        return await super().settle(
            lease, expected, result, correlation_id, now
        )


class _Broker:
    """Approve the one test-profile policy review route."""

    async def decide(self, request: object) -> BrokerDecision:
        """Return the policy-matching single reviewer approval."""
        requirements = getattr(request, "requirements")
        subject = getattr(request, "subject")
        return BrokerDecision(
            requirements.broker,
            (
                ReviewerDecision(
                    PatchPrincipalId("container-reviewer"),
                    subject.tenant,
                    requirements.reviewer_role,
                    ApprovalDecisionState.APPROVED,
                ),
            ),
        )


def _subject() -> ExecutionSubject:
    """Return the host-owned principal for the selected container context."""
    return ExecutionSubject(
        PatchPrincipalId("container-principal"),
        PatchTenantId("container-tenant"),
        PatchRunId("container-run"),
        PatchSessionId("container-session"),
        PatchTaskId("container-task"),
        PatchAgentId("container-agent"),
    )


def _policy() -> TrustedPatchPolicy:
    """Authorize one reviewed update through the standard patch pipeline."""
    reader = PreauthorizationClass("container-read")
    return TrustedPatchPolicy(
        PolicyRevision("policy-v2"),
        frozenset((OperationType.EDIT, OperationType.APPLY)),
        (
            PolicyRule(
                PolicyPathSelector(None),
                (
                    CapabilityMode(
                        Capability.UPDATE, ApprovalMode.REQUIRE_REVIEW
                    ),
                    CapabilityMode(
                        Capability.CREATE, ApprovalMode.REQUIRE_REVIEW
                    ),
                    CapabilityMode(
                        Capability.DELETE, ApprovalMode.REQUIRE_REVIEW
                    ),
                    CapabilityMode(
                        Capability.MOVE, ApprovalMode.REQUIRE_REVIEW
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
            PolicyRouteId("container-route"),
            PolicyBrokerId("container-broker"),
            PolicyReviewerRole("container-reviewer"),
            1,
        ),
    )


def _settings(root: Path, image: str) -> ContainerPatchRuntimeSettings:
    """Return one immutable selected Docker service settings value."""
    token = sha256(str(root).encode()).hexdigest()[:16]
    implementation = SandboxWorkerImplementationId("container-runtime-v1")
    identity = TargetIdentity(
        PatchContextId("context_" + token),
        PatchWorkspaceId("workspace_" + token),
        PatchDomainId("domain_" + token),
        PatchTargetId("target_" + token),
        container_protocol_id(),
        "docker-volume-" + token,
        "docker-mount-" + token,
        "policy-v2",
        "persistent-lease-" + token,
        PatchApprovalId("approval_" + token),
        implementation,
    )
    return ContainerPatchRuntimeSettings(
        ContainerPatchImage(image),
        ContainerPatchRuntimeContext(
            identity,
            _limits(),
            ByteSize(65_536),
            None,
            SandboxChannelId("container-channel-v1"),
            SandboxContextLifetimeId("container-context-v1"),
            implementation,
        ),
        root,
        SandboxExecutionPlanFingerprint("container-test-plan-v1"),
        _TEST_LEASE_AUTHORITY,
        test_profile=True,
    )


def test_patch_phase_11_parent_worker_dispatch_matches_container_mutation(
    tmp_path: Path,
) -> None:
    """Exercise sealed container-worker semantics in the parent test.

    Run the worker behavior without relaxing its sealed subprocess boundary.
    """
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "note.txt").write_bytes(b"before\n")
    root_status = root.stat()
    witness, mount_binding = _ROOTED.capture_rooted_root_binding(root)
    identity = {
        "context": "context",
        "workspace": "workspace",
        "domain": "domain",
        "target": "target",
        "protocol": "protocol",
        "filesystem": witness.filesystem_id,
        "mount": witness.mount_id,
        "policy": "policy",
        "persistent_lease": "lease",
        "approval": "approval",
        "implementation": "implementation",
    }
    config = cast(
        _WORKER._RuntimeChildConfig,
        {
            "root": str(root),
            "namespace": "/private",
            "cwd": None,
            "maximum": 1_024,
            "aggregate_maximum": 4_096,
            "token": "a" * 64,
            "receipt": "receipt",
            "identity": identity,
            "channel_id": "channel",
            "implementation_id": "implementation",
            "implementation_digest": "implementation-digest",
            "source_digest": "source-digest",
            "implementation_root": "/implementation",
            "read_canary": str(tmp_path / "host-canary"),
            "session_id": "session",
            "execution_plan": _WORKER._ExecutionPlanFingerprint(
                "execution-plan"
            ),
            "backend": "container",
            "workspace_view": "/workspace",
            "private_view": "/private",
            "context_lifetime": "lifetime",
            "protocol": "protocol",
            "persistent_lease": "lease",
            "filesystem": witness.filesystem_id,
            "mount": witness.mount_id,
        },
    )
    request = cast(
        _WORKER._RuntimeRequestPayload,
        {
            "version": _WORKER._MESSAGE_VERSION,
            "sequence": 1,
            "kind": "inspect",
            "receipt": "receipt",
            "identity": identity,
            "channel_id": "channel",
            "implementation_id": "implementation",
            "body": {},
        },
    )

    observed_witness, witness_closed = _WORKER._child_dispatch(
        "witness", {}, config, witness, request, b"a" * 32, mount_binding
    )
    assert observed_witness == {"root": _WORKER._root_payload(witness)}
    assert not witness_closed
    canary, canary_closed = _WORKER._child_dispatch(
        "canary", {}, config, witness, request, b"a" * 32, mount_binding
    )
    assert canary["outside_read_denied"] is True
    assert isinstance(canary["metadata_probe"], str)
    assert len(canary["metadata_probe"]) == 64
    assert not canary_closed
    inspected, inspected_closed = _WORKER._child_dispatch(
        "inspect",
        {"paths": ["note.txt"], "root": _WORKER._root_payload(witness)},
        config,
        witness,
        request,
        b"a" * 32,
        mount_binding,
    )
    snapshots = cast(list[Mapping[str, object]], inspected["snapshots"])
    assert len(snapshots) == 1
    assert snapshots[0]["path"] == "note.txt"
    assert snapshots[0]["bytes"] == b64encode(b"before\n").decode()
    assert not inspected_closed

    path = LogicalPath("created.txt")
    value = ProposedBytes(b"created\n")
    command = _ROOTED.RootedMutationCommand(
        PatchPlanId("plan_" + "a" * 16),
        (
            PlannedLineage(
                PatchLineageId("lineage_" + "a" * 16),
                PlannedFile(path, False, None, None, None, ByteSize(0)),
                PlannedFile(
                    path,
                    True,
                    value,
                    MetadataProfile(FileMode(0o644), False, "lf"),
                    value.digest(),
                    value.size(),
                ),
                None,
                path,
                frozenset((Capability.CREATE,)),
                (),
                (),
                ("created",),
                (path,),
                "single_step",
                ("create",),
                "target_private",
                b"",
                ((None, (root_status.st_dev, root_status.st_ino)),),
            ),
        ),
        frozenset((Capability.CREATE,)),
    )
    fences: list[None] = []
    report = _ROOTED._commit_rooted(
        command,
        _ROOTED.RootedMutationProfile(root, None, FileMode(0o644)),
        witness,
        lambda: fences.append(None),
        mount_binding=mount_binding,
    )
    assert fences
    assert report.journal is not None
    assert report.journal.postcondition.value == "established"
    assert report.journal.steps[0].state.value == "committed"
    assert (root / path.value).read_bytes() == b"created\n"

    closed, should_close = _WORKER._child_dispatch(
        "close", {}, config, witness, request, b"a" * 32, mount_binding
    )
    assert closed == {}
    assert should_close


def test_patch_phase_11_parent_service_rejects_forged_access() -> None:
    """Reject forged reads and mismatched endpoint contexts."""

    async def exercise() -> None:
        """Exercise only the parent-owned service access decisions."""
        service = object.__new__(SandboxPatchSdkService)
        endpoint = object()
        service.runtime = SimpleNamespace(
            _bind_sandbox_endpoint=lambda scope: endpoint
        )
        service.scope = SimpleNamespace(context_kind=ContextKind.CONTAINER)
        with pytest.raises(TargetInspectionError) as sandbox_endpoint:
            service._patch_sandbox_endpoint()
        assert sandbox_endpoint.value.code is TargetErrorCode.WITNESS_STALE
        assert service._patch_container_endpoint() is endpoint

        service.scope = SimpleNamespace(context_kind=ContextKind.SANDBOX)
        with pytest.raises(TargetInspectionError) as container_endpoint:
            service._patch_container_endpoint()
        assert container_endpoint.value.code is TargetErrorCode.WITNESS_STALE

        service._latest = None
        with pytest.raises(TargetInspectionError) as latest:
            await service._inspect_latest()
        assert latest.value.code is TargetErrorCode.WORKER_UNAVAILABLE

        service._requests = {}
        service._reader_tasks = set()
        forged = _SANDBOX._SandboxSettlementPort(service).inspect(
            PatchInvocationHandle(object())
        )
        with pytest.raises(TargetInspectionError) as forged_error:
            await forged
        assert forged_error.value.code is TargetErrorCode.WITNESS_STALE

        unknown = service._inspect_request_future(
            PatchRequestId("request_" + "a" * 16),
            PatchObserverCorrelationId("correlation_" + "a" * 16),
        )
        with pytest.raises(TargetInspectionError) as unknown_error:
            await unknown
        assert unknown_error.value.code is TargetErrorCode.WITNESS_STALE

    run(exercise())


async def _restart_process_from_config(config_path: Path) -> None:
    """Run one host process against a persisted authenticated Docker lease."""
    payload = loads(config_path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise AssertionError("restart process configuration is invalid")
    root_value = payload.get("root")
    image = payload.get("image")
    authority_value = payload.get("authority")
    old_text = payload.get("old_text")
    new_text = payload.get("new_text")
    domain = payload.get("domain")
    barrier_ready = payload.get("barrier_ready")
    barrier_release = payload.get("barrier_release")
    barrier_stage = payload.get("barrier_stage", "volume_create")
    startup_failure = payload.get("startup_failure")
    if (
        type(root_value) is not str
        or type(image) is not str
        or type(authority_value) is not str
        or type(old_text) is not str
        or type(new_text) is not str
        or domain is not None
        and type(domain) is not str
        or barrier_ready is not None
        and type(barrier_ready) is not str
        or barrier_release is not None
        and type(barrier_release) is not str
        or (barrier_ready is None) != (barrier_release is None)
        or type(barrier_stage) is not str
        or barrier_stage
        not in {"volume_create", "guard_acquired", "guard_released"}
        or startup_failure is not None
        and type(startup_failure) is not str
        or startup_failure is not None
        and startup_failure not in {"attach"}
    ):
        raise AssertionError("restart process configuration is invalid")
    authority = ContainerPersistentLeaseAuthority.from_bytes(
        b64decode(authority_value.encode(), validate=True)
    )
    settings = replace(
        _settings(Path(root_value), image),
        persistent_lease_authority=authority,
    )
    if domain is not None:
        settings = replace(
            settings,
            context=replace(
                settings.context,
                identity=replace(
                    settings.context.identity,
                    domain_id=PatchDomainId(domain),
                ),
            ),
        )
    if barrier_ready is not None and barrier_release is not None:
        original_output = _CONTAINER._docker_output
        paused = False
        guard_name = _CONTAINER._docker_name(
            "avalan_patch_lease_",
            settings.context.identity.persistent_lease_id,
        )

        async def gated_output(
            command: tuple[str, ...],
            required: bool = True,
            timeout: float = _CONTAINER._DOCKER_COMMAND_TIMEOUT_SECONDS,
        ) -> str | None:
            """Pause a child after labeling and before worker startup."""
            nonlocal paused
            output = await original_output(command, required, timeout)
            is_volume_create = command[:3] == (
                "docker",
                "volume",
                "create",
            )
            is_guard_create = command[:2] == ("docker", "run") and (
                "--name" in command
            )
            is_guard_release = command == (
                "docker",
                "rm",
                "--force",
                guard_name,
            )
            if not paused and (
                (barrier_stage == "volume_create" and is_volume_create)
                or (barrier_stage == "guard_acquired" and is_guard_create)
                or (barrier_stage == "guard_released" and is_guard_release)
            ):
                paused = True
                Path(barrier_ready).write_text("ready\n", encoding="utf-8")
                for _ in range(1_000):
                    if Path(barrier_release).is_file():
                        break
                    await sleep(0.01)
                else:
                    raise AssertionError(
                        "initial volume race barrier timed out"
                    )
            return output

        _CONTAINER._docker_output = gated_output
    if startup_failure == "attach":
        original_start = _CONTAINER.create_subprocess_exec

        async def failed_attach(*command: str, **kwargs: object) -> Process:
            """Abort only the fresh worker attach after seeding its volume."""
            if command[:2] == ("docker", "start"):
                raise OSError("controlled worker attach failure")
            return await original_start(*command, **kwargs)

        _CONTAINER.create_subprocess_exec = failed_attach
    approval_authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=approval_authority)
    )
    binder = _binder(
        settings, _configuration(_Clock(), approval_authority), store
    )
    try:
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.edit"])
    except TargetInspectionError as error:
        print(dumps({"error": error.code.value}), flush=True)
        return
    assert bundle.toolset is not None
    await bundle.manager.__aenter__()
    try:
        outcome = await bundle.toolset.sdk_host().invoke_json(
            OperationType.EDIT,
            {
                "path": "note.txt",
                "edits": [{"old_text": old_text, "new_text": new_text}],
            },
        )
        assert type(outcome) is PatchResult
        assert outcome.status is PatchStatus.COMMITTED
        scope = await binder.runtime.resolve(
            ScopeSelection(ContextKind.CONTAINER)
        )
        inspected = await ContainerInspectionTarget(binder.runtime).inspect(
            InspectionRequest(scope, (LogicalPath("note.txt"),))
        )
        snapshot = inspected.snapshots[0]
        assert snapshot.bytes_value is not None
        print(
            dumps(
                {
                    "status": outcome.status.value,
                    "bytes": b64encode(snapshot.bytes_value._value).decode(),
                }
            ),
            flush=True,
        )
    finally:
        await bundle.manager.__aexit__(None, None, None)
        await binder.runtime.close()


async def _run_restart_process(
    config_path: Path,
) -> Mapping[str, object]:
    """Run one isolated calling process and return its closed result."""
    process = await create_subprocess_exec(
        executable,
        "-c",
        _RESTART_PROCESS_BOOTSTRAP,
        str(Path(__file__).resolve().parent),
        str(Path(__file__).resolve()),
        "--phase11-restart-process",
        str(config_path),
        stdin=DEVNULL,
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode("utf-8")
    payload = loads(stdout.decode("utf-8"))
    assert type(payload) is dict
    return payload


def test_patch_phase_11_container_authority_rejects_unavailable_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed at each direct container authority boundary."""
    seed = tmp_path / "seed"
    seed.mkdir()
    image = "sha256:" + "a" * 64
    settings = _settings(seed, image)

    async def output(
        command: tuple[str, ...], required: bool = True
    ) -> str | None:
        """Return only bounded Docker fixture responses."""
        del command, required
        return "container-id\n"

    async def start_process(*_: object, **__: object) -> _ContainerProcess:
        """Return a finite no-op Docker attach process."""
        return _ContainerProcess(
            stdin=_ContainerPipe(), stdout=_ContainerPipe()
        )

    async def exercise() -> None:
        """Exercise direct worker failures without host mutation fallback."""
        assert not _PatchAuthorityValidator.container_endpoint_is_issued(
            object()
        )
        with pytest.raises(TargetInspectionError):
            _CONTAINER.container_protocol_id(1)
        with pytest.raises(TargetInspectionError):
            ContainerPatchImage("python:latest")
        with pytest.raises(TargetInspectionError):
            ContainerPatchRuntimeContext(
                settings.context.identity,
                settings.context.limits,
                settings.context.max_snapshot_bytes,
                settings.context.cwd,
                settings.context.channel_id,
                settings.context.context_lifetime_id,
                SandboxWorkerImplementationId("wrong-implementation"),
            )
        with pytest.raises(TargetInspectionError):
            ContainerPatchRuntimeSettings(
                settings.image,
                settings.context,
                seed,
                settings.execution_plan_fingerprint,
                settings.persistent_lease_authority,
            )
        with pytest.raises(TargetInspectionError):
            _CONTAINER.ContainerRuntimeReceipt(
                "", "", object(), object(), {}, "", "", "", ""
            )

        process = _CONTAINER._ContainerRuntimeProcess(settings)
        process._closed = True
        with pytest.raises(TargetInspectionError) as closed:
            await process.start()
        assert closed.value.code is TargetErrorCode.WORKER_UNAVAILABLE

        process = _CONTAINER._ContainerRuntimeProcess(settings)
        process._process = _ContainerProcess()
        process._root = _CONTAINER.RootWitness(FileIdentity(1, 1), "mount")
        with pytest.raises(TargetInspectionError):
            await process.start()

        process = _CONTAINER._ContainerRuntimeProcess(settings)
        with pytest.raises(TargetInspectionError) as unstarted:
            await process.inspect(
                (), _CONTAINER.RootWitness(FileIdentity(1, 1), "mount")
            )
        assert unstarted.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        with pytest.raises(TargetInspectionError):
            await process.commit(object(), object())
        with pytest.raises(TargetInspectionError):
            await process._runtime_receipt_locked()

        process._process = _ContainerProcess()
        process._token = b"token"
        process._receipt = "receipt"
        with pytest.raises(TargetInspectionError):
            await process._request_locked("inspect", {})

        process = _CONTAINER._ContainerRuntimeProcess(settings)
        process._process = _ContainerProcess(
            stdin=_ContainerPipe(),
            stdout=_ContainerPipe(lines=[b"response"]),
        )
        process._token = b"token"
        process._receipt = "receipt"
        monkeypatch.setattr(
            _CONTAINER,
            "_response_payload",
            lambda *_: {"control": "wrong", "effect": 1},
        )
        with pytest.raises(TargetInspectionError):
            await process._request_locked("commit", {})

        process = _CONTAINER._ContainerRuntimeProcess(settings)
        process._process = _ContainerProcess(
            stdin=_ContainerPipe(), stdout=_ContainerPipe()
        )
        process._token = b"token"
        process._receipt = "receipt"
        with pytest.raises(TargetInspectionError):
            await process._request_locked("inspect", {})

        process = _CONTAINER._ContainerRuntimeProcess(settings)
        permit = _PermitPipe()
        process._process = _ContainerProcess(
            stdin=permit, stdout=_ContainerPipe(lines=[b"response"])
        )
        process._token = b"token"
        process._receipt = "receipt"
        monkeypatch.setattr(
            _CONTAINER,
            "_response_payload",
            lambda *_: {"control": "fence", "effect": 1},
        )

        class Validator:
            """Approve the one synthetic fence before its I/O failure."""

            async def is_rooted_command_current(self, command: object) -> bool:
                """Approve only the synthetic command fixture."""
                del command
                return True

        with pytest.raises(TargetInspectionError):
            await process._request_locked(
                "commit", {}, command=object(), validator=Validator()
            )

        process._process = _ContainerProcess(
            stdin=_ContainerPipe(), stdout=_ContainerPipe()
        )
        process._token = b"token"
        process._receipt = "receipt"
        with pytest.raises(TargetInspectionError) as oversized:
            await process._request_locked("inspect", {"body": "x" * 2_000_000})
        assert oversized.value.code is TargetErrorCode.LIMIT_EXCEEDED

        process._process = _ContainerProcess(
            stdin=_ContainerPipe(drain_error=TimeoutError()),
            stdout=_ContainerPipe(),
        )
        with pytest.raises(TargetInspectionError):
            await process._request_locked("inspect", {})

        process = _CONTAINER._ContainerRuntimeProcess(settings)
        process._process = _ContainerProcess(
            stdin=_ContainerPipe(),
            stdout=_ContainerPipe(lines=[TimeoutError()]),
        )
        process._token = b"token"
        process._receipt = "receipt"
        with pytest.raises(TargetInspectionError):
            await process._request_locked("inspect", {})

        reaped = _ContainerProcess(returncode=None, wait_error=TimeoutError())
        process = _CONTAINER._ContainerRuntimeProcess(settings)
        process._process = reaped
        await process._reap()
        assert reaped.terminated and reaped.killed

        runtime = settings.create_runtime()
        with pytest.raises(TargetInspectionError):
            await runtime.resolve(ScopeSelection(ContextKind.SANDBOX))
        with pytest.raises(TargetInspectionError):
            await runtime._require_scope(object())
        with pytest.raises(TargetInspectionError):
            await runtime.worker(object())

        real_worker = _CONTAINER.ContainerPatchRuntime.worker

        async def handshake(_: object, __: object) -> object:
            """Avoid worker startup while testing endpoint issuance guards."""
            return object()

        async def inspected(_: object, __: object) -> object:
            """Return one opaque inspection receipt."""
            return object()

        async def worker(_: object, __: object) -> object:
            """Return one opaque worker receipt."""
            return object()

        monkeypatch.setattr(
            _CONTAINER.ContainerPatchRuntime, "handshake", handshake
        )
        monkeypatch.setattr(
            _CONTAINER.ContainerPatchRuntime, "inspect", inspected
        )
        monkeypatch.setattr(_CONTAINER.ContainerPatchRuntime, "worker", worker)
        runtime = settings.create_runtime()
        with pytest.raises(TargetInspectionError):
            await real_worker(runtime, object())
        scope = object()
        runtime._scope = scope
        runtime._receipt = object()
        runtime._receipt_guard = runtime._receipt
        with pytest.raises(TargetInspectionError):
            runtime._bind_sandbox_endpoint(object())
        runtime._endpoint = object()
        with pytest.raises(TargetInspectionError):
            runtime._bind_sandbox_endpoint(scope)
        inspection_target = ContainerInspectionTarget(runtime)
        mutation_target = ContainerPatchTarget(runtime)
        await inspection_target.handshake(scope)
        await inspection_target.inspect(object())
        await mutation_target.handshake(scope)
        await mutation_target.worker(scope)
        with pytest.raises(TargetInspectionError):
            ContainerPatchRuntimeBinder(
                object(), object(), object(), object(), object(), object()
            )
        with pytest.raises(TargetInspectionError):
            ContainerPatchRuntimeBinder.from_shared_store(
                object(), object(), object(), object(), object()
            )
        authority = HmacDurableApprovalAuthority.random()
        store = _BoundStore(
            InMemoryDurablePatchBackend(approval_verifier=authority)
        )
        binding = DurablePatchStoreBinding(store, store)
        binder = ContainerPatchRuntimeBinder.from_shared_store(
            settings,
            _configuration(_Clock(), authority),
            _policy(),
            PatchApprovalBinding(True),
            binding,
        )
        assert type(binder) is ContainerPatchRuntimeBinder
        with pytest.raises(TargetInspectionError):
            ContainerPatchRuntimeBinder(
                settings.create_runtime(),
                _configuration(_Clock(), authority),
                _policy(),
                PatchApprovalBinding(True),
                PatchCoordinatorBinding(True),
                PatchPersistenceBinding(True),
            )

        monkeypatch.setattr(_CONTAINER, "_docker_output", output)
        monkeypatch.setattr(
            _CONTAINER, "create_subprocess_exec", start_process
        )
        monkeypatch.setattr(
            _CONTAINER._ImplementationBundle,
            "create",
            classmethod(lambda *_args, **_kwargs: _Bundle()),
        )
        monkeypatch.setattr(
            _CONTAINER,
            "_make_container_bundle_readable",
            _mock_bundle_readable,
        )

        async def failing_output(
            command: tuple[str, ...], required: bool = True
        ) -> str | None:
            """Fail creation while allowing deterministic cleanup."""
            if command[:3] == ("docker", "volume", "inspect"):
                return None
            if command[:3] == ("docker", "volume", "create") and required:
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            return None

        monkeypatch.setattr(_CONTAINER, "_docker_output", failing_output)
        with pytest.raises(TargetInspectionError):
            await _CONTAINER._ContainerRuntimeProcess(settings).start()

        async def runtime_error_output(
            command: tuple[str, ...], required: bool = True
        ) -> str | None:
            """Raise a non-patch setup failure while allowing cleanup."""
            if command[:3] == ("docker", "volume", "inspect"):
                return None
            if command[:3] == ("docker", "volume", "create") and required:
                raise RuntimeError("docker failure")
            return None

        monkeypatch.setattr(_CONTAINER, "_docker_output", runtime_error_output)
        with pytest.raises(TargetInspectionError):
            await _CONTAINER._ContainerRuntimeProcess(settings).start()

        async def unavailable_process(
            *_: object, **__: object
        ) -> _DockerOutputProcess:
            """Raise the Docker launch failure without a host fallback."""
            raise OSError("docker unavailable")

        monkeypatch.setattr(
            _CONTAINER, "create_subprocess_exec", unavailable_process
        )
        assert await _docker_output(("docker", "bad"), False) is None
        with pytest.raises(TargetInspectionError):
            await _docker_output(("docker", "bad"))

        async def output_process(
            *_: object, **__: object
        ) -> _DockerOutputProcess:
            """Return a non-UTF8 Docker CLI output fixture."""
            return _DockerOutputProcess(b"\xff")

        monkeypatch.setattr(
            _CONTAINER, "create_subprocess_exec", output_process
        )
        with pytest.raises(TargetInspectionError):
            await _docker_output(("docker", "bad"))

        async def failing_process(
            *_: object, **__: object
        ) -> _DockerOutputProcess:
            """Return a failed Docker CLI status fixture."""
            return _DockerOutputProcess(b"", returncode=1)

        monkeypatch.setattr(
            _CONTAINER, "create_subprocess_exec", failing_process
        )
        assert await _docker_output(("docker", "bad"), False) is None
        with pytest.raises(TargetInspectionError):
            await _docker_output(("docker", "bad"))

        timed_out: list[_SlowDockerOutputProcess] = []

        async def slow_process(
            *_: object, **__: object
        ) -> _SlowDockerOutputProcess:
            """Return one Docker CLI process that requires reaping."""
            process = _SlowDockerOutputProcess(wait_error=TimeoutError())
            timed_out.append(process)
            return process

        monkeypatch.setattr(_CONTAINER, "create_subprocess_exec", slow_process)
        with pytest.raises(TargetInspectionError) as timeout:
            await _docker_output(("docker", "bad"), timeout=0.001)
        assert timeout.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
        assert timed_out[0].terminated
        assert timed_out[0].killed

        unreapable: list[_UnreapableDockerOutputProcess] = []

        async def unreapable_process(
            *_: object, **__: object
        ) -> _UnreapableDockerOutputProcess:
            """Return a child that remains unreaped after SIGKILL."""
            process = _UnreapableDockerOutputProcess()
            unreapable.append(process)
            return process

        monkeypatch.setattr(
            _CONTAINER, "create_subprocess_exec", unreapable_process
        )
        assert (
            await _docker_output(("docker", "bad"), False, timeout=0.001)
            is None
        )
        assert unreapable[0].terminated
        assert unreapable[0].killed
        with pytest.raises(TargetInspectionError) as invalid_timeout:
            await _docker_output(
                ("docker", "bad"),
                timeout=_CONTAINER._DOCKER_BUILD_TIMEOUT_SECONDS + 1.0,
            )
        assert (
            invalid_timeout.value.code
            is TargetErrorCode.CAPABILITY_UNAVAILABLE
        )

        process = _CONTAINER._ContainerRuntimeProcess(settings)

        async def invalid_canary(
            _: object, kind: str, body: object, **__: object
        ) -> Mapping[str, object]:
            """Return an invalid startup proof without reading the host."""
            del kind, body
            return {}

        monkeypatch.setattr(
            _CONTAINER._ContainerRuntimeProcess,
            "_request_locked",
            invalid_canary,
        )
        monkeypatch.setattr(_CONTAINER, "_docker_output", output)
        monkeypatch.setattr(
            _CONTAINER, "create_subprocess_exec", start_process
        )
        with pytest.raises(TargetInspectionError):
            await process.start()
        assert process._process is None

        async def missing_snapshots(
            *_: object, **__: object
        ) -> Mapping[str, object]:
            """Return an invalid worker inspection payload."""
            return {}

        monkeypatch.setattr(
            _CONTAINER._ContainerRuntimeProcess, "_request", missing_snapshots
        )
        with pytest.raises(TargetInspectionError):
            await _CONTAINER._ContainerRuntimeProcess(settings).inspect(
                (), _CONTAINER.RootWitness(FileIdentity(1, 1), "mount")
            )

    run(exercise())


def test_patch_phase_11_pristine_container_authority_is_default_denied() -> (
    None
):
    """Keep the unloaded container authority hook closed by default."""
    namespace = run_path(str(Path("src/avalan/_patch_authority.py").resolve()))
    validator = cast(
        type[_DefaultContainerAuthority], namespace["_PatchAuthorityValidator"]
    )

    assert not validator.container_endpoint_is_issued(object())


@pytest.mark.parametrize(
    "failure",
    ("worker_create", "copy", "bootstrap", "attach", "attestation"),
)
def test_patch_phase_11_owned_volume_startup_failures_cleanup_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Remove only a new authenticated volume after each startup failure."""
    seed = tmp_path / failure
    seed.mkdir()
    settings = _settings(seed, "sha256:" + "a" * 64)
    state: dict[str, object] = {"exists": False, "labels": {}, "commands": []}
    active_failure: dict[str, str | None] = {"value": failure}

    async def output(
        command: tuple[str, ...], required: bool = True
    ) -> str | None:
        """Model owned Docker volume state without exposing a host path."""
        del required
        commands = cast(list[tuple[str, ...]], state["commands"])
        commands.append(command)
        if command[:3] == ("docker", "volume", "inspect"):
            if not state["exists"]:
                return None
            return dumps([{"Labels": state["labels"]}])
        if command[:3] == ("docker", "volume", "create"):
            labels = {
                item.split("=", 1)[0]: item.split("=", 1)[1]
                for item in (command[4], command[6])
            }
            state["labels"] = labels
            state["exists"] = True
            return command[-1] + "\n"
        if command[:2] == ("docker", "ps"):
            return ""
        if (
            command[:2] == ("docker", "create")
            and active_failure["value"] == "worker_create"
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        if (
            command[:2] == ("docker", "cp")
            and active_failure["value"] == "copy"
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        if (
            command[:2] == ("docker", "run")
            and active_failure["value"] == "bootstrap"
            and "--name" not in command
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        if command[:3] == ("docker", "volume", "rm"):
            state["exists"] = False
            return command[-1] + "\n"
        return "container-id\n"

    async def start_process(
        *command: object, **_: object
    ) -> _ContainerProcess:
        """Fail only the configured no-network worker attachment stage."""
        if (
            tuple(command[:2]) == ("docker", "start")
            and active_failure["value"] == "attach"
        ):
            raise OSError("test attachment failure")
        return _ContainerProcess(
            stdin=_ContainerPipe(), stdout=_ContainerPipe()
        )

    async def worker_request(
        _: object, kind: str, __: object, **___: object
    ) -> Mapping[str, object]:
        """Return only startup attestation facts for a successful retry."""
        if active_failure["value"] == "attestation" and kind == "canary":
            return {}
        if kind == "canary":
            return {
                "pid": 1,
                "outside_read_denied": True,
                "metadata_probe": "sealed",
            }
        assert kind == "witness"
        return {
            "root": {
                "device": 1,
                "inode": 1,
                "mount": "docker-mount",
                "filesystem": "docker-volume",
            }
        }

    async def exercise() -> None:
        """Fail one stage, remove the new volume, and retry from no state."""
        monkeypatch.setattr(_CONTAINER, "_docker_output", output)
        monkeypatch.setattr(
            _CONTAINER, "create_subprocess_exec", start_process
        )
        monkeypatch.setattr(
            _CONTAINER._ImplementationBundle,
            "create",
            classmethod(lambda *_args, **_kwargs: _Bundle()),
        )
        monkeypatch.setattr(
            _CONTAINER,
            "_make_container_bundle_readable",
            _mock_bundle_readable,
        )
        monkeypatch.setattr(
            _CONTAINER._ContainerRuntimeProcess,
            "_request_locked",
            worker_request,
        )
        with pytest.raises(TargetInspectionError):
            await _CONTAINER._ContainerRuntimeProcess(settings).start()
        assert not state["exists"]
        commands = cast(list[tuple[str, ...]], state["commands"])
        assert any(
            command[:3] == ("docker", "volume", "rm") for command in commands
        )

        active_failure["value"] = None
        retry = _CONTAINER._ContainerRuntimeProcess(settings)
        await retry.start()
        assert state["exists"]
        await retry.dispose()
        assert not state["exists"]

    run(exercise())


def test_patch_phase_11_refuses_foreign_or_partial_owned_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject labels not authenticated by this authority before attachment."""
    seed = tmp_path / "foreign"
    seed.mkdir()
    settings = _settings(seed, "sha256:" + "a" * 64)
    commands: list[tuple[str, ...]] = []
    volume = _CONTAINER._docker_name(
        "avalan_patch_", settings.context.identity.persistent_lease_id
    )
    resource_digest = _CONTAINER._persistent_resource_digest(settings)
    owner_receipt = _CONTAINER._volume_owner_receipt(
        settings.persistent_lease_authority, resource_digest
    )

    async def output(
        command: tuple[str, ...], required: bool = True
    ) -> str | None:
        """Expose an existing foreign or partial volume only to inspection."""
        del required
        commands.append(command)
        if command[:2] == ("docker", "run"):
            return "guard\n"
        if command[:3] == ("docker", "volume", "inspect"):
            return dumps([{"Labels": {"avalan.patch.resource": "foreign"}}])
        if command[:2] == ("docker", "inspect"):
            return dumps(
                [
                    {
                        "Config": {
                            "Labels": {
                                "avalan.patch.resource": resource_digest,
                                "avalan.patch.owner": owner_receipt,
                            }
                        }
                    }
                ]
            )
        if command[:3] == ("docker", "rm", "--force"):
            return "guard\n"
        raise AssertionError("foreign volume was attached or removed")

    async def exercise() -> None:
        """Keep the foreign volume untouched and start no worker process."""
        monkeypatch.setattr(_CONTAINER, "_docker_output", output)
        process = _CONTAINER._ContainerRuntimeProcess(settings)
        with pytest.raises(TargetInspectionError) as rejected:
            await process.start()
        assert rejected.value.code is TargetErrorCode.WITNESS_STALE
        assert process.volume_name is None
        assert commands[0][:2] == ("docker", "run")
        assert commands[1] == ("docker", "volume", "inspect", volume)
        assert not any(
            command[:3] == ("docker", "volume", "rm") for command in commands
        )

    run(exercise())


def test_patch_phase_11_owned_volume_dispose_never_removes_live_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wait for every authenticated attachment before deleting one volume."""
    seed = tmp_path / "live"
    seed.mkdir()
    settings = _settings(seed, "sha256:" + "a" * 64)
    state: dict[str, object] = {"exists": False, "labels": {}, "removed": 0}

    async def output(
        command: tuple[str, ...], required: bool = True
    ) -> str | None:
        """Keep one volume alive until the final owned service disposes."""
        del required
        if command[:3] == ("docker", "volume", "inspect"):
            if not state["exists"]:
                return None
            return dumps([{"Labels": state["labels"]}])
        if command[:3] == ("docker", "volume", "create"):
            state["labels"] = {
                item.split("=", 1)[0]: item.split("=", 1)[1]
                for item in (command[4], command[6])
            }
            state["exists"] = True
            return command[-1] + "\n"
        if command[:2] == ("docker", "ps"):
            return ""
        if command[:3] == ("docker", "volume", "rm"):
            state["removed"] = cast(int, state["removed"]) + 1
            state["exists"] = False
            return command[-1] + "\n"
        return "container-id\n"

    async def start_process(*_: object, **__: object) -> _ContainerProcess:
        """Return one fake attached worker process for each trusted runtime."""
        return _ContainerProcess(
            stdin=_ContainerPipe(), stdout=_ContainerPipe()
        )

    async def worker_request(
        _: object, kind: str, __: object, **___: object
    ) -> Mapping[str, object]:
        """Return the minimal attestation for each owned attachment."""
        if kind == "canary":
            return {
                "pid": 1,
                "outside_read_denied": True,
                "metadata_probe": "sealed",
            }
        return {
            "root": {
                "device": 1,
                "inode": 1,
                "mount": "docker-mount",
                "filesystem": "docker-volume",
            }
        }

    async def exercise() -> None:
        """Dispose one service while its peer still owns the attachment."""
        monkeypatch.setattr(_CONTAINER, "_docker_output", output)
        monkeypatch.setattr(
            _CONTAINER, "create_subprocess_exec", start_process
        )
        monkeypatch.setattr(
            _CONTAINER._ImplementationBundle,
            "create",
            classmethod(lambda *_args, **_kwargs: _Bundle()),
        )
        monkeypatch.setattr(
            _CONTAINER,
            "_make_container_bundle_readable",
            _mock_bundle_readable,
        )
        monkeypatch.setattr(
            _CONTAINER._ContainerRuntimeProcess,
            "_request_locked",
            worker_request,
        )
        first = _CONTAINER._ContainerRuntimeProcess(settings)
        second = _CONTAINER._ContainerRuntimeProcess(settings)
        await first.start()
        await second.start()
        await first.dispose()
        assert state["exists"]
        assert state["removed"] == 0
        await second.dispose()
        assert not state["exists"]
        assert state["removed"] == 1

    run(exercise())


def test_patch_phase_11_owned_volume_defensive_cleanup_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject malformed, conflicting, and no-longer-owned volume state."""
    seed = tmp_path / "defensive-volume"
    seed.mkdir()
    process = _CONTAINER._ContainerRuntimeProcess(
        _settings(seed, "sha256:" + "a" * 64)
    )
    resource_digest = "a" * 64
    owner_receipt = "b" * 64
    volume = "phase11-defensive-volume"
    inspection: dict[str, str | None] = {"value": None}
    removal: dict[str, str | None] = {"value": None}

    async def output(
        command: tuple[str, ...], required: bool = True
    ) -> str | None:
        """Return only the scripted inspection and remove outcomes."""
        del required
        if command[:3] == ("docker", "volume", "inspect"):
            return inspection["value"]
        if command[:3] == ("docker", "volume", "rm"):
            return removal["value"]
        if command[:2] == ("docker", "ps"):
            return ""
        if command[:2] == ("docker", "run"):
            return "guard\n"
        if command[:2] == ("docker", "inspect"):
            return dumps(
                [
                    {
                        "Labels": {
                            "avalan.patch.resource": resource_digest,
                            "avalan.patch.owner": owner_receipt,
                        }
                    }
                ]
            )
        if command[:3] == ("docker", "rm", "--force"):
            return command[-1] + "\n"
        raise AssertionError("unexpected Docker command")

    async def exercise() -> None:
        """Exercise every fail-closed owned-volume cleanup branch."""
        monkeypatch.setattr(_CONTAINER, "_docker_output", output)
        assert not _CONTAINER._owned_volume_matches(
            "[]", resource_digest, owner_receipt
        )
        try:
            await process._claim_volume(
                volume, resource_digest, owner_receipt, created=True
            )
            duplicate = _CONTAINER._ContainerRuntimeProcess(process.settings)
            with pytest.raises(TargetInspectionError) as duplicate_error:
                await duplicate._claim_volume(
                    volume, resource_digest, owner_receipt, created=True
                )
            assert duplicate_error.value.code is TargetErrorCode.WITNESS_STALE
            mismatch = _CONTAINER._ContainerRuntimeProcess(process.settings)
            with pytest.raises(TargetInspectionError) as mismatch_error:
                await mismatch._claim_volume(
                    volume, "c" * 64, "d" * 64, created=False
                )
            assert mismatch_error.value.code is TargetErrorCode.WITNESS_STALE
            async with _CONTAINER._OWNED_VOLUMES_LOCK:
                claim = _CONTAINER._OWNED_VOLUMES[volume]
                claim.active_attachments = 0
            await process._release_volume_attachment()
            assert not process._volume_attached
            assert not process._volume_owned

            idle = _CONTAINER._ContainerRuntimeProcess(process.settings)
            await idle._claim_volume(
                volume, resource_digest, owner_receipt, created=False
            )
            await idle._release_volume_attachment()
            idle._volume_owned = False
            await idle._dispose_owned_volume()
            idle._volume_owned = True

            inspection["value"] = None
            await idle._dispose_owned_volume()
            inspection["value"] = dumps(
                [
                    {
                        "Labels": {
                            "avalan.patch.resource": resource_digest,
                            "avalan.patch.owner": owner_receipt,
                        }
                    }
                ]
            )
            removal["value"] = None
            await idle._dispose_owned_volume()
        finally:
            async with _CONTAINER._OWNED_VOLUMES_LOCK:
                _CONTAINER._OWNED_VOLUMES.pop(volume, None)

    run(exercise())


def test_patch_phase_11_volume_recovery_defenses_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject every unproven volume reclaim and cleanup boundary."""
    seed = tmp_path / "volume-recovery-defenses"
    seed.mkdir()
    settings = _settings(seed, "sha256:" + "a" * 64)
    process = _CONTAINER._ContainerRuntimeProcess(settings)
    resource_digest = "a" * 64
    owner_receipt = "b" * 64
    volume = "phase11-volume-recovery-defenses"

    with pytest.raises(TargetInspectionError) as invalid_authority:
        ContainerPersistentLeaseAuthority.from_bytes(b"short")
    assert (
        invalid_authority.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )

    def inspected(labels: Mapping[str, str], *, stopped: bool = False) -> str:
        """Return one labelled Docker inspection payload for the test state."""
        row: dict[str, object] = {"Labels": labels}
        if stopped:
            row["State"] = {"Running": False}
        return dumps([row])

    async def exercise() -> None:
        """Drive the exact stale and authenticated reclaim branches."""

        async def live_attachment(
            command: tuple[str, ...], required: bool = True
        ) -> str | None:
            """Prove a live attachment without allowing a guard creation."""
            del required
            if command[:2] == ("docker", "ps"):
                return "live-container\n"
            raise AssertionError("live volume unexpectedly acquired a guard")

        monkeypatch.setattr(_CONTAINER, "_docker_output", live_attachment)
        with pytest.raises(TargetInspectionError) as live_rejected:
            await process._claim_volume(
                volume, resource_digest, owner_receipt, created=False
            )
        assert live_rejected.value.code is TargetErrorCode.WITNESS_STALE

        async def unknown_guard(
            command: tuple[str, ...], required: bool = True
        ) -> str | None:
            """Make an existing guard absent and therefore unreclaimable."""
            del required
            if command[:2] == ("docker", "run"):
                return None
            if command[:2] == ("docker", "inspect"):
                return None
            raise AssertionError("unknown guard attempted an unsafe recovery")

        monkeypatch.setattr(_CONTAINER, "_docker_output", unknown_guard)
        with pytest.raises(TargetInspectionError) as unknown_rejected:
            await process._acquire_volume_guard(
                volume, resource_digest, owner_receipt
            )
        assert unknown_rejected.value.code is TargetErrorCode.WITNESS_STALE

        async def foreign_guard(
            command: tuple[str, ...], required: bool = True
        ) -> str | None:
            """Keep a mismatched guard from being reclaimed or removed."""
            del required
            if command[:2] == ("docker", "run"):
                return None
            if command[:2] == ("docker", "inspect"):
                return inspected({"avalan.patch.resource": "foreign"})
            raise AssertionError("foreign guard attempted an unsafe recovery")

        monkeypatch.setattr(_CONTAINER, "_docker_output", foreign_guard)
        with pytest.raises(TargetInspectionError) as foreign_rejected:
            await process._acquire_volume_guard(
                volume, resource_digest, owner_receipt
            )
        assert foreign_rejected.value.code is TargetErrorCode.WITNESS_STALE

        labels = {
            "avalan.patch.resource": resource_digest,
            "avalan.patch.owner": owner_receipt,
        }

        async def failed_remove(
            command: tuple[str, ...], required: bool = True
        ) -> str | None:
            """Reject a proven stale guard when removal itself is unproven."""
            del required
            if command[:2] == ("docker", "run"):
                return None
            if command[:2] == ("docker", "inspect"):
                return inspected(labels, stopped=True)
            if command[:2] == ("docker", "ps"):
                return ""
            if command[:3] == ("docker", "rm", "--force"):
                return None
            raise AssertionError("unexpected stale-guard command")

        monkeypatch.setattr(_CONTAINER, "_docker_output", failed_remove)
        with pytest.raises(TargetInspectionError) as remove_rejected:
            await process._acquire_volume_guard(
                volume, resource_digest, owner_receipt
            )
        assert remove_rejected.value.code is TargetErrorCode.WITNESS_STALE

        retried_runs = iter((None, None))

        async def failed_retry(
            command: tuple[str, ...], required: bool = True
        ) -> str | None:
            """Reject a reclaimed guard when the second create is absent."""
            del required
            if command[:2] == ("docker", "run"):
                return next(retried_runs)
            if command[:2] == ("docker", "inspect"):
                return inspected(labels, stopped=True)
            if command[:2] == ("docker", "ps"):
                return ""
            if command[:3] == ("docker", "rm", "--force"):
                return "guard\n"
            raise AssertionError("unexpected retry-guard command")

        monkeypatch.setattr(_CONTAINER, "_docker_output", failed_retry)
        with pytest.raises(TargetInspectionError) as retry_rejected:
            await process._acquire_volume_guard(
                volume, resource_digest, owner_receipt
            )
        assert retry_rejected.value.code is TargetErrorCode.WITNESS_STALE

        successful_runs = iter((None, "guard\n"))

        async def successful_retry(
            command: tuple[str, ...], required: bool = True
        ) -> str | None:
            """Permit only an authenticated, idle stale-guard reclaim."""
            del required
            if command[:2] == ("docker", "run"):
                return next(successful_runs)
            if command[:2] == ("docker", "inspect"):
                return inspected(labels, stopped=True)
            if command[:2] == ("docker", "ps"):
                return ""
            if command[:3] == ("docker", "rm", "--force"):
                return "guard\n"
            raise AssertionError("unexpected successful-retry command")

        monkeypatch.setattr(_CONTAINER, "_docker_output", successful_retry)
        assert await process._acquire_volume_guard(
            volume, resource_digest, owner_receipt
        ) == _CONTAINER._docker_name(
            "avalan_patch_lease_",
            settings.context.identity.persistent_lease_id,
        )

        await process._cleanup_new_volume(
            volume, resource_digest, owner_receipt
        )
        process._volume_owned = True
        process._volume_name = volume
        process._volume_resource_digest = resource_digest
        process._volume_owner_receipt = owner_receipt

        async def authenticated_cleanup(
            command: tuple[str, ...], required: bool = True
        ) -> str | None:
            """Authenticate cleanup while retaining one exact guard."""
            del required
            if command[:2] == ("docker", "run"):
                return "guard\n"
            if command[:2] == ("docker", "inspect"):
                return inspected(labels)
            if command[:2] == ("docker", "ps"):
                return ""
            if command[:3] == ("docker", "rm", "--force"):
                return "guard\n"
            if command[:3] == ("docker", "volume", "inspect"):
                return inspected(labels)
            if command[:3] == ("docker", "volume", "rm"):
                return volume + "\n"
            raise AssertionError("unexpected cleanup command")

        monkeypatch.setattr(
            _CONTAINER, "_docker_output", authenticated_cleanup
        )
        async with _CONTAINER._OWNED_VOLUMES_LOCK:
            _CONTAINER._OWNED_VOLUMES[volume] = _CONTAINER._OwnedVolumeClaim(
                resource_digest, owner_receipt
            )
        await process._cleanup_new_volume(
            volume, resource_digest, owner_receipt
        )
        async with _CONTAINER._OWNED_VOLUMES_LOCK:
            assert volume not in _CONTAINER._OWNED_VOLUMES

        process._volume_owned = True
        process._volume_name = volume
        process._volume_resource_digest = resource_digest
        process._volume_owner_receipt = owner_receipt

        async def live_dispose(
            command: tuple[str, ...], required: bool = True
        ) -> str | None:
            """Keep a live attachment from volume inspection after guarding."""
            del required
            if command[:2] == ("docker", "run"):
                return "guard\n"
            if command[:2] == ("docker", "inspect"):
                return inspected(labels)
            if command[:2] == ("docker", "ps"):
                return "live-container\n"
            if command[:3] == ("docker", "rm", "--force"):
                return "guard\n"
            raise AssertionError(
                "live volume reached an unsafe dispose command"
            )

        monkeypatch.setattr(_CONTAINER, "_docker_output", live_dispose)
        await process._dispose_owned_volume()

    try:
        run(exercise())
    finally:
        _CONTAINER._OWNED_VOLUMES.pop(volume, None)


def test_patch_phase_11_local_claim_guard_invariants_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject mismatched local claims and preclaimed durable guards."""
    seed = tmp_path / "local-claim-guard"
    seed.mkdir()
    settings = _settings(seed, "sha256:" + "a" * 64)
    process = _CONTAINER._ContainerRuntimeProcess(settings)
    volume = "phase11-local-claim-guard"
    resource_digest = "a" * 64
    owner_receipt = "b" * 64

    async def no_live_attachment(
        command: tuple[str, ...], required: bool = True
    ) -> str | None:
        """Prove guard mismatch fails before any external recovery action."""
        del required
        assert command[:2] == ("docker", "ps")
        return ""

    async def exercise() -> None:
        """Exercise local exact, mismatched, and preclaimed-guard branches."""
        assert not await process._has_local_volume_claim(
            volume, resource_digest, owner_receipt
        )
        async with _CONTAINER._OWNED_VOLUMES_LOCK:
            _CONTAINER._OWNED_VOLUMES[volume] = _CONTAINER._OwnedVolumeClaim(
                resource_digest, owner_receipt
            )
        assert await process._has_local_volume_claim(
            volume, resource_digest, owner_receipt
        )
        with pytest.raises(TargetInspectionError) as mismatch:
            await process._has_local_volume_claim(
                volume, "c" * 64, owner_receipt
            )
        assert mismatch.value.code is TargetErrorCode.WITNESS_STALE
        async with _CONTAINER._OWNED_VOLUMES_LOCK:
            _CONTAINER._OWNED_VOLUMES.pop(volume, None)
        process._volume_guard_name = "expected-guard"
        monkeypatch.setattr(_CONTAINER, "_docker_output", no_live_attachment)
        with pytest.raises(TargetInspectionError) as guard_mismatch:
            await process._claim_volume(
                volume,
                resource_digest,
                owner_receipt,
                created=False,
                guard_name="other-guard",
            )
        assert guard_mismatch.value.code is TargetErrorCode.WITNESS_STALE

    try:
        run(exercise())
    finally:
        _CONTAINER._OWNED_VOLUMES.pop(volume, None)


def test_patch_phase_11_guard_recovery_rejects_inconsistent_inspection_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when a second guard inspection decode is inconsistent."""
    resource_digest = "a" * 64
    owner_receipt = "b" * 64
    owned_row = [
        {
            "Labels": {
                "avalan.patch.resource": resource_digest,
                "avalan.patch.owner": owner_receipt,
            }
        }
    ]
    calls = 0

    def malformed_second_decode(_: str) -> object:
        """Return ownership once and reject a changed second inspection."""
        nonlocal calls
        calls += 1
        if calls == 1:
            return owned_row
        raise JSONDecodeError("changed inspection", "{", 0)

    monkeypatch.setattr(_CONTAINER, "loads", malformed_second_decode)
    assert not _CONTAINER._owned_volume_guard_is_stopped(
        "inspection", resource_digest, owner_receipt
    )

    calls = 0

    def malformed_second_shape(_: str) -> object:
        """Return ownership once and a malformed second inspection shape."""
        nonlocal calls
        calls += 1
        return owned_row if calls == 1 else []

    monkeypatch.setattr(_CONTAINER, "loads", malformed_second_shape)
    assert not _CONTAINER._owned_volume_guard_is_stopped(
        "inspection", resource_digest, owner_receipt
    )


def test_patch_phase_11_rethrows_unattached_incomplete_journal() -> None:
    """Preserve a terminal journal fault when no pending branch exists."""
    phase_eight = run_path("tests/patch/phase_8_durable_continuation_test.py")
    claimed = phase_eight["_claimed"]
    correlation = phase_eight["_correlation"]
    report = phase_eight["_report"]
    result = phase_eight["_result"]
    assert callable(claimed)
    assert callable(correlation)
    assert callable(report)
    assert callable(result)

    class RejectingStore(InMemoryDurablePatchStore):
        """Reject only terminal settlement after the durable journal append."""

        async def settle(
            self,
            lease: DurableCommitLease,
            expected: DurableJournalCursor,
            terminal_result: PatchResult,
            correlation_id: PatchObserverCorrelationId,
            now: ExpiryTick,
        ) -> DurableTerminalRecord:
            """Raise the precise retained journal incompleteness signal."""
            del lease, expected, terminal_result, correlation_id, now
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_INCOMPLETE)

    async def exercise() -> None:
        """Attempt settlement without a pending continuation.

        Preserve the durable error rather than manufacture pending state.
        """
        backend, _, identity, reservation, plan, lease = await claimed("a")
        reconciler = phase_eight["DurablePatchReconciler"](
            RejectingStore(backend)
        )
        with pytest.raises(DurableStoreError) as rejected:
            await reconciler.reconcile(
                phase_eight["DurableRequestAccess"](
                    reservation.request_id, identity
                ),
                lease,
                report(
                    plan,
                    (phase_eight["CommitStepState"].COMMITTED,),
                ),
                result(
                    reservation.request_id,
                    plan,
                    phase_eight["MutationState"].COMMITTED,
                ),
                correlation("a"),
                phase_eight["ExpiryTick"](20),
            )
        assert rejected.value.code is DurableStoreErrorCode.JOURNAL_INCOMPLETE

    run(exercise())


def test_patch_phase_11_forged_settlement_inspection_is_stale() -> None:
    """Reject an unissued handle without exposing durable latest state."""

    async def exercise() -> None:
        """Exercise the exact forged and empty inspection branches."""
        service = object.__new__(SandboxPatchSdkService)
        service._latest = None
        with pytest.raises(TargetInspectionError) as unavailable:
            await service._inspect_latest()
        assert unavailable.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        future = _SANDBOX._SandboxSettlementPort(service).inspect(
            PatchInvocationHandle(object())
        )
        with pytest.raises(TargetInspectionError) as stale:
            await future
        assert stale.value.code is TargetErrorCode.WITNESS_STALE

    run(exercise())


def test_patch_phase_11_rethrows_nonstale_reconciliation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not relabel a non-stale reconciliation fault as forged identity."""

    async def invoke(*arguments: object) -> PatchResult:
        """Fail after the host issues the synthetic invocation handle."""
        del arguments
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)

    async def reconcile(
        handle: PatchInvocationHandle, error: Exception
    ) -> PatchResult:
        """Return a distinct durable failure from the reconciliation port."""
        del handle, error
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)

    async def exercise() -> None:
        """Keep the original non-stale reconciliation error observable."""
        snapshot = PatchCapabilitySnapshot(
            edit_available=True, apply_available=False
        )
        host = cast(
            PatchSdkHost,
            SimpleNamespace(
                _snapshot=snapshot,
                _is_active=lambda: True,
                _capability=object(),
                _service=SimpleNamespace(invoke=invoke),
                _reconcile_after_dispatch=reconcile,
            ),
        )
        monkeypatch.setattr(
            _TOOLSET,
            "_bound_issue_invocation",
            lambda *_: PatchInvocationHandle(object()),
        )
        with pytest.raises(TargetInspectionError) as rejected:
            await PatchSdkHost._invoke_raw_with_identity(
                host,
                OperationType.EDIT,
                b'{"path":"note.txt","edits":['
                b'{"old_text":"old","new_text":"new"}]}',
                PatchRequestId("request_" + "a" * 16),
                PatchObserverCorrelationId("correlation_" + "a" * 16),
            )
        assert rejected.value.code is TargetErrorCode.WORKER_UNAVAILABLE

    run(exercise())


def test_patch_phase_11_container_bundle_is_only_readable_to_the_child(
    tmp_path: Path,
) -> None:
    """Expose an immutable copied bundle without links or write authority."""
    bundle = tmp_path / "bundle"
    package = bundle / "avalan" / "patch"
    package.mkdir(parents=True)
    worker = package / "sandbox_worker.py"
    worker.write_text("worker", encoding="utf-8")

    _CONTAINER._make_container_bundle_readable(bundle)

    assert bundle.stat().st_mode & 0o777 == 0o555
    assert package.stat().st_mode & 0o777 == 0o555
    assert worker.stat().st_mode & 0o777 == 0o444

    not_bundle = tmp_path / "not-bundle"
    not_bundle.write_text("not a bundle", encoding="utf-8")
    with pytest.raises(TargetInspectionError) as not_directory:
        _CONTAINER._make_container_bundle_readable(not_bundle)
    assert not_directory.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE

    linked = tmp_path / "linked"
    linked.mkdir()
    target = linked / "target.py"
    target.write_text("target", encoding="utf-8")
    (linked / "symlink.py").symlink_to(target)
    with pytest.raises(TargetInspectionError) as symlink:
        _CONTAINER._make_container_bundle_readable(linked)
    assert symlink.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE

    hardlinked = tmp_path / "hardlinked"
    hardlinked.mkdir()
    source = hardlinked / "source.py"
    source.write_text("source", encoding="utf-8")
    (hardlinked / "hardlink.py").hardlink_to(source)
    with pytest.raises(TargetInspectionError) as hardlink:
        _CONTAINER._make_container_bundle_readable(hardlinked)
    assert hardlink.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE


def _test_image_target_architecture(value: str) -> str:
    """Return the Docker architecture matching one current Linux runtime."""
    architectures = {
        "aarch64": "arm64",
        "amd64": "amd64",
        "arm64": "arm64",
        "x86_64": "amd64",
    }
    target = architectures.get(value.lower())
    assert target is not None, "unsupported Docker test architecture"
    return target


async def _test_image() -> str:
    """Build and return the sealed Linux worker image for one E2E."""
    base = await _CONTAINER._docker_output(
        (
            "docker",
            "image",
            "inspect",
            _TEST_WORKER_BASE_IMAGE,
        )
    )
    assert base is not None
    target_architecture = _test_image_target_architecture(machine())
    image = await _CONTAINER._docker_output(
        (
            "docker",
            "build",
            "--quiet",
            "--network=none",
            "--pull=false",
            "--build-arg",
            "TARGETARCH=" + target_architecture,
            "--file",
            str(_FIXTURES / "container_worker.Dockerfile"),
            str(_FIXTURES),
        ),
        timeout=_CONTAINER._DOCKER_BUILD_TIMEOUT_SECONDS,
    )
    assert image is not None
    return image.strip()


@pytest.mark.parametrize(
    ("runtime_architecture", "target_architecture"),
    (("aarch64", "arm64"), ("amd64", "amd64"), ("x86_64", "amd64")),
)
def test_patch_phase_11_image_target_architecture_is_closed(
    runtime_architecture: str, target_architecture: str
) -> None:
    """Map only supported Docker runtime architectures to committed wheels."""
    assert _test_image_target_architecture(runtime_architecture) == (
        target_architecture
    )


@pytest.mark.parametrize("runtime_architecture", ("", "ppc64le"))
def test_patch_phase_11_image_target_architecture_rejects_unknown_values(
    runtime_architecture: str,
) -> None:
    """Fail before the image build when no wheel architecture is committed."""
    with pytest.raises(AssertionError, match="unsupported Docker"):
        _test_image_target_architecture(runtime_architecture)


@pytest.mark.parametrize(
    ("target_architecture", "platform_tag"),
    (("amd64", "manylinux2014_x86_64"), ("arm64", "manylinux2014_aarch64")),
)
def test_patch_phase_11_wheelhouse_installs_offline_per_architecture(
    target_architecture: str, platform_tag: str, tmp_path: Path
) -> None:
    """Install each committed Linux wheel set without an index or fallback."""
    requirements = _TEST_WORKER_WHEELHOUSE / "requirements.txt"
    dockerfile = _FIXTURES / "container_worker.Dockerfile"
    expected = set(_TEST_WORKER_WHEEL_SHA256) | {"requirements.txt"}
    assert {
        path.name for path in _TEST_WORKER_WHEELHOUSE.iterdir()
    } == expected
    for name, expected_hash in _TEST_WORKER_WHEEL_SHA256.items():
        assert sha256(
            (_TEST_WORKER_WHEELHOUSE / name).read_bytes()
        ).hexdigest() == (expected_hash)
        assert expected_hash in requirements.read_text(encoding="utf-8")
        assert (
            "COPY container_wheels/" + name + " /wheelhouse/"
            in dockerfile.read_text(encoding="utf-8")
        )
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    assert dockerfile_text.startswith("FROM " + _TEST_WORKER_BASE_IMAGE)
    assert 'case "$TARGETARCH" in' in dockerfile_text
    assert "--no-deps --no-index" in dockerfile_text
    assert "--only-binary=:all:" in dockerfile_text
    assert "--require-hashes" in dockerfile_text
    installed = tmp_path / target_architecture
    completed = run_process(
        (
            executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-compile",
            "--no-deps",
            "--no-index",
            "--find-links",
            str(_TEST_WORKER_WHEELHOUSE),
            "--only-binary=:all:",
            "--require-hashes",
            "--platform",
            platform_tag,
            "--implementation",
            "cp",
            "--python-version",
            "3.11",
            "--abi",
            "cp311",
            "--target",
            str(installed),
            "--requirement",
            str(requirements),
        ),
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (installed / "cffi").is_dir()
    assert (installed / "cryptography").is_dir()
    assert (installed / "pycparser").is_dir()


def test_patch_phase_11_worker_wheelhouse_rejects_missing_or_wrong_inputs(
    tmp_path: Path,
) -> None:
    """Reject absent and altered wheel inputs without permitting an index."""
    requirements = _TEST_WORKER_WHEELHOUSE / "requirements.txt"
    missing = run_process(
        (
            executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(tmp_path / "missing"),
            "--only-binary=:all:",
            "--require-hashes",
            "--no-deps",
            "--platform",
            "manylinux2014_aarch64",
            "--implementation",
            "cp",
            "--python-version",
            "3.11",
            "--abi",
            "cp311",
            "--target",
            str(tmp_path / "missing-target"),
            "--requirement",
            str(requirements),
        ),
        capture_output=True,
        check=False,
        text=True,
    )
    assert missing.returncode != 0
    altered = tmp_path / "altered-requirements.txt"
    altered.write_text(
        requirements.read_text(encoding="utf-8").replace(
            next(iter(_TEST_WORKER_WHEEL_SHA256.values())), "0" * 64
        ),
        encoding="utf-8",
    )
    wrong_hash = run_process(
        (
            executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(_TEST_WORKER_WHEELHOUSE),
            "--only-binary=:all:",
            "--require-hashes",
            "--no-deps",
            "--platform",
            "manylinux2014_aarch64",
            "--implementation",
            "cp",
            "--python-version",
            "3.11",
            "--abi",
            "cp311",
            "--target",
            str(tmp_path / "wrong-hash-target"),
            "--requirement",
            str(altered),
        ),
        capture_output=True,
        check=False,
        text=True,
    )
    assert wrong_hash.returncode != 0


def test_patch_phase_11_image_build_uses_a_bounded_cold_start_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give the immutable image build its distinct bounded deadline."""
    observed: list[tuple[tuple[str, ...], bool, float]] = []

    async def output(
        command: tuple[str, ...],
        required: bool = True,
        timeout: float = _CONTAINER._DOCKER_COMMAND_TIMEOUT_SECONDS,
    ) -> str | None:
        """Record the exact bounded Docker build invocation."""
        observed.append((command, required, timeout))
        return "sha256:" + "a" * 64 + "\n"

    monkeypatch.setattr(_CONTAINER, "_docker_output", output)

    assert run(_test_image()) == "sha256:" + "a" * 64
    base_command, required, timeout = observed[0]
    assert base_command == (
        "docker",
        "image",
        "inspect",
        _TEST_WORKER_BASE_IMAGE,
    )
    assert required
    assert timeout == _CONTAINER._DOCKER_COMMAND_TIMEOUT_SECONDS
    command, required, timeout = observed[1]
    assert command[:3] == ("docker", "build", "--quiet")
    assert "--network=none" in command
    assert "--pull=false" in command
    assert "--build-arg" in command
    assert any(value.startswith("TARGETARCH=") for value in command)
    assert required
    assert timeout == _CONTAINER._DOCKER_BUILD_TIMEOUT_SECONDS


def _configuration(
    clock: _Clock, authority: HmacDurableApprovalAuthority
) -> SandboxPatchServiceConfiguration:
    """Return production service wiring over the test's durable authority."""
    approvals = ApprovalService(_Broker(), clock, RuntimeGrantStore())
    return SandboxPatchServiceConfiguration(
        _subject(),
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, authority),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )


def _binder(
    settings: ContainerPatchRuntimeSettings,
    configuration: SandboxPatchServiceConfiguration,
    store: InMemoryDurablePatchStore,
) -> ContainerPatchRuntimeBinder:
    """Bind one Docker lease to the selected durable patch service."""
    return ContainerPatchRuntimeBinder(
        settings.create_runtime(),
        configuration,
        _policy(),
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True, store),
        PatchPersistenceBinding(True, store),
    )


def _multi_file_apply(before: str, after: str) -> Mapping[str, object]:
    """Return one real two-file apply transaction with exact preconditions."""
    return {
        "patch": "\n".join(
            (
                "*** Begin Patch v1",
                "*** Update File: note.txt",
                "@@",
                "-" + before,
                "+" + after,
                "*** Update File: second.txt",
                "@@",
                "-second-before",
                "+second-after",
                "*** End Patch",
            )
        )
    }


def _ordinary_read_command(
    image: str, volume: str, path: str
) -> tuple[str, ...]:
    """Return an ordinary read-only container command without host mounts."""
    return (
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--mount",
        "type=volume,src=" + volume + ",dst=/workspace,readonly",
        image,
        "python3",
        "-I",
        "-c",
        "from pathlib import Path;print(Path('/workspace/"
        + path
        + "').read_text(),end='')",
    )


def _volume_command(
    image: str, volume: str, script: str, *, read_only: bool = False
) -> tuple[str, ...]:
    """Return one test-controlled volume probe with no host bind mount."""
    mount = "type=volume,src=" + volume + ",dst=/workspace"
    if read_only:
        mount += ",readonly"
    return (
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=65536",
        "--mount",
        mount,
        image,
        "python3",
        "-I",
        "-c",
        script,
    )


async def _volume_bytes(image: str, volume: str, path: str) -> bytes:
    """Read one persistent-volume file through a no-network observer."""
    value = await _docker_output(
        _volume_command(
            image,
            volume,
            "from base64 import b64encode;"
            "from pathlib import Path;"
            "print(b64encode(Path('/workspace/"
            + path
            + "').read_bytes()).decode(),end='')",
            read_only=True,
        )
    )
    assert value is not None
    return b64decode(value)


def _replacement_settings(
    settings: ContainerPatchRuntimeSettings, replacement: str
) -> ContainerPatchRuntimeSettings:
    """Return a distinct plan-bound identity for stale-fence coverage."""
    identity = settings.context.identity
    match replacement:
        case "execution_plan":
            return replace(
                settings,
                execution_plan_fingerprint=SandboxExecutionPlanFingerprint(
                    "replacement-plan-v1"
                ),
            )
        case "persistent_lease":
            identity = replace(
                identity, persistent_lease_id="persistent-lease-replaced"
            )
        case "workspace":
            identity = replace(
                identity,
                workspace_id=PatchWorkspaceId("workspace_" + "a" * 16),
            )
        case "context":
            identity = replace(
                identity,
                context_id=PatchContextId("context_" + "a" * 16),
            )
        case "domain":
            identity = replace(
                identity, domain_id=PatchDomainId("domain_" + "a" * 16)
            )
        case "target":
            identity = replace(
                identity, target_id=PatchTargetId("target_" + "a" * 16)
            )
        case "filesystem":
            identity = replace(identity, filesystem_id="docker-filesystem-v2")
        case "mount":
            identity = replace(identity, mount_id="docker-mount-v2")
        case "policy":
            identity = replace(identity, policy_revision="policy-v3")
        case "approval":
            identity = replace(
                identity,
                approval_channel_id=PatchApprovalId("approval_" + "a" * 16),
            )
        case "channel":
            return replace(
                settings,
                context=replace(
                    settings.context,
                    channel_id=SandboxChannelId("container-channel-v2"),
                ),
            )
        case "context_lifetime":
            return replace(
                settings,
                context=replace(
                    settings.context,
                    context_lifetime_id=SandboxContextLifetimeId(
                        "container-context-v2"
                    ),
                ),
            )
        case "implementation":
            implementation = SandboxWorkerImplementationId(
                "container-runtime-v2"
            )
            return replace(
                settings,
                context=replace(
                    settings.context,
                    identity=replace(
                        identity, implementation_id=implementation
                    ),
                    implementation_id=implementation,
                ),
            )
        case "cwd":
            return replace(
                settings,
                context=replace(settings.context, cwd=LogicalPath("nested")),
            )
        case "image":
            return replace(
                settings, image=ContainerPatchImage("sha256:" + "a" * 64)
            )
        case _:
            raise AssertionError("unknown plan-bound replacement")
    return replace(
        settings, context=replace(settings.context, identity=identity)
    )


async def _replace_volume_root(
    image: str, volume: str, root_subdirectory: LogicalPath
) -> None:
    """Replace one selected Docker root through an adversary mount."""
    root = "/workspace/" + root_subdirectory.value
    command = (
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=65536",
        "--mount",
        "type=volume,src=" + volume + ",dst=/workspace",
        image,
        "python3",
        "-I",
        "-c",
        (
            "from pathlib import Path;"
            f"root=Path({root!r});"
            "root.rename(root.with_name('parked-root'));"
            "root.mkdir();"
            "(root/'note.txt').write_bytes(b'canary\\n');"
            "print('replaced',end='')"
        ),
    )
    assert await _docker_output(command) == "replaced"


@pytest.mark.parametrize(
    "case",
    SHARED_CONTEXT_CORPUS,
    ids=lambda case: "container-" + case.case_id,
)
def test_patch_phase_11_reuses_shared_context_contract_corpus(
    tmp_path: Path,
    case: ContextCorpusCase,
) -> None:
    """Run inherited semantic, fault, and projection cases in Docker."""
    source = tmp_path / case.case_id
    source.mkdir()
    root_subdirectory = (
        LogicalPath("replaceable-root") if case.replace_root else None
    )
    seed_root = (
        source
        if root_subdirectory is None
        else source / root_subdirectory.value
    )
    seed_root.mkdir(exist_ok=True)
    for path, value in case.initial_files:
        target = seed_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    clock = _Clock()
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    configuration = _configuration(clock, authority)

    async def exercise() -> None:
        """Invoke one inherited request only through the container binder."""
        image = await _test_image()
        settings = replace(
            _settings(source, image), root_subdirectory=root_subdirectory
        )
        binder = _binder(settings, configuration, store)
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.edit", "patch.apply"])
        assert bundle.toolset is not None
        await bundle.manager.__aenter__()
        try:
            host = bundle.toolset.sdk_host()
            scope = await binder.runtime.resolve(
                ScopeSelection(ContextKind.CONTAINER)
            )
            if case.replace_root:
                volume = binder.runtime._process.volume_name
                assert volume is not None
                assert root_subdirectory is not None
                await _replace_volume_root(image, volume, root_subdirectory)
                with pytest.raises(PatchToolError):
                    await host.invoke_json(case.operation, case.arguments)
                for path, value in case.expected_files:
                    assert (
                        await _volume_bytes(
                            image, volume, "parked-root/" + path
                        )
                        == value
                    )
                assert (
                    await _volume_bytes(
                        image,
                        volume,
                        root_subdirectory.value + "/note.txt",
                    )
                    == b"canary\n"
                )
                return
            if case.inspection_only:
                inspected = await ContainerInspectionTarget(
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
                return
            if case.expected_error:
                with pytest.raises(PatchToolError):
                    await host.invoke_json(case.operation, case.arguments)
            else:
                outcome = await host.invoke_json(
                    case.operation, case.arguments
                )
                assert isinstance(outcome, PatchResult)
                assert outcome.status is case.expected_status
                assert set(project_model_result(outcome)) == {
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
                public = repr(project_model_result(outcome))
                assert str(source) not in public
                assert all(
                    value.decode("utf-8", errors="ignore").strip()
                    not in public
                    for _path, value in case.initial_files
                    if value.strip()
                )
            paths = tuple(
                LogicalPath(path)
                for path in dict.fromkeys(
                    path
                    for path, _value in (
                        *case.initial_files,
                        *case.expected_files,
                    )
                )
            )
            inspected = await ContainerInspectionTarget(
                binder.runtime
            ).inspect(InspectionRequest(scope, paths))
            observed = {
                snapshot.path.value: (
                    b""
                    if snapshot.bytes_value is None
                    else snapshot.bytes_value._value
                )
                for snapshot in inspected.snapshots
                if snapshot.present
            }
            assert tuple(sorted(observed.items())) == case.expected_files
        finally:
            await bundle.manager.__aexit__(None, None, None)
            await binder.runtime.dispose()

    run(exercise())


def test_patch_phase_11_subroot_commit_and_root_replacement_race(
    tmp_path: Path,
) -> None:
    """Commit inside one sub-root, then reject its recreated-root race."""
    source = tmp_path / "subroot-race"
    source.mkdir()
    selected = LogicalPath("selected-root")
    selected_root = source / selected.value
    selected_root.mkdir()
    (selected_root / "note.txt").write_text("before\n", encoding="utf-8")
    approval_authority = HmacDurableApprovalAuthority.random()
    store = _BlockingFenceStore(
        InMemoryDurablePatchBackend(approval_verifier=approval_authority)
    )

    async def exercise() -> None:
        """Preserve the parked selected root when a later fence races it."""
        image = await _test_image()
        settings = replace(
            _settings(source, image), root_subdirectory=selected
        )
        binder = _binder(
            settings, _configuration(_Clock(), approval_authority), store
        )
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        await bundle.manager.__aenter__()
        try:
            host = bundle.toolset.sdk_host()
            store.release_effect.set()
            first = await host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "before", "new_text": "after"}],
                },
            )
            assert type(first) is PatchResult
            assert first.status is PatchStatus.COMMITTED
            scope = await binder.runtime.resolve(
                ScopeSelection(ContextKind.CONTAINER)
            )
            inspected = await ContainerInspectionTarget(
                binder.runtime
            ).inspect(InspectionRequest(scope, (LogicalPath("note.txt"),)))
            assert inspected.snapshots[0].bytes_value is not None
            assert inspected.snapshots[0].bytes_value._value == b"after\n"
            volume = binder.runtime._process.volume_name
            assert volume is not None
            assert (
                await _volume_bytes(
                    image, volume, selected.value + "/note.txt"
                )
                == b"after\n"
            )

            store.checks = 0
            store.effect_reached.clear()
            store.release_effect.clear()
            raced = create_task(
                host.invoke_json(
                    OperationType.EDIT,
                    {
                        "path": "note.txt",
                        "edits": [{"old_text": "after", "new_text": "later"}],
                    },
                )
            )
            await store.effect_reached.wait()
            await _replace_volume_root(image, volume, selected)
            store.release_effect.set()
            outcome = await raced
            assert type(outcome) is PatchPending
            assert outcome.lifecycle is LifecyclePhase.SETTLEMENT_PENDING
            request = bundle.toolset._service._requests[outcome.request_id]
            settlement = await store.inspect(request.access)
            assert settlement.terminal is None
            assert settlement.pending is not None
            assert (
                await _volume_bytes(image, volume, "parked-root/note.txt")
                == b"after\n"
            )
            assert (
                await _volume_bytes(
                    image, volume, selected.value + "/note.txt"
                )
                == b"canary\n"
            )
        finally:
            await bundle.manager.__aexit__(None, None, None)
            await binder.runtime.dispose()

    run(exercise())


def test_patch_phase_11_commits_container_move_with_private_artifact_cleanup(
    tmp_path: Path,
) -> None:
    """Commit a no-replace move and settle its Docker artifact journal."""
    source = tmp_path / "move"
    source.mkdir()
    (source / "source.txt").write_text("source\n", encoding="utf-8")
    clock = _Clock()
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    configuration = _configuration(clock, authority)
    arguments = {
        "patch": "\n".join(
            (
                "*** Begin Patch v1",
                "*** Update File: source.txt",
                "*** Move to: moved.txt",
                "*** End Patch",
            )
        )
    }

    async def exercise() -> None:
        """Execute the exact move only through the Docker worker."""
        image = await _test_image()
        binder = _binder(_settings(source, image), configuration, store)
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.apply"])
        assert bundle.toolset is not None
        await bundle.manager.__aenter__()
        try:
            outcome = await bundle.toolset.sdk_host().invoke_json(
                OperationType.APPLY, arguments
            )
            assert isinstance(outcome, PatchResult)
            assert outcome.status is PatchStatus.COMMITTED
            scope = await binder.runtime.resolve(
                ScopeSelection(ContextKind.CONTAINER)
            )
            inspected = await ContainerInspectionTarget(
                binder.runtime
            ).inspect(
                InspectionRequest(
                    scope,
                    (LogicalPath("source.txt"), LogicalPath("moved.txt")),
                )
            )
            assert not inspected.snapshots[0].present
            assert inspected.snapshots[1].bytes_value is not None
            assert inspected.snapshots[1].bytes_value._value == b"source\n"
        finally:
            await bundle.manager.__aexit__(None, None, None)
            await binder.runtime.dispose()

    run(exercise())


def test_patch_phase_11_scopes_container_mutation_to_trusted_cwd(
    tmp_path: Path,
) -> None:
    """Use only the selected container cwd as the relative mutation root."""
    source = tmp_path / "seed"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (nested / "note.txt").write_text("before\n", encoding="utf-8")
    clock = _Clock()
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    configuration = _configuration(clock, authority)

    async def exercise() -> None:
        """Apply through the real worker and retain the root-level decoy."""
        image = await _test_image()
        settings = _settings(source, image)
        settings = replace(
            settings,
            context=replace(settings.context, cwd=LogicalPath("nested")),
        )
        binder = _binder(settings, configuration, store)
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        await bundle.manager.__aenter__()
        try:
            scope = await binder.runtime.resolve(
                ScopeSelection(ContextKind.CONTAINER)
            )
            assert scope.cwd == LogicalPath("nested")
            outcome = await bundle.toolset.sdk_host().invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "before", "new_text": "after"}],
                },
            )
            assert isinstance(outcome, PatchResult)
            assert outcome.status is PatchStatus.COMMITTED
            inspected = await ContainerInspectionTarget(
                binder.runtime
            ).inspect(InspectionRequest(scope, (LogicalPath("note.txt"),)))
            assert inspected.snapshots[0].bytes_value is not None
            assert inspected.snapshots[0].bytes_value._value == b"after\n"
        finally:
            await bundle.manager.__aexit__(None, None, None)
            await binder.runtime.dispose()

    run(exercise())


def test_patch_phase_11_inactive_profile_never_starts_or_advertises_runtime(
    tmp_path: Path,
) -> None:
    """Keep the Docker authority absent for each incapable test profile."""
    source = tmp_path / "seed"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority = HmacDurableApprovalAuthority.random()
    configuration = _configuration(_Clock(), authority)
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Reject activation without creating a volume or service process."""
        image = await _test_image()
        for profile in (
            PatchTestHostProfile(enabled=False, authenticated=True),
            PatchTestHostProfile(enabled=True, authenticated=False),
        ):
            binder = _binder(_settings(source, image), configuration, store)
            with pytest.raises(PatchToolError):
                await PatchToolLoader(binder, profile).load(
                    enable_tools=["patch.edit"]
                )
            assert binder.runtime._process._process is None
            assert binder.runtime._process.volume_name is None

    run(exercise())


def test_patch_phase_11_missing_container_endpoint_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject Docker activation when its private endpoint is absent."""
    source = tmp_path / "seed"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority = HmacDurableApprovalAuthority.random()
    configuration = _configuration(_Clock(), authority)
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Remove the sealed endpoint before capability issuance."""
        image = await _test_image()
        binder = _binder(_settings(source, image), configuration, store)
        monkeypatch.delattr(
            SandboxPatchSdkService, "_patch_container_endpoint"
        )
        with pytest.raises(PatchToolError, match="container endpoint"):
            await PatchToolLoader(
                binder, PatchTestHostProfile(enabled=True, authenticated=True)
            ).load(enable_tools=["patch.edit"])
        assert binder.runtime._process._process is None
        await binder.runtime.dispose()

    run(exercise())


def test_patch_phase_11_wrong_container_binder_reaps_runtime(
    tmp_path: Path,
) -> None:
    """Reject a substituted binder and close its live Docker service."""
    source = tmp_path / "seed"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority = HmacDurableApprovalAuthority.random()
    configuration = _configuration(_Clock(), authority)
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Use a forged structural binder after a real Docker bind."""
        image = await _test_image()
        binder = _binder(_settings(source, image), configuration, store)
        binding = await binder.bind()

        class WrongBinder:
            """Return a real binding without the selected runtime identity."""

            async def bind(self) -> PatchRuntimeBinding:
                """Return the already-live binding for rejection coverage."""
                return binding

        with pytest.raises(PatchToolError, match="selected runtime"):
            await PatchToolLoader(
                WrongBinder(),
                PatchTestHostProfile(enabled=True, authenticated=True),
            ).load(enable_tools=["patch.edit"])
        assert binder.runtime._process._process is None
        await binder.runtime.dispose()

    run(exercise())


def test_patch_phase_11_requirements(tmp_path: Path) -> None:
    """Commit through Docker while ordinary container mounts stay read-only."""
    source = tmp_path / "seed"
    source.mkdir()
    note = source / "note.txt"
    note.write_text("before\n", encoding="utf-8")
    note.chmod(0o644)
    second = source / "second.txt"
    second.write_text("second-before\n", encoding="utf-8")
    clock = _Clock()
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    configuration = _configuration(clock, authority)

    async def exercise() -> bytes:
        """Exercise immutable mounts, sealed edit, and lease reattachment."""
        image = await _test_image()
        settings = _settings(source, image)
        binder = _binder(settings, configuration, store)
        runtime = binder.runtime
        loader = PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        )
        bundle = await loader.load(enable_tools=["patch.edit", "patch.apply"])
        assert bundle.toolset is not None
        scope = await runtime.resolve(ScopeSelection(ContextKind.CONTAINER))
        target = ContainerPatchTarget(runtime)
        assert (await target.handshake(scope)).identity == scope.identity
        service = bundle.toolset._service
        endpoint = service._patch_container_endpoint()
        assert _PatchAuthorityValidator.container_endpoint_is_issued(endpoint)
        with pytest.raises(TargetInspectionError) as sandbox_endpoint:
            service._patch_sandbox_endpoint()
        assert sandbox_endpoint.value.code is TargetErrorCode.WITNESS_STALE
        service.scope = replace(scope, context_kind=ContextKind.SANDBOX)
        with pytest.raises(TargetInspectionError) as container_endpoint:
            service._patch_container_endpoint()
        assert container_endpoint.value.code is TargetErrorCode.WITNESS_STALE
        service.scope = scope
        assert runtime._receipt is not None
        assert runtime._receipt.canary_receipt
        assert runtime._receipt.backend_policy_digest
        assert runtime._receipt.runtime_command_digest
        initial = await ContainerInspectionTarget(runtime).inspect(
            InspectionRequest(scope, (LogicalPath("note.txt"),))
        )
        assert initial.snapshots[0].bytes_value is not None
        assert initial.snapshots[0].bytes_value._value == b"before\n"
        volume = runtime._process.volume_name
        assert volume is not None
        ordinary = _ordinary_read_command(image, volume, "note.txt")
        assert await _docker_output(ordinary) == "before\n"
        denied = ordinary[:-1] + (
            (
                "from pathlib import Path;"
                "Path('/workspace/note.txt').write_text('no')"
            ),
        )
        assert await _docker_output(denied, False) is None
        host = bundle.toolset.sdk_host()
        outcome = await host.invoke_json(
            OperationType.EDIT,
            {
                "path": "note.txt",
                "edits": [{"old_text": "before", "new_text": "after"}],
            },
        )
        assert outcome.status.value == "committed"
        assert note.read_text(encoding="utf-8") == "before\n"
        apply = await host.invoke_json(
            OperationType.APPLY, _multi_file_apply("after", "after-apply")
        )
        assert apply.status.value == "committed"
        later = await ContainerInspectionTarget(runtime).inspect(
            InspectionRequest(scope, (LogicalPath("note.txt"),))
        )
        assert later.snapshots[0].bytes_value is not None
        assert later.snapshots[0].bytes_value._value == b"after-apply\n"
        assert await _docker_output(ordinary) == "after-apply\n"
        assert (
            await _docker_output(
                ordinary[:-1]
                + (
                    (
                        "from pathlib import Path;"
                        "path=Path('/workspace/note.txt');"
                        "print(oct(path.stat().st_mode & 0o777),end='')"
                    ),
                )
            )
            == "0o644"
        )
        assert (
            await _docker_output(
                _ordinary_read_command(image, volume, "second.txt")
            )
            == "second-after\n"
        )
        projection = repr((outcome, await host.inspect()))
        note.write_text("host-divergent\n", encoding="utf-8")
        second.write_text("host-divergent\n", encoding="utf-8")
        assert await _docker_output(ordinary) == "after-apply\n"
        scratch = (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/scratch:rw,noexec,nosuid,size=65536",
            "--mount",
            "type=volume,src=" + volume + ",dst=/workspace,readonly",
            image,
            "python3",
            "-I",
            "-c",
            (
                "from pathlib import Path;"
                "Path('/scratch/output.txt').write_text('scratch');"
                "print(Path('/scratch/output.txt').read_text(),end='')"
            ),
        )
        assert await _docker_output(scratch) == "scratch"
        assert await _docker_output(ordinary) == "after-apply\n"
        process = runtime._process._process
        assert process is not None
        process.terminate()
        await process.wait()
        with pytest.raises(TargetInspectionError) as unavailable:
            await ContainerInspectionTarget(runtime).inspect(
                InspectionRequest(scope, (LogicalPath("note.txt"),))
            )
        assert unavailable.value.code is TargetErrorCode.WORKER_UNAVAILABLE
        assert await _docker_output(ordinary) == "after-apply\n"
        assert note.read_text(encoding="utf-8") == "host-divergent\n"
        object.__setattr__(
            runtime,
            "settings",
            replace(
                settings,
                execution_plan_fingerprint=SandboxExecutionPlanFingerprint(
                    "replaced-container-plan"
                ),
            ),
        )
        try:
            await runtime.resolve(ScopeSelection(ContextKind.CONTAINER))
        except TargetInspectionError:
            pass
        else:
            raise AssertionError("container plan replacement was accepted")
        assert volume not in projection
        assert str(runtime._process._bundle) not in projection
        await runtime.close()
        resumed = settings.create_runtime()
        try:
            resumed_scope = await resumed.resolve(
                ScopeSelection(ContextKind.CONTAINER)
            )
            retained = await ContainerInspectionTarget(resumed).inspect(
                InspectionRequest(resumed_scope, (LogicalPath("note.txt"),))
            )
            assert retained.snapshots[0].bytes_value is not None
            retained_bytes = retained.snapshots[0].bytes_value._value
            assert retained_bytes == b"after-apply\n"
            return retained_bytes
        finally:
            await resumed.dispose()

    retained_bytes = run(exercise())
    assert retained_bytes.startswith(b"after-apply")


def test_patch_phase_11_recovers_authenticated_lease_across_process_restart(
    tmp_path: Path,
) -> None:
    """Reclaim one exact idle Docker lease from a fresh calling process."""
    source = tmp_path / "process-restart"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority_key = b"x" * 32

    async def exercise() -> None:
        """Run two hosts, then reject wrong and concurrent recovery."""
        image = await _test_image()
        settings = replace(
            _settings(source, image),
            persistent_lease_authority=(
                ContainerPersistentLeaseAuthority.from_bytes(authority_key)
            ),
        )
        assert authority_key.hex() not in repr(settings)
        config = tmp_path / "restart-host-configuration.json"
        base = {
            "root": str(source),
            "image": image,
            "authority": b64encode(authority_key).decode(),
            "old_text": "before",
            "new_text": "after",
        }
        volume = _CONTAINER._docker_name(
            "avalan_patch_", settings.context.identity.persistent_lease_id
        )
        live = settings.create_runtime()
        try:
            config.write_text(dumps(base), encoding="utf-8")
            first = await _run_restart_process(config)
            assert first == {
                "status": PatchStatus.COMMITTED.value,
                "bytes": b64encode(b"after\n").decode(),
            }

            second_config = {
                **base,
                "old_text": "after",
                "new_text": "final",
            }
            config.write_text(dumps(second_config), encoding="utf-8")
            second = await _run_restart_process(config)
            assert second == {
                "status": PatchStatus.COMMITTED.value,
                "bytes": b64encode(b"final\n").decode(),
            }
            assert await _volume_bytes(image, volume, "note.txt") == b"final\n"

            wrong_key = {
                **second_config,
                "authority": b64encode(b"y" * 32).decode(),
            }
            config.write_text(dumps(wrong_key), encoding="utf-8")
            rejected_key = await _run_restart_process(config)
            assert rejected_key == {
                "error": TargetErrorCode.WITNESS_STALE.value
            }

            wrong_domain = {
                **second_config,
                "domain": "domain_" + "f" * 16,
            }
            config.write_text(dumps(wrong_domain), encoding="utf-8")
            rejected_domain = await _run_restart_process(config)
            assert rejected_domain == {
                "error": TargetErrorCode.WITNESS_STALE.value
            }
            assert await _volume_bytes(image, volume, "note.txt") == b"final\n"

            await live.resolve(ScopeSelection(ContextKind.CONTAINER))
            container_id = live._process._container_id
            assert container_id is not None
            inspected = await _docker_output(
                (
                    "docker",
                    "inspect",
                    container_id,
                )
            )
            assert inspected is not None
            assert authority_key.hex() not in inspected
            config.write_text(dumps(second_config), encoding="utf-8")
            rejected_live = await _run_restart_process(config)
            assert rejected_live == {
                "error": TargetErrorCode.WITNESS_STALE.value
            }
            assert await _volume_bytes(image, volume, "note.txt") == b"final\n"
        finally:
            await live.close()
            await _docker_output(
                ("docker", "volume", "rm", "--force", volume), False
            )

    run(exercise())


def test_patch_phase_11_serializes_initial_volume_creation_across_processes(
    tmp_path: Path,
) -> None:
    """Fence raced persistent-volume creation before worker attachment."""
    source = tmp_path / "initial-volume-race"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority_key = b"z" * 32

    async def exercise() -> None:
        """Hold labeled creation and require one durable guard owner."""
        image = await _test_image()
        settings = replace(
            _settings(source, image),
            persistent_lease_authority=(
                ContainerPersistentLeaseAuthority.from_bytes(authority_key)
            ),
        )
        base = {
            "root": str(source),
            "image": image,
            "authority": b64encode(authority_key).decode(),
            "old_text": "before",
            "new_text": "after",
        }
        ready = tmp_path / "initial-volume-race-ready"
        release = tmp_path / "initial-volume-race-release"
        first_config = tmp_path / "initial-volume-race-first.json"
        contender_config = tmp_path / "initial-volume-race-contender.json"
        recovery_config = tmp_path / "initial-volume-race-recovery.json"
        volume = _CONTAINER._docker_name(
            "avalan_patch_", settings.context.identity.persistent_lease_id
        )
        guard = _CONTAINER._docker_name(
            "avalan_patch_lease_",
            settings.context.identity.persistent_lease_id,
        )
        try:
            first_config.write_text(
                dumps(
                    {
                        **base,
                        "barrier_ready": str(ready),
                        "barrier_release": str(release),
                    }
                ),
                encoding="utf-8",
            )
            first_task = create_task(_run_restart_process(first_config))
            for _ in range(1_000):
                if ready.is_file():
                    break
                if first_task.done():
                    raise AssertionError(
                        "initial volume race exited before the barrier: "
                        + repr(first_task.result())
                    )
                await sleep(0.01)
            else:
                raise AssertionError(
                    "initial volume race did not reach barrier"
                )

            contender_config.write_text(dumps(base), encoding="utf-8")
            contender = await _run_restart_process(contender_config)
            assert contender == {"error": TargetErrorCode.WITNESS_STALE.value}
            assert (
                await _docker_output(("docker", "inspect", guard)) is not None
            )
            assert (
                await _docker_output(
                    (
                        "docker",
                        "ps",
                        "--quiet",
                        "--filter",
                        "volume=" + volume,
                    )
                )
                == ""
            )

            release.write_text("release\n", encoding="utf-8")
            first = await first_task
            assert first == {
                "status": PatchStatus.COMMITTED.value,
                "bytes": b64encode(b"after\n").decode(),
            }
            assert await _volume_bytes(image, volume, "note.txt") == b"after\n"
            assert (
                await _docker_output(("docker", "inspect", guard), False)
                is None
            )
            assert (
                await _docker_output(
                    (
                        "docker",
                        "ps",
                        "--quiet",
                        "--filter",
                        "volume=" + volume,
                    )
                )
                == ""
            )

            recovery_config.write_text(
                dumps({**base, "old_text": "after", "new_text": "final"}),
                encoding="utf-8",
            )
            recovery = await _run_restart_process(recovery_config)
            assert recovery == {
                "status": PatchStatus.COMMITTED.value,
                "bytes": b64encode(b"final\n").decode(),
            }
            assert await _volume_bytes(image, volume, "note.txt") == b"final\n"
            assert (
                await _docker_output(("docker", "inspect", guard), False)
                is None
            )
        finally:
            release.write_text("release\n", encoding="utf-8")
            await _docker_output(("docker", "rm", "--force", guard), False)
            await _docker_output(
                ("docker", "volume", "rm", "--force", volume), False
            )

    run(exercise())


def test_patch_phase_11_dispose_fails_closed_while_reclaim_owns_guard(
    tmp_path: Path,
) -> None:
    """Keep disposal fenced while another process reclaims before attach."""
    source = tmp_path / "dispose-reclaim-race"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority_key = b"q" * 32

    async def exercise() -> None:
        """Race retired disposal against a guard-held restarting process."""
        image = await _test_image()
        settings = replace(
            _settings(source, image),
            persistent_lease_authority=(
                ContainerPersistentLeaseAuthority.from_bytes(authority_key)
            ),
        )
        base = {
            "root": str(source),
            "image": image,
            "authority": b64encode(authority_key).decode(),
            "old_text": "before",
            "new_text": "after",
        }
        volume = _CONTAINER._docker_name(
            "avalan_patch_", settings.context.identity.persistent_lease_id
        )
        guard = _CONTAINER._docker_name(
            "avalan_patch_lease_",
            settings.context.identity.persistent_lease_id,
        )
        ready = tmp_path / "dispose-reclaim-ready"
        release = tmp_path / "dispose-reclaim-release"
        config = tmp_path / "dispose-reclaim.json"
        reclaim_task = None
        try:
            config.write_text(dumps(base), encoding="utf-8")
            assert await _run_restart_process(config) == {
                "status": PatchStatus.COMMITTED.value,
                "bytes": b64encode(b"after\n").decode(),
            }

            retired = _CONTAINER._ContainerRuntimeProcess(settings)
            await retired.start()
            await retired.close()
            assert retired._volume_owned
            assert retired._volume_guard_name is None

            config.write_text(
                dumps(
                    {
                        **base,
                        "old_text": "after",
                        "new_text": "final",
                        "barrier_ready": str(ready),
                        "barrier_release": str(release),
                        "barrier_stage": "guard_acquired",
                    }
                ),
                encoding="utf-8",
            )
            reclaim_task = create_task(_run_restart_process(config))
            for _ in range(1_000):
                if ready.is_file():
                    break
                if reclaim_task.done():
                    raise AssertionError(
                        "guard reclaim exited before the barrier: "
                        + repr(reclaim_task.result())
                    )
                await sleep(0.01)
            else:
                raise AssertionError("guard reclaim did not reach the barrier")

            with pytest.raises(TargetInspectionError) as rejected:
                await retired._dispose_owned_volume()
            assert rejected.value.code is TargetErrorCode.WITNESS_STALE
            assert await _volume_bytes(image, volume, "note.txt") == b"after\n"
            assert (
                await _docker_output(("docker", "inspect", guard)) is not None
            )
            assert (
                await _docker_output(
                    (
                        "docker",
                        "ps",
                        "--quiet",
                        "--filter",
                        "volume=" + volume,
                    )
                )
                == ""
            )

            release.write_text("release\n", encoding="utf-8")
            assert await reclaim_task == {
                "status": PatchStatus.COMMITTED.value,
                "bytes": b64encode(b"final\n").decode(),
            }
            reclaim_task = None
            assert await _volume_bytes(image, volume, "note.txt") == b"final\n"
            assert (
                await _docker_output(("docker", "inspect", guard), False)
                is None
            )
            assert (
                await _docker_output(
                    (
                        "docker",
                        "ps",
                        "--quiet",
                        "--filter",
                        "volume=" + volume,
                    )
                )
                == ""
            )

            await retired._dispose_owned_volume()
            assert (
                await _docker_output(
                    ("docker", "volume", "inspect", volume), False
                )
                is None
            )
            assert (
                await _docker_output(("docker", "inspect", guard), False)
                is None
            )
        finally:
            release.write_text("release\n", encoding="utf-8")
            if reclaim_task is not None:
                await reclaim_task
            await _docker_output(("docker", "rm", "--force", guard), False)
            await _docker_output(
                ("docker", "volume", "rm", "--force", volume), False
            )

    run(exercise())


def test_patch_phase_11_failed_start_cleanup_never_deletes_reclaimed_volume(
    tmp_path: Path,
) -> None:
    """Fence failed-start cleanup while a second process owns its guard."""
    source = tmp_path / "failed-start-cleanup-race"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority_key = b"s" * 32

    async def exercise() -> None:
        """Seed, fail, reclaim, and dispose one exact persistent volume."""
        image = await _test_image()
        settings = replace(
            _settings(source, image),
            persistent_lease_authority=(
                ContainerPersistentLeaseAuthority.from_bytes(authority_key)
            ),
        )
        base = {
            "root": str(source),
            "image": image,
            "authority": b64encode(authority_key).decode(),
            "old_text": "before",
            "new_text": "after",
        }
        volume = _CONTAINER._docker_name(
            "avalan_patch_", settings.context.identity.persistent_lease_id
        )
        guard = _CONTAINER._docker_name(
            "avalan_patch_lease_",
            settings.context.identity.persistent_lease_id,
        )
        failed_ready = tmp_path / "failed-start-cleanup-ready"
        failed_release = tmp_path / "failed-start-cleanup-release"
        reclaim_ready = tmp_path / "failed-start-reclaim-ready"
        reclaim_release = tmp_path / "failed-start-reclaim-release"
        failed_config = tmp_path / "failed-start-cleanup.json"
        reclaim_config = tmp_path / "failed-start-reclaim.json"
        failed_task = None
        reclaim_task = None
        try:
            failed_config.write_text(
                dumps(
                    {
                        **base,
                        "startup_failure": "attach",
                        "barrier_ready": str(failed_ready),
                        "barrier_release": str(failed_release),
                        "barrier_stage": "guard_released",
                    }
                ),
                encoding="utf-8",
            )
            failed_task = create_task(_run_restart_process(failed_config))
            for _ in range(1_000):
                if failed_ready.is_file():
                    break
                if failed_task.done():
                    raise AssertionError(
                        "failed starter exited before cleanup barrier: "
                        + repr(failed_task.result())
                    )
                await sleep(0.01)
            else:
                raise AssertionError(
                    "failed starter did not release its guard"
                )

            reclaim_config.write_text(
                dumps(
                    {
                        **base,
                        "barrier_ready": str(reclaim_ready),
                        "barrier_release": str(reclaim_release),
                        "barrier_stage": "guard_acquired",
                    }
                ),
                encoding="utf-8",
            )
            reclaim_task = create_task(_run_restart_process(reclaim_config))
            for _ in range(1_000):
                if reclaim_ready.is_file():
                    break
                if reclaim_task.done():
                    raise AssertionError(
                        "reclaimer exited before guard barrier: "
                        + repr(reclaim_task.result())
                    )
                await sleep(0.01)
            else:
                raise AssertionError("reclaimer did not acquire the guard")

            assert (
                await _volume_bytes(image, volume, "note.txt") == b"before\n"
            )
            assert (
                await _docker_output(("docker", "inspect", guard)) is not None
            )
            assert (
                await _docker_output(
                    (
                        "docker",
                        "ps",
                        "--quiet",
                        "--filter",
                        "volume=" + volume,
                    )
                )
                == ""
            )

            failed_release.write_text("release\n", encoding="utf-8")
            assert await failed_task == {
                "error": TargetErrorCode.CAPABILITY_UNAVAILABLE.value
            }
            failed_task = None
            assert (
                await _volume_bytes(image, volume, "note.txt") == b"before\n"
            )
            assert (
                await _docker_output(("docker", "inspect", guard)) is not None
            )

            reclaim_release.write_text("release\n", encoding="utf-8")
            assert await reclaim_task == {
                "status": PatchStatus.COMMITTED.value,
                "bytes": b64encode(b"after\n").decode(),
            }
            reclaim_task = None
            assert await _volume_bytes(image, volume, "note.txt") == b"after\n"
            assert (
                await _docker_output(("docker", "inspect", guard), False)
                is None
            )

            retired = _CONTAINER._ContainerRuntimeProcess(settings)
            await retired.start()
            await retired.dispose()
            assert (
                await _docker_output(
                    ("docker", "volume", "inspect", volume), False
                )
                is None
            )
            assert (
                await _docker_output(("docker", "inspect", guard), False)
                is None
            )
        finally:
            failed_release.write_text("release\n", encoding="utf-8")
            reclaim_release.write_text("release\n", encoding="utf-8")
            if failed_task is not None:
                await failed_task
            if reclaim_task is not None:
                await reclaim_task
            await _docker_output(("docker", "rm", "--force", guard), False)
            await _docker_output(
                ("docker", "volume", "rm", "--force", volume), False
            )

    run(exercise())


def test_patch_phase_11_failed_start_cleanup_defers_to_live_volume(
    tmp_path: Path,
) -> None:
    """Keep a failed starter from removing a live authenticated volume."""
    source = tmp_path / "failed-start-live-volume"
    source.mkdir()

    async def exercise() -> None:
        """Acquire cleanup's guard and reject the live-attachment race."""
        image = await _test_image()
        settings = _settings(source, image)
        process = _CONTAINER._ContainerRuntimeProcess(settings)
        resource_digest = _CONTAINER._persistent_resource_digest(settings)
        owner_receipt = _CONTAINER._volume_owner_receipt(
            settings.persistent_lease_authority, resource_digest
        )
        resource_label, owner_label = _CONTAINER._volume_labels(
            resource_digest, owner_receipt
        )
        volume = _CONTAINER._docker_name(
            "avalan_patch_", settings.context.identity.persistent_lease_id
        )
        guard = _CONTAINER._docker_name(
            "avalan_patch_lease_",
            settings.context.identity.persistent_lease_id,
        )
        live = _CONTAINER._docker_name(
            "avalan_patch_live_", settings.context.identity.persistent_lease_id
        )
        try:
            await _docker_output(
                (
                    "docker",
                    "volume",
                    "create",
                    "--label",
                    resource_label,
                    "--label",
                    owner_label,
                    volume,
                )
            )
            container_id = await _docker_output(
                (
                    "docker",
                    "run",
                    "--detach",
                    "--name",
                    live,
                    "--network",
                    "none",
                    "--mount",
                    "type=volume,src=" + volume + ",dst=/workspace",
                    image,
                    "python3",
                    "-I",
                    "-c",
                    "from time import sleep;sleep(30)",
                )
            )
            assert container_id is not None
            process._volume_name = volume
            process._volume_resource_digest = resource_digest
            process._volume_owner_receipt = owner_receipt
            process._volume_owned = True
            async with _CONTAINER._OWNED_VOLUMES_LOCK:
                _CONTAINER._OWNED_VOLUMES[volume] = (
                    _CONTAINER._OwnedVolumeClaim(
                        resource_digest, owner_receipt
                    )
                )

            await process._cleanup_new_volume(
                volume, resource_digest, owner_receipt
            )

            assert (
                await _docker_output(
                    ("docker", "volume", "inspect", volume), False
                )
                is not None
            )
            assert (
                await _docker_output(("docker", "inspect", live), False)
                is not None
            )
            assert (
                await _docker_output(("docker", "inspect", guard), False)
                is None
            )
            assert not process._volume_owned
            async with _CONTAINER._OWNED_VOLUMES_LOCK:
                assert volume not in _CONTAINER._OWNED_VOLUMES
        finally:
            await _docker_output(("docker", "rm", "--force", live), False)
            await _docker_output(("docker", "rm", "--force", guard), False)
            await _docker_output(
                ("docker", "volume", "rm", "--force", volume), False
            )

    run(exercise())


def test_patch_phase_11_e2e_020_reconciles_cancelled_multifile_apply(
    tmp_path: Path,
) -> None:
    """Reattach a cancelled Docker apply and record one terminal effect."""
    source = tmp_path / "seed"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    (source / "second.txt").write_text("second-before\n", encoding="utf-8")
    clock = _Clock()
    authority = HmacDurableApprovalAuthority.random()
    store = _BlockingFenceStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    configuration = _configuration(clock, authority)
    arguments = _multi_file_apply("before", "after")

    async def exercise() -> None:
        """Cancel the caller at the fence and await durable reconciliation."""
        image = await _test_image()
        settings = _settings(source, image)
        binder = _binder(settings, configuration, store)
        loader = PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        )
        bundle = await loader.load(enable_tools=["patch.apply"])
        assert bundle.toolset is not None
        await bundle.toolset.__aenter__()
        runtime = binder.runtime
        volume = runtime._process.volume_name
        assert volume is not None
        invocation = create_task(
            bundle.toolset.sdk_host().invoke_json(
                OperationType.APPLY, arguments
            )
        )
        await store.effect_reached.wait()
        invocation.cancel()
        pending = await invocation
        assert isinstance(pending, PatchPending)
        assert pending.lifecycle is LifecyclePhase.SETTLEMENT_PENDING
        assert (
            await _docker_output(
                _ordinary_read_command(image, volume, "note.txt")
            )
            == "before\n"
        )
        assert (
            await _docker_output(
                _ordinary_read_command(image, volume, "second.txt")
            )
            == "second-before\n"
        )
        fresh = await loader.load(enable_tools=["patch.apply"])
        assert fresh.toolset is not None
        attached = await fresh.toolset.sdk_host().retransmit_json(
            OperationType.APPLY,
            arguments,
            pending.request_id,
            pending.correlation_id,
        )
        assert attached == pending
        terminal = create_task(
            fresh.toolset.sdk_host().await_terminal(attached)
        )
        store.release_effect.set()
        result = await terminal
        assert isinstance(result, PatchResult)
        assert result.status is PatchStatus.COMMITTED
        assert result.lifecycle is LifecyclePhase.REQUEST_COMPLETED
        assert (
            await _docker_output(
                _ordinary_read_command(image, volume, "note.txt")
            )
            == "after\n"
        )
        assert (
            await _docker_output(
                _ordinary_read_command(image, volume, "second.txt")
            )
            == "second-after\n"
        )
        service = bundle.toolset._service
        access = service._requests[pending.request_id].access
        records = await store.outbox(access, SequenceNumber(0), 1024)
        assert (
            sum(
                record.lifecycle is LifecyclePhase.REQUEST_COMPLETED
                for record in records
            )
            == 1
        )
        await bundle.toolset.__aexit__(None, None, None)
        await runtime.dispose()

    run(exercise())


@pytest.mark.parametrize(
    "replacement",
    (
        "execution_plan",
        "persistent_lease",
        "workspace",
        "context",
        "domain",
        "target",
        "filesystem",
        "mount",
        "policy",
        "approval",
        "channel",
        "context_lifetime",
        "implementation",
        "cwd",
        "image",
    ),
)
def test_patch_phase_11_e2e_021_fences_replaced_plan_bound_context(
    tmp_path: Path, replacement: str
) -> None:
    """Reject a changed plan, lease, or workspace before Docker writes."""
    source = tmp_path / replacement
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    (source / "second.txt").write_text("second-before\n", encoding="utf-8")
    clock = _Clock()
    authority = HmacDurableApprovalAuthority.random()
    store = _BlockingFenceStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )
    configuration = _configuration(clock, authority)

    async def exercise() -> None:
        """Change one plan-bound witness while the durable fence is paused."""
        image = await _test_image()
        settings = _settings(source, image)
        binder = _binder(settings, configuration, store)
        loader = PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        )
        bundle = await loader.load(enable_tools=["patch.apply"])
        assert bundle.toolset is not None
        await bundle.toolset.__aenter__()
        runtime = binder.runtime
        volume = runtime._process.volume_name
        assert volume is not None
        invocation = create_task(
            bundle.toolset.sdk_host().invoke_json(
                OperationType.APPLY, _multi_file_apply("before", "after")
            )
        )
        await store.effect_reached.wait()
        object.__setattr__(
            runtime, "settings", _replacement_settings(settings, replacement)
        )
        with pytest.raises(TargetInspectionError) as stale:
            await runtime.resolve(ScopeSelection(ContextKind.CONTAINER))
        assert stale.value.code is TargetErrorCode.WITNESS_STALE
        store.release_effect.set()
        pending = await invocation
        assert isinstance(pending, PatchPending)
        service = bundle.toolset._service
        snapshot = await store.inspect(
            service._requests[pending.request_id].access
        )
        assert snapshot.terminal is None
        assert snapshot.pending is not None
        assert (
            await _docker_output(
                _ordinary_read_command(image, volume, "note.txt")
            )
            == "before\n"
        )
        assert (
            await _docker_output(
                _ordinary_read_command(image, volume, "second.txt")
            )
            == "second-before\n"
        )
        await bundle.toolset.__aexit__(None, None, None)
        await runtime.dispose()

    run(exercise())


def test_patch_phase_11_reconciles_post_dispatch_stale_after_first_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return exact truth when a later Docker effect loses its fence."""
    source = tmp_path / "post-dispatch-stale"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    (source / "second.txt").write_text("second-before\n", encoding="utf-8")
    authority = HmacDurableApprovalAuthority.random()
    store = _SecondEffectFenceStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Fence the second effect, then reconcile after dispatch."""
        image = await _test_image()
        settings = _settings(source, image)
        binder = _binder(settings, _configuration(_Clock(), authority), store)
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.apply"])
        assert bundle.toolset is not None
        await bundle.manager.__aenter__()
        service = bundle.toolset._service
        original_invoke = SandboxPatchSdkService.invoke

        async def stale_after_dispatch(
            current: SandboxPatchSdkService, *arguments: object
        ) -> PatchResult | PatchPending:
            """Raise after the real service has stored durable truth."""
            outcome = await original_invoke(current, *arguments)
            if current is service:
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            return outcome

        monkeypatch.setattr(
            SandboxPatchSdkService, "invoke", stale_after_dispatch
        )
        try:
            volume = binder.runtime._process.volume_name
            assert volume is not None
            invocation = create_task(
                bundle.toolset.sdk_host().invoke_json(
                    OperationType.APPLY, _multi_file_apply("before", "after")
                )
            )
            await store.effect_reached.wait()
            object.__setattr__(
                binder.runtime,
                "settings",
                _replacement_settings(settings, "filesystem"),
            )
            store.release_effect.set()
            outcome = await invocation
            assert isinstance(outcome, PatchPending)
            snapshot = await store.inspect(
                service._requests[outcome.request_id].access
            )
            assert snapshot.terminal is None
            assert snapshot.pending is not None
            assert [
                entry.state.value
                for entry in snapshot.journal.steps
                if entry.state.value in {"committed", "not_committed"}
            ] == ["committed", "not_committed"]
            assert await _volume_bytes(image, volume, "note.txt") == b"after\n"
            assert (
                await _volume_bytes(image, volume, "second.txt")
                == b"second-before\n"
            )
            records = await store.outbox(
                service._requests[outcome.request_id].access,
                SequenceNumber(0),
                1024,
            )
            assert not any(
                record.lifecycle is LifecyclePhase.REQUEST_COMPLETED
                for record in records
            )
        finally:
            await bundle.manager.__aexit__(None, None, None)
            await binder.runtime.dispose()

    run(exercise())


@pytest.mark.parametrize("attack", ("forged", "replayed", "out_of_order"))
def test_patch_phase_11_rejects_forged_replayed_and_out_of_order_channel(
    tmp_path: Path, attack: str
) -> None:
    """Fail closed when the sole Docker control channel loses integrity."""
    source = tmp_path / attack
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")

    async def exercise() -> None:
        """Send a bad channel message and retain the immutable volume."""
        image = await _test_image()
        settings = _settings(source, image)
        runtime = settings.create_runtime()
        scope = await runtime.resolve(ScopeSelection(ContextKind.CONTAINER))
        request = InspectionRequest(scope, (LogicalPath("note.txt"),))
        assert (
            await ContainerInspectionTarget(runtime).inspect(request)
        ).snapshots[0].bytes_value is not None
        process = runtime._process
        volume = process.volume_name
        assert volume is not None
        match attack:
            case "forged":
                process._token = b"\\x00" * 32
            case "replayed":
                process._sequence -= 1
            case "out_of_order":
                process._sequence -= 2
            case _:
                raise AssertionError("unknown channel attack")
        with pytest.raises(TargetInspectionError) as rejected:
            await ContainerInspectionTarget(runtime).inspect(request)
        assert rejected.value.code in {
            TargetErrorCode.WITNESS_STALE,
            TargetErrorCode.WORKER_UNAVAILABLE,
        }
        assert (
            await _docker_output(
                _ordinary_read_command(image, volume, "note.txt")
            )
            == "before\n"
        )
        await runtime.dispose()

    run(exercise())


def test_patch_phase_11_preserves_container_representation_and_metadata(
    tmp_path: Path,
) -> None:
    """Round-trip Docker text, mode, and protected xattr metadata exactly."""
    source = tmp_path / "representation"
    source.mkdir()
    (source / "bom.txt").write_bytes(b"\xef\xbb\xbfbefore\r\n")
    (source / "none.txt").write_bytes(b"before")
    (source / "metadata.txt").write_bytes(b"before\n")
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Use only the mounted volume to set and verify native metadata."""
        image = await _test_image()
        binder = _binder(
            _settings(source, image),
            _configuration(_Clock(), authority),
            store,
        )
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.apply"])
        assert bundle.toolset is not None
        await bundle.manager.__aenter__()
        try:
            volume = binder.runtime._process.volume_name
            assert volume is not None
            prepared = await _docker_output(
                _volume_command(
                    image,
                    volume,
                    "import os;"
                    "path='/workspace/metadata.txt';"
                    "os.chmod(path,0o640);"
                    "os.setxattr(path,b'user.avalan.phase11',b'retained')",
                )
            )
            assert prepared == ""
            outcome = await bundle.toolset.sdk_host().invoke_json(
                OperationType.APPLY,
                {
                    "patch": "\n".join(
                        (
                            "*** Begin Patch v1",
                            "*** Update File: bom.txt",
                            "@@",
                            "-before",
                            "+after",
                            "*** Update File: none.txt",
                            "@@",
                            "-before",
                            "\\ No newline at end of file",
                            "+after",
                            "\\ No newline at end of file",
                            "*** End of File",
                            "*** Update File: metadata.txt",
                            "@@",
                            "-before",
                            "+after",
                            "*** End Patch",
                        )
                    )
                },
            )
            assert isinstance(outcome, PatchResult)
            assert outcome.status is PatchStatus.COMMITTED
            assert (
                await _volume_bytes(image, volume, "bom.txt")
                == b"\xef\xbb\xbfafter\r\n"
            )
            assert await _volume_bytes(image, volume, "none.txt") == b"after"
            assert (
                await _volume_bytes(image, volume, "metadata.txt")
                == b"after\n"
            )
            protected = await _docker_output(
                _volume_command(
                    image,
                    volume,
                    "import os;"
                    "path='/workspace/metadata.txt';"
                    "print(oct(os.stat(path).st_mode & 0o777)+':'"
                    "+os.getxattr(path,b'user.avalan.phase11').decode(),end='')",
                    read_only=True,
                )
            )
            assert protected == "0o640:retained"
            receipt = binder.runtime._receipt
            assert receipt is not None
            assert receipt.canary_receipt
            assert receipt.primitive_receipts[
                _CONTAINER.TargetPrimitive.METADATA_PRESERVATION
            ]
        finally:
            await bundle.manager.__aexit__(None, None, None)
            await binder.runtime.dispose()

    run(exercise())


@pytest.mark.parametrize(
    ("attack", "setup", "path", "old_text", "canary"),
    (
        (
            "symlink",
            "import os;os.symlink('outside.txt','/workspace/linked.txt')",
            "linked.txt",
            "outside",
            "outside.txt",
        ),
        (
            "hardlink_alias",
            (
                "import os;"
                "os.link('/workspace/note.txt','/workspace/note-alias.txt')"
            ),
            "note.txt",
            "before",
            "note-alias.txt",
        ),
        (
            "special",
            "import os;os.mkfifo('/workspace/special.txt')",
            "special.txt",
            "before",
            "note.txt",
        ),
        (
            "ancestor_link",
            (
                "import os;os.rename('/workspace/nested','/workspace/parked');"
                "os.symlink('/workspace/outside','/workspace/nested')"
            ),
            "nested/note.txt",
            "outside",
            "outside/note.txt",
        ),
    ),
)
def test_patch_phase_11_rejects_hostile_container_volume_topology(
    tmp_path: Path,
    attack: str,
    setup: str,
    path: str,
    old_text: str,
    canary: str,
) -> None:
    """Reject links, aliases, special files, and linked ancestors in Docker."""
    source = tmp_path / attack
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    (source / "outside.txt").write_text("outside\n", encoding="utf-8")
    (source / "nested").mkdir()
    (source / "nested" / "note.txt").write_text("before\n", encoding="utf-8")
    (source / "outside").mkdir()
    (source / "outside" / "note.txt").write_text("outside\n", encoding="utf-8")
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Inject one hostile entry only through the selected Docker volume."""
        image = await _test_image()
        binder = _binder(
            _settings(source, image),
            _configuration(_Clock(), authority),
            store,
        )
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        await bundle.manager.__aenter__()
        try:
            volume = binder.runtime._process.volume_name
            assert volume is not None
            assert (
                await _docker_output(_volume_command(image, volume, setup))
                == ""
            )
            with pytest.raises((PatchToolError, TargetInspectionError)):
                await bundle.toolset.sdk_host().invoke_json(
                    OperationType.EDIT,
                    {
                        "path": path,
                        "edits": [{"old_text": old_text, "new_text": "after"}],
                    },
                )
            assert await _volume_bytes(image, volume, canary) in {
                b"before\n",
                b"outside\n",
            }
            assert not tuple(source.glob(".avalan-patch-*"))
        finally:
            await bundle.manager.__aexit__(None, None, None)
            await binder.runtime.dispose()

    run(exercise())


def test_patch_phase_11_container_public_lifecycle_and_surfaces_are_redacted(
    tmp_path: Path,
) -> None:
    """Keep Docker internals out of result, event, and SDK projections."""
    source = tmp_path / "private-source"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Collect the complete public lifecycle after one container effect."""
        image = await _test_image()
        settings = _settings(source, image)
        binder = _binder(settings, _configuration(_Clock(), authority), store)
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        await bundle.manager.__aenter__()
        try:
            runtime = binder.runtime
            process = runtime._process
            volume = process.volume_name
            token = process._token
            receipt = runtime._receipt
            assert volume is not None
            assert token is not None
            assert receipt is not None
            host = bundle.toolset.sdk_host()
            outcome = await host.invoke_json(
                OperationType.EDIT,
                {
                    "path": "note.txt",
                    "edits": [{"old_text": "before", "new_text": "after"}],
                },
            )
            assert isinstance(outcome, PatchResult)
            assert outcome.status is PatchStatus.COMMITTED
            events = [event async for event in host.lifecycle()]
            assert [event.lifecycle for event in events] == [
                LifecyclePhase.REQUEST_COMPLETED
            ]
            assert [event.sequence.value for event in events] == [1]
            assert events[0].request_id == outcome.request_id
            assert len({event.event_id for event in events}) == 1
            service = bundle.toolset._service
            access = service._requests[outcome.request_id].access
            snapshot = await store.inspect(access)
            assert snapshot.plan is not None
            assert snapshot.plan.plan_id == outcome.plan_id
            assert (
                snapshot.plan.context_id
                == settings.context.identity.context_id
            )
            assert snapshot.plan.workspace_id == (
                settings.context.identity.workspace_id
            )
            assert (
                snapshot.plan.domain_id == settings.context.identity.domain_id
            )
            assert snapshot.terminal is not None
            assert snapshot.terminal.result == outcome
            assert snapshot.terminal.outbox.event_id == events[0].event_id
            forged_pending = PatchPending(
                1,
                PatchPendingOperationId("pending_" + "f" * 16),
                outcome.request_id,
                events[0].correlation_id,
                LifecyclePhase.SETTLEMENT_PENDING,
            )
            with pytest.raises(TargetInspectionError) as forged_handle:
                await host.await_terminal(forged_pending)
            assert forged_handle.value.code is TargetErrorCode.WITNESS_STALE
            fresh_host = bundle.toolset.sdk_host()
            with pytest.raises(PatchToolError, match="request identity"):
                await fresh_host.retransmit_json(
                    OperationType.EDIT,
                    {
                        "path": "note.txt",
                        "edits": [{"old_text": "before", "new_text": "after"}],
                    },
                    outcome.request_id,
                    PatchObserverCorrelationId("correlation_" + "f" * 16),
                )
            public_values = (
                outcome,
                events,
                project_model_result(outcome),
                await host.inspect(),
            )
            private_markers = (
                str(source),
                volume,
                token.hex(),
                receipt.session_id,
                settings.context.channel_id,
                settings.context.implementation_id,
                "/private",
                "/workspace",
                "AVALAN_SANDBOX_PATCH_SESSION",
            )
            assert all(
                marker not in repr(value)
                for value in public_values
                for marker in private_markers
            )
            assert await _volume_bytes(image, volume, "note.txt") == b"after\n"
        finally:
            await bundle.manager.__aexit__(None, None, None)
            await binder.runtime.dispose()

    run(exercise())


def test_patch_phase_11_serializes_container_contexts_in_one_domain(
    tmp_path: Path,
) -> None:
    """Fence two Docker services that point at one persistent workspace."""
    source = tmp_path / "shared-volume"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority = HmacDurableApprovalAuthority.random()
    store = _BlockingFenceStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Compete distinct container contexts at a shared durable fence."""
        image = await _test_image()
        first_settings = _settings(source, image)
        first_identity = first_settings.context.identity
        second_settings = replace(
            first_settings,
            context=replace(
                first_settings.context,
                identity=replace(
                    first_identity,
                    context_id=PatchContextId("context_" + "b" * 16),
                    target_id=PatchTargetId("target_" + "b" * 16),
                ),
            ),
        )
        first = _binder(
            first_settings, _configuration(_Clock(), authority), store
        )
        second = _binder(
            second_settings, _configuration(_Clock(), authority), store
        )
        first_bundle = await PatchToolLoader(
            first, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.edit"])
        second_bundle = await PatchToolLoader(
            second, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.edit"])
        assert first_bundle.toolset is not None
        assert second_bundle.toolset is not None
        await first_bundle.manager.__aenter__()
        await second_bundle.manager.__aenter__()
        try:
            volume = first.runtime._process.volume_name
            assert volume is not None
            assert second.runtime._process.volume_name == volume
            assert (
                first_bundle.toolset._service.store
                is second_bundle.toolset._service.store
                is store
            )
            first_invocation = create_task(
                first_bundle.toolset.sdk_host().invoke_json(
                    OperationType.EDIT,
                    {
                        "path": "note.txt",
                        "edits": [{"old_text": "before", "new_text": "after"}],
                    },
                )
            )
            await store.effect_reached.wait()
            with pytest.raises(PatchToolError, match="reconciliation"):
                await second_bundle.toolset.sdk_host().invoke_json(
                    OperationType.EDIT,
                    {
                        "path": "note.txt",
                        "edits": [
                            {"old_text": "before", "new_text": "second"}
                        ],
                    },
                )
            active = store._backend.active_leases[first_identity.domain_id]
            assert active.request_id in first_bundle.toolset._service._requests
            assert (
                await _volume_bytes(image, volume, "note.txt") == b"before\n"
            )
            store.release_effect.set()
            first_result = await first_invocation
            assert isinstance(first_result, PatchResult)
            assert first_result.status is PatchStatus.COMMITTED
            assert await _volume_bytes(image, volume, "note.txt") == b"after\n"
        finally:
            await second_bundle.manager.__aexit__(None, None, None)
            await first_bundle.manager.__aexit__(None, None, None)
            await second.runtime.dispose()
            await first.runtime.dispose()

    run(exercise())


def test_patch_phase_11_rejects_mismatched_domain_for_one_persistent_volume(
    tmp_path: Path,
) -> None:
    """Refuse a second domain before it can attach the same Docker volume."""
    source = tmp_path / "mismatched-domain"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Start one valid lease then reject a colliding foreign domain."""
        image = await _test_image()
        settings = _settings(source, image)
        first = _binder(settings, _configuration(_Clock(), authority), store)
        first_bundle = await PatchToolLoader(
            first, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.edit"])
        assert first_bundle.toolset is not None
        mismatched = replace(
            settings,
            context=replace(
                settings.context,
                identity=replace(
                    settings.context.identity,
                    domain_id=PatchDomainId("domain_" + "f" * 16),
                ),
            ),
        )
        second = _binder(
            mismatched, _configuration(_Clock(), authority), store
        )
        try:
            volume = first.runtime._process.volume_name
            assert volume is not None
            with pytest.raises(TargetInspectionError) as rejected:
                await PatchToolLoader(
                    second,
                    PatchTestHostProfile(enabled=True, authenticated=True),
                ).load(enable_tools=["patch.edit"])
            assert rejected.value.code is TargetErrorCode.WITNESS_STALE
            assert second.runtime._process.volume_name is None
            assert (
                await _volume_bytes(image, volume, "note.txt") == b"before\n"
            )
        finally:
            await first_bundle.manager.__aexit__(None, None, None)
            await first.runtime.dispose()

    run(exercise())


def test_patch_phase_11_rejects_destination_race_at_container_fence(
    tmp_path: Path,
) -> None:
    """Preserve a raced Docker destination and settle no committed effect."""
    source = tmp_path / "destination-race"
    source.mkdir()
    authority = HmacDurableApprovalAuthority.random()
    store = _BlockingFenceStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Create a foreign name after the sealed plan reaches its fence."""
        image = await _test_image()
        binder = _binder(
            _settings(source, image),
            _configuration(_Clock(), authority),
            store,
        )
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.apply"])
        assert bundle.toolset is not None
        await bundle.manager.__aenter__()
        try:
            volume = binder.runtime._process.volume_name
            assert volume is not None
            invocation = create_task(
                bundle.toolset.sdk_host().invoke_json(
                    OperationType.APPLY,
                    {
                        "patch": "\n".join(
                            (
                                "*** Begin Patch v1",
                                "*** Add File: created.txt",
                                "+planned",
                                "*** End Patch",
                            )
                        )
                    },
                )
            )
            await store.effect_reached.wait()
            assert (
                await _docker_output(
                    _volume_command(
                        image,
                        volume,
                        "from pathlib import Path;"
                        "Path('/workspace/created.txt').write_text('foreign\\n')",
                    )
                )
                == ""
            )
            store.release_effect.set()
            outcome = await invocation
            assert isinstance(outcome, PatchResult)
            assert outcome.status is PatchStatus.COMMIT_FAILED
            assert outcome.truth.requested_effect_occurred.value == "false"
            assert (
                await _volume_bytes(image, volume, "created.txt")
                == b"foreign\n"
            )
            artifacts = await _docker_output(
                _volume_command(
                    image,
                    volume,
                    "from pathlib import Path;"
                    "print(','.join(sorted(path.name for path in "
                    "Path('/workspace').glob('.avalan-patch-*'))),end='')",
                    read_only=True,
                )
            )
            assert artifacts == ""
        finally:
            await bundle.manager.__aexit__(None, None, None)
            await binder.runtime.dispose()

    run(exercise())


def test_patch_phase_11_service_loss_at_fence_stays_pending_without_effect(
    tmp_path: Path,
) -> None:
    """Reconcile a lost Docker channel as pending instead of guessing truth."""
    source = tmp_path / "lost-service"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority = HmacDurableApprovalAuthority.random()
    store = _BlockingFenceStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Sever the attached service while its fence is blocked."""
        image = await _test_image()
        binder = _binder(
            _settings(source, image),
            _configuration(_Clock(), authority),
            store,
        )
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        await bundle.toolset.__aenter__()
        try:
            volume = binder.runtime._process.volume_name
            assert volume is not None
            invocation = create_task(
                bundle.toolset.sdk_host().invoke_json(
                    OperationType.EDIT,
                    {
                        "path": "note.txt",
                        "edits": [{"old_text": "before", "new_text": "after"}],
                    },
                )
            )
            await store.effect_reached.wait()
            process = binder.runtime._process._process
            assert process is not None
            process.terminate()
            await process.wait()
            store.release_effect.set()
            outcome = await invocation
            assert isinstance(outcome, PatchPending)
            assert outcome.lifecycle is LifecyclePhase.SETTLEMENT_PENDING
            request = bundle.toolset._service._requests[outcome.request_id]
            snapshot = await store.inspect(request.access)
            assert snapshot.terminal is None
            assert snapshot.pending is not None
            assert (
                await _volume_bytes(image, volume, "note.txt") == b"before\n"
            )
        finally:
            await bundle.toolset.__aexit__(None, None, None)
            await binder.runtime.dispose()

    run(exercise())


def test_patch_phase_11_container_service_has_only_its_sealed_authority(
    tmp_path: Path,
) -> None:
    """Inspect the live Docker profile and exclude ordinary authority."""
    source = tmp_path / "sealed-service"
    source.mkdir()
    (source / "note.txt").write_text("before\n", encoding="utf-8")
    authority = HmacDurableApprovalAuthority.random()
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(approval_verifier=authority)
    )

    async def exercise() -> None:
        """Verify the running worker has only its fixed IPC and mounts."""
        image = await _test_image()
        binder = _binder(
            _settings(source, image),
            _configuration(_Clock(), authority),
            store,
        )
        bundle = await PatchToolLoader(
            binder, PatchTestHostProfile(enabled=True, authenticated=True)
        ).load(enable_tools=["patch.edit"])
        assert bundle.toolset is not None
        try:
            process = binder.runtime._process
            container_id = process._container_id
            volume = process.volume_name
            assert container_id is not None
            assert volume is not None
            inspected_raw = await _docker_output(
                (
                    "docker",
                    "inspect",
                    container_id,
                )
            )
            assert inspected_raw is not None
            inspected = loads(inspected_raw)
            assert isinstance(inspected, list) and len(inspected) == 1
            service = inspected[0]
            assert isinstance(service, dict)
            host_config = service["HostConfig"]
            assert isinstance(host_config, dict)
            assert host_config["NetworkMode"] == "none"
            assert host_config["ReadonlyRootfs"] is True
            assert host_config["Privileged"] is False
            assert host_config["CapDrop"] == ["ALL"]
            assert not host_config["CapAdd"]
            mounts = service["Mounts"]
            assert isinstance(mounts, list)
            assert {
                (item["Destination"], item["RW"], item["Type"])
                for item in mounts
                if isinstance(item, dict)
            } == {
                ("/implementation", False, "bind"),
                ("/workspace", True, "volume"),
            }
            ordinary = await _docker_output(
                _volume_command(
                    image,
                    volume,
                    "import os;from pathlib import Path;"
                    "print(f'{Path(\"/implementation\").exists()}:'"
                    "f'{\"AVALAN_SANDBOX_PATCH_SESSION\" in os.environ}',"
                    "end='')",
                    read_only=True,
                )
            )
            assert ordinary == "False:False"
        finally:
            await binder.runtime.dispose()

    run(exercise())


if __name__ == "__main__":
    if len(argv) != 3 or argv[1] != "--phase11-restart-process":
        raise SystemExit("expected one Phase 11 restart process configuration")
    run(_restart_process_from_config(Path(argv[2])))
