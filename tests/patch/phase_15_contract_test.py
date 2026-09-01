"""Exercise Phase 15 activation through explicit runtime factory seams."""

from asyncio import create_subprocess_exec, create_task, run, sleep
from dataclasses import replace
from json import loads
from os import mkfifo
from pathlib import Path
from sys import executable
from sys import platform as runtime_platform
from threading import Event
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)
from patch_activation_support import patch_test_activation_factory

import avalan.patch.local_commit as local_commit_module
import avalan.patch.target as target_module
from avalan.agent.loader import OrchestratorLoader
from avalan.patch.activation import (
    PatchActivationPlatform,
    PatchActivationRegistry,
    PatchActivationRuntime,
    PatchActivationRuntimeFactory,
    PatchProfileState,
    _build_activation_verifier,
    _new_activation_authority,
    build_patch_production_manifest,
    build_patch_runtime_activation_factory,
    render_patch_production_manifest,
)
from avalan.patch.coordinator import (
    InMemoryCoordinatorStore,
    InMemoryLeaseManager,
    InMemoryPatchCoordinator,
    RetransmissionKey,
    RevalidationFact,
    RevalidationField,
    RevalidationSnapshot,
    RuntimeIdentity,
    ScriptedReconciler,
)
from avalan.patch.domain import (
    ApprovalMode,
    ByteSize,
    Capability,
    ContextKind,
    DurationTicks,
    ExpiryTick,
    LogicalPath,
    OperationType,
    PatchApprovalId,
    PatchCommitOwnerId,
    PatchContextId,
    PatchDomainId,
    PatchExecutionId,
    PatchInput,
    PatchLimits,
    PatchPlanId,
    PatchProtocolId,
    PatchRequest,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PatchTargetId,
    PatchWorkspaceId,
    SequenceNumber,
)
from avalan.patch.durable_store import (
    DurableCommitLease,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.local_commit import LocalCommitTarget
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
from avalan.patch.planner import plan
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
    PlanBinding,
    PlanReviewRequest,
    PolicyAuthorizer,
    PolicyBrokerId,
    PolicyPathSelector,
    PolicyReviewerRole,
    PolicyRevision,
    PolicyRouteId,
    PolicyRule,
    PreauthorizationClass,
    PreflightRequest,
    ReviewerDecision,
    RuntimeGrantStore,
    TrustedPatchPolicy,
    compose_limits,
    seal_plan,
)
from avalan.patch.target import (
    AliasMode,
    InspectionRequest,
    LocalInspectionTarget,
    LocalPlatformProfile,
    LocalScopeResolver,
    LocalTargetProfile,
    ScopeSelection,
    TargetErrorCode,
    TargetIdentity,
    TargetInspectionError,
    TrustedLocalRoot,
)
from avalan.tool.manager import ToolManager

_ROOT = Path(__file__).resolve().parents[2]
_DOCUMENTATION = _ROOT / "docs" / "PATCH.md"
_FIXTURE = _ROOT / "tests" / "fixtures" / "patch" / "activation_manifest.json"


class _ObservedService:
    """Expose the production service's activation-observer attach seam."""

    def __init__(self) -> None:
        """Initialize one empty attached observer slot."""
        self.observer: object | None = None

    def set_activation_observer(self, observer: object) -> None:
        """Attach one factory-owned observer exactly once."""
        if self.observer is not None:
            raise ValueError("observer already attached")
        self.observer = observer


def _runtime_binding(
    store: InMemoryDurablePatchStore, service: _ObservedService
) -> object:
    """Return a completed authenticated binding over one durable store."""
    identity = SimpleNamespace(
        target_id=SimpleNamespace(value="target_phase15"),
        workspace_id=SimpleNamespace(value="workspace_phase15"),
        domain_id=SimpleNamespace(value="domain_phase15"),
    )
    return SimpleNamespace(
        scope=SimpleNamespace(
            context_kind=ContextKind.SANDBOX,
            identity=identity,
        ),
        handshake=SimpleNamespace(
            platform=SimpleNamespace(value=PatchActivationPlatform.MACOS.value)
        ),
        coordinator=SimpleNamespace(durable_store=store),
        persistence=SimpleNamespace(durable_store=store),
        policy=SimpleNamespace(revision=SimpleNamespace(value="policy-v1")),
        service=service,
    )


def _lease(request_id: PatchRequestId, fence: int) -> DurableCommitLease:
    """Return one typed coordinator-issued durable lease fixture."""
    return DurableCommitLease(
        request_id,
        PatchDomainId("domain_" + "a" * 16),
        PatchCommitOwnerId("owner_" + "b" * 16),
        SequenceNumber(fence),
        ExpiryTick(100),
    )


class _Phase15LocalClock(ApprovalClock):
    """Return the unexpired tick used by the direct local integration."""

    async def now(self) -> ExpiryTick:
        """Return one fixed monotonic tick before the plan expiry."""
        return ExpiryTick(1)


class _Phase15LocalBroker:
    """Approve the exact sealed local plan through the production service."""

    async def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Return the required reviewer decision for this plan only."""
        return BrokerDecision(
            request.requirements.broker,
            (
                ReviewerDecision(
                    PatchPrincipalId("reviewer_phase15"),
                    request.subject.tenant,
                    request.requirements.reviewer_role,
                    ApprovalDecisionState.APPROVED,
                ),
            ),
        )


def _phase15_local_limits() -> PatchLimits:
    """Return finite limits for the direct rooted local integration."""
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


def _phase15_local_profile(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> LocalTargetProfile:
    """Build one explicit test deployment authority for the selected root."""
    signer = Ed25519PrivateKey.generate()
    verifier = signer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    original_bootstrap = target_module._WORKER_BOOTSTRAP
    child_bootstrap = original_bootstrap.replace(
        "from avalan.patch.target import _worker_main\n"
        "raise SystemExit(_worker_main())",
        "import avalan.patch.target as target_module\n"
        "target_module._RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES = "
        + repr(verifier)
        + "\nraise SystemExit(target_module._worker_main())",
    )
    assert child_bootstrap != original_bootstrap
    monkeypatch.setattr(
        target_module, "_RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES", verifier
    )
    monkeypatch.setattr(target_module, "_WORKER_BOOTSTRAP", child_bootstrap)
    authority = target_module._RuntimeTargetAuthority(
        signer.sign(target_module._runtime_target_authority_message(root))
    )
    trusted_root = TrustedLocalRoot(root, _runtime_authority=authority)
    witness = target_module._capture_root_witness(trusted_root)
    identity = TargetIdentity(
        PatchContextId("context_" + "f" * 16),
        PatchWorkspaceId("workspace_" + "f" * 16),
        PatchDomainId("domain_" + "f" * 16),
        PatchTargetId("target_" + "f" * 16),
        PatchProtocolId("protocol_" + "f" * 16),
        witness.filesystem_id,
        witness.mount_id,
        "policy-phase15",
        "workspace-lease-phase15",
        PatchApprovalId("approval_" + "f" * 16),
    )
    namespace = root.parent / ".avalan-patch-phase15-private"
    namespace.mkdir(mode=0o700)
    return LocalTargetProfile(
        identity,
        trusted_root,
        None,
        _phase15_local_limits(),
        ByteSize(10_000),
        _runtime_authority=authority,
        platform=(
            LocalPlatformProfile.DARWIN
            if runtime_platform == "darwin"
            else LocalPlatformProfile.LINUX
        ),
        mutation_test_profile=True,
        commit_namespace=namespace,
    )


def _phase15_revalidation_snapshot() -> RevalidationSnapshot:
    """Return the complete coordinator witness for the direct target run."""
    return RevalidationSnapshot(
        tuple(
            sorted(
                (
                    RevalidationFact(field, "phase15-" + field.value, "bound")
                    for field in RevalidationField
                ),
                key=lambda fact: (fact.field.value, fact.key, fact.value),
            )
        )
    )


def _example(name: str) -> bytes:
    """Return one named canonical JSON documentation example as UTF-8 bytes."""
    document = _DOCUMENTATION.read_text(encoding="utf-8")
    prefix = f"<!-- patch-example: {name} -->\n~~~json\n"
    start = document.index(prefix) + len(prefix)
    end = document.index("\n~~~", start)
    return document[start:end].encode("utf-8")


def _parse(operation: RawPatchInputKind, raw: bytes) -> None:
    """Parse one documentation example through the production JSON codec."""
    PatchRequestParser(PatchInputLimits()).parse(
        RawPatchIngress(
            RawProviderProfile("phase15-documentation"),
            RawToolCallId("phase15-documentation"),
            operation,
            RawPatchInputState.COMPLETE,
            raw,
        )
    )


def test_patch_phase_15_requirements() -> None:
    """Keep the documented Version 1 JSON examples executable."""
    document = _DOCUMENTATION.read_text(encoding="utf-8")
    for required in (
        "## Selection is not authority",
        "## Exact Version 1 schemas",
        "## Contexts and container limits",
        "## Explicit non-features",
        "kill switch",
        "automatic rebase",
    ):
        assert required in document
    _parse(RawPatchInputKind.EDIT_JSON, _example("valid-edit"))
    _parse(RawPatchInputKind.APPLY_JSON, _example("valid-apply"))
    with pytest.raises(PatchInputError):
        _parse(
            RawPatchInputKind.EDIT_JSON,
            _example("invalid-edit-empty-old-text"),
        )


def test_patch_phase_15_documentation_specification_heading() -> None:
    """Require documentation to state its specification role."""
    document = _DOCUMENTATION.read_text(encoding="utf-8")
    assert document.startswith("# Patch tools\n")
    assert "Selection is not authority" in document
    assert "does not grant mutation capability" in document
    assert "Exact Version 1 schemas" in document
    assert "Both JSON function schemas are closed" in document
    assert "A pending operation is not a terminal tool result" in document
    assert "Explicit non-features" in document
    assert ToolManager.__module__ == "avalan.tool.manager"


def test_patch_activation_manifest_is_source_derived_and_frozen() -> None:
    """Freeze generated schemas, protocols, and an incomplete profile."""
    manifest = build_patch_production_manifest()
    expected = loads(_FIXTURE.read_text(encoding="utf-8"))
    assert loads(render_patch_production_manifest(manifest)) == expected
    assert manifest.profiles[0].state is PatchProfileState.INCOMPLETE
    assert manifest.tool_inventory == ("patch.edit", "patch.apply")


def test_patch_e2e_037_rooted_foreign_writer_never_clobbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a real rooted local commit after a foreign writer changes it."""
    root = tmp_path / "selected-root"
    root.mkdir()
    note = root / "note.txt"
    note.write_bytes(b"before\n")
    outside_canary = tmp_path / "outside-canary.txt"
    outside_canary.write_bytes(b"outside remains private\n")
    assert outside_canary.read_bytes() == b"outside remains private\n"
    symlink = root / "outside-link.txt"
    symlink.symlink_to(outside_canary)
    ancestor = root / "outside-ancestor"
    ancestor.symlink_to(tmp_path, target_is_directory=True)
    privileged = root / "privileged.txt"
    privileged.write_bytes(b"metadata canary\n")
    privileged.chmod(0o4755)
    directory = root / "directory-canary"
    directory.mkdir()
    fifo = root / "special-fifo"
    mkfifo(fifo)
    profile = _phase15_local_profile(root, monkeypatch)

    async def scenario() -> None:
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        inspection = LocalInspectionTarget(profile)
        alias_profile = replace(profile, alias_mode=AliasMode.CASE_INSENSITIVE)
        alias_scope = await LocalScopeResolver(alias_profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        with pytest.raises(TargetInspectionError) as alias_error:
            await LocalInspectionTarget(alias_profile).inspect(
                InspectionRequest(
                    alias_scope,
                    (LogicalPath("note.txt"), LogicalPath("NOTE.TXT")),
                )
            )
        assert alias_error.value.code is TargetErrorCode.ALIAS_DENIED
        for denied_path, expected_codes in (
            (
                LogicalPath("outside-link.txt"),
                frozenset((TargetErrorCode.LINK_DENIED,)),
            ),
            (
                LogicalPath("outside-ancestor/outside-canary.txt"),
                frozenset((TargetErrorCode.LINK_DENIED,)),
            ),
            (
                LogicalPath("privileged.txt"),
                frozenset((TargetErrorCode.METADATA_DENIED,)),
            ),
            (
                LogicalPath("directory-canary"),
                frozenset((TargetErrorCode.SPECIAL_FILE_DENIED,)),
            ),
            (
                LogicalPath("special-fifo"),
                frozenset((TargetErrorCode.SPECIAL_FILE_DENIED,)),
            ),
        ):
            with pytest.raises(TargetInspectionError) as denial:
                await inspection.inspect(
                    InspectionRequest(scope, (denied_path,))
                )
            assert denial.value.code in expected_codes
            assert denied_path.value not in str(denial.value)
        eof_document = (
            b"*** Begin Patch v1\n"
            b"*** Update File: note.txt\n"
            b"@@\n"
            b"-before\n"
            b"\\ No newline at end of file\n"
            b"+after\n"
            b"\\ No newline at end of file\n"
            b"*** End of File\n"
            b"*** End Patch\n"
        )
        eof_request = PatchRequestParser(PatchInputLimits()).parse(
            RawPatchIngress(
                RawProviderProfile("phase15-local-provider"),
                RawToolCallId("phase15-eof-call"),
                RawPatchInputKind.VERIFIED_FREEFORM,
                RawPatchInputState.COMPLETE,
                eof_document,
            )
        )
        assert eof_request.operation is OperationType.APPLY
        for truncated_document in (
            eof_document.removesuffix(b"*** End Patch\n"),
            eof_document.replace(
                b"*** End of File\n", b"*** End of File\n+x\n"
            ),
        ):
            with pytest.raises(PatchInputError):
                PatchRequestParser(PatchInputLimits()).parse(
                    RawPatchIngress(
                        RawProviderProfile("phase15-local-provider"),
                        RawToolCallId("phase15-truncated-call"),
                        RawPatchInputKind.VERIFIED_FREEFORM,
                        RawPatchInputState.COMPLETE,
                        truncated_document,
                    )
                )
        worker = await target.worker(scope)
        inspected = await inspection.inspect(
            InspectionRequest(scope, (LogicalPath("note.txt"),))
        )
        hardlink = root / "note-hardlink.txt"
        hardlink.hardlink_to(note)
        with pytest.raises(TargetInspectionError) as hardlink_error:
            await inspection.inspect(
                InspectionRequest(scope, (LogicalPath("note-hardlink.txt"),))
            )
        assert hardlink_error.value.code is TargetErrorCode.HARDLINK_DENIED
        hardlink.unlink()
        request = PatchRequestParser(PatchInputLimits()).parse(
            RawPatchIngress(
                RawProviderProfile("phase15-local-provider"),
                RawToolCallId("phase15-local-call"),
                RawPatchInputKind.EDIT_JSON,
                RawPatchInputState.COMPLETE,
                b'{"path":"note.txt","edits":[{"old_text":"before\\n",'
                b'"new_text":"after\\n"}]}',
            )
        )
        with pytest.raises(PatchInputError):
            PatchRequestParser(PatchInputLimits()).parse(
                RawPatchIngress(
                    RawProviderProfile("phase15-local-provider"),
                    RawToolCallId("phase15-outside-call"),
                    RawPatchInputKind.EDIT_JSON,
                    RawPatchInputState.COMPLETE,
                    b'{"path":"../outside-canary.txt","edits":[]}',
                )
            )
        for index, unsafe_path in enumerate(
            (
                "/outside-canary.txt",
                "C:/outside-canary.txt",
                "//outside-canary.txt",
                "../../outside-canary.txt",
                "note.txt/../outside-canary.txt",
                "~/.private",
            )
        ):
            with pytest.raises(PatchInputError):
                PatchRequestParser(PatchInputLimits()).parse(
                    RawPatchIngress(
                        RawProviderProfile("phase15-local-provider"),
                        RawToolCallId("phase15-unsafe-" + str(index)),
                        RawPatchInputKind.EDIT_JSON,
                        RawPatchInputState.COMPLETE,
                        (
                            b'{"path":"'
                            + unsafe_path.encode("ascii")
                            + b'","edits":[{"old_text":"before",'
                            b'"new_text":"after"}]}'
                        ),
                    )
                )
        candidate = plan(request, inspected.planner_workspace())
        limits = _phase15_local_limits()
        reader = PreauthorizationClass("phase15-local-reader")
        requirements = ApprovalRequirements(
            ApprovalMode.PREAUTHORIZED,
            PolicyRouteId("route-phase15"),
            PolicyBrokerId("broker-phase15"),
            PolicyReviewerRole("reviewer-phase15"),
            1,
            reader,
        )
        policy = TrustedPatchPolicy(
            PolicyRevision("policy-phase15"),
            frozenset((OperationType.EDIT,)),
            (
                PolicyRule(
                    PolicyPathSelector(None),
                    tuple(
                        CapabilityMode(
                            capability,
                            ApprovalMode.PREAUTHORIZED,
                            reader,
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
            limits,
            requirements,
        )
        authorizer = PolicyAuthorizer(policy)
        paths = (LogicalPath("note.txt"),)
        effects = frozenset(
            capability
            for lineage in candidate.lineages
            for capability in lineage.capabilities
        )
        preflight = await authorizer.authorize_preinspection(
            PreflightRequest(
                OperationType.EDIT,
                paths,
                effects,
                frozenset(paths),
                compose_limits(limits, limits, limits, limits, limits),
            )
        )
        final = await authorizer.authorize_final(
            preflight, candidate, await target.handshake(scope)
        )
        subject = ExecutionSubject(
            PatchPrincipalId("principal-phase15"),
            PatchTenantId("tenant-phase15"),
            PatchRunId("run-phase15"),
            PatchSessionId("session-phase15"),
            PatchTaskId("task-phase15"),
            PatchAgentId("agent-phase15"),
        )
        sealed = seal_plan(
            PatchPlanId("plan_" + "f" * 16),
            PlanBinding(
                PatchRequest(
                    1,
                    PatchRequestId("request_" + "f" * 16),
                    PatchExecutionId("execution_" + "f" * 16),
                    OperationType.EDIT,
                    PatchInput(b"phase15-local"),
                    paths,
                ),
                candidate.request_digest,
                subject,
                ContextKind.LOCAL,
                profile.identity,
                None,
                preflight,
                final,
            ),
            candidate,
            ExpiryTick(100),
        )
        approvals = ApprovalService(
            _Phase15LocalBroker(),
            _Phase15LocalClock(),
            RuntimeGrantStore(),
        )
        decision = await approvals.await_review(
            PlanReviewRequest(sealed, subject, final.approval)
        )
        assert decision.grant is not None
        coordinator_store = InMemoryCoordinatorStore(approvals)
        coordinator = InMemoryPatchCoordinator(
            coordinator_store,
            InMemoryLeaseManager(coordinator_store),
            ScriptedReconciler(_phase15_revalidation_snapshot()),
        )
        reservation = await coordinator.reserve(
            RuntimeIdentity(
                subject,
                final.approval.route,
                RetransmissionKey("phase15-rooted-foreign-writer"),
            ),
            candidate.request_digest,
        )
        destination = note
        final_revalidation_reached = Event()
        foreign_writer_completed = Event()
        original_barrier = local_commit_module._commit_barrier

        def interleave_foreign_writer(stage: str) -> None:
            """Hold the authenticated final revalidation before replacement."""
            if stage == "target.namespace_after_final_check":
                final_revalidation_reached.set()
                assert foreign_writer_completed.wait(2)
            original_barrier(stage)

        monkeypatch.setattr(
            local_commit_module, "_commit_barrier", interleave_foreign_writer
        )
        execution = create_task(
            coordinator.execute(
                reservation,
                sealed,
                decision.grant,
                _phase15_revalidation_snapshot(),
                worker,
                "phase15-rooted-controller",
            )
        )
        while not final_revalidation_reached.is_set():
            if execution.done():
                completed = await execution
                pytest.fail(
                    "local worker completed before the final "
                    "revalidation barrier: " + str(completed)
                )
            await sleep(0)
        foreign_writer = await create_subprocess_exec(
            executable,
            "-c",
            "from os import replace; from pathlib import Path; "
            + "target = Path("
            + repr(str(destination))
            + "); stage = target.with_name('.foreign-writer'); "
            + (
                "stage.write_bytes(b'foreign writer\\n'); "
                "replace(stage, target)"
            ),
        )
        assert await foreign_writer.wait() == 0
        foreign_writer_completed.set()
        result = await execution
        if not isinstance(result, PatchResult):
            result = await coordinator.execute(
                reservation,
                sealed,
                decision.grant,
                _phase15_revalidation_snapshot(),
                worker,
                "phase15-rooted-controller",
            )
        assert isinstance(result, PatchResult)
        assert result.status is PatchStatus.COMMIT_FAILED
        assert destination.read_bytes() == b"foreign writer\n"
        assert outside_canary.read_bytes() == b"outside remains private\n"
        assert not tuple(root.glob(".avalan-patch-*"))
        assert profile.commit_namespace is not None
        assert not tuple(profile.commit_namespace.iterdir())

    run(scenario())


def test_patch_phase_15_default_incomplete_profile_advertises_no_tools() -> (
    None
):
    """Keep production profile selection and activation authority absent."""

    async def inspect() -> tuple[tuple[str, ...], int]:
        manifest = build_patch_production_manifest()
        registry = PatchActivationRegistry(
            manifest,
            _build_activation_verifier(
                manifest,
                _new_activation_authority(b"q" * 32),
                production=False,
            ),
        )
        profile = manifest.profiles[0]
        return (
            await registry.advertised_tools(profile.key),
            await registry.active_binding_count(profile.key),
        )

    assert run(inspect()) == ((), 0)
    assert OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID


def test_patch_phase_15_default_factory_is_inert_without_proof() -> (  # noqa: E501
    None
):
    """Keep the production loader factory inert without a proven profile."""

    async def inspect() -> None:
        factory = build_patch_runtime_activation_factory()
        store = InMemoryDurablePatchStore(InMemoryDurablePatchBackend())
        assert (
            await factory.activate(_runtime_binding(store, _ObservedService()))
            is None
        )

    run(inspect())
    assert OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID


def test_patch_activation_fails_closed_and_deactivation_keeps_owner() -> None:
    """Block reactivation until the exact retained owner has settled."""

    async def scenario() -> (
        tuple[PatchActivationRuntime, PatchActivationRuntime]
    ):
        factory = patch_test_activation_factory()
        assert isinstance(factory, PatchActivationRuntimeFactory)
        store = InMemoryDurablePatchStore(InMemoryDurablePatchBackend())
        first_service = _ObservedService()
        runtime = await factory.activate(
            _runtime_binding(store, first_service)
        )
        assert runtime is not None
        lease = _lease(PatchRequestId.new(), 1)
        await runtime.bind_durable_commit(lease)
        await runtime.deactivate()
        assert (
            await factory.activate(_runtime_binding(store, _ObservedService()))
            is None
        )
        await runtime.release_durable_commit(lease)
        await runtime.release_durable_commit(lease)
        reactivated = await factory.activate(
            _runtime_binding(store, _ObservedService())
        )
        assert reactivated is not None
        await reactivated.deactivate()
        return runtime, reactivated

    runtime, reactivated = run(scenario())
    assert runtime.lease.key == reactivated.lease.key
    assert reactivated.lease.epoch == runtime.lease.epoch + 1


def test_patch_phase_15_documentation_exposes_only_unproven_profiles() -> None:
    """Document selection state without granting activation authority."""
    document = _DOCUMENTATION.read_text(encoding="utf-8")
    manifest = build_patch_production_manifest()
    assert "Selection is not authority" in document
    assert manifest.profiles[0].state is PatchProfileState.INCOMPLETE
    assert not manifest.profiles[0].proven


def test_patch_phase_15_documentation_replays_schema_examples() -> None:
    """Replay every documented schema example without creating a receipt."""
    _parse(RawPatchInputKind.EDIT_JSON, _example("valid-edit"))
    _parse(RawPatchInputKind.APPLY_JSON, _example("valid-apply"))
    with pytest.raises(PatchInputError):
        _parse(
            RawPatchInputKind.EDIT_JSON,
            _example("invalid-edit-empty-old-text"),
        )
