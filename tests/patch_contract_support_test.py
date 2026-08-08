"""Exercise deterministic in-memory support for dormant patch contracts."""

from asyncio import create_task
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from sys import modules
from sys import path as sys_path
from types import ModuleType
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase

if TYPE_CHECKING:
    from patch_contract_support import (
        ApprovalBinding,
        ApprovalDecision,
        FaultLabel,
        ScriptedApprovalBroker,
        WorkspaceEntry,
        WorkspaceOracle,
    )

_ROOT = Path(__file__).resolve().parents[1]


def _load_support() -> ModuleType:
    """Load the standalone scripted patch-test support module."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    name = "_patch_contract_support"
    spec = spec_from_file_location(
        name,
        _ROOT / "scripts" / "patch_contract_support.py",
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_SUPPORT = _load_support()


def _directory(
    name: str,
    identity: str,
    children: tuple["WorkspaceEntry", ...],
) -> "WorkspaceEntry":
    """Return one scripted directory entry with sorted child names."""
    entry: WorkspaceEntry = _SUPPORT.WorkspaceEntry(
        name=name,
        entry_type=_SUPPORT.WorkspaceEntryType.DIRECTORY,
        content=b"",
        symlink_target=None,
        link_count=1,
        identity=_SUPPORT.PatchWorkspaceEntryId(identity),
        mode=0o755,
        children=children,
    )
    return entry


def _workspace_oracle(
    *,
    content: bytes = b"before",
    symlink_target: str = "visible.txt",
    link_count: int = 1,
    identity: str = "entry-visible",
    mode: int = 0o644,
    metadata_value: str = "user:rw",
    canary_content: bytes = b"outside",
    artifact_content: bytes = b"staged",
) -> "WorkspaceOracle":
    """Return one complete recursive workspace-oracle fixture."""
    notes = _SUPPORT.WorkspaceEntry(
        name="notes.txt",
        entry_type=_SUPPORT.WorkspaceEntryType.FILE,
        content=b"notes",
        symlink_target=None,
        link_count=1,
        identity=_SUPPORT.PatchWorkspaceEntryId("entry-notes"),
        mode=0o600,
    )
    docs = _directory("docs", "entry-docs", (notes,))
    link = _SUPPORT.WorkspaceEntry(
        name="link",
        entry_type=_SUPPORT.WorkspaceEntryType.SYMLINK,
        content=b"",
        symlink_target=symlink_target,
        link_count=1,
        identity=_SUPPORT.PatchWorkspaceEntryId("entry-link"),
        mode=0o777,
    )
    visible = _SUPPORT.WorkspaceEntry(
        name="visible.txt",
        entry_type=_SUPPORT.WorkspaceEntryType.FILE,
        content=content,
        symlink_target=None,
        link_count=link_count,
        identity=_SUPPORT.PatchWorkspaceEntryId(identity),
        mode=mode,
        security_metadata=(
            _SUPPORT.WorkspaceSecurityMetadata(
                name="acl",
                value=metadata_value,
            ),
        ),
    )
    canary = _SUPPORT.WorkspaceEntry(
        name="outside.txt",
        entry_type=_SUPPORT.WorkspaceEntryType.FILE,
        content=canary_content,
        symlink_target=None,
        link_count=1,
        identity=_SUPPORT.PatchWorkspaceEntryId("canary-outside"),
        mode=0o600,
    )
    artifact = _SUPPORT.WorkspaceEntry(
        name="receipt.txt",
        entry_type=_SUPPORT.WorkspaceEntryType.FILE,
        content=artifact_content,
        symlink_target=None,
        link_count=1,
        identity=_SUPPORT.PatchWorkspaceEntryId("artifact-receipt"),
        mode=0o600,
    )
    oracle: WorkspaceOracle = _SUPPORT.WorkspaceOracle(
        root=_directory("", "entry-root", (docs, link, visible)),
        outside_root_canaries=(canary,),
        artifact_namespaces=(
            _SUPPORT.ArtifactNamespace(
                name="staging",
                root=_directory("", "artifact-root", (artifact,)),
            ),
        ),
    )
    return oracle


def _approval_binding() -> "ApprovalBinding":
    """Return one complete exact approval-binding fixture."""
    binding: ApprovalBinding = _SUPPORT.ApprovalBinding(
        plan_id=_SUPPORT.PatchPlanId("plan-1"),
        principal_id=_SUPPORT.PatchPrincipalId("principal-1"),
        tenant_id=_SUPPORT.PatchTenantId("tenant-1"),
        run_id=_SUPPORT.PatchRunId("run-1"),
        context_id=_SUPPORT.PatchContextId("context-1"),
        workspace_id=_SUPPORT.PatchWorkspaceId("workspace-1"),
        policy_id=_SUPPORT.PatchPolicyId("policy-1"),
        broker_id=_SUPPORT.PatchBrokerId("broker-1"),
        quorum=2,
    )
    return binding


def _approval_broker(
    binding: "ApprovalBinding",
    *,
    decision: "ApprovalDecision",
    expires_tick: int = 9,
    delay_label: "FaultLabel | None" = None,
) -> "ScriptedApprovalBroker":
    """Return one immutable scripted approval broker fixture."""
    grant = None
    if decision is _SUPPORT.ApprovalDecision.APPROVE:
        grant = _SUPPORT.PatchApprovalGrant(
            identifier=_SUPPORT.PatchGrantId("grant-1"),
            binding=binding,
            issued_tick=2,
            expires_tick=expires_tick,
        )
    broker: ScriptedApprovalBroker = _SUPPORT.ScriptedApprovalBroker(
        binding=binding,
        decision=decision,
        grant=grant,
        delay_label=delay_label,
    )
    return broker


class PatchContractSupportTest(IsolatedAsyncioTestCase):
    """Exercise deterministic script-only dormant patch collaborators."""

    async def test_reproducibly_records_boundaries(self) -> None:
        """Record every scripted target boundary without workspace access."""
        clock = _SUPPORT.ManualClock(tick=4)
        self.assertEqual(clock.advance(3).tick, 7)
        assert clock.advance(3).tick == 7

        factories = _SUPPORT.PatchFactories.create()
        issued = tuple(
            factory.issue()[0]
            for factory in (
                factories.requests,
                factories.calls,
                factories.plans,
                factories.approvals,
                factories.operations,
                factories.lineages,
                factories.steps,
                factories.domains,
                factories.contexts,
                factories.workspaces,
                factories.events,
                factories.observers,
                factories.digest_inputs,
                factories.grants,
                factories.leases,
                factories.fences,
                factories.correlations,
            )
        )
        self.assertEqual(
            issued,
            (
                "request-0000",
                "call-0000",
                "plan-0000",
                "approval-0000",
                "operation-0000",
                "lineage-0000",
                "step-0000",
                "domain-0000",
                "context-0000",
                "workspace-0000",
                "event-0000",
                "observer-0000",
                "digest-input-0000",
                "grant-0000",
                "lease-0000",
                "fence-0000",
                "correlation-0000",
            ),
        )
        _, advanced_requests = factories.requests.issue()
        self.assertEqual(advanced_requests.issue()[0], "request-0001")

        controller = _SUPPORT.FaultController.create()
        label = _SUPPORT.FaultLabel.COMMIT_BEFORE
        arrived = create_task(controller.arrive(label))
        await controller.wait_until_entered(label)
        self.assertFalse(arrived.done())
        controller.release(label)
        await arrived

        target = _SUPPORT.ScriptedMutationTarget(
            capabilities=(
                _SUPPORT.PatchCapability("commit"),
                _SUPPORT.PatchCapability("inspect"),
            )
        )
        capabilities, target = await target.negotiate_capabilities()
        self.assertEqual(capabilities, ("commit", "inspect"))
        _, target = await target.inspect(_SUPPORT.PatchPath("visible.txt"))
        _, target = await target.observe_precondition(
            _SUPPORT.PatchPath("visible.txt")
        )
        _, target = await target.open_handle(
            _SUPPORT.PatchHandleId("handle-1")
        )
        _, target = await target.acquire_lock(_SUPPORT.PatchLockId("lock-1"))
        _, target = await target.stage_artifact(
            _SUPPORT.PatchStagingArtifactId("staging-1")
        )
        _, target = await target.commit_step(_SUPPORT.PatchStepId("step-1"))
        _, target = await target.verify(_SUPPORT.PatchPath("visible.txt"))
        _, target = await target.clean_staging_artifact(
            _SUPPORT.PatchStagingArtifactId("staging-1")
        )
        _, target = await target.release_lock(_SUPPORT.PatchLockId("lock-1"))
        _, target = await target.close_handle(
            _SUPPORT.PatchHandleId("handle-1")
        )

        self.assertEqual(
            tuple(record.action for record in target.trace),
            (
                "negotiate_capabilities",
                "inspect",
                "observe_precondition",
                "open_handle",
                "acquire_lock",
                "stage_artifact",
                "commit_step",
                "verify",
                "clean_staging_artifact",
                "release_lock",
                "close_handle",
            ),
        )
        self.assertFalse(
            any(record.workspace_namespace_mutation for record in target.trace)
        )
        self.assertEqual(
            tuple(record.await_receipt.boundary for record in target.trace),
            (
                _SUPPORT.AwaitBoundary.TARGET_NEGOTIATION,
                _SUPPORT.AwaitBoundary.TARGET_INSPECTION,
                _SUPPORT.AwaitBoundary.TARGET_PRECONDITION,
                _SUPPORT.AwaitBoundary.TARGET_HANDLE_OPEN,
                _SUPPORT.AwaitBoundary.TARGET_LOCK_ACQUIRE,
                _SUPPORT.AwaitBoundary.TARGET_STAGE,
                _SUPPORT.AwaitBoundary.TARGET_COMMIT,
                _SUPPORT.AwaitBoundary.TARGET_VERIFICATION,
                _SUPPORT.AwaitBoundary.TARGET_CLEANUP,
                _SUPPORT.AwaitBoundary.TARGET_LOCK_RELEASE,
                _SUPPORT.AwaitBoundary.TARGET_HANDLE_CLOSE,
            ),
        )
        self.assertEqual(
            target.trace[6].await_receipt.depths,
            _SUPPORT.ResourceDepths(
                coordinator_lease=1,
                target_handle=1,
                target_worker=1,
                staging_resource=1,
            ),
        )
        self.assertEqual(
            tuple(record.await_receipt.depths for record in target.trace),
            (
                _SUPPORT.ResourceDepths(),
                _SUPPORT.ResourceDepths(),
                _SUPPORT.ResourceDepths(),
                _SUPPORT.ResourceDepths(),
                _SUPPORT.ResourceDepths(target_handle=1),
                _SUPPORT.ResourceDepths(
                    coordinator_lease=1,
                    target_handle=1,
                ),
                _SUPPORT.ResourceDepths(
                    coordinator_lease=1,
                    target_handle=1,
                    target_worker=1,
                    staging_resource=1,
                ),
                _SUPPORT.ResourceDepths(
                    coordinator_lease=1,
                    target_handle=1,
                    staging_resource=1,
                ),
                _SUPPORT.ResourceDepths(
                    coordinator_lease=1,
                    target_handle=1,
                    staging_resource=1,
                ),
                _SUPPORT.ResourceDepths(
                    coordinator_lease=1,
                    target_handle=1,
                ),
                _SUPPORT.ResourceDepths(target_handle=1),
            ),
        )

    async def test_target_rejects_retained_owner_at_its_fault_wait(
        self,
    ) -> None:
        """Reject a live target handle at the target's own fault wait."""
        target = _SUPPORT.ScriptedMutationTarget(
            faults=_SUPPORT.FaultController.create(),
            fault_label=_SUPPORT.FaultLabel.TARGET_BEFORE,
        )
        _, target = await target.open_handle(
            _SUPPORT.PatchHandleId("handle-retained")
        )

        with self.assertRaisesRegex(
            _SUPPORT.AllowedAwaitViolation,
            "boundary=fault_wait owners=target_handle:1",
        ):
            await target.close_handle(
                _SUPPORT.PatchHandleId("handle-retained")
            )

    async def test_rejects_ambiguous_json_and_labels(self) -> None:
        """Reject duplicate JSON names and incomplete frozen fault labels."""
        self.assertEqual(
            _SUPPORT.load_strict_json('{"outer":{"value":1}}'),
            {"outer": {"value": 1}},
        )
        assert _SUPPORT.load_strict_json('{"outer":{"value":1}}') == {
            "outer": {"value": 1}
        }
        with self.assertRaisesRegex(ValueError, "duplicate JSON object name"):
            _SUPPORT.load_strict_json('{"outer":{"value":1,"value":2}}')

        with self.assertRaises(AssertionError):
            _SUPPORT.FaultController.create(
                (_SUPPORT.FaultLabel.LIFECYCLE_BEFORE,)
            )
        with self.assertRaises(AssertionError):
            _SUPPORT.FaultController.create(
                (
                    _SUPPORT.FaultLabel.LIFECYCLE_BEFORE,
                    _SUPPORT.FaultLabel.LIFECYCLE_BEFORE,
                )
            )

        before = _workspace_oracle()
        (
            _,
            mutated_target,
        ) = await _SUPPORT.ScriptedMutationTarget().record_namespace_mutation(
            _SUPPORT.PatchPath("visible.txt")
        )
        self.assertFalse(
            _SUPPORT.has_zero_write_evidence(
                before,
                before,
                mutated_target,
            )
        )

    async def test_workspace_oracle_captures_recursive_zero_write_facts(
        self,
    ) -> None:
        """Capture every protected tree fact and both zero-write predicates."""
        before = _workspace_oracle()
        same = _workspace_oracle()
        changed_oracles = (
            _workspace_oracle(content=b"after"),
            _workspace_oracle(symlink_target="docs/notes.txt"),
            _workspace_oracle(link_count=2),
            _workspace_oracle(identity="entry-visible-replaced"),
            _workspace_oracle(mode=0o600),
            _workspace_oracle(metadata_value="user:r"),
            _workspace_oracle(canary_content=b"canary-changed"),
            _workspace_oracle(artifact_content=b"artifact-changed"),
        )

        self.assertTrue(before.equals(same))
        assert before.equals(same)
        self.assertEqual(before.root.children[0].children[0].content, b"notes")
        self.assertTrue(
            all(
                before.digest() != changed.digest()
                for changed in changed_oracles
            )
        )
        self.assertTrue(
            _SUPPORT.has_zero_write_evidence(
                before,
                same,
                _SUPPORT.ScriptedMutationTarget(),
            )
        )
        self.assertFalse(
            _SUPPORT.has_zero_write_evidence(
                before,
                changed_oracles[0],
                _SUPPORT.ScriptedMutationTarget(),
            )
        )
        (
            _,
            transient_target,
        ) = await _SUPPORT.ScriptedMutationTarget().record_namespace_mutation(
            _SUPPORT.PatchPath("visible.txt")
        )
        self.assertFalse(
            _SUPPORT.has_zero_write_evidence(
                before,
                same,
                transient_target,
            )
        )

        with self.assertRaises(AssertionError):
            _directory(
                "",
                "entry-root",
                (
                    _SUPPORT.WorkspaceEntry(
                        name="z",
                        entry_type=_SUPPORT.WorkspaceEntryType.FILE,
                        content=b"",
                        symlink_target=None,
                        link_count=1,
                        identity=_SUPPORT.PatchWorkspaceEntryId("entry-z"),
                        mode=0o600,
                    ),
                    _SUPPORT.WorkspaceEntry(
                        name="a",
                        entry_type=_SUPPORT.WorkspaceEntryType.FILE,
                        content=b"",
                        symlink_target=None,
                        link_count=1,
                        identity=_SUPPORT.PatchWorkspaceEntryId("entry-a"),
                        mode=0o600,
                    ),
                ),
            )

    async def test_scripted_approval_broker_enforces_bindings_and_grants(
        self,
    ) -> None:
        """Approve only exact unexpired bindings and consume grants once."""
        binding = _approval_binding()
        clock = _SUPPORT.ManualClock(tick=3)
        broker = _approval_broker(
            binding,
            decision=_SUPPORT.ApprovalDecision.APPROVE,
        )
        outcome, reviewed_broker = await broker.decide(binding, clock)
        self.assertEqual(outcome.kind, _SUPPORT.ApprovalOutcomeKind.APPROVED)
        assert outcome.grant is not None
        self.assertEqual(reviewed_broker.calls, (binding,))

        consumed, consumed_broker = await reviewed_broker.consume(
            outcome.grant,
            binding,
            clock,
        )
        self.assertEqual(consumed.kind, _SUPPORT.ApprovalOutcomeKind.APPROVED)
        replayed, _ = await consumed_broker.consume(
            outcome.grant,
            binding,
            clock,
        )
        self.assertEqual(replayed.kind, _SUPPORT.ApprovalOutcomeKind.REPLAYED)

        (
            consumptions,
            arbitrated_broker,
        ) = await reviewed_broker.consume_concurrently(
            outcome.grant,
            binding,
            clock,
            (
                _SUPPORT.PatchObserverId("observer-a"),
                _SUPPORT.PatchObserverId("observer-b"),
            ),
        )
        self.assertEqual(
            tuple(item.outcome.kind for item in consumptions),
            (
                _SUPPORT.ApprovalOutcomeKind.APPROVED,
                _SUPPORT.ApprovalOutcomeKind.REPLAYED,
            ),
        )
        self.assertEqual(arbitrated_broker.consumed_grants, ("grant-1",))

        mismatch_cases = (
            (
                replace(binding, plan_id=_SUPPORT.PatchPlanId("plan-2")),
                "PLAN",
            ),
            (
                replace(
                    binding,
                    principal_id=_SUPPORT.PatchPrincipalId("principal-2"),
                ),
                "PRINCIPAL",
            ),
            (
                replace(
                    binding,
                    tenant_id=_SUPPORT.PatchTenantId("tenant-2"),
                ),
                "TENANT",
            ),
            (
                replace(binding, run_id=_SUPPORT.PatchRunId("run-2")),
                "RUN",
            ),
            (
                replace(
                    binding,
                    context_id=_SUPPORT.PatchContextId("context-2"),
                ),
                "CONTEXT",
            ),
            (
                replace(
                    binding,
                    workspace_id=_SUPPORT.PatchWorkspaceId("workspace-2"),
                ),
                "WORKSPACE",
            ),
            (
                replace(
                    binding,
                    policy_id=_SUPPORT.PatchPolicyId("policy-2"),
                ),
                "POLICY",
            ),
            (
                replace(
                    binding,
                    broker_id=_SUPPORT.PatchBrokerId("broker-2"),
                ),
                "BROKER",
            ),
            (replace(binding, quorum=3), "QUORUM"),
        )
        for mismatched_binding, expected_mismatch in mismatch_cases:
            rejected, _ = await broker.decide(
                mismatched_binding,
                clock,
            )
            self.assertEqual(
                rejected.kind,
                _SUPPORT.ApprovalOutcomeKind.BINDING_MISMATCH,
            )
            assert rejected.mismatch is not None
            self.assertEqual(rejected.mismatch.name, expected_mismatch)

        for decision, expected_kind in (
            (
                _SUPPORT.ApprovalDecision.DENY,
                _SUPPORT.ApprovalOutcomeKind.DENIED,
            ),
            (
                _SUPPORT.ApprovalDecision.UNAVAILABLE,
                _SUPPORT.ApprovalOutcomeKind.UNAVAILABLE,
            ),
        ):
            decided, _ = await _approval_broker(
                binding,
                decision=decision,
            ).decide(binding, clock)
            self.assertEqual(decided.kind, expected_kind)

        expired, _ = await _approval_broker(
            binding,
            decision=_SUPPORT.ApprovalDecision.APPROVE,
            expires_tick=3,
        ).decide(binding, clock)
        self.assertEqual(expired.kind, _SUPPORT.ApprovalOutcomeKind.EXPIRED)

        controller = _SUPPORT.FaultController.create()
        delayed_broker = _approval_broker(
            binding,
            decision=_SUPPORT.ApprovalDecision.APPROVE,
            delay_label=_SUPPORT.FaultLabel.APPROVAL_BEFORE,
        )
        delayed = create_task(
            delayed_broker.decide(binding, clock, faults=controller)
        )
        await controller.wait_until_entered(
            _SUPPORT.FaultLabel.APPROVAL_BEFORE
        )
        self.assertFalse(delayed.done())
        controller.release(_SUPPORT.FaultLabel.APPROVAL_BEFORE)
        delayed_outcome, _ = await delayed
        self.assertEqual(
            delayed_outcome.kind,
            _SUPPORT.ApprovalOutcomeKind.APPROVED,
        )
        self.assertEqual(
            reviewed_broker.await_receipts[0].boundary,
            _SUPPORT.AwaitBoundary.APPROVAL_DECISION,
        )
        self.assertEqual(
            consumed_broker.await_receipts[-1].boundary,
            _SUPPORT.AwaitBoundary.APPROVAL_CONSUME,
        )

    async def test_store_and_target_conformance_definitions(self) -> None:
        """Define backend-neutral storage and target-profile conformance."""
        suite = _SUPPORT.StoreConformanceSuite.create()
        self.assertEqual(
            suite.backends,
            (
                _SUPPORT.StoreBackend.IN_MEMORY,
                _SUPPORT.StoreBackend.POSTGRESQL,
            ),
        )
        self.assertEqual(
            tuple(case.identifier for case in suite.cases),
            (
                "close_owned_boundary",
                "compare_and_set_conflict",
                "compare_and_set_success",
                "create_absent_record",
                "read_absent_record",
            ),
        )
        assert suite.cases[-1].expected_result is None

        profiles = (
            _SUPPORT.TargetConformanceProfile(
                kind=_SUPPORT.TargetProfileKind.SCRIPTED,
                capable=True,
                required_capabilities=(_SUPPORT.PatchCapability("inspect"),),
            ),
            _SUPPORT.TargetConformanceProfile(
                kind=_SUPPORT.TargetProfileKind.LOCAL,
                capable=False,
            ),
            _SUPPORT.TargetConformanceProfile(
                kind=_SUPPORT.TargetProfileKind.SANDBOX,
                capable=False,
            ),
            _SUPPORT.TargetConformanceProfile(
                kind=_SUPPORT.TargetProfileKind.CONTAINER,
                capable=False,
            ),
        )
        runner = _SUPPORT.TargetFactoryConformanceRunner(profiles=profiles)

        class CompleteFactory:
            """Create only the one target profile that declares capability."""

            async def create(self, profile: object) -> object:
                """Return one scripted target only for a capable profile."""
                assert isinstance(profile, _SUPPORT.TargetConformanceProfile)
                if not profile.capable:
                    return None
                return _SUPPORT.ScriptedMutationTarget(
                    capabilities=profile.required_capabilities
                )

        class IncompleteFactory:
            """Incorrectly create a target for every profile state."""

            async def create(self, profile: object) -> object:
                """Return an invalid target for one incapable profile."""
                assert isinstance(profile, _SUPPORT.TargetConformanceProfile)
                return _SUPPORT.ScriptedMutationTarget()

        results = await runner.run(CompleteFactory())
        self.assertEqual(
            tuple(result.capable for result in results),
            (True, False, False, False),
        )
        self.assertEqual(results[0].capabilities, ("inspect",))
        self.assertEqual(
            tuple(receipt.boundary for receipt in results[0].await_receipts),
            (
                _SUPPORT.AwaitBoundary.TARGET_FACTORY_CREATE,
                _SUPPORT.AwaitBoundary.TARGET_FACTORY_NEGOTIATE,
            ),
        )
        with self.assertRaises(AssertionError):
            await runner.run(IncompleteFactory())
        with self.assertRaises(AssertionError):
            _SUPPORT.TargetConformanceProfile(
                kind=_SUPPORT.TargetProfileKind.LOCAL,
                capable=False,
                required_capabilities=(_SUPPORT.PatchCapability("inspect"),),
            )

    async def test_ipc_crash_harness_uses_explicit_barriers(self) -> None:
        """Prove child outcomes through pipe barriers, not timing guesses."""
        released = _SUPPORT.ChildProcessCrashHarness.start().release()
        self.assertEqual(released.exit_code, 0)
        self.assertEqual(released.stderr, b"")

        crashed = _SUPPORT.ChildProcessCrashHarness.start().trigger_crash()
        self.assertEqual(crashed.barrier, "barrier-ready")
        self.assertEqual(crashed.exit_code, 17)
        assert crashed.stderr == b""

    async def test_phase_evidence_codec_and_resource_matrix(self) -> None:
        """Seal canonical review evidence and reject unlisted ownership."""
        evidence = _SUPPORT.PhaseEvidence(
            phase=0,
            status=_SUPPORT.PhaseEvidenceStatus.COMPLETE,
            active_node_ids=(
                "tests/patch_contract_support_test.py::PatchContractSupportTest::test_phase_evidence_codec_and_resource_matrix",
            ),
            commands=(
                _SUPPORT.PhaseCommandEvidence(
                    command=(
                        "poetry run pytest -q "
                        "tests/patch_contract_support_test.py"
                    ),
                    exit_code=0,
                ),
            ),
            artifact_digests=(
                _SUPPORT.ArtifactDigest(
                    name="support-module",
                    sha256=_SUPPORT.PatchArtifactDigest("a" * 64),
                ),
            ),
            reviewer_findings=(
                _SUPPORT.ReviewerFinding(
                    identifier="finding-1",
                    severity=_SUPPORT.ReviewerSeverity.P3,
                    disposition=_SUPPORT.ReviewerDisposition.OPEN,
                    rationale="Documented for a later phase.",
                ),
            ),
        )
        sealed = _SUPPORT.PhaseEvidenceCodec.seal(
            evidence,
            _SUPPORT.PatchObserverId("reviewer-1"),
        )
        self.assertTrue(_SUPPORT.PhaseEvidenceCodec.verify(sealed))
        assert _SUPPORT.PhaseEvidenceCodec.verify(sealed)
        self.assertFalse(
            _SUPPORT.PhaseEvidenceCodec.verify(
                replace(
                    sealed,
                    canonical_digest=_SUPPORT.PatchArtifactDigest("0" * 64),
                    signing=replace(
                        sealed.signing,
                        signed_digest=_SUPPORT.PatchArtifactDigest("0" * 64),
                    ),
                )
            )
        )
        with self.assertRaises(AssertionError):
            _SUPPORT.PhaseEvidence(
                phase=0,
                status=_SUPPORT.PhaseEvidenceStatus.COMPLETE,
                active_node_ids=("test",),
                commands=(
                    _SUPPORT.PhaseCommandEvidence(command="test", exit_code=0),
                ),
                artifact_digests=(),
                reviewer_findings=(
                    _SUPPORT.ReviewerFinding(
                        identifier="blocking",
                        severity=_SUPPORT.ReviewerSeverity.P1,
                        disposition=_SUPPORT.ReviewerDisposition.OPEN,
                        rationale="Blocking issue remains.",
                    ),
                ),
            )

        sentinel = _SUPPORT.ResourceDepthSentinel()
        for boundary in _SUPPORT.AwaitBoundary:
            allowed = _SUPPORT._allowed_await_depths(boundary)
            checked = await _SUPPORT.ResourceDepthSentinel(
                depths=allowed
            ).at_await(boundary)
            self.assertEqual(checked.receipts[-1].boundary, boundary)
            for owner in _SUPPORT.ResourceOwner:
                if getattr(allowed, owner.value) == 0:
                    forbidden = allowed.acquire(owner)
                else:
                    forbidden = allowed.release(owner)
                with self.assertRaisesRegex(
                    _SUPPORT.AllowedAwaitViolation,
                    f"boundary={boundary.value} owners=",
                ):
                    await _SUPPORT.ResourceDepthSentinel(
                        depths=forbidden
                    ).at_await(boundary)
        with self.assertRaises(AssertionError):
            sentinel.release(_SUPPORT.ResourceOwner.APPROVAL_WAIT)
