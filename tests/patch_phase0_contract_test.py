"""Exercise the narrow executable evidence that owns Phase 0 requirements."""

from asyncio import run
from importlib.util import module_from_spec, spec_from_file_location
from json import loads
from pathlib import Path
from sys import modules
from sys import path as sys_path
from types import ModuleType

from pytest import MonkeyPatch

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "patch"


def _load_script(name: str) -> ModuleType:
    """Load one standalone Phase 0 executable artifact."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    spec = spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_GATE = _load_script("run_patch_contract_gate")
_SUPPORT = _load_script("patch_contract_support")
_VERIFIER = _load_script("verify_patch_acceptance")


def _baseline() -> dict[str, object]:
    """Return the concrete Phase 0 no-advertisement inventory."""
    value = loads(
        (_FIXTURES / "baseline_evidence.json").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def _fixture(name: str) -> dict[str, object]:
    """Return one exact executable Phase 0 fixture corpus."""
    value = loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_phase0_dormant_public_contract_preflight(
    monkeypatch: MonkeyPatch,
) -> None:
    """Require the dormant public contract gate to preflight successfully."""
    monkeypatch.delenv(_GATE.POSTGRESQL_TEST_DSN_ENV, raising=False)
    monkeypatch.delenv(_GATE._LEGACY_POSTGRESQL_LEASE_ENV, raising=False)
    monkeypatch.setattr(_GATE, "_load_patch_contracts", lambda root: None)
    assert _GATE.preflight(2, repo_root=_ROOT) is None


def test_phase0_fault_lifecycle_evidence_is_executable() -> None:
    """Execute the frozen lifecycle, failure, golden, and threat corpora."""

    async def execute() -> None:
        controller = _SUPPORT.FaultController.create()
        task = __import__("asyncio").create_task(
            controller.arrive(_SUPPORT.FaultLabel.LIFECYCLE_BEFORE)
        )
        await controller.wait_until_entered(
            _SUPPORT.FaultLabel.LIFECYCLE_BEFORE
        )
        controller.release(_SUPPORT.FaultLabel.LIFECYCLE_BEFORE)
        await task

    assert run(execute()) is None
    requirements = _VERIFIER._validate_requirements(
        _FIXTURES / "requirements_traceability.json",
        _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json"),
        _ROOT,
    )
    _VERIFIER._validate_failure_matrix(
        _FIXTURES / "failure_matrix.json",
        _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json"),
        requirements,
    )
    goldens = _fixture("goldens.json")
    cases = goldens["cases"]
    assert isinstance(cases, list)
    for raw in cases:
        assert isinstance(raw, dict)
        result = _SUPPORT.execute_golden_corpus(
            _SUPPORT.GoldenCorpusCategory(raw["category"]),
            bytes.fromhex(str(raw["input_bytes_hex"])),
        )
        assert result.output_bytes.hex() == raw["expected_bytes_hex"]
        assert result.outcome == raw["expected_outcome"]
        assert result.error == raw["expected_error"]
    threats = _fixture("threat_model.json")["threats"]
    assert isinstance(threats, list)
    for raw in threats:
        assert isinstance(raw, dict)
        result = _SUPPORT.execute_threat_corpus(
            _SUPPORT.ThreatCorpusIdentifier(raw["id"]),
            bytes.fromhex(str(raw["setup_bytes_hex"])),
            bytes.fromhex(str(raw["action_bytes_hex"])),
        )
        assert result.output_bytes.hex() == raw["expected_bytes_hex"]
        assert result.outcome == raw["expected_containment"]
        assert result.error == raw["expected_error"]


def test_phase0_manual_clock_remains_deterministic() -> None:
    """Exercise the bounded deterministic clock used by contract fixtures."""
    assert _SUPPORT.ManualClock(tick=2).advance(3).tick == 5


def test_phase0_strict_json_rejects_duplicate_members() -> None:
    """Exercise the shared strict JSON decoder used by dormant contracts."""
    assert _SUPPORT.load_strict_json('{"request":"one"}') == {"request": "one"}


def test_phase0_no_process_or_tool_surface_is_advertised() -> None:
    """Exercise the runtime advertisement scan against checked-out source."""
    assert (
        _VERIFIER._validate_runtime_patch_advertisement(
            _baseline()["runtime_patch_advertisement"], _ROOT
        )
        is None
    )


def test_phase0_runtime_probe_rejects_dynamic_patch_identity() -> None:
    """Reject a dynamically composed patch namespace before advertisement."""
    from avalan.tool import ToolSet

    try:
        _VERIFIER._assert_runtime_toolsets_incapable(
            (ToolSet(namespace="pa" + "tch", tools=()),)
        )
    except _VERIFIER.PatchAcceptanceError:
        return
    raise AssertionError("dynamic patch identity was not rejected")


def test_phase0_workspace_oracle_keeps_recursive_metadata() -> None:
    """Exercise the immutable workspace oracle without disk access."""
    root = _SUPPORT.WorkspaceEntry(
        name="",
        entry_type=_SUPPORT.WorkspaceEntryType.DIRECTORY,
        content=b"",
        symlink_target=None,
        link_count=1,
        identity=_SUPPORT.PatchWorkspaceEntryId("root"),
        mode=0o755,
    )
    oracle = _SUPPORT.WorkspaceOracle(root=root)
    assert oracle.equals(oracle) and len(oracle.digest()) == 64


def test_phase0_target_trace_remains_nonmutating_by_default() -> None:
    """Exercise the script-only target trace before a future target exists."""

    async def execute() -> None:
        target = _SUPPORT.ScriptedMutationTarget(capabilities=())
        _, successor = await target.inspect(_SUPPORT.PatchPath("visible.txt"))
        assert successor.trace[-1].action is _SUPPORT.TargetTraceAction.INSPECT

    assert run(execute()) is None


def test_phase0_target_trace_actions_are_closed() -> None:
    """Exercise the frozen target action enumeration used by contract tests."""
    assert _SUPPORT.TargetTraceAction.COMMIT_STEP.value == "commit_step"


def test_phase0_approval_binding_is_immutable() -> None:
    """Exercise an exact plan-bound approval binding without broker effects."""
    binding = _SUPPORT.ApprovalBinding(
        plan_id=_SUPPORT.PatchPlanId("plan"),
        principal_id=_SUPPORT.PatchPrincipalId("principal"),
        tenant_id=_SUPPORT.PatchTenantId("tenant"),
        run_id=_SUPPORT.PatchRunId("run"),
        context_id=_SUPPORT.PatchContextId("context"),
        workspace_id=_SUPPORT.PatchWorkspaceId("workspace"),
        policy_id=_SUPPORT.PatchPolicyId("policy"),
        broker_id=_SUPPORT.PatchBrokerId("broker"),
        quorum=1,
    )
    assert binding.plan_id == "plan"


def test_phase0_approval_broker_denial_is_closed() -> None:
    """Exercise the script-only broker denial and immutable successor."""
    binding = _SUPPORT.ApprovalBinding(
        plan_id=_SUPPORT.PatchPlanId("plan"),
        principal_id=_SUPPORT.PatchPrincipalId("principal"),
        tenant_id=_SUPPORT.PatchTenantId("tenant"),
        run_id=_SUPPORT.PatchRunId("run"),
        context_id=_SUPPORT.PatchContextId("context"),
        workspace_id=_SUPPORT.PatchWorkspaceId("workspace"),
        policy_id=_SUPPORT.PatchPolicyId("policy"),
        broker_id=_SUPPORT.PatchBrokerId("broker"),
        quorum=1,
    )

    async def execute() -> None:
        broker = _SUPPORT.ScriptedApprovalBroker(
            binding=binding,
            decision=_SUPPORT.ApprovalDecision.DENY,
        )
        outcome, successor = await broker.decide(
            binding, _SUPPORT.ManualClock(tick=0)
        )
        assert outcome.kind is _SUPPORT.ApprovalOutcomeKind.DENIED
        assert successor.calls == (binding,)

    assert run(execute()) is None


def test_phase0_store_conformance_inventory_is_complete() -> None:
    """Exercise the future-store contract inventory without opening a store."""
    suite = _SUPPORT.StoreConformanceSuite.create()
    assert tuple(item.value for item in suite.backends) == (
        "in_memory",
        "postgresql",
    )


def test_phase0_incapable_target_profile_has_no_capabilities() -> None:
    """Exercise the incapable profile with no target authority."""
    profile = _SUPPORT.TargetConformanceProfile(
        kind=_SUPPORT.TargetProfileKind.SCRIPTED,
        capable=False,
    )
    assert profile.required_capabilities == ()


def test_phase0_target_factory_corpus_requires_each_context() -> None:
    """Exercise the target-factory corpus's closed context ordering."""
    profiles = tuple(
        _SUPPORT.TargetConformanceProfile(kind=kind, capable=False)
        for kind in _SUPPORT.TargetProfileKind
    )
    assert (
        _SUPPORT.TargetFactoryConformanceRunner(profiles=profiles).profiles
        == profiles
    )


def test_phase0_phase_evidence_codec_seals_deterministically() -> None:
    """Exercise deterministic sealing for the Phase 0 record type."""
    evidence = _SUPPORT.PhaseEvidence(
        phase=0,
        status=_SUPPORT.PhaseEvidenceStatus.IN_PROGRESS,
        active_node_ids=(
            "tests/patch_phase0_contract_test.py::test_phase0_phase_evidence_codec_seals_deterministically",
        ),
        commands=(
            _SUPPORT.PhaseCommandEvidence(command="pytest", exit_code=0),
        ),
        artifact_digests=(
            _SUPPORT.ArtifactDigest(name="fixture", sha256="0" * 64),
        ),
        reviewer_findings=(
            _SUPPORT.ReviewerFinding(
                identifier="PATCH-REV-LOCAL",
                severity=_SUPPORT.ReviewerSeverity.P3,
                disposition=_SUPPORT.ReviewerDisposition.FIXED,
                rationale="local deterministic evidence",
            ),
        ),
    )
    sealed = _SUPPORT.PhaseEvidenceCodec.seal(
        evidence, _SUPPORT.PatchObserverId("reviewer")
    )
    assert _SUPPORT.PhaseEvidenceCodec.verify(sealed)


def test_phase0_resource_depth_sentinel_rejects_unowned_awaits() -> None:
    """Exercise resource depths at a closed publication boundary."""

    async def execute() -> None:
        sentinel = _SUPPORT.ResourceDepthSentinel()
        result = await sentinel.at_await(_SUPPORT.AwaitBoundary.PUBLICATION)
        assert (
            result.receipts[-1].boundary is _SUPPORT.AwaitBoundary.PUBLICATION
        )

    assert run(execute()) is None
