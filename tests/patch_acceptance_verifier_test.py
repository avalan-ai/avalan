"""Exercise the dormant patch acceptance-manifest verifier."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from pathlib import Path
from sys import modules
from sys import path as sys_path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "patch"


def _load_verifier() -> ModuleType:
    """Load the standalone verifier with its shared script dependencies."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    name = "_patch_acceptance_verifier"
    spec = spec_from_file_location(
        name, _ROOT / "scripts" / "verify_patch_acceptance.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_VERIFIER = _load_verifier()


def _copy_bundle(destination: Path) -> None:
    """Copy every tracked JSON fixture into one disposable bundle."""
    destination.mkdir()
    for source in _FIXTURES.glob("*.json"):
        (destination / source.name).write_bytes(source.read_bytes())


def _read(path: Path) -> dict[str, object]:
    """Read one mutable JSON object fixture."""
    value = loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: object) -> None:
    """Write one deterministic disposable JSON fixture."""
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def _resign(payload: dict[str, object], field: str) -> None:
    """Update one fixture digest after a focused mutation."""
    if field == "catalog_sha256":
        canonical = {
            "record_layout": payload["record_layout"],
            "requirements": payload["requirements"],
        }
    else:
        canonical = {
            key: value for key, value in payload.items() if key != field
        }
    payload[field] = _VERIFIER.canonical_sha256(canonical)


def test_patch_acceptance_positive_load() -> None:
    """Load the complete dormant Phase 0 patch contract bundle."""
    manifest = _VERIFIER.load_phase0_contracts(_FIXTURES, repo_root=_ROOT)

    assert manifest.current_phase == 0
    assert len(manifest.active_nodes(0)) == 15


def test_patch_acceptance_validates_complete_phase_evidence() -> None:
    """Validate required typed fields in the in-progress Phase 0 record."""
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")

    _VERIFIER._validate_phase_evidence(
        _FIXTURES / "phase_evidence.json", manifest, _ROOT
    )


def test_patch_acceptance_history_snapshot_is_pinned() -> None:
    """Keep every active semantic field and executable digest immutable."""
    payload = _read(_FIXTURES / "acceptance_history.json")
    snapshot = payload["snapshot"]
    assert isinstance(snapshot, dict)
    nodes = snapshot["nodes"]
    assert isinstance(nodes, list) and isinstance(nodes[0], dict)
    assert set(nodes[0]) == {
        "id",
        "requirement_ids",
        "category",
        "surface",
        "context",
        "platform",
        "operation",
        "authority",
        "commit_boundary",
        "evidence_class",
        "node_id",
        "lifecycle",
        "active_from_phase",
        "executable_sha256",
    }
    assert snapshot["snapshot_sha256"] == (
        _VERIFIER._PINNED_ACCEPTANCE_HISTORY_SNAPSHOT_SHA256
    )


def test_patch_acceptance_rejects_resigned_history_erasure(
    tmp_path: Path,
) -> None:
    """Reject a re-signed snapshot that removes historical active evidence."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    path = fixtures / "acceptance_history.json"
    payload = _read(path)
    snapshot = payload["snapshot"]
    assert isinstance(snapshot, dict)
    nodes = snapshot["nodes"]
    assert isinstance(nodes, list)
    nodes.pop()
    snapshot["snapshot_sha256"] = _VERIFIER.canonical_sha256(nodes)
    _resign(payload, "history_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError, match="history snapshot is not pinned"
    ):
        _VERIFIER._validate_acceptance_history(
            path,
            _VERIFIER.load_manifest(fixtures / "acceptance_manifest.json"),
            _ROOT,
        )


def test_patch_acceptance_rejects_unreviewed_history_rename(
    tmp_path: Path,
) -> None:
    """Reject an active-node rename without a reviewed replacement ledger."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    manifest_path = fixtures / "acceptance_manifest.json"
    manifest_payload = _read(manifest_path)
    nodes = manifest_payload["nodes"]
    assert isinstance(nodes, list) and isinstance(nodes[0], dict)
    nodes[0]["id"] = "PATCH-A-REPLACED"
    _resign(manifest_payload, "manifest_sha256")
    _write(manifest_path, manifest_payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="semantic change is unreviewed",
    ):
        _VERIFIER._validate_acceptance_history(
            fixtures / "acceptance_history.json",
            _VERIFIER.load_manifest(manifest_path),
            _ROOT,
        )


def test_patch_acceptance_accepts_reviewed_history_replacement(
    tmp_path: Path,
) -> None:
    """Accept a monotonic active replacement tied to a named review round."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    manifest_path = fixtures / "acceptance_manifest.json"
    manifest_payload = _read(manifest_path)
    nodes = manifest_payload["nodes"]
    assert isinstance(nodes, list) and isinstance(nodes[0], dict)
    nodes[0]["id"] = "PATCH-A-REPLACED"
    _resign(manifest_payload, "manifest_sha256")
    _write(manifest_path, manifest_payload)
    history_path = fixtures / "acceptance_history.json"
    history = _read(history_path)
    replacements = history["replacements"]
    assert isinstance(replacements, list)
    snapshot = history["snapshot"]
    assert isinstance(snapshot, dict)
    historical_nodes = snapshot["nodes"]
    assert isinstance(historical_nodes, list) and isinstance(
        historical_nodes[0], dict
    )
    replacement = deepcopy(historical_nodes[0])
    replacement["id"] = "PATCH-A-REPLACED"
    replacements.append(
        {
            "old": historical_nodes[0],
            "new": replacement,
            "review_round": 3,
            "reviewer": "round-3-contract-review",
            "rationale": "Reviewed replacement seals every semantic field.",
        }
    )
    _resign(history, "history_sha256")
    _write(history_path, history)

    _VERIFIER._validate_acceptance_history(
        history_path,
        _VERIFIER.load_manifest(manifest_path),
        _ROOT,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("requirement_ids", ["PATCH-R-0001"]),
        ("category", "negative"),
        ("surface", "sdk"),
        ("context", "local"),
        ("platform", "linux"),
        ("operation", "replacement"),
        ("authority", "read_only"),
        ("commit_boundary", "commit_started"),
        ("evidence_class", "replacement"),
        (
            "node_id",
            (
                "tests/patch_phase0_contract_test.py::"
                "test_phase0_runtime_probe_rejects_dynamic_patch_identity"
            ),
        ),
        ("lifecycle_phase", ("planned", 1)),
    ),
)
def test_patch_acceptance_rejects_unreviewed_history_semantic_change(
    tmp_path: Path, field: str, value: object
) -> None:
    """Reject every semantic history weakening without a reviewed record."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    manifest_path = fixtures / "acceptance_manifest.json"
    manifest = _read(manifest_path)
    nodes = manifest["nodes"]
    assert isinstance(nodes, list) and isinstance(nodes[0], dict)
    if field == "lifecycle_phase":
        assert isinstance(value, tuple) and len(value) == 2
        lifecycle, phase = value
        assert isinstance(lifecycle, str) and type(phase) is int
        nodes[0]["lifecycle"] = lifecycle
        nodes[0]["active_from_phase"] = phase
    else:
        nodes[0][field] = value
    _resign(manifest, "manifest_sha256")
    _write(manifest_path, manifest)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError, match="semantic change is unreviewed"
    ):
        _VERIFIER._validate_acceptance_history(
            fixtures / "acceptance_history.json",
            _VERIFIER.load_manifest(manifest_path),
            _ROOT,
        )


def test_patch_acceptance_rejects_complete_phase_with_pending_gates(
    tmp_path: Path,
) -> None:
    """Reject a terminal evidence status while quality receipts are pending."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    path = fixtures / "phase_evidence.json"
    payload = _read(path)
    payload["status"] = "complete"
    _resign(payload, "record_sha256")
    _write(path, payload)
    manifest = _VERIFIER.load_manifest(fixtures / "acceptance_manifest.json")

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="complete evidence exact gate is not complete",
    ):
        _VERIFIER._validate_phase_evidence(path, manifest, _ROOT)


def test_patch_acceptance_rejects_missing_fixture(tmp_path: Path) -> None:
    """Reject a bundle missing a required tracked Phase 0 record."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    (fixtures / "goldens.json").unlink()

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="required patch fixture is missing",
    ):
        _VERIFIER.load_phase0_contracts(fixtures, repo_root=_ROOT)


def test_patch_acceptance_rejects_duplicate_node(tmp_path: Path) -> None:
    """Reject duplicate acceptance-node identities before collection."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    path = fixtures / "acceptance_manifest.json"
    payload = _read(path)
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    nodes.append(deepcopy(nodes[0]))
    _resign(payload, "manifest_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError, match="duplicate acceptance node ID"
    ):
        _VERIFIER.load_phase0_contracts(fixtures, repo_root=_ROOT)


def test_patch_acceptance_rejects_unowned_requirement(tmp_path: Path) -> None:
    """Reject requirements whose recorded test evidence no longer owns them."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    path = fixtures / "requirements_traceability.json"
    payload = _read(path)
    requirements = payload["requirements"]
    assert isinstance(requirements, list)
    requirement = requirements[0]
    assert isinstance(requirement, list)
    requirement[11] = "tests/patch_acceptance_verifier_test.py::missing_owner"
    _resign(payload, "catalog_sha256")
    _write(path, payload)

    with pytest.raises(_VERIFIER.PatchAcceptanceError, match="unowned"):
        _VERIFIER.load_phase0_contracts(fixtures, repo_root=_ROOT)


def test_patch_acceptance_rejects_duplicate_requirement_id(
    tmp_path: Path,
) -> None:
    """Reject duplicate stable requirement IDs in the traceability record."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    path = fixtures / "requirements_traceability.json"
    payload = _read(path)
    requirements = payload["requirements"]
    assert isinstance(requirements, list)
    first = requirements[0]
    second = requirements[1]
    assert isinstance(first, list) and isinstance(second, list)
    second[0] = first[0]
    _resign(payload, "catalog_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError, match="duplicate requirement ID"
    ):
        _VERIFIER.load_phase0_contracts(fixtures, repo_root=_ROOT)


def test_patch_acceptance_rejects_requirement_id_gap(tmp_path: Path) -> None:
    """Reject a traceability catalog with a gap in stable requirement IDs."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    path = fixtures / "requirements_traceability.json"
    payload = _read(path)
    requirements = payload["requirements"]
    assert isinstance(requirements, list)
    second = requirements[1]
    assert isinstance(second, list)
    second[0] = "PATCH-R-0003"
    _resign(payload, "catalog_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="requirement IDs are not contiguous",
    ):
        _VERIFIER.load_phase0_contracts(fixtures, repo_root=_ROOT)


def test_patch_acceptance_rejects_duplicate_requirement_span(
    tmp_path: Path,
) -> None:
    """Reject duplicate source spans in the traceability catalog."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    path = fixtures / "requirements_traceability.json"
    payload = _read(path)
    requirements = payload["requirements"]
    assert isinstance(requirements, list)
    first = requirements[0]
    second = requirements[1]
    assert isinstance(first, list) and isinstance(second, list)
    second[1:4] = first[1:4]
    _resign(payload, "catalog_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="duplicate requirement source span",
    ):
        _VERIFIER.load_phase0_contracts(fixtures, repo_root=_ROOT)


def test_patch_acceptance_rejects_missing_required_source_area(
    tmp_path: Path,
) -> None:
    """Reject a catalog that omits one required appendix area."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    path = fixtures / "requirements_traceability.json"
    payload = _read(path)
    requirements = payload["requirements"]
    assert isinstance(requirements, list)
    for requirement in requirements:
        assert isinstance(requirement, list)
        if requirement[1] == "A.4":
            requirement[1] = "A.3"
    _resign(payload, "catalog_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="required source areas are missing",
    ):
        _VERIFIER.load_phase0_contracts(fixtures, repo_root=_ROOT)


def test_patch_acceptance_rejects_inventory_drift(tmp_path: Path) -> None:
    """Reject a frozen source inventory that differs from the checkout."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    path = fixtures / "contract_decisions.json"
    payload = _read(path)
    inventories = payload["inventories"]
    assert isinstance(inventories, dict)
    src_inventory = inventories["src"]
    assert isinstance(src_inventory, list)
    source = src_inventory[0]
    assert isinstance(source, dict)
    source["source_sha256"] = "0" * 64
    payload["inventory_sha256"] = _VERIFIER.canonical_sha256(inventories)
    _resign(payload, "record_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError, match="baseline src inventory drifted"
    ):
        _VERIFIER.load_phase0_contracts(fixtures, repo_root=_ROOT)


def test_patch_acceptance_rejects_premature_active_node(
    tmp_path: Path,
) -> None:
    """Reject future evidence that claims active lifecycle status early."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    path = fixtures / "acceptance_manifest.json"
    payload = _read(path)
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    future = nodes[15]
    assert isinstance(future, dict)
    future["lifecycle"] = "active"
    _resign(payload, "manifest_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError, match="prematurely active"
    ):
        _VERIFIER.load_phase0_contracts(fixtures, repo_root=_ROOT)


def test_patch_acceptance_rejects_source_artifact_read(tmp_path: Path) -> None:
    """Reject a tracked module that reads one ignored patch design artifact."""
    source = tmp_path / "src"
    source.mkdir()
    path = source / "reader.py"
    path.write_text(
        "from pathlib import Path\nPath('specs/PATCH.md').read_text()\n",
        encoding="utf-8",
    )

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError, match="design artifact"
    ):
        _VERIFIER.verify_source_artifact_reads(tmp_path)


@pytest.mark.parametrize(
    "source",
    (
        (
            "from pathlib import Path as P\n"
            "P('specs').joinpath('PATCH.md').read_text()\n"
        ),
        (
            "from pathlib import Path\n"
            "(Path('specs') / 'PATCH-agenda.md').read_bytes()\n"
        ),
        (
            "import os as system\n"
            "name = 'PATCH.md'\n"
            "path = system.path.join('specs', name)\n"
            "system.open(path, 0)\n"
        ),
        (
            "from io import open as reader\n"
            "prefix = 'specs/'\n"
            "name = prefix + 'PATCH.md'\n"
            "reader(name)\n"
        ),
        (
            "import builtins as runtime\n"
            "DESIGN = 'specs/PATCH-agenda.md'\n"
            "runtime.open(DESIGN)\n"
        ),
        (
            "import builtins as runtime\n"
            "reader = getattr(runtime, 'open')\n"
            "reader('specs/PATCH.md')\n"
        ),
        "name = 'PATCH.md'\nopen(f'specs/{name}')\n",
        "name = '/'.join(('specs', 'PATCH-agenda.md'))\nopen(name)\n",
        "from builtins import open\nreader = open\nreader('specs/PATCH.md')\n",
    ),
)
def test_patch_acceptance_rejects_source_artifact_read_bypasses(
    tmp_path: Path,
    source: str,
) -> None:
    """Reject join, concatenation, variable, and reader-alias bypasses."""
    directory = tmp_path / "tests"
    directory.mkdir()
    (directory / "reader.py").write_text(source, encoding="utf-8")

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError, match="design artifact"
    ):
        _VERIFIER.verify_source_artifact_reads(tmp_path)


def test_patch_acceptance_audits_actual_design_artifact_opens(
    tmp_path: Path,
) -> None:
    """Reject an actual interpreter-level artifact open in contract tests."""
    artifact = tmp_path / "specs" / "PATCH-agenda.md"
    source = (
        b"from builtins import open as reader\nreader(artifact)\n"
    ).decode("utf-8")

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError, match="design artifact at runtime"
    ):
        _VERIFIER.verify_source_artifact_runtime_open(
            lambda: exec(source, {"artifact": artifact})
        )


def test_patch_acceptance_rejects_incomplete_advertisement(
    tmp_path: Path,
) -> None:
    """Reject an advertised public surface without complete prerequisites."""
    fixtures = tmp_path / "fixtures"
    _copy_bundle(fixtures)
    path = fixtures / "surface_conformance.json"
    payload = _read(path)
    surfaces = payload["surfaces"]
    assert isinstance(surfaces, list)
    surface = surfaces[0]
    assert isinstance(surface, dict)
    surface["advertised"] = True
    _resign(payload, "manifest_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="advertised surface has incomplete capability evidence",
    ):
        _VERIFIER.load_phase0_contracts(fixtures, repo_root=_ROOT)


def _validate_failure_fixture(path: Path) -> None:
    """Validate one mutable failure matrix against the frozen Phase 0 owner."""
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")
    requirements = _VERIFIER._validate_requirements(
        _FIXTURES / "requirements_traceability.json", manifest, _ROOT
    )
    _VERIFIER._validate_failure_matrix(path, manifest, requirements)


def test_patch_acceptance_rejects_missing_failure_boundary(
    tmp_path: Path,
) -> None:
    """Reject a matrix that omits one frozen failure boundary."""
    path = tmp_path / "failure_matrix.json"
    payload = _read(_FIXTURES / "failure_matrix.json")
    cells = payload["cells"]
    assert isinstance(cells, list)
    cells.pop()
    _resign(payload, "matrix_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="failure boundary catalog is incomplete",
    ):
        _validate_failure_fixture(path)


def test_patch_acceptance_rejects_duplicate_failure_boundary(
    tmp_path: Path,
) -> None:
    """Reject a matrix that represents one frozen boundary twice."""
    path = tmp_path / "failure_matrix.json"
    payload = _read(_FIXTURES / "failure_matrix.json")
    cells = payload["cells"]
    assert isinstance(cells, list) and isinstance(cells[0], dict)
    duplicate = deepcopy(cells[0])
    duplicate["id"] = "PATCH-F-999"
    cells.append(duplicate)
    _resign(payload, "matrix_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError, match="duplicate failure boundary"
    ):
        _validate_failure_fixture(path)


def test_patch_acceptance_rejects_failure_count_drift(tmp_path: Path) -> None:
    """Reject a count that no longer matches the target inspection oracle."""
    path = tmp_path / "failure_matrix.json"
    payload = _read(_FIXTURES / "failure_matrix.json")
    cells = payload["cells"]
    assert isinstance(cells, list) and isinstance(cells[10], dict)
    cells[10]["expected_inspection_count"] = 2
    _resign(payload, "matrix_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="failure target inspection count drifted",
    ):
        _validate_failure_fixture(path)


def test_patch_acceptance_rejects_failure_state_drift(tmp_path: Path) -> None:
    """Reject a requested-effect step state outside the closed algebra."""
    path = tmp_path / "failure_matrix.json"
    payload = _read(_FIXTURES / "failure_matrix.json")
    cells = payload["cells"]
    assert isinstance(cells, list) and isinstance(cells[0], dict)
    states = cells[0]["per_step_states"]
    assert isinstance(states, list) and isinstance(states[0], dict)
    states[0]["state"] = "unknown_state"
    _resign(payload, "matrix_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="failure step state is invalid",
    ):
        _validate_failure_fixture(path)


def test_patch_acceptance_rejects_precommit_oracle_drift(
    tmp_path: Path,
) -> None:
    """Reject a precommit row that lacks either zero-write oracle."""
    path = tmp_path / "failure_matrix.json"
    payload = _read(_FIXTURES / "failure_matrix.json")
    cells = payload["cells"]
    assert isinstance(cells, list) and isinstance(cells[0], dict)
    cells[0]["workspace_oracle_equal"] = False
    _resign(payload, "matrix_sha256")
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="precommit failure lacks both zero-write evidence oracles",
    ):
        _validate_failure_fixture(path)


def test_patch_acceptance_freezes_all_boundary_semantics() -> None:
    """Require one complete exact semantic truth record per frozen boundary."""
    catalog = _VERIFIER._FAILURE_SEMANTICS
    effect_facts = _VERIFIER._FAILURE_EFFECT_FACTS

    assert tuple(catalog) == _VERIFIER._FAILURE_BOUNDARY_CATALOG
    assert len(catalog) == 46
    for semantic in catalog.values():
        assert len(semantic.counts) == 5
        assert semantic.step_state in _VERIFIER._FAILURE_STEP_STATES
        assert semantic.lineage_state in _VERIFIER._FAILURE_LINEAGE_STATES
        assert semantic.artifact_state in _VERIFIER._FAILURE_ARTIFACT_STATES
        assert semantic.requested_effect_fact in effect_facts
        assert (
            semantic.workspace_change_fact
            in _VERIFIER._FAILURE_WORKSPACE_FACTS
        )
        assert semantic.events[
            -1
        ] == "request_completed" or semantic.pending_behavior.startswith(
            "pending_"
        )


@pytest.mark.parametrize(
    ("name", "boundary"),
    (
        ("commit_started", "lifecycle.received"),
        ("count_vector", "target.inspect"),
        ("step_state", "target.commit_step"),
        ("lineage_state", "target.commit_step"),
        ("artifact_state", "target.stage_artifact"),
        ("requested_effect_fact", "target.commit_step"),
        ("workspace_change_fact", "target.commit_step"),
        ("event_sequence", "target.inspect"),
        ("pending_behavior", "cancellation.after_commit"),
        ("retryability", "target.commit_step"),
        ("workspace_oracle_equal", "lifecycle.received"),
        ("public_projection", "target.commit_step"),
    ),
)
def test_patch_acceptance_rejects_each_failure_semantic_mutation(
    tmp_path: Path, name: str, boundary: str
) -> None:
    """Reject one valid-looking mutation in every matrix semantic category."""
    path = tmp_path / f"failure-{name}.json"
    payload = _read(_FIXTURES / "failure_matrix.json")
    cells = payload["cells"]
    assert isinstance(cells, list)
    cell = next(
        item
        for item in cells
        if isinstance(item, dict) and item["boundary"] == boundary
    )
    assert isinstance(cell, dict)
    match name:
        case "commit_started":
            cell["commit_started"] = not bool(cell["commit_started"])
        case "count_vector":
            cell["expected_inspection_count"] = 2
            cell["target_inspection_count"] = 2
        case "step_state":
            states = cell["per_step_states"]
            assert isinstance(states, list) and isinstance(states[0], dict)
            states[0]["state"] = "indeterminate"
        case "lineage_state":
            states = cell["per_lineage_states"]
            assert isinstance(states, list) and isinstance(states[0], dict)
            states[0]["state"] = "indeterminate"
        case "artifact_state":
            cell["artifact_state"] = "leaked"
        case "requested_effect_fact":
            cell["requested_effect_fact"] = "partially_committed"
        case "workspace_change_fact":
            cell["workspace_change_fact"] = "unknown"
        case "event_sequence":
            cell["event_sequence"] = ["request_received", "request_completed"]
        case "pending_behavior":
            cell["pending_behavior"] = "terminal_not_pending"
        case "retryability":
            cell["retryability"] = "retransmit_only"
        case "workspace_oracle_equal":
            cell["workspace_oracle_equal"] = not bool(
                cell["workspace_oracle_equal"]
            )
        case "public_projection":
            cell["public_projection"] = "patch_indeterminate"
        case _:
            pytest.fail(f"uncovered failure mutation category: {name}")
    _resign(payload, "matrix_sha256")
    _write(path, payload)

    with pytest.raises(_VERIFIER.PatchAcceptanceError):
        _validate_failure_fixture(path)
