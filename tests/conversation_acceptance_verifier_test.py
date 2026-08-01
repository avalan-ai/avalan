"""Exercise strict conversation acceptance verification."""

from copy import deepcopy
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from pathlib import Path
from shutil import copy2, copytree
from sys import modules
from sys import path as sys_path
from types import ModuleType
from typing import Any, Protocol, cast

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "conversation"


class _AcceptanceManifest(Protocol):
    """Describe manifest attributes exercised by Phase 1 simulations."""

    current_phase: int

    def active_nodes(self, through_phase: int) -> tuple[object, ...]:
        """Return active nodes through one phase."""


def _load_verifier() -> ModuleType:
    """Return the conversation acceptance verifier module."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    name = "_conversation_acceptance_verifier"
    spec = spec_from_file_location(
        name, _ROOT / "scripts" / "verify_conversation_acceptance.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_VERIFIER = _load_verifier()


def _read(name: str) -> dict[str, object]:
    """Return one mutable conversation fixture object."""
    value: object = loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return {str(key): item for key, item in value.items()}


def _write(path: Path, value: object) -> None:
    """Write deterministic JSON fixture evidence."""
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def _resign(payload: dict[str, object], digest_field: str) -> None:
    """Update one fixture's canonical digest."""
    canonical = {
        key: value for key, value in payload.items() if key != digest_field
    }
    payload[digest_field] = _VERIFIER.canonical_sha256(canonical)


def _manifest() -> object:
    """Return the validated Phase 0 acceptance manifest."""
    return _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")


def _phase1_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_AcceptanceManifest, dict[str, object]]:
    """Return a simulated Phase 1 manifest with appended source anchors."""
    payload = _read("acceptance_manifest.json")
    payload["current_phase"] = 1
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    for node in nodes:
        assert isinstance(node, dict)
        if node.get("active_from_phase") == 1:
            node["lifecycle"] = "active"
    active_ids = [
        node["node_id"]
        for node in nodes
        if isinstance(node, dict)
        and node.get("lifecycle") == "active"
        and isinstance(node.get("active_from_phase"), int)
        and node["active_from_phase"] <= 1
    ]
    assert all(isinstance(node_id, str) for node_id in active_ids)
    history = payload["activation_history"]
    assert isinstance(history, list)
    phase1_digest = _VERIFIER._text_digest(tuple(active_ids))
    phase1_history = {
        "phase": 1,
        "node_ids": active_ids,
        "sha256": phase1_digest,
    }
    history.append(phase1_history)
    _resign(payload, "manifest_sha256")
    phase1_nodes = [
        node
        for node in nodes
        if isinstance(node, dict) and node.get("active_from_phase") == 1
    ]
    normalized_phase1_nodes = [
        {key: value for key, value in node.items() if key != "lifecycle"}
        for node in phase1_nodes
    ]
    node_anchors = dict(_VERIFIER._NODE_PAYLOAD_SHA256_BY_PHASE)
    node_anchors[1] = _VERIFIER.canonical_sha256(normalized_phase1_nodes)
    history_anchors = dict(_VERIFIER._ACTIVATION_HISTORY_BY_PHASE)
    history_anchors[1] = phase1_digest
    replacement_anchors = dict(_VERIFIER._REPLACEMENT_HISTORY_BY_PHASE)
    replacement_anchors[1] = replacement_anchors[0]
    monkeypatch.setattr(
        _VERIFIER,
        "_NODE_PAYLOAD_SHA256_BY_PHASE",
        node_anchors,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_ACTIVATION_HISTORY_BY_PHASE",
        history_anchors,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_REPLACEMENT_HISTORY_BY_PHASE",
        replacement_anchors,
    )
    path = tmp_path / "acceptance-phase1.json"
    _write(path, payload)
    return _VERIFIER.load_manifest(path), payload


def _phase1_replacement_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_AcceptanceManifest, dict[str, object], str, str]:
    """Return a Phase 1 manifest with one retained reviewed tombstone."""
    payload = _read("acceptance_manifest.json")
    payload["current_phase"] = 1
    nodes = payload["nodes"]
    history = payload["activation_history"]
    replacements = payload["replacements"]
    assert isinstance(nodes, list)
    assert isinstance(history, list)
    assert isinstance(replacements, list)
    old = nodes[0]
    assert isinstance(old, dict)
    old_node_id = str(old["node_id"])
    old["lifecycle"] = "replaced"
    for node in nodes:
        assert isinstance(node, dict)
        if node.get("active_from_phase") == 1:
            node["lifecycle"] = "active"
    target = deepcopy(old)
    target["id"] = "phase1-reviewed-replacement"
    target["lifecycle"] = "active"
    target["active_from_phase"] = 1
    target_node_id = (
        "tests/conversation/replacement_contract_test.py::"
        "test_phase1_reviewed_replacement"
    )
    target["node_id"] = target_node_id
    nodes.append(target)
    activated_ids = [
        str(node["node_id"])
        for node in nodes
        if isinstance(node, dict)
        and node.get("lifecycle") in {"active", "replaced"}
        and isinstance(node.get("active_from_phase"), int)
        and node["active_from_phase"] <= 1
    ]
    history_sha256 = _VERIFIER._text_digest(tuple(activated_ids))
    phase1_history = {
        "phase": 1,
        "node_ids": activated_ids,
        "sha256": history_sha256,
    }
    history.append(phase1_history)
    replacement = {
        "phase": 1,
        "old_node_id": old_node_id,
        "replacement_node_ids": [target_node_id],
        "reviewed_by": "phase1-review",
        "evidence": "phase1-replacement-evidence",
    }
    replacements.append(replacement)
    _resign(payload, "manifest_sha256")
    phase1_nodes = [
        node
        for node in nodes
        if isinstance(node, dict) and node.get("active_from_phase") == 1
    ]
    normalized_phase1_nodes = [
        {key: value for key, value in node.items() if key != "lifecycle"}
        for node in phase1_nodes
    ]
    node_anchors = dict(_VERIFIER._NODE_PAYLOAD_SHA256_BY_PHASE)
    node_anchors[1] = _VERIFIER.canonical_sha256(normalized_phase1_nodes)
    history_anchors = dict(_VERIFIER._ACTIVATION_HISTORY_BY_PHASE)
    history_anchors[1] = history_sha256
    replacement_anchors = dict(_VERIFIER._REPLACEMENT_HISTORY_BY_PHASE)
    replacement_anchors[1] = (1, _VERIFIER.canonical_sha256([replacement]))
    monkeypatch.setattr(
        _VERIFIER,
        "_NODE_PAYLOAD_SHA256_BY_PHASE",
        node_anchors,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_ACTIVATION_HISTORY_BY_PHASE",
        history_anchors,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_REPLACEMENT_HISTORY_BY_PHASE",
        replacement_anchors,
    )
    path = tmp_path / "acceptance-phase1-replacement.json"
    _write(path, payload)
    return _VERIFIER.load_manifest(path), payload, old_node_id, target_node_id


def test_phase0_inventory_is_complete_and_executable() -> None:
    """Validate every integrated Phase 0 fixture without executing pytest."""
    manifest = _VERIFIER.verify_acceptance(
        _FIXTURES / "acceptance_manifest.json",
        repo_root=_ROOT,
        through_phase=0,
        execute=False,
    )
    requirements = _VERIFIER.load_requirements(
        _FIXTURES / "requirements_traceability.json",
        manifest,
        repo_root=_ROOT,
    )
    matrix = _VERIFIER.load_failure_matrix(
        _FIXTURES / "failure_matrix.json",
        manifest=manifest,
        requirement_ids=frozenset(item.id for item in requirements),
    )

    assert len(manifest.active_nodes(0)) == 9
    assert len(manifest.planned_nodes()) == 13
    assert len(requirements) == 144
    assert len(matrix.boundaries) == 11
    assert len(matrix.surfaces) == 9
    assert len(matrix.cells) == 99


def test_phase1_activation_appends_anchors_without_rewriting_phase0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept Phase 1 activation through new phase-scoped anchors only."""
    phase0_node_anchor = _VERIFIER._NODE_PAYLOAD_SHA256_BY_PHASE[0]
    phase0_history_anchor = _VERIFIER._ACTIVATION_HISTORY_BY_PHASE[0]

    manifest, _payload = _phase1_manifest(tmp_path, monkeypatch)

    assert manifest.current_phase == 1
    assert len(manifest.active_nodes(0)) == 9
    assert len(manifest.active_nodes(1)) == 10
    assert _VERIFIER._NODE_PAYLOAD_SHA256_BY_PHASE[0] == phase0_node_anchor
    assert _VERIFIER._ACTIVATION_HISTORY_BY_PHASE[0] == phase0_history_anchor


def test_phase1_replacement_retains_tombstone_history_and_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept one reviewed replacement without deleting prior evidence."""
    manifest, payload, old_node_id, target_node_id = (
        _phase1_replacement_manifest(tmp_path, monkeypatch)
    )
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    by_node_id = {
        str(node["node_id"]): node for node in nodes if isinstance(node, dict)
    }
    assert by_node_id[old_node_id]["lifecycle"] == "replaced"
    assert by_node_id[target_node_id]["lifecycle"] == "active"
    requirements = _VERIFIER.load_requirements(
        _FIXTURES / "requirements_traceability.json",
        manifest,
        repo_root=_ROOT,
    )
    assert len(requirements) == 144

    relative_paths = {
        "scripts/contract_gate.py",
        "scripts/contract_startup/avalan_contract_gate_plugin.py",
        "scripts/contract_startup/sitecustomize.py",
        "scripts/run_conversation_contract_gate.py",
        "scripts/verify_conversation_acceptance.py",
        "scripts/verify_conversation_types.py",
        *_VERIFIER._PHASE0_ACTIVE_SOURCE_SHA256,
    }
    for relative in relative_paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(_ROOT / relative, destination)
    target_relative = target_node_id.split("::", 1)[0]
    target_source = tmp_path / target_relative
    target_source.parent.mkdir(parents=True, exist_ok=True)
    target_source.write_text(
        "def test_phase1_reviewed_replacement() -> None:\n"
        "    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    domain_relative = "tests/conversation/domain_contract_test.py"
    domain_source = tmp_path / domain_relative
    domain_source.parent.mkdir(parents=True, exist_ok=True)
    domain_source.write_text(
        "def test_normative_domain_contract() -> None:\n"
        "    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    source_anchors = dict(_VERIFIER._ACTIVE_SOURCE_SHA256_BY_PHASE)
    source_anchors[1] = {
        domain_relative: (
            _VERIFIER.sha256(domain_source.read_bytes()).hexdigest()
        ),
        target_relative: (
            _VERIFIER.sha256(target_source.read_bytes()).hexdigest()
        ),
    }
    monkeypatch.setattr(
        _VERIFIER,
        "_ACTIVE_SOURCE_SHA256_BY_PHASE",
        source_anchors,
    )

    _VERIFIER.verify_gate_source_isolation(tmp_path, manifest)


def test_phase1_replacement_rejects_unreviewed_lifecycle_reversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a re-sign attack that revives the retained old record."""
    _manifest_value, payload, old_node_id, _target_node_id = (
        _phase1_replacement_manifest(tmp_path, monkeypatch)
    )
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    old = next(
        node
        for node in nodes
        if isinstance(node, dict) and node.get("node_id") == old_node_id
    )
    old["lifecycle"] = "active"
    _resign(payload, "manifest_sha256")
    path = tmp_path / "acceptance-replacement-revival.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="replaced acceptance records and reviewed ledger entries differ",
    ):
        _VERIFIER.load_manifest(path)


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("outside-chain", "outside its reviewed replacement chain"),
        ("inactive-leaf", "lacks active replacement-chain evidence"),
    ),
)
def test_requirement_ownership_requires_active_replacement_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
    message: str,
) -> None:
    """Reject extra owners and inactive leaves despite a valid replacement."""
    manifest, _payload, _old_node_id, target_node_id = (
        _phase1_replacement_manifest(tmp_path, monkeypatch)
    )
    dynamic_manifest = cast(Any, manifest)
    target = next(
        node
        for node in dynamic_manifest.nodes
        if node.node_id == target_node_id
    )
    if attack == "outside-chain":
        outsider = replace(
            target,
            id="phase1-unreviewed-owner",
            node_id=(
                "tests/conversation/unreviewed_test.py::test_unreviewed_owner"
            ),
        )
        attacked = replace(
            dynamic_manifest,
            nodes=(*dynamic_manifest.nodes, outsider),
        )
    else:
        inactive = replace(target, lifecycle="replaced")
        attacked = replace(
            dynamic_manifest,
            nodes=tuple(
                inactive if node.node_id == target_node_id else node
                for node in dynamic_manifest.nodes
            ),
        )

    with pytest.raises(_VERIFIER.ConversationAcceptanceError, match=message):
        _VERIFIER.load_requirements(
            _FIXTURES / "requirements_traceability.json",
            attacked,
            repo_root=_ROOT,
        )


def test_phase1_sources_append_without_rewriting_phase0_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept one new active source while preserving Phase 0 byte pins."""
    manifest, _payload = _phase1_manifest(tmp_path, monkeypatch)
    relative_paths = {
        "scripts/contract_gate.py",
        "scripts/contract_startup/avalan_contract_gate_plugin.py",
        "scripts/contract_startup/sitecustomize.py",
        "scripts/run_conversation_contract_gate.py",
        "scripts/verify_conversation_acceptance.py",
        "scripts/verify_conversation_types.py",
        *_VERIFIER._PHASE0_ACTIVE_SOURCE_SHA256,
    }
    for relative in relative_paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(_ROOT / relative, destination)
    phase1_relative = "tests/conversation/domain_contract_test.py"
    phase1_source = tmp_path / phase1_relative
    phase1_source.parent.mkdir(parents=True, exist_ok=True)
    phase1_source.write_text(
        "def test_normative_domain_contract() -> None:\n"
        "    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    source_anchors = dict(_VERIFIER._ACTIVE_SOURCE_SHA256_BY_PHASE)
    source_anchors[1] = {
        phase1_relative: (
            _VERIFIER.sha256(phase1_source.read_bytes()).hexdigest()
        )
    }
    monkeypatch.setattr(
        _VERIFIER,
        "_ACTIVE_SOURCE_SHA256_BY_PHASE",
        source_anchors,
    )

    _VERIFIER.verify_gate_source_isolation(tmp_path, manifest)


def test_phase1_failure_and_threat_snapshots_are_append_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept later metadata with new anchors and unchanged Phase 0 slices."""
    manifest, _payload = _phase1_manifest(tmp_path, monkeypatch)
    requirements = _VERIFIER.load_requirements(
        _FIXTURES / "requirements_traceability.json",
        manifest,
        repo_root=_ROOT,
    )
    requirement_ids = frozenset(item.id for item in requirements)
    failure_anchors = dict(_VERIFIER._FAILURE_STRUCTURE_BY_PHASE)
    failure_anchors[1] = failure_anchors[0]
    monkeypatch.setattr(
        _VERIFIER,
        "_FAILURE_STRUCTURE_BY_PHASE",
        failure_anchors,
    )
    failure = _read("failure_matrix.json")
    failure["current_phase"] = 1
    _resign(failure, "matrix_sha256")
    failure_path = tmp_path / "failure-phase1.json"
    _write(failure_path, failure)
    matrix = _VERIFIER.load_failure_matrix(
        failure_path,
        manifest=manifest,
        requirement_ids=requirement_ids,
    )
    assert len(matrix.cells) == 99

    threat_anchors = dict(_VERIFIER._THREAT_STRUCTURE_BY_PHASE)
    threat_anchors[1] = threat_anchors[0]
    monkeypatch.setattr(
        _VERIFIER,
        "_THREAT_STRUCTURE_BY_PHASE",
        threat_anchors,
    )
    threats = _read("threat_model.json")
    threats["current_phase"] = 1
    _resign(threats, "threat_model_sha256")
    threat_path = tmp_path / "threat-phase1.json"
    _write(threat_path, threats)
    _VERIFIER._validate_threat_model(
        threat_path,
        manifest=manifest,
        requirement_ids=requirement_ids,
    )


def test_phase1_resign_cannot_rewrite_phase0_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject rewritten Phase 0 history beneath a valid Phase 1 anchor."""
    _manifest_value, payload = _phase1_manifest(tmp_path, monkeypatch)
    nodes = payload["nodes"]
    history = payload["activation_history"]
    assert isinstance(nodes, list)
    assert isinstance(history, list)
    phase0_nodes = [
        node
        for node in nodes
        if isinstance(node, dict) and node.get("active_from_phase") == 0
    ]
    later_nodes = [node for node in nodes if node not in phase0_nodes]
    nodes[:] = [*reversed(phase0_nodes), *later_nodes]
    phase0_ids = [node["node_id"] for node in reversed(phase0_nodes)]
    phase1_ids = [
        *phase0_ids,
        *[
            node["node_id"]
            for node in later_nodes
            if isinstance(node, dict) and node.get("active_from_phase") == 1
        ],
    ]
    for entry, node_ids in zip(history, (phase0_ids, phase1_ids), strict=True):
        assert isinstance(entry, dict)
        entry["node_ids"] = node_ids
        entry["sha256"] = _VERIFIER._text_digest(tuple(node_ids))
    _resign(payload, "manifest_sha256")
    path = tmp_path / "history-drift.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="activation history differs from its immutable phase anchor",
    ):
        _VERIFIER.load_manifest(path)


def test_acceptance_rejects_unanchored_phase0_replacement(
    tmp_path: Path,
) -> None:
    """Reject a locally signed replacement appended to frozen history."""
    payload = _read("acceptance_manifest.json")
    nodes = payload["nodes"]
    replacements = payload["replacements"]
    assert isinstance(nodes, list)
    assert isinstance(replacements, list)
    old = nodes[0]
    target = nodes[1]
    assert isinstance(old, dict)
    assert isinstance(target, dict)
    replacement = {
        "phase": 0,
        "old_node_id": old["node_id"],
        "replacement_node_ids": [target["node_id"]],
        "reviewed_by": "test-review",
        "evidence": "test-evidence",
    }
    replacements.append(replacement)
    _resign(payload, "manifest_sha256")
    path = tmp_path / "acceptance-replacement-drift.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="replacement history anchors are not append-only",
    ):
        _VERIFIER.load_manifest(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("schema", "schema_version must be 1"),
        (
            "history",
            "activation history must preserve every implemented phase",
        ),
        ("replacement", "replacement history anchors are not append-only"),
    ),
)
def test_acceptance_reuses_complete_type_manifest_validation(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Reject re-signed type schema, history, and replacement drift."""
    payload = _read("type_contract_manifest.json")
    if mutation == "schema":
        payload["schema_version"] = 2
    elif mutation == "history":
        payload["activation_history"] = []
    else:
        replacements = payload["replacements"]
        assert isinstance(replacements, list)
        replacement = {
            "phase": 0,
            "old_fixture_id": "phase0-contract-positive",
            "replacement_fixture_ids": [
                "phase0-identity-interchange-negative"
            ],
            "reviewed_by": "test-review",
            "evidence": "test-evidence",
        }
        replacements.append(replacement)
    _resign(payload, "manifest_sha256")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    _write(fixtures / "type_contract_manifest.json", payload)

    with pytest.raises(_VERIFIER.ConversationAcceptanceError, match=message):
        _VERIFIER._validate_type_manifest(fixtures, 0, _ROOT)


def test_acceptance_type_boundary_rejects_source_byte_drift(
    tmp_path: Path,
) -> None:
    """Bind acceptance to shared external type-source byte anchors."""
    source = _ROOT / "tests" / "conversation_type_contracts"
    destination = tmp_path / "tests" / "conversation_type_contracts"
    copytree(source, destination)
    positive = destination / "phase0_positive.py"
    positive.write_text(
        positive.read_text(encoding="utf-8") + "\nVALUE = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="type fixture source digest changed",
    ):
        _VERIFIER._validate_type_manifest(_FIXTURES, 0, tmp_path)


def test_active_source_drift_is_rejected_by_external_hash(
    tmp_path: Path,
) -> None:
    """Reject a re-signed manifest backed by changed active test bytes."""
    manifest = _manifest()
    relative_paths = {
        "scripts/contract_gate.py",
        "scripts/contract_startup/avalan_contract_gate_plugin.py",
        "scripts/contract_startup/sitecustomize.py",
        "scripts/run_conversation_contract_gate.py",
        "scripts/verify_conversation_acceptance.py",
        "scripts/verify_conversation_types.py",
        "tests/conversation_contract_gate_test.py",
        "tests/conversation_phase0_contract_test.py",
        "tests/conversation_response_dormancy_test.py",
    }
    for relative in relative_paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(_ROOT / relative, destination)
    changed = tmp_path / "tests" / "conversation_phase0_contract_test.py"
    changed.write_text(
        changed.read_text(encoding="utf-8") + "\nVALUE = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="active acceptance source digest changed",
    ):
        _VERIFIER.verify_gate_source_isolation(tmp_path, manifest)


def test_manifest_rejects_duplicate_nodes(tmp_path: Path) -> None:
    """Reject duplicate evidence even when the document remains valid JSON."""
    payload = _read("acceptance_manifest.json")
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    nodes.append(deepcopy(nodes[0]))
    _resign(payload, "manifest_sha256")
    path = tmp_path / "acceptance.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="duplicate acceptance node ID",
    ):
        _VERIFIER.load_manifest(path)


def test_manifest_rejects_dimension_drift(tmp_path: Path) -> None:
    """Reject removal of one mandatory acceptance dimension."""
    payload = _read("acceptance_manifest.json")
    dimensions = payload["required_dimensions"]
    assert isinstance(dimensions, dict)
    provider = dimensions["provider"]
    assert isinstance(provider, list)
    provider.pop()
    _resign(payload, "manifest_sha256")
    path = tmp_path / "acceptance.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="mandatory dimension inventory changed",
    ):
        _VERIFIER.load_manifest(path)


def test_manifest_rejects_resigned_semantic_drift(tmp_path: Path) -> None:
    """Reject structurally valid drift signed only by the mutable fixture."""
    payload = _read("acceptance_manifest.json")
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["surface"] = "resigned-but-unreviewed"
    _resign(payload, "manifest_sha256")
    path = tmp_path / "acceptance.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="independent phase anchor",
    ):
        _VERIFIER.load_manifest(path)


def test_requirement_catalog_rejects_missing_ordinal(tmp_path: Path) -> None:
    """Reject a catalog with fewer than all 144 normative occurrences."""
    manifest = _manifest()
    payload = _read("requirements_traceability.json")
    requirements = payload["requirements"]
    assert isinstance(requirements, list)
    requirements.pop()
    payload["catalog_sha256"] = _VERIFIER.canonical_sha256(requirements)
    path = tmp_path / "requirements.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="does not contain every normative occurrence",
    ):
        _VERIFIER.load_requirements(path, manifest, repo_root=_ROOT)


def test_requirement_catalog_rejects_resigned_paraphrase_drift(
    tmp_path: Path,
) -> None:
    """Bind the complete paraphrased catalog outside its own signature."""
    manifest = _manifest()
    payload = _read("requirements_traceability.json")
    requirements = payload["requirements"]
    assert isinstance(requirements, list)
    requirement = requirements[0]
    assert isinstance(requirement, dict)
    requirement["paraphrase"] = "A locally re-signed substitute."
    payload["catalog_sha256"] = _VERIFIER.canonical_sha256(requirements)
    path = tmp_path / "requirements.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="independent Phase 0 anchor",
    ):
        _VERIFIER.load_requirements(path, manifest, repo_root=_ROOT)


def test_failure_matrix_rejects_missing_cartesian_cell(
    tmp_path: Path,
) -> None:
    """Reject a resigned matrix missing one boundary/surface cell."""
    manifest = _manifest()
    requirements = _VERIFIER.load_requirements(
        _FIXTURES / "requirements_traceability.json",
        manifest,
        repo_root=_ROOT,
    )
    payload = _read("failure_matrix.json")
    cells = payload["cells"]
    assert isinstance(cells, list)
    cells.pop()
    _resign(payload, "matrix_sha256")
    path = tmp_path / "matrix.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="complete Cartesian inventory",
    ):
        _VERIFIER.load_failure_matrix(
            path,
            manifest=manifest,
            requirement_ids=frozenset(item.id for item in requirements),
        )


def test_failure_matrix_rejects_resigned_semantic_drift(
    tmp_path: Path,
) -> None:
    """Reject a complete matrix whose rationale was locally re-signed."""
    manifest = _manifest()
    requirements = _VERIFIER.load_requirements(
        _FIXTURES / "requirements_traceability.json",
        manifest,
        repo_root=_ROOT,
    )
    payload = _read("failure_matrix.json")
    cells = payload["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell["rationale"] = "Locally re-signed semantic drift."
    _resign(payload, "matrix_sha256")
    path = tmp_path / "matrix.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="immutable phase anchor",
    ):
        _VERIFIER.load_failure_matrix(
            path,
            manifest=manifest,
            requirement_ids=frozenset(item.id for item in requirements),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "expected_dispatch_count",
            2,
            "closed zero-or-one inventory",
        ),
        ("public_mapping", "future_state", "closed Phase 0 inventory"),
    ),
)
def test_failure_matrix_rejects_values_outside_closed_inventories(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Reject unknown count and state values even after local re-signing."""
    manifest = _manifest()
    requirements = _VERIFIER.load_requirements(
        _FIXTURES / "requirements_traceability.json",
        manifest,
        repo_root=_ROOT,
    )
    payload = _read("failure_matrix.json")
    cells = payload["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell[field] = value
    _resign(payload, "matrix_sha256")
    path = tmp_path / "matrix.json"
    _write(path, payload)

    with pytest.raises(_VERIFIER.ConversationAcceptanceError, match=message):
        _VERIFIER.load_failure_matrix(
            path,
            manifest=manifest,
            requirement_ids=frozenset(item.id for item in requirements),
        )


def test_provider_fixture_rejects_premature_dispatch() -> None:
    """Reject dispatch before provider activation evidence exists."""
    payload = _read("provider_conformance.json")
    payload["production_dispatch_enabled"] = True

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="must not advertise or dispatch",
    ):
        _VERIFIER._validate_provider_conformance(payload)


def test_threat_model_rejects_missing_required_threat(tmp_path: Path) -> None:
    """Reject a resigned threat model missing one required attack class."""
    manifest = _manifest()
    requirements = _VERIFIER.load_requirements(
        _FIXTURES / "requirements_traceability.json",
        manifest,
        repo_root=_ROOT,
    )
    payload = _read("threat_model.json")
    threats = payload["threats"]
    assert isinstance(threats, list)
    threats.pop()
    _resign(payload, "threat_model_sha256")
    path = tmp_path / "threats.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="threat inventory is incomplete",
    ):
        _VERIFIER._validate_threat_model(
            path,
            manifest=manifest,
            requirement_ids=frozenset(item.id for item in requirements),
        )


def test_threat_model_rejects_resigned_semantic_drift(
    tmp_path: Path,
) -> None:
    """Reject an attack description changed with only a local signature."""
    manifest = _manifest()
    requirements = _VERIFIER.load_requirements(
        _FIXTURES / "requirements_traceability.json",
        manifest,
        repo_root=_ROOT,
    )
    payload = _read("threat_model.json")
    threats = payload["threats"]
    assert isinstance(threats, list)
    threat = threats[0]
    assert isinstance(threat, dict)
    threat["attack"] = "Locally re-signed threat drift."
    _resign(payload, "threat_model_sha256")
    path = tmp_path / "threats.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="immutable phase anchor",
    ):
        _VERIFIER._validate_threat_model(
            path,
            manifest=manifest,
            requirement_ids=frozenset(item.id for item in requirements),
        )


def test_deterministic_fixture_requires_its_canonical_digest() -> None:
    """Reject deterministic evidence whose bytes outgrow its signature."""
    payload = _read("deterministic_fixtures.json")
    payload["schema_version"] = 2

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="not the version 1 inventory",
    ):
        _VERIFIER._validate_deterministic_fixtures(payload)
