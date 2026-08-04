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


def _resign_scoped(payload: dict[str, object]) -> None:
    """Update one whole-object canonical digest value."""
    canonical = dict(payload)
    digest = cast(dict[str, object], canonical.pop("canonical_digest"))
    digest["value"] = _VERIFIER.canonical_sha256(canonical)


def _resign_activation(payload: dict[str, object]) -> None:
    """Update one activation review signature and canonical digest."""
    signed = dict(payload)
    signature = cast(dict[str, object], signed.pop("review_signature"))
    signed.pop("canonical_digest")
    signature["value"] = _VERIFIER.canonical_sha256(signed)
    _resign_scoped(payload)


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


def test_phase3_execution_inherits_only_the_postgresql_test_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the owned database capability into acceptance subprocesses."""
    observed: list[tuple[str, ...]] = []

    def execute_nodes(
        root: Path,
        node_ids: tuple[str, ...],
        *,
        junit_path: Path,
        expected_evidence: dict[str, str],
        inherited_names: tuple[str, ...],
    ) -> None:
        assert root == _ROOT
        assert len(node_ids) == 42
        assert junit_path.name == "pytest.xml"
        assert set(expected_evidence) == set(node_ids)
        observed.append(inherited_names)

    monkeypatch.setattr(_VERIFIER, "execute_pytest_nodes", execute_nodes)

    _VERIFIER.verify_acceptance(
        _FIXTURES / "acceptance_manifest.phase3.json",
        repo_root=_ROOT,
        through_phase=3,
        execute=True,
    )

    assert observed == [(_VERIFIER.POSTGRESQL_TEST_DSN_ENV,)]


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
    phase5_transitions = {
        relative: transition
        for relative, transition in _VERIFIER._phase5_provider_transitions(
            _ROOT
        ).items()
        if relative in relative_paths
    }
    monkeypatch.setattr(
        _VERIFIER,
        "_phase5_provider_transitions",
        lambda _root: phase5_transitions,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_phase6_provider_transitions",
        lambda _root: {},
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_phase7_provider_transitions",
        lambda _root: {},
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_phase8_provider_transitions",
        lambda _root: {},
    )
    phase9_transitions = {
        relative: transition
        for relative, transition in _VERIFIER._phase9_provider_transitions(
            _ROOT
        ).items()
        if relative in relative_paths
    }
    monkeypatch.setattr(
        _VERIFIER,
        "_phase9_provider_transitions",
        lambda _root: phase9_transitions,
    )
    phase10_transitions = {
        relative: transition
        for relative, transition in _VERIFIER._phase10_provider_transitions(
            _ROOT
        ).items()
        if relative in relative_paths
    }
    monkeypatch.setattr(
        _VERIFIER,
        "_phase10_provider_transitions",
        lambda _root: phase10_transitions,
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


def test_replacement_cover_rejects_extra_requirement_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject overlapping replacement evidence that adds ownership."""
    manifest, payload, _old_node_id, target_node_id = (
        _phase1_replacement_manifest(tmp_path, monkeypatch)
    )
    dynamic_manifest = cast(Any, manifest)
    target = next(
        node
        for node in dynamic_manifest.nodes
        if node.node_id == target_node_id
    )
    attacked_target = replace(
        target,
        requirement_ids=(*target.requirement_ids, "CONV-N-999"),
    )
    attacked_nodes = tuple(
        attacked_target if node.node_id == target_node_id else node
        for node in dynamic_manifest.nodes
    )
    history = payload["activation_history"]
    assert isinstance(history, list)
    activation_history = tuple(
        tuple(entry["node_ids"])
        for entry in history
        if isinstance(entry, dict) and isinstance(entry.get("node_ids"), list)
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="exact nonempty cover",
    ):
        _VERIFIER._validate_replacement_transitions(
            dynamic_manifest.replacements,
            attacked_nodes,
            activation_history,
        )


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
    phase5_transitions = {
        relative: transition
        for relative, transition in _VERIFIER._phase5_provider_transitions(
            _ROOT
        ).items()
        if relative in relative_paths
    }
    monkeypatch.setattr(
        _VERIFIER,
        "_phase5_provider_transitions",
        lambda _root: phase5_transitions,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_phase6_provider_transitions",
        lambda _root: {},
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_phase7_provider_transitions",
        lambda _root: {},
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_phase8_provider_transitions",
        lambda _root: {},
    )
    phase9_transitions = {
        relative: transition
        for relative, transition in _VERIFIER._phase9_provider_transitions(
            _ROOT
        ).items()
        if relative in relative_paths
    }
    monkeypatch.setattr(
        _VERIFIER,
        "_phase9_provider_transitions",
        lambda _root: phase9_transitions,
    )
    phase10_transitions = {
        relative: transition
        for relative, transition in _VERIFIER._phase10_provider_transitions(
            _ROOT
        ).items()
        if relative in relative_paths
    }
    monkeypatch.setattr(
        _VERIFIER,
        "_phase10_provider_transitions",
        lambda _root: phase10_transitions,
    )

    _VERIFIER.verify_gate_source_isolation(tmp_path, manifest)


def test_phase5_provider_transition_target_has_independent_byte_anchor(
    tmp_path: Path,
) -> None:
    """Reject a re-signed transition that rewrites its target bytes."""
    payload = _read("provider_transition.phase5.json")
    transitions = payload["transitions"]
    assert isinstance(transitions, list)
    target = transitions[0]
    assert isinstance(target, dict)
    target["to_size"] = int(cast(int, target["to_size"])) + 1
    target["to_sha256"] = "0" * 64
    _resign(payload, "canonical_sha256")
    transition_path = (
        tmp_path
        / "tests"
        / "fixtures"
        / "conversation"
        / "provider_transition.phase5.json"
    )
    transition_path.parent.mkdir(parents=True)
    _write(transition_path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="target differs from its independent anchor",
    ):
        _VERIFIER._phase5_provider_transitions(tmp_path)


def test_phase6_lifecycle_transition_pins_exact_bytes() -> None:
    """Bind lifecycle transition evidence to executable fallback bytes."""
    expected = (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        21_208,
        "edea170ab18293bcb91dcef38a95e7359cc0ee099636472cf24a52b1e288e943",
    )

    assert (
        _VERIFIER._PHASE6_PROVIDER_SOURCE_BYTE_ANCHORS[
            "src/avalan/conversation/lifecycle.py"
        ]
        == expected
    )
    assert (
        _VERIFIER._phase6_provider_transitions(_ROOT)[
            "src/avalan/conversation/lifecycle.py"
        ]
        == expected
    )


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
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setattr(
        _VERIFIER,
        "_phase5_provider_transitions",
        lambda _root: {},
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_phase6_provider_transitions",
        lambda _root: {},
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_phase7_provider_transitions",
        lambda _root: {},
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_phase8_provider_transitions",
        lambda _root: {},
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_phase9_provider_transitions",
        lambda _root: {},
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_phase10_provider_transitions",
        lambda _root: {},
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


@pytest.mark.parametrize(
    ("node_id", "dimension", "value", "message"),
    (
        (
            (
                "tests/conversation/agent_integration_e2e_test.py::"
                "test_parent_tool_effect_failure_fences_unsafe_retry"
            ),
            "execution",
            "no_tool",
            "must declare one_tool execution",
        ),
        (
            (
                "tests/interaction/stores/conversation_atomic_pgsql_test.py::"
                "test_fresh_worker_applies_atomic_conversation_answer_once"
            ),
            "scenario_lifecycle",
            "same_process",
            "must declare fresh_process lifecycle",
        ),
        (
            (
                "tests/conversation/agent_integration_pgsql_test.py::"
                "test_pgsql_tool_boundaries_recover_without_duplicate_effect"
            ),
            "local_retention",
            "direct_process_local",
            "must declare durable_local retention",
        ),
    ),
)
def test_phase8_manifest_rejects_contradictory_evidence_axes(
    tmp_path: Path,
    node_id: str,
    dimension: str,
    value: str,
    message: str,
) -> None:
    """Reject re-signed no-tool, process-local, and same-process claims."""
    payload = _read("acceptance_manifest.phase8.json")
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    node = next(
        item
        for item in nodes
        if isinstance(item, dict) and item.get("node_id") == node_id
    )
    node[dimension] = [value]
    _resign(payload, "manifest_sha256")
    path = tmp_path / "acceptance-phase8-axis-drift.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match=message,
    ):
        _VERIFIER.load_manifest(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint_commit_count", 1),
        ("retry_decision", "never"),
    ),
)
def test_phase8_failure_matrix_rejects_agent_tool_contradictions(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Reject re-signed agent tool semantics that contradict evidence."""
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "acceptance_manifest.phase8.json"
    )
    requirements = _VERIFIER.load_requirements(
        _FIXTURES / "requirements_traceability.json",
        manifest,
        repo_root=_ROOT,
    )
    payload = _read("failure_matrix.phase8.json")
    cells = payload["cells"]
    assert isinstance(cells, list)
    cell = next(
        item
        for item in cells
        if isinstance(item, dict)
        and item.get("id") == "tool_effect--agent_sdk"
    )
    cell[field] = value
    _resign(payload, "matrix_sha256")
    path = tmp_path / "failure-matrix-phase8-agent-tool-drift.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="agent tool failure semantics contradict",
    ):
        _VERIFIER.load_failure_matrix(
            path,
            manifest=manifest,
            requirement_ids=frozenset(item.id for item in requirements),
        )


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


def test_phase11_threat_model_rejects_placeholder_ownership(
    tmp_path: Path,
) -> None:
    """Reject locally signed placeholder ownership in active controls."""
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "acceptance_manifest.phase11.json"
    )
    requirements = _VERIFIER.load_requirements(
        _FIXTURES / "requirements_traceability.json",
        manifest,
        repo_root=_ROOT,
    )
    payload = _read("threat_model.phase11.json")
    threats = payload["threats"]
    assert isinstance(threats, list)
    threat = threats[-1]
    assert isinstance(threat, dict)
    owners = threat["control_owners"]
    assert isinstance(owners, list)
    owner = owners[0]
    assert isinstance(owner, dict)
    owner["owner"] = "todo-runtime-owner"
    _resign(payload, "threat_model_sha256")
    path = tmp_path / "threats.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="placeholder text",
    ):
        _VERIFIER._validate_threat_model(
            path,
            manifest=manifest,
            requirement_ids=frozenset(item.id for item in requirements),
        )


def test_phase11_threat_model_rejects_missing_operational_response(
    tmp_path: Path,
) -> None:
    """Reject active hardening threats missing operational response data."""
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "acceptance_manifest.phase11.json"
    )
    requirements = _VERIFIER.load_requirements(
        _FIXTURES / "requirements_traceability.json",
        manifest,
        repo_root=_ROOT,
    )
    payload = _read("threat_model.phase11.json")
    threats = payload["threats"]
    assert isinstance(threats, list)
    threat = threats[-1]
    assert isinstance(threat, dict)
    threat.pop("incident_response")
    _resign(payload, "threat_model_sha256")
    path = tmp_path / "threats.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="invalid keys",
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


def _phase12_candidate_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    *,
    anchor_candidate: bool = True,
    anchor_mappings: bool = True,
) -> Path:
    """Write and optionally re-anchor one resigned candidate variant."""
    _resign(payload, "canonical_sha256")
    path = tmp_path / "acceptance_candidate.phase12.json"
    _write(path, payload)
    monkeypatch.setattr(
        _VERIFIER,
        "_PHASE12_TRACEABILITY_CANDIDATE_PATH",
        str(path),
    )
    if anchor_candidate:
        monkeypatch.setattr(
            _VERIFIER,
            "_PHASE12_TRACEABILITY_CANDIDATE_BYTE_SHA256",
            _VERIFIER.sha256(path.read_bytes()).hexdigest(),
        )
        monkeypatch.setattr(
            _VERIFIER,
            "_PHASE12_TRACEABILITY_CANDIDATE_CANONICAL_SHA256",
            payload["canonical_sha256"],
        )
    if anchor_mappings:
        monkeypatch.setattr(
            _VERIFIER,
            "_PHASE12_TRACEABILITY_MAPPING_CANONICAL_SHA256",
            _VERIFIER.canonical_sha256(
                {
                    "public_e2e_inventory": payload["public_e2e_inventory"],
                    "normative_requirements": payload[
                        "normative_requirements"
                    ],
                }
            ),
        )
    return path


def _phase12_live_identity(
    live_results: dict[str, object],
    **changes: str,
) -> Any:
    """Return the exact identity encoded by the tracked Terra receipt."""
    azure = cast(dict[str, object], live_results["azure_openai_matrix"])
    results = cast(list[dict[str, object]], azure["results"])
    terra = next(
        row for row in results if row["deployment"] == "gpt-5.6-terra"
    )
    receipt = cast(dict[str, object], terra["tracked_cli_receipt"])
    values = {
        "provider_family": cast(str, receipt["provider_family"]),
        "profile": cast(str, receipt["model_or_deployment"]),
        "revision": cast(str, receipt["model_or_deployment_revision"]),
        "structural_observations_digest": cast(
            str,
            receipt["structural_observations_digest"],
        ),
    }
    values.update(changes)
    return _VERIFIER._Phase12LiveReceiptIdentity(**values)


def _phase12_live_proof_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    activation: dict[str, object],
    live_results: dict[str, object],
) -> Path:
    """Write and source-anchor one linked activation/live evidence pair."""
    _resign_scoped(live_results)
    live_path = tmp_path / "live_conformance_results.phase12.json"
    _write(live_path, live_results)
    live_digest = cast(dict[str, object], live_results["canonical_digest"])[
        "value"
    ]
    live_link = cast(dict[str, object], activation["live_evidence"])
    live_link["path"] = live_path.name
    live_link["byte_sha256"] = _VERIFIER.sha256(
        live_path.read_bytes()
    ).hexdigest()
    live_link["canonical_digest"] = live_digest
    _resign_activation(activation)
    activation_path = tmp_path / "activation_manifest.phase12.json"
    _write(activation_path, activation)
    activation_digest = cast(
        dict[str, object], activation["canonical_digest"]
    )["value"]
    monkeypatch.setattr(
        _VERIFIER,
        "_PHASE12_ACTIVATION_DECISION_PATH",
        activation_path.name,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_PHASE12_ACTIVATION_DECISION_BYTE_SHA256",
        _VERIFIER.sha256(activation_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_PHASE12_ACTIVATION_DECISION_CANONICAL_SHA256",
        activation_digest,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_PHASE12_LIVE_RESULTS_PATH",
        live_path.name,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_PHASE12_LIVE_RESULTS_BYTE_SHA256",
        _VERIFIER.sha256(live_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_PHASE12_LIVE_RESULTS_CANONICAL_SHA256",
        live_digest,
    )
    return tmp_path


def test_phase12_live_proof_resolves_to_exact_current_receipt() -> None:
    """Resolve the full-digest proof to one exact Terra receipt identity."""
    activation = _read("activation_manifest.phase12.json")
    live_results = _read("live_conformance_results.phase12.json")
    proof_ids = cast(list[str], activation["live_proof_ids"])
    identity = _phase12_live_identity(live_results)

    assert proof_ids == [identity.proof_id]
    assert proof_ids[0].endswith(
        ":structural-sha256:"
        "f76c0c145f3775c5e445cb55efb3c9cb5b9293a01695e6850f3764ee6badc5f3"
    )
    _VERIFIER._validate_phase12_live_proof_resolution(_ROOT)


def test_phase12_live_proof_rejects_stale_structural_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject the prior prefix-only proof after locally valid resealing."""
    activation = _read("activation_manifest.phase12.json")
    activation["live_proof_ids"] = [
        "azure-openai-gpt-5.6-terra-2026-07-09-dd2482cf"
    ]
    root = _phase12_live_proof_root(
        tmp_path,
        monkeypatch,
        activation,
        _read("live_conformance_results.phase12.json"),
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="live proof",
    ):
        _VERIFIER._validate_phase12_live_proof_resolution(root)


def test_phase12_live_proof_rejects_resigned_activation_without_source_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject locally resigned proof drift absent source-owned authority."""
    activation = _read("activation_manifest.phase12.json")
    activation["live_proof_ids"] = ["locally-resigned-unknown-proof"]
    _resign_activation(activation)
    path = tmp_path / "activation_manifest.phase12.json"
    _write(path, activation)
    monkeypatch.setattr(
        _VERIFIER,
        "_PHASE12_ACTIVATION_DECISION_PATH",
        path.name,
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="activation decision byte anchor is invalid",
    ):
        _VERIFIER._validate_phase12_live_proof_resolution(tmp_path)


def test_phase12_live_proof_rejects_unknown_well_formed_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a well-formed proof absent from the current receipt set."""
    activation = _read("activation_manifest.phase12.json")
    activation["live_proof_ids"] = [
        f"{_VERIFIER._PHASE12_LIVE_PROOF_PREFIX}:identity-sha256:"
        f"{'f' * 64}:structural-sha256:{'e' * 64}"
    ]
    root = _phase12_live_proof_root(
        tmp_path,
        monkeypatch,
        activation,
        _read("live_conformance_results.phase12.json"),
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="live proof does not resolve",
    ):
        _VERIFIER._validate_phase12_live_proof_resolution(root)


def test_phase12_live_proof_rejects_mismatched_referenced_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a full digest that disagrees with its bound receipt identity."""
    activation = _read("activation_manifest.phase12.json")
    proof_id = cast(list[str], activation["live_proof_ids"])[0]
    activation["live_proof_ids"] = [
        f"{proof_id.rsplit(':', maxsplit=1)[0]}:{'0' * 64}"
    ]
    root = _phase12_live_proof_root(
        tmp_path,
        monkeypatch,
        activation,
        _read("live_conformance_results.phase12.json"),
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="live proof digest does not match",
    ):
        _VERIFIER._validate_phase12_live_proof_resolution(root)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("provider_family", "openai"),
        ("profile", "gpt-5.6-sol"),
        ("revision", "2026-07-10"),
        ("structural_observations_digest", "0" * 64),
    ),
)
def test_phase12_live_proof_rejects_wrong_receipt_identity_or_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    """Reject a proof bound to any wrong provider/profile/revision/digest."""
    activation = _read("activation_manifest.phase12.json")
    live_results = _read("live_conformance_results.phase12.json")
    activation["live_proof_ids"] = [
        _phase12_live_identity(live_results, **{field: value}).proof_id
    ]
    root = _phase12_live_proof_root(
        tmp_path,
        monkeypatch,
        activation,
        live_results,
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="live proof does not resolve",
    ):
        _VERIFIER._validate_phase12_live_proof_resolution(root)


def test_phase12_live_proof_rejects_duplicate_identifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a duplicate activation proof before receipt resolution."""
    activation = _read("activation_manifest.phase12.json")
    proof_id = cast(list[str], activation["live_proof_ids"])[0]
    activation["live_proof_ids"] = [proof_id, proof_id]
    root = _phase12_live_proof_root(
        tmp_path,
        monkeypatch,
        activation,
        _read("live_conformance_results.phase12.json"),
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="duplicate or noncanonical",
    ):
        _VERIFIER._validate_phase12_live_proof_resolution(root)


def test_phase12_live_proof_rejects_ambiguous_duplicate_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject one proof resolving to duplicate current receipt identities."""
    activation = _read("activation_manifest.phase12.json")
    live_results = _read("live_conformance_results.phase12.json")
    azure = cast(dict[str, object], live_results["azure_openai_matrix"])
    results = cast(list[dict[str, object]], azure["results"])
    terra = next(
        row for row in results if row["deployment"] == "gpt-5.6-terra"
    )
    results.append(deepcopy(terra))
    live_results["completed_full_matrix_profile_count"] = 2
    root = _phase12_live_proof_root(
        tmp_path,
        monkeypatch,
        activation,
        live_results,
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="resolves ambiguously",
    ):
        _VERIFIER._validate_phase12_live_proof_resolution(root)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("provider_family", "openai"),
        ("model_or_deployment", "gpt-5.6-sol"),
        ("model_or_deployment_revision", "2026-07-10"),
    ),
)
def test_phase12_live_proof_rejects_receipt_profile_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    """Reject a receipt whose identity differs from its containing profile."""
    activation = _read("activation_manifest.phase12.json")
    live_results = _read("live_conformance_results.phase12.json")
    azure = cast(dict[str, object], live_results["azure_openai_matrix"])
    results = cast(list[dict[str, object]], azure["results"])
    terra = next(
        row for row in results if row["deployment"] == "gpt-5.6-terra"
    )
    receipt = cast(dict[str, object], terra["tracked_cli_receipt"])
    receipt[field] = value
    root = _phase12_live_proof_root(
        tmp_path,
        monkeypatch,
        activation,
        live_results,
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="receipt identity differs from its provider profile",
    ):
        _VERIFIER._validate_phase12_live_proof_resolution(root)


def test_phase12_traceability_candidate_is_exact_and_non_promoting() -> None:
    """Validate exact planned, public, normative, and blocker evidence."""
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "acceptance_manifest.phase11.json"
    )

    _VERIFIER._validate_phase12_traceability_candidate(_ROOT, manifest)


def test_phase12_candidate_rejects_resigned_self_digest_without_source_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a locally resigned candidate absent source-owned authority."""
    payload = _read("acceptance_candidate.phase12.json")
    payload["candidate_state"] = "locally_resigned_substitution"
    _phase12_candidate_path(
        tmp_path,
        monkeypatch,
        payload,
        anchor_candidate=False,
        anchor_mappings=False,
    )
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "acceptance_manifest.phase11.json"
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="byte anchor is invalid",
    ):
        _VERIFIER._validate_phase12_traceability_candidate(_ROOT, manifest)


def test_phase12_mapping_anchor_rejects_unrelated_same_class_active_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject same-class active-node substitution in a normative mapping."""
    payload = _read("acceptance_candidate.phase12.json")
    requirements = cast(
        list[dict[str, object]], payload["normative_requirements"]
    )
    evidence = cast(list[dict[str, object]], requirements[1]["evidence"])
    evidence[0] = {
        "node_id": (
            "tests/conversation/compaction_e2e_test.py::"
            "test_tool_cycles_across_two_boundaries_keep_exact_final_order"
        ),
        "evidence_class": "wire",
        "evidence_state": "active",
    }
    _phase12_candidate_path(
        tmp_path,
        monkeypatch,
        payload,
        anchor_mappings=False,
    )
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "acceptance_manifest.phase11.json"
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="mapping authority digest is invalid",
    ):
        _VERIFIER._validate_phase12_traceability_candidate(_ROOT, manifest)


def test_phase12_candidate_rejects_resigned_live_outcome_overclaim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep exact external live blocker states non-interchangeable."""
    payload = _read("acceptance_candidate.phase12.json")
    blockers = cast(list[dict[str, object]], payload["external_blockers"])
    blockers[0][
        "state"
    ] = "requires_operator_authority_credentials_and_cost_acknowledgement"
    _phase12_candidate_path(tmp_path, monkeypatch, payload)
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "acceptance_manifest.phase11.json"
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="external blocker is not precise",
    ):
        _VERIFIER._validate_phase12_traceability_candidate(_ROOT, manifest)


def test_phase12_candidate_rejects_resigned_broad_planned_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject broad provider or dimension labels absent from the node."""
    payload = _read("acceptance_candidate.phase12.json")
    planned = cast(list[dict[str, object]], payload["planned_nodes"])
    providers = cast(list[str], planned[0]["provider_families"])
    providers.append("incapable_generic_compatible")
    _phase12_candidate_path(tmp_path, monkeypatch, payload)
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "acceptance_manifest.phase11.json"
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="label-only or broad claims",
    ):
        _VERIFIER._validate_phase12_traceability_candidate(_ROOT, manifest)


def test_phase12_candidate_rejects_resigned_provider_e2e_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject CONV-E2E-015 without exact native and compatible evidence."""
    payload = _read("acceptance_candidate.phase12.json")
    inventory = cast(list[dict[str, object]], payload["public_e2e_inventory"])
    evidence = cast(list[dict[str, object]], inventory[-1]["evidence"])
    evidence.pop()
    _phase12_candidate_path(tmp_path, monkeypatch, payload)
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "acceptance_manifest.phase11.json"
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="CONV-E2E-015 provider evidence is not exact",
    ):
        _VERIFIER._validate_phase12_traceability_candidate(_ROOT, manifest)


def test_phase12_candidate_rejects_resigned_live_requirement_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the live-provider normative requirement externally planned."""
    payload = _read("acceptance_candidate.phase12.json")
    requirements = cast(
        list[dict[str, object]], payload["normative_requirements"]
    )
    live_evidence = cast(list[dict[str, object]], requirements[6]["evidence"])
    live_evidence[0] = {
        "node_id": _VERIFIER._PHASE12_MATRIX_NODE_ID,
        "evidence_class": "matrix",
        "evidence_state": "candidate_deterministic",
    }
    _phase12_candidate_path(tmp_path, monkeypatch, payload)
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "acceptance_manifest.phase11.json"
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="live provider verification must remain externally planned",
    ):
        _VERIFIER._validate_phase12_traceability_candidate(_ROOT, manifest)


def test_phase12_candidate_rejects_resigned_active_evidence_overclaim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an executable historical node relabeled as currently active."""
    payload = _read("acceptance_candidate.phase12.json")
    inventory = cast(list[dict[str, object]], payload["public_e2e_inventory"])
    evidence = cast(list[dict[str, object]], inventory[2]["evidence"])
    evidence[0]["evidence_state"] = "active"
    _phase12_candidate_path(tmp_path, monkeypatch, payload)
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "acceptance_manifest.phase11.json"
    )

    with pytest.raises(
        _VERIFIER.ConversationAcceptanceError,
        match="labels inactive evidence active",
    ):
        _VERIFIER._validate_phase12_traceability_candidate(_ROOT, manifest)
