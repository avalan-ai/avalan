"""Verify append-only Phase 1 conversation snapshot selection."""

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import loads
from pathlib import Path
from sys import modules
from sys import path as sys_path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests/fixtures/conversation"
_PHASE0_BYTES = {
    "acceptance_manifest.json": (
        "b046e73982c85dccb6f6d16ff09e036ec10e05cc4b91aca376b8c74118df3a05"
    ),
    "failure_matrix.json": (
        "2c8443c83c966f0f05cc0fd54956aae5d6865e173733eb8a33fe1de3ec1fa751"
    ),
    "threat_model.json": (
        "f1448630d12320eaf70e7394dbb063a7caaf65bbf22221a35a72e2c1b48c0120"
    ),
    "type_contract_manifest.json": (
        "87fc9af5ccc661b70010d6399b4a0235b05ff322a87306ed870b1dacfcd7416f"
    ),
}


def _load(name: str, relative: str) -> ModuleType:
    """Load one gate module from the tracked script directory."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    spec = spec_from_file_location(name, _ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


def _payload(name: str) -> dict[str, object]:
    """Return one strict object-shaped fixture payload."""
    value: object = loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return {str(key): item for key, item in value.items()}


def test_phase2_defaults_select_the_complete_snapshot_family() -> None:
    """Select Phase 2 acceptance, type, failure, and threat snapshots."""
    type_verifier = _load(
        "_phase1_type_verifier",
        "scripts/verify_conversation_types.py",
    )
    acceptance = _load(
        "_phase1_acceptance_verifier",
        "scripts/verify_conversation_acceptance.py",
    )
    runner = _load(
        "_phase1_contract_runner",
        "scripts/run_conversation_contract_gate.py",
    )

    acceptance_path = acceptance.default_manifest_path()
    assert acceptance_path.name == "acceptance_manifest.phase2.json"
    assert (
        type_verifier.default_manifest_path().name
        == "type_contract_manifest.phase2.json"
    )
    assert (
        acceptance.companion_fixture_path(
            acceptance_path, "failure_matrix"
        ).name
        == "failure_matrix.phase2.json"
    )
    assert (
        acceptance.companion_fixture_path(acceptance_path, "threat_model").name
        == "threat_model.phase2.json"
    )
    assert (
        acceptance.companion_fixture_path(
            acceptance_path, "type_contract_manifest"
        ).name
        == "type_contract_manifest.phase2.json"
    )
    assert acceptance.load_manifest(acceptance_path).current_phase == 2
    assert (
        type_verifier.load_manifest(
            type_verifier.default_manifest_path()
        ).current_phase
        == 2
    )
    assert runner._CONVERSATION_CURRENT_PHASE == 2
    runner._validate_through_phase(_ROOT, 2)
    with pytest.raises(runner.ContractGateError):
        runner._validate_through_phase(_ROOT, 3)


def test_phase1_acceptance_validates_selected_companions() -> None:
    """Validate every selected Phase 1 companion before pytest execution."""
    acceptance = _load(
        "_phase1_acceptance_validation",
        "scripts/verify_conversation_acceptance.py",
    )
    manifest = acceptance.verify_acceptance(
        _FIXTURES / "acceptance_manifest.phase1.json",
        repo_root=_ROOT,
        through_phase=1,
        execute=False,
    )
    assert manifest.current_phase == 1
    assert len(manifest.active_nodes(1)) == 21


def test_phase0_manifests_remain_byte_pinned_and_history_is_retained() -> None:
    """Preserve Phase 0 bytes and append Phase 1 history monotonically."""
    for name, expected in _PHASE0_BYTES.items():
        assert sha256((_FIXTURES / name).read_bytes()).hexdigest() == expected

    acceptance0 = _payload("acceptance_manifest.json")
    acceptance1 = _payload("acceptance_manifest.phase1.json")
    history0 = acceptance0["activation_history"]
    history1 = acceptance1["activation_history"]
    nodes0 = acceptance0["nodes"]
    nodes1 = acceptance1["nodes"]
    assert isinstance(history0, list) and isinstance(history1, list)
    assert isinstance(nodes0, list) and isinstance(nodes1, list)
    assert history1[0] == history0[0]
    assert [
        node
        for node in nodes1
        if isinstance(node, dict) and node.get("active_from_phase") == 0
    ] == [
        node
        for node in nodes0
        if isinstance(node, dict) and node.get("active_from_phase") == 0
    ]

    types0 = _payload("type_contract_manifest.json")
    types1 = _payload("type_contract_manifest.phase1.json")
    type_history0 = types0["activation_history"]
    type_history1 = types1["activation_history"]
    fixtures0 = types0["fixtures"]
    fixtures1 = types1["fixtures"]
    assert isinstance(type_history0, list) and isinstance(type_history1, list)
    assert isinstance(fixtures0, list) and isinstance(fixtures1, list)
    assert type_history1[0] == type_history0[0]
    assert fixtures1[: len(fixtures0)] == fixtures0

    for stem, digest_key in (
        ("failure_matrix", "matrix_sha256"),
        ("threat_model", "threat_model_sha256"),
    ):
        base = _payload(f"{stem}.json")
        snapshot = _payload(f"{stem}.phase1.json")
        base.pop("current_phase")
        snapshot.pop("current_phase")
        base.pop(digest_key)
        snapshot.pop(digest_key)
        assert snapshot == base
