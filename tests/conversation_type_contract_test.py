"""Exercise strict conversation static type-contract verification."""

from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from pathlib import Path
from shutil import copytree
from sys import modules
from sys import path as sys_path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "conversation"


def _load_verifier() -> ModuleType:
    """Return the conversation type verifier module."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    name = "_conversation_type_verifier"
    spec = spec_from_file_location(
        name, _ROOT / "scripts" / "verify_conversation_types.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_VERIFIER = _load_verifier()


def _read_manifest() -> dict[str, object]:
    """Return a mutable type-contract manifest."""
    value: object = loads(
        (_FIXTURES / "type_contract_manifest.json").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return {str(key): item for key, item in value.items()}


def _write(path: Path, value: object) -> None:
    """Write deterministic JSON fixture evidence."""
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def _resign(payload: dict[str, object]) -> None:
    """Update the mutable manifest signature after a test mutation."""
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    payload["manifest_sha256"] = _VERIFIER.canonical_sha256(canonical)


def test_phase0_type_fixtures_match_exact_mypy_diagnostics() -> None:
    """Run one positive and two negative strict type fixtures exactly."""
    manifest = _VERIFIER.verify_conversation_types(
        _FIXTURES / "type_contract_manifest.json",
        repo_root=_ROOT,
        through_phase=0,
    )

    assert manifest.current_phase == 0
    assert len(manifest.fixtures) == 3
    assert [fixture.kind for fixture in manifest.fixtures] == [
        "positive",
        "negative",
        "negative",
    ]


def test_phase1_type_payload_history_and_source_anchors_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept a retained Phase 0 tombstone and reviewed Phase 1 target."""
    source_root = _ROOT / "tests" / "conversation_type_contracts"
    destination = tmp_path / "tests" / "conversation_type_contracts"
    copytree(source_root, destination)
    phase1_relative = "tests/conversation_type_contracts/phase1_positive.py"
    phase1_source = tmp_path / phase1_relative
    phase1_source.write_text("VALUE: int = 1\n", encoding="utf-8")
    source_sha256 = _VERIFIER.sha256(phase1_source.read_bytes()).hexdigest()
    payload = _read_manifest()
    payload["current_phase"] = 1
    fixtures = payload["fixtures"]
    history = payload["activation_history"]
    replacements = payload["replacements"]
    assert isinstance(fixtures, list)
    assert isinstance(history, list)
    assert isinstance(replacements, list)
    old_fixture = fixtures[0]
    assert isinstance(old_fixture, dict)
    old_fixture["lifecycle"] = "replaced"
    phase1_fixture = {
        "id": "phase1-contract-positive",
        "kind": "positive",
        "lifecycle": "active",
        "active_from_phase": 1,
        "path": phase1_relative,
        "source_sha256": source_sha256,
        "expected_diagnostics": [],
    }
    fixtures.append(phase1_fixture)
    fixture_ids = [
        fixture["id"] for fixture in fixtures if isinstance(fixture, dict)
    ]
    assert all(isinstance(identifier, str) for identifier in fixture_ids)
    history_sha256 = _VERIFIER.sha256(
        "\n".join(fixture_ids).encode("utf-8")
    ).hexdigest()
    phase1_history = {
        "phase": 1,
        "fixture_ids": fixture_ids,
        "sha256": history_sha256,
    }
    history.append(phase1_history)
    replacement = {
        "phase": 1,
        "old_fixture_id": old_fixture["id"],
        "replacement_fixture_ids": [phase1_fixture["id"]],
        "reviewed_by": "phase1-review",
        "evidence": "phase1-type-replacement",
    }
    replacements.append(replacement)
    _resign(payload)
    fixture_anchors = dict(_VERIFIER._TYPE_FIXTURE_PAYLOAD_SHA256_BY_PHASE)
    normalized_phase1_fixture = {
        key: value
        for key, value in phase1_fixture.items()
        if key != "lifecycle"
    }
    phase1_fixture_payload = [normalized_phase1_fixture]
    fixture_anchors[1] = _VERIFIER.canonical_sha256(phase1_fixture_payload)
    history_anchors = dict(_VERIFIER._TYPE_ACTIVATION_HISTORY_BY_PHASE)
    history_anchors[1] = history_sha256
    replacement_anchors = dict(_VERIFIER._TYPE_REPLACEMENT_HISTORY_BY_PHASE)
    replacement_anchors[1] = (1, _VERIFIER.canonical_sha256([replacement]))
    source_anchors = dict(_VERIFIER._TYPE_SOURCE_SHA256_BY_PHASE)
    source_anchors[1] = {phase1_relative: source_sha256}
    monkeypatch.setattr(
        _VERIFIER,
        "_TYPE_FIXTURE_PAYLOAD_SHA256_BY_PHASE",
        fixture_anchors,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_TYPE_ACTIVATION_HISTORY_BY_PHASE",
        history_anchors,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_TYPE_REPLACEMENT_HISTORY_BY_PHASE",
        replacement_anchors,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_TYPE_SOURCE_SHA256_BY_PHASE",
        source_anchors,
    )
    path = tmp_path / "type-phase1.json"
    _write(path, payload)

    manifest = _VERIFIER.load_manifest(path)
    _VERIFIER.validate_type_source_phase_anchors(manifest, tmp_path)

    assert manifest.current_phase == 1
    assert len(manifest.fixtures) == 4
    assert manifest.fixtures[0].lifecycle == "replaced"
    assert manifest.fixtures[-1].lifecycle == "active"
    assert (
        _VERIFIER._TYPE_FIXTURE_PAYLOAD_SHA256_BY_PHASE[0]
        == _VERIFIER._PHASE0_TYPE_FIXTURE_PAYLOAD_SHA256
    )

    old_fixture["lifecycle"] = "active"
    _resign(payload)
    attack_path = tmp_path / "type-phase1-revival.json"
    _write(attack_path, payload)
    with pytest.raises(
        _VERIFIER.ConversationTypeContractError,
        match="replaced type records and reviewed ledger entries differ",
    ):
        _VERIFIER.load_manifest(attack_path)


def test_type_history_cannot_rewrite_phase0_with_current_resign(
    tmp_path: Path,
) -> None:
    """Reject reordered Phase 0 type history under a new manifest digest."""
    payload = _read_manifest()
    fixtures = payload["fixtures"]
    history = payload["activation_history"]
    assert isinstance(fixtures, list)
    assert isinstance(history, list)
    fixtures.reverse()
    fixture_ids = [
        fixture["id"] for fixture in fixtures if isinstance(fixture, dict)
    ]
    entry = history[0]
    assert isinstance(entry, dict)
    entry["fixture_ids"] = fixture_ids
    entry["sha256"] = _VERIFIER.sha256(
        "\n".join(fixture_ids).encode("utf-8")
    ).hexdigest()
    _resign(payload)
    path = tmp_path / "type-history-drift.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationTypeContractError,
        match="type activation history differs from its immutable phase",
    ):
        _VERIFIER.load_manifest(path)


def test_type_manifest_rejects_empty_negative_diagnostics(
    tmp_path: Path,
) -> None:
    """Reject a negative type fixture with no expected diagnostics."""
    payload = _read_manifest()
    fixtures = payload["fixtures"]
    assert isinstance(fixtures, list)
    negative = fixtures[1]
    assert isinstance(negative, dict)
    negative["expected_diagnostics"] = []
    path = tmp_path / "types.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationTypeContractError,
        match="expected diagnostics must be non-empty",
    ):
        _VERIFIER.load_manifest(path)


def test_type_manifest_rejects_resigned_fixture_drift(tmp_path: Path) -> None:
    """Reject validly shaped fixture drift with a local signature."""
    payload = _read_manifest()
    fixtures = payload["fixtures"]
    assert isinstance(fixtures, list)
    fixture = fixtures[0]
    assert isinstance(fixture, dict)
    fixture["source_sha256"] = "0" * 64
    _resign(payload)
    path = tmp_path / "types.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.ConversationTypeContractError,
        match="independent phase anchor",
    ):
        _VERIFIER.load_manifest(path)


def test_type_runner_rejects_changed_fixture_bytes(tmp_path: Path) -> None:
    """Reject fixture changes before invoking the static checker."""
    source = _ROOT / "tests" / "conversation_type_contracts"
    destination = tmp_path / "tests" / "conversation_type_contracts"
    copytree(source, destination)
    positive = destination / "phase0_positive.py"
    positive.write_text(
        positive.read_text(encoding="utf-8") + "\nVALUE = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        _VERIFIER.ConversationTypeContractError,
        match="source digest changed",
    ):
        _VERIFIER.verify_conversation_types(
            _FIXTURES / "type_contract_manifest.json",
            repo_root=tmp_path,
            through_phase=0,
        )


def test_type_runner_rejects_unimplemented_phase() -> None:
    """Reject a requested phase beyond the tracked static inventory."""
    with pytest.raises(
        _VERIFIER.ConversationTypeContractError,
        match="through-phase must be implemented",
    ):
        _VERIFIER.verify_conversation_types(
            _FIXTURES / "type_contract_manifest.json",
            repo_root=_ROOT,
            through_phase=1,
        )
