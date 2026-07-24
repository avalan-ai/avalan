"""Exercise strict structured-input type-contract verification."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from pathlib import Path
from sys import modules
from sys import path as sys_path
from types import ModuleType
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "input"


def _load_verifier() -> ModuleType:
    """Return the type-contract verifier module."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    name = "_input_contract_type_verifier"
    spec = spec_from_file_location(
        name, _ROOT / "scripts" / "verify_input_types.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_VERIFIER = _load_verifier()


def test_acceptance_only_phase_lag_rejects_new_type_obligations() -> None:
    """Allow one type-neutral phase and reject wider or typed phase drift."""
    manifest = _VERIFIER.load_manifest(
        _FIXTURES / "type_contract_manifest.json"
    )

    _VERIFIER._validate_acceptance_phase(manifest, 7)

    obligation = replace(manifest.fixtures[0], active_from_phase=7)
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError,
        match="acceptance-only phase without new type obligations",
    ):
        _VERIFIER._validate_acceptance_phase(
            replace(
                manifest,
                fixtures=(*manifest.fixtures, obligation),
            ),
            7,
        )
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError,
        match="acceptance-only phase without new type obligations",
    ):
        _VERIFIER._validate_acceptance_phase(manifest, 8)


def _read_manifest() -> dict[str, Any]:
    """Return a mutable type-contract manifest copy."""
    value = loads(
        (_FIXTURES / "type_contract_manifest.json").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: object) -> None:
    """Write one deterministic JSON document."""
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def _snapshot_digest(values: list[str]) -> str:
    """Return one type snapshot digest."""
    return sha256("\n".join(values).encode()).hexdigest()


def _ledger_digest(payload: dict[str, Any]) -> str:
    """Return the type activation ledger digest."""
    value = {
        "activation_snapshots": payload["activation_snapshots"],
        "planned_replacements": payload["planned_replacements"],
        "replacements": payload["replacements"],
    }
    return sha256(
        dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()


def _replacement_manifest() -> dict[str, Any]:
    """Return a valid synthetic type fixture replacement ledger."""
    new_path = "tests/input_type_contracts/replacement_positive.py"
    snapshots = [
        {
            "phase": 0,
            "fixture_ids": ["historical-positive"],
            "sha256": _snapshot_digest(["historical-positive"]),
        },
        {
            "phase": 1,
            "fixture_ids": ["replacement-positive"],
            "sha256": _snapshot_digest(["replacement-positive"]),
        },
    ]
    return {
        "schema_version": 1,
        "feature": "structured_task_input",
        "current_phase": 1,
        "activation_history": [
            {"phase": 0, "fixture_ids": []},
            {"phase": 1, "fixture_ids": ["replacement-positive"]},
        ],
        "activation_snapshots": snapshots,
        "planned_replacements": [],
        "replacements": [
            {
                "phase": 1,
                "old_fixture_id": "historical-positive",
                "replacement_fixture_ids": ["replacement-positive"],
                "reviewed_by": "synthetic-ledger-reviewer",
                "evidence": "synthetic reviewed replacement",
            }
        ],
        "fixtures": [
            {
                "id": "replacement-positive",
                "kind": "positive",
                "lifecycle": "active",
                "active_from_phase": 1,
                "path": new_path,
                "expected_diagnostics": [],
            }
        ],
    }


def _planned_replacement_manifest() -> dict[str, Any]:
    """Return a valid synthetic planned-fixture replacement ledger."""
    phase_zero = ["base-positive"]
    phase_one = ["base-positive", "replacement-positive", "other-positive"]
    return {
        "schema_version": 1,
        "feature": "structured_task_input",
        "current_phase": 1,
        "activation_history": [
            {"phase": 0, "fixture_ids": ["base-positive"]},
            {
                "phase": 1,
                "fixture_ids": [
                    "replacement-positive",
                    "other-positive",
                ],
            },
        ],
        "activation_snapshots": [
            {
                "phase": 0,
                "fixture_ids": phase_zero,
                "sha256": _snapshot_digest(phase_zero),
            },
            {
                "phase": 1,
                "fixture_ids": phase_one,
                "sha256": _snapshot_digest(phase_one),
            },
        ],
        "planned_replacements": [
            {
                "phase": 1,
                "old_fixture_id": "planned-negative",
                "replacement_fixture_ids": ["replacement-positive"],
                "reason": "The frozen public callable has a different name.",
                "reviewed_by": "synthetic-reviewer",
                "evidence": "strict synthetic mypy evidence",
            }
        ],
        "replacements": [],
        "fixtures": [
            {
                "id": "base-positive",
                "kind": "positive",
                "lifecycle": "active",
                "active_from_phase": 0,
                "path": "tests/input_type_contracts/base.py",
                "expected_diagnostics": [],
            },
            {
                "id": "planned-negative",
                "kind": "negative",
                "lifecycle": "replaced",
                "active_from_phase": 1,
                "path": "tests/input_type_contracts/replacement.py",
                "expected_diagnostics": ["historical diagnostic"],
            },
            {
                "id": "replacement-positive",
                "kind": "positive",
                "lifecycle": "active",
                "active_from_phase": 1,
                "path": "tests/input_type_contracts/replacement.py",
                "expected_diagnostics": [],
            },
            {
                "id": "other-positive",
                "kind": "positive",
                "lifecycle": "active",
                "active_from_phase": 1,
                "path": "tests/input_type_contracts/other.py",
                "expected_diagnostics": [],
            },
        ],
    }


def test_type_contract_manifest_and_runner_are_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run active mypy evidence and reject diagnostic or ledger drift."""
    manifest = _read_manifest()
    monkeypatch.setattr(
        _VERIFIER, "_EXPECTED_TYPE_LEDGER_SHA256", _ledger_digest(manifest)
    )
    real_path = _FIXTURES / "type_contract_manifest.json"
    loaded = _VERIFIER.verify_input_types(
        real_path,
        repo_root=_ROOT,
        through_phase=6,
        acceptance_manifest_path=_FIXTURES / "acceptance_manifest.json",
    )
    assert loaded.current_phase == 6
    active_fixture_ids = [
        fixture.id
        for fixture in loaded.fixtures
        if fixture.lifecycle == "active"
    ]
    assert len(active_fixture_ids) == 34
    assert active_fixture_ids == [
        "deterministic-fixtures-positive",
        "canonical-answers-positive",
        "async-handler-positive",
        "synchronous-handler-negative",
        "untyped-handler-result-negative",
        "broker-identity-interchange-negative",
        "store-revision-interchange-negative",
        "controller-identity-interchange-negative",
        "synchronous-resumer-negative",
        "candidate-resolution-variants-negative",
        "strict-resolution-variants-negative",
        "typed-resolution-payload-negative",
        "input-required-identity-negative",
        "unchecked-any-leak-negative",
        "model-capability-catalog-positive",
        "tool-catalog-interchange-negative",
        "capability-binding-identity-interchange-negative",
        "capability-payload-result-shape-negative",
        "execution-state-positive",
        "execution-context-interchange-negative",
        "synchronous-execution-id-factory-negative",
        "portable-continuation-positive",
        "continuation-store-cas-negative",
        "provider-snapshot-identity-interchange-negative",
        "synchronous-continuation-runtime-resolver-negative",
        "task-target-outcome-positive",
        "task-target-outcome-raw-negative",
        "task-target-suspension-as-success-negative",
        "public-sdk-attached-positive",
        "public-sdk-construction-positive",
        "public-sdk-sync-handler-negative",
        "public-sdk-untyped-submission-negative",
        "public-sdk-result-conversion-negative",
        "public-sdk-ref-interchange-negative",
    ]

    invalid = deepcopy(manifest)
    negative = next(
        fixture
        for fixture in invalid["fixtures"]
        if fixture["kind"] == "negative"
    )
    negative["expected_diagnostics"] = []
    invalid_path = tmp_path / "invalid.json"
    _write(invalid_path, invalid)
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError, match="diagnostics"
    ):
        _VERIFIER.load_manifest(invalid_path)

    replacement = _replacement_manifest()
    monkeypatch.setattr(
        _VERIFIER, "_EXPECTED_TYPE_LEDGER_SHA256", _ledger_digest(replacement)
    )
    _write(invalid_path, replacement)
    assert _VERIFIER.load_manifest(invalid_path).current_phase == 1
    replacement["replacements"] = []
    monkeypatch.setattr(
        _VERIFIER, "_EXPECTED_TYPE_LEDGER_SHA256", _ledger_digest(replacement)
    )
    _write(invalid_path, replacement)
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError, match="tombstone"
    ):
        _VERIFIER.load_manifest(invalid_path)

    root = tmp_path / "runner"
    fixture_dir = root / "tests" / "input_type_contracts"
    fixture_dir.mkdir(parents=True)
    fixture_path = fixture_dir / "bad.py"
    fixture_path.write_text('value: int = "wrong"\n', encoding="utf-8")
    runner_manifest = {
        "schema_version": 1,
        "feature": "structured_task_input",
        "current_phase": 0,
        "activation_history": [
            {"phase": 0, "fixture_ids": ["strict-negative"]}
        ],
        "activation_snapshots": [
            {
                "phase": 0,
                "fixture_ids": ["strict-negative"],
                "sha256": _snapshot_digest(["strict-negative"]),
            }
        ],
        "planned_replacements": [],
        "replacements": [],
        "fixtures": [
            {
                "id": "strict-negative",
                "kind": "negative",
                "lifecycle": "active",
                "active_from_phase": 0,
                "path": "tests/input_type_contracts/bad.py",
                "expected_diagnostics": ["intentionally wrong diagnostic"],
            }
        ],
    }
    runner_path = root / "type_manifest.json"
    acceptance_path = root / "acceptance_manifest.json"
    _write(runner_path, runner_manifest)
    _write(acceptance_path, {"current_phase": 0})
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_TYPE_LEDGER_SHA256",
        _ledger_digest(runner_manifest),
    )
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError, match="diagnostics changed"
    ):
        _VERIFIER.verify_input_types(
            runner_path,
            repo_root=root,
            through_phase=0,
            acceptance_manifest_path=acceptance_path,
        )


def test_current_type_fixture_inventory_and_sources_fail_closed(
    tmp_path: Path,
) -> None:
    """Reject omitted descriptors and any current fixture source drift."""
    raw_manifest = _read_manifest()
    loaded = _VERIFIER.load_manifest(_FIXTURES / "type_contract_manifest.json")
    omitted_id = _VERIFIER._EXPECTED_CURRENT_FIXTURE_IDS[0]
    fixtures = tuple(
        fixture for fixture in loaded.fixtures if fixture.id != omitted_id
    )
    raw_fixtures = [
        fixture
        for fixture in raw_manifest["fixtures"]
        if fixture["id"] != omitted_id
    ]
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError,
        match="reviewed type fixture inventory changed",
    ):
        _VERIFIER._validate_current_fixture_inventory(
            fixtures,
            raw_fixtures,
            6,
        )

    changed_descriptors = deepcopy(raw_manifest["fixtures"])
    changed_current = next(
        fixture
        for fixture in changed_descriptors
        if fixture["id"] == omitted_id
    )
    changed_current["path"] = (
        "tests/input_type_contracts/unreviewed_execution_state.py"
    )
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError,
        match="reviewed type fixture inventory changed",
    ):
        _VERIFIER._validate_current_fixture_inventory(
            loaded.fixtures,
            changed_descriptors,
            6,
        )

    for fixture in loaded.fixtures:
        if fixture.active_from_phase not in {
            *_VERIFIER._FROZEN_FIXTURE_INVENTORIES,
            _VERIFIER._CURRENT_BOUNDARY_PHASE,
        }:
            continue
        source = _ROOT / fixture.path
        destination = tmp_path / fixture.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    _VERIFIER._validate_current_fixture_sources(loaded, tmp_path)
    changed_source = tmp_path / next(
        fixture.path
        for fixture in loaded.fixtures
        if fixture.active_from_phase == _VERIFIER._CURRENT_BOUNDARY_PHASE
    )
    changed_source.write_text(
        changed_source.read_text(encoding="utf-8") + "# source drift\n",
        encoding="utf-8",
    )
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError,
        match="reviewed type fixture sources changed",
    ):
        _VERIFIER._validate_current_fixture_sources(loaded, tmp_path)


def test_planned_type_replacements_are_exact_and_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accept only reviewed one-time replacements of planned fixtures."""
    manifest = _planned_replacement_manifest()
    expected_digest = _ledger_digest(manifest)
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_TYPE_LEDGER_SHA256",
        expected_digest,
    )
    path = tmp_path / "planned-replacement.json"
    _write(path, manifest)

    loaded = _VERIFIER.load_manifest(path)

    assert loaded.current_phase == 1
    assert [
        fixture.id
        for fixture in loaded.fixtures
        if fixture.lifecycle == "replaced"
    ] == ["planned-negative"]

    not_planned = deepcopy(manifest)
    old = next(
        fixture
        for fixture in not_planned["fixtures"]
        if fixture["id"] == "planned-negative"
    )
    old["lifecycle"] = "active"
    old["path"] = "tests/input_type_contracts/old.py"
    _write(path, not_planned)
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError,
        match="not genuinely planned",
    ):
        _VERIFIER.load_manifest(path)

    duplicate = deepcopy(manifest)
    duplicate["planned_replacements"].append(
        deepcopy(duplicate["planned_replacements"][0])
    )
    _write(path, duplicate)
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError,
        match="replaced more than once",
    ):
        _VERIFIER.load_manifest(path)

    wrong_target = deepcopy(manifest)
    wrong_target["planned_replacements"][0]["replacement_fixture_ids"] = [
        "base-positive"
    ]
    _write(path, wrong_target)
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError,
        match="exact same-phase active fixture",
    ):
        _VERIFIER.load_manifest(path)

    for field_name, changed_value in (
        ("replacement_fixture_ids", ["other-positive"]),
        ("reason", "A different non-empty reason."),
        ("reviewed_by", "another-reviewer"),
        ("evidence", "different non-empty evidence"),
    ):
        changed = deepcopy(manifest)
        changed["planned_replacements"][0][field_name] = changed_value
        _write(path, changed)
        with pytest.raises(
            _VERIFIER.TypeContractVerificationError,
            match="ledger changed without verifier review",
        ):
            _VERIFIER.load_manifest(path)

    missing = deepcopy(manifest)
    missing["planned_replacements"] = []
    _write(path, missing)
    with pytest.raises(
        _VERIFIER.TypeContractVerificationError,
        match="does not exactly preserve",
    ):
        _VERIFIER.load_manifest(path)
