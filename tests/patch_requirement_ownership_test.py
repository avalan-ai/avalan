"""Exercise Phase 0 requirement ownership without loading the whole bundle."""

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
    """Load the standalone ownership validator with its local dependencies."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    name = "_patch_requirement_ownership_verifier"
    spec = spec_from_file_location(
        name, _ROOT / "scripts" / "verify_patch_acceptance.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_VERIFIER = _load_verifier()


def _read(path: Path) -> dict[str, object]:
    """Read one mutable ownership fixture payload."""
    value = loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, payload: object) -> None:
    """Write one deterministic temporary fixture payload."""
    path.write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resign_requirements(payload: dict[str, object]) -> None:
    """Recalculate the immutable requirement catalog digest."""
    payload["catalog_sha256"] = _VERIFIER.canonical_sha256(
        {
            "record_layout": payload["record_layout"],
            "requirements": payload["requirements"],
        }
    )


def _resign_manifest(payload: dict[str, object]) -> None:
    """Recalculate the immutable acceptance-manifest digest."""
    payload["manifest_sha256"] = _VERIFIER.canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_sha256"
        }
    )


def _write_bundle(
    directory: Path,
    requirements: dict[str, object],
    manifest: dict[str, object],
) -> tuple[Path, Path]:
    """Write the trace and acceptance records needed by ownership checks."""
    requirements_path = directory / "requirements_traceability.json"
    manifest_path = directory / "acceptance_manifest.json"
    _write(requirements_path, requirements)
    _write(manifest_path, manifest)
    return requirements_path, manifest_path


def test_phase0_requirements_have_exact_semantic_evidence() -> None:
    """Require 1,017 records with bounded active executable ownership."""
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")
    ownership = _VERIFIER._validate_requirements(
        _FIXTURES / "requirements_traceability.json", manifest, _ROOT
    )

    assert len(ownership) == 1017
    assert len(manifest.active_nodes(0)) == 15
    assert all(
        item.implementation_artifact.startswith("scripts/")
        for item in ownership.values()
        if item.owning_phase == 0
    )


def test_phase0_requirement_rejects_fixture_as_implementation(
    tmp_path: Path,
) -> None:
    """Reject an active requirement that masquerades a fixture as code."""
    requirements = _read(_FIXTURES / "requirements_traceability.json")
    manifest = _read(_FIXTURES / "acceptance_manifest.json")
    records = requirements["requirements"]
    assert isinstance(records, list) and isinstance(records[0], list)
    records[0][9] = "tests/fixtures/patch/acceptance_manifest.json"
    _resign_requirements(requirements)
    requirements_path, manifest_path = _write_bundle(
        tmp_path, requirements, manifest
    )

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="active requirement implementation artifact is not executable",
    ):
        _VERIFIER._validate_requirements(
            requirements_path, _VERIFIER.load_manifest(manifest_path), _ROOT
        )


def test_phase0_requirement_rejects_active_evidence_for_future_work(
    tmp_path: Path,
) -> None:
    """Reject a future requirement that borrows an active Phase 0 node."""
    requirements = _read(_FIXTURES / "requirements_traceability.json")
    manifest = _read(_FIXTURES / "acceptance_manifest.json")
    records = requirements["requirements"]
    nodes = manifest["nodes"]
    assert isinstance(records, list) and isinstance(nodes, list)
    future = records[87]
    active = nodes[0]
    assert isinstance(future, list) and isinstance(active, dict)
    future[11] = active["node_id"]
    _resign_requirements(requirements)
    requirements_path, manifest_path = _write_bundle(
        tmp_path, requirements, manifest
    )

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="requirement is unowned or differs from acceptance evidence",
    ):
        _VERIFIER._validate_requirements(
            requirements_path, _VERIFIER.load_manifest(manifest_path), _ROOT
        )


def test_phase0_requirement_rejects_generic_bundle_load_evidence(
    tmp_path: Path,
) -> None:
    """Reject a Phase 0 requirement owned by a generic contract bundle load."""
    requirements = _read(_FIXTURES / "requirements_traceability.json")
    manifest = _read(_FIXTURES / "acceptance_manifest.json")
    nodes = manifest["nodes"]
    assert isinstance(nodes, list) and isinstance(nodes[0], dict)
    generic = (
        "tests/patch_acceptance_verifier_test.py::"
        "test_patch_acceptance_positive_load"
    )
    nodes[0]["node_id"] = generic
    records = requirements["requirements"]
    assert isinstance(records, list)
    for record in records[:6]:
        assert isinstance(record, list)
        record[11] = generic
    _resign_manifest(manifest)
    _resign_requirements(requirements)
    requirements_path, manifest_path = _write_bundle(
        tmp_path, requirements, manifest
    )

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="generic bundle load",
    ):
        _VERIFIER._validate_requirements(
            requirements_path, _VERIFIER.load_manifest(manifest_path), _ROOT
        )


def test_phase0_requirement_rejects_node_cap_bypass(tmp_path: Path) -> None:
    """Reject a ninth active requirement assigned to one evidence node."""
    requirements = _read(_FIXTURES / "requirements_traceability.json")
    manifest = _read(_FIXTURES / "acceptance_manifest.json")
    records = requirements["requirements"]
    nodes = manifest["nodes"]
    assert isinstance(records, list) and isinstance(nodes, list)
    owner = nodes[0]
    assert isinstance(owner, dict)
    owner["requirement_ids"] = [f"PATCH-R-{item:04d}" for item in range(1, 10)]
    displaced = nodes[1]
    assert isinstance(displaced, dict)
    displaced["requirement_ids"] = [
        f"PATCH-R-{item:04d}" for item in range(10, 13)
    ]
    node_id = owner["node_id"]
    assert isinstance(node_id, str)
    for record in records[:9]:
        assert isinstance(record, list)
        record[11] = node_id
    _resign_manifest(manifest)
    _resign_requirements(requirements)
    requirements_path, manifest_path = _write_bundle(
        tmp_path, requirements, manifest
    )

    with pytest.raises(
        _VERIFIER.PatchAcceptanceError,
        match="owns too many requirements",
    ):
        _VERIFIER._validate_requirements(
            requirements_path, _VERIFIER.load_manifest(manifest_path), _ROOT
        )


def test_phase9_requirement_rejects_same_symbol_from_wrong_artifact() -> None:
    """Bind a Phase 9 runtime reference to its exact imported source path."""
    node_id = (
        "tests/patch/phase_9_contract_test.py::"
        "test_patch_phase_9_malformed_provider_arguments_cannot_fall_back"
    )
    function = _VERIFIER._test_node_function(_ROOT, node_id)
    bindings = _VERIFIER._test_node_import_bindings(_ROOT, node_id)

    assert bindings["ToolCall"] == ("src/avalan/entities.py", "ToolCall")
    assert _VERIFIER._function_uses_bound_symbol(
        function,
        bindings,
        "src/avalan/entities.py",
        "ToolCall",
    )
    assert not _VERIFIER._function_uses_bound_symbol(
        function,
        bindings,
        "src/avalan/model/response/parsers/tool.py",
        "ToolCall",
    )
