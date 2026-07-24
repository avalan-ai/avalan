"""Exercise compact structured-input acceptance verification."""

from copy import deepcopy
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from os import utime
from pathlib import Path
from sys import modules
from sys import path as sys_path
from types import ModuleType
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "input"


def _load_verifier() -> ModuleType:
    """Return the acceptance verifier module."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    name = "_compact_input_acceptance_verifier"
    spec = spec_from_file_location(
        name, _ROOT / "scripts" / "verify_input_acceptance.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_VERIFIER = _load_verifier()


def _read(name: str) -> dict[str, Any]:
    """Return one mutable fixture copy."""
    value = loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: object) -> None:
    """Write deterministic JSON."""
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def _resign(payload: dict[str, Any], field: str) -> None:
    """Update one fixture's canonical digest."""
    canonical = {key: value for key, value in payload.items() if key != field}
    payload[field] = _VERIFIER._digest(canonical)


@pytest.mark.parametrize(
    "mutation, match",
    (
        (
            lambda value: value["nodes"].append(deepcopy(value["nodes"][0])),
            "duplicate acceptance node ID",
        ),
        (
            lambda value: value["nodes"][0].update(lifecycle="planned"),
            "lifecycle disagrees",
        ),
        (
            lambda value: value["nodes"][0].update(category="unknown"),
            "category is invalid",
        ),
        (
            lambda value: value.update(schema_version=True),
            "schema_version",
        ),
    ),
)
def test_acceptance_rejects_invalid_inventory(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    """Reject malformed, duplicated, or lifecycle-invalid nodes."""
    payload = _read("acceptance_manifest.json")
    mutation(payload)
    path = tmp_path / "manifest.json"
    _write(path, payload)

    with pytest.raises(_VERIFIER.AcceptanceVerificationError, match=match):
        _VERIFIER.load_manifest(path)


def test_acceptance_rejects_na_reason_without_exact_ids(
    tmp_path: Path,
) -> None:
    """Reject overlapping rules instead of storing narrative N/A cells."""
    payload = _read("failure_matrix.json")
    payload["applicability_rules"][1]["condition_id"] = payload[
        "applicability_rules"
    ][0]["condition_id"]
    payload["applicability_rules"][1]["surface_ids"] = payload[
        "applicability_rules"
    ][0]["surface_ids"]
    _resign(payload, "matrix_sha256")
    path = tmp_path / "matrix.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="duplicate applicable failure cell",
    ):
        _VERIFIER.load_failure_matrix(path)


def test_acceptance_cli_executes_exact_synthetic_node(
    tmp_path: Path,
) -> None:
    """Use ordinary pytest collection to derive parametrized instances."""
    tests = tmp_path / "tests"
    tests.mkdir()
    path = tests / "sample_test.py"
    path.write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('value', (1, 2, 3))\n"
        "def test_value(value: int) -> None:\n"
        "    assert value > 0\n",
        encoding="utf-8",
    )
    node = _VERIFIER.AcceptanceNode(
        id="synthetic",
        category="unit",
        lifecycle="active",
        active_from_phase=0,
        requirement_ids=("INPUT-N-001",),
        node_id="tests/sample_test.py::test_value",
    )

    collected = _VERIFIER._verify_nodes((node,), tmp_path)

    assert collected == (
        "tests/sample_test.py::test_value[1]",
        "tests/sample_test.py::test_value[2]",
        "tests/sample_test.py::test_value[3]",
    )


def test_acceptance_rejects_execution_for_different_collected_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject JUnit evidence whose count hides a different test instance."""
    tests = tmp_path / "tests"
    tests.mkdir()
    path = tests / "sample_test.py"
    path.write_text(
        "def test_value() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    node = _VERIFIER.AcceptanceNode(
        id="synthetic",
        category="unit",
        lifecycle="active",
        active_from_phase=0,
        requirement_ids=("INPUT-N-001",),
        node_id="tests/sample_test.py::test_value",
    )

    def pytest_result(
        root: Path,
        arguments: tuple[str, ...],
        *,
        timeout: int,
    ) -> Any:
        assert root == tmp_path
        assert timeout > 0
        if "--collect-only" in arguments:
            return _VERIFIER.CompletedProcess(
                arguments,
                0,
                stdout="tests/sample_test.py::test_value\n",
                stderr="",
            )
        junit_argument = next(
            argument
            for argument in arguments
            if argument.startswith("--junitxml=")
        )
        junit = Path(junit_argument.split("=", 1)[1])
        junit.write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase file="tests/sample_test.py" '
            'classname="tests.sample_test" name="test_other" />'
            "</testsuite>\n",
            encoding="utf-8",
        )
        return _VERIFIER.CompletedProcess(
            arguments,
            0,
            stdout="1 passed\n",
            stderr="",
        )

    monkeypatch.setattr(_VERIFIER, "_pytest", pytest_result)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="does not match collected instance IDs",
    ):
        _VERIFIER._verify_nodes((node,), tmp_path)


def test_acceptance_rejects_pytest_non_evidence(tmp_path: Path) -> None:
    """Reject skipped tests before they can count as acceptance evidence."""
    tests = tmp_path / "tests"
    tests.mkdir()
    path = tests / "sample_test.py"
    prohibited_marker = "pytest.mark." + "skip"
    path.write_text(
        "import pytest\n\n"
        f"@{prohibited_marker}\n"
        "def test_value() -> None:\n"
        "    assert True\n",
        encoding="utf-8",
    )
    node = _VERIFIER.AcceptanceNode(
        id="synthetic",
        category="unit",
        lifecycle="active",
        active_from_phase=0,
        requirement_ids=("INPUT-N-001",),
        node_id="tests/sample_test.py::test_value",
    )

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="skipped",
    ):
        _VERIFIER._verify_nodes((node,), tmp_path)


@pytest.mark.parametrize("name", ("ex" + "ec", "com" + "pile"))
def test_acceptance_rejects_placeholder_and_execution_tricks(
    tmp_path: Path,
    name: str,
) -> None:
    """Reject dynamic-code tricks without a custom AST language."""
    tests = tmp_path / "tests"
    tests.mkdir()
    path = tests / "sample_test.py"
    path.write_text(
        f"def test_value() -> None:\n    {name}('assert True')\n",
        encoding="utf-8",
    )
    node = _VERIFIER.AcceptanceNode(
        id="synthetic",
        category="unit",
        lifecycle="active",
        active_from_phase=0,
        requirement_ids=("INPUT-N-001",),
        node_id="tests/sample_test.py::test_value",
    )

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="prohibited coverage trick",
    ):
        _VERIFIER._verify_nodes((node,), tmp_path)


def test_current_runtime_manifest_inventory_fails_closed(
    tmp_path: Path,
) -> None:
    """Reject planned/current drift directly from node phase metadata."""
    payload = _read("acceptance_manifest.json")
    current = next(
        node for node in payload["nodes"] if node["active_from_phase"] == 7
    )
    current["lifecycle"] = "planned"
    path = tmp_path / "manifest.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="lifecycle disagrees",
    ):
        _VERIFIER.load_manifest(path)


def test_current_runtime_executes_and_reports_exact_phase_nodes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Execute every current-phase node, including both public scenarios."""
    executed: tuple[Any, ...] = ()

    def verify_nodes(nodes: tuple[Any, ...], root: Path) -> tuple[str, ...]:
        nonlocal executed
        assert root == _ROOT
        executed = nodes
        return tuple(node.node_id for node in nodes)

    monkeypatch.setattr(_VERIFIER, "_require_database_harness", lambda: None)
    monkeypatch.setattr(
        _VERIFIER,
        "_validate_contract_fixtures",
        lambda *arguments: None,
    )
    monkeypatch.setattr(_VERIFIER, "_verify_nodes", verify_nodes)
    monkeypatch.setattr(
        _VERIFIER,
        "_parse_args",
        lambda: _VERIFIER.Namespace(
            through_phase=7,
            manifest=_FIXTURES / "acceptance_manifest.json",
            repo_root=_ROOT,
            runtime_only=True,
        ),
    )

    assert _VERIFIER.main() == 0

    requirements = {
        requirement_id
        for node in executed
        for requirement_id in node.requirement_ids
    }
    assert {
        "INPUT-26.1",
        "INPUT-26.2",
        "INPUT-26.3",
        "INPUT-26.6",
    } == {value for value in requirements if value.startswith("INPUT-26.")}
    assert len(executed) == 16
    assert all(node.active_from_phase == 7 for node in executed)
    assert f"nodes={len(executed)}" in capsys.readouterr().out


def test_current_runtime_file_inventory_fails_closed(
    tmp_path: Path,
) -> None:
    """Require every active pytest file to exist inside tests."""
    node = _VERIFIER.AcceptanceNode(
        id="missing",
        category="unit",
        lifecycle="active",
        active_from_phase=6,
        requirement_ids=("INPUT-N-001",),
        node_id="tests/missing_test.py::test_missing",
    )

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="does not exist",
    ):
        _VERIFIER._verify_nodes((node,), tmp_path)


def test_current_regression_classification_fails_closed(
    tmp_path: Path,
) -> None:
    """Reject weakened exact-gate invariants and stale evidence digests."""
    payload = _read("baseline_evidence.json")
    payload["invariants"]["fail_closed"] = False
    _resign(payload, "evidence_sha256")
    path = tmp_path / "evidence.json"
    _write(path, payload)
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="invariants changed",
    ):
        _VERIFIER._validate_evidence(path, manifest)


def test_failure_matrix_rules_bind_real_manifest_nodes() -> None:
    """Bind compact applicable rules to existing lifecycle-aware tests."""
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")
    requirements = _VERIFIER._validate_requirements(
        _FIXTURES / "requirements_traceability.json", manifest
    )
    surfaces, envelopes = _VERIFIER._validate_decisions(
        _FIXTURES / "contract_decisions.json"
    )

    matrix = _VERIFIER.load_failure_matrix(
        _FIXTURES / "failure_matrix.json",
        manifest=manifest,
        requirement_ids=requirements,
        decision_surface_ids=surfaces,
        public_envelope_ids=envelopes,
    )

    assert len(matrix.rules) == 169
    assert len(matrix.applicable_cells()) == 564
    assert len(matrix.all_cells() - matrix.applicable_cells()) == 696


def test_failure_matrix_rejects_resigned_transition_tampering(
    tmp_path: Path,
) -> None:
    """Reject a valid-looking transition that contradicts its condition."""
    payload = _read("failure_matrix.json")
    rule = next(
        item
        for item in payload["applicability_rules"]
        if item["condition_id"] == "INPUT-F-02"
    )
    rule["expected_transition"] = "pending->expired"
    _resign(payload, "matrix_sha256")
    path = tmp_path / "failure_matrix.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="does not match condition and surface semantics",
    ):
        _VERIFIER.load_failure_matrix(path)


def test_current_phase_requires_real_postgresql_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject current acceptance outside the provisioned PostgreSQL gate."""
    monkeypatch.delenv("AVALAN_TASK_TEST_POSTGRESQL_DSN", raising=False)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="real PostgreSQL harness",
    ):
        _VERIFIER.verify_acceptance(
            _FIXTURES / "acceptance_manifest.json",
            repo_root=_ROOT,
            through_phase=7,
        )


def test_acceptance_only_phase_lag_rejects_new_type_obligations() -> None:
    """Allow one type-neutral phase and reject wider or typed phase drift."""
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")
    type_manifest = _VERIFIER.load_type_manifest(
        _FIXTURES / "type_contract_manifest.json"
    )

    _VERIFIER._validate_type_contract_phase(manifest, type_manifest)

    obligation = replace(
        type_manifest.fixtures[0],
        active_from_phase=manifest.current_phase,
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="acceptance-only phase without new type obligations",
    ):
        _VERIFIER._validate_type_contract_phase(
            manifest,
            replace(
                type_manifest,
                fixtures=(*type_manifest.fixtures, obligation),
            ),
        )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="acceptance-only phase without new type obligations",
    ):
        _VERIFIER._validate_type_contract_phase(
            manifest,
            replace(type_manifest, current_phase=5),
        )


def test_fresh_coverage_binding_rejects_older_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject coverage evidence older than source, tests, or gate scripts."""
    source = tmp_path / "src"
    tests = tmp_path / "tests"
    scripts = tmp_path / "scripts"
    source.mkdir()
    tests.mkdir()
    scripts.mkdir()
    report = tmp_path / "coverage.json"
    report.write_text("{}\n", encoding="utf-8")
    changed = tests / "changed_test.py"
    changed.write_text("def test_changed(): pass\n", encoding="utf-8")
    utime(report, (1, 1))
    utime(changed, (2, 2))
    called = False

    def verify(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(_VERIFIER, "verify_src_coverage", verify)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="predates current",
    ):
        _VERIFIER._validate_fresh_coverage(tmp_path)
    assert called is False


def test_strict_json_rejects_duplicate_fixture_keys(tmp_path: Path) -> None:
    """Fail closed when JSON repeats a key."""
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema_version":2,"schema_version":2}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="duplicate",
    ):
        _VERIFIER.load_manifest(path)
