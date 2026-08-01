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
from xml.etree.ElementTree import Element, SubElement

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
    """Reject duplicate cells across compact applicability rules."""
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


def test_pytest_database_handoff_is_explicit_and_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward only an explicitly requested valid test database DSN."""
    dsn_name = "AVALAN_TASK_TEST_POSTGRESQL_DSN"
    dsn = "postgresql://test/avalan_task_test_0123456789abcdef"
    tests = tmp_path / "tests"
    tests.mkdir()
    path = tests / "database_environment_test.py"
    path.write_text(
        "from os import environ\n\n"
        "def test_scoped_handoff() -> None:\n"
        f"    assert environ[{dsn_name!r}] == {dsn!r}\n\n"
        "def test_unscoped_environment() -> None:\n"
        f"    assert {dsn_name!r} not in environ\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(dsn_name, dsn)

    scoped = _VERIFIER._pytest(
        tmp_path,
        ("-q", "tests/database_environment_test.py::test_scoped_handoff"),
        timeout=30,
    )
    unscoped = _VERIFIER.run_pytest(
        tmp_path,
        (
            "-q",
            "tests/database_environment_test.py::test_unscoped_environment",
        ),
        timeout=30,
    )

    assert scoped.returncode == 0, scoped.stdout + scoped.stderr
    assert unscoped.returncode == 0, unscoped.stdout + unscoped.stderr

    for value, match in (
        ("", "PostgreSQL DSN is empty"),
        (
            "postgresql://test/db?service=production",
            "malformed or ambiguous",
        ),
    ):
        monkeypatch.setenv(dsn_name, value)
        with pytest.raises(_VERIFIER.ContractGateError, match=match):
            _VERIFIER._pytest(
                tmp_path,
                (
                    "-q",
                    "tests/database_environment_test.py::test_scoped_handoff",
                ),
                timeout=30,
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


def test_acceptance_collection_failure_includes_bounded_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the useful tail of a failed pytest collection."""
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
    hidden_prefix = "not-in-bounded-tail"
    diagnostic = "ERROR: test node does not exist"

    def collection_failure(
        root: Path,
        arguments: tuple[str, ...],
        *,
        timeout: int,
    ) -> Any:
        assert root == tmp_path
        assert "--collect-only" in arguments
        assert timeout == 180
        return _VERIFIER.CompletedProcess(
            arguments,
            4,
            stdout="collection stopped\n",
            stderr=hidden_prefix + "x" * 5000 + diagnostic,
        )

    monkeypatch.setattr(_VERIFIER, "_pytest", collection_failure)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="pytest collection failed",
    ) as error_info:
        _VERIFIER._verify_nodes((node,), tmp_path)

    message = str(error_info.value)
    assert "stdout:\ncollection stopped" in message
    assert f"stderr:\n{'x' * (4000 - len(diagnostic))}{diagnostic}" in message
    assert hidden_prefix not in message


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
    """Execute every current orchestration acceptance node."""
    executed: tuple[Any, ...] = ()
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")
    current_phase = manifest.current_phase

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
            through_phase=current_phase,
            manifest=_FIXTURES / "acceptance_manifest.json",
            repo_root=_ROOT,
            runtime_only=True,
        ),
    )

    assert _VERIFIER.main() == 0

    assert executed == manifest.current_phase_nodes()
    assert len(executed) == 18
    assert not manifest.planned_nodes()
    assert all(node.active_from_phase == current_phase for node in executed)
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

    assert len(matrix.rules) == 93
    assert len(matrix.non_applicability_rules) == 17
    assert len(matrix.applicable_cells()) == 145
    assert len(matrix.non_applicable_cells()) == 1115
    assert matrix.applicable_cells().isdisjoint(matrix.non_applicable_cells())
    assert (
        matrix.applicable_cells() | matrix.non_applicable_cells()
        == matrix.all_cells()
    )


def test_contract_decisions_reject_invalid_capability_activation(
    tmp_path: Path,
) -> None:
    """Reject unsupported, unevidenced, or malformed production rows."""
    cases = (
        ("planned_enabled", "enabled capability evidence"),
        ("unsupported_provider", "unsupported provider or local"),
        ("disabled_consumer", "consumer must remain production enabled"),
        ("unknown_field", "capability row has invalid keys"),
        (
            "unadvertised_active_evidence",
            "unadvertised capability evidence must remain planned",
        ),
        ("missing_evidence", "enabled capability evidence test is missing"),
    )
    active_evidence = (
        "active:tests/model/nlp/vendor_openai_continuation_test.py::"
        "test_native_openai_model_registers_exact_durable_revision"
    )
    for case, match in cases:
        payload = _read("contract_decisions.json")
        rows = {row["id"]: row for row in payload["capability_matrix"]["rows"]}
        if case == "planned_enabled":
            evidence = "planned:src/avalan/model/nlp/text/vendor/openai.py"
            rows["provider-openai"]["evidence"] = evidence
        elif case == "unsupported_provider":
            rows["provider-anthropic"].update(
                production_advertised=True,
                evidence=active_evidence,
            )
        elif case == "disabled_consumer":
            rows["sdk-attached"]["production_advertised"] = False
        elif case == "unknown_field":
            rows["sdk-attached"]["unexpected"] = True
        elif case == "unadvertised_active_evidence":
            rows["cli-model-run-attached-tty"]["evidence"] = active_evidence
        else:
            rows["provider-openai"]["evidence"] = (
                "active:tests/model/nlp/vendor_openai_continuation_test.py::"
                "test_missing"
            )
        _resign(payload, "contract_sha256")
        path = tmp_path / f"{case}.json"
        _write(path, payload)

        with pytest.raises(
            _VERIFIER.AcceptanceVerificationError,
            match=match,
        ):
            _VERIFIER._validate_decisions(path)


def test_failure_matrix_rejects_resigned_applicability_overlap(
    tmp_path: Path,
) -> None:
    """Reject one cell declared both applicable and non-applicable."""
    payload = _read("failure_matrix.json")
    applicable = payload["applicability_rules"][0]
    surface_id = applicable["surface_ids"][0]
    rule = next(
        item
        for item in payload["non_applicability_rules"]
        if surface_id in item["surface_ids"]
    )
    rule["condition_ids"].append(applicable["condition_id"])
    _resign(payload, "matrix_sha256")
    path = tmp_path / "failure_matrix.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="applicable and non-applicable failure cells overlap",
    ):
        _VERIFIER.load_failure_matrix(path)


def test_failure_matrix_rejects_resigned_unexplained_omission(
    tmp_path: Path,
) -> None:
    """Reject one cell omitted from both explicit matrix partitions."""
    payload = _read("failure_matrix.json")
    payload["non_applicability_rules"][0]["condition_ids"].pop()
    _resign(payload, "matrix_sha256")
    path = tmp_path / "failure_matrix.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="failure matrix has unexplained cells",
    ):
        _VERIFIER.load_failure_matrix(path)


def test_failure_matrix_rejects_active_planned_na_evidence(
    tmp_path: Path,
) -> None:
    """Reject planned prose masquerading as active N/A evidence."""
    payload = _read("failure_matrix.json")
    payload["non_applicability_rules"][0]["evidence"] = "planned: decide later"
    _resign(payload, "matrix_sha256")
    path = tmp_path / "failure_matrix.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="active non-applicability evidence cannot be planned",
    ):
        _VERIFIER.load_failure_matrix(path)


def test_failure_matrix_rejects_missing_na_evidence_path(
    tmp_path: Path,
) -> None:
    """Reject a non-applicability claim backed by a missing file."""
    payload = _read("failure_matrix.json")
    evidence = "tests/input/missing_evidence_test.py"
    payload["non_applicability_rules"][0]["evidence"] = evidence
    _resign(payload, "matrix_sha256")
    path = tmp_path / "failure_matrix.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="evidence path is missing",
    ):
        _VERIFIER.load_failure_matrix(path)


def test_failure_matrix_rejects_missing_na_json_fragment(
    tmp_path: Path,
) -> None:
    """Reject a non-applicability claim backed by a missing JSON path."""
    payload = _read("failure_matrix.json")
    payload["non_applicability_rules"][0]["evidence"] = (
        "tests/fixtures/input/contract_decisions.json"
        "#error_status.public_failure_surfaces"
    )
    _resign(payload, "matrix_sha256")
    path = tmp_path / "failure_matrix.json"
    _write(path, payload)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="JSON fragment is missing",
    ):
        _VERIFIER.load_failure_matrix(path)


@pytest.mark.parametrize(
    ("reference", "match"),
    (
        (
            "src/avalan/cli/commands/agent.py#missing_runtime",
            "evidence symbol is missing",
        ),
        (
            "tests/input_contract_test.py::test_missing_contract",
            "evidence test is missing",
        ),
    ),
)
def test_failure_matrix_rejects_missing_na_python_symbol(
    tmp_path: Path,
    reference: str,
    match: str,
) -> None:
    """Reject missing Python symbols in both evidence reference forms."""
    payload = _read("failure_matrix.json")
    payload["non_applicability_rules"][0]["evidence"] = reference
    _resign(payload, "matrix_sha256")
    path = tmp_path / "failure_matrix.json"
    _write(path, payload)

    with pytest.raises(_VERIFIER.AcceptanceVerificationError, match=match):
        _VERIFIER.load_failure_matrix(path)


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


@pytest.mark.parametrize(
    "case, match",
    (
        ("missing", "needs one evidence"),
        ("malformed", "not strict JSON"),
        ("multi", "must own one surface"),
        ("duplicate", "duplicate dynamic"),
        ("wrong_owner", "unassigned failure evidence"),
        ("wrong_surface", "differs from instance"),
        ("wrong_claim", "differs from rule"),
        ("schema_invalid", "violates frozen schema"),
    ),
)
def test_failure_evidence_rejects_unowned_or_inaccurate_rows(
    case: str,
    match: str,
) -> None:
    """Reject every way one JUnit property can overclaim a matrix cell."""
    node = "tests/sample_test.py::test_value"
    rule = _VERIFIER.ApplicabilityRule(
        condition_id="INPUT-F-01",
        surface_ids=("surface",),
        active_from_phase=0,
        evidence_claim=(
            "INPUT-F-01",
            "created",
            "unavailable",
            "test.unavailable.v1",
            "exit",
            "69",
            0,
            0,
        ),
        negative_e2e_node=node,
    )
    matrix = _VERIFIER.FailureMatrix(
        surfaces=(
            _VERIFIER.FailureSurface(id="surface", active_from_phase=0),
        ),
        conditions=(
            _VERIFIER.FailureCondition(
                id="INPUT-F-01",
                active_from_phase=0,
                requirement_id="INPUT-N-106",
            ),
        ),
        rules=(rule,),
    )
    observation = {
        "condition_id": "INPUT-F-01",
        "surface_id": "surface",
        "transition_from": "created",
        "transition_to": "unavailable",
        "public_result_id": "test.unavailable.v1",
        "public_result": {"kind": "unavailable"},
        "status_key": "exit",
        "status_value": "69",
        "provider_call_count": 0,
        "domain_side_effect_count": 0,
    }
    testcase = Element(
        "testcase",
        file="tests/sample_test.py",
        classname="tests.sample_test",
        name="test_value[surface]",
    )
    properties = SubElement(testcase, "properties")
    property_element = SubElement(
        properties,
        "property",
        name="failure_matrix_evidence",
        value=dumps([observation]),
    )
    testcases: tuple[Element, ...] = (testcase,)
    schemas = {
        "test.unavailable.v1": {
            "type": "object",
            "required": ["kind"],
            "properties": {"kind": {"const": "unavailable"}},
        }
    }
    _VERIFIER._verify_failure_matrix_evidence(testcases, matrix, schemas)
    if case == "missing":
        properties.remove(property_element)
    elif case == "malformed":
        property_element.set("value", "{")
    elif case == "multi":
        property_element.set("value", dumps([observation, observation]))
    elif case == "duplicate":
        testcases = (testcase, deepcopy(testcase))
    elif case == "wrong_owner":
        testcase.set("name", "test_other[surface]")
    elif case == "wrong_surface":
        testcase.set("name", "test_value[other]")
    elif case == "wrong_claim":
        observation["status_value"] = "0"
        property_element.set("value", dumps([observation]))
    else:
        observation["public_result"] = {}
        property_element.set("value", dumps([observation]))

    with pytest.raises(_VERIFIER.AcceptanceVerificationError, match=match):
        _VERIFIER._verify_failure_matrix_evidence(
            testcases,
            matrix,
            schemas,
        )


def test_current_phase_requires_real_postgresql_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject current acceptance outside the provisioned PostgreSQL gate."""
    monkeypatch.delenv("AVALAN_TASK_TEST_POSTGRESQL_DSN", raising=False)
    manifest_path = _FIXTURES / "acceptance_manifest.json"
    current_phase = _VERIFIER.load_manifest(manifest_path).current_phase

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="real PostgreSQL harness",
    ):
        _VERIFIER.verify_acceptance(
            manifest_path,
            repo_root=_ROOT,
            through_phase=current_phase,
        )


def test_postgresql_harness_begins_at_first_real_e2e_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require PostgreSQL without omitting any selected acceptance node."""
    manifest_path = _FIXTURES / "acceptance_manifest.json"
    manifest = _VERIFIER.load_manifest(manifest_path)
    all_postgresql = manifest.postgresql_nodes(manifest.current_phase)
    assert len(all_postgresql) == 11
    assert {node.active_from_phase for node in all_postgresql} == {5}
    observed: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        _VERIFIER,
        "_validate_contract_fixtures",
        lambda manifest, fixtures, root: None,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_verify_nodes",
        lambda nodes, root: observed.append(
            tuple(node.node_id for node in nodes)
        ),
    )
    monkeypatch.delenv("AVALAN_TASK_TEST_POSTGRESQL_DSN", raising=False)

    _VERIFIER.verify_acceptance(
        manifest_path,
        repo_root=_ROOT,
        through_phase=4,
    )
    assert observed == [
        tuple(node.node_id for node in manifest.active_nodes(4))
    ]
    assert len(observed[0]) == 352

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="real PostgreSQL harness",
    ):
        _VERIFIER.verify_acceptance(
            manifest_path,
            repo_root=_ROOT,
            through_phase=5,
        )

    monkeypatch.setenv(
        "AVALAN_TASK_TEST_POSTGRESQL_DSN",
        "postgresql://test",
    )
    _VERIFIER.verify_acceptance(
        manifest_path,
        repo_root=_ROOT,
        through_phase=5,
    )
    assert observed[-1] == tuple(
        node.node_id for node in manifest.active_nodes(5)
    )
    assert len(observed[-1]) == 788
    assert set(node.node_id for node in all_postgresql).issubset(observed[-1])


def test_acceptance_only_phase_lag_rejects_new_type_obligations() -> None:
    """Allow one type-neutral phase and reject wider or typed phase drift."""
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")
    type_manifest = _VERIFIER.load_type_manifest(
        _FIXTURES / "type_contract_manifest.json"
    )
    lagged_type_manifest = replace(
        type_manifest,
        current_phase=manifest.current_phase - 1,
    )

    _VERIFIER._validate_type_contract_phase(manifest, lagged_type_manifest)

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
                lagged_type_manifest,
                fixtures=(*lagged_type_manifest.fixtures, obligation),
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
