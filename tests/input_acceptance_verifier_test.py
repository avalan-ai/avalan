"""Exercise the fail-closed structured-input acceptance verifier."""

from copy import deepcopy
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


def _load_script(name: str) -> ModuleType:
    """Return one repository script as an importable module."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    module_name = f"_input_contract_{name}"
    spec = spec_from_file_location(
        module_name, _ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_VERIFIER = _load_script("verify_input_acceptance")


def _read(name: str) -> dict[str, Any]:
    """Return one mutable fixture copy."""
    value = loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: object) -> None:
    """Write deterministic JSON for a synthetic verifier input."""
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def _snapshot_digest(values: list[str]) -> str:
    """Return an activation snapshot digest."""
    return sha256("\n".join(values).encode()).hexdigest()


def _ledger_digest(payload: dict[str, Any]) -> str:
    """Return the acceptance ledger digest."""
    value = {
        "activation_snapshots": payload["activation_snapshots"],
        "replacements": payload["replacements"],
    }
    return sha256(
        dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()


def _matrix_digest(payload: dict[str, Any]) -> str:
    """Return the failure-matrix contract digest."""
    value = {
        key: payload[key]
        for key in (
            "observation_window",
            "domain_side_effect_scope",
            "surfaces",
            "conditions",
            "cells",
        )
    }
    return sha256(
        dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _canonical_digest(value: object) -> str:
    """Return a canonical JSON digest."""
    return sha256(
        dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _synthetic_contract_inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one complete synthetic manifest and requirements catalog."""
    categories = [
        "unit",
        "integration",
        "negative",
        "race",
        "security",
        "public_e2e",
    ]
    nodes: list[dict[str, Any]] = []
    requirements: list[dict[str, Any]] = []
    for index, category in enumerate(categories, start=1):
        requirement_id = f"SYNTHETIC-{index:03d}"
        node_id = f"tests/synthetic_contract_test.py::test_{category}_contract"
        nodes.append(
            {
                "id": f"synthetic-{category}",
                "category": category,
                "lifecycle": "active",
                "active_from_phase": 0,
                "requirement_ids": [requirement_id],
                "node_id": node_id,
            }
        )
        requirements.append(
            {
                "id": requirement_id,
                "source_section": str(index + 6),
                "normative_level": "MUST",
                "paraphrase": f"Exercise the synthetic {category} contract.",
                "owner": "synthetic-contract-owner",
                "implementation_artifacts": [
                    "tests/synthetic_contract_test.py"
                ],
                "test_node_ids": [node_id],
            }
        )
    active_node_ids = [node["node_id"] for node in nodes]
    manifest = {
        "schema_version": 1,
        "feature": "structured_task_input",
        "current_phase": 0,
        "categories": categories,
        "activation_history": [
            {"phase": 0, "node_ids": [node["id"] for node in nodes]}
        ],
        "activation_snapshots": [
            {
                "phase": 0,
                "node_ids": active_node_ids,
                "sha256": _snapshot_digest(active_node_ids),
            }
        ],
        "replacements": [],
        "nodes": nodes,
    }
    catalog = {
        "schema_version": 1,
        "feature": "structured_task_input",
        "source_sections": [str(value) for value in range(7, 27)],
        "catalog_sha256": _canonical_digest(requirements),
        "requirements": requirements,
    }
    return manifest, catalog


def _replacement_manifest() -> dict[str, Any]:
    """Return a valid synthetic manifest with one reviewed tombstone."""
    categories = [
        "unit",
        "integration",
        "negative",
        "race",
        "security",
        "public_e2e",
    ]
    old_node_id = "tests/synthetic_contract_test.py::test_historical_contract"
    nodes = [
        {
            "id": "replacement-unit",
            "category": "unit",
            "lifecycle": "active",
            "active_from_phase": 1,
            "requirement_ids": ["INPUT-GATE-001"],
            "node_id": (
                "tests/synthetic_contract_test.py::test_replacement_contract"
            ),
        }
    ]
    nodes.extend(
        {
            "id": f"stable-{category}",
            "category": category,
            "lifecycle": "active",
            "active_from_phase": 0,
            "requirement_ids": [f"INPUT-GATE-{index:03d}"],
            "node_id": (
                f"tests/synthetic_contract_test.py::test_{category}_contract"
            ),
        }
        for index, category in enumerate(categories[1:], start=2)
    )
    stable_node_ids: list[str] = [str(node["node_id"]) for node in nodes[1:]]
    phase_zero: list[str] = [old_node_id, *stable_node_ids]
    phase_one: list[str] = [str(node["node_id"]) for node in nodes]
    return {
        "schema_version": 1,
        "feature": "structured_task_input",
        "current_phase": 1,
        "categories": categories,
        "activation_history": [
            {"phase": 0, "node_ids": [node["id"] for node in nodes[1:]]},
            {"phase": 1, "node_ids": [nodes[0]["id"]]},
        ],
        "activation_snapshots": [
            {
                "phase": 0,
                "node_ids": phase_zero,
                "sha256": _snapshot_digest(phase_zero),
            },
            {
                "phase": 1,
                "node_ids": phase_one,
                "sha256": _snapshot_digest(phase_one),
            },
        ],
        "replacements": [
            {
                "phase": 1,
                "old_node_id": old_node_id,
                "replacement_node_ids": [nodes[0]["node_id"]],
                "requirement_ids": ["INPUT-GATE-001"],
                "reviewed_by": "synthetic-ledger-reviewer",
                "evidence": "synthetic reviewed replacement",
            }
        ],
        "nodes": nodes,
    }


def _protocol_mutations(decisions: dict[str, Any]) -> list[dict[str, Any]]:
    """Return invalid copies for every pinned protocol correction."""
    mutations: list[dict[str, Any]] = []

    def mutate(path: tuple[str, ...], value: object) -> None:
        payload = deepcopy(decisions)
        target = payload
        for name in path[:-1]:
            target = target[name]
        target[path[-1]] = value
        mutations.append(payload)

    task_states = decisions["protocol_projection"]["a2a"]["task_states"]
    mutate(("protocol_projection", "a2a", "task_states"), task_states[:-1])
    mutate(
        ("error_status", "a2a", "core", "push_notification_not_supported"), -1
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "elicitation",
            "requestedSchema",
            "allowed_top_level_keys",
        ),
        ["type", "properties", "required"],
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "elicitation",
            "requestedSchema",
            "properties",
            "primitive_types",
        ),
        ["string", "boolean"],
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "elicitation",
            "requestedSchema",
            "properties",
            "single_select",
            "type",
        ),
        "array",
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "elicitation",
            "requestedSchema",
            "properties",
            "multiple_select",
            "type",
        ),
        "string",
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "elicitation",
            "requestedSchema",
            "properties",
            "multiple_select",
            "items",
            "type",
        ),
        "number",
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "elicitation",
            "requestedSchema",
            "properties",
            "multiple_select",
            "uniqueItems",
        ),
        False,
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "params_task_schema",
            "properties",
            "ttl",
            "unit",
        ),
        "seconds",
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "params_task_schema",
            "properties",
            "ttl",
            "type",
        ),
        ["number", "null"],
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "request_type_task_capability_absent",
        ),
        "reject params.task with -32601",
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "generic_receiver_task_requirement",
            "omission_error",
        ),
        -32601,
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "tool_execution_task_support",
            "absent",
        ),
        "process normally and ignore params.task augmentation",
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "tool_execution_task_support",
            "required",
        ),
        "client MUST invoke with params.task; omission MUST return -32602",
    )
    mutate(
        ("protocol_projection", "mcp", "tasks", "ttl_mapping"),
        "round seconds",
    )
    mutate(
        ("protocol_projection", "mcp", "tasks", "initial_state"),
        "input_required",
    )
    mutate(
        ("protocol_projection", "mcp", "tasks", "legal_transitions"),
        [
            ["working", "input_required"],
            ["input_required", "working"],
            ["working", "completed"],
            ["working", "failed"],
            ["working", "cancelled"],
            ["input_required", "cancelled"],
        ],
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "CreateTaskResult",
            "required",
        ),
        ["task", "_meta"],
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "CreateTaskResult",
            "additionalProperties",
        ),
        False,
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "CreateTaskResult",
            "properties",
            "_meta",
            "minProperties",
        ),
        1,
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "task_schema",
            "properties",
            "ttl",
            "minimum",
        ),
        1,
    )
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "task_schema",
            "properties",
            "pollInterval",
            "minimum",
        ),
        0,
    )
    task_properties = deepcopy(
        decisions["protocol_projection"]["mcp"]["tasks"]["task_schema"][
            "properties"
        ]
    )
    task_properties["_meta"] = {"type": "object"}
    mutate(
        (
            "protocol_projection",
            "mcp",
            "tasks",
            "task_schema",
            "properties",
        ),
        task_properties,
    )
    mutate(
        ("error_status", "mcp", "receiver_task_augmentation_required"),
        -32601,
    )
    mutate(
        ("error_status", "mcp", "tool_task_augmentation_forbidden"),
        -32602,
    )
    mutate(
        ("error_status", "mcp", "tool_task_augmentation_required"),
        -32602,
    )
    mutate(
        (
            "error_status",
            "public_envelope_catalog",
            "mcp.extension_required_error.v1",
            "properties",
            "error",
            "properties",
            "code",
            "const",
        ),
        -32602,
    )
    mutate(
        (
            "protocol_projection",
            "a2a",
            "extension",
            "message_metadata_schema",
            "$defs",
            "question",
            "additionalProperties",
        ),
        True,
    )
    mutate(
        (
            "protocol_projection",
            "a2a",
            "extension",
            "message_metadata_examples",
        ),
        {},
    )
    mutate(("error_status", "public_envelope_examples"), {})
    mutate(
        (
            "error_status",
            "public_envelope_catalog_contract",
            "mutation_requirements",
        ),
        [
            "missing_required_field",
            "extra_field",
            "wrong_const",
            "wrong_type",
        ],
    )
    mutate(
        ("error_status", "public_envelope_cross_field_mutations"),
        {},
    )
    return mutations


def test_acceptance_rejects_invalid_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject inventory drift and unreviewed ledger removal."""
    manifest = _read("acceptance_manifest.json")
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_ACCEPTANCE_LEDGER_SHA256",
        _ledger_digest(manifest),
    )
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    assert _VERIFIER.load_manifest(path).current_phase == 0

    invalid = deepcopy(manifest)
    invalid["categories"].append("unit")
    _write(path, invalid)
    with pytest.raises(_VERIFIER.AcceptanceVerificationError):
        _VERIFIER.load_manifest(path)

    duplicate_node = deepcopy(manifest)
    duplicate_node["nodes"][1]["node_id"] = duplicate_node["nodes"][0][
        "node_id"
    ]
    _write(path, duplicate_node)
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="duplicate pytest node ID",
    ):
        _VERIFIER.load_manifest(path)

    _write(path, manifest)
    loaded_manifest = _VERIFIER.load_manifest(path)
    requirements_path = tmp_path / "requirements.json"
    requirements = _read("requirements_traceability.json")
    missing_requirement = deepcopy(requirements)
    missing_requirement["requirements"] = missing_requirement["requirements"][
        1:
    ]
    missing_digest = _canonical_digest(missing_requirement["requirements"])
    missing_requirement["catalog_sha256"] = missing_digest
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_REQUIREMENTS_SHA256",
        missing_digest,
    )
    _write(requirements_path, missing_requirement)
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="requirements inventory mismatch",
    ):
        _VERIFIER._validate_requirements(requirements_path, loaded_manifest)

    unmapped_requirement = deepcopy(requirements)
    unmapped_requirement["requirements"][0]["test_node_ids"] = [
        "tests/synthetic_contract_test.py::test_missing_contract"
    ]
    unmapped_digest = _canonical_digest(unmapped_requirement["requirements"])
    unmapped_requirement["catalog_sha256"] = unmapped_digest
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_REQUIREMENTS_SHA256",
        unmapped_digest,
    )
    _write(requirements_path, unmapped_requirement)
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="unmapped requirement node",
    ):
        _VERIFIER._validate_requirements(requirements_path, loaded_manifest)

    replacement = _replacement_manifest()
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_ACCEPTANCE_LEDGER_SHA256",
        _ledger_digest(replacement),
    )
    _write(path, replacement)
    assert _VERIFIER.load_manifest(path).current_phase == 1
    replacement["replacements"] = []
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_ACCEPTANCE_LEDGER_SHA256",
        _ledger_digest(replacement),
    )
    _write(path, replacement)
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError, match="tombstone"
    ):
        _VERIFIER.load_manifest(path)

    decisions = _read("contract_decisions.json")
    _VERIFIER._validate_protocol_decision_shapes(decisions)
    for mutation in _protocol_mutations(decisions):
        with pytest.raises(_VERIFIER.AcceptanceVerificationError):
            _VERIFIER._validate_protocol_decision_shapes(mutation)

    synthetic_manifest, synthetic_requirements = (
        _synthetic_contract_inventory()
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_REQUIREMENT_IDS",
        frozenset(
            requirement["id"]
            for requirement in synthetic_requirements["requirements"]
        ),
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_REQUIREMENTS_SHA256",
        synthetic_requirements["catalog_sha256"],
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_ACCEPTANCE_LEDGER_SHA256",
        _ledger_digest(synthetic_manifest),
    )
    _write(path, synthetic_manifest)
    _write(requirements_path, synthetic_requirements)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "unrelated.md").write_text("sentinel\n", encoding="utf-8")
    original_read_text = Path.read_text
    observed_reads: list[Path] = []

    def read_non_markdown(
        target: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        observed_reads.append(target)
        assert target.suffix.lower() != ".md"
        return original_read_text(target, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_non_markdown)
    loaded_synthetic = _VERIFIER.load_manifest(path)
    assert _VERIFIER._validate_requirements(
        requirements_path, loaded_synthetic
    ) == frozenset(
        requirement["id"]
        for requirement in synthetic_requirements["requirements"]
    )
    assert observed_reads


def test_acceptance_rejects_na_reason_without_exact_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require every reviewed N/A reason to name its exact matrix cell."""
    manifest_payload = _read("acceptance_manifest.json")
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_ACCEPTANCE_LEDGER_SHA256",
        _ledger_digest(manifest_payload),
    )
    manifest_path = tmp_path / "acceptance_manifest.json"
    _write(manifest_path, manifest_payload)
    manifest = _VERIFIER.load_manifest(manifest_path)
    requirements_payload = _read("requirements_traceability.json")
    requirements = frozenset(
        requirement["id"]
        for requirement in requirements_payload["requirements"]
    )
    decisions = _read("contract_decisions.json")
    decision_surfaces = frozenset(
        row["public_failure_surface"]
        for row in decisions["capability_matrix"]["rows"]
        if row["public_failure_surface"] is not None
    )
    public_envelopes = frozenset(
        decisions["error_status"]["public_envelope_catalog"]
    )
    matrix = _read("failure_matrix.json")
    cell = next(item for item in matrix["cells"] if not item["applicable"])
    condition_id = cell["condition_id"]
    surface_id = cell["surface_id"]
    matrix_path = tmp_path / "failure_matrix.json"

    def validate(payload: dict[str, Any]) -> None:
        digest = _matrix_digest(payload)
        payload["matrix_sha256"] = digest
        monkeypatch.setattr(
            _VERIFIER,
            "_EXPECTED_FAILURE_MATRIX_SHA256",
            digest,
        )
        _write(matrix_path, payload)
        _VERIFIER._validate_failure_matrix(
            matrix_path,
            manifest,
            requirements,
            decision_surfaces,
            public_envelopes,
        )

    validate(deepcopy(matrix))
    missing_condition = deepcopy(matrix)
    missing_condition["cells"][matrix["cells"].index(cell)][
        "non_applicability_reason"
    ] = (
        f"{surface_id} is a concrete read-only surface that cannot create, "
        "advertise, resolve, or mutate this reviewed interaction lifecycle."
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="concrete reviewed reason",
    ):
        validate(missing_condition)
    missing_surface = deepcopy(matrix)
    missing_surface["cells"][matrix["cells"].index(cell)][
        "non_applicability_reason"
    ] = (
        f"{condition_id} cannot occur on this concrete read-only control "
        "surface because no request is created, advertised, or mutated."
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="concrete reviewed reason",
    ):
        validate(missing_surface)

    unavailable_fallback = deepcopy(matrix)
    fallback_cell = next(
        item
        for item in unavailable_fallback["cells"]
        if item["condition_id"] == "INPUT-F-15"
        and item["surface_id"] == "mcp-inbound-task"
    )
    fallback_cell.update(
        {
            "expected_transition": "created->unavailable",
            "public_result": "envelope=mcp.unavailable_error.v1",
            "status_or_exit": "jsonrpc_error=-32001",
            "provider_call_count": 0,
        }
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="MCP capability-absent fallback",
    ):
        validate(unavailable_fallback)

    unowned_cell = deepcopy(matrix)
    unrelated_node = next(
        node
        for node in manifest_payload["nodes"]
        if node["node_id"]
        == "tests/input/public_interaction_e2e_test.py::"
        "test_attached_cli_clarification"
    )
    applicable = next(
        item
        for item in unowned_cell["cells"]
        if item["applicable"]
        and item["active_from_phase"] == unrelated_node["active_from_phase"]
        and item["condition_id"] != "INPUT-F-15"
    )
    applicable["negative_e2e_node"] = unrelated_node["node_id"]
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="does not own its requirement",
    ):
        validate(unowned_cell)


def test_acceptance_cli_executes_exact_synthetic_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collect and execute only the requested synthetic pytest node."""
    tests = tmp_path / "tests"
    tests.mkdir()
    source = tests / "synthetic_contract_test.py"
    source.write_text(
        "def test_selected():\n"
        "    value = 6 * 7\n"
        "    assert value == 42\n\n"
        "def test_unselected():\n"
        "    assert False\n",
        encoding="utf-8",
    )
    nodes = ("tests/synthetic_contract_test.py::test_selected",)
    collection = _VERIFIER._run_probe(
        _VERIFIER._COLLECT_DRIVER,
        _VERIFIER._COLLECT_SENTINEL,
        nodes,
        tmp_path,
    )
    _VERIFIER._verify_collection(nodes, collection)
    execution = _VERIFIER._run_probe(
        _VERIFIER._EXECUTE_DRIVER,
        _VERIFIER._EXECUTE_SENTINEL,
        nodes,
        tmp_path,
    )
    _VERIFIER._verify_execution(nodes, execution)
    assert execution["exit_code"] == 0

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "unrelated.md").write_text("sentinel\n", encoding="utf-8")
    original_read_text = Path.read_text
    observed_reads: list[Path] = []

    def read_non_markdown(
        target: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        observed_reads.append(target)
        assert target.suffix.lower() != ".md"
        return original_read_text(target, encoding=encoding, errors=errors)

    def successful_probe(
        driver: str,
        sentinel: str,
        node_ids: tuple[str, ...],
        root: Path,
    ) -> dict[str, object]:
        del driver, root
        common: dict[str, object] = {
            "exit_code": 0,
            "deselected": [],
            "collection_reports": [],
            "probe_stdout": "",
            "probe_stderr": "",
        }
        if sentinel == _VERIFIER._COLLECT_SENTINEL:
            return {
                **common,
                "items": [
                    {"nodeid": node_id, "markers": []} for node_id in node_ids
                ],
            }
        assert sentinel == _VERIFIER._EXECUTE_SENTINEL
        return {
            **common,
            "items": list(node_ids),
            "reports": [
                {
                    "nodeid": node_id,
                    "when": when,
                    "outcome": "passed",
                    "wasxfail": "",
                    "detail": "",
                }
                for node_id in node_ids
                for when in ("setup", "call", "teardown")
            ],
        }

    monkeypatch.setattr(Path, "read_text", read_non_markdown)
    monkeypatch.setattr(_VERIFIER, "_run_probe", successful_probe)
    live_manifest = _VERIFIER.verify_acceptance(
        _FIXTURES / "acceptance_manifest.json",
        repo_root=_ROOT,
        through_phase=0,
        contract_fixture_root=_FIXTURES,
    )
    assert len(live_manifest.active_nodes(0)) == 24
    assert observed_reads


def test_acceptance_rejects_pytest_non_evidence() -> None:
    """Reject skipped, deselected, partial, and duplicate pytest evidence."""
    manifest = _read("acceptance_manifest.json")
    node = next(
        item["node_id"]
        for item in manifest["nodes"]
        if item["lifecycle"] == "active"
    )
    planned_node = next(
        item["node_id"]
        for item in manifest["nodes"]
        if item["lifecycle"] == "planned"
    )
    collection: dict[str, Any] = {
        "exit_code": 0,
        "items": [{"nodeid": node, "markers": ["skip"]}],
        "deselected": [],
        "collection_reports": [],
        "probe_stdout": "",
        "probe_stderr": "",
    }
    with pytest.raises(_VERIFIER.AcceptanceVerificationError, match="markers"):
        _VERIFIER._verify_collection((node,), collection)
    collection["items"][0]["markers"] = ["skipif"]
    with pytest.raises(_VERIFIER.AcceptanceVerificationError, match="markers"):
        _VERIFIER._verify_collection((node,), collection)
    collection["items"][0]["markers"] = []
    collection["deselected"] = [node]
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError, match="deselected"
    ):
        _VERIFIER._verify_collection((node,), collection)
    collection["deselected"] = []
    collection["collection_reports"] = [
        {"outcome": "failed", "detail": "synthetic collection error"}
    ]
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="collection was skipped or failed",
    ):
        _VERIFIER._verify_collection((node,), collection)
    collection["collection_reports"] = []
    collection["items"] = []
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="missing=",
    ):
        _VERIFIER._verify_collection((node,), collection)
    collection["items"] = [
        {"nodeid": node, "markers": []},
        {"nodeid": planned_node, "markers": []},
    ]
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="unexpected=",
    ):
        _VERIFIER._verify_collection((node,), collection)
    execution: dict[str, Any] = {
        "exit_code": 0,
        "items": [node],
        "deselected": [],
        "collection_reports": [],
        "reports": [
            {
                "nodeid": node,
                "when": "call",
                "outcome": "passed",
                "wasxfail": "",
                "detail": "",
            }
        ],
        "probe_stdout": "",
        "probe_stderr": "",
    }
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError, match="exactly once"
    ):
        _VERIFIER._verify_execution((node,), execution)

    execution["reports"] = [
        {
            "nodeid": node,
            "when": when,
            "outcome": "failed" if when == "call" else "passed",
            "wasxfail": "",
            "detail": "synthetic failure" if when == "call" else "",
        }
        for when in ("setup", "call", "teardown")
    ]
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="call outcome was failed",
    ):
        _VERIFIER._verify_execution((node,), execution)

    execution["reports"] = [
        {
            "nodeid": node,
            "when": when,
            "outcome": "passed",
            "wasxfail": "synthetic expectation" if when == "call" else "",
            "detail": "",
        }
        for when in ("setup", "call", "teardown")
    ]
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="xfail/xpass",
    ):
        _VERIFIER._verify_execution((node,), execution)


def test_acceptance_rejects_placeholder_and_execution_tricks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject assertions hidden behind unreachable or prohibited code."""
    tests = tmp_path / "tests"
    tests.mkdir()
    path = tests / "case_test.py"
    node = "tests/case_test.py::test_case"
    invalid_sources = (
        (
            "def test_case():\n    if True:\n        return\n    assert"
            " len('x') == 1\n"
        ),
        "def test_case():\n    for value in []:\n        assert value\n",
        (
            "def test_case():\n    try:\n        pass\n    except Exception:\n"
            "        assert len('x') == 1\n"
        ),
        "def test_case():\n    value = object()\n    assert value == value\n",
        (
            "def test_case():\n    value = object()\n    assert value is not"
            " None\n    compile('1', 'x', 'eval')\n"
        ),
        (
            "def test_case():\n    value = object()\n    assert value is not"
            " None\n    exec('value = 1')\n"
        ),
        (
            "def enabled():\n    return False\n\ndef test_case():\n"
            "    if enabled():\n        assert len('x') == 1\n"
        ),
        (
            "import pytest\n\ndef test_case():\n    if False:\n"
            "        pytest.skip('dormant')\n    assert len('x') == 1\n"
        ),
        (
            "import pytest\n\ndef test_case():\n    if False:\n"
            "        pytest.importorskip('missing')\n"
            "    assert len('x') == 1\n"
        ),
        (
            "import pytest\n\ndef test_case():\n    if False:\n"
            "        pytest.xfail('dormant')\n    assert len('x') == 1\n"
        ),
        (
            "from pytest import skip as finish\n\ndef test_case():\n"
            "    assert len('x') == 1\n    finish('aliased')\n"
        ),
        (
            "from pytest import importorskip as optional\n\ndef test_case():\n"
            "    assert len('x') == 1\n    optional('missing')\n"
        ),
        (
            "from pytest import xfail as expected\n\ndef test_case():\n"
            "    assert len('x') == 1\n    expected('aliased')\n"
        ),
        (
            "import unittest\n\n@unittest.skip('disabled')\n"
            "def test_case():\n    assert len('x') == 1\n"
        ),
        (
            "import unittest\n\n@unittest.skipIf(False, 'conditional')\n"
            "def test_case():\n    assert len('x') == 1\n"
        ),
        (
            "import unittest\n\n@unittest.skipUnless(True, 'conditional')\n"
            "def test_case():\n    assert len('x') == 1\n"
        ),
        (
            "from unittest import skipIf as optional\n\n"
            "@optional(False, 'conditional')\ndef test_case():\n"
            "    assert len('x') == 1\n"
        ),
        (
            "import builtins\n\ndef test_case():\n    assert len('x') == 1\n"
            "    builtins.exec('value = 1')\n"
        ),
        (
            "import builtins as runtime\n\ndef test_case():\n"
            "    assert len('x') == 1\n"
            "    runtime.compile('1', 'x', 'eval')\n"
        ),
        (
            "from builtins import exec as evaluate\n\ndef test_case():\n"
            "    assert len('x') == 1\n    evaluate('value = 1')\n"
        ),
        (
            "from builtins import compile as build\n\ndef test_case():\n"
            "    assert len('x') == 1\n    build('1', 'x', 'eval')\n"
        ),
    )
    for source in invalid_sources:
        path.write_text(source, encoding="utf-8")
        with pytest.raises(_VERIFIER.AcceptanceVerificationError):
            _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def test_case():\n    value = len('contract')\n    assert value"
        " == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    assert path.is_file()

    tracked_paths = {"tests/case_test.py"}
    untracked_paths = {
        "docs/examples/skills/code/example.py",
        "tests/fixtures/input/example.json",
    }

    def fake_git_lines(_root: Path, *args: str) -> tuple[str, ...]:
        match args[0]:
            case "diff":
                return tuple(sorted(tracked_paths))
            case "ls-files":
                return tuple(sorted(untracked_paths))
            case _:
                raise AssertionError(args)

    monkeypatch.setattr(_VERIFIER, "_git_lines", fake_git_lines)
    declared_paths = ("tests/case_test.py", "tests/fixtures/input/")
    preserved_untracked = ("docs/examples/skills/code/",)
    _VERIFIER._validate_live_boundary(
        tmp_path,
        declared_paths,
        preserved_untracked,
    )
    tracked_paths.add("undeclared.txt")
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="live changed paths differ",
    ):
        _VERIFIER._validate_live_boundary(
            tmp_path,
            declared_paths,
            preserved_untracked,
        )
    tracked_paths.remove("undeclared.txt")
    untracked_paths.add("undeclared.txt")
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="live changed paths differ",
    ):
        _VERIFIER._validate_live_boundary(
            tmp_path,
            declared_paths,
            preserved_untracked,
        )
