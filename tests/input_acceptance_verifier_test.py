"""Exercise the fail-closed structured-input acceptance verifier."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from pathlib import Path
from subprocess import run
from sys import modules
from sys import path as sys_path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

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


def _git(root: Path, *arguments: str) -> None:
    """Run one synthetic repository command successfully."""
    completed = run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _snapshot_digest(values: list[str]) -> str:
    """Return an activation snapshot digest."""
    return sha256("\n".join(values).encode()).hexdigest()


def _ledger_digest(payload: dict[str, Any]) -> str:
    """Return the acceptance ledger digest."""
    value = {
        "activation_snapshots": payload["activation_snapshots"],
        "replacements": payload["replacements"],
        "requirement_activation_slices": payload[
            "requirement_activation_slices"
        ],
        "parameter_expansions": payload["parameter_expansions"],
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
            "activation_schedule_corrections",
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
        "requirement_activation_slices": [],
        "parameter_expansions": [],
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
        "requirement_activation_slices": [],
        "parameter_expansions": [],
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


def _manifest_with_planned_expansion() -> (
    tuple[dict[str, Any], str, list[str], str, list[str]]
):
    """Return active and planned parameter expansions in manifest order."""
    manifest, _ = _synthetic_contract_inventory()
    active_node_id = manifest["nodes"][0]["node_id"]
    assert isinstance(active_node_id, str)
    active_instances = [
        f"{active_node_id}[first]",
        f"{active_node_id}[second]",
    ]
    planned_node_id = (
        "tests/synthetic_contract_test.py::test_future_parameter_contract"
    )
    planned_instances = [
        f"{planned_node_id}[capable]",
        f"{planned_node_id}[incapable]",
    ]
    manifest["nodes"].append(
        {
            "id": "synthetic-future-parameter-contract",
            "category": "unit",
            "lifecycle": "planned",
            "active_from_phase": 1,
            "requirement_ids": ["SYNTHETIC-007"],
            "node_id": planned_node_id,
        }
    )
    manifest["parameter_expansions"] = [
        {
            "node_id": active_node_id,
            "instance_node_ids": active_instances,
            "sha256": _snapshot_digest(active_instances),
        },
        {
            "node_id": planned_node_id,
            "instance_node_ids": planned_instances,
            "sha256": _snapshot_digest(planned_instances),
        },
    ]
    return (
        manifest,
        active_node_id,
        active_instances,
        planned_node_id,
        planned_instances,
    )


def test_planned_parameter_expansion_is_validated_without_early_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate planned instances while selecting only active instances."""
    (
        manifest,
        active_node_id,
        active_instances,
        planned_node_id,
        planned_instances,
    ) = _manifest_with_planned_expansion()
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_ACCEPTANCE_LEDGER_SHA256",
        _ledger_digest(manifest),
    )
    path = tmp_path / "manifest.json"
    _write(path, manifest)

    loaded = _VERIFIER.load_manifest(path)

    assert {
        expansion.node_id: expansion.instance_node_ids
        for expansion in loaded.parameter_expansions
    } == {
        active_node_id: tuple(active_instances),
        planned_node_id: tuple(planned_instances),
    }
    assert loaded.active_pytest_instances(0) == (
        *active_instances,
        *(
            node["node_id"]
            for node in manifest["nodes"][1:6]
            if isinstance(node["node_id"], str)
        ),
    )
    assert not set(planned_instances) & set(loaded.active_pytest_instances(0))


def test_planned_parameter_expansion_digest_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a stale digest for an exact planned parameter expansion."""
    manifest, _, _, _, _ = _manifest_with_planned_expansion()
    manifest["parameter_expansions"][1]["sha256"] = "0" * 64
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_ACCEPTANCE_LEDGER_SHA256",
        _ledger_digest(manifest),
    )
    path = tmp_path / "manifest.json"
    _write(path, manifest)

    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="parameter expansion digest mismatch",
    ):
        _VERIFIER.load_manifest(path)


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
    assert _VERIFIER.load_manifest(path).current_phase == 6

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

    unsliced = deepcopy(manifest)
    unsliced["requirement_activation_slices"].pop()
    _write(path, unsliced)
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="mixed-lifecycle requirements lack exact activation slices",
    ):
        _VERIFIER.load_manifest(path)

    unreviewed_slice = deepcopy(manifest)
    unreviewed_slice["requirement_activation_slices"][0][
        "reviewed_by"
    ] = "pending"
    _write(path, unreviewed_slice)
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="lacks implementation review",
    ):
        _VERIFIER.load_manifest(path)

    incomplete_slice = deepcopy(manifest)
    incomplete_slice["requirement_activation_slices"][0][
        "active_node_ids"
    ].pop()
    _write(path, incomplete_slice)
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="active inventory changed",
    ):
        _VERIFIER.load_manifest(path)

    stale_expansion_hash = deepcopy(manifest)
    stale_expansion_hash["parameter_expansions"][0]["instance_node_ids"].pop()
    _write(path, stale_expansion_hash)
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="parameter expansion digest mismatch",
    ):
        _VERIFIER.load_manifest(path)

    duplicate_expansion = deepcopy(manifest)
    duplicate_instances = duplicate_expansion["parameter_expansions"][0][
        "instance_node_ids"
    ]
    duplicate_instances.append(duplicate_instances[0])
    duplicate_expansion["parameter_expansions"][0]["sha256"] = (
        _snapshot_digest(duplicate_instances)
    )
    _write(path, duplicate_expansion)
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="duplicate parameter instance",
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


def test_current_runtime_manifest_inventory_fails_closed() -> None:
    """Reject missing, extra, moved, inactive, or gate-owned current nodes."""
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")
    runtime_nodes = tuple(
        node
        for node in manifest.nodes
        if node.active_from_phase == _VERIFIER._CURRENT_BOUNDARY_PHASE
        and node.lifecycle == "active"
        and node.node_id.split("::", 1)[0]
        in _VERIFIER._EXPECTED_CURRENT_RUNTIME_FILES
    )
    assert runtime_nodes

    missing = tuple(
        node for node in manifest.nodes if node is not runtime_nodes[0]
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="reviewed runtime acceptance inventory changed",
    ):
        _VERIFIER._validate_current_manifest_inventory(missing)

    extra = replace(
        runtime_nodes[0],
        id="current-unreviewed-runtime",
        node_id=(
            runtime_nodes[0].node_id.split("::", 1)[0]
            + "::test_unreviewed_runtime"
        ),
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="reviewed runtime acceptance inventory changed",
    ):
        _VERIFIER._validate_current_manifest_inventory(
            (
                *manifest.nodes,
                extra,
            )
        )

    moved = replace(
        runtime_nodes[0],
        node_id="tests/agent/unreviewed_semantic_test.py::test_runtime",
    )
    moved_nodes = tuple(
        moved if node is runtime_nodes[0] else node for node in manifest.nodes
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="reviewed runtime acceptance inventory changed",
    ):
        _VERIFIER._validate_current_manifest_inventory(moved_nodes)

    for changed in (
        replace(runtime_nodes[0], lifecycle="planned"),
        replace(runtime_nodes[0], requirement_ids=("INPUT-GATE-001",)),
    ):
        changed_nodes = tuple(
            changed if node is runtime_nodes[0] else node
            for node in manifest.nodes
        )
        with pytest.raises(
            _VERIFIER.AcceptanceVerificationError,
            match="reviewed runtime acceptance inventory changed",
        ):
            _VERIFIER._validate_current_manifest_inventory(changed_nodes)


def test_current_runtime_file_inventory_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject file or expanded-instance drift before current execution."""
    for relative in _VERIFIER._EXPECTED_CURRENT_RUNTIME_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")
    definitions = _VERIFIER._current_runtime_node_ids(manifest.nodes)
    instances = _VERIFIER._current_runtime_instance_ids(manifest)
    assert len(definitions) == 26
    assert len(instances) == 26
    requested: list[tuple[str, ...]] = []

    def collection_payload(node_ids: tuple[str, ...]) -> dict[str, object]:
        return {
            "exit_code": 0,
            "items": [
                {"nodeid": node_id, "markers": []} for node_id in node_ids
            ],
            "deselected": [],
            "collection_reports": [],
            "probe_stdout": "",
            "probe_stderr": "",
        }

    def collect_instances(
        driver: str,
        sentinel: str,
        node_ids: tuple[str, ...],
        root: Path,
    ) -> dict[str, object]:
        del driver, sentinel, root
        requested.append(node_ids)
        return collection_payload(instances)

    monkeypatch.setattr(_VERIFIER, "_run_probe", collect_instances)
    _VERIFIER._validate_current_runtime_collection(manifest, tmp_path)
    assert requested == [definitions]

    for drifted in (
        instances[1:],
        (*instances, "tests/unexpected_test.py::test_x"),
    ):
        monkeypatch.setattr(
            _VERIFIER,
            "_run_probe",
            lambda *_args, drifted=drifted: collection_payload(drifted),
        )
        with pytest.raises(
            _VERIFIER.AcceptanceVerificationError,
            match="acceptance nodes were not exactly collected",
        ):
            _VERIFIER._validate_current_runtime_collection(manifest, tmp_path)

    missing = tmp_path / _VERIFIER._EXPECTED_CURRENT_RUNTIME_FILES[0]
    missing.unlink()
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="current runtime test-file inventory changed",
    ):
        _VERIFIER._validate_current_runtime_collection(manifest, tmp_path)


def test_task_failure_schedule_is_frozen() -> None:
    """Reject lifecycle or activation-boundary drift for task failures."""
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")
    scheduled = tuple(
        node
        for node in manifest.nodes
        if node.node_id in _VERIFIER._EXPECTED_TASK_FAILURE_NODES
    )
    assert frozenset(node.node_id for node in scheduled) == (
        _VERIFIER._EXPECTED_TASK_FAILURE_NODES
    )
    assert all(
        node.lifecycle == "active"
        and node.active_from_phase == _VERIFIER._CURRENT_BOUNDARY_PHASE
        and node.category == "public_e2e"
        for node in scheduled
    )

    moved = replace(
        scheduled[0],
        lifecycle="planned",
        active_from_phase=_VERIFIER._CURRENT_BOUNDARY_PHASE + 1,
    )
    moved_nodes = tuple(
        moved if node is scheduled[0] else node for node in manifest.nodes
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="scheduled task failure-node inventory changed",
    ):
        _VERIFIER._validate_task_failure_schedule(moved_nodes)


def test_current_regression_classification_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject structural drift and any new test hidden as mechanical."""
    manifest = _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")
    evidence = _read("baseline_evidence.json")
    classification = evidence["current_regression_classification"]
    _VERIFIER._validate_current_regression_classification(
        classification,
        manifest,
        _ROOT,
    )
    semantic_partition = _VERIFIER._current_semantic_definition_partition(
        frozenset(("equal", "modified", "new")),
        {"equal": "same", "modified": "before"},
        {"equal": "same", "modified": "after", "new": "added"},
    )
    assert semantic_partition == (
        frozenset(("new",)),
        frozenset(("modified",)),
        frozenset(("equal",)),
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="current semantic definitions do not partition",
    ):
        _VERIFIER._current_semantic_definition_partition(
            frozenset(("missing",)),
            {},
            {},
        )

    baseline, current, paths = _VERIFIER._current_changed_test_definitions(
        _ROOT
    )
    support_baseline, support_current = (
        _VERIFIER._current_changed_test_support_surfaces(_ROOT, paths)
    )
    changed_support = {
        relative
        for relative in support_baseline
        if support_baseline[relative] != support_current[relative]
    }
    assert len(support_baseline) == _VERIFIER._EXPECTED_CURRENT_TEST_FILE_COUNT
    assert (
        len(changed_support)
        == _VERIFIER._EXPECTED_CURRENT_SUPPORT_SURFACE_COUNT
    )
    assert (
        len(support_baseline.keys() - changed_support)
        == _VERIFIER._EXPECTED_CURRENT_UNCHANGED_SUPPORT_SURFACE_COUNT
    )
    assert changed_support == set(
        _VERIFIER._EXPECTED_CURRENT_CHANGED_SUPPORT_PATHS
    )

    real_probe = _VERIFIER._run_probe

    def inject_unreviewed_inherited(
        driver: str,
        sentinel: str,
        node_ids: tuple[str, ...],
        root: Path,
    ) -> dict[str, object]:
        payload = real_probe(driver, sentinel, node_ids, root)
        if sentinel == _VERIFIER._COLLECT_SENTINEL:
            items = payload["items"]
            assert isinstance(items, list)
            items.append(
                {
                    "nodeid": (
                        "tests/task/stores/pgsql_contract_test.py::"
                        "PgsqlStoreContractTest::test_unreviewed_inherited"
                    ),
                    "markers": [],
                }
            )
        return cast(dict[str, object], payload)

    with monkeypatch.context() as exact_inherited:
        exact_inherited.setattr(
            _VERIFIER,
            "_run_probe",
            inject_unreviewed_inherited,
        )
        with pytest.raises(
            _VERIFIER.AcceptanceVerificationError,
            match="does not map to one static definition",
        ):
            _VERIFIER._current_changed_test_definitions(_ROOT)

    support_source = (
        "import pytest\n\nVALUE = 1\n\nclass Base:\n    pass\n\n"
        "class TestCase(Base):\n    def helper(self):\n"
        "        return VALUE\n\n    def test_case(self):\n"
        "        assert self.helper() == 1\n"
    )
    mechanical_source = support_source.replace(
        "        assert self.helper() == 1\n",
        "        # formatting-only\n"
        "        assert (\n            self.helper()\n        ) == 1\n",
    )
    decorated_test_source = support_source.replace(
        "    def test_case(self):\n",
        "    @pytest.mark.skip\n    def test_case(self):\n",
    )
    support_digest = _VERIFIER._test_support_surface_digest(
        support_source,
        "tests/example_test.py",
    )
    assert (
        _VERIFIER._test_support_surface_digest(
            mechanical_source,
            "tests/example_test.py",
        )
        == support_digest
    )
    assert (
        _VERIFIER._test_support_surface_digest(
            decorated_test_source,
            "tests/example_test.py",
        )
        == support_digest
    )
    support_mutations = (
        support_source.replace(
            "class TestCase(Base):",
            "@pytest.mark.skip\nclass TestCase(Base):",
        ),
        support_source.replace("VALUE = 1", "pytestmark = pytest.mark.skip"),
        support_source.replace(
            "class Base:",
            "@pytest.fixture\ndef shared_value():\n    return 1\n\n"
            "class Base:",
        ),
        support_source.replace("return VALUE", "return VALUE + 1"),
        support_source.replace("class TestCase(Base):", "class TestCase:"),
    )
    for changed_source in support_mutations:
        assert (
            _VERIFIER._test_support_surface_digest(
                changed_source,
                "tests/example_test.py",
            )
            != support_digest
        )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="cannot inspect changed test definitions",
    ):
        _VERIFIER._test_support_surface_digest(
            "def test_case(:\n    pass\n",
            "tests/example_test.py",
        )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="duplicate changed test definition",
    ):
        _VERIFIER._test_definition_digests(
            "def test_case():\n    pass\n\ndef test_case():\n    pass\n",
            "tests/example_test.py",
        )
    legacy_duplicate_path = "tests/memory/permanent/pgsql_test.py"
    legacy_duplicate_source = (
        "class PgsqlMessageMemoryTestCase:\n"
        "    def test_search_messages(self):\n        assert value == 1\n"
        "    def test_search_messages(self):\n        assert value == 2\n"
    )
    reversed_legacy_duplicate_source = (
        legacy_duplicate_source.replace(
            "assert value == 1",
            "assert value == temporary",
        )
        .replace("assert value == 2", "assert value == 1")
        .replace(
            "assert value == temporary",
            "assert value == 2",
        )
    )
    legacy_duplicate_id = (
        f"{legacy_duplicate_path}::PgsqlMessageMemoryTestCase::"
        "test_search_messages"
    )
    ordered_duplicate_digest = _VERIFIER._test_definition_digests(
        legacy_duplicate_source,
        legacy_duplicate_path,
    )[legacy_duplicate_id]
    reversed_duplicate_digest = _VERIFIER._test_definition_digests(
        reversed_legacy_duplicate_source,
        legacy_duplicate_path,
    )[legacy_duplicate_id]
    assert ordered_duplicate_digest != reversed_duplicate_digest
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="duplicate changed test definition",
    ):
        _VERIFIER._test_definition_digests(
            legacy_duplicate_source
            + "    def test_search_messages(self):\n"
            "        assert value == 3\n",
            legacy_duplicate_path,
        )
    assert _VERIFIER._common_test_definition_order_is_preserved(
        {"first": "1", "second": "2"},
        {"first": "3", "second": "4"},
    )
    assert not _VERIFIER._common_test_definition_order_is_preserved(
        {"first": "1", "second": "2"},
        {"second": "4", "first": "3"},
    )
    structural_source = (
        "if True:\n"
        "    class TestConditional:\n"
        "        try:\n"
        "            def test_visible(self):\n"
        "                assert True\n"
        "        except Exception:\n"
        "            pass\n\n"
        "def helper():\n"
        "    def test_local():\n"
        "        assert False\n"
        "    return test_local\n"
    )
    structural_definitions = _VERIFIER._test_definition_digests(
        structural_source,
        "tests/test_structural.py",
    )
    assert tuple(structural_definitions) == (
        "tests/test_structural.py::TestConditional::test_visible",
    )

    changed_paths = (
        "tests/combined_test.py",
        "tests/test_changed_pattern.py",
    )
    for relative, source in (
        (
            changed_paths[0],
            "def test_suffix_pattern():\n    assert True\n",
        ),
        (
            changed_paths[1],
            (
                "import pytest\n\n"
                "if True:\n"
                "    @pytest.mark.parametrize(\n"
                "        'value', [1], ids=['value::[edge]']\n"
                "    )\n"
                "    def test_conditional(value):\n"
                "        assert value == 1\n\n"
                "def helper():\n"
                "    def test_local():\n"
                "        assert False\n"
                "    return test_local\n"
            ),
        ),
    ):
        changed_path = tmp_path / relative
        changed_path.parent.mkdir(parents=True, exist_ok=True)
        changed_path.write_text(source, encoding="utf-8")
    collect_calls: list[tuple[str, ...]] = []
    original_run_probe = _VERIFIER._run_probe

    def collect_once(
        driver: str,
        sentinel: str,
        node_ids: tuple[str, ...],
        root: Path,
    ) -> dict[str, object]:
        collect_calls.append(node_ids)
        result: dict[str, object] = original_run_probe(
            driver,
            sentinel,
            node_ids,
            root,
        )
        return result

    with monkeypatch.context() as patch_context:
        patch_context.setattr(_VERIFIER, "_git_returncode", lambda *_args: 0)
        patch_context.setattr(
            _VERIFIER,
            "_git_lines",
            lambda *_args: changed_paths,
        )
        patch_context.setattr(
            _VERIFIER,
            "_current_baseline_source",
            lambda *_args: None,
        )
        patch_context.setattr(
            _VERIFIER,
            "_validate_frozen_duplicate_test_definitions",
            lambda *_args: None,
        )
        patch_context.setattr(
            _VERIFIER,
            "_EXPECTED_NONCONVENTION_TEST_PATHS",
            frozenset(),
        )
        patch_context.setattr(
            _VERIFIER,
            "_EXPECTED_INHERITED_COLLECTIONS",
            frozenset(),
        )
        patch_context.setattr(
            _VERIFIER,
            "_EXPECTED_INHERITED_COLLECTION_SHA256",
            sha256(b"").hexdigest(),
        )
        patch_context.setattr(_VERIFIER, "_run_probe", collect_once)
        synthetic_baseline, synthetic_current, synthetic_paths = (
            _VERIFIER._current_changed_test_definitions(tmp_path)
        )
    assert not synthetic_baseline
    assert synthetic_paths == frozenset(changed_paths)
    assert set(synthetic_current) == {
        "tests/combined_test.py::test_suffix_pattern",
        "tests/test_changed_pattern.py::test_conditional",
    }
    assert collect_calls == [tuple(sorted(changed_paths))]

    mismatch_path = "tests/test_collection_mismatch.py"
    mismatch_file = tmp_path / mismatch_path
    support_module = tmp_path / "imported_support.py"
    support_module.write_text(
        "def test_imported():\n    assert True\n",
        encoding="utf-8",
    )
    mismatch_sources = (
        "def template():\n    assert True\n\ntest_dynamic = template\n",
        "from imported_support import test_imported\n",
    )
    for mismatch_source in mismatch_sources:
        mismatch_file.write_text(mismatch_source, encoding="utf-8")
        with monkeypatch.context() as patch_context:
            patch_context.setattr(
                _VERIFIER,
                "_git_returncode",
                lambda *_args: 0,
            )
            patch_context.setattr(
                _VERIFIER,
                "_git_lines",
                lambda *_args: (mismatch_path,),
            )
            patch_context.setattr(
                _VERIFIER,
                "_current_baseline_source",
                lambda *_args: None,
            )
            patch_context.setattr(
                _VERIFIER,
                "_validate_frozen_duplicate_test_definitions",
                lambda *_args: None,
            )
            patch_context.setattr(
                _VERIFIER,
                "_EXPECTED_NONCONVENTION_TEST_PATHS",
                frozenset(),
            )
            with pytest.raises(
                _VERIFIER.AcceptanceVerificationError,
                match="does not map to one static definition",
            ):
                _VERIFIER._current_changed_test_definitions(tmp_path)

    support_path = "tests/test_new_support.py"
    support_file = tmp_path / support_path
    support_source = (
        "VALUE = 1\n\n"
        "if VALUE:\n"
        "    def helper():\n"
        "        return VALUE\n\n"
        "    def test_case():\n"
        "        assert helper() == 1\n"
    )
    support_file.write_text(support_source, encoding="utf-8")
    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            _VERIFIER,
            "_current_baseline_source",
            lambda *_args: None,
        )
        new_baseline, new_current = (
            _VERIFIER._current_changed_test_support_surfaces(
                tmp_path,
                frozenset((support_path,)),
            )
        )
        assert new_baseline == {
            support_path: _VERIFIER._ABSENT_TEST_SUPPORT_SHA256
        }
        original_support_digest = new_current[support_path]
        support_file.write_text(
            support_source.replace("helper() == 1", "helper() == VALUE"),
            encoding="utf-8",
        )
        _, test_body_current = (
            _VERIFIER._current_changed_test_support_surfaces(
                tmp_path,
                frozenset((support_path,)),
            )
        )
        assert test_body_current[support_path] == original_support_digest
        for drifted_support in (
            support_source.replace("return VALUE", "return VALUE + 1"),
            support_source.replace("if VALUE:", "if not VALUE:"),
        ):
            support_file.write_text(drifted_support, encoding="utf-8")
            _, drifted_current = (
                _VERIFIER._current_changed_test_support_surfaces(
                    tmp_path,
                    frozenset((support_path,)),
                )
            )
            assert drifted_current[support_path] != original_support_digest

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            _VERIFIER,
            "_git_returncode",
            lambda *_args: 0,
        )
        patch_context.setattr(
            _VERIFIER,
            "_git_lines",
            lambda *_args: ("tests/deleted_test.py",),
        )
        patch_context.setattr(
            _VERIFIER,
            "_current_baseline_source",
            lambda *_args: "def test_deleted():\n    assert value\n",
        )
        patch_context.setattr(
            _VERIFIER,
            "_EXPECTED_NONCONVENTION_TEST_PATHS",
            frozenset(),
        )
        with pytest.raises(
            _VERIFIER.AcceptanceVerificationError,
            match="baseline test file was deleted",
        ):
            _VERIFIER._current_changed_test_definitions(tmp_path)
    duplicate_id, duplicate_record = next(
        iter(_VERIFIER._EXPECTED_CURRENT_DUPLICATE_TEST_DEFINITIONS.items())
    )
    duplicate_baseline = {duplicate_id: duplicate_record[1]}
    duplicate_current = {duplicate_id: duplicate_record[2]}
    _VERIFIER._validate_frozen_duplicate_test_definitions(
        duplicate_baseline,
        duplicate_current,
    )
    duplicate_drifts: tuple[tuple[dict[str, str], dict[str, str]], ...] = (
        ({}, duplicate_current),
        (duplicate_baseline, {duplicate_id: "0" * 64}),
    )
    for drifted_duplicates in duplicate_drifts:
        with pytest.raises(
            _VERIFIER.AcceptanceVerificationError,
            match="frozen duplicate test definition changed",
        ):
            _VERIFIER._validate_frozen_duplicate_test_definitions(
                *drifted_duplicates
            )

    stale_support = deepcopy(classification)
    stale_support["support_surfaces"][0]["current_ast_sha256"] = "0" * 64
    stale_digest_value = {
        "mechanical_nodes": stale_support["mechanical_nodes"],
        "reviewed_nonsemantic_nodes": stale_support[
            "reviewed_nonsemantic_nodes"
        ],
        "support_surfaces": stale_support["support_surfaces"],
    }
    stale_digest = _canonical_digest(stale_digest_value)
    stale_support["catalog_sha256"] = stale_digest
    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            _VERIFIER,
            "_EXPECTED_CURRENT_REGRESSION_SHA256",
            stale_digest,
        )
        with pytest.raises(
            _VERIFIER.AcceptanceVerificationError,
            match="support surface differs from its exact reviewed file",
        ):
            _VERIFIER._validate_current_regression_classification(
                stale_support,
                manifest,
                _ROOT,
            )

    support_variants: list[tuple[dict[str, Any], str]] = []
    equal_support = deepcopy(classification)
    equal_support["support_surfaces"][0]["current_ast_sha256"] = equal_support[
        "support_surfaces"
    ][0]["baseline_ast_sha256"]
    support_variants.append(
        (
            equal_support,
            "support surface differs from its exact reviewed file",
        )
    )
    missing_support = deepcopy(classification)
    missing_support["support_surfaces"].pop()
    support_variants.append(
        (
            missing_support,
            "support surfaces lack exact classification",
        )
    )
    duplicate_support = deepcopy(classification)
    duplicate_support["support_surfaces"].append(
        deepcopy(duplicate_support["support_surfaces"][0])
    )
    support_variants.append(
        (
            duplicate_support,
            "duplicate current support-surface path",
        )
    )
    new_only_support = deepcopy(classification)
    new_only_support["support_surfaces"][0]["path"] = "tests/new_only_test.py"
    support_variants.append(
        (
            new_only_support,
            "support surface differs from its exact reviewed file",
        )
    )
    for support_variant, message in support_variants:
        variant_digest_value = {
            "mechanical_nodes": support_variant["mechanical_nodes"],
            "reviewed_nonsemantic_nodes": support_variant[
                "reviewed_nonsemantic_nodes"
            ],
            "support_surfaces": support_variant["support_surfaces"],
        }
        variant_digest = _canonical_digest(variant_digest_value)
        support_variant["catalog_sha256"] = variant_digest
        with monkeypatch.context() as patch_context:
            patch_context.setattr(
                _VERIFIER,
                "_EXPECTED_CURRENT_REGRESSION_SHA256",
                variant_digest,
            )
            patch_context.setattr(
                _VERIFIER,
                "_EXPECTED_CURRENT_SUPPORT_SURFACE_COUNT",
                len(support_variant["support_surfaces"]),
            )
            with pytest.raises(
                _VERIFIER.AcceptanceVerificationError,
                match=message,
            ):
                _VERIFIER._validate_current_regression_classification(
                    support_variant,
                    manifest,
                    _ROOT,
                )

    legacy = (
        "tests/agent/additional_coverage_test.py::"
        "EngineAgentCoverageTestCase::test_output_property"
    )
    drifted = {**current, legacy: "0" * 64}
    monkeypatch.setattr(
        _VERIFIER,
        "_current_changed_test_definitions",
        lambda root: (baseline, drifted, paths),
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="new test definitions lack semantic acceptance",
    ):
        _VERIFIER._validate_current_regression_classification(
            classification, manifest, _ROOT
        )

    removed = dict(current)
    removed_id = next(
        node.node_id
        for node in manifest.nodes
        if node.lifecycle == "active"
        and node.node_id in baseline
        and node.node_id in current
        and baseline[node.node_id] != current[node.node_id]
    )
    removed.pop(removed_id)
    monkeypatch.setattr(
        _VERIFIER,
        "_current_changed_test_definitions",
        lambda root: (baseline, removed, paths),
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="removed test definitions lack exact semantic replacement",
    ):
        _VERIFIER._validate_current_regression_classification(
            classification, manifest, _ROOT
        )

    hidden = "tests/agent/unreviewed_semantic_test.py::test_hidden_behavior"
    monkeypatch.setattr(
        _VERIFIER,
        "_current_changed_test_definitions",
        lambda root: (baseline, {**current, hidden: "0" * 64}, paths),
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="new test definitions lack semantic acceptance",
    ):
        _VERIFIER._validate_current_regression_classification(
            classification, manifest, _ROOT
        )

    mechanical = deepcopy(classification)
    reviewed = mechanical["reviewed_nonsemantic_nodes"].pop(0)
    mechanical["mechanical_nodes"].append(
        {
            "node_id": hidden,
            "baseline_ast_sha256": reviewed["baseline_ast_sha256"],
            "current_ast_sha256": reviewed["current_ast_sha256"],
            "disposition": "mechanical_fixture_or_assertion_migration",
            "evidence": reviewed["evidence"],
        }
    )
    digest_value = {
        "mechanical_nodes": mechanical["mechanical_nodes"],
        "reviewed_nonsemantic_nodes": mechanical["reviewed_nonsemantic_nodes"],
        "support_surfaces": mechanical["support_surfaces"],
    }
    digest = _canonical_digest(digest_value)
    mechanical["catalog_sha256"] = digest
    monkeypatch.setattr(
        _VERIFIER, "_EXPECTED_CURRENT_REGRESSION_SHA256", digest
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_CURRENT_REGRESSION_NODE_COUNT",
        len(mechanical["mechanical_nodes"]),
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="mechanical regression differs",
    ):
        _VERIFIER._validate_current_regression_classification(
            mechanical, manifest, _ROOT
        )


def test_production_capability_history_is_atomic() -> None:
    """Reject capability advertisement before the atomic boundary."""
    history = [
        {"phase": 0, "state": "absent"},
        {"phase": 1, "state": "dormant_unadvertised"},
    ]
    assert _VERIFIER._production_capability_history(history, 1) == (
        "absent",
        "dormant_unadvertised",
    )
    final_history = [
        {
            "phase": phase,
            "state": (
                "absent"
                if phase == 0
                else "active" if phase == 12 else "dormant_unadvertised"
            ),
        }
        for phase in range(13)
    ]
    assert (
        _VERIFIER._production_capability_history(final_history, 12)[-1]
        == "active"
    )
    invalid_histories: tuple[object, ...] = (
        None,
        history[:1],
        [history[0], "invalid"],
        [history[0], {"phase": 1, "state": "active", "extra": True}],
        [history[0], {"phase": 0, "state": "dormant_unadvertised"}],
        [history[0], {"phase": 1, "state": "active"}],
    )
    for invalid in invalid_histories:
        with pytest.raises(_VERIFIER.AcceptanceVerificationError):
            _VERIFIER._production_capability_history(invalid, 1)


def test_tree_binding_is_commit_stable_and_fail_closed(
    tmp_path: Path,
) -> None:
    """Bind file content independently of tracked working-tree state."""
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "acceptance@example.invalid")
    _git(tmp_path, "config", "user.name", "Acceptance Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _git(tmp_path, "config", "core.filemode", "true")
    verifier = tmp_path / "scripts" / "verify_input_acceptance.py"
    verifier.parent.mkdir()
    verifier.write_text(
        f'EXPECTED = "{_VERIFIER._EXPECTED_EVIDENCE_SHA256}"\n',
        encoding="utf-8",
    )
    evidence_path = (
        tmp_path / "tests" / "fixtures" / "input" / "baseline_evidence.json"
    )
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("{}\n", encoding="utf-8")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "--quiet", "-m", "baseline")

    tracked.write_text("updated\n", encoding="utf-8")
    added = tmp_path / "added.txt"
    added.write_text("new\n", encoding="utf-8")
    preserved = tmp_path / "preserved" / "example.txt"
    preserved.parent.mkdir()
    preserved.write_text("unrelated\n", encoding="utf-8")
    evidence = {"quality_gate": {"tree_binding": {"stale": True}}}
    before_commit = _VERIFIER._current_tree_binding(
        tmp_path,
        ("preserved/",),
        evidence,
    )
    assert before_commit.keys() == {
        "baseline_head",
        "inventory_file_count",
        "inventory_sha256",
        "normalized_evidence_kind",
        "normalized_evidence_sha256",
        "normalized_verifier_kind",
        "normalized_verifier_sha256",
        "source_tree_file_count",
        "source_tree_inventory_sha256",
        "test_tree_file_count",
        "test_tree_inventory_sha256",
        "script_tree_file_count",
        "script_tree_inventory_sha256",
        "support_tree_boundary",
        "support_tree_file_count",
        "support_tree_inventory_sha256",
        "tree_sha256",
    }
    assert before_commit["inventory_file_count"] == 4
    assert before_commit["source_tree_file_count"] == 0
    assert before_commit["test_tree_file_count"] == 1
    assert before_commit["script_tree_file_count"] == 1
    assert before_commit["support_tree_file_count"] == 2
    assert (
        before_commit["support_tree_boundary"]
        == "all normalized files outside src/, tests/, and scripts/"
    )
    _git(tmp_path, "add", "tracked.txt", "added.txt")
    _git(tmp_path, "commit", "--quiet", "-m", "update")
    assert (
        _VERIFIER._current_tree_binding(
            tmp_path,
            ("preserved/",),
            evidence,
        )
        == before_commit
    )

    for protected in (evidence_path, verifier):
        original_mode = protected.stat().st_mode
        protected.chmod(original_mode | 0o100)
        assert (
            _VERIFIER._current_tree_binding(
                tmp_path,
                ("preserved/",),
                evidence,
            )
            != before_commit
        )
        protected.chmod(original_mode)
        assert (
            _VERIFIER._current_tree_binding(
                tmp_path,
                ("preserved/",),
                evidence,
            )
            == before_commit
        )

    _git(tmp_path, "update-index", "--chmod=+x", "added.txt")
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="staged and unstaged changes|modes differ",
    ):
        _VERIFIER._current_tree_binding(
            tmp_path,
            ("preserved/",),
            evidence,
        )
    _git(tmp_path, "update-index", "--chmod=-x", "added.txt")

    added.write_text("staged\n", encoding="utf-8")
    _git(tmp_path, "add", "added.txt")
    added.write_text("unstaged\n", encoding="utf-8")
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="staged and unstaged changes",
    ):
        _VERIFIER._current_tree_binding(
            tmp_path,
            ("preserved/",),
            evidence,
        )
    added.write_text("new\n", encoding="utf-8")
    _git(tmp_path, "add", "added.txt")

    tracked.unlink()
    before_deletion_commit = _VERIFIER._current_tree_binding(
        tmp_path,
        ("preserved/",),
        evidence,
    )
    _git(tmp_path, "add", "--update", "tracked.txt")
    _git(tmp_path, "commit", "--quiet", "-m", "delete")
    assert (
        _VERIFIER._current_tree_binding(
            tmp_path,
            ("preserved/",),
            evidence,
        )
        == before_deletion_commit
    )

    added.write_text("changed\n", encoding="utf-8")
    assert (
        _VERIFIER._current_tree_binding(
            tmp_path,
            ("preserved/",),
            evidence,
        )
        != before_deletion_commit
    )
    unsafe = tmp_path / "unsafe-link"
    unsafe.symlink_to(added)
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="entry is a symlink",
    ):
        _VERIFIER._current_tree_binding(
            tmp_path,
            ("preserved/",),
            evidence,
        )


def test_evidence_state_and_review_history_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject premature gate claims and non-append-only review changes."""
    evidence = _read("baseline_evidence.json")
    inventory = evidence["inventory"]
    assert isinstance(inventory, dict)
    active_acceptance_nodes = inventory["active_acceptance_nodes"]
    active_pytest_instances = inventory["active_pytest_instances"]
    assert active_acceptance_nodes == 814
    type_manifest = _read("type_contract_manifest.json")
    active_type_fixtures = sum(
        fixture["lifecycle"] == "active"
        for fixture in type_manifest["fixtures"]
    )
    complete_evidence = deepcopy(evidence["quality_gate"])
    pending = {
        "state": "pending",
        "required_commands": complete_evidence["required_commands"],
        "state_details": {
            "requested_at": "2026-07-24T10:03:24Z",
            "reason": (
                "Synthetic pending evidence has no completed results or"
                " bindings."
            ),
        },
        "results": [],
        "tree_binding": {},
        "coverage_binding": {},
    }
    _VERIFIER._validate_quality_gate_evidence(
        pending,
        active_acceptance_nodes=active_acceptance_nodes,
        active_pytest_instances=active_pytest_instances,
        active_type_fixtures=active_type_fixtures,
        root=_ROOT,
        preserved_untracked=("docs/examples/skills/code/",),
        evidence_payload=evidence,
    )
    incomplete_focused_command = deepcopy(pending)
    incomplete_focused_command["required_commands"][
        -1
    ] = "poetry run pytest --verbose -s tests/agent/execution_test.py"
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="exact ordered common gate commands",
    ):
        _VERIFIER._validate_quality_gate_evidence(
            incomplete_focused_command,
            active_acceptance_nodes=active_acceptance_nodes,
            active_pytest_instances=active_pytest_instances,
            active_type_fixtures=active_type_fixtures,
            root=_ROOT,
            preserved_untracked=("docs/examples/skills/code/",),
            evidence_payload=evidence,
        )
    premature = deepcopy(pending)
    premature["results"] = [{"command": "not executed"}]
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="cannot claim completed",
    ):
        _VERIFIER._validate_quality_gate_evidence(
            premature,
            active_acceptance_nodes=active_acceptance_nodes,
            active_pytest_instances=active_pytest_instances,
            active_type_fixtures=active_type_fixtures,
            root=_ROOT,
            preserved_untracked=("docs/examples/skills/code/",),
            evidence_payload=evidence,
        )

    commands = pending["required_commands"]
    focused = commands[2]
    results = [
        {
            "command": commands[0],
            "exit_code": 0,
            "source_files_typechecked": 10,
            "script_files_typechecked": 6,
        },
        {
            "command": commands[1],
            "exit_code": 0,
            "active_fixtures": active_type_fixtures,
        },
        {
            "command": focused,
            "exit_code": 0,
            "active_nodes": active_acceptance_nodes,
            "active_instances": active_pytest_instances,
            "covered_statements": 100,
            "total_statements": 100,
            "source_files": 10,
            "missing_lines": 0,
            "missing_files": 0,
            "passed": 1,
            "skipped": 0,
            "subtests_passed": 1,
            "seconds": 1.0,
            "deselected": 0,
            "xfail": 0,
            "xpass": 0,
        },
        {"command": commands[3], "exit_code": 0},
    ]
    tree_binding = {
        "head": "a" * 40,
        "diff_sha256": "b" * 64,
        "untracked_inventory_sha256": "c" * 64,
        "normalized_evidence_sha256": "d" * 64,
        "normalized_verifier_sha256": "e" * 64,
        "tree_sha256": "f" * 64,
    }
    inventory = ("1" * 64, 10, 100, 5)
    complete = {
        "state": "complete",
        "required_commands": commands,
        "state_details": {
            "completed_at": "2026-07-21T04:00:00-03:00",
            "gate_run_id": "round-5-gate-run",
            "final_review": {
                "reviewer": "/root/public_sdk_gate_review",
                "status": "approved",
                "approval_sealed": True,
            },
            "prior_failed_attempts": [
                {
                    "attempt": attempt,
                    "command": focused,
                    "outcome": "failed",
                    "passed": 12000,
                    "skipped": 59,
                    "subtests_passed": 8678,
                    "seconds": seconds,
                    "failure_stage": failure_stage,
                    "failure": (
                        "Concrete failed gate provenance remains preserved."
                    ),
                }
                for attempt, seconds, failure_stage in (
                    (1, 366.53, "coverage_exclusion_verification"),
                    (2, 362.09, "coverage_exclusion_verification"),
                    (3, 366.48, "exact_source_coverage"),
                )
            ],
            "diagnostic_coverage": {
                "purpose": "diagnostic_only",
                "passed": 12000,
                "skipped": 59,
                "subtests_passed": 8678,
                "seconds": 368.56,
                "report_sha256": (
                    "48b0587756849bc1c22fdf437f0f03643ad9e2c9788e4166"
                    "168239e31be2cf7f"
                ),
            },
            "hard_coverage_audit": {
                "command": "make test-coverage -- -100 src/",
                "exit_code": 0,
                "below_threshold_files": [],
            },
        },
        "results": results,
        "tree_binding": tree_binding,
        "coverage_binding": {
            "report_sha256": "2" * 64,
            "xml_report_sha256": "3" * 64,
            "source_inventory_sha256": inventory[0],
            "source_file_count": inventory[1],
            "statement_count": inventory[2],
            "excluded_line_count": inventory[3],
        },
    }
    monkeypatch.setattr(
        _VERIFIER,
        "_current_tree_binding",
        lambda *args: tree_binding,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_source_statement_inventory",
        lambda root: inventory,
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_coverage_report_binding",
        lambda root: ("2" * 64, *inventory),
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_coverage_xml_report_digest",
        lambda root: "3" * 64,
    )
    verified_coverage = SimpleNamespace(
        files=tuple(f"src/file_{index}.py" for index in range(inventory[1])),
        summary=SimpleNamespace(
            covered_lines=inventory[2],
            excluded_lines=inventory[3],
            missing_lines=0,
            num_statements=inventory[2],
        ),
    )
    monkeypatch.setattr(
        _VERIFIER,
        "verify_src_coverage",
        lambda **kwargs: verified_coverage,
    )
    _VERIFIER._validate_quality_gate_evidence(
        complete,
        active_acceptance_nodes=active_acceptance_nodes,
        active_pytest_instances=active_pytest_instances,
        active_type_fixtures=active_type_fixtures,
        root=_ROOT,
        preserved_untracked=("docs/examples/skills/code/",),
        evidence_payload=evidence,
    )
    unsealed_review = deepcopy(complete)
    unsealed_review["state_details"]["final_review"]["approval_sealed"] = False
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="approved sealed verdict",
    ):
        _VERIFIER._validate_quality_gate_evidence(
            unsealed_review,
            active_acceptance_nodes=active_acceptance_nodes,
            active_pytest_instances=active_pytest_instances,
            active_type_fixtures=active_type_fixtures,
            root=_ROOT,
            preserved_untracked=("docs/examples/skills/code/",),
            evidence_payload=evidence,
        )
    stale_focused = deepcopy(complete)
    stale_focused["results"][2]["active_instances"] -= 1
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="exact PostgreSQL acceptance evidence is incomplete",
    ):
        _VERIFIER._validate_quality_gate_evidence(
            stale_focused,
            active_acceptance_nodes=active_acceptance_nodes,
            active_pytest_instances=active_pytest_instances,
            active_type_fixtures=active_type_fixtures,
            root=_ROOT,
            preserved_untracked=("docs/examples/skills/code/",),
            evidence_payload=evidence,
        )
    missing_report = {
        "meta": {"format": 3},
        "files": {
            "src/sample.py": {
                "executed_lines": [1],
                "missing_lines": [2],
                "excluded_lines": [],
                "summary": {
                    "covered_lines": 1,
                    "excluded_lines": 0,
                    "missing_lines": 1,
                    "num_statements": 2,
                },
            }
        },
        "totals": {
            "covered_lines": 1,
            "excluded_lines": 0,
            "missing_lines": 1,
            "num_statements": 2,
        },
    }
    missing_report_path = tmp_path / "coverage.json"
    _write(missing_report_path, missing_report)

    def reject_missing_report(*, report_path: Path, repo_root: Path) -> object:
        assert report_path == missing_report_path
        assert repo_root == tmp_path
        payload = loads(report_path.read_text(encoding="utf-8"))
        summary = payload["files"]["src/sample.py"]["summary"]
        if summary["missing_lines"]:
            raise _VERIFIER.CoverageVerificationError(
                "source coverage is not exact"
            )
        return verified_coverage

    monkeypatch.setattr(
        _VERIFIER,
        "verify_src_coverage",
        reject_missing_report,
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="live exact source coverage is invalid",
    ):
        _VERIFIER._validate_quality_gate_evidence(
            complete,
            active_acceptance_nodes=active_acceptance_nodes,
            active_pytest_instances=active_pytest_instances,
            active_type_fixtures=active_type_fixtures,
            root=tmp_path,
            preserved_untracked=("docs/examples/skills/code/",),
            evidence_payload=evidence,
        )
    monkeypatch.setattr(
        _VERIFIER,
        "verify_src_coverage",
        lambda **kwargs: verified_coverage,
    )
    stale_tree = deepcopy(complete)
    stale_tree["tree_binding"]["tree_sha256"] = "0" * 64
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="live git tree",
    ):
        _VERIFIER._validate_quality_gate_evidence(
            stale_tree,
            active_acceptance_nodes=active_acceptance_nodes,
            active_pytest_instances=active_pytest_instances,
            active_type_fixtures=active_type_fixtures,
            root=_ROOT,
            preserved_untracked=("docs/examples/skills/code/",),
            evidence_payload=evidence,
        )
    stale_report = deepcopy(complete)
    stale_report["coverage_binding"]["report_sha256"] = "4" * 64
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="live report",
    ):
        _VERIFIER._validate_quality_gate_evidence(
            stale_report,
            active_acceptance_nodes=active_acceptance_nodes,
            active_pytest_instances=active_pytest_instances,
            active_type_fixtures=active_type_fixtures,
            root=_ROOT,
            preserved_untracked=("docs/examples/skills/code/",),
            evidence_payload=evidence,
        )
    stale_xml_report = deepcopy(complete)
    stale_xml_report["coverage_binding"]["xml_report_sha256"] = "4" * 64
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="coverage XML digest",
    ):
        _VERIFIER._validate_quality_gate_evidence(
            stale_xml_report,
            active_acceptance_nodes=active_acceptance_nodes,
            active_pytest_instances=active_pytest_instances,
            active_type_fixtures=active_type_fixtures,
            root=_ROOT,
            preserved_untracked=("docs/examples/skills/code/",),
            evidence_payload=evidence,
        )
    stale_coverage = deepcopy(complete)
    stale_coverage["coverage_binding"]["statement_count"] = 99
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="live source tree",
    ):
        _VERIFIER._validate_quality_gate_evidence(
            stale_coverage,
            active_acceptance_nodes=active_acceptance_nodes,
            active_pytest_instances=active_pytest_instances,
            active_type_fixtures=active_type_fixtures,
            root=_ROOT,
            preserved_untracked=("docs/examples/skills/code/",),
            evidence_payload=evidence,
        )

    history = deepcopy(evidence["review_history"])
    _VERIFIER._validate_review_history(
        history,
        evidence["review_history_sha256"],
        evidence["review_history_phase0_sha256"],
        evidence["review_history_phase1_sha256"],
        evidence["review_history_phase2_sha256"],
        evidence["review_history_prior_sha256"],
        6,
        "/root",
    )
    self_review = deepcopy(history)
    self_review[-1]["reviewer"] = "/root"
    self_review_digest = _canonical_digest(self_review)
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_REVIEW_HISTORY_SHA256",
        self_review_digest,
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="implementation owner cannot review its own evidence",
    ):
        _VERIFIER._validate_review_history(
            self_review,
            self_review_digest,
            evidence["review_history_phase0_sha256"],
            evidence["review_history_phase1_sha256"],
            evidence["review_history_phase2_sha256"],
            evidence["review_history_prior_sha256"],
            6,
            "/root",
        )
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_REVIEW_HISTORY_SHA256",
        evidence["review_history_sha256"],
    )
    quality_history = deepcopy(evidence["quality_history"])
    _VERIFIER._validate_quality_history(
        quality_history,
        evidence["quality_history_sha256"],
        6,
    )
    rewritten_quality = deepcopy(quality_history)
    rewritten_quality[0]["quality_gate_sha256"] = "0" * 64
    rewritten_quality_digest = _canonical_digest(rewritten_quality)
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_QUALITY_HISTORY_SHA256",
        rewritten_quality_digest,
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="lost its phase-1 record",
    ):
        _VERIFIER._validate_quality_history(
            rewritten_quality,
            rewritten_quality_digest,
            6,
        )

    rewritten_phase2_quality = deepcopy(quality_history)
    rewritten_phase2_quality[1]["quality_gate_sha256"] = "1" * 64
    rewritten_phase2_quality_digest = _canonical_digest(
        rewritten_phase2_quality
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_QUALITY_HISTORY_SHA256",
        rewritten_phase2_quality_digest,
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="lost its phase-2 record",
    ):
        _VERIFIER._validate_quality_history(
            rewritten_phase2_quality,
            rewritten_phase2_quality_digest,
            6,
        )

    rewritten_prior_quality = deepcopy(quality_history)
    rewritten_prior_quality[2]["quality_gate_sha256"] = "2" * 64
    rewritten_prior_quality_digest = _canonical_digest(rewritten_prior_quality)
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_QUALITY_HISTORY_SHA256",
        rewritten_prior_quality_digest,
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="lost its phase-3 record",
    ):
        _VERIFIER._validate_quality_history(
            rewritten_prior_quality,
            rewritten_prior_quality_digest,
            6,
        )

    rewritten_frozen_quality = deepcopy(quality_history)
    rewritten_frozen_quality[3]["quality_gate_sha256"] = "3" * 64
    rewritten_frozen_quality_digest = _canonical_digest(
        rewritten_frozen_quality
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_QUALITY_HISTORY_SHA256",
        rewritten_frozen_quality_digest,
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="quality history lost its .* record",
    ):
        _VERIFIER._validate_quality_history(
            rewritten_frozen_quality,
            rewritten_frozen_quality_digest,
            6,
        )

    rewritten_phase5_quality = deepcopy(quality_history)
    rewritten_phase5_quality[4]["quality_gate_sha256"] = "4" * 64
    rewritten_phase5_quality_digest = _canonical_digest(
        rewritten_phase5_quality
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_QUALITY_HISTORY_SHA256",
        rewritten_phase5_quality_digest,
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="quality history lost its phase-5 record",
    ):
        _VERIFIER._validate_quality_history(
            rewritten_phase5_quality,
            rewritten_phase5_quality_digest,
            6,
        )

    wrong_terminal_reviewer = deepcopy(history)
    wrong_terminal_reviewer[16]["reviewer"] = "/root/acceptance_review"
    wrong_terminal_reviewer_digest = _canonical_digest(wrong_terminal_reviewer)
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_REVIEW_HISTORY_SHA256",
        wrong_terminal_reviewer_digest,
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="occurrence identity or status changed",
    ):
        _VERIFIER._validate_review_history(
            wrong_terminal_reviewer,
            wrong_terminal_reviewer_digest,
            evidence["review_history_phase0_sha256"],
            evidence["review_history_phase1_sha256"],
            evidence["review_history_phase2_sha256"],
            evidence["review_history_prior_sha256"],
            6,
            "/root",
        )
    for index, message in (
        (0, "phase-0 review prefix digest mismatch"),
        (3, "phase-1 review prefix digest mismatch"),
        (5, "phase-2 pending review prefix digest mismatch"),
        (7, "phase-2 review prefix digest mismatch"),
        (9, "historical review prefix digest mismatch"),
    ):
        rewritten = deepcopy(history)
        rewritten[index]["evidence"] += " rewritten"
        rewritten_digest = _canonical_digest(rewritten)
        monkeypatch.setattr(
            _VERIFIER,
            "_EXPECTED_REVIEW_HISTORY_SHA256",
            rewritten_digest,
        )
        with pytest.raises(
            _VERIFIER.AcceptanceVerificationError,
            match=message,
        ):
            _VERIFIER._validate_review_history(
                rewritten,
                rewritten_digest,
                evidence["review_history_phase0_sha256"],
                evidence["review_history_phase1_sha256"],
                evidence["review_history_phase2_sha256"],
                evidence["review_history_prior_sha256"],
                6,
                "/root",
            )


def test_pending_structural_inventory_is_key_order_independent_and_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept reordered fields and reject each incorrect inventory field."""
    expected = _VERIFIER._EXPECTED_PENDING_SOURCE_INVENTORY
    reordered = {
        "excluded_line_count": expected[3],
        "statement_count": expected[2],
        "source_file_count": expected[1],
        "source_inventory_sha256": expected[0],
    }
    monkeypatch.setattr(
        _VERIFIER, "_source_statement_inventory", lambda root: expected
    )
    _VERIFIER._validate_pending_structural_inventory(reordered, _ROOT)

    mutations = {
        "source_inventory_sha256": "0" * 64,
        "source_file_count": expected[1] + 1,
        "statement_count": expected[2] + 1,
        "excluded_line_count": expected[3] + 1,
    }
    for field, value in mutations.items():
        invalid = dict(reordered)
        invalid[field] = value
        with pytest.raises(
            _VERIFIER.AcceptanceVerificationError,
            match="differs from the live source tree",
        ):
            _VERIFIER._validate_pending_structural_inventory(invalid, _ROOT)


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


def test_acceptance_rejects_failure_schedule_evidence_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require the exact reviewed 58-cell forward schedule correction."""
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

    missing_group = deepcopy(matrix)
    missing_group["activation_schedule_corrections"].pop()
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="frozen 58-cell review",
    ):
        validate(missing_group)

    overlap = deepcopy(matrix)
    overlap["activation_schedule_corrections"].append(
        deepcopy(overlap["activation_schedule_corrections"][0])
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="overlaps cell",
    ):
        validate(overlap)

    backward = deepcopy(matrix)
    backward["activation_schedule_corrections"][0][
        "corrected_active_from_phase"
    ] = 5
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="move forward from the current implemented boundary",
    ):
        validate(backward)

    unledgered_phase = deepcopy(matrix)
    unledgered_cell = next(
        cell
        for cell in unledgered_phase["cells"]
        if cell["condition_id"] == "INPUT-F-04"
        and cell["surface_id"] == "task-target-agent-direct"
    )
    unledgered_cell["active_from_phase"] = 6
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="frozen natural or reviewed corrected phase",
    ):
        validate(unledgered_phase)

    wrong_owner = deepcopy(matrix)
    corrected_cell = next(
        cell
        for cell in wrong_owner["cells"]
        if cell["condition_id"] == "INPUT-F-04"
        and cell["surface_id"] == "task-client-inspect"
    )
    corrected_cell["negative_e2e_node"] = (
        "tests/input/failure_matrix_sdk_e2e_test.py::test_input_f_04"
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="wrong planned E2E owner",
    ):
        validate(wrong_owner)


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
    collected = _VERIFIER._verify_collection(nodes, collection)
    execution = _VERIFIER._run_probe(
        _VERIFIER._EXECUTE_DRIVER,
        _VERIFIER._EXECUTE_SENTINEL,
        nodes,
        tmp_path,
    )
    _VERIFIER._verify_execution(nodes, execution, collected)
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
        del driver
        common: dict[str, object] = {
            "exit_code": 0,
            "deselected": [],
            "collection_reports": [],
            "probe_stdout": "",
            "probe_stderr": "",
        }
        if sentinel == _VERIFIER._COLLECT_SENTINEL:
            collected_node_ids = node_ids
            if all(node_id.endswith(".py") for node_id in node_ids):
                collected_node_ids = tuple(
                    definition
                    for relative in node_ids
                    for definition in _VERIFIER._test_definition_digests(
                        (root / relative).read_text(encoding="utf-8"),
                        relative,
                    )
                )
            return {
                **common,
                "items": [
                    {"nodeid": node_id, "markers": []}
                    for node_id in collected_node_ids
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
                    "user_properties": [],
                }
                for node_id in node_ids
                for when in ("setup", "call", "teardown")
            ],
        }

    monkeypatch.setattr(Path, "read_text", read_non_markdown)
    monkeypatch.setattr(_VERIFIER, "_run_probe", successful_probe)
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_INHERITED_COLLECTIONS",
        frozenset(),
    )
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_INHERITED_COLLECTION_SHA256",
        sha256(b"").hexdigest(),
    )
    live_manifest = _VERIFIER.verify_acceptance(
        _FIXTURES / "acceptance_manifest.json",
        repo_root=_ROOT,
        through_phase=0,
        contract_fixture_root=_FIXTURES,
    )
    assert len(live_manifest.active_nodes(0)) == 23
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
    parameter_instances = (f"{node}[case-a]", f"{node}[case-b]")
    collection["items"] = [
        {"nodeid": parameter, "markers": []}
        for parameter in parameter_instances
    ]
    assert (
        _VERIFIER._verify_collection(parameter_instances, collection)
        == parameter_instances
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="unexpected=",
    ):
        _VERIFIER._verify_collection(
            (parameter_instances[0],),
            collection,
        )
    for mutated_instances in (
        parameter_instances[:1],
        (*parameter_instances, f"{node}[case-c]"),
        (parameter_instances[0], f"{node}[renamed]"),
        (parameter_instances[0], parameter_instances[0]),
    ):
        mutated_collection = deepcopy(collection)
        mutated_collection["items"] = [
            {"nodeid": parameter, "markers": []}
            for parameter in mutated_instances
        ]
        with pytest.raises(
            _VERIFIER.AcceptanceVerificationError,
            match=(
                "acceptance nodes were not exactly collected"
                "|duplicate collected node ID"
            ),
        ):
            _VERIFIER._verify_collection(
                parameter_instances,
                mutated_collection,
            )
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
                "user_properties": [],
            }
        ],
        "probe_stdout": "",
        "probe_stderr": "",
    }
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError, match="exactly once"
    ):
        _VERIFIER._verify_execution((node,), execution, (node,))

    execution["reports"] = [
        {
            "nodeid": node,
            "when": when,
            "outcome": "failed" if when == "call" else "passed",
            "wasxfail": "",
            "detail": "synthetic failure" if when == "call" else "",
            "user_properties": [],
        }
        for when in ("setup", "call", "teardown")
    ]
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="call outcome was failed",
    ):
        _VERIFIER._verify_execution((node,), execution, (node,))

    execution["reports"] = [
        {
            "nodeid": node,
            "when": when,
            "outcome": "passed",
            "wasxfail": "synthetic expectation" if when == "call" else "",
            "detail": "",
            "user_properties": [],
        }
        for when in ("setup", "call", "teardown")
    ]
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="xfail/xpass",
    ):
        _VERIFIER._verify_execution((node,), execution, (node,))

    execution["reports"] = [
        {
            "nodeid": node,
            "when": when,
            "outcome": "passed",
            "wasxfail": "",
            "detail": "",
            "user_properties": [],
        }
        for when in ("setup", "call", "call", "teardown")
    ]
    _VERIFIER._verify_execution((node,), execution, (node,))
    execution["reports"][2]["outcome"] = "failed"
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="call outcome was failed",
    ):
        _VERIFIER._verify_execution((node,), execution, (node,))

    execution["items"] = list(parameter_instances)
    execution["reports"] = []
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="unexpected=",
    ):
        _VERIFIER._verify_execution((node,), execution, (node,))


def test_acceptance_binds_failure_cells_to_dynamic_postconditions() -> None:
    """Reject inventory-only or drifted failure-matrix execution evidence."""
    all_expectations = _VERIFIER._failure_evidence_expectations(
        _FIXTURES,
        5,
    )
    node = "tests/input/failure_matrix_task_e2e_test.py::test_input_f_04"
    expectations = all_expectations[node]
    examples = _read("contract_decisions.json")["error_status"][
        "public_envelope_examples"
    ]
    observed: list[dict[str, object]] = []
    for expectation in expectations:
        transition_from, transition_to = expectation.expected_transition.split(
            "->"
        )
        observed.append(
            {
                "condition_id": expectation.condition_id,
                "surface_id": expectation.surface_id,
                "transition_from": transition_from,
                "transition_to": transition_to,
                "public_result_id": expectation.public_result_id,
                "public_result": deepcopy(
                    examples[expectation.public_result_id]
                ),
                "status_key": expectation.status_key,
                "status_value": expectation.status_value,
                "provider_call_count": expectation.provider_call_count,
                "domain_side_effect_count": (
                    expectation.domain_side_effect_count
                ),
            }
        )

    def execution(
        evidence: list[dict[str, object]] | None,
    ) -> dict[str, object]:
        properties = (
            [] if evidence is None else [["failure_matrix_evidence", evidence]]
        )
        return {
            "exit_code": 0,
            "items": [node],
            "deselected": [],
            "collection_reports": [],
            "reports": [
                {
                    "nodeid": node,
                    "when": "setup",
                    "outcome": "passed",
                    "wasxfail": "",
                    "detail": "",
                    "user_properties": [],
                },
                {
                    "nodeid": node,
                    "when": "call",
                    "outcome": "passed",
                    "wasxfail": "",
                    "detail": "",
                    "user_properties": deepcopy(properties),
                },
                {
                    "nodeid": node,
                    "when": "teardown",
                    "outcome": "passed",
                    "wasxfail": "",
                    "detail": "",
                    "user_properties": deepcopy(properties),
                },
            ],
            "probe_stdout": "",
            "probe_stderr": "",
        }

    _VERIFIER._verify_execution(
        (node,),
        execution(observed),
        (node,),
        failure_expectations={node: expectations},
    )
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="exactly one stable dynamic",
    ):
        _VERIFIER._verify_execution(
            (node,),
            execution(None),
            (node,),
            failure_expectations={node: expectations},
        )

    def omit_last_evidence(evidence: list[dict[str, object]]) -> None:
        evidence.pop()

    mutations: tuple[
        tuple[str, Callable[[list[dict[str, object]]], None]],
        ...,
    ] = (
        (
            "unowned failure-matrix evidence",
            lambda evidence: evidence[0].__setitem__(
                "surface_id",
                "task-target-flow-direct",
            ),
        ),
        (
            "dynamic failure transition drifted",
            lambda evidence: evidence[0].__setitem__(
                "transition_to",
                "answered",
            ),
        ),
        (
            "dynamic public envelope drifted",
            lambda evidence: cast(
                dict[str, object], evidence[0]["public_result"]
            ).__setitem__("task_state", "failed"),
        ),
        (
            "dynamic failure status drifted",
            lambda evidence: evidence[0].__setitem__(
                "status_value",
                "failed",
            ),
        ),
        (
            "dynamic failure counts drifted",
            lambda evidence: evidence[0].__setitem__(
                "provider_call_count",
                99,
            ),
        ),
        (
            "duplicate failure-matrix evidence",
            lambda evidence: evidence.append(deepcopy(evidence[0])),
        ),
        (
            "omitted active failure-matrix evidence",
            omit_last_evidence,
        ),
    )
    for message, mutate in mutations:
        drifted = deepcopy(observed)
        mutate(drifted)
        with pytest.raises(
            _VERIFIER.AcceptanceVerificationError,
            match=message,
        ):
            _VERIFIER._verify_execution(
                (node,),
                execution(drifted),
                (node,),
                failure_expectations={node: expectations},
            )


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
        (
            "def test_case():\n    try:\n"
            "        assert len('x') == 2\n"
            "    finally:\n        return\n"
        ),
        (
            "def test_case():\n    for _ in (1,):\n        try:\n"
            "            assert len('x') == 2\n"
            "        finally:\n            break\n"
        ),
        (
            "def test_case():\n    for _ in (1,):\n        try:\n"
            "            assert len('x') == 2\n"
            "        finally:\n            continue\n"
        ),
        (
            "def test_case():\n    try:\n"
            "        assert len('x') == 2\n"
            "    except AssertionError:\n        pass\n"
        ),
        (
            "def test_case():\n    try:\n"
            "        assert len('x') == 2\n"
            "    except Exception:\n        return\n"
        ),
        (
            "def test_case():\n    try:\n"
            "        assert len('x') == 2\n"
            "    except BaseException:\n        pass\n"
        ),
        (
            "def test_case():\n    try:\n        pass\n"
            "    except RuntimeError:\n        raise\n"
            "    else:\n        assert len('x') == 2\n"
            "    finally:\n        return\n"
        ),
        (
            "def test_case():\n    def checked():\n        try:\n"
            "            assert len('x') == 2\n"
            "        finally:\n            return\n"
            "    checked()\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    try:\n        checked()\n"
            "    finally:\n        return\n"
        ),
        (
            "def test_case():\n    def checked():\n        try:\n"
            "            assert len('x') == 2\n"
            "        except AssertionError:\n            pass\n"
            "    checked()\n"
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
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 1\n    value = object()\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 1\n    False and checked()\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 1\n    1 < 0 < checked()\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 1\n    return\n    checked()\n"
        ),
        (
            "def checked():\n    assert len('x') == 1\n\n"
            "def test_case():\n    callback = lambda: checked()\n"
            "    callback()\n"
        ),
        (
            "import unittest\n\ndef test_case():\n"
            "    case = unittest.TestCase()\n"
            "    callback = lambda: case.assertEqual(len('x'), 1)\n"
            "    callback()\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 1\n"
            "    alias = checked\n    alias()\n"
        ),
        (
            "def invoke(callback):\n    callback()\n\n"
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 1\n    invoke(checked)\n"
        ),
        (
            "def checked():\n    assert len('x') == 1\n\n"
            "def test_case():\n    alias = checked\n"
            "    def rebind():\n        nonlocal alias\n"
            "        alias = checked\n"
            "    rebind()\n    alias()\n"
        ),
        (
            "def checked():\n    assert len('x') == 1\n\n"
            "def test_case():\n    globals()['alias'] = checked\n"
            "    globals()['alias']()\n"
        ),
        (
            "async def test_case():\n    async def checked():\n"
            "        assert len('x') == 1\n    await checked()\n"
        ),
        (
            "def test_case():\n    def unchecked():\n"
            "        return len('x')\n    unchecked()\n"
        ),
        (
            "def enabled():\n    return True\n\ndef test_case():\n"
            "    def checked():\n        assert len('x') == 1\n"
            "    if enabled():\n        checked()\n"
        ),
        (
            "def test_case():\n    def first():\n        second()\n"
            "    def second():\n        first()\n    first()\n"
        ),
        "from helpers import checked\n\ndef test_case():\n    checked()\n",
        (
            "def test_case():\n    values = ('x',)\n    return\n"
            "    for value in values:\n        assert len(value) == 1\n"
        ),
        (
            "def test_case():\n    values = ('x',)\n"
            "    for value in values:\n        if value:\n"
            "            break\n        assert len(value) == 1\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 1\n"
            "    [checked() for _ in ()]\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    with suppress(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress as guard\n\ndef test_case():\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "def test_case():\n"
            "    from contextlib import suppress as guard\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "import contextlib as guards\n\ndef test_case():\n"
            "    with guards.suppress(Exception):\n"
            "        with guards.suppress(AssertionError):\n"
            "            assert len('x') == 2\n"
        ),
        (
            "import contextlib\nimport pytest\n\ndef test_case():\n"
            "    with contextlib.suppress(Exception):\n"
            "        with pytest.raises(ValueError):\n"
            "            int('x')\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    def checked():\n"
            "        assert len('x') == 2\n"
            "    with suppress(AssertionError):\n"
            "        checked()\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    guard = suppress\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "import contextlib\n\ndef test_case():\n"
            "    guards = contextlib\n"
            "    with guards.suppress(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "import contextlib\n\n"
            "guard = contextlib.suppress\n\ndef test_case():\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    guard: object = suppress\n"
            "    failure = AssertionError\n"
            "    failures = (ValueError, failure)\n"
            "    with guard(*failures):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    with suppress(BaseException):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    *rest, guard = (suppress,)\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    guard, *rest = (suppress,)\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    values = (suppress,)\n"
            "    *rest, guard = values\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n"
            "    guard = suppress\n"
            "    if False:\n"
            "        guard = nullcontext\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "guard = suppress\n"
            "if False:\n"
            "    guard = nullcontext\n\n"
            "def test_case():\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n"
            "    guard = suppress\n"
            "    for guard in ():\n"
            "        pass\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n"
            "    guard = suppress\n"
            "    False and (guard := nullcontext)\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import *\n\ndef test_case():\n"
            "    with suppress(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def enabled():\n"
            "    return True\n\ndef test_case():\n"
            "    if enabled():\n"
            "        guard = suppress\n"
            "    else:\n"
            "        guard = nullcontext\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n"
            "    guard = suppress\n"
            "    while False:\n"
            "        guard = nullcontext\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n"
            "    guard = suppress\n"
            "    try:\n"
            "        pass\n"
            "    except RuntimeError:\n"
            "        guard = nullcontext\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef enabled():\n"
            "    return False\n\ndef test_case():\n"
            "    guard = suppress\n"
            "    if enabled():\n"
            "        del guard\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    for guard in (suppress,):\n"
            "        pass\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n"
            "    with nullcontext(suppress) as guard:\n"
            "        with guard(AssertionError):\n"
            "            assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    errors = (error for error in (AssertionError,))\n"
            "    with suppress(*errors):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    base = (AssertionError,)\n"
            "    errors = (error for error in base)\n"
            "    with suppress(*errors):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    errors = (ValueError,) + (AssertionError,)\n"
            "    with suppress(*errors):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def enabled():\n"
            "    return True\n\ndef test_case():\n"
            "    guard = suppress if enabled() else nullcontext\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n"
            "    guard = (nullcontext, suppress)[1]\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def choose(*values):\n"
            "    return values[0]\n\ndef test_case():\n"
            "    guard = choose(nullcontext, suppress)\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n"
            "    guard = suppress\n"
            "    [guard := nullcontext for _ in ()]\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n"
            "    guard = suppress\n"
            "    match object():\n"
            "        case None:\n"
            "            guard = nullcontext\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    with (guard := suppress)(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    match suppress:\n        case guard:\n            pass\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n    guard = suppress\n"
            "    1 > 2 > (guard := nullcontext)\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 1\n"
            "    checked = lambda: None\n    checked()\n"
        ),
        (
            "async def test_case():\n    async def checked():\n"
            "        assert len('x') == 1\n    checked()\n"
        ),
        (
            "from .contextlib import nullcontext\n\n"
            "def test_case():\n    with nullcontext():\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from .asyncio import run\n\n"
            "def test_case():\n    async def checked():\n"
            "        assert len('x') == 2\n    run(checked())\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    checked.__code__ = (lambda: None).__code__\n"
            "    checked()\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    alias = checked\n"
            "    alias.__code__ = (lambda: None).__code__\n"
            "    checked()\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    first = checked\n    second = first\n"
            "    second.__code__ = (lambda: None).__code__\n"
            "    first()\n"
        ),
        (
            "def choose(value):\n    return value\n\n"
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    alias = choose(checked)\n"
            "    alias.__code__ = (lambda: None).__code__\n"
            "    checked()\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    first = second\n    second = first\n"
            "    first.__code__ = (lambda: None).__code__\n"
            "    checked()\n"
        ),
        (
            "import asyncio\n\ndef test_case():\n"
            "    async def checked():\n"
            "        assert len('x') == 2\n"
            "    asyncio.run.__code__ = (lambda _value: None).__code__\n"
            "    asyncio.run(checked())\n"
        ),
        (
            "import asyncio\n\ndef test_case():\n"
            "    async def checked():\n"
            "        assert len('x') == 2\n"
            "    runner = asyncio.run\n"
            "    runner.__code__ = (lambda _value: None).__code__\n"
            "    asyncio.run(checked())\n"
        ),
        (
            "from asyncio import run\n\ndef test_case():\n"
            "    async def checked():\n"
            "        assert len('x') == 2\n"
            "    run.__code__ = (lambda _value: None).__code__\n"
            "    run(checked())\n"
        ),
        (
            "from asyncio import run\n\ndef test_case():\n"
            "    async def checked():\n"
            "        assert len('x') == 2\n"
            "    first = run\n    second = first\n"
            "    second.__code__ = (lambda _value: None).__code__\n"
            "    run(checked())\n"
        ),
        (
            "import asyncio\n\ndef test_case():\n"
            "    def checked():\n        assert len('x') == 2\n"
            "    asyncio.checked = checked\n"
            "    asyncio.checked.__code__ = (lambda: None).__code__\n"
            "    checked()\n"
        ),
        (
            "import asyncio\n\ndef test_case():\n"
            "    async def checked():\n"
            "        assert len('x') == 2\n"
            "    asyncio.runner = asyncio.run\n"
            "    asyncio.runner.__code__ = "
            "(lambda _value: None).__code__\n"
            "    asyncio.run(checked())\n"
        ),
        (
            "import asyncio\n\ndef test_case():\n"
            "    def checked():\n        assert len('x') == 2\n"
            "    asyncio.first = checked\n"
            "    asyncio.second = asyncio.first\n"
            "    asyncio.second.__defaults__ = ()\n"
            "    checked()\n"
        ),
        (
            "import asyncio\n\ndef enabled():\n    return True\n\n"
            "def test_case():\n"
            "    def checked():\n        assert len('x') == 2\n"
            "    def unrelated():\n        return None\n"
            "    asyncio.target = (checked if enabled() else unrelated)\n"
            "    asyncio.target.__code__ = (lambda: None).__code__\n"
            "    checked()\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    mutate = setattr\n"
            "    mutate(checked, '__code__', (lambda: None).__code__)\n"
            "    checked()\n"
        ),
        (
            "import builtins\n\ndef test_case():\n"
            "    def checked():\n        assert len('x') == 2\n"
            "    runtime = builtins\n    mutate = runtime.setattr\n"
            "    mutate(checked, '__code__', (lambda: None).__code__)\n"
            "    checked()\n"
        ),
        (
            "import asyncio\n\ndef test_case():\n"
            "    def checked():\n        assert len('x') == 2\n"
            "    asyncio.mutate = setattr\n"
            "    mutate = asyncio.mutate\n"
            "    mutate(checked, '__code__', (lambda: None).__code__)\n"
            "    checked()\n"
        ),
        (
            "from builtins import setattr as replace\n\ndef test_case():\n"
            "    def checked():\n        assert len('x') == 2\n"
            "    first = replace\n    mutate = first\n"
            "    mutate(checked, '__defaults__', ())\n"
            "    checked()\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    mutate = checked.__setattr__\n"
            "    mutate('__code__', (lambda: None).__code__)\n"
            "    checked()\n"
        ),
        (
            "import asyncio\n\ndef test_case():\n"
            "    async def checked():\n"
            "        assert len('x') == 2\n"
            "    mutate = asyncio.run.__setattr__\n"
            "    mutate('__code__', (lambda _value: None).__code__)\n"
            "    asyncio.run(checked())\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    mutate = object.__setattr__\n"
            "    mutate(checked, '__code__', (lambda: None).__code__)\n"
            "    checked()\n"
        ),
        (
            "from builtins import object as root\n\ndef test_case():\n"
            "    def checked():\n        assert len('x') == 2\n"
            "    first = root.__setattr__\n    mutate = first\n"
            "    mutate(checked, '__code__', (lambda: None).__code__)\n"
            "    checked()\n"
        ),
        (
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    remove = delattr\n"
            "    remove(checked, '__defaults__')\n"
            "    checked()\n"
        ),
        (
            "def mutate(*_args):\n    return None\n\n"
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    mutate(checked, 'metadata')\n"
            "    checked()\n"
        ),
        (
            "def mutate(*_args):\n    return None\n\n"
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    mutate('__kwdefaults__')\n"
            "    checked()\n"
        ),
        (
            "def mutate(*_args):\n    return None\n\n"
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    attribute = '__code__'\n    mutate(attribute)\n"
            "    checked()\n"
        ),
        (
            "def mutate(**_values):\n    return None\n\n"
            "def test_case():\n    def checked():\n"
            "        assert len('x') == 2\n"
            "    mutate(__code__=(lambda: None).__code__)\n"
            "    checked()\n"
        ),
        (
            "def test_case():\n    values = ()\n"
            "    for value in values:\n        assert len(value) == 1\n"
            "    values = ('x',)\n"
        ),
        (
            "def test_case():\n    values = ('x',)\n    values = ()\n"
            "    for value in values:\n        assert len(value) == 1\n"
        ),
        (
            "def test_case():\n    values = ['x']\n    values.clear()\n"
            "    for value in values:\n        assert len(value) == 1\n"
        ),
        (
            "import pytest\nfrom contextlib import suppress\n\n"
            "def test_case():\n    pytest.raises = suppress\n"
            "    with pytest.raises(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "import pytest\nfrom contextlib import suppress\n\n"
            "def test_case():\n"
            "    setattr(pytest, 'raises', suppress)\n"
            "    with pytest.raises(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "import pytest\nfrom contextlib import suppress\n\n"
            "def test_case():\n    name = 'raises'\n"
            "    pytest.__dict__[name] = suppress\n"
            "    with pytest.raises(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "import contextlib\n\ndef test_case():\n"
            "    contextlib.__dict__.update("
            "{'nullcontext': contextlib.suppress})\n"
            "    with contextlib.nullcontext(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "import asyncio\n\ndef test_case():\n"
            "    async def checked():\n"
            "        assert len('x') == 2\n"
            "    name = 'run'\n"
            "    setattr(asyncio, name, lambda _coroutine: None)\n"
            "    asyncio.run(checked())\n"
        ),
        (
            "import asyncio\n"
            "asyncio.__dict__['run'] = lambda _coroutine: None\n"
            "from asyncio import run\n\ndef test_case():\n"
            "    async def checked():\n"
            "        assert len('x') == 2\n"
            "    run(checked())\n"
        ),
        (
            "from contextlib import suppress\n\nclass Guards:\n"
            "    guard = staticmethod(suppress)\n\ndef test_case():\n"
            "    with Guards.guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\nclass Guard(suppress):\n"
            "    pass\n\ndef test_case():\n"
            "    with Guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\nclass Guards:\n"
            "    @property\n    def guard(self):\n        return suppress\n\n"
            "def test_case():\n"
            "    with Guards().guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n"
            "from functools import partial\n\n"
            "def test_case():\n"
            "    guard = partial(suppress, AssertionError)\n"
            "    with guard():\n        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    guard = lambda error: suppress(error)\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef factory(guard=suppress):\n"
            "    return guard(AssertionError)\n\ndef test_case():\n"
            "    with factory():\n        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n    guard = suppress or nullcontext\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    guard = {'active': suppress}['active']\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "import contextlib\n\ndef test_case():\n"
            "    guard = getattr(contextlib, 'suppress')\n"
            "    with guard(AssertionError):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    context = suppress(AssertionError)\n    with context:\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def enabled():\n    return True\n\ndef test_case():\n"
            "    context = (suppress(AssertionError) if enabled() "
            "else nullcontext())\n    with context:\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import nullcontext, suppress\n\n"
            "def test_case():\n"
            "    context = (nullcontext(), suppress(AssertionError))[1]\n"
            "    with context:\n        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef factory():\n"
            "    return suppress(AssertionError)\n\ndef test_case():\n"
            "    with factory():\n        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef choose(value):\n"
            "    return value\n\ndef test_case():\n"
            "    failure = choose(AssertionError)\n"
            "    with suppress(failure):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "from contextlib import suppress\n\ndef test_case():\n"
            "    failures = [ValueError]\n"
            "    failures.append(AssertionError)\n"
            "    with suppress(*failures):\n"
            "        assert len('x') == 2\n"
        ),
        (
            "class Guard:\n    def __enter__(self):\n        return self\n"
            "    def __exit__(self, *_args):\n        return True\n\n"
            "def test_case():\n    with Guard():\n"
            "        assert len('x') == 2\n"
        ),
    )
    for source in invalid_sources:
        path.write_text(source, encoding="utf-8")
        with pytest.raises(_VERIFIER.AcceptanceVerificationError):
            _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "import contextlib\n\nclass TestCase:\n"
        "    guards = contextlib\n"
        "    guard = guards.suppress\n\n"
        "    def test_case(self):\n"
        "        with self.guard(AssertionError):\n"
        "            assert len('x') == 2\n",
        encoding="utf-8",
    )
    with pytest.raises(_VERIFIER.AcceptanceVerificationError):
        _VERIFIER._validate_test_implementation(
            "tests/case_test.py::TestCase::test_case",
            tmp_path,
        )
    for source in (
        (
            "class TestCase:\n"
            "    def checked(self):\n"
            "        assert len('x') == 1\n"
            "    def test_case(self):\n"
            "        self.checked()\n"
        ),
        (
            "import unittest\nfrom contextlib import suppress\n\n"
            "class TestCase(unittest.TestCase):\n"
            "    assertRaises = suppress\n\n"
            "    def test_case(self):\n"
            "        with self.assertRaises(AssertionError):\n"
            "            assert len('x') == 2\n"
        ),
        (
            "import unittest\nfrom contextlib import suppress\n\n"
            "class TestCase(unittest.TestCase):\n"
            "    def test_case(self):\n"
            "        self.assertRaises = suppress\n"
            "        with self.assertRaises(AssertionError):\n"
            "            assert len('x') == 2\n"
        ),
        (
            "import unittest\nfrom contextlib import suppress\n\n"
            "class Override:\n    assertRaises = suppress\n\n"
            "class TestCase(Override, unittest.TestCase):\n"
            "    def test_case(self):\n"
            "        with self.assertRaises(AssertionError):\n"
            "            assert len('x') == 2\n"
        ),
        (
            "import unittest\nfrom contextlib import suppress\n\n"
            "class Override:\n"
            "    def subTest(self, **_kwargs):\n"
            "        return suppress(AssertionError)\n\n"
            "class TestCase(Override, unittest.TestCase):\n"
            "    def test_case(self):\n"
            "        with self.subTest():\n"
            "            assert len('x') == 2\n"
        ),
        (
            "import unittest\nfrom contextlib import suppress\n\n"
            "class TestCase(unittest.TestCase):\n"
            "    def test_case(self):\n"
            "        self.__dict__['assertRaises'] = suppress\n"
            "        with self.assertRaises(AssertionError):\n"
            "            assert len('x') == 2\n"
        ),
        (
            "import unittest\nfrom contextlib import suppress\n\n"
            "class TestCase(unittest.TestCase):\n"
            "    def test_case(self):\n"
            "        self.__dict__.update("
            "{'subTest': lambda **_kwargs: suppress(AssertionError)})\n"
            "        with self.subTest():\n"
            "            assert len('x') == 2\n"
        ),
        (
            "class TestCase:\n"
            "    def checked(self):\n"
            "        assert len('x') == 1\n"
            "    def test_case(self):\n"
            "        name = 'checked'\n"
            "        setattr(self, name, lambda: None)\n"
            "        self.checked()\n"
        ),
        (
            "class TestCase:\n"
            "    def checked(self):\n"
            "        assert len('x') == 1\n"
            "    def test_case(self):\n"
            "        self.__dict__['checked'] = lambda: None\n"
            "        self.checked()\n"
        ),
        (
            "import unittest\nfrom contextlib import suppress\n\n"
            "class Override:\n    assertRaises = suppress\n\n"
            "unittest.__dict__['TestCase'] = Override\n"
            "from unittest import TestCase as Base\n\n"
            "class TestCase(Base):\n"
            "    def test_case(self):\n"
            "        with self.assertRaises(AssertionError):\n"
            "            assert len('x') == 2\n"
        ),
    ):
        path.write_text(source, encoding="utf-8")
        with pytest.raises(_VERIFIER.AcceptanceVerificationError):
            _VERIFIER._validate_test_implementation(
                "tests/case_test.py::TestCase::test_case",
                tmp_path,
            )
    path.write_text(
        "def test_case():\n    value = len('contract')\n    assert value"
        " == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def test_case():\n    def checked():\n"
        "        assert len('contract') == 8\n    checked()\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def checked():\n    assert len('contract') == 8\n\n"
        "def test_case():\n    checked()\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from asyncio import run as runner\n\ndef test_case():\n"
        "    async def checked():\n"
        "        assert len('contract') == 8\n    runner(checked())\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def test_case():\n    def exercise():\n"
        "        assert len('contract') == 8\n"
        "        return 'verified'\n"
        "    result = exercise()\n"
        "    assert result == 'verified'\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def test_case():\n    def exercise():\n"
        "        assert len('contract') == 8\n"
        "        return 'verified'\n"
        "    alias = exercise\n"
        "    result = alias()\n"
        "    assert result == 'verified'\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def test_case():\n    def exercise():\n"
        "        assert len('contract') == 8\n"
        "        return 'verified'\n"
        "    first = exercise\n    second = first\n"
        "    result = second()\n"
        "    assert result == 'verified'\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def test_case():\n    def exercise():\n"
        "        assert len('contract') == 8\n"
        "        return 'verified'\n"
        "    alias = exercise\n"
        "    result = alias()\n"
        "    alias.__code__ = (lambda: None).__code__\n"
        "    assert result == 'verified'\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def test_case():\n    cleaned = []\n    try:\n"
        "        assert len('contract') == 8\n"
        "    finally:\n        cleaned.append(True)\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def test_case():\n    cleaned = []\n    def checked():\n"
        "        try:\n            assert len('contract') == 8\n"
        "        finally:\n            cleaned.append(True)\n"
        "    checked()\n"
        "    assert cleaned == [True]\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def test_case():\n    cleaned = []\n    try:\n"
        "        assert len('contract') == 8\n"
        "    except AssertionError:\n        raise\n"
        "    finally:\n        cleaned.append(True)\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def test_case():\n    values = ('contract',)\n"
        "    for value in values:\n        assert len(value) == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "def test_case():\n    values = ['contract']\n"
        "    values.append('second')\n"
        "    for value in values:\n        assert len(value) > 0\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from contextlib import nullcontext\n\ndef test_case():\n"
        "    with nullcontext():\n"
        "        assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from contextlib import nullcontext, suppress\n\n"
        "guard = suppress\n"
        "guard = nullcontext\n\ndef test_case():\n"
        "    with guard():\n"
        "        assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from contextlib import nullcontext, suppress\n\n"
        "def test_case():\n"
        "    guard = nullcontext\n"
        "    if False:\n"
        "        guard = suppress\n"
        "    with guard():\n"
        "        assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from contextlib import nullcontext, suppress\n\n"
        "def test_case():\n"
        "    guard = (suppress, nullcontext)[1]\n"
        "    with guard():\n"
        "        assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from contextlib import suppress\n\ndef test_case():\n"
        "    with suppress(ValueError):\n"
        "        assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from contextlib import nullcontext, suppress\n\n"
        "def test_case():\n"
        "    guard = suppress\n"
        "    guard = nullcontext\n"
        "    with guard():\n"
        "        assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from contextlib import suppress\n\ndef test_case():\n"
        "    failure = ValueError\n"
        "    failures = (failure,)\n"
        "    with suppress(*failures):\n"
        "        assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from contextlib import suppress\n\ndef test_case():\n"
        "    with suppress(AssertionError):\n"
        "        assert len('contract') == 7\n"
        "    assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "import pytest\n\ndef test_case():\n"
        "    with pytest.raises(ValueError):\n        int('x')\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from pytest import raises as check\n\ndef test_case():\n"
        "    with check(ValueError):\n        int('x')\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "import unittest\n\nclass TestCase(unittest.TestCase):\n"
        "    def test_case(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            int('x')\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(
        "tests/case_test.py::TestCase::test_case",
        tmp_path,
    )
    for source in (
        (
            "import unittest\n\n"
            "class TestCase(unittest.IsolatedAsyncioTestCase):\n"
            "    async def test_case(self):\n"
            "        with self.subTest(label='direct'):\n"
            "            assert len('contract') == 8\n"
        ),
        (
            "import unittest\nfrom contextlib import suppress\n\n"
            "class Override:\n    assertRaises = suppress\n\n"
            "class TestCase(unittest.TestCase, Override):\n"
            "    def test_case(self):\n"
            "        with self.assertRaises(ValueError):\n"
            "            int('x')\n"
        ),
        (
            "import unittest\nfrom contextlib import suppress\n\n"
            "class Override:\n"
            "    def subTest(self, **_kwargs):\n"
            "        return suppress(AssertionError)\n\n"
            "class TestCase(unittest.TestCase, Override):\n"
            "    def test_case(self):\n"
            "        self.__dict__['unrelated'] = suppress\n"
            "        with self.subTest(label='safe-order'):\n"
            "            assert len('contract') == 8\n"
        ),
    ):
        path.write_text(source, encoding="utf-8")
        _VERIFIER._validate_test_implementation(
            "tests/case_test.py::TestCase::test_case",
            tmp_path,
        )
    path.write_text(
        "from tempfile import TemporaryDirectory\n"
        "from unittest.mock import patch\n\ndef test_case():\n"
        "    with TemporaryDirectory() as directory, patch.dict({}, {}):\n"
        "        with open(f'{directory}/value', 'w') as stream:\n"
        "            stream.write('value')\n"
        "        assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from contextlib import nullcontext\n\ndef test_case():\n"
        "    with (guard := nullcontext)():\n"
        "        assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from contextlib import nullcontext\n\ndef test_case():\n"
        "    match nullcontext:\n        case guard:\n            pass\n"
        "    with guard():\n        assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from contextlib import nullcontext, suppress\n\n"
        "def test_case():\n    guard = suppress\n"
        "    None is (guard := nullcontext) is None\n"
        "    with guard():\n        assert len('contract') == 8\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "async def test_case():\n    async def exercise():\n"
        "        assert len('contract') == 8\n"
        "        return 'verified'\n"
        "    result = await exercise()\n"
        "    assert result == 'verified'\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "from asyncio import run\n\ndef test_case():\n"
        "    async def exercise():\n"
        "        assert len('contract') == 8\n"
        "        return 'verified'\n"
        "    result = run(exercise())\n"
        "    assert result == 'verified'\n",
        encoding="utf-8",
    )
    _VERIFIER._validate_test_implementation(node, tmp_path)
    path.write_text(
        "class Guard:\n    def __enter__(self):\n        return self\n"
        "    def __exit__(self, *_args):\n        return True\n\n"
        "def test_case():\n    with Guard():\n        pass\n"
        "    assert len('contract') == 8\n",
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
        (),
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
            (),
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
            (),
            preserved_untracked,
        )
    untracked_paths.remove("undeclared.txt")
    tracked_paths.add("src/canonical.py")
    declared_with_source = (*declared_paths, "src/canonical.py")
    with pytest.raises(
        _VERIFIER.AcceptanceVerificationError,
        match="live production source changes differ",
    ):
        _VERIFIER._validate_live_boundary(
            tmp_path,
            declared_with_source,
            (),
            preserved_untracked,
        )
    _VERIFIER._validate_live_boundary(
        tmp_path,
        declared_with_source,
        ("src/canonical.py",),
        preserved_untracked,
    )
