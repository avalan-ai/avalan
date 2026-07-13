"""Exercise the self-contained reasoning-summary release gate."""

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps
from pathlib import Path
from sys import modules
from types import ModuleType
from typing import Any, cast

_ROOT = Path(__file__).resolve().parents[2]


def _load_script_loader() -> ModuleType:
    path = _ROOT / "tests" / "reasoning_summary_script_loader.py"
    module_name = "_reasoning_summary_release_gate_script_loader"
    spec = spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SCRIPT_LOADER = _load_script_loader()
_BENCHMARK = _SCRIPT_LOADER.load_reasoning_summary_script(
    "benchmark_reasoning_summary"
)
_PHASE9_DIMENSION = "phase 9 public acceptance and release gates"
_PHASE9_NODE_IDS = (
    (
        "tests/e2e/reasoning_summary_public_e2e_test.py"
        "::test_summary_request_is_prompt_independent"
    ),
    (
        "tests/e2e/reasoning_summary_public_e2e_test.py"
        "::test_request_and_display_controls_are_independent"
    ),
    (
        "tests/e2e/reasoning_summary_release_gate_test.py"
        "::test_full_quality_and_coverage_gate"
    ),
    (
        "tests/server/streaming_acceptance_inventory_test.py"
        "::StreamingAcceptanceInventoryTestCase"
        "::test_streaming_acceptance_inventory_collects_unskipped_tests"
    ),
    (
        "tests/server/streaming_acceptance_inventory_test.py"
        "::StreamingAcceptanceInventoryTestCase"
        "::test_final_gate_inventory_collects_unskipped_tests"
    ),
)
_FINAL_GATE_DIMENSIONS = (
    "stream accumulation/to_str",
    "SDK losslessness",
    "FancyTheme lossless isolation",
    "local backpressure",
    "hosted cleanup",
    "live tool output",
    "parallel tool ordering",
    "cross-protocol projection",
    "channel done boundaries",
    "event stats/history boundedness",
    "reasoning parsing",
    "listener-less memory boundedness",
    "final negative/e2e suites",
    "reasoning summary public activation",
    "reasoning summary multipart and tool continuation",
    "reasoning summary invalid and unsupported",
    "reasoning summary performance heartbeat queues and replay bounds",
    "reasoning summary privacy",
    "reasoning summary Responses MCP and A2A",
)
_FINAL_GATE_CATALOG_SHA256 = (
    "78125cd5555582fd857354cbb56c9c54adf1619208cac5b43496fe2527e90bc6"
)
_REQUIREMENTS_CATALOG_SHA256 = (
    "ffb70eee04a87b05ccf5202b6b55da3324184492a5553dd8ecf79313e7bfa368"
)
_PHASE9_TRACE_TARGETS = (
    (
        "tests/e2e/reasoning_summary_public_e2e_test.py"
        "::test_summary_request_is_prompt_independent"
    ),
    (
        "tests/e2e/reasoning_summary_public_e2e_test.py"
        "::test_request_and_display_controls_are_independent"
    ),
    (
        "tests/e2e/reasoning_summary_release_gate_test.py"
        "::test_full_quality_and_coverage_gate"
    ),
)


def _load_inventory_module() -> ModuleType:
    path = (
        _ROOT / "tests" / "server" / "streaming_acceptance_inventory_test.py"
    )
    module_name = "_reasoning_summary_release_gate_inventory"
    spec = spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _final_gate_catalog(module: ModuleType) -> dict[str, object]:
    inventory = cast(dict[str, Any], module.FINAL_GATE_ACCEPTANCE_HARNESSES)
    return {
        dimension: {
            "synthetic": list(evidence.synthetic),
            "integration": list(evidence.integration),
        }
        for dimension, evidence in inventory.items()
    }


def test_full_quality_and_coverage_gate() -> None:
    """Assert the real hard gate and exact release catalogs."""
    report = _BENCHMARK.run_phase9_benchmark()
    hard_gate = _BENCHMARK.evaluate_phase9_hard_gate(report)
    assert hard_gate.passed
    assert hard_gate.failure_reasons == ()
    assert report["hard_gate"] == {
        "passed": True,
        "failure_reasons": (),
    }
    metrics = cast(dict[str, object], report["phase9_metrics"])
    heartbeat = cast(dict[str, object], metrics["heartbeat"])
    assert heartbeat["delta_count"] == 8192
    assert heartbeat["warmups"] == 3
    assert heartbeat["samples"] == 20
    assert heartbeat["interval_milliseconds"] == 10
    assert cast(float, heartbeat["maximum_drift_milliseconds"]) <= 100
    coalescing = cast(dict[str, object], metrics["responses_coalescing"])
    assert coalescing["source_delta_count"] == 4097
    assert coalescing["maximum_delta_characters"] == 4096
    assert coalescing["source_close_count"] == 1
    queue_pressure = cast(
        dict[str, dict[str, object]], metrics["queue_pressure"]
    )
    assert len(queue_pressure) == 11
    assert all(
        evidence["passed"] is True for evidence in queue_pressure.values()
    )

    manifest_payload = _SCRIPT_LOADER.strict_json_loads(
        (
            _ROOT
            / "tests"
            / "fixtures"
            / "reasoning_summary"
            / "acceptance_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert isinstance(manifest_payload, dict)
    assert manifest_payload["active_phase"] == 9
    dimensions = cast(dict[str, list[str]], manifest_payload["dimensions"])
    assert tuple(dimensions[_PHASE9_DIMENSION]) == _PHASE9_NODE_IDS
    prior_nodes = [
        node_id
        for dimension, node_ids in dimensions.items()
        if dimension != _PHASE9_DIMENSION
        for node_id in node_ids
    ]
    all_nodes = [
        node_id for node_ids in dimensions.values() for node_id in node_ids
    ]
    assert len(prior_nodes) == 360
    assert len(all_nodes) == 365
    assert len(all_nodes) == len(set(all_nodes))

    inventory_module = _load_inventory_module()
    assert (
        tuple(inventory_module.REQUIRED_FINAL_GATE_DIMENSIONS)
        == _FINAL_GATE_DIMENSIONS
    )
    final_gate_catalog = _final_gate_catalog(inventory_module)
    assert tuple(final_gate_catalog) == _FINAL_GATE_DIMENSIONS
    canonical_final_gate = dumps(
        final_gate_catalog,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert sha256(canonical_final_gate).hexdigest() == (
        _FINAL_GATE_CATALOG_SHA256
    )

    traceability = _SCRIPT_LOADER.strict_json_loads(
        (
            _ROOT
            / "tests"
            / "fixtures"
            / "reasoning_summary"
            / "requirements_traceability.json"
        ).read_text(encoding="utf-8")
    )
    assert isinstance(traceability, dict)
    invariant = cast(dict[str, object], traceability["catalog_invariant"])
    assert invariant == {
        "canonicalization": "compact_sorted_keys_utf8_json_v1",
        "normative_count": 54,
        "acceptance_count": 22,
        "sha256": _REQUIREMENTS_CATALOG_SHA256,
    }
    acceptance = cast(
        list[dict[str, object]], traceability["acceptance_criteria"]
    )
    phase9_targets = tuple(
        cast(str, entry["test_target"])
        for entry in acceptance
        if entry["phase"] == 9
    )
    assert phase9_targets == _PHASE9_TRACE_TARGETS
    canonical_requirements = dumps(
        {
            "normative_requirements": traceability["normative_requirements"],
            "acceptance_criteria": acceptance,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert sha256(canonical_requirements).hexdigest() == (
        _REQUIREMENTS_CATALOG_SHA256
    )
