"""Exercise compact structured-input contract evidence."""

from importlib.util import module_from_spec, spec_from_file_location
from json import loads
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


def _fixture(name: str) -> dict[str, Any]:
    """Return one fixture object."""
    value = loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _manifest() -> Any:
    """Return the validated compact acceptance manifest."""
    return _VERIFIER.load_manifest(_FIXTURES / "acceptance_manifest.json")


def test_requirement_catalog_is_complete() -> None:
    """Derive exact requirement coverage from manifest node metadata."""
    manifest = _manifest()
    requirement_ids = _VERIFIER._validate_requirements(
        _FIXTURES / "requirements_traceability.json", manifest
    )
    catalog = _fixture("requirements_traceability.json")

    assert len(requirement_ids) == 131
    assert len(catalog["requirements"]) == 131
    assert requirement_ids == {
        requirement_id
        for node in manifest.nodes
        for requirement_id in node.requirement_ids
    }


def test_acceptance_manifest_lifecycle_is_monotonic() -> None:
    """Derive history and planned/current slices from node metadata."""
    manifest = _manifest()
    history = manifest.activation_history()

    assert manifest.current_phase == 7
    assert len(manifest.nodes) == 944
    assert len(manifest.active_nodes(7)) == 830
    assert len(manifest.planned_nodes()) == 114
    assert tuple(map(len, history)) == (
        23,
        79,
        86,
        99,
        352,
        788,
        814,
        830,
    )
    assert all(
        set(history[phase]).issubset(history[phase + 1])
        for phase in range(manifest.current_phase)
    )
    assert all(
        node.active_from_phase <= manifest.current_phase
        for node in manifest.active_nodes(7)
    )
    assert all(
        node.active_from_phase > manifest.current_phase
        for node in manifest.planned_nodes()
    )
    requirement_ids = {
        requirement_id
        for node in manifest.nodes
        for requirement_id in node.requirement_ids
    }
    for requirement_id in requirement_ids:
        active, remaining = manifest.requirement_slice(requirement_id, 7)
        expected = {
            node.node_id
            for node in manifest.nodes
            if requirement_id in node.requirement_ids
        }
        assert set(active).isdisjoint(remaining)
        assert set(active) | set(remaining) == expected
    assert {node.node_id for node in manifest.current_phase_nodes()} == {
        "tests/cli/agent_interaction_test.py::AgentRunInteractionInjectionTestCase::test_agent_run_passes_opened_runtime_per_call_and_reads_once",
        "tests/cli/agent_interaction_test.py::CliInteractionRuntimeTestCase::test_missing_control_terminal_disables_attached_runtime",
        "tests/cli/agent_interaction_test.py::CliInteractionRuntimeTestCase::test_runtime_owns_channel_and_pauses_only_active_display",
        "tests/cli/display_reducer_test.py::DisplayReducerTestCase::test_input_required_is_terminal_but_not_completed_outcome",
        "tests/cli/interaction_channel_test.py::CliInteractionChannelTestCase::test_cancelled_reader_preserves_bytes_for_next_prompt",
        "tests/cli/interaction_channel_test.py::CliInteractionChannelTestCase::test_terminal_disappearance_returns_eof_without_hanging",
        "tests/cli/interaction_cli_pty_e2e_test.py::test_decline_input_cancel_run_cancel_and_disappearance_are_distinct",
        "tests/cli/interaction_cli_pty_e2e_test.py::test_piped_prompt_and_pty_clarification_complete_one_run",
        "tests/cli/interaction_cli_pty_e2e_test.py::test_real_orchestrator_engine_agent_resumes_same_run",
        "tests/cli/interaction_cli_pty_e2e_test.py::test_real_orchestrator_run_cancel_owns_containing_run_cleanup",
        "tests/cli/interaction_cli_pty_e2e_test.py::test_semantic_text_multiline_and_multiple_other_rows",
        "tests/cli/interaction_renderer_test.py::CliInteractionRendererHelperTestCase::test_source_has_no_stdout_or_sync_event_loop_entrypoint",
        "tests/cli/interaction_renderer_test.py::CliInteractionRendererTestCase::test_bundle_help_feedback_and_safe_control_output",
        "tests/cli/interaction_renderer_test.py::CliInteractionRendererTestCase::test_channel_lifetime_remains_with_its_owner",
        "tests/cli/interaction_renderer_test.py::CliInteractionRendererTestCase::test_controls_work_inside_multiline_and_other_prompts",
        "tests/interaction/interaction_broker_test.py::test_handler_cancel_is_canonical_cancel_not_handler_loss",
    }


def test_failure_matrix_is_complete() -> None:
    """Derive every applicable and non-applicable failure cell."""
    manifest = _manifest()
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

    assert len(matrix.surfaces) == 84
    assert len(matrix.conditions) == 15
    assert len(matrix.rules) == 169
    assert len(matrix.all_cells()) == 1260
    assert len(matrix.applicable_cells()) == 564
    assert len(matrix.all_cells() - matrix.applicable_cells()) == 696
    assert set(matrix.evidence_nodes(6)).issubset(
        {node.node_id for node in manifest.active_nodes(6)}
    )


def test_deterministic_fixtures_are_reproducible() -> None:
    """Require every deterministic test dependency to stay explicit."""
    _VERIFIER._validate_deterministic_fixtures(
        _FIXTURES / "deterministic_fixtures.json"
    )
    fixture = _fixture("deterministic_fixtures.json")

    assert fixture["clock"]["initial"] == 1000
    assert fixture["clock"]["expected"][-1] == 1005
    assert len(fixture["id_factory"]["expected"]) == 13
    assert len(fixture["provider_calls"]) == 2
    assert fixture["barrier"]["parties"] == 3


def test_contract_decisions_are_frozen() -> None:
    """Validate contract digests, public schemas, and examples directly."""
    surfaces, envelopes = _VERIFIER._validate_decisions(
        _FIXTURES / "contract_decisions.json"
    )
    decisions = _fixture("contract_decisions.json")
    error_status = decisions["error_status"]

    assert len(surfaces) == 84
    assert envelopes == set(error_status["public_envelope_catalog"])
    validator = _VERIFIER._draft_validator()
    for envelope_id in sorted(envelopes):
        schema = error_status["public_envelope_catalog"][envelope_id]
        example = error_status["public_envelope_examples"][envelope_id]
        validator.check_schema(schema)
        assert validator(schema).is_valid(example)


def test_no_bc_removal_inventory_is_frozen() -> None:
    """Require explicit no-backward-compatibility removal evidence."""
    _VERIFIER._validate_no_bc(_FIXTURES / "no_bc_removals.json")
    inventory = _fixture("no_bc_removals.json")

    assert len(inventory["removals"]) >= 5
    assert all(item["replacement"] for item in inventory["removals"])
    assert all(item["evidence"] for item in inventory["removals"])


def test_baseline_evidence_is_complete() -> None:
    """Require the authoritative fresh exact gate and fail-closed policy."""
    manifest = _manifest()
    _VERIFIER._validate_evidence(
        _FIXTURES / "baseline_evidence.json", manifest
    )
    evidence = _fixture("baseline_evidence.json")

    assert (
        evidence["authoritative_gate"]["command"]
        == "make test-pgsql-exact no-install INPUT_PHASE=7"
    )
    assert evidence["authoritative_gate"]["fresh_report_required"] is True
    assert evidence["invariants"] == {
        "planned_nodes_are_not_evidence": True,
        "activation_is_derived_from_nodes": True,
        "failure_cells_are_derived_from_rules": True,
        "reject_skip_xfail_deselection": True,
        "reject_exec_compile_coverage_tricks": True,
        "exact_source_coverage": True,
        "fail_closed": True,
    }


def test_capability_remains_dormant() -> None:
    """Keep the production capability absent until atomic activation."""
    decisions = _fixture("contract_decisions.json")

    assert decisions["activation"]["production_default"] == "absent"
    assert (
        "activate only after every enabled consumer"
        in decisions["activation"]["atomic_rule"]
    )
    assert "retaining resolver capacity" in decisions["activation"]["rollback"]


@pytest.mark.parametrize(
    "name",
    (
        "acceptance_manifest.json",
        "baseline_evidence.json",
        "contract_decisions.json",
        "deterministic_fixtures.json",
        "failure_matrix.json",
        "no_bc_removals.json",
        "requirements_traceability.json",
    ),
)
def test_contract_fixture_json_is_deterministic(name: str) -> None:
    """Keep compact fixture formatting stable and newline-terminated."""
    content = (_FIXTURES / name).read_text(encoding="utf-8")

    assert content.endswith("\n")
    assert "\t" not in content
