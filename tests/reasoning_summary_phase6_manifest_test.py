"""Test the exact Phase 6 reasoning-summary acceptance inventory."""

from asyncio import run
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from sys import modules
from types import ModuleType
from typing import Any, cast

from reasoning_summary_script_loader import load_reasoning_summary_script

_ACCEPTANCE_SCRIPT = load_reasoning_summary_script(
    "verify_reasoning_summary_acceptance"
)
_PHASE6_DIMENSION = "phase 6 CLI visibility themes and recording"
_PHASE6_NODE_IDS = (
    (
        "tests/reasoning_summary_phase6_manifest_test.py::"
        "test_phase6_acceptance_manifest_pins_exact_catalog"
    ),
    (
        "tests/cli/display_test.py::CliStreamDisplayConfigTestCase::"
        "test_noninteractive_reasoning_uses_stderr_diagnostics"
    ),
    (
        "tests/cli/display_snapshot_test.py::"
        "DisplaySnapshotBuilderTestCase::"
        "test_visible_reasoning_retains_structured_identity_and_completion"
    ),
    (
        "tests/cli/display_snapshot_test.py::"
        "DisplaySnapshotBuilderTestCase::"
        "test_hidden_reasoning_has_no_text_owner_or_retained_content"
    ),
    (
        "tests/reasoning_summary_phase6_manifest_test.py::"
        "test_phase6_separator_limit_matrix_acceptance"
    ),
    (
        "tests/cli/display_reducer_test.py::DisplayReducerTestCase::"
        "test_hidden_reasoning_updates_metrics_without_retaining_or_refreshing"
    ),
    (
        "tests/cli/stream_presenter_test.py::"
        "StreamPresenterContractTestCase::"
        "test_reasoning_blocks_group_contiguous_representations"
    ),
    (
        "tests/cli/theme_basic_test.py::BasicStreamPresenterTestCase::"
        "test_reasoning_live_frame_is_opt_in_bounded_and_subtle"
    ),
    (
        "tests/cli/theme_basic_test.py::BasicStreamPresenterTestCase::"
        "test_stderr_reasoning_emits_only_unseen_suffixes_before_tool"
    ),
    (
        "tests/cli/theme_fancy_test.py::FancyStreamPresenterTestCase::"
        "test_reasoning_gate_prevents_hidden_panel_construction"
    ),
    (
        "tests/cli/theme_fancy_test.py::FancyStreamPresenterTestCase::"
        "test_reasoning_panels_label_mixed_representations"
    ),
    (
        "tests/cli/stream_coordinator_test.py::CliStreamCoordinatorTestCase::"
        "test_live_roles_share_one_owner"
    ),
    (
        "tests/cli/stream_coordinator_test.py::CliStreamCoordinatorTestCase::"
        "test_confirm_tool_call_records_once_after_prompt_resume"
    ),
    (
        "tests/cli/stream_coordinator_test.py::CliStreamCoordinatorTestCase::"
        "test_stderr_reasoning_append_preserves_repeated_suffixes"
    ),
    (
        "tests/reasoning_summary_phase6_manifest_test.py::"
        "test_phase6_pairwise_matrix_oracle_acceptance"
    ),
    (
        "tests/reasoning_summary_phase6_manifest_test.py::"
        "test_phase6_reducer_presenter_matrix_acceptance"
    ),
    (
        "tests/reasoning_summary_phase6_manifest_test.py::"
        "test_phase6_public_boundary_matrix_acceptance"
    ),
)


def _load_test_module(relative_path: str) -> ModuleType:
    assert isinstance(relative_path, str) and relative_path
    path = Path(__file__).resolve().parent / relative_path
    module_name = f"_phase6_acceptance_{path.stem}"
    spec = spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        modules.pop(module_name, None)
    return module


def _run_test_case_method(
    relative_path: str,
    class_name: str,
    method_name: str,
) -> None:
    module = _load_test_module(relative_path)
    test_class = getattr(module, class_name)
    test_case = test_class(methodName=method_name)
    getattr(test_case, method_name)()


async def _run_async_test_case_method(
    relative_path: str,
    class_name: str,
    method_name: str,
) -> None:
    module = _load_test_module(relative_path)
    test_class = getattr(module, class_name)
    test_case = test_class(methodName=method_name)
    await getattr(test_case, method_name)()


def test_phase6_separator_limit_matrix_acceptance() -> None:
    _run_test_case_method(
        "cli/display_snapshot_test.py",
        "DisplaySnapshotBuilderTestCase",
        "test_visible_reasoning_limits_charge_separator_and_unicode_bytes",
    )


def test_phase6_pairwise_matrix_oracle_acceptance() -> None:
    _run_test_case_method(
        "cli/theme_matrix_e2e_test.py",
        "ReasoningDisplayPairwiseMatrixTestCase",
        "test_reasoning_display_matrix_is_exact_and_pairwise_complete",
    )


def test_phase6_reducer_presenter_matrix_acceptance() -> None:
    run(
        _run_async_test_case_method(
            "cli/theme_matrix_e2e_test.py",
            "ReasoningDisplayIntegrationMatrixTestCase",
            "test_reasoning_reducer_presenter_integration_rows",
        )
    )


def test_phase6_public_boundary_matrix_acceptance() -> None:
    run(
        _run_async_test_case_method(
            "cli/theme_matrix_e2e_test.py",
            "ThemeMatrixE2ETestCase",
            "test_reasoning_critical_public_boundary_rows",
        )
    )


def test_phase6_acceptance_manifest_pins_exact_catalog() -> None:
    manifest = cast(
        Any,
        _ACCEPTANCE_SCRIPT.load_manifest(
            _ACCEPTANCE_SCRIPT.default_manifest_path()
        ),
    )

    assert manifest.active_phase >= 6
    assert manifest.dimensions[_PHASE6_DIMENSION] == _PHASE6_NODE_IDS
    assert len(_PHASE6_NODE_IDS) == 17
    assert len(manifest.node_ids) >= 343
    assert len(manifest.node_ids) == len(set(manifest.node_ids))
