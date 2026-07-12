"""Test the exact Phase 5 reasoning-summary acceptance inventory."""

from typing import Any, cast

from reasoning_summary_script_loader import load_reasoning_summary_script

_ACCEPTANCE_SCRIPT = load_reasoning_summary_script(
    "verify_reasoning_summary_acceptance"
)
_PHASE5_DIMENSION = "phase 5 SDK accumulation orchestration and non-stream"
_RETENTION_TEST = "tests/model/model_stream_retention_isolation_test.py::"
_RESPONSE_TEST = (
    "tests/model/text_generation_response_reasoning_summary_test.py::"
)
_SUMMARY_TEST = "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
_OPENAI_TEST = "tests/model/nlp/vendor_openai_test.py::"
_ORCHESTRATOR_TEST = (
    "tests/agent/orchestrator_response_test.py::"
    "OrchestratorResponseMethodsTestCase::"
)
_ENGINE_TEST = "tests/agent/engine_agent_test.py::EngineAgentRunTestCase::"
_PHASE5_NODE_IDS = (
    (
        "tests/reasoning_summary_phase5_manifest_test.py::"
        "test_phase5_acceptance_manifest_pins_exact_catalog"
    ),
    f"{_RETENTION_TEST}test_summary_parts_have_readable_boundaries",
    f"{_RETENTION_TEST}test_flat_reasoning_preserves_part_order",
    f"{_RETENTION_TEST}test_structured_reasoning_preserves_representation",
    f"{_RETENTION_TEST}test_summary_obeys_reasoning_retention",
    f"{_RETENTION_TEST}test_multipart_summary_is_ordered_and_readable",
    (
        f"{_RETENTION_TEST}"
        "test_reasoning_merges_only_contiguous_exact_segment_identity"
    ),
    (
        f"{_RETENTION_TEST}"
        "test_reasoning_terminal_finalizes_status_without_boundary_leak"
    ),
    (
        f"{_RETENTION_TEST}"
        "test_reasoning_character_limit_retains_utf8_safe_suffix"
    ),
    f"{_RETENTION_TEST}test_reasoning_utf8_byte_limit_never_splits_character",
    (
        f"{_RETENTION_TEST}"
        "test_reasoning_separator_is_charged_before_following_segment"
    ),
    (
        f"{_RETENTION_TEST}"
        "test_reasoning_history_is_independent_from_canonical_item_retention"
    ),
    (
        f"{_RETENTION_TEST}"
        "test_reasoning_accumulator_instances_isolate_concurrent_sentinels"
    ),
    (
        f"{_RETENTION_TEST}"
        "test_reasoning_flat_view_has_no_second_retained_history"
    ),
    (
        f"{_RETENTION_TEST}"
        "test_reasoning_character_retention_limit_minus_equal_plus_one"
    ),
    (
        f"{_RETENTION_TEST}"
        "test_reasoning_utf8_retention_limit_minus_equal_plus_one"
    ),
    (
        f"{_RETENTION_TEST}"
        "test_reasoning_segment_retention_limit_minus_equal_plus_one"
    ),
    (
        f"{_RETENTION_TEST}"
        "test_reasoning_separator_budget_limit_minus_equal_plus_one"
    ),
    (
        f"{_RETENTION_TEST}"
        "test_canonical_item_retention_limit_minus_equal_plus_one"
    ),
    (
        f"{_RETENTION_TEST}"
        "test_large_unicode_reasoning_owner_work_and_storage_are_bounded"
    ),
    f"{_RESPONSE_TEST}test_summary_never_contaminates_answer",
    f"{_RESPONSE_TEST}test_summary_isolated_from_answer_tools_and_memory",
    (
        f"{_RESPONSE_TEST}"
        "test_reasoning_tokens_and_summary_characters_are_separate"
    ),
    (
        f"{_RESPONSE_TEST}"
        "test_rich_non_stream_result_uses_canonical_normalization"
    ),
    (
        f"{_RESPONSE_TEST}"
        "test_non_stream_terminal_status_preserves_partial_answer"
    ),
    f"{_SUMMARY_TEST}test_non_stream_openai_preserves_summary_tool_and_answer",
    f"{_SUMMARY_TEST}test_non_stream_openai_unknown_output_item_is_ignored",
    (
        f"{_SUMMARY_TEST}"
        "test_non_stream_openai_replay_owner_follows_valid_tool_cycle"
    ),
    f"{_SUMMARY_TEST}test_non_stream_openai_no_tool_releases_replay_owner",
    f"{_SUMMARY_TEST}test_non_stream_openai_mapping_failure_rolls_back_owner",
    (
        f"{_SUMMARY_TEST}"
        "test_openai_stream_and_non_stream_semantics_are_equivalent"
    ),
    f"{_SUMMARY_TEST}test_non_stream_openai_incomplete_tool_is_not_executable",
    (
        f"{_SUMMARY_TEST}"
        "test_non_stream_openai_client_close_owns_temporary_stream"
    ),
    f"{_OPENAI_TEST}NonStreamingResponseTestCase::test_response_single_stream",
    (
        f"{_OPENAI_TEST}"
        "TemplateMessagesFormatTestCase::test_non_stream_tool_call_output"
    ),
    (
        f"{_ORCHESTRATOR_TEST}"
        "test_non_stream_structured_response_uses_canonical_path"
    ),
    f"{_ENGINE_TEST}test_sync_messages_keeps_rich_reasoning_out_of_memory",
)


def test_phase5_acceptance_manifest_pins_exact_catalog() -> None:
    manifest = cast(
        Any,
        _ACCEPTANCE_SCRIPT.load_manifest(
            _ACCEPTANCE_SCRIPT.default_manifest_path()
        ),
    )

    assert manifest.active_phase >= 5
    assert manifest.dimensions[_PHASE5_DIMENSION] == _PHASE5_NODE_IDS
    assert len(_PHASE5_NODE_IDS) == 37
    assert len(manifest.node_ids) == len(set(manifest.node_ids))
