"""Test the exact Phase 3 reasoning-summary acceptance inventory."""

from typing import Any, cast

from reasoning_summary_script_loader import load_reasoning_summary_script

_ACCEPTANCE_SCRIPT = load_reasoning_summary_script(
    "verify_reasoning_summary_acceptance"
)
_PHASE3_DIMENSION = "phase 3 OpenAI request construction, retry, and replay"
_PHASE3_NODE_IDS = (
    (
        "tests/reasoning_summary_phase3_manifest_test.py::"
        "test_phase3_acceptance_manifest_pins_exact_catalog"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_reasoning_request_shapes_are_exact_and_omission_safe"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_summary_only_request_is_forwarded"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_reasoning_request_streaming_and_non_streaming_are_identical"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_summary_request_tool_matrix_keeps_include_deduplicated"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_effort_none_max_and_disabled_request_semantics"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_invalid_summary_fails_before_responses_create"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_declared_unsupported_summary_fails_before_provider_call"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_upstream_summary_rejection_is_actionable_and_non_retryable"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_rejected_summary_is_not_retried_without_summary"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_summary_preserves_encrypted_replay"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_encrypted_replay_survives_tool_cycles"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_replay_requires_encrypted_content_and_preserves_provider_fields"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_replay_retention_policy_has_dedicated_exact_limits"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_replay_item_limit_is_prospective_at_limit_boundaries"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_replay_summary_node_limit_counts_nested_empty_and_scalar_entries"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_replay_summary_character_limit_counts_keys_and_values"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_replay_summary_serialized_byte_limit_uses_compact_unicode_json"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_replay_rejects_non_json_values_without_coercion"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_replay_overflow_is_safe_atomic_non_retryable_and_zero_dispatch"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_replay_rollback_release_and_request_isolation"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_encrypted_reasoning_remains_opaque"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_encrypted_reasoning_is_not_displayed_or_relabelled"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_pre_output_retry_preserves_request_and_resets_state"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_visible_native_or_summary_reasoning_disables_retry"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_failed_attempt_reasoning_and_function_calls_roll_back_together"
    ),
)


def test_phase3_acceptance_manifest_pins_exact_catalog() -> None:
    manifest = cast(
        Any,
        _ACCEPTANCE_SCRIPT.load_manifest(
            _ACCEPTANCE_SCRIPT.default_manifest_path()
        ),
    )

    assert manifest.dimensions[_PHASE3_DIMENSION] == _PHASE3_NODE_IDS
    assert len(_PHASE3_NODE_IDS) == 26
    assert len(manifest.node_ids) == len(set(manifest.node_ids))
