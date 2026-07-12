"""Test the exact Phase 2 reasoning-summary acceptance inventory."""

from typing import Any, cast

from reasoning_summary_script_loader import load_reasoning_summary_script

_ACCEPTANCE_SCRIPT = load_reasoning_summary_script(
    "verify_reasoning_summary_acceptance"
)
_PHASE2_DIMENSION = (
    "phase 2 typed canonical representation and native migration"
)
_PHASE2_NODE_IDS = (
    (
        "tests/reasoning_summary_phase2_manifest_test.py::"
        "test_phase2_acceptance_manifest_pins_exact_catalog"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_summary_reuses_canonical_reasoning_channel"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_reasoning_representation_is_not_relabelled"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_summary_uses_only_reasoning_channel"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_summary_reuses_reasoning_kinds"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_no_summary_specific_canonical_kind"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_reasoning_representation_validation"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_reasoning_identity_is_structured_without_payload"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_represented_reasoning_is_private"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_every_summary_delta_is_typed"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_reasoning_identity_round_trip_preserves_optional_correlations"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_reasoning_segment_boundaries_and_response_resets"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_reasoning_empty_deltas_are_operationally_invisible"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_reasoning_segment_allocator_is_constant_space_for_ten_thousand_segments"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_provider_capabilities_serialize_native_and_summary_support"
    ),
    (
        "tests/model/model_stream_contract_test.py::StreamContractTestCase::"
        "test_reasoning_observability_is_content_free"
    ),
    (
        "tests/model/reasoning_summary_native_provider_regression_test.py::"
        "test_representation_does_not_change_provider_semantics"
    ),
    (
        "tests/model/reasoning_summary_native_provider_regression_test.py::"
        "test_native_reasoning_is_not_summary_fallback"
    ),
    (
        "tests/model/reasoning_summary_native_provider_regression_test.py::"
        "test_every_native_delta_is_typed"
    ),
    (
        "tests/model/reasoning_summary_native_provider_regression_test.py::"
        "test_native_providers_remain_compatible"
    ),
    (
        "tests/model/reasoning_summary_native_provider_regression_test.py::"
        "test_shared_local_reasoning_parser_backend_parity"
    ),
    (
        "tests/model/reasoning_summary_native_provider_regression_test.py::"
        "test_mlx_reasoning_parser_preserves_untagged_cancel_and_close"
    ),
    (
        "tests/model/reasoning_summary_native_provider_regression_test.py::"
        "test_public_stream_capabilities_keep_summary_support_disabled"
    ),
    (
        "tests/model/nlp/vendor_anthropic_test.py::"
        "test_canonical_stream_maps_anthropic_events"
    ),
    (
        "tests/model/nlp/vendor_bedrock_test.py::BedrockTestCase::"
        "test_canonical_stream_maps_bedrock_events"
    ),
    (
        "tests/model/nlp/vendor_extra_test.py::LiteLLMTestCase::"
        "test_canonical_stream_maps_chat_chunks"
    ),
    (
        "tests/model/nlp/vendor_openai_test.py::OpenAITestCase::"
        "test_canonical_stream_maps_responses_events"
    ),
    (
        "tests/model/text_generation_response_golden_trace_test.py::"
        "TextGenerationResponseGoldenTraceTestCase::"
        "test_canonical_stream_matches_current_golden_trace"
    ),
    (
        "tests/model/text_generation_response_golden_trace_test.py::"
        "TextGenerationResponseGoldenTraceTestCase::"
        "test_usage_callbacks_and_cancelled_terminal_trace"
    ),
    (
        "tests/model/text_generation_response_golden_trace_test.py::"
        "TextGenerationResponseGoldenTraceTestCase::"
        "test_provider_error_characterizes_error_terminal_trace"
    ),
    (
        "tests/agent/orchestrator_response_test.py::"
        "OrchestratorResponseCanonicalLifecycleTestCase::"
        "test_reasoning_provenance_survives_provider_and_canonical_copy_paths"
    ),
    (
        "tests/agent/orchestrator_response_test.py::"
        "OrchestratorResponseCanonicalLifecycleTestCase::"
        "test_two_tool_continuations_preserve_reasoning_identity"
    ),
    (
        "tests/agent/orchestrator_response_test.py::"
        "OrchestratorResponseCanonicalLifecycleTestCase::"
        "test_conflicting_reasoning_continuation_is_rejected"
    ),
    (
        "tests/task/event_test.py::SanitizedTaskEventTest::"
        "test_token_event_keeps_canonical_stream_observability"
    ),
)


def test_phase2_acceptance_manifest_pins_exact_catalog() -> None:
    manifest = cast(
        Any,
        _ACCEPTANCE_SCRIPT.load_manifest(
            _ACCEPTANCE_SCRIPT.default_manifest_path()
        ),
    )

    assert manifest.dimensions[_PHASE2_DIMENSION] == _PHASE2_NODE_IDS
    assert len(_PHASE2_NODE_IDS) == 34
    assert len(manifest.node_ids) == len(set(manifest.node_ids))
