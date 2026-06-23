from dataclasses import dataclass
from json import loads
from pathlib import Path
from subprocess import run
from sys import executable
from unittest import TestCase

REQUIRED_ACCEPTANCE_DIMENSIONS = (
    "stream equivalence",
    "SDK losslessness",
    "UI lossy isolation",
    "provider/local backpressure",
    "live tool output",
    "tool ordering",
    "protocol consistency",
    "channel boundaries",
    "event stats isolation",
    "reasoning parser correctness",
    "server no-listener memory bounds",
)

REQUIRED_HARDENING_DIMENSIONS = (
    "incremental tool parser state",
    "static protocol envelopes",
    "optional enrichment opt-outs",
    "lazy render cleanup",
    "malformed tool calls",
    "duplicate terminal items",
    "content after terminal state",
    "missing correlation ids",
    "subscriber overflow",
    "cancellation races",
    "retention limits",
    "protocol route e2e surfaces",
    "streaming conformance negatives",
    "protocol route negatives",
    "parser negatives",
    "CLI/SDK projection negatives",
    "benchmark and latency guard negatives",
    "inventory guard negatives",
)

REQUIRED_FINAL_NEGATIVE_E2E_DIMENSIONS = (
    "canonical validation",
    "producers",
    "local parser",
    "orchestration/tools",
    "eventing",
    "SDK",
    "stdout/CLI/FancyTheme",
    "Chat SSE",
    "Responses SSE",
    "MCP",
    "A2A",
    "flow",
    "cancellation/disconnect",
    "backpressure",
    "bounded memory",
    "reasoning parsing",
    "acceptance inventory",
)

REQUIRED_FINAL_GATE_DIMENSIONS = (
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
)


@dataclass(frozen=True)
class _FinalGateHarnessEvidence:
    synthetic: tuple[str, ...]
    integration: tuple[str, ...]


ACCEPTANCE_HARNESS_TESTS = {
    "stream equivalence": (
        (
            "tests/model/text_generation_response_golden_trace_test.py"
            "::TextGenerationResponseGoldenTraceTestCase"
            "::test_canonical_stream_matches_current_golden_trace"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_stream_accumulation_and_to_str_match_answer_semantics"
        ),
        (
            "tests/server/streaming_conformance_test.py"
            "::test_canonical_trace_conforms_across_public_stream_surfaces"
        ),
        (
            "tests/server/streaming_conformance_test.py"
            "::test_canonical_golden_trace_locks_item_semantics"
        ),
    ),
    "SDK losslessness": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_consumer_projection_iterator_preserves_order_losslessly"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_consumer_projection_iterator_rejects_sequence_gap"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_consumer_projections_stream_lossless_items"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_async_iteration_rejects_semantic_sequence_discontinuity"
        ),
    ),
    "UI lossy isolation": (
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_lossy_cli_frames_do_not_drop_lossless_public_surfaces"
        ),
        (
            "tests/cli/model_test.py"
            "::CliRenderFrameTestCase"
            "::test_frame_rate_renderer_coalesces_fancy_theme_frames"
        ),
        (
            "tests/event/event_manager_test.py"
            "::EventManagerTestCase"
            "::test_coalescing_ui_stream_subscriber_keeps_lossless_items"
        ),
    ),
    "provider/local backpressure": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_provider_and_local_streams_wait_for_slow_consumers"
        ),
        (
            "tests/model/nlp/vendor_openai_test.py"
            "::OpenAITestCase"
            "::test_canonical_disconnect_closes_provider_no_read_ahead"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_hosted_projection_disconnect_closes_after_first_item"
        ),
        (
            "tests/model/nlp/text_generation_methods_more_test.py"
            "::StreamGeneratorTestCase"
            "::test_lossless_streamer_handoff_blocks_until_consumed"
        ),
        (
            "tests/model/nlp/text_generation_methods_more_test.py"
            "::StreamGeneratorTestCase"
            "::test_stream_generator_blocks_under_slow_consumer_pressure"
        ),
        (
            "tests/model/ds4_model_test.py"
            "::test_ds4_stream_queue_backpressure_does_not_deadlock_event_loop"
        ),
    ),
    "live tool output": (
        (
            "tests/agent/orchestrator_response_test.py"
            "::OrchestratorResponseCanonicalLifecycleTestCase"
            "::test_iteration_records_live_tool_output_before_completion"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_tool_lifecycle_rejects_live_items_after_completion"
        ),
        (
            "tests/server/streaming_conformance_test.py"
            "::test_canonical_trace_conforms_across_public_stream_surfaces"
        ),
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_same_canonical_stream_projects_through_protocols"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_updates_resource_from_canonical_tool_output"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_preserves_consecutive_tool_output_metadata"
        ),
    ),
    "tool ordering": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_tool_lifecycle_assembles_planned_order_observations"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_tool_lifecycle_observation_assembly_rejects_planned_mismatches"
        ),
        (
            "tests/agent/orchestrator_response_test.py"
            "::OrchestratorResponseCanonicalLifecycleTestCase"
            "::test_iteration_parallel_results_emit_before_continuation"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolTestCase"
            "::test_call_streams_answer_tool_and_status_events"
        ),
    ),
    "protocol consistency": (
        (
            "tests/server/streaming_conformance_test.py"
            "::test_canonical_trace_conforms_across_public_stream_surfaces"
        ),
        (
            "tests/server/primary_consumer_projection_golden_test.py"
            "::PrimaryConsumerProjectionGoldenTestCase"
            "::test_primary_consumers_project_same_canonical_trace"
        ),
        (
            "tests/server/streaming_conformance_test.py"
            "::test_terminal_outcome_traces_project_protocol_terminal_events"
        ),
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_terminal_outcome_traces_project_through_protocols"
        ),
        (
            "tests/model/text_generation_response_golden_trace_test.py"
            "::TextGenerationResponseGoldenTraceTestCase"
            "::test_protocol_output_traces_capture_current_projections"
        ),
    ),
    "channel boundaries": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_sequence_validator_rejects_channel_boundary_errors"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_stream_projection_state_projects_canonical_and_projection_items"
        ),
        (
            "tests/server/streaming_conformance_test.py"
            "::test_canonical_trace_rejects_content_after_terminal"
        ),
    ),
    "event stats isolation": (
        (
            "tests/event/event_manager_test.py"
            "::EventManagerTestCase"
            "::test_event_stats_instances_are_isolated"
        ),
        (
            "tests/event/event_manager_test.py"
            "::EventManagerTestCase"
            "::test_server_mode_stats_are_explicit_opt_in"
        ),
    ),
    "reasoning parser correctness": (
        (
            "tests/agent/reasoning_parser_split_tag_test.py"
            "::ReasoningParserSplitTagTestCase"
            "::test_split_channel_start_and_end_tags"
        ),
        (
            "tests/agent/reasoning_parser_edge_test.py"
            "::ReasoningParserEdgeTestCase"
            "::test_flush_reasoning_tokens"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_parses_split_reasoning_tags"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_preserves_split_marker_whitespace"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_handles_adjacent_reasoning_sections"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_handles_split_adjacent_sections"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_closes_unterminated_reasoning"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_keeps_partial_end_marker_reasoning"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_keeps_malformed_markers_as_answer"
        ),
        (
            "tests/model/nlp/vendor_openai_test.py"
            "::OpenAITestCase"
            "::test_canonical_stream_preserves_hosted_reasoning_whitespace"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_consumer_projections_preserve_split_reasoning_whitespace"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_consumer_projections_handle_adjacent_reasoning_sections"
        ),
    ),
    "server no-listener memory bounds": (
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_repeated_chat_stream_requests_bound_no_listener_state"
        ),
        (
            "tests/server/responses_test.py"
            "::ResponsesEndpointTestCase"
            "::test_repeated_requests_without_ui_listener_do_not_retain_events"
        ),
        (
            "tests/server/responses_test.py"
            "::ResponsesEndpointTestCase"
            "::test_repeated_streaming_requests_do_not_retain_events"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_repeated_response_stream_requests_release_sources"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_repeated_streams_bound_resource_state_and_keep_result"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_repeated_stream_responses_bound_shared_resource_store"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_close_after_terminal_result_cleans_sources"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_cleans_up_when_terminal_emit_closes"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolHttpE2ETestCase"
            "::test_calls_sdk_v1_router_and_streams_status"
        ),
        (
            "tests/server/router_streaming_test.py"
            "::RouterStreamingTestCase"
            "::test_default_server_stream_retention_surfaces_are_bounded"
        ),
        (
            "tests/server/orchestrator_di_test.py"
            "::OrchestratorDiTestCase"
            "::test_di_get_orchestrator_from_file"
        ),
        (
            "tests/server/orchestrator_di_test.py"
            "::OrchestratorDiTestCase"
            "::test_di_get_orchestrator_from_settings"
        ),
    ),
}

HARDENING_COVERAGE_TESTS = {
    "incremental tool parser state": (
        (
            "tests/agent/tool_call_parser_extra_test.py"
            "::ToolCallParserExtraTestCase"
            "::test_streaming_hot_path_uses_incremental_marker_state"
        ),
        (
            "tests/agent/tool_call_parser_extra_test.py"
            "::ToolCallParserExtraTestCase"
            "::test_split_harmony_final_marker_stays_visible_suffix"
        ),
        (
            "tests/agent/tool_call_parser_extra_test.py"
            "::ToolCallParserExtraTestCase"
            "::test_stream_reports_malformed_open_sibling_after_valid_call"
        ),
    ),
    "static protocol envelopes": (
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_chat_chunk_envelope_does_not_leak_mutable_state"
        ),
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_chat_chunk_envelope_rejects_invalid_state"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_response_sse_event_rejects_incompatible_coalesce"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_response_sse_event_coalesces_only_adjacent_matching_events"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_preserves_consecutive_tool_output_metadata"
        ),
    ),
    "optional enrichment opt-outs": (
        (
            "tests/event/event_manager_test.py"
            "::EventManagerTestCase"
            "::test_server_mode_stats_are_explicit_opt_in"
        ),
        (
            "tests/agent/orchestrator_response_test.py"
            "::OrchestratorResponseIterationTestCase"
            "::test_token_events_skip_tokenizer_when_enrichment_disabled"
        ),
        (
            "tests/cli/model_test.py"
            "::CliTokenGenerationTestCase"
            "::test_projection_display_token_uses_canonical_metadata"
        ),
        (
            "tests/cli/model_test.py"
            "::CliTokenGenerationTestCase"
            "::test_token_stream_skips_tokenizer_config_without_display_tokens"
        ),
    ),
    "lazy render cleanup": (
        (
            "tests/cli/model_test.py"
            "::CliTokenGenerationTestCase"
            "::test_token_stream_sets_stop_signal_on_pause_cancel"
        ),
        (
            "tests/cli/model_test.py"
            "::CliTokenGenerationTestCase"
            "::test_token_stream_consumes_sync_theme_frames"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_protocol_projection_state_construction_is_canonical_only"
        ),
    ),
    "malformed tool calls": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_stream_negative_acceptance_edges"
        ),
        (
            "tests/agent/tool_call_parser_extra_test.py"
            "::ToolCallParserExtraTestCase"
            "::test_malformed_tool_tag_stream_emits_diagnostic_event"
        ),
    ),
    "duplicate terminal items": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_stream_negative_acceptance_edges"
        ),
        (
            "tests/server/router_streaming_test.py"
            "::RouterStreamingTestCase"
            "::test_protocol_stream_accumulator_rejects_duplicate_terminal"
        ),
    ),
    "content after terminal state": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_stream_negative_acceptance_edges"
        ),
        (
            "tests/model/model_stream_benchmark_regression_test.py"
            "::StreamBenchmarkRegressionTestCase"
            "::test_rejects_late_content_during_projection_benchmark"
        ),
    ),
    "missing correlation ids": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_stream_negative_acceptance_edges"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_response_sse_item_state_rejects_invalid_values"
        ),
    ),
    "subscriber overflow": (
        (
            "tests/event/event_manager_test.py"
            "::EventManagerTestCase"
            "::test_stream_item_subscriber_overflow_fails_closed"
        ),
    ),
    "cancellation races": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_provider_stream_cancellation_race_closes_source"
        ),
        (
            "tests/model/nlp/vendor_openai_test.py"
            "::OpenAITestCase"
            "::test_canonical_cancel_closes_pending_provider_pull"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_hosted_response_preopen_cancel_cancels_provider"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_hosted_projection_cancel_closes_pending_read"
        ),
        (
            "tests/server/streaming_latency_budget_test.py"
            "::StreamingLatencyBudgetTestCase"
            "::test_cancellation_and_close_latency_budget_across_surfaces"
        ),
    ),
    "retention limits": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_stream_negative_acceptance_edges"
        ),
        (
            "tests/server/router_streaming_test.py"
            "::RouterStreamingTestCase"
            "::test_protocol_stream_retention_settings_reject_invalid_values"
        ),
        (
            "tests/server/router_streaming_test.py"
            "::RouterStreamingTestCase"
            "::test_default_server_stream_retention_surfaces_are_bounded"
        ),
        (
            "tests/model/model_stream_benchmark_regression_test.py"
            "::StreamBenchmarkRegressionTestCase"
            "::test_long_stream_retention_peak_memory_within_budget"
        ),
        (
            "tests/server/a2a_v1_router_test.py"
            "::test_install_a2a_routes_mounts_v1_sdk_routes"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolHttpE2ETestCase"
            "::test_calls_sdk_v1_router_and_streams_status"
        ),
    ),
    "protocol route e2e surfaces": (
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_create_chat_completion_streams_canonical_items"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_emits_canonical_items"
        ),
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_same_canonical_stream_projects_through_protocols"
        ),
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_terminal_outcome_traces_project_through_protocols"
        ),
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_lossy_cli_frames_do_not_drop_lossless_public_surfaces"
        ),
        (
            "tests/server/streaming_latency_budget_test.py"
            "::StreamingLatencyBudgetTestCase"
            "::test_cancellation_and_close_latency_budget_across_surfaces"
        ),
    ),
    "streaming conformance negatives": (
        (
            "tests/server/streaming_conformance_test.py"
            "::test_canonical_trace_rejects_content_after_terminal"
        ),
        (
            "tests/server/streaming_conformance_test.py"
            "::test_canonical_trace_rejects_missing_terminal"
        ),
        (
            "tests/server/streaming_conformance_test.py"
            "::test_public_projection_helpers_reject_unsupported_items"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_rejects_empty_stream_missing_terminal"
        ),
    ),
    "protocol route negatives": (
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_create_chat_completion_stream_rejects_late_content"
        ),
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_create_chat_completion_stream_rejects_late_projection"
        ),
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_create_chat_completion_stream_rejects_mixed_surfaces"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_rejects_canonical_content_after_terminal"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_rejects_projected_content_after_terminal"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_rejects_canonical_stream_missing_terminal"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_rejects_mixed_stream_surfaces"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_rejects_duplicate_canonical_terminal"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_legacy_rejection_first_item"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_rejects_legacy_after_terminal"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_rejects_mixed_surfaces"
        ),
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_default_protocol_routes_legacy_rejection_first_item"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolTestCase"
            "::test_failed_terminal_status_raises"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolTestCase"
            "::test_sdk_client_error_response_raises"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolTestCase"
            "::test_response_without_terminal_status_raises"
        ),
    ),
    "parser negatives": (
        (
            "tests/agent/reasoning_parser_split_tag_test.py"
            "::ReasoningParserSplitTagTestCase"
            "::test_malformed_embedded_partial_tag_stays_visible"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_keeps_malformed_markers_as_answer"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_reports_malformed_tool_call_text"
        ),
        (
            "tests/agent/tool_call_parser_extra_test.py"
            "::ToolCallParserExtraTestCase"
            "::test_malformed_tool_tag_stream_emits_diagnostic_event"
        ),
        (
            "tests/agent/tool_call_parser_extra_test.py"
            "::ToolCallParserExtraTestCase"
            "::test_stream_reports_malformed_open_sibling_after_valid_call"
        ),
    ),
    "CLI/SDK projection negatives": (
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_async_iteration_rejects_semantic_sequence_discontinuity"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_async_iteration_rejects_mixed_semantic_and_legacy_output"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_async_iteration_rejects_semantic_output_missing_terminal"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_project_stream_consumer_item_uses_shared_projection_state"
        ),
        (
            "tests/cli/model_test.py"
            "::CliTokenGenerationTestCase"
            "::test_token_generation_no_stats_rejects_late_projection"
        ),
        (
            "tests/cli/model_test.py"
            "::CliTokenGenerationTestCase"
            "::test_render_projections_rejects_legacy_mixed_stream"
        ),
        (
            "tests/cli/model_test.py"
            "::CliTokenGenerationTestCase"
            "::test_render_projections_rejects_bad_api_item"
        ),
        (
            "tests/cli/model_test.py"
            "::CliTokenGenerationTestCase"
            "::test_plain_stdout_legacy_rejection_first_item"
        ),
        (
            "tests/cli/model_test.py"
            "::CliTokenGenerationTestCase"
            "::test_token_generation_no_stats_rejects_bad_response_projection"
        ),
    ),
    "benchmark and latency guard negatives": (
        (
            "tests/model/model_stream_benchmark_regression_test.py"
            "::StreamBenchmarkRegressionTestCase"
            "::test_latency_and_overhead_budgets_reject_non_positive_values"
        ),
        (
            "tests/server/streaming_latency_budget_test.py"
            "::StreamingLatencyBudgetValidationTestCase"
            "::test_cancellation_and_close_budgets_reject_non_positive_values"
        ),
        (
            "tests/model/model_stream_benchmark_regression_test.py"
            "::StreamBenchmarkRegressionTestCase"
            "::test_projection_benchmark_rejects_unsupported_item"
        ),
        (
            "tests/model/model_stream_benchmark_regression_test.py"
            "::StreamBenchmarkRegressionTestCase"
            "::test_final_benchmark_markdown_parser_rejects_missing_phase_row"
        ),
        (
            "tests/model/model_stream_benchmark_regression_test.py"
            "::StreamBenchmarkRegressionTestCase"
            "::test_final_benchmark_markdown_parser_rejects_missing_phase1_reference"
        ),
        (
            "tests/model/model_stream_benchmark_regression_test.py"
            "::StreamBenchmarkRegressionTestCase"
            "::test_final_benchmark_markdown_parser_rejects_missing_phase6_reference"
        ),
        (
            "tests/model/model_stream_benchmark_regression_test.py"
            "::StreamBenchmarkRegressionTestCase"
            "::test_final_benchmark_markdown_parser_rejects_missing_openai_acceptance"
        ),
        (
            "tests/model/model_stream_benchmark_regression_test.py"
            "::StreamBenchmarkRegressionTestCase"
            "::test_final_benchmark_markdown_parser_rejects_missing_gpt_oss_accept"
        ),
        (
            "tests/model/model_stream_benchmark_regression_test.py"
            "::StreamBenchmarkRegressionTestCase"
            "::test_benchmark_comparison_rejects_missing_model_row"
        ),
        (
            "tests/model/model_stream_benchmark_regression_test.py"
            "::StreamBenchmarkRegressionTestCase"
            "::test_final_benchmark_markdown_parser_rejects_missing_baseline"
        ),
    ),
    "inventory guard negatives": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_legacy_classifier_guard_detects_new_direct_classifiers"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_legacy_classifier_guard_strict_mode_reports_rejection_guards"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_production_streaming_source_has_no_legacy_classifiers"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_forbidden_text_guard_detects_tokens_with_narrow_allowances"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_forbidden_production_streaming_text_tokens_are_absent"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_legacy_classifier_guard_detects_grouped_aliases"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_legacy_classifier_guard_detects_type_and_match_classifiers"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_legacy_classifier_guard_detects_delayed_alias_classifiers"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_legacy_classification_import_guard_detects_common_shapes"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_runtime_streaming_source_has_no_legacy_classification_imports"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_legacy_classifier_guard_detects_tracked_string_classifiers"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_public_streaming_return_guard_detects_legacy_item_aliases"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_public_canonical_to_legacy_projection_guard_detects_shapes"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_public_canonical_to_legacy_projections_are_absent"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_protocol_projection_mapper_guard_detects_merged_constant_keys"
        ),
        (
            "tests/model/nlp/mlxlm_extra_test.py"
            "::MlxLmStreamTestCase"
            "::test_stream_accepts_non_legacy_token_and_text_chunks"
        ),
        (
            "tests/model/nlp/mlxlm_extra_test.py"
            "::MlxLmStreamTestCase"
            "::test_stream_rejects_legacy_token_subclass_chunk"
        ),
        (
            "tests/model/nlp/vllm_extra_test.py"
            "::VllmStreamTestCase"
            "::test_stream_accepts_non_legacy_token_and_text_chunks"
        ),
        (
            "tests/model/nlp/vllm_extra_test.py"
            "::VllmStreamTestCase"
            "::test_stream_rejects_legacy_token_subclass_chunk"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_legacy_classifier_inventory_rejects_malformed_entries"
        ),
        (
            "tests/server/streaming_acceptance_inventory_test.py"
            "::StreamingAcceptanceInventoryTestCase"
            "::test_streaming_acceptance_inventory_rejects_uncollectable_node"
        ),
        (
            "tests/server/streaming_acceptance_inventory_test.py"
            "::StreamingAcceptanceInventoryTestCase"
            "::test_streaming_acceptance_inventory_rejects_skip_or_xfail_marker"
        ),
        (
            "tests/server/streaming_acceptance_inventory_test.py"
            "::StreamingAcceptanceInventoryTestCase"
            "::test_streaming_hardening_inventory_rejects_missing_dimension"
        ),
    ),
}

FINAL_NEGATIVE_E2E_SUITE_TESTS = {
    "canonical validation": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_stream_negative_acceptance_edges"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_sequence_validator_rejects_terminal_and_order_errors"
        ),
        (
            "tests/server/streaming_conformance_test.py"
            "::test_canonical_trace_rejects_missing_terminal"
        ),
    ),
    "producers": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_provider_stream_normalizer_assigns_identity_and_metadata"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_provider_stream_normalizer_assigns_identity_and_metadata"
        ),
        (
            "tests/model/nlp/vendor_openai_test.py"
            "::OpenAITestCase"
            "::test_canonical_cancel_closes_pending_provider_pull"
        ),
        (
            "tests/model/nlp/text_generation_methods_more_test.py"
            "::StreamGeneratorTestCase"
            "::test_stream_generator_blocks_under_slow_consumer_pressure"
        ),
    ),
    "local parser": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_parses_split_reasoning_tags"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_closes_unterminated_reasoning"
        ),
        (
            "tests/agent/tool_call_parser_extra_test.py"
            "::ToolCallParserExtraTestCase"
            "::test_stream_reports_malformed_open_sibling_after_valid_call"
        ),
    ),
    "orchestration/tools": (
        (
            "tests/agent/orchestrator_response_test.py"
            "::OrchestratorResponseCanonicalLifecycleTestCase"
            "::test_iteration_records_live_tool_output_before_completion"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_tool_lifecycle_assembles_planned_order_observations"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_tool_lifecycle_rejects_live_items_after_completion"
        ),
    ),
    "eventing": (
        (
            "tests/event/event_manager_test.py"
            "::EventManagerTestCase"
            "::test_trigger_stream_item_publishes_canonical_projection"
        ),
        (
            "tests/event/event_manager_test.py"
            "::EventManagerTestCase"
            "::test_stream_item_subscriber_overflow_fails_closed"
        ),
        (
            "tests/event/event_manager_test.py"
            "::EventManagerTestCase"
            "::test_event_stats_instances_are_isolated"
        ),
    ),
    "SDK": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_consumer_projection_iterator_preserves_order_losslessly"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_consumer_projection_iterator_rejects_sequence_gap"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_consumer_projections_stream_lossless_items"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_async_iteration_rejects_semantic_sequence_discontinuity"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_project_stream_consumer_item_uses_shared_projection_state"
        ),
    ),
    "stdout/CLI/FancyTheme": (
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_lossy_cli_frames_do_not_drop_lossless_public_surfaces"
        ),
        (
            "tests/cli/model_test.py"
            "::CliTokenGenerationTestCase"
            "::test_render_projections_rejects_unsupported_item"
        ),
        (
            "tests/cli/model_test.py"
            "::CliTokenGenerationTestCase"
            "::test_plain_stdout_legacy_rejection_first_item"
        ),
        (
            "tests/cli/model_test.py"
            "::CliRenderFrameTestCase"
            "::test_frame_rate_renderer_coalesces_fancy_theme_frames"
        ),
    ),
    "Chat SSE": (
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_create_chat_completion_streams_canonical_items"
        ),
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_create_chat_completion_stream_rejects_late_content"
        ),
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_create_chat_completion_stream_disconnect_closes_source"
        ),
    ),
    "Responses SSE": (
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_emits_canonical_items"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_rejects_canonical_content_after_terminal"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_response_disconnect_closes_source_before_pull"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_response_cancellation_cancels_source"
        ),
    ),
    "MCP": (
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_emits_canonical_notifications"
        ),
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_terminal_outcome_traces_project_through_protocols"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_rejects_missing_canonical_terminal"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_rejects_duplicate_canonical_terminal"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_legacy_rejection_first_item"
        ),
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_default_protocol_routes_legacy_rejection_first_item"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_cleans_up_when_consumer_closes"
        ),
    ),
    "A2A": (
        (
            "tests/server/a2a_v1_router_test.py"
            "::test_install_a2a_routes_mounts_v1_sdk_routes"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolHttpE2ETestCase"
            "::test_calls_sdk_v1_router_and_streams_status"
        ),
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_terminal_outcome_traces_project_through_protocols"
        ),
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_default_protocol_routes_legacy_rejection_first_item"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolTestCase"
            "::test_failed_terminal_status_raises"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolTestCase"
            "::test_response_without_terminal_status_raises"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolTestCase"
            "::test_cancellation_checker_runs_before_stream_emit"
        ),
    ),
    "flow": (
        (
            "tests/server/protocol_streaming_e2e_test.py"
            "::test_same_canonical_stream_projects_through_protocols"
        ),
        (
            "tests/flow/runtime_test.py"
            "::FlowPlanExecutionTestCase"
            "::test_execute_flow_plan_projects_flow_events_to_canonical_items"
        ),
        (
            "tests/flow/runtime_test.py"
            "::FlowPlanExecutionTestCase"
            "::test_execute_flow_plan_rejects_bad_stream_session"
        ),
        (
            "tests/flow/runtime_test.py"
            "::FlowRuntimeEndToEndTestCase"
            "::test_loaded_definition_routes_cancellation"
        ),
    ),
    "cancellation/disconnect": (
        (
            "tests/server/streaming_latency_budget_test.py"
            "::StreamingLatencyBudgetTestCase"
            "::test_cancellation_and_close_latency_budget_across_surfaces"
        ),
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_create_chat_completion_stream_cancellation_cancels_source"
        ),
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_response_cancellation_cancels_source"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_stream_response_interrupts_pending_pull_on_cancellation"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolTestCase"
            "::test_cancellation_checker_runs_before_stream_emit"
        ),
    ),
    "backpressure": (
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_provider_and_local_streams_wait_for_slow_consumers"
        ),
        (
            "tests/model/nlp/text_generation_methods_more_test.py"
            "::StreamGeneratorTestCase"
            "::test_lossless_streamer_handoff_blocks_until_consumed"
        ),
        (
            "tests/model/nlp/text_generation_methods_more_test.py"
            "::StreamGeneratorTestCase"
            "::test_stream_generator_blocks_under_slow_consumer_pressure"
        ),
        (
            "tests/model/ds4_model_test.py"
            "::test_ds4_stream_queue_backpressure_does_not_deadlock_event_loop"
        ),
    ),
    "bounded memory": (
        (
            "tests/server/chat_router_unit_test.py"
            "::ChatRouterUnitTest"
            "::test_repeated_chat_stream_requests_bound_no_listener_state"
        ),
        (
            "tests/server/responses_test.py"
            "::ResponsesEndpointTestCase"
            "::test_repeated_streaming_requests_do_not_retain_events"
        ),
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterAsyncTestCase"
            "::test_repeated_stream_responses_bound_shared_resource_store"
        ),
        (
            "tests/tool/a2a_tool_test.py"
            "::A2ACallToolHttpE2ETestCase"
            "::test_calls_sdk_v1_router_and_streams_status"
        ),
        (
            "tests/server/router_streaming_test.py"
            "::RouterStreamingTestCase"
            "::test_default_server_stream_retention_surfaces_are_bounded"
        ),
    ),
    "reasoning parsing": (
        (
            "tests/agent/reasoning_parser_split_tag_test.py"
            "::ReasoningParserSplitTagTestCase"
            "::test_split_channel_start_and_end_tags"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_preserves_split_marker_whitespace"
        ),
        (
            "tests/model/model_stream_contract_test.py"
            "::StreamContractTestCase"
            "::test_local_stream_normalizer_handles_adjacent_reasoning_sections"
        ),
        (
            "tests/model/text_generation_response_more_test.py"
            "::TextGenerationResponseMoreTestCase"
            "::test_consumer_projections_handle_adjacent_reasoning_sections"
        ),
    ),
    "acceptance inventory": (
        (
            "tests/server/streaming_acceptance_inventory_test.py"
            "::StreamingAcceptanceInventoryTestCase"
            "::test_streaming_acceptance_inventory_collects_unskipped_tests"
        ),
        (
            "tests/server/streaming_acceptance_inventory_test.py"
            "::StreamingAcceptanceInventoryTestCase"
            "::test_streaming_hardening_inventory_collects_unskipped_tests"
        ),
        (
            "tests/server/streaming_acceptance_inventory_test.py"
            "::StreamingAcceptanceInventoryTestCase"
            "::test_streaming_acceptance_inventory_rejects_uncollectable_node"
        ),
        (
            "tests/server/streaming_acceptance_inventory_test.py"
            "::StreamingAcceptanceInventoryTestCase"
            "::test_final_negative_e2e_inventory_rejects_missing_dimension"
        ),
    ),
}

FINAL_GATE_ACCEPTANCE_HARNESSES = {
    "stream accumulation/to_str": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/model/text_generation_response_more_test.py"
                "::TextGenerationResponseMoreTestCase"
                "::test_stream_accumulation_and_to_str_match_answer_semantics"
            ),
            (
                "tests/model/text_generation_response_more_test.py"
                "::TextGenerationResponseMoreTestCase"
                "::test_to_str_preserves_stream_terminal_failure_semantics"
            ),
        ),
        integration=(
            (
                "tests/server/streaming_conformance_test.py"
                "::test_canonical_trace_conforms_across_public_stream_surfaces"
            ),
            (
                "tests/model/text_generation_response_golden_trace_test.py"
                "::TextGenerationResponseGoldenTraceTestCase"
                "::test_to_str_matches_accumulated_golden_answer"
            ),
        ),
    ),
    "SDK losslessness": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_consumer_projection_iterator_preserves_order_losslessly"
            ),
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_consumer_projection_iterator_rejects_sequence_gap"
            ),
            (
                "tests/model/text_generation_response_more_test.py"
                "::TextGenerationResponseMoreTestCase"
                "::test_consumer_projections_stream_lossless_items"
            ),
        ),
        integration=(
            (
                "tests/server/protocol_streaming_e2e_test.py"
                "::test_lossy_cli_frames_do_not_drop_lossless_public_surfaces"
            ),
            (
                "tests/server/streaming_conformance_test.py"
                "::test_canonical_trace_conforms_across_public_stream_surfaces"
            ),
        ),
    ),
    "FancyTheme lossless isolation": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/cli/model_test.py"
                "::CliRenderFrameTestCase"
                "::test_frame_rate_renderer_coalesces_fancy_theme_frames"
            ),
            (
                "tests/event/event_manager_test.py"
                "::EventManagerTestCase"
                "::test_coalescing_ui_stream_subscriber_keeps_lossless_items"
            ),
        ),
        integration=(
            (
                "tests/server/protocol_streaming_e2e_test.py"
                "::test_lossy_cli_frames_do_not_drop_lossless_public_surfaces"
            ),
        ),
    ),
    "local backpressure": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/model/nlp/text_generation_methods_more_test.py"
                "::StreamGeneratorTestCase"
                "::test_lossless_streamer_handoff_blocks_until_consumed"
            ),
            (
                "tests/model/nlp/text_generation_methods_more_test.py"
                "::StreamGeneratorTestCase"
                "::test_stream_generator_blocks_under_slow_consumer_pressure"
            ),
            (
                "tests/model/ds4_model_test.py"
                "::test_ds4_stream_queue_backpressure_does_not_deadlock_event_loop"
            ),
        ),
        integration=(
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_provider_and_local_streams_wait_for_slow_consumers"
            ),
        ),
    ),
    "hosted cleanup": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/model/text_generation_response_more_test.py"
                "::TextGenerationResponseMoreTestCase"
                "::test_hosted_response_preopen_close_closes_provider"
            ),
            (
                "tests/model/text_generation_response_more_test.py"
                "::TextGenerationResponseMoreTestCase"
                "::test_hosted_projection_disconnect_closes_after_first_item"
            ),
            (
                "tests/model/text_generation_response_more_test.py"
                "::TextGenerationResponseMoreTestCase"
                "::test_hosted_projection_cancel_closes_pending_read"
            ),
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_provider_stream_normalizer_closes_on_consumer_disconnect"
            ),
        ),
        integration=(
            (
                "tests/server/chat_router_unit_test.py"
                "::ChatRouterUnitTest"
                "::test_create_chat_completion_stream_disconnect_closes_source"
            ),
            (
                "tests/server/create_response_sse_test.py"
                "::CreateResponseSSEEventsTestCase"
                "::test_streaming_response_disconnect_closes_source_before_pull"
            ),
            (
                "tests/server/streaming_latency_budget_test.py"
                "::StreamingLatencyBudgetTestCase"
                "::test_cancellation_and_close_latency_budget_across_surfaces"
            ),
        ),
    ),
    "live tool output": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/agent/orchestrator_response_test.py"
                "::OrchestratorResponseCanonicalLifecycleTestCase"
                "::test_iteration_records_live_tool_output_before_completion"
            ),
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_tool_lifecycle_rejects_live_items_after_completion"
            ),
        ),
        integration=(
            (
                "tests/server/protocol_streaming_e2e_test.py"
                "::test_same_canonical_stream_projects_through_protocols"
            ),
            (
                "tests/server/mcp_router_test.py"
                "::MCPRouterAsyncTestCase"
                "::test_stream_response_updates_resource_from_canonical_tool_output"
            ),
            (
                "tests/server/create_response_sse_test.py"
                "::CreateResponseSSEEventsTestCase"
                "::test_streaming_preserves_consecutive_tool_output_metadata"
            ),
        ),
    ),
    "parallel tool ordering": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_tool_lifecycle_assembles_planned_order_observations"
            ),
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_tool_lifecycle_observation_assembly_rejects_planned_mismatches"
            ),
        ),
        integration=(
            (
                "tests/agent/orchestrator_response_test.py"
                "::OrchestratorResponseCanonicalLifecycleTestCase"
                "::test_iteration_parallel_results_emit_before_continuation"
            ),
            (
                "tests/tool/a2a_tool_test.py"
                "::A2ACallToolTestCase"
                "::test_call_streams_answer_tool_and_status_events"
            ),
        ),
    ),
    "cross-protocol projection": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/server/primary_consumer_projection_golden_test.py"
                "::PrimaryConsumerProjectionGoldenTestCase"
                "::test_primary_consumers_project_same_canonical_trace"
            ),
            (
                "tests/server/streaming_conformance_test.py"
                "::test_canonical_trace_conforms_across_public_stream_surfaces"
            ),
            (
                "tests/server/streaming_conformance_test.py"
                "::test_canonical_golden_trace_locks_item_semantics"
            ),
        ),
        integration=(
            (
                "tests/server/protocol_streaming_e2e_test.py"
                "::test_same_canonical_stream_projects_through_protocols"
            ),
            (
                "tests/server/protocol_streaming_e2e_test.py"
                "::test_terminal_outcome_traces_project_through_protocols"
            ),
            (
                "tests/server/protocol_streaming_e2e_test.py"
                "::test_lossy_cli_frames_do_not_drop_lossless_public_surfaces"
            ),
        ),
    ),
    "channel done boundaries": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_normalizers_emit_done_items_at_deterministic_boundaries"
            ),
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_sequence_validator_rejects_channel_boundary_errors"
            ),
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_sequence_validator_rejects_done_before_content"
            ),
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_sequence_validator_rejects_items_after_channel_terminal"
            ),
        ),
        integration=(
            (
                "tests/server/create_response_sse_test.py"
                "::CreateResponseSSEEventsTestCase"
                "::test_streaming_emits_done_events_for_multiple_groups"
            ),
            (
                "tests/server/streaming_conformance_test.py"
                "::test_canonical_trace_rejects_content_after_terminal"
            ),
        ),
    ),
    "event stats/history boundedness": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/event/event_manager_test.py"
                "::EventManagerTestCase"
                "::test_event_stats_instances_are_isolated"
            ),
            (
                "tests/event/event_manager_test.py"
                "::EventManagerTestCase"
                "::test_server_mode_stats_are_explicit_opt_in"
            ),
            (
                "tests/event/event_manager_test.py"
                "::EventManagerTestCase"
                "::test_history_evicts_oversized_events_by_byte_limit"
            ),
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_accumulator_bounds_retained_history_losslessly"
            ),
        ),
        integration=(
            (
                "tests/server/router_streaming_test.py"
                "::RouterStreamingTestCase"
                "::test_default_server_stream_retention_surfaces_are_bounded"
            ),
            (
                "tests/server/a2a_v1_router_test.py"
                "::test_install_a2a_routes_mounts_v1_sdk_routes"
            ),
        ),
    ),
    "reasoning parsing": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/agent/reasoning_parser_split_tag_test.py"
                "::ReasoningParserSplitTagTestCase"
                "::test_split_channel_start_and_end_tags"
            ),
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_local_stream_normalizer_preserves_split_marker_whitespace"
            ),
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_local_stream_normalizer_handles_adjacent_reasoning_sections"
            ),
            (
                "tests/model/model_stream_contract_test.py"
                "::StreamContractTestCase"
                "::test_local_stream_normalizer_closes_unterminated_reasoning"
            ),
        ),
        integration=(
            (
                "tests/model/nlp/vendor_openai_test.py"
                "::OpenAITestCase"
                "::test_canonical_stream_preserves_hosted_reasoning_whitespace"
            ),
            (
                "tests/model/text_generation_response_more_test.py"
                "::TextGenerationResponseMoreTestCase"
                "::test_consumer_projections_preserve_split_reasoning_whitespace"
            ),
            (
                "tests/model/text_generation_response_more_test.py"
                "::TextGenerationResponseMoreTestCase"
                "::test_consumer_projections_handle_adjacent_reasoning_sections"
            ),
        ),
    ),
    "listener-less memory boundedness": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/server/router_streaming_test.py"
                "::RouterStreamingTestCase"
                "::test_default_server_stream_retention_surfaces_are_bounded"
            ),
            (
                "tests/server/a2a_v1_router_test.py"
                "::test_install_a2a_routes_mounts_v1_sdk_routes"
            ),
        ),
        integration=(
            (
                "tests/server/chat_router_unit_test.py"
                "::ChatRouterUnitTest"
                "::test_repeated_chat_stream_requests_bound_no_listener_state"
            ),
            (
                "tests/server/responses_test.py"
                "::ResponsesEndpointTestCase"
                "::test_repeated_requests_without_ui_listener_do_not_retain_events"
            ),
            (
                "tests/server/create_response_sse_test.py"
                "::CreateResponseSSEEventsTestCase"
                "::test_repeated_response_stream_requests_release_sources"
            ),
            (
                "tests/server/mcp_router_test.py"
                "::MCPRouterAsyncTestCase"
                "::test_repeated_stream_responses_bound_shared_resource_store"
            ),
            (
                "tests/tool/a2a_tool_test.py"
                "::A2ACallToolHttpE2ETestCase"
                "::test_calls_sdk_v1_router_and_streams_status"
            ),
        ),
    ),
    "final negative/e2e suites": _FinalGateHarnessEvidence(
        synthetic=(
            (
                "tests/server/streaming_acceptance_inventory_test.py"
                "::StreamingAcceptanceInventoryTestCase"
                "::test_final_negative_e2e_inventory_collects_unskipped_tests"
            ),
            (
                "tests/server/streaming_acceptance_inventory_test.py"
                "::StreamingAcceptanceInventoryTestCase"
                "::test_final_negative_e2e_inventory_rejects_missing_dimension"
            ),
        ),
        integration=(
            (
                "tests/server/protocol_streaming_e2e_test.py"
                "::test_same_canonical_stream_projects_through_protocols"
            ),
            (
                "tests/flow/runtime_test.py"
                "::FlowRuntimeEndToEndTestCase"
                "::test_loaded_definition_routes_cancellation"
            ),
            (
                "tests/server/streaming_latency_budget_test.py"
                "::StreamingLatencyBudgetTestCase"
                "::test_cancellation_and_close_latency_budget_across_surfaces"
            ),
        ),
    ),
}

_COLLECT_SENTINEL = "__STREAMING_ACCEPTANCE_COLLECT__"
_COLLECT_TIMEOUT_SECONDS = 120
_DISALLOWED_MARKERS = frozenset(("skip", "skipif", "xfail"))
_PYTEST_COLLECT_SCRIPT = f"""
from json import dumps
from sys import argv, exit

from pytest import main


class Collector:
    def __init__(self):
        self.items = []

    def pytest_collection_finish(self, session):
        self.items = [
            {{
                "nodeid": item.nodeid,
                "markers": sorted(
                    marker.name for marker in item.iter_markers()
                ),
            }}
            for item in session.items
        ]


collector = Collector()
exit_code = main(["--collect-only", "-q", *argv[1:]], plugins=[collector])
print("{_COLLECT_SENTINEL}" + dumps({{"items": collector.items}}))
exit(exit_code)
"""


@dataclass(frozen=True)
class _CollectedNode:
    node_id: str
    markers: tuple[str, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_complete_inventory(
    inventory: dict[str, tuple[str, ...]],
    required_dimensions: tuple[str, ...] = REQUIRED_ACCEPTANCE_DIMENSIONS,
) -> None:
    assert set(inventory) == set(required_dimensions)
    for dimension, node_ids in inventory.items():
        assert dimension in required_dimensions
        assert node_ids


def _assert_complete_final_gate_inventory(
    inventory: dict[str, _FinalGateHarnessEvidence],
) -> None:
    assert set(inventory) == set(REQUIRED_FINAL_GATE_DIMENSIONS)
    for dimension, evidence in inventory.items():
        assert dimension in REQUIRED_FINAL_GATE_DIMENSIONS
        assert evidence.synthetic
        assert evidence.integration


def _inventory_node_ids(
    inventory: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            node_id for node_ids in inventory.values() for node_id in node_ids
        )
    )


def _final_gate_node_ids(
    inventory: dict[str, _FinalGateHarnessEvidence],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            node_id
            for evidence in inventory.values()
            for node_id in (*evidence.synthetic, *evidence.integration)
        )
    )


def _collect_pytest_nodes(
    node_ids: tuple[str, ...],
) -> tuple[_CollectedNode, ...]:
    completed = run(
        [executable, "-c", _PYTEST_COLLECT_SCRIPT, *node_ids],
        cwd=_repo_root(),
        capture_output=True,
        check=False,
        text=True,
        timeout=_COLLECT_TIMEOUT_SECONDS,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    payload_lines = [
        line[len(_COLLECT_SENTINEL) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_COLLECT_SENTINEL)
    ]
    assert len(payload_lines) == 1, output
    payload = loads(payload_lines[0])
    assert isinstance(payload, dict)
    raw_items = payload.get("items")
    assert isinstance(raw_items, list)
    items: list[_CollectedNode] = []
    for raw_item in raw_items:
        assert isinstance(raw_item, dict)
        node_id = raw_item.get("nodeid")
        markers = raw_item.get("markers")
        assert isinstance(node_id, str)
        assert isinstance(markers, list)
        marker_names: list[str] = []
        for marker in markers:
            assert isinstance(marker, str)
            marker_names.append(marker)
        items.append(
            _CollectedNode(node_id=node_id, markers=tuple(marker_names))
        )
    return tuple(items)


def _assert_exact_collected_nodes(
    requested_node_ids: tuple[str, ...],
    collected_nodes: tuple[_CollectedNode, ...],
) -> None:
    collected_node_ids = tuple(item.node_id for item in collected_nodes)
    assert len(collected_node_ids) == len(requested_node_ids)
    assert set(collected_node_ids) == set(requested_node_ids)


def _assert_no_skip_or_xfail_markers(
    collected_nodes: tuple[_CollectedNode, ...],
) -> None:
    for item in collected_nodes:
        disallowed = sorted(set(item.markers) & _DISALLOWED_MARKERS)
        assert not disallowed, f"{item.node_id}: {disallowed}"


def _assert_collectable_unskipped_node_ids(
    node_ids: tuple[str, ...],
) -> None:
    collected_nodes = _collect_pytest_nodes(node_ids)
    _assert_exact_collected_nodes(node_ids, collected_nodes)
    _assert_no_skip_or_xfail_markers(collected_nodes)


class StreamingAcceptanceInventoryTestCase(TestCase):
    def test_streaming_acceptance_inventory_covers_required_dimensions(
        self,
    ) -> None:
        _assert_complete_inventory(ACCEPTANCE_HARNESS_TESTS)

    def test_streaming_acceptance_inventory_collects_unskipped_tests(
        self,
    ) -> None:
        _assert_collectable_unskipped_node_ids(
            _inventory_node_ids(ACCEPTANCE_HARNESS_TESTS)
        )

    def test_streaming_acceptance_inventory_rejects_missing_dimension(
        self,
    ) -> None:
        inventory = dict(ACCEPTANCE_HARNESS_TESTS)
        del inventory["stream equivalence"]

        with self.assertRaises(AssertionError):
            _assert_complete_inventory(inventory)

    def test_streaming_acceptance_inventory_rejects_uncollectable_node(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            _assert_collectable_unskipped_node_ids(
                (
                    (
                        "tests/server/streaming_conformance_test.py"
                        "::test_missing_acceptance_harness"
                    ),
                )
            )

    def test_streaming_acceptance_inventory_rejects_skip_or_xfail_marker(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            _assert_no_skip_or_xfail_markers(
                (
                    _CollectedNode(
                        node_id="tests/example_test.py::test_example",
                        markers=("skip",),
                    ),
                )
            )

    def test_streaming_hardening_inventory_covers_required_dimensions(
        self,
    ) -> None:
        _assert_complete_inventory(
            HARDENING_COVERAGE_TESTS,
            REQUIRED_HARDENING_DIMENSIONS,
        )

    def test_streaming_hardening_inventory_collects_unskipped_tests(
        self,
    ) -> None:
        _assert_collectable_unskipped_node_ids(
            _inventory_node_ids(HARDENING_COVERAGE_TESTS)
        )

    def test_streaming_hardening_inventory_rejects_missing_dimension(
        self,
    ) -> None:
        inventory = dict(HARDENING_COVERAGE_TESTS)
        del inventory["malformed tool calls"]

        with self.assertRaises(AssertionError):
            _assert_complete_inventory(
                inventory,
                REQUIRED_HARDENING_DIMENSIONS,
            )

    def test_final_negative_e2e_inventory_covers_required_dimensions(
        self,
    ) -> None:
        _assert_complete_inventory(
            FINAL_NEGATIVE_E2E_SUITE_TESTS,
            REQUIRED_FINAL_NEGATIVE_E2E_DIMENSIONS,
        )

    def test_final_negative_e2e_inventory_collects_unskipped_tests(
        self,
    ) -> None:
        _assert_collectable_unskipped_node_ids(
            _inventory_node_ids(FINAL_NEGATIVE_E2E_SUITE_TESTS)
        )

    def test_final_negative_e2e_inventory_rejects_missing_dimension(
        self,
    ) -> None:
        inventory = dict(FINAL_NEGATIVE_E2E_SUITE_TESTS)
        del inventory["canonical validation"]

        with self.assertRaises(AssertionError):
            _assert_complete_inventory(
                inventory,
                REQUIRED_FINAL_NEGATIVE_E2E_DIMENSIONS,
            )

    def test_final_gate_inventory_covers_required_dimensions(self) -> None:
        _assert_complete_final_gate_inventory(FINAL_GATE_ACCEPTANCE_HARNESSES)

    def test_final_gate_inventory_collects_unskipped_tests(self) -> None:
        _assert_collectable_unskipped_node_ids(
            _final_gate_node_ids(FINAL_GATE_ACCEPTANCE_HARNESSES)
        )

    def test_final_gate_inventory_rejects_missing_dimension(self) -> None:
        inventory = dict(FINAL_GATE_ACCEPTANCE_HARNESSES)
        del inventory["hosted cleanup"]

        with self.assertRaises(AssertionError):
            _assert_complete_final_gate_inventory(inventory)

    def test_final_gate_inventory_rejects_missing_synthetic_harness(
        self,
    ) -> None:
        inventory = dict(FINAL_GATE_ACCEPTANCE_HARNESSES)
        evidence = inventory["SDK losslessness"]
        inventory["SDK losslessness"] = _FinalGateHarnessEvidence(
            synthetic=(),
            integration=evidence.integration,
        )

        with self.assertRaises(AssertionError):
            _assert_complete_final_gate_inventory(inventory)

    def test_final_gate_inventory_rejects_missing_integration_harness(
        self,
    ) -> None:
        inventory = dict(FINAL_GATE_ACCEPTANCE_HARNESSES)
        evidence = inventory["SDK losslessness"]
        inventory["SDK losslessness"] = _FinalGateHarnessEvidence(
            synthetic=evidence.synthetic,
            integration=(),
        )

        with self.assertRaises(AssertionError):
            _assert_complete_final_gate_inventory(inventory)
