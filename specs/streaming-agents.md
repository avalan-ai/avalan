# Streaming Agent Coordination

This file tracks delegated agent work for Phase 7 of
`specs/streaming-agenda.md`.

## Phase 7 Runs

### Run 1: 7.2 Protocol Classifier Cleanup

- Status: Complete; awaiting user confirmation before next slice.
- Worker: Russell (`019ecd52-8788-7693-ad21-9c6348580fcd`), launched.
- Reviewer: Pauli (`019ecd5a-819b-7aa2-b22d-a0cbd023fc42`), completed.
- Task: Replace a remaining internal protocol classifier or duplicate
  streaming state-machine path with canonical stream adapters/shared
  accumulator behavior, add focused positive and negative tests, preserve
  100% coverage, and amend changes to HEAD without changing the commit
  message.
- Notes: Removed the A2A-local `StreamState` classifier and switched
  artifact boundary handling to canonical stream projection channels while
  keeping the shared accumulator as the terminal/order validator. Changed
  `src/avalan/server/a2a/router.py` and
  `tests/server/a2a_router_unit_extra_test.py`. Focused tests run:
  `poetry run pytest --verbose -s tests/server/a2a_router_unit_extra_test.py
  tests/server/test_a2a.py tests/server/protocol_streaming_e2e_test.py
  tests/server/streaming_conformance_test.py` (63 passed). Next slice:
  replace the Responses SSE `ResponseState` classifier with canonical
  stream projection/adaptor state.
- Review feedback: A2A tool streams can fail on raw legacy `Event` items
  emitted by `OrchestratorResponse` because the fallback path sends every
  non-canonical item through `canonical_item_from_token`, which rejects
  `Event`. Follow-up requested from the worker.
- Follow-up: Added live `OrchestratorResponse.consumer_projections()`
  backed by canonical items, with display-token projection through the
  canonical token adapter and existing canonical tool lifecycle items for
  legacy tool events. Updated the temporary classifier inventory entry for
  the new helper. Changed
  `src/avalan/agent/orchestrator/response/orchestrator_response.py`,
  `src/avalan/model/stream.py`, `tests/server/test_a2a.py`, and
  `tests/model/text_generation_response_full_test.py`. Agenda checkboxes
  unchanged. Tests run: `make lint`; `poetry run pytest --verbose -s
  tests/server/test_a2a.py tests/server/a2a_router_unit_extra_test.py
  tests/server/protocol_streaming_e2e_test.py
  tests/server/streaming_conformance_test.py
  tests/agent/orchestrator_response_test.py
  tests/cli/model_test.py::CliRenderFrameTestCase::test_token_stream_start_thinking_orchestrator_response`
  (155 passed); `poetry run pytest --verbose -s` (6179 passed, 42
  skipped); `make test-coverage` (all source files 100%). Next slice:
  replace the Responses SSE `ResponseState` classifier with canonical
  stream projection/adaptor state.
- Second review feedback: `OrchestratorResponse.consumer_projections()`
  drives the stream through `__anext__()`, but `_next_item()` drops
  canonical non-answer items. A2A now prefers the projection API, so an
  `OrchestratorResponse` wrapping canonical reasoning, usage, tool, done,
  or control items can silently lose those artifacts while completing.
  Follow-up requested from the worker.
- Second follow-up: Preserved canonical non-answer items from
  `OrchestratorResponse` projection streams by appending inner canonical
  items to the wrapper's canonical stream with wrapper stream identity,
  sequence, channel, and terminal closure handling. Kept the prior legacy
  tool-event projection fix in place. Changed
  `src/avalan/agent/orchestrator/response/orchestrator_response.py` and
  `tests/server/test_a2a.py`. Agenda checkboxes unchanged. Tests run:
  `make lint`; `poetry run pytest --verbose -s tests/server/test_a2a.py
  tests/server/a2a_router_unit_extra_test.py
  tests/server/protocol_streaming_e2e_test.py
  tests/server/streaming_conformance_test.py
  tests/agent/orchestrator_response_test.py
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source`
  (156 passed, 13 subtests passed); `poetry run pytest --verbose -s`
  (6180 passed, 42 skipped, 3247 subtests passed); `make test-coverage`
  (all source files 100%). Next slice: replace the Responses SSE
  `ResponseState` classifier with canonical stream projection/adaptor
  state.
- Third review feedback: Wrapped canonical streams preserve the
  `usage.completed` item but later overwrite accumulated final usage with
  synthesized terminal usage when the inner terminal has no usage. The
  regression test needs to assert the usage value survives, and the
  follow-up note should not overstate terminal usage handling until fixed.
- Third follow-up: Kept wrapper terminals from synthesizing fallback usage
  after an explicit `usage.completed` item, so the shared accumulator's
  final usage remains the inner canonical usage value. Strengthened the
  A2A regression to assert the wrapped projection terminal carries no
  fallback usage and that projection/A2A accumulators retain
  `{"output_tokens": 1}`. Changed
  `src/avalan/agent/orchestrator/response/orchestrator_response.py` and
  `tests/server/test_a2a.py`. Agenda checkboxes unchanged. Tests run:
  `make lint`; `poetry run pytest --verbose -s tests/server/test_a2a.py
  tests/server/a2a_router_unit_extra_test.py
  tests/server/protocol_streaming_e2e_test.py
  tests/server/streaming_conformance_test.py
  tests/agent/orchestrator_response_test.py
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source`
  (156 passed, 13 subtests passed); `poetry run pytest --verbose -s`
  (6180 passed, 42 skipped, 3247 subtests passed); `make test-coverage`
  (all source files 100%). Next slice: replace the Responses SSE
  `ResponseState` classifier with canonical stream projection/adaptor
  state.
- Fourth review: No findings. Verified explicit `usage.completed` values
  survive through `OrchestratorResponse.consumer_projections()` and A2A
  accumulation, canonical non-answer items remain preserved, and the
  legacy `Event` tool-stream path still holds. Follow-up is not required.

### Run 2: 7.2 Responses SSE Classifier Cleanup

- Status: Complete; awaiting user confirmation before next slice.
- Worker: Hooke (`019ecd90-7acb-7b71-8d47-782269eebb27`), launched.
- Reviewer: Maxwell (`019ecd9b-b1da-75a3-b953-b8181597b26f`),
  completed.
- Task: Replace the Responses SSE `ResponseState` classifier with
  canonical stream projection/adaptor state, add focused positive and
  negative tests, preserve 100% coverage, and amend changes to HEAD
  without changing the commit message.
- Notes: Removed the public Responses SSE `ResponseState` enum and the
  `_new_state`/`_is_tool_response_state` helpers, replacing them with
  private projection adapter state derived from `StreamConsumerProjection`
  kinds and tool-call correlation. The Responses SSE loop now uses the
  adapter for answer, reasoning, function-call, custom-tool-call, and
  terminal boundary events while preserving existing delta, usage,
  diagnostic, tool execution, cancellation, and failure behavior. Updated
  the conformance/golden tests to exercise the adapter path directly and
  added focused positive/negative unit coverage for adapter state and
  protocol id handling. Changed
  `src/avalan/server/routers/responses.py`,
  `tests/server/responses_utils_test.py`,
  `tests/server/create_response_sse_test.py`,
  `tests/server/primary_consumer_projection_golden_test.py`, and
  `tests/server/streaming_conformance_test.py`. Agenda checkboxes
  unchanged. Tests run: `make lint`; `poetry run pytest --verbose -s
  tests/server/responses_utils_test.py tests/server/create_response_sse_test.py
  tests/server/primary_consumer_projection_golden_test.py
  tests/server/streaming_conformance_test.py` (62 passed, 4 subtests
  passed); `poetry run pytest --verbose -s` (6183 passed, 42 skipped,
  3247 subtests passed); `make test-coverage` (all source files 100%).
  Next slice: migrate MCP `_MCPStreamProjectionState` and its legacy item
  mapper toward canonical stream adapters/shared accumulator behavior.
- Review feedback: Consecutive `response.tool_execution.output` events can
  coalesce while dropping later structured payloads because the coalescer
  concatenates deltas but keeps the first event data. Follow-up requested
  from the worker.
- Follow-up: Made Responses SSE tool-output coalescing skip events with
  structured `data`, preserving per-chunk stdout/stderr category and other
  metadata while leaving text/reasoning/custom-tool deltas coalescible.
  Added direct coalescer coverage and a Responses SSE streaming regression
  with consecutive tool-output chunks carrying different `data`. Changed
  `src/avalan/server/routers/responses.py`,
  `tests/server/responses_utils_test.py`, and
  `tests/server/create_response_sse_test.py`. Agenda checkboxes unchanged.
  Tests run: `poetry run pytest --verbose -s
  tests/server/responses_utils_test.py tests/server/create_response_sse_test.py`
  (60 passed, 4 subtests passed); `make lint`; `make test-coverage` (all
  source files 100%); `poetry run pytest --verbose -s` (6185 passed, 42
  skipped, 3247 subtests passed). Next slice remains: migrate MCP
  `_MCPStreamProjectionState` and its legacy item mapper toward canonical
  stream adapters/shared accumulator behavior.
- Second review: No findings. Verified structured tool-output chunks no
  longer coalesce and lose per-chunk metadata, and the Responses SSE
  adapter path still preserves answer, reasoning, tool-call, usage, and
  terminal behavior. Follow-up is not required.

### Run 3: 7.2 MCP Projection Cleanup

- Status: Complete; continuing to next requested slice.
- Worker: Tesla (`019ecdb0-ba12-7d31-9720-1d56ba38606a`), launched.
- Reviewer: Newton (`019ecdba-7d7f-7ad2-b993-bbaf9bb4d389`),
  completed.
- Task: Migrate MCP `_MCPStreamProjectionState` and its legacy item mapper
  toward canonical stream adapters/shared accumulator behavior, add
  focused positive and negative tests, preserve 100% coverage, and amend
  changes to HEAD without changing the commit message.
- Notes: Added `_MCPLegacyStreamAdapter` so MCP legacy `Token`/string and
  tool `Event` inputs are confined to one temporary canonical-item mapper,
  then simplified `_MCPStreamProjectionState` to compose that adapter with
  `ProtocolStreamProjectionState` and the shared protocol accumulator.
  Removed MCP-local legacy sequence/start synthesis from the notification
  loop while preserving answer text, reasoning, tool calls, tool output
  resources, usage fallback, terminal events, cancellation, and error
  behavior. Updated the legacy classifier inventory with explicit MCP
  removal criteria. Added focused positive/negative tests for adapter
  token projection and malformed tool events. Changed
  `src/avalan/server/routers/mcp.py`, `src/avalan/model/stream.py`, and
  `tests/server/mcp_router_test.py`. Agenda checkboxes unchanged. Tests
  run: `poetry run pytest --verbose -s tests/server/mcp_router_test.py
  tests/server/protocol_streaming_e2e_test.py
  tests/server/streaming_conformance_test.py
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source`
  (132 passed, 16 subtests passed); `make lint`; `poetry run pytest
  --verbose -s` (6187 passed, 42 skipped, 3248 subtests passed);
  `make test-coverage` (all source files 100%). Next slice: continue
  Phase 7.2 by replacing the next remaining protocol/UI projection
  classifier or reconstruction path, starting with the A2A/CLI/flow
  legacy projection state still visible in the classifier inventory and
  source scan.
- Review feedback: Legacy MCP `ToolCallToken(call=...)` metadata can be
  dropped because canonical tool notifications ignore preserved call
  metadata and return nothing for empty deltas. The MCP adapter inventory
  also under-documents handled string and tool `Event` legacy surfaces.
  Follow-up requested from the worker.
- Follow-up: Preserved canonical tool-call metadata in MCP notification
  output by making `_canonical_tool_notification()` include
  name/arguments on non-empty input deltas and emit a `tool.call`
  notification for empty argument deltas that carry call metadata. Kept
  the handling inside the MCP adapter/projection path and made the adapter
  explicitly classify `Event` inputs so the legacy classifier inventory
  now documents string, token, and event surfaces. Added regressions for
  `ToolCallToken(token="", call=call)` and data-bearing tool argument
  notifications. Changed `src/avalan/server/routers/mcp.py`,
  `src/avalan/model/stream.py`, `tests/server/mcp_router_test.py`, and
  `specs/streaming-agents.md`. Agenda checkboxes unchanged. Tests run:
  `poetry run pytest --verbose -s
  tests/server/mcp_router_test.py::MCPUtilityTestCase::test_canonical_tool_notification_variants
  tests/server/mcp_router_test.py::MCPRouterAsyncTestCase::test_stream_item_notifications_project_empty_tool_call_token
  tests/server/mcp_router_test.py::MCPRouterAsyncTestCase::test_stream_item_notifications_project_legacy_tokens
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source`
  (4 passed, 14 subtests passed); `poetry run pytest --verbose -s
  tests/server/mcp_router_test.py tests/server/protocol_streaming_e2e_test.py
  tests/server/streaming_conformance_test.py
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source`
  (133 passed, 16 subtests passed); `make lint`; `poetry run pytest
  --verbose -s` (6188 passed, 42 skipped, 3248 subtests passed);
  `make test-coverage` (all source files 100%). Next slice remains:
  continue Phase 7.2 by replacing the next remaining protocol/UI
  projection classifier or reconstruction path, starting with the
  A2A/CLI/flow legacy projection state still visible in the classifier
  inventory and source scan.
- Second review: No findings. Verified MCP legacy token/string/Event
  handling is confined to the adapter path, tool-call metadata is
  preserved for non-empty and empty argument deltas, and the inventory now
  documents the handled legacy surfaces. Follow-up is not required.

### Run 4: 7.2 CLI/Flow Projection Cleanup

- Status: Complete.
- Worker: Halley (`019ecdcb-6cbc-70f1-9bcc-312416ca51db`), launched.
- Reviewer: Singer (`019ecdd5-6bfc-76a2-9610-535d615b19f5`),
  completed.
- Task: Replace the next remaining protocol/UI projection classifier or
  reconstruction path, starting with CLI/flow legacy projection state
  visible in the classifier inventory and source scan, with canonical
  stream adapters/shared accumulator behavior. Add focused positive and
  negative tests, preserve 100% coverage, and amend changes to HEAD
  without changing the commit message.
- Notes: Removed the inventory-backed `FancyTheme.tokens` `TokenDetail`
  classifier by selecting probability-rendering display tokens from
  canonical display-token metadata structurally instead of branching on the
  legacy token-detail class. Removed the corresponding legacy classifier
  inventory entry. Added positive coverage for a non-`TokenDetail` display
  token carrying alternative-token metadata and negative coverage for a
  plain display token without that metadata.
- Files changed:
  `src/avalan/cli/theme/fancy.py`, `src/avalan/model/stream.py`,
  `tests/cli/theme_fancy_test.py`, `specs/streaming-agents.md`.
- Tests:
  `poetry run pytest tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source -q`
  (passed);
  `poetry run pytest tests/cli/theme_fancy_test.py::FancyThemeTokensTestCase -q`
  (passed);
  `poetry run pytest tests/cli/theme_fancy_test.py tests/model/model_stream_contract_test.py -q`
  (186 passed, 185 subtests passed);
  `poetry run pytest tests/cli/model_test.py::CliTokenGenerationTestCase -q`
  (37 passed, 2 subtests passed);
  `make lint` (passed);
  `make tests` (6190 passed, 42 skipped, 3247 subtests passed);
  `make test-coverage` (all source files 100%).
- Next slice: replace or confine the CLI flow progress monitor's
  `token_generated` `token_type` fallback once the event producer exposes
  canonical stream observability for those token events, or explicitly
  document that fallback as a temporary compatibility shim with removal
  criteria.
- Review: No findings. Verified `FancyTheme.tokens` no longer branches on
  `TokenDetail`, probability rendering uses structural display-token
  metadata, malformed projection metadata is filtered, and the classifier
  inventory still matches source. Follow-up is not required.

### Run 5: 7.2 CLI Flow Progress Fallback Cleanup

- Status: Complete.
- Worker: Beauvoir (`019ecde4-850c-7822-88ad-e277cedf4031`),
  launched.
- Reviewer: Parfit (`019ecdf3-3830-7b70-b8c2-c69d3fe11210`),
  completed.
- Task: Replace or explicitly confine the CLI flow progress monitor's
  `token_generated` `token_type` fallback with canonical stream
  observability or a documented temporary shim with removal criteria. Add
  focused positive and negative tests, preserve 100% coverage, and amend
  changes to HEAD without changing the commit message.
- Notes: Replaced the CLI flow progress monitor's reasoning/output
  classification from `token_type` with canonical stream kind metadata.
  Token events emitted by `OrchestratorResponse` now carry canonical stream
  observability metadata, and task event sanitization preserves safe
  canonical stream metadata under `canonical_stream` without retaining raw
  token text.
  Legacy `token_type` by itself remains output-only fallback behavior for
  missing stream metadata.
- Files changed:
  `src/avalan/agent/orchestrator/response/orchestrator_response.py`,
  `src/avalan/cli/commands/flow.py`, `src/avalan/task/event.py`,
  `src/avalan/task/privacy.py`,
  `tests/agent/default_orchestrator_test.py`,
  `tests/agent/orchestrator_response_additional_test.py`,
  `tests/agent/orchestrator_response_test.py`,
  `tests/cli/flow_command_test.py`, `tests/task/event_test.py`,
  `tests/task/privacy_test.py`, `specs/streaming-agents.md`.
- Tests:
  `poetry run pytest tests/agent/orchestrator_response_test.py::OrchestratorResponseIterationTestCase::test_consumer_projections_align_token_event_sequences tests/agent/orchestrator_response_test.py::OrchestratorResponseIterationTestCase::test_consumer_projections_align_canonical_answer_event_sequence tests/agent/orchestrator_response_test.py::OrchestratorResponseIterationTestCase::test_projection_rejects_mismatched_token_event_sequence tests/agent/orchestrator_response_test.py::OrchestratorResponseIterationTestCase::test_to_str_emits_streamed_token_events tests/agent/orchestrator_response_additional_test.py::OrchestratorResponseAdditionalCoverageTestCase::test_response_text_and_calls_skips_events -q`
  (5 passed);
  `poetry run pytest tests/cli/flow_command_test.py tests/task/event_test.py tests/task/privacy_test.py tests/agent/orchestrator_response_test.py tests/agent/orchestrator_response_additional_test.py tests/agent/default_orchestrator_test.py tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source -q`
  (273 passed, 91 subtests passed);
  `make lint` (passed);
  `make tests` (6195 passed, 42 skipped, 3247 subtests passed);
  `make test-coverage` (all source files 100%).
- Next slice: continue Phase 7.2 with the next remaining public legacy
  projection or protocol/UI classifier visible in the current inventory and
  source scan; do not move into later phases.
- Review feedback: Token event canonical sequence can drift from emitted
  projection sequence because synthetic observability uses
  `_canonical_sequence + _step`; two legacy string tokens produced
  projection sequences `1, 2` but task event canonical sequences `1, 3`.
  The Run 5 note also overstates sanitized metadata as preserving
  summaries when only safe canonical stream metadata is retained.
  Follow-up requested from the worker.
- Follow-up: Fixed token event canonical stream observability to use the
  same current canonical sequence consumed by legacy token projections.
  Added a two-token `consumer_projections()` regression that compares
  projected `answer.delta` sequences with token event `canonical_stream`
  sequences, and corrected the Run 5 sanitization wording.
- Second review feedback: `to_str()`/`_response_text_and_calls()` can
  emit duplicate token event canonical sequences for multiple legacy
  tokens because no token deltas are appended to `_canonical_items` on that
  path. Canonical `answer.delta` inputs can still drift because the answer
  item is appended before `_emit()` creates the token event. Follow-up
  requested from the worker.
- Second follow-up: Token events now reserve canonical sequence positions
  on event-only text paths and reuse matching reserved token items when
  `consumer_projections()` appends legacy token projections. Canonical
  `answer.delta` inputs pass the already-appended sequence into token event
  emission. Added regressions for legacy `to_str()`,
  `_response_text_and_calls()`, canonical answer projection, and a negative
  mismatched-token reservation case.
- Third review: No findings. Verified token event canonical sequences
  align with emitted projections across legacy projection, canonical
  answer, and event-only paths without terminal/control collisions. The
  CLI flow progress classification and safe metadata sanitization remain
  intact. Follow-up is not required.

### Run 6: 7.2 TextGenerationResponse Projection Cleanup

- Status: Complete.
- Worker: Linnaeus (`019ece14-dfa9-7fd3-bc28-bc989dfbfa65`),
  launched.
- Reviewer: Kuhn (`019ece1f-89be-7101-873d-a1e303a853a2`),
  completed.
- Task: Reduce or explicitly confine the `TextGenerationResponse` legacy
  public iteration/reconstruction paths in the classifier inventory by
  deriving behavior from canonical stream items/projections and the shared
  accumulator where practical. Add focused positive and negative tests,
  preserve 100% coverage, and amend changes to HEAD without changing the
  commit message.
- Notes: Removed the
  `TextGenerationResponse._ensure_non_stream_prefetched` legacy token
  classifier by deriving non-stream `Token`/`TokenDetail` text through
  `canonical_item_from_token()` and `TextGenerationSingleStream` final text
  through its canonical accumulator. Removed the corresponding classifier
  inventory entry and added TokenDetail prefetch coverage while keeping the
  generic-result fallback behavior.
- Files changed:
  `src/avalan/model/response/text.py`, `src/avalan/model/stream.py`,
  `tests/model/text_generation_response_non_stream_test.py`.
- Tests run:
  `poetry run pytest tests/model/text_generation_response_non_stream_test.py tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source -q`
  (10 passed, 12 subtests passed);
  `poetry run pytest tests/model/text_generation_response_non_stream_test.py tests/model/text_generation_response_more_test.py tests/model/text_generation_response_golden_trace_test.py tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source -q`
  (71 passed, 30 subtests passed);
  `poetry run pytest tests/model/text_generation_response_non_stream_test.py tests/model/text_generation_response_more_test.py tests/model/text_generation_response_golden_trace_test.py --cov=avalan.model.response.text --cov-report=term-missing -q`
  (70 passed, 18 subtests passed, targeted subset reported 95% for
  `src/avalan/model/response/text.py`);
  `make test-coverage -- -100 src/avalan/model/response` (no files below
  100% reported);
  `poetry run ruff check src/avalan/model/response/text.py src/avalan/model/stream.py tests/model/text_generation_response_non_stream_test.py`
  (passed);
  `make lint` (passed);
  `poetry run pytest --verbose -s` (6197 passed, 42 skipped, 3246 subtests
  passed).
- Next slice: Confine or replace the remaining
  `TextGenerationResponse.__aiter__`/`__anext__` public legacy iteration
  shim, keeping `async for` compatibility while moving final text,
  terminal-state, usage, reasoning, and tool-call reconstruction toward
  canonical projections/shared accumulator behavior.
- Review feedback: `TextGenerationResponse.__aiter__()` carries
  parser-derived `is_thinking` across `_reset_iteration_state()`, so a
  fresh iterator after partial `<think>` consumption can emit the restarted
  first answer token as a `ReasoningToken`. Follow-up requested from the
  worker with a focused restart-after-reasoning regression.
- Follow-up: Split manual thinking state from parser-derived thinking state.
  `set_thinking()` now records an explicit manual flag and iterator resets
  reapply only that flag, so partial reasoning-marker consumption no longer
  leaks into a fresh iterator. Added regression coverage that partially
  consumes into a reasoning span, restarts iteration, and verifies the
  restarted first token remains answer text while the existing manual
  `set_thinking(True)` restart behavior is preserved.
- Follow-up files changed: `src/avalan/model/response/text.py`,
  `tests/model/text_generation_response_full_test.py`.
- Follow-up tests run:
  `poetry run pytest tests/model/text_generation_response_full_test.py -q`
  (9 passed);
  `poetry run pytest tests/model/text_generation_response_full_test.py tests/model/text_generation_response_non_stream_test.py tests/model/text_generation_response_more_test.py tests/model/text_generation_response_golden_trace_test.py tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source -q`
  (80 passed, 30 subtests passed);
  `make lint` (passed);
  `make test-coverage -- -100 src/avalan/model/response` (reported
  unrelated `src/avalan/model/response/parsers/tool.py: 92%`);
  `poetry run pytest --cov=avalan.model.response.text --cov-report=term-missing -q`
  (6198 passed, 42 skipped, 3246 subtests passed;
  `src/avalan/model/response/text.py` at 100%).
- Second review: No findings. Verified iterator reset now preserves only
  explicit manual thinking state, restart-after-parser-derived reasoning is
  covered, the non-stream prefetch cleanup remains canonicalized, and a
  rerun of `make test-coverage -- -100 src/avalan/model/response` produced
  no below-100 output. Follow-up is not required.

### Run 7: 7.2 TextGenerationResponse Iteration Shim Cleanup

- Status: Complete.
- Worker: Pasteur (`019ece2e-74dc-7383-b5d8-0974e9b05945`),
  launched.
- Reviewer: Ampere (`019ece39-7c13-7fc2-96bc-fd0be8883108`),
  completed.
- Task: Confine or replace the remaining
  `TextGenerationResponse.__aiter__`/`__anext__` public legacy iteration
  shim while preserving `async for` compatibility and moving final text,
  terminal-state, usage, reasoning, and tool-call reconstruction toward
  canonical projections/shared accumulator behavior. Add focused positive
  and negative tests, preserve 100% coverage, and amend changes to HEAD
  without changing the commit message.
- Notes: Kept the public `async for` legacy shape but confined it behind a
  single iterator-level canonical accumulator. Legacy tokens, parsed
  reasoning tokens, and tool-call tokens now update that accumulator as
  they are returned, and iterator exhaustion finalizes terminal state,
  usage, and final text through the same accumulator used by `to_str()`.
  `to_str()` no longer rebuilds a second legacy accumulator, and semantic
  stream usage now wins over provider fallback usage after canonical usage
  has been observed. Reclassified the remaining
  `TextGenerationResponse.__aiter__`/`__anext__` inventory entries as
  temporary compatibility shims with public SDK removal criteria. Agenda
  checkboxes unchanged.
- Files changed:
  `src/avalan/model/response/text.py`, `src/avalan/model/stream.py`,
  `tests/model/text_generation_response_more_test.py`,
  `specs/streaming-agents.md`.
- Tests run:
  `poetry run pytest tests/model/text_generation_response_more_test.py
  tests/model/text_generation_response_full_test.py
  tests/model/text_generation_response_non_stream_test.py
  tests/model/text_generation_response_golden_trace_test.py
  tests/model/text_generation_response_additional_test.py
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source
  -q` (85 passed, 30 subtests passed);
  `poetry run pytest --cov=avalan.model.response.text
  --cov-report=term-missing
  tests/model/text_generation_response_more_test.py
  tests/model/text_generation_response_full_test.py
  tests/model/text_generation_response_non_stream_test.py
  tests/model/text_generation_response_golden_trace_test.py
  tests/model/text_generation_response_additional_test.py -q` (84 passed,
  18 subtests passed; `src/avalan/model/response/text.py` at 100%);
  `make lint` (passed);
  `poetry run pytest --verbose -s` (6200 passed, 42 skipped, 3246
  subtests passed);
  `make test-coverage` (all source files 100%).
- Next slice: continue Phase 7.2 by replacing or confining the remaining
  inventory-backed `model.stream` legacy token classifiers
  (`stream_consumer_projection_from_token`, `normalize_local_stream.events`,
  `token_text`, `canonical_item_from_token`, and `_token_metadata`) without
  moving into retention or hot-path hardening work.
- Review: No findings. Verified the iterator-level canonical accumulator
  records legacy, reasoning, tool-call, canonical, and projection
  iteration consistently, `to_str()` reuses that state, and the inventory
  reclassification is accurate. Follow-up is not required.

### Run 8: 7.2 Model Stream Classifier Cleanup

- Status: Complete.
- Worker: Sartre (`019ece3e-6730-7f73-8c38-1bd5ed416e14`),
  launched.
- Reviewer: James (`019ece4c-1c2f-7b43-b781-5610331de809`),
  completed.
- Task: Replace or explicitly confine the remaining inventory-backed
  `model.stream` legacy token classifiers:
  `stream_consumer_projection_from_token`, `normalize_local_stream.events`,
  `token_text`, `canonical_item_from_token`, and `_token_metadata`. Add
  focused positive and negative tests, preserve 100% coverage, and amend
  changes to HEAD without changing the commit message.
- Notes: Confined legacy token canonicalization behind
  `_LegacyTokenStreamAdapter.item_from_token`, kept local raw-string parsing
  in `_LegacyTokenStreamAdapter.events_from_token` as an explicit temporary
  ingestion shim, updated `StreamProjectionState` and
  `normalize_local_stream` to consume adapter canonical item/provider-event
  output, and removed `stream_consumer_projection_from_token`,
  `normalize_local_stream.events`, `token_text`, `canonical_item_from_token`,
  and `_token_metadata` from the legacy classifier inventory. Preserved
  answer text, reasoning visibility, tool-call argument deltas, tool-call
  ready/done boundaries, display-token metadata, provider metadata,
  sequence/order validation, terminal error handling, and legacy/canonical
  mixing guardrails. Added positive and negative contract coverage for the
  confined adapter, metadata shim, local tool-boundary behavior, invalid
  local legacy input, and inventory guardrails. Changed
  `src/avalan/model/stream.py` and
  `tests/model/model_stream_contract_test.py`. Agenda checkboxes unchanged.
  Tests run: `poetry run pytest
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_maps_legacy_tokens_losslessly
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_marks_complete_legacy_tool_calls
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_closes_complete_tool_calls_before_next
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_reports_invalid_legacy_tokens
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_consumer_projection_from_legacy_tool_call_token_preserves_call
  -q` (6 passed, 13 subtests passed);
  `poetry run pytest tests/model/model_stream_contract_test.py -q` (111
  passed, 178 subtests passed); `poetry run pytest
  tests/model/text_generation_response_golden_trace_test.py
  tests/model/text_generation_response_non_stream_test.py
  tests/model/text_generation_response_more_test.py -q` (72 passed, 18
  subtests passed); `make lint`; `poetry run pytest --verbose -s` (6202
  passed, 42 skipped, 3247 subtests passed); `make test-coverage` (all
  source files 100%). Next slice: continue Phase 7.2 with the remaining
  `model.stream` temporary ingestion shims, then the public SDK iteration
  compatibility shim or the orchestrator response compatibility classifiers.
- Review feedback: `_LegacyTokenStreamAdapter.events_from_token()` still
  directly branches on legacy string input while only
  `_LegacyTokenStreamAdapter.item_from_token` is inventoried, so the Run 8
  confinement and next-slice notes overclaim that all remaining
  `model.stream` classifiers are behind `item_from_token`. Follow-up
  requested from the worker.
- Follow-up: Inventoried `_LegacyTokenStreamAdapter.events_from_token` as
  the remaining local raw-string ingestion shim with owner/removal criteria
  and updated the source-scan guard to detect that string surface without
  treating generic string validation as a stream classifier. Updated the
  Run 8 notes and next-slice wording so the remaining `model.stream` shim
  is explicit. Changed `src/avalan/model/stream.py`,
  `tests/model/model_stream_contract_test.py`, and
  `specs/streaming-agents.md`. Agenda checkboxes unchanged. Tests run:
  `poetry run pytest
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_guard_detects_tracked_string_classifiers
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_parses_split_reasoning_tags
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_closes_complete_tool_calls_before_next
  -q` (4 passed, 14 subtests passed); `poetry run pytest
  tests/model/model_stream_contract_test.py -q` (112 passed, 179 subtests
  passed); `make lint`; `poetry run pytest --verbose -s` (6203 passed, 42
  skipped, 3248 subtests passed); `make test-coverage` (all source files
  100%).
- Second review: No findings. Verified
  `_LegacyTokenStreamAdapter.events_from_token` is inventoried as a local
  raw-string ingestion shim, the source guard catches that classifier
  without generic string-validation noise, Run 8 wording no longer
  overclaims full `item_from_token` confinement, and local parsing behavior
  remains intact. Follow-up is not required.

### Run 9: 7.2 Orchestrator Compatibility Cleanup

- Status: Complete.
- Worker: Ohm (`019ece5b-57c6-7051-a37f-1726bc6d8c7d`), launched.
- Reviewer: Gibbs (`019ece6a-b9c0-7683-88cb-323d8dd99feb`),
  completed.
- Task: Replace or explicitly confine the remaining
  `OrchestratorResponse` compatibility classifiers
  (`_next_item`, `_response_text_and_calls`,
  `_append_canonical_projection_item`, and `_emit`) by deriving behavior
  from canonical tool lifecycle items/projections and shared accumulator
  state where practical. Add focused positive and negative tests, preserve
  100% coverage, and amend changes to HEAD without changing the commit
  message.
- Notes: Removed the four `OrchestratorResponse` method-level classifier
  inventory entries by routing `_next_item`, `_response_text_and_calls`,
  `_append_canonical_projection_item`, and `_emit` through a single
  `_stream_item_projection` compatibility shim. The shim maps canonical
  items/projections through canonical helpers, confines legacy string,
  `ToolCallToken`, and `Event` handling with owner/removal criteria, and
  leaves tool-call projection output derived from the existing canonical
  tool lifecycle items rather than appending duplicate legacy token
  projections. Added positive and negative guardrail tests for the helper,
  parser/event routing, and projection reuse.
- Files changed:
  `src/avalan/agent/orchestrator/response/orchestrator_response.py`,
  `src/avalan/model/stream.py`,
  `tests/agent/orchestrator_response_additional_test.py`,
  `tests/model/model_stream_contract_test.py`, and
  `specs/streaming-agents.md`.
- Tests:
  `poetry run pytest tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source -q`
  (passed);
  `poetry run pytest tests/agent/orchestrator_response_additional_test.py::OrchestratorResponseAdditionalCoverageTestCase::test_stream_item_projection_confines_legacy_surfaces tests/agent/orchestrator_response_additional_test.py::OrchestratorResponseAdditionalCoverageTestCase::test_stream_item_projection_rejects_unsupported_item tests/agent/orchestrator_response_additional_test.py::OrchestratorResponseAdditionalCoverageTestCase::test_projection_reuses_recorded_tool_lifecycle_item -q`
  (3 passed);
  `poetry run pytest tests/agent/orchestrator_response_test.py tests/agent/orchestrator_response_additional_test.py tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source -q`
  (118 passed, 27 subtests passed);
  `poetry run pytest tests/server/test_a2a.py tests/server/mcp_router_test.py tests/cli/model_test.py::CliTokenGenerationTestCase tests/flow/runtime_test.py::FlowPlanExecutionTestCase::test_execute_flow_plan_projects_flow_events_to_canonical_items -q`
  (194 passed, 4 subtests passed);
  `make lint`;
  `poetry run pytest --verbose -s` (6207 passed, 42 skipped, 3249 subtests
  passed);
  `make test-coverage` (all source files 100%).
- Next slice: continue Phase 7.2 with the remaining inventory-backed
  compatibility shims outside `OrchestratorResponse`, starting with the
  public `TextGenerationResponse.__aiter__`/`__anext__` legacy iteration
  surface or the MCP legacy mapper if protocol risk takes priority.
- Review: No findings. Verified the four old `OrchestratorResponse`
  classifier inventory entries are absent, `_stream_item_projection` is
  the recorded compatibility shim, token/event/tool/terminal behavior is
  preserved in touched paths, and the Run 9 log is accurate. Follow-up is
  not required.

### Run 10: 7.2 Completion Audit And Final Cleanup

- Status: Complete.
- Worker: Gauss (`019ece70-1766-7673-8a15-37fea4fc56cb`), launched.
- Reviewer: Hubble (`019ece7a-7431-7e32-a14f-5da9a5869422`),
  completed.
- Task: Audit all remaining Phase 7.2 classifier/projection/reconstruction
  surfaces, fix any remaining gaps, and mark the 7.2 agenda checkboxes only
  if each item is genuinely complete as a whole task. Add or update
  focused tests if changes are needed, preserve 100% coverage, and amend
  changes to HEAD without changing the commit message.
- Notes: Audited the remaining model-stream, SDK, stdout/CLI, Chat SSE,
  Responses SSE, MCP, A2A, flow, and orchestrator classifier/projection
  surfaces. Found one remaining A2A legacy fallback that still kept its own
  legacy stream sequence/start state and called `canonical_item_from_token`
  directly. Replaced it with `_A2ALegacyStreamAdapter` plus
  `ProtocolStreamProjectionState`, added it to the explicit legacy
  classifier inventory with owner/removal criteria, and kept A2A artifact
  state as protocol projection state over canonical items. The remaining
  legacy classifiers are named temporary shims with removal criteria, and
  protocol projections derive from canonical items, `StreamProjectionState`,
  `ProtocolStreamProjectionState`, or the shared accumulators. Marked the
  four Phase 7.2 agenda checkboxes complete. Files changed:
  `src/avalan/server/a2a/router.py`, `src/avalan/model/stream.py`,
  `tests/server/a2a_router_unit_extra_test.py`,
  `tests/model/model_stream_contract_test.py`,
  `specs/streaming-agenda.md`, and this coordination file. Tests run:
  `poetry run pytest --verbose -s
  tests/server/a2a_router_unit_extra_test.py::test_translator_legacy_items_use_projection_adapter
  tests/server/a2a_router_unit_extra_test.py::test_translator_rejects_mixed_stream_surfaces
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source`
  (3 passed, 16 subtests passed); `poetry run pytest --verbose -s
  tests/server/a2a_router_unit_extra_test.py tests/server/test_a2a.py
  tests/server/protocol_streaming_e2e_test.py
  tests/server/streaming_conformance_test.py
  tests/model/model_stream_contract_test.py` (179 passed, 181 subtests
  passed); `poetry run pytest --verbose -s tests/server/router_streaming_test.py
  tests/server/mcp_router_test.py tests/server/create_response_sse_test.py
  tests/server/responses_utils_test.py
  tests/server/primary_consumer_projection_golden_test.py
  tests/cli/model_test.py::CliTokenGenerationTestCase
  tests/cli/theme_fancy_test.py tests/flow/runtime_test.py` (457 passed,
  89 subtests passed); `make lint` (passed, mypy clean);
  `poetry run pytest --verbose -s` (6208 passed, 42 skipped, 3250 subtests
  passed); `make test-coverage` (all source files 100%). Remaining Phase 7
  work: 7.3 bounded retention and 7.4 hot-path hardening.
- Review: No findings. Validated the Phase 7.2 completion claim: A2A's
  remaining legacy fallback is confined to `_A2ALegacyStreamAdapter` plus
  `ProtocolStreamProjectionState`, terminal/order enforcement uses shared
  accumulator paths, the remaining legacy classifier inventory has owners
  and removal criteria, and the four Phase 7.2 agenda boxes are correctly
  checked. Follow-up is not required.

### Run 11: 7.3 Retention Bounds Foundation

- Status: Complete.
- Worker: Bernoulli (`019ece82-1119-7ec2-8670-fc8afce7574b`),
  completed.
- Reviewer: Darwin (`019ece8b-bf20-7631-b108-e008276f2117`),
  completed.
- Task: Start Phase 7.3 by enforcing or auditing bounded defaults for
  event history, subscriber queues, and adjacent retention surfaces. Add
  focused positive and negative tests, preserve 100% coverage, and amend
  changes to HEAD without changing the commit message.
- Notes: Audited EventManager, A2A TaskStore, MCP resource store, and flow
  canonical listener retention defaults; those surfaces already have
  bounded histories/queues and pressure tests. Implemented the first
  missing retention guard in the shared canonical accumulator: retained
  semantic item history, diagnostics/control replay views, flow history,
  and usage/metrics history now evict oldest retained entries using
  `StreamRetentionPolicy`, while active final answer, reasoning, tool
  arguments/output, usage, terminal validation, and terminal projection
  remain lossless. Pressure behavior for these retained views is explicit
  oldest-entry eviction; final response reconstruction is unaffected during
  active sessions. Changed `src/avalan/model/stream.py` and
  `tests/model/model_stream_contract_test.py`. Agenda checkboxes
  unchanged because the full 7.3 item still spans server-mode no-listener
  memory and remaining adjacent retention surfaces. Tests run: `poetry run
  pytest --verbose -s tests/model/model_stream_contract_test.py` (116
  passed, 181 subtests passed); `poetry run pytest --verbose -s
  tests/server/router_streaming_test.py
  tests/server/protocol_streaming_e2e_test.py
  tests/server/create_response_sse_test.py tests/server/test_a2a.py` (94
  passed, 6 subtests passed); `poetry run pytest --verbose -s
  tests/event/event_manager_test.py tests/server/a2a_store_test.py
  tests/server/mcp_router_test.py tests/flow/runtime_test.py` (327
  passed, 84 subtests passed); `make lint`; `poetry run pytest --verbose
  -s` (6212 passed, 42 skipped, 3250 subtests passed); `make
  test-coverage` (all source files 100%). Next slice: verify repeated
  default server-mode streaming requests without UI listeners do not
  allocate or retain unbounded event/listen queues, accumulator histories,
  task records, MCP resources, or protocol adapter state.
- Review: No findings. Verified retained accumulator views are bounded by
  `StreamRetentionPolicy` with oldest-entry eviction, lossless active state
  remains separate, terminal projection still works after eviction, and
  Run 11 does not overclaim Phase 7.3 completion. Follow-up is not
  required.

### Run 12: 7.3 Server-Mode No-Listener Retention

- Status: Complete.
- Worker: Epicurus (`019ece8f-4714-79f2-9bfb-92f94386451c`),
  completed.
- Reviewers: Lovelace (`019ece9d-28e7-79a0-a59a-9d78ce1dafde`),
  Hilbert (`019ecea5-e4b3-75b1-a778-47d8d56251c9`), and Mencius
  (`019eceae-d394-7db0-b708-db0f8acbeacc`), completed.
- Task: Verify and enforce repeated default server-mode streaming requests
  without UI listeners do not accumulate unbounded event/listen queues,
  accumulator histories, task records, MCP resources, protocol adapter
  state, or related retained buffers. Add focused positive and negative
  tests, preserve 100% coverage, and amend changes to HEAD without
  changing the commit message.
- Notes: Enforced the missing server-mode retention pieces for this slice.
  MCP resource stores now cap retained resource records, evict oldest
  closed resources, protect active stream resources through terminal
  response reconstruction, and close successful stream resources before
  the final JSON-RPC result. A2A default task-record retention now uses
  the shared stream retention policy instead of a separate hard-coded
  default. Added repeated no-listener/default request coverage for A2A
  task records and MCP resource state, plus pressure tests confirming A2A
  final text reconstruction survives bounded task events/artifact deltas.
  Changed `src/avalan/server/routers/mcp.py`,
  `src/avalan/server/a2a/store.py`,
  `tests/server/mcp_router_test.py`,
  `tests/server/a2a_store_test.py`, `tests/server/test_a2a.py`, and
  `tests/server/protocol_streaming_e2e_test.py`. Agenda checkboxes
  unchanged because a final 7.3 closeout pass should still verify any
  remaining adjacent server-mode surfaces before marking the whole item
  complete. Tests run: `poetry run pytest --verbose -s
  tests/server/mcp_router_test.py tests/server/protocol_streaming_e2e_test.py`
  (133 passed, 2 subtests passed); `poetry run pytest --verbose -s
  tests/server/a2a_store_test.py tests/server/test_a2a.py
  tests/server/router_streaming_test.py tests/event/event_manager_test.py
  tests/server/responses_test.py` (157 passed, 12 subtests passed);
  `make lint`; `poetry run pytest --verbose -s` (6219 passed, 42
  skipped, 3250 subtests passed); `make test-coverage` (all source files
  100%). Next slice: final 7.3 audit across remaining default
  server-mode no-listener retention surfaces and decide whether the
  agenda item can be checked off.
- Review feedback: A single oversized completed MCP stream can create more
  distinct resources than the cap and leave all current-stream resources
  protected after `close_many()`, with no later unprotected prune before
  returning the final result. Follow-up requested from the worker.
- Follow-up: Added an explicit `MCPResourceStore.prune_closed()` step and
  call it after the successful MCP final JSON-RPC result is assembled, so
  current-stream resources stay protected through close notifications and
  response reconstruction but oldest closed resources are pruned before
  completion. Added a regression where one completed stream creates more
  distinct `tool_call_id:category` resources than `resource_limit`,
  verifies all close notifications and final tool-call summaries are still
  reconstructed, and asserts retained MCP resource state is bounded
  immediately after completion. Changed
  `src/avalan/server/routers/mcp.py`,
  `tests/server/mcp_router_test.py`, and
  `specs/streaming-agents.md`. Agenda checkboxes unchanged. Tests run:
  `poetry run pytest --verbose -s tests/server/mcp_router_test.py` (134
  passed, 2 subtests passed); `poetry run pytest --verbose -s
  tests/server/protocol_streaming_e2e_test.py` (1 passed); `make lint`
  (passed, mypy clean); `make test-coverage` (all source files 100%);
  `poetry run pytest --verbose -s` (6220 passed, 42 skipped, 3250
  subtests passed). Next slice remains: final 7.3 audit across remaining
  default server-mode no-listener retention surfaces and decide whether
  the agenda item can be checked off.
- Second review feedback: Oversized non-success MCP streams still bypass
  the unprotected prune. Cancellation, validation-error,
  canonical-cancelled, and canonical-errored branches close resources and
  return without `resource_store.prune_closed()`, so a single oversized
  non-success stream can retain more closed resources than `resource_limit`
  until a later store operation. Follow-up requested from the worker.
- Second follow-up: Routed all MCP terminal result/error exits through a
  shared helper that closes resources, emits close notifications, prunes
  closed retained resources, and then emits the terminal JSON-RPC message.
  The generator-close cleanup path now also prunes after closing. Added
  oversized non-success regression coverage for validation-error,
  canonical-cancelled, canonical-errored, and cancellation exits; each
  verifies close notifications and terminal errors are preserved while
  retained MCP resources, chunks, and order state are bounded immediately
  after completion. Changed `src/avalan/server/routers/mcp.py`,
  `tests/server/mcp_router_test.py`, and
  `specs/streaming-agents.md`. Agenda checkboxes unchanged. Tests run:
  `poetry run pytest --verbose -s tests/server/mcp_router_test.py` (135
  passed, 6 subtests passed); `poetry run pytest --verbose -s
  tests/server/protocol_streaming_e2e_test.py` (1 passed); `make lint`
  (passed, mypy clean); `poetry run pytest --verbose -s` (6221 passed,
  42 skipped, 3254 subtests passed); `make test-coverage` (all source
  files 100%). Next slice remains: final 7.3 audit across remaining
  default server-mode no-listener retention surfaces and decide whether
  the agenda item can be checked off.
- Second follow-up review: No findings. Verified MCP cancellation,
  validation-error, canonical-cancelled, canonical-errored, and success
  terminal paths use the shared terminal helper, preserve close
  notifications before pruning, prune closed retained resources, and then
  emit the terminal JSON-RPC message. Verified the oversized non-success
  regression covers validation, canonical-cancelled, canonical-errored,
  and cancellation exits with bounded retained resources, chunks, and
  order state. Follow-up is not required.

### Run 13: 7.3 Final Retention Closeout

- Status: Complete.
- Worker: Cicero (`019eceb2-8a5e-7812-a4e4-7749b9a3f1c2`),
  completed.
- Reviewers: Hegel (`019eceba-4709-74b0-acf9-533f8333d360`) and
  Hypatia (`019ecebe-418a-78e2-b9c4-415dac8f30dd`), completed.
- Task: Finalize Phase 7.3 by auditing all retention surfaces, adding any
  missing focused coverage, checking the four 7.3 agenda boxes only if
  genuinely satisfied, and amending changes to HEAD without changing the
  commit message.
- Notes: Audited the completed Run 11/12 retention work across
  EventManager history/listen queues/subscriber policies, shared
  canonical accumulator replay/flow/metrics histories, A2A task records,
  task events, messages, artifacts and artifact deltas, MCP resource
  text/chunks/resources, flow listener exact/UI histories, and
  repeated server-mode requests without UI listeners. Tightened the
  remaining A2A retention validation gap so `TaskStoreRetention` rejects
  `bool` and non-numeric bounds instead of accepting truthy values through
  comparisons, and strengthened negative tests for those invalid settings.
  Confirmed pressure behavior is explicit: lossless subscribers block,
  UI/listen projections coalesce or drop per configured class, critical
  subscribers fail closed, cancellation paths cancel/close stream sources,
  and retained histories/stores evict oldest entries or chunks while
  active final response accumulation remains lossless. Checked all four
  Phase 7.3 boxes in the ignored workspace checklist
  `specs/streaming-agenda.md` and did not begin 7.4. Tracked changes:
  `src/avalan/server/a2a/store.py`,
  `tests/server/a2a_store_test.py`, and
  `specs/streaming-agents.md`.
- Tests run: `poetry run pytest --verbose -s
  tests/server/a2a_store_test.py` (9 passed);
  `poetry run pytest --verbose -s tests/model/model_stream_contract_test.py
  tests/event/event_manager_test.py tests/server/a2a_store_test.py
  tests/server/test_a2a.py tests/server/mcp_router_test.py
  tests/flow/runtime_test.py tests/server/responses_test.py
  tests/server/router_streaming_test.py
  tests/server/protocol_streaming_e2e_test.py` (524 passed, 271 subtests
  passed); `make lint` (passed, mypy clean); `poetry run pytest
  --verbose -s` (6221 passed, 42 skipped, 3254 subtests passed);
  `make test-coverage` (all source files 100%).
- Phase completed: Yes, Phase 7.3 is complete.
- Next slice: Start Phase 7.4 hot-path hardening with an audit of
  full-buffer parser scans and the first incremental parser-state
  replacement.
- Review feedback: No code-level findings. Verified
  `TaskStoreRetention` validation uses shared helpers that reject bool and
  non-numeric bounds, negative tests cover the meaningful invalid paths,
  bounded retention behavior and final reconstruction checks support the
  7.3 completion claim, and the focused 7.3 suite passed. Documentation
  follow-up requested because the worker line named Codex instead of
  Cicero and the Run 13 tracked-change note implied ignored
  `specs/streaming-agenda.md` was part of signed HEAD.
- Follow-up: Corrected the worker identity to Cicero and clarified that
  the 7.3 boxes were checked in the ignored workspace checklist while the
  signed HEAD tracked changes are `src/avalan/server/a2a/store.py`,
  `tests/server/a2a_store_test.py`, and `specs/streaming-agents.md`.
- Follow-up review: No findings. Verified Run 13 now names Cicero,
  describes `specs/streaming-agenda.md` as an ignored workspace checklist,
  excludes it from the tracked-change list, and leaves 7.3 checked with
  7.4 unchecked. Follow-up is not required.

### Run 14: 7.4 Tool Parser Hot-Path State

- Status: Complete.
- Worker: Anscombe (`019ecec0-db8d-7820-8848-e5d8b28d99d1`), completed.
- Reviewers: Huygens (`019eced0-bbd7-76e3-a979-d0fcc5d0687c`) and
  Chandrasekhar (`019ecede-0086-7b21-b07d-6176ebe91d50`), completed.
- Task: Start Phase 7.4 hot-path hardening by replacing repeated
  full-buffer scans in `ToolCallResponseParser.push()` with incremental
  parser state while preserving XML-style, Harmony, and DSML tool marker
  behavior.
- Notes: Added streaming state for visible markdown fence and same-line
  quote suppression, plus incremental close detection for active tool-call
  buffers. `push()` now uses that state for executable marker detection,
  visible-prefix splitting, close-suffix splitting, and split close-marker
  handling. The legacy helper methods remain available for direct callers
  and compatibility tests. Added positive and negative parser coverage for
  fenced and quoted marker suppression, split close markers, self-closing
  XML tags with quoted attributes, Harmony visible suffixes, DSML wrappers,
  and a deterministic hot-path regression that fails if the old full-buffer
  fence, quote, or close-scan helpers are used by the streaming path.
- Changed files: `src/avalan/model/response/parsers/tool.py`,
  `tests/agent/tool_call_parser_extra_test.py`, and this run log. The
  ignored workspace checklist `specs/streaming-agenda.md` was updated
  separately.
- Tests run: `poetry run pytest --verbose -s
  tests/agent/tool_call_parser_extra_test.py
  tests/agent/tool_call_parser_test.py
  tests/agent/tool_call_parser_fixture_test.py
  tests/model/response_parsers_additional_test.py` (103 passed, 6
  subtests passed); `make lint` (passed, mypy clean);
  `poetry run pytest --verbose -s` (6229 passed, 42 skipped, 3254
  subtests passed); `make test-coverage` (all source files 100%).
- 7.4 checkbox checked: Yes, only the first 7.4 checkbox in the ignored
  workspace checklist.
- Review loop 1: Huygens found that split Harmony final-channel markers
  could leak as tool tokens and that active in-call chunks still rescanned
  the accumulated buffer through `tool_call_status()`. Follow-up kept the
  first 7.4 checkbox checked after adding pending visible-suffix state,
  preserving split final markers as visible suffix text, and skipping
  non-terminal `tool_call_status()` rescans for the real `ToolManager`
  while retaining the compatibility fallback for non-standard managers.
  Added regressions for both cases; focused parser tests now report `103
  passed, 6 subtests passed`.
- Follow-up review: No findings. Verified split Harmony final markers are
  covered and preserve `TOOL_PROCESS`, visible suffix text, no leaked
  `ToolCallToken`, and no diagnostic. Verified exact `ToolManager` open
  chunks return before accumulated-buffer `tool_call_status()` rescans,
  while non-standard manager compatibility fallback remains. The first
  7.4 checklist item remains justified, and the remaining 7.4 items stay
  unchecked. Follow-up is not required.
- Next slice: Continue Phase 7.4 by auditing provider/protocol envelope
  allocation on per-token paths and reusing static state where semantics
  are unchanged.

### Run 15: 7.4 Protocol Envelope Hot-Path State

- Status: Complete.
- Task: Continue Phase 7.4 by addressing the second checkbox: reuse static
  provider and protocol envelope state where semantics are unchanged.
- Notes: Added a frozen Chat SSE chunk envelope that precomputes the
  stable response id, timestamp, model, object, choice, and delta JSON
  prefix/suffix once per stream while encoding only the content delta per
  token. Added a Responses SSE stream envelope for stable
  `response.created` provider state and immutable shared output/content
  index mappings for repeated protocol event envelopes, copying them into
  fresh event dictionaries so chunk mutations cannot leak across events or
  streams. Preserved terminal, usage, cancellation, error, coalescing, and
  protocol ordering behavior.
- Changed files: `src/avalan/server/routers/chat.py`,
  `src/avalan/server/routers/responses.py`,
  `tests/server/chat_router_unit_test.py`,
  `tests/server/responses_utils_test.py`, and this run log. The ignored
  workspace checklist `specs/streaming-agenda.md` was updated separately.
- Tests run:
  `poetry run pytest --verbose -s tests/server/chat_router_unit_test.py
  tests/server/responses_utils_test.py tests/server/create_response_sse_test.py
  tests/server/streaming_conformance_test.py
  tests/server/primary_consumer_projection_golden_test.py
  tests/server/chat_test.py tests/server/responses_test.py` (126 passed,
  3 skipped, 10 subtests passed); `git diff --check` (passed);
  `make lint` (passed; ruff format, black, ruff check --fix, mypy);
  `poetry run pytest --verbose -s` (6239 passed, 42 skipped, 3254
  subtests passed); `make test-coverage` (all source files 100%).
- 7.4 checkbox checked: Yes, only the second 7.4 checkbox in the ignored
  workspace checklist was checked by this run. The remaining 7.4 boxes
  stay unchecked.
- Review feedback: No findings. Verified the Chat SSE envelope reuse and
  Responses SSE static index reuse are real, reused state does not share
  mutable per-event data across chunks or streams, and terminal, usage,
  cancellation, validation-error, and failed/cancelled terminal paths are
  preserved. Reviewer checks included `git diff --check`, focused server
  streaming/router tests (126 passed, 3 skipped, 10 subtests passed), a
  signed HEAD check for `e0644c4a34eecae2368f711d47fd8e40958b32ff`, and
  confirmation that only the first two 7.4 implementation boxes are
  checked in the ignored workspace checklist.
- Next slice: Continue Phase 7.4 with the third checkbox by auditing
  timestamp generation, tokenizer enrichment, and detailed metrics on
  streaming hot paths, then making any unchanged or optional work opt-in
  or subscriber-driven.

### Run 16: 7.4 Optional Stream Enrichment

- Status: Complete.
- Task: Complete the third 7.4 checkbox by making timestamp-adjacent
  optional work, tokenizer enrichment, and detailed metrics opt-in or
  subscriber-driven on streaming hot paths.
- Notes: Added explicit event stats collection control with server-mode
  managers defaulting detailed stats off while still delivering events to
  listeners. The CLI render stream now skips display-token projection and
  tokenizer-config table reads unless token display or probability output
  requests them. Token-generated event enrichment now avoids engine
  tokenizer access unless token-id enrichment is enabled. Required
  protocol timestamps and explicit opt-in behavior are preserved.
- Changed files: `src/avalan/event/manager.py`,
  `src/avalan/cli/commands/model.py`,
  `src/avalan/agent/orchestrator/response/orchestrator_response.py`,
  `tests/event/event_manager_test.py`, `tests/cli/model_test.py`,
  `tests/agent/orchestrator_response_test.py`,
  `tests/server/responses_test.py`, and this run log. The ignored
  workspace checklist `specs/streaming-agenda.md` was updated separately.
- Tests run: `poetry run pytest --verbose -s
  tests/event/event_manager_test.py` (78 passed, 10 subtests passed);
  `poetry run pytest --verbose -s tests/cli/model_test.py -k
  'stream_render_items or token_stream_skips_tokenizer_config_without_display_tokens
  or token_generation_with_stats_uses_canonical_projection'` (16 passed,
  106 deselected, 2 subtests passed); `poetry run pytest --verbose -s
  tests/agent/orchestrator_response_test.py -k
  'tokenizer_when_enrichment_disabled or to_str_emits_streamed_token_events
  or iteration_emits_events_and_end'` (3 passed, 90 deselected);
  `poetry run pytest --verbose -s tests/server/responses_test.py -k
  'retain_events or stats_can_be_enabled'` (3 passed, 5 deselected);
  `poetry run pytest --verbose -s tests/event/event_manager_test.py
  tests/cli/model_test.py` (202 passed, 16 subtests passed);
  `poetry run pytest --verbose -s
  tests/agent/orchestrator_response_test.py tests/server/responses_test.py`
  (101 passed); `poetry run pytest --cov=avalan.event.manager
  --cov=avalan.cli.commands.model --cov-report=term-missing --verbose -s
  tests/event/event_manager_test.py tests/cli/model_test.py` (202 passed,
  16 subtests passed; both touched modules 100%); `make lint` (passed;
  ruff format, black, ruff check --fix, mypy); `poetry run pytest
  --verbose -s` (6248 passed, 42 skipped, 3254 subtests passed);
  `make test-coverage` (all source files 100%).
- Coverage note: Two concurrent filtered `make test-coverage` probes
  collided on `coverage.json` and failed before the final serial full
  `make test-coverage` pass above. A file-path `--cov` probe produced no
  useful coverage data and was replaced with the module-name command
  listed above.
- 7.4 checkbox checked: Yes, only the third 7.4 checkbox in the ignored
  workspace checklist was checked by this run. Later 7.4 boxes remain
  unchecked.
- Review feedback: No findings. Verified optional hot-path work is gated
  before event or enrichment construction, explicit subscriber and option
  paths still deliver expected enrichment/stats, server-mode defaults keep
  token events and detailed stats off unless requested, and the ignored
  checklist has the first three 7.4 implementation boxes checked with the
  fourth and fifth unchecked. Reviewer checks included `git diff --check
  HEAD^ HEAD`, signature status `G` for
  `b139e7a999c40066f7b0f66fb071477a3e49b380`, a focused opt-in suite
  (12 passed, 291 deselected), and the broader event/CLI/orchestrator/
  responses suite (303 passed, 16 subtests passed).
- Next slice: Continue Phase 7.4 by removing unnecessary per-token model
  construction and render rebuilds.

### Run 17: 7.4 Render Construction Hot-Path State

- Status: Complete.
- Task: Complete the fourth 7.4 checkbox by removing unnecessary
  per-token model construction and render rebuilds.
- Notes: Cached the legacy token stream adapter on each
  `StreamProjectionState` so legacy projection no longer constructs a new
  adapter for every token while preserving per-state isolation. Cached CLI
  tokenizer display settings once per stream instead of reading them on
  every rendered token. Replaced eager materialization of all theme token
  frames with lazy first-frame consumption, closing unused extra frames
  when probability display is off while still consuming probability-detail
  frames when requested. CLI output semantics, frame coalescing,
  cancellation cleanup, display-token behavior, and stream validation were
  preserved.
- Changed files: `src/avalan/model/stream.py`,
  `src/avalan/cli/commands/model.py`,
  `tests/model/model_stream_contract_test.py`,
  `tests/cli/model_test.py`, and this run log. The ignored workspace
  checklist `specs/streaming-agenda.md` was updated separately.
- Tests run: `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py -k
  'stream_projection_state_reuses_legacy_adapter or
  stream_projection_state_does_not_share_legacy_adapters or
  stream_projection_state_projects_legacy_tokens_with_fallback_ids'`
  (3 passed, 115 deselected); `poetry run pytest --verbose -s
  tests/cli/model_test.py -k
  'token_stream_reuses_tokenizer_config_per_stream or
  token_stream_closes_unused_theme_frames or
  token_stream_accepts_empty_theme_frame_stream or
  token_stream_consumes_probability_theme_frames'` (4 passed, 123
  deselected); `poetry run pytest --verbose -s tests/cli/model_test.py`
  (126 passed, 6 subtests passed); `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py` (118 passed, 181 subtests
  passed); `poetry run pytest --cov=avalan.cli.commands.model
  --cov=avalan.model.stream --cov-report=term-missing --verbose -s
  tests/cli/model_test.py tests/model/model_stream_contract_test.py` (246
  passed, 187 subtests passed; touched modules 100%); `make lint`
  (passed; ruff format, black, ruff check --fix, mypy);
  `poetry run pytest --verbose -s` (6255 passed, 42 skipped, 3254
  subtests passed); `make test-coverage` (all source files 100%).
- 7.4 checkbox checked: Yes, only the fourth 7.4 checkbox in the ignored
  workspace checklist was checked by this run. The benchmark checkbox
  remains unchecked.
- Review feedback loop 1: Fixed a cancellation cleanup gap in the lazy
  CLI token-frame path. Each acquired theme frame stream is now closed
  from a `finally`, including cancellation during display pauses and both
  probability and non-probability paths. Added a regression test that
  cancels after the first lazy frame and asserts the frame generator
  cleanup runs and the stop signal is set. Follow-up checks:
  `poetry run pytest --verbose -s tests/cli/model_test.py -k
  'token_stream_closes_theme_frames_on_pause_cancel or
  token_stream_closes_unused_theme_frames or
  token_stream_consumes_probability_theme_frames or
  token_stream_accepts_empty_theme_frame_stream'` (4 passed, 124
  deselected); `poetry run pytest --verbose -s tests/cli/model_test.py`
  (128 passed, 6 subtests passed); `poetry run pytest
  --cov=avalan.cli.commands.model --cov-report=term-missing --verbose -s
  tests/cli/model_test.py` (128 passed, 6 subtests passed;
  `avalan.cli.commands.model` 100%); `make lint` (passed after
  shortening the regression test name; ruff format, black, ruff check
  --fix, mypy); `poetry run pytest --verbose -s` (6256 passed, 42
  skipped, 3254 subtests passed); `make test-coverage` (all source files
  100%).
- Follow-up review: No findings. Verified every acquired
  `token_frames_stream` is closed from the cleanup path, including empty
  streams, cancellation during first display pause, non-probability early
  close, and probability-frame iteration. Verified the new cancellation
  regression fails on the previous leaked-generator path, adapter reuse
  remains per `StreamProjectionState`, tokenizer config is cached per
  stream, lazy first-frame rendering preserves probability and empty-stream
  behavior, cancellation still sets the stop signal, and frame coalescing
  still renders the latest dirty frame. Follow-up review checks included
  `git diff --check a7873da3972cf7cc629403911af3cadaea4d225a..HEAD`, a
  focused lazy-frame regression suite (4 passed), adapter reuse suite (3
  passed), render/coalescing suite (23 passed, 2 subtests passed),
  `poetry run pytest --verbose -s tests/cli/model_test.py` (128 passed, 6
  subtests passed), and `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py` (119 passed, 181 subtests
  passed).
- Next slice: Finish Phase 7.4 by benchmarking long streams, slow
  consumers, cancellation, memory retention, and per-token adapter
  overhead, then record the benchmark results.

### Run 18: 7.4 Streaming Benchmark Coverage

- Status: Complete; 7.4 implementation is done pending review.
- Task: Complete the fifth 7.4 checkbox by benchmarking long streams,
  slow consumers, cancellation, memory retention, and per-token adapter
  overhead, without marking broader Phase 7.4 acceptance items.
- Notes: Added deterministic synthetic benchmark regression coverage for
  long canonical projection streams, slow local consumers, cancellation
  latency, bounded retained history, peak retained-memory behavior, and
  cached legacy adapter projection overhead. The same test file includes
  negative coverage for content after terminal state and unsupported
  legacy adapter inputs. Runtime behavior was unchanged.
- Synthetic benchmark results from the final focused run
  `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py`: cached legacy
  adapter projection handled 8192 tokens at 3.797 us/token; cancellation
  latency was 0.058 ms; long canonical projection handled 8196 items at
  5.445 us/item; retained-history memory for 8196 items kept 128 retained
  items with 122912 peak traced bytes; slow local consumption of 256
  tokens had 0 max read-ahead and completed in 6.184 ms. The focused
  coverage probe ran with coverage instrumentation and stayed within the
  250 us/item budget despite higher measured overhead.
- Full benchmark run: `/bin/zsh -lc 'source .env.sh; poetry run python
  specs/streaming/benchmark.py --model all --repeats 1 --warmups 0
  --max-new-tokens 1024 --baseline phase-7 --json-out
  specs/streaming/latest.json --append-benchmarks
  specs/streaming/BENCHMARKS.md --continue-on-error'` completed for
  `openai`, `hermes`, and `gpt-oss`. Results: `openai` TTFT 5.429s,
  total 7.730s, 31.95 estimated tokens/s; `hermes` TTFT 0.353s, total
  11.122s, 29.67 estimated tokens/s; `gpt-oss` TTFT 0.488s, total
  4.138s, 92.32 estimated tokens/s.
- Comparison: `openai` was slower than `main` and phase-6 on TTFT/total,
  with a shorter hosted output and no reasoning tokens, so the delta is
  recorded as hosted provider variance. `hermes` stayed effectively flat
  versus `main`/phase-6 and slightly improved total time. `gpt-oss`
  improved TTFT, total time, and estimated throughput versus both `main`
  and phase-6.
- Changed files: `tests/model/model_stream_benchmark_regression_test.py`
  and this run log. The ignored workspace checklist
  `specs/streaming-agenda.md` was updated separately; the ignored
  `specs/streaming/BENCHMARKS.md` and `specs/streaming/latest.json` were
  updated by the full benchmark harness but are not tracked in this
  checkout.
- Tests and benchmarks run: `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py` (7 passed);
  `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py
  tests/model/model_stream_contract_test.py
  tests/server/protocol_streaming_e2e_test.py` (127 passed, 181 subtests
  passed); `make lint` (passed; ruff format, black, ruff check --fix,
  mypy); `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py` (7 passed after
  lint); `poetry run pytest --cov=avalan.model.stream
  --cov-report=term-missing --verbose -s
  tests/model/model_stream_benchmark_regression_test.py
  tests/model/model_stream_contract_test.py` (126 passed, 181 subtests
  passed; focused module-only coverage 99%, with remaining branches
  covered by broader suites); `poetry run pytest --verbose -s` (6263
  passed, 42 skipped, 3254 subtests passed); `make test-coverage` (all
  source files 100%).
- 7.4 checkbox checked: Yes, only the fifth/final implementation checkbox
  in the ignored workspace checklist was checked by this run.
- All 7.4 implementation checkboxes checked: Yes.
- Review feedback: No findings. Verified HEAD
  `969ff80b17dbd9095e09073123209acc5de28c92` is signed with status `G`,
  the tracked worktree is clean, and the tracked slice is limited to
  `tests/model/model_stream_benchmark_regression_test.py` plus this run
  log. Verified the benchmark regression coverage maps to all requested
  dimensions: long streams, slow consumer/read-ahead, cancellation
  latency/cleanup, retention peak memory, per-token cached adapter
  overhead, and negative guardrails. Focused review test
  `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py` passed (7
  passed in 0.35s). Confirmed ignored benchmark outputs
  `specs/streaming/BENCHMARKS.md`, `specs/streaming/latest.json`, and
  `specs/streaming-agenda.md` staying untracked is consistent with
  `.gitignore`, and all five 7.4 implementation boxes are checked while
  broader 7.4 exit-condition boxes remain unchecked.
- Next slice: Address remaining Phase 7.4 test-focus checkboxes as
  separate acceptance work.

### Run 19: 7.4 Test Focus Acceptance Coverage

- Status: Complete; 7.4 Test Focus is done pending review. Broader 7.4
  Exit Conditions and Final Acceptance Gates remain unchecked.
- Task: Address the remaining unchecked 7.4 Test Focus boxes only,
  adapting to Runs 14-18 without changing runtime behavior.
- Notes: Added a public-surface e2e test that drives one canonical stream
  through SDK projections, stdout generation, CLI/FancyTheme rendering,
  MCP, and A2A. The CLI branch proves derived token frames can be closed
  after the first frame while SDK/stdout/protocol consumers still receive
  the full ordered semantic stream. Added provider and local gated-pull
  tests proving slow consumers do not cause read-ahead. Strengthened the
  live tool stream test so stdout/stderr/progress items are sequenced
  before the tool execution terminal and before the model-facing final
  observation. Added focused negative coverage for malformed tool calls,
  duplicate stream terminals, content after terminal state, missing tool
  correlation IDs, stream subscriber overflow, provider cancellation
  races, and retention limits. Runtime behavior was unchanged.
- Changed files: `tests/server/protocol_streaming_e2e_test.py`,
  `tests/model/model_stream_contract_test.py`,
  `tests/event/event_manager_test.py`,
  `tests/agent/orchestrator_response_test.py`, and this run log. The
  ignored workspace checklist `specs/streaming-agenda.md` was updated
  separately.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/server/protocol_streaming_e2e_test.py
  tests/model/model_stream_contract_test.py tests/event/event_manager_test.py
  tests/agent/orchestrator_response_test.py -k
  'lossy_cli_frames_do_not_drop_lossless_public_surfaces or
  provider_and_local_streams_wait_for_slow_consumers or
  stream_negative_acceptance_edges or
  provider_stream_cancellation_race_closes_source or
  stream_item_subscriber_overflow_fails_closed or
  iteration_records_live_tool_output_before_completion or
  tool_lifecycle_assembles_planned_order_observations'` (7 passed, 290
  deselected); `poetry run pytest --verbose -s
  tests/server/protocol_streaming_e2e_test.py
  tests/model/model_stream_contract_test.py tests/event/event_manager_test.py
  tests/agent/orchestrator_response_test.py` (297 passed, 191 subtests
  passed); `make lint` (passed; ruff format, black, ruff check --fix with
  2 fixes, mypy success); `poetry run pytest --verbose -s
  tests/server/protocol_streaming_e2e_test.py
  tests/model/model_stream_contract_test.py tests/event/event_manager_test.py
  tests/agent/orchestrator_response_test.py` (297 passed, 191 subtests
  passed after lint); `poetry run pytest --verbose -s` (6268 passed, 42
  skipped, 3254 subtests passed); `make test-coverage` (all source files
  100%).
- 7.4 Test Focus boxes checked: Yes, all five previously unchecked Test
  Focus boxes were checked in the ignored workspace checklist:
  UI/FancyTheme lossy-frame isolation without loss to SDK/stdout/protocol
  consumers; provider and local backpressure; live tool output and final
  observation ordering; negative streaming edge cases; representative e2e
  public-surface runs.
- Remaining 7.4 boxes unchecked: All 7.4 Exit Conditions remain unchecked.
  Final Acceptance Gates remain unchecked. No Exit Conditions or Final
  Acceptance Gates were changed by this run.
- Next slice: Address the remaining 7.4 Exit Conditions as acceptance
  audit work, starting with the internal-consumer/ad hoc-classifier and
  legacy-shim removal checks.

#### Run 19 Review Feedback Loop 1

- Reviewer feedback: Dirac found medium-risk determinism gaps in
  `tests/model/model_stream_contract_test.py`; three acceptance tests
  waited on internal async events without timeouts, so a regression that
  prevents a pending provider/local pull from starting could hang instead
  of failing deterministically.
- Follow-up: Wrapped the flagged internal event waits with
  `asyncio.wait_for(...)` and bounded the pending pull awaits. The
  provider cancellation race and provider/local backpressure branches now
  cancel and await unfinished pull tasks in `finally` before closing the
  stream, so failed expectations do not leave pending tasks behind.
- Checklist status: The five 7.4 Test Focus boxes remain checked. This
  follow-up makes the existing provider/local backpressure and
  cancellation-race coverage deterministic enough to keep those boxes
  justified. No Exit Conditions or Final Acceptance Gates were changed.
- Files changed: `tests/model/model_stream_contract_test.py` and this run
  log.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/server/protocol_streaming_e2e_test.py
  tests/model/model_stream_contract_test.py tests/event/event_manager_test.py
  tests/agent/orchestrator_response_test.py -k
  'lossy_cli_frames_do_not_drop_lossless_public_surfaces or
  provider_and_local_streams_wait_for_slow_consumers or
  stream_negative_acceptance_edges or
  provider_stream_cancellation_race_closes_source or
  stream_item_subscriber_overflow_fails_closed or
  iteration_records_live_tool_output_before_completion or
  tool_lifecycle_assembles_planned_order_observations'` (7 passed, 290
  deselected); `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py` (122 passed, 181 subtests
  passed); `make lint` (passed; ruff format, black, ruff check --fix, and
  mypy); focused acceptance selection after lint (7 passed, 290
  deselected); `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py` after lint (122 passed, 181
  subtests passed); `poetry run pytest --verbose -s` (6268 passed, 42
  skipped, 3254 subtests passed); `make test-coverage` (all source files
  100%).
- Next slice: Continue with the remaining 7.4 Exit Conditions audit,
  starting with internal-consumer/ad hoc-classifier and legacy-shim
  removal checks.
- Follow-up review: No findings. Verified the three previously unbounded
  async event waits are now bounded with `asyncio.wait_for(...)`, and
  unfinished pending pulls are cancelled and awaited before stream close in
  cleanup paths. The fix converts the prior hang mode into deterministic
  failure without weakening assertions. Review checks included `git diff
  --check 4f66b3c6bd9467e19cd701ff91ac75b091de7c0d..HEAD`, focused
  provider cancellation and slow-consumer tests, and the Run 19 focused
  acceptance selection (7 passed, 290 deselected). Confirmed all five 7.4
  Test Focus boxes remain checked, 7.4 Exit Conditions and Final
  Acceptance Gates remain unchecked, the tracked worktree is clean, and
  HEAD `d01e285fea892f943fe41b14f7b49298fa829608` has signature status
  `G`.

### Run 20: 7.4 Exit Conditions Audit

- Status: Complete; first three 7.4 Exit Conditions are checked. Later
  Exit Conditions and Final Acceptance Gates remain unchecked.
- Task: Audit and complete only the first three Phase 7.4 Exit Conditions:
  internal consumers must not depend on ad hoc legacy stream classifiers,
  remaining legacy token/event stream surfaces must be documented temporary
  shims with owners and removal criteria, and default server-mode paths
  must not accumulate unbounded event history, resource text, artifact
  deltas, task events, metrics history, or subscriber queues.
- Notes: Strengthened the legacy stream classifier source guard to scan all
  server modules in addition to the existing streaming-owned projection
  modules. The guard still only treats raw string checks as stream
  classifiers at explicit legacy stream sites, avoiding unrelated string
  validation noise, and now asserts the remaining inventoried classifiers
  are only temporary ingestion or compatibility shims. Added a server-mode
  retention acceptance guard covering protocol retention settings,
  EventManager server defaults, A2A task retention defaults, and MCP
  resource text retention. Runtime behavior was unchanged.
- Files changed: `tests/model/model_stream_contract_test.py`,
  `tests/server/router_streaming_test.py`, this run log, and the ignored
  workspace checklist `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_inventory_matches_source
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_guard_detects_new_direct_classifiers
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_classifier_guard_detects_tracked_string_classifiers`
  (3 passed, 16 subtests passed); `poetry run pytest --verbose -s
  tests/server/router_streaming_test.py::RouterStreamingTestCase::test_default_server_stream_retention_surfaces_are_bounded
  tests/server/a2a_store_test.py::test_task_store_bounds_records_and_histories
  tests/event/event_manager_test.py::AsyncEventManagerTestCase::test_server_mode_disables_history_and_drops_observability`
  (no tests ran; stale node id for the EventManager test); corrected
  `poetry run pytest --verbose -s
  tests/server/router_streaming_test.py::RouterStreamingTestCase::test_default_server_stream_retention_surfaces_are_bounded
  tests/server/a2a_store_test.py::test_task_store_bounds_records_and_histories
  tests/event/event_manager_test.py::EventManagerTestCase::test_server_mode_disables_history_and_drops_observability`
  (3 passed, 5 subtests passed); `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py tests/server/router_streaming_test.py
  tests/server/a2a_store_test.py tests/server/mcp_router_test.py
  tests/event/event_manager_test.py` (384 passed, 204 subtests passed);
  `git diff --check` (passed); `make lint` (passed; ruff format, black,
  ruff check --fix, mypy); focused file suite after lint (384 passed, 204
  subtests passed); `poetry run pytest --verbose -s` (6269 passed, 42
  skipped, 3259 subtests passed); `make test-coverage` (all source files
  100%).
- 7.4 Exit Conditions checked: Yes, only the first three Exit Conditions
  in the ignored workspace checklist were checked by this run.
- Remaining 7.4 boxes unchecked: latency/per-token-overhead exit;
  cancellation/close latency exit; acceptance harness coverage exit;
  negative/e2e changed-hardening exit; phase-7 benchmark recording exit;
  `src/` coverage exit. Final Acceptance Gates remain unchecked.
- Follow-up review: No blocking findings. Verified the ignored checklist
  marks only the first three 7.4 Exit Conditions checked, the legacy
  classifier guard now scans streaming-owned and server modules with
  negative coverage, the remaining legacy surfaces are documented temporary
  shims with owners and removal criteria, and default server retention
  coverage matches EventManager, MCP resource text, and A2A task-store
  bounded defaults. Reviewer verification ran the focused classifier and
  default retention tests (`4 passed, 21 subtests passed`). Residual risk:
  the AST guard targets `isinstance` classifiers in selected stream/server
  modules and would not catch a future `type(...) is ...` or pattern-match
  classifier without extending the guard.
- Next slice: Continue Phase 7.4 Exit Conditions with the latency and
  per-token overhead acceptance audit.

### Run 21: 7.4 Local Latency And Overhead Exit

- Status: Complete; only the local streaming latency/per-token overhead
  Exit Condition is checked. Later Exit Conditions and Final Acceptance
  Gates remain unchecked.
- Task: Audit and complete only this Phase 7.4 Exit Condition: local
  streaming latency does not regress and per-token overhead stays within
  defined budgets.
- Notes: Audited the existing budget definitions and benchmark evidence.
  `StreamPerformanceBudget` keeps the enforced thresholds at
  `time_to_first_item_ms=5000` and `per_item_overhead_us=250`. The
  existing Phase 7 benchmark rows show local TTFT did not regress versus
  `main`: `hermes` moved from 0.353s to 0.353s, and `gpt-oss` moved from
  0.591s to 0.488s. Existing regression tests already covered long
  canonical projection overhead and cached legacy adapter projection
  overhead. Added a synthetic guard for the actual `normalize_local_stream`
  path so startup latency and local per-item overhead are asserted against
  the same budgets, and added negative coverage for non-positive latency
  and overhead budgets. Runtime behavior was unchanged.
- Files changed: `tests/model/model_stream_benchmark_regression_test.py`,
  this run log, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py` (9 passed, 2
  subtests passed; local first item 0.035 ms, local first answer 0.021 ms,
  local normalization 8.274 us/item, long projection 5.195 us/item, cached
  legacy adapter 3.614 us/token); `git diff --check` (passed);
  `make lint` (passed; ruff format, black, ruff check --fix, mypy);
  `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py` after lint (9
  passed, 2 subtests passed; local first item 0.032 ms, local first answer
  0.024 ms, local normalization 8.692 us/item, long projection 5.326
  us/item, cached legacy adapter 3.746 us/token); `git diff --check`
  after lint (passed); `poetry run pytest --verbose -s` (6271 passed, 42
  skipped, 3261 subtests passed); `make test-coverage` (all source files
  100%).
- 7.4 Exit Conditions checked: Yes, only the local streaming
  latency/per-token overhead Exit Condition in the ignored workspace
  checklist was checked by this run.
- Remaining 7.4 boxes unchecked: cancellation/close latency exit;
  acceptance harness coverage exit; negative/e2e changed-hardening exit;
  phase-7 benchmark recording exit; `src/` coverage exit. Final Acceptance
  Gates remain unchecked.
- Next slice: Continue Phase 7.4 Exit Conditions with cancellation and
  close latency across hosted providers, local generation, tools, fanout
  subscribers, MCP, A2A, and HTTP adapters.

### Run 22: 7.4 Cancellation And Close Latency Exit

- Status: Complete; only the cancellation/close latency Exit Condition is
  checked. Later Exit Conditions and Final Acceptance Gates remain
  unchecked.
- Task: Audit existing cancellation and close-latency evidence across
  hosted provider normalization, local generation/`normalize_local_stream`,
  tool streaming, fanout subscribers, MCP, A2A, and HTTP/SSE adapters.
  Complete only the Phase 7.4 Exit Condition: `Cancellation and close
  latency meet budgets across hosted providers, local generation, tools,
  fanout subscribers, MCP, A2A, and HTTP adapters.`
- Notes: Existing coverage already proved provider/local close correctness,
  provider/local backpressure, provider cancellation races, HTTP cleanup on
  disconnect/cancellation, MCP pending-pull interruption and cleanup, A2A
  cancellation cleanup, EventManager fanout subscriber draining, and tool
  cancellation guard behavior. Added a focused synthetic latency guard
  against `StreamPerformanceBudget(cancellation_latency_ms=1000,
  close_latency_ms=1000)` for the Avalan-owned close/cancel paths: hosted
  provider normalization via `normalize_provider_stream` with hosted
  capabilities, local generation via `normalize_local_stream`, cooperative
  native tool streaming with `ToolCallContext.stream_event`, fanout
  subscriber `EventManager.aclose()`, MCP pending-pull cancellation plus
  MCP cleanup wrapper, A2A cleanup wrapper, and the shared HTTP/SSE
  `cleanup_stream_sources()` adapter. Added negative coverage for
  non-positive cancellation and close budgets. Residual limitation: live
  external hosted-provider transport latency is not directly exercised;
  the deterministic hosted-provider source proves Avalan's normalizer
  promptly cancels and closes the provider iterator. Arbitrary already
  running tools remain cooperative unless the surrounding task is
  cancelled; the guard proves the Avalan-owned cooperative streaming tool
  path.
- Files changed: `tests/server/streaming_latency_budget_test.py`, this run
  log, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/server/streaming_latency_budget_test.py` before lint (2 passed, 15
  subtests passed); `git diff --check` (passed); `make lint` (passed;
  ruff format, black, ruff check --fix, mypy); `poetry run pytest
  --verbose -s tests/server/streaming_latency_budget_test.py` after lint
  (2 passed, 15 subtests passed; all measured latencies below 1 ms);
  `poetry run pytest --verbose -s` (6273 passed, 42 skipped, 3276
  subtests passed); `make test-coverage` (all source files 100%).
- 7.4 Exit Conditions checked: Yes, only the cancellation/close latency
  Exit Condition in the ignored workspace checklist was checked by this
  run.
- Remaining 7.4 boxes unchecked: acceptance harness coverage exit;
  negative/e2e changed-hardening exit; phase-7 benchmark recording exit;
  `src/` coverage exit. Final Acceptance Gates remain unchecked.
- Next slice: Continue Phase 7.4 Exit Conditions with the acceptance
  harness coverage audit.

#### Run 22 Review Feedback Loop 1

- Reviewer feedback: Blocking gaps remained in the cancellation/close
  latency exit claim. Tool close latency was missing from the close matrix,
  and MCP/A2A/HTTP entries only exercised cleanup helpers rather than the
  actual adapter-level stream paths.
- Follow-up: Extended `tests/server/streaming_latency_budget_test.py` with
  deterministic adapter-level guards. Tool close now measures
  `ToolManager.__aexit__` cascading through `ToolSet.__aexit__` and
  `Tool.__aexit__`. HTTP coverage now drives the Chat SSE and Responses
  SSE route `StreamingResponse.body_iterator` cancellation and close paths.
  MCP coverage now drives `_stream_mcp_response()` cancellation via
  `cancel_event` and close via generator `aclose()`. A2A coverage now
  drives the `/tasks` route stream body iterator for cancellation and
  consumer close, including the translator cleanup path underneath. The
  cancellation/close latency Exit Condition remains checked because these
  named surfaces now all measure below the 1000 ms budget. Residual
  limitation remains live external hosted-provider transport latency; the
  synthetic hosted-provider source proves Avalan-owned normalizer
  cancellation and close behavior without external services.
- Files changed: `tests/server/streaming_latency_budget_test.py` and this
  run log. The ignored checklist `specs/streaming-agenda.md` remains
  checked for only the cancellation/close latency Exit Condition.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/server/streaming_latency_budget_test.py` before lint (2 passed, 24
  subtests passed); `git diff --check` (passed); `make lint` (passed; ruff
  format, black, ruff check --fix, mypy); `poetry run pytest --verbose -s
  tests/server/streaming_latency_budget_test.py` after lint (2 passed, 24
  subtests passed; all measured latencies below 2 ms); `poetry run pytest
  --verbose -s` (6273 passed, 42 skipped, 3285 subtests passed);
  `make test-coverage` (all source files 100%).
- 7.4 Exit Conditions checked: unchanged; only the cancellation/close
  latency Exit Condition is checked.
- Remaining 7.4 boxes unchecked: acceptance harness coverage exit;
  negative/e2e changed-hardening exit; phase-7 benchmark recording exit;
  `src/` coverage exit. Final Acceptance Gates remain unchecked.
- Follow-up review: No blocking findings. Verified tool close latency now
  covers the actual `ToolManager.__aexit__` cascade through `ToolSet` and
  `Tool`, and adapter-level coverage now drives Chat SSE, Responses SSE,
  MCP `_stream_mcp_response()`, and A2A `/tasks` stream body/generator
  cancellation and close paths with cleanup assertions. Reviewer
  verification ran `poetry run pytest --verbose -s
  tests/server/streaming_latency_budget_test.py` (`2 passed, 24 subtests
  passed`) and `git diff --check
  c13ac411d1d430042202a91c068c3d207b056c49..HEAD` (passed). Residual risk:
  deterministic synthetic coverage does not measure live external hosted
  transport latency or the full ASGI server disconnect stack, but it covers
  Avalan-owned cancellation/close behavior at normalizer and route-generator
  boundaries.
- Next slice: Continue Phase 7.4 Exit Conditions with the acceptance
  harness coverage audit.

#### Run 21 Review Feedback Loop 1

- Reviewer feedback: Medium false-positive risk in
  `test_local_stream_latency_and_overhead_within_budget`; the first-answer
  latency measurement started after `STREAM_STARTED`, so startup and the
  first answer could each consume nearly the full
  `time_to_first_item_ms` budget independently.
- Follow-up: Changed the synthetic local latency guard so
  `first_answer_ms` is measured from the original stream start timestamp,
  matching the first-item measurement and asserting cumulative first-answer
  latency against the same budget. The local latency/per-token overhead
  Exit Condition remains checked because the tightened focused run measured
  first item at 0.024 ms, cumulative first answer at 0.043 ms, and local
  normalization at 8.483 us/item against the 5000 ms and 250 us budgets.
- Files changed: `tests/model/model_stream_benchmark_regression_test.py`
  and this run log.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py` (9 passed, 2
  subtests passed; first item 0.024 ms, cumulative first answer 0.043 ms,
  local normalization 8.483 us/item); `git diff --check` (passed);
  `make lint` (passed; ruff format, black, ruff check --fix, mypy);
  `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py` after lint (9
  passed, 2 subtests passed; first item 0.024 ms, cumulative first answer
  0.044 ms, local normalization 8.498 us/item); `poetry run pytest
  --verbose -s` (6271 passed, 42 skipped, 3261 subtests passed);
  `make test-coverage` (all source files 100%).
- Checklist status: The local streaming latency/per-token overhead Exit
  Condition remains checked. No other 7.4 Exit Conditions or Final
  Acceptance Gates were changed.
- Follow-up review: No blocking findings. Verified `first_answer_ms` is
  measured cumulatively from the original stream-start timestamp set before
  the first pull, so the synthetic guard no longer has the split-timer
  false positive. Reviewer verification ran
  `poetry run python -B -m pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py -p
  no:cacheprovider` (`9 passed, 2 subtests passed`; cumulative first
  answer 0.045 ms). Residual risk: this remains a synthetic immediate-token
  benchmark with a generous 5000 ms budget, so real model/provider startup
  latency is covered by the recorded benchmark rows rather than this unit
  guard.
- Next slice: Continue Phase 7.4 Exit Conditions with cancellation and
  close latency across hosted providers, local generation, tools, fanout
  subscribers, MCP, A2A, and HTTP adapters.

### Run 23: 7.4 Acceptance Harness Coverage Exit

- Status: Complete; only the acceptance harness coverage Exit Condition
  is checked. Later Exit Conditions and Final Acceptance Gates remain
  unchecked.
- Task: Audit and complete only this Phase 7.4 Exit Condition if truly
  satisfied: `Acceptance harnesses cover stream equivalence, SDK
  losslessness, UI lossy isolation, provider/local backpressure, live tool
  output, tool ordering, protocol consistency, channel boundaries, event
  stats isolation, and reasoning parser correctness.`
- Notes: Audited the existing acceptance coverage across the named
  dimensions. Existing behavior harnesses cover golden-trace stream
  equivalence, SDK ordered/lossless projection, lossy CLI frame isolation
  without loss to public surfaces, provider and local slow-consumer
  backpressure, live MCP/Responses tool-output streaming, planned tool
  observation ordering, cross-protocol projection consistency, channel
  boundary rejection, event stats isolation/opt-in behavior, and reasoning
  parser plus local-normalizer correctness. Added an inventory guard that
  maps each required dimension to concrete pytest node IDs.
- Files changed: `tests/server/streaming_acceptance_inventory_test.py`,
  this run log, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` before lint (4
  passed, 10 subtests passed); mapped acceptance harness nodes from the
  inventory before lint (22 passed, 11 subtests passed); `git diff
  --check` (passed); `make lint` (passed; ruff format, black, ruff check
  --fix, mypy); `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` plus mapped
  acceptance harness nodes after lint (26 passed, 21 subtests passed);
  `poetry run pytest --verbose -s` (6277 passed, 42 skipped, 3295
  subtests passed); `make test-coverage` (all source files 100%); final
  `git diff --check` (passed).
- Boxes checked: The acceptance harness coverage Exit Condition in the
  ignored workspace checklist.
- Boxes left unchecked: negative/e2e changed-hardening exit; phase-7
  benchmark recording exit; `src/` coverage exit. Final Acceptance Gates
  remain unchecked.
- Next slice: Continue Phase 7.4 Exit Conditions with the negative/e2e
  changed-hardening audit, unless the phase-7 benchmark recording or
  coverage exit is explicitly requested next.

#### Run 23 Review Feedback Loop 1

- Reviewer feedback: The committed inventory guard was weaker than
  intended because it validated mapped node IDs with AST existence only.
  That proved file/function names existed, but not that pytest could
  collect those nodes or that a mapped node had not become skip/xfail-only.
- Follow-up: Replaced the AST-only node check with a subprocess
  `pytest --collect-only -q` probe over the deduplicated mapped node IDs.
  The helper now asserts pytest collection succeeds, the collected node ID
  set exactly matches the requested set, and collected items do not carry
  `skip`, `skipif`, or `xfail` markers. Negative coverage now rejects an
  uncollectable pytest node through the collect-only path and rejects a
  collected item with a skip/xfail marker. The acceptance harness coverage
  Exit Condition remains checked; no other checklist boxes changed.
- Files changed: `tests/server/streaming_acceptance_inventory_test.py` and
  this run log. The ignored workspace checklist
  `specs/streaming-agenda.md` remains checked only for the acceptance
  harness coverage Exit Condition from Run 23.
- Tests and checks run: initial tightened guard run exposed a duplicate
  mapped reference and failed as intended for exact collection; after
  deduplicating collection requests, `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` (5 passed); mapped
  acceptance harness nodes (22 passed, 11 subtests passed); `git diff
  --check` (passed); `make lint` (passed; ruff format, black, ruff check
  --fix, mypy); `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` plus mapped harness
  nodes after lint (27 passed, 11 subtests passed); final `git diff
  --check` (passed); `poetry run pytest --verbose -s` (6278 passed, 42
  skipped, 3285 subtests passed); `make test-coverage` (all source files
  100%).
- Boxes checked: unchanged; only the acceptance harness coverage Exit
  Condition in the ignored workspace checklist.
- Boxes left unchecked: negative/e2e changed-hardening exit; phase-7
  benchmark recording exit; `src/` coverage exit. Final Acceptance Gates
  remain unchecked.
- Follow-up review: No blocking findings. Verified the inventory guard now
  uses real pytest subprocess collection, checks exact deduplicated node ID
  matches, and rejects `skip`, `skipif`, or `xfail` markers. Reviewer
  verification ran `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` (5 passed), the
  mapped acceptance nodes (22 passed, 11 subtests passed), and
  `git diff --check
  d62f15ac0cf27a90e8fb42b37f3c440f25186438..HEAD` (passed). Residual risk:
  collect-only validation would not catch a future unconditional runtime
  `pytest.skip()` inside a mapped test, though the current mapped nodes run
  cleanly; the guard also intentionally rejects any `skipif` marker even if
  its condition is false.
- Next slice: Continue Phase 7.4 Exit Conditions with the negative/e2e
  changed-hardening audit, unless the phase-7 benchmark recording or
  coverage exit is explicitly requested next.

### Run 24: 7.4 Negative And E2E Hardening Exit

- Status: Complete; only the negative/e2e changed-hardening Exit
  Condition is checked. Later Exit Conditions and Final Acceptance Gates
  remain unchecked.
- Worker: xhigh Run24.
- Task: Audit changed Phase 7 hardening behavior and complete only this
  Phase 7.4 Exit Condition if truly satisfied: `Negative and e2e tests
  cover all changed hardening behavior.`
- Notes: Audited the Phase 7 hardening slices from incremental tool
  parsing, static protocol envelopes, optional enrichment gates, lazy
  render cleanup, benchmark/latency guards, retention/server-mode guards,
  classifier guardrails, and acceptance inventory guardrails. Added a
  hardening inventory in `tests/server/streaming_acceptance_inventory_test.py`
  that maps required negative and e2e dimensions to concrete pytest node
  IDs, then verifies those nodes collect exactly and do not carry
  `skip`, `skipif`, or `xfail` markers. The inventory explicitly covers
  malformed tool calls, duplicate terminal items, content after terminal
  state, missing correlation IDs, subscriber overflow, cancellation
  races, retention limits, protocol route e2e surfaces, benchmark and
  latency guard negatives, inventory guard negatives, and the hardening
  added in Runs 20-23. Runtime behavior was unchanged.
- Files changed: `tests/server/streaming_acceptance_inventory_test.py`,
  this run log, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` first failed as
  intended on two stale Chat router node IDs, then passed after correction
  (8 passed); mapped hardening inventory nodes passed before lint (36
  passed, 41 subtests passed); `git diff --check` (passed); `make lint`
  (passed; ruff format, black, ruff check --fix, mypy); post-lint
  `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` (8 passed);
  post-lint mapped hardening inventory nodes (36 passed, 41 subtests
  passed); post-lint `git diff --check` (passed);
  `poetry run pytest --verbose -s` (6281 passed, 42 skipped, 3285
  subtests passed); `make test-coverage` (all source files 100%).
- Boxes checked: The negative/e2e changed-hardening Exit Condition in the
  ignored workspace checklist.
- Boxes left unchecked: phase-7 benchmark recording exit; `src/` coverage
  exit. Final Acceptance Gates remain unchecked.
- Next slice: Continue Phase 7.4 Exit Conditions with phase-7 benchmark
  recording or the `src/` coverage exit, whichever is requested next.

#### Run 24 Review Feedback Loop 1

- Reviewer feedback: Blocking inventory gaps remained. The retention
  bucket omitted A2A retention bounds and A2A final reconstruction under
  retention pressure. The protocol route e2e bucket omitted Chat SSE and
  Responses SSE canonical route projection coverage. The inventory guard
  negatives bucket omitted the new hardening-specific missing-dimension
  negative.
- Follow-up: Added exact mapped nodes for
  `tests/server/a2a_store_test.py::test_task_store_bounds_records_and_histories`,
  `tests/server/test_a2a.py::test_translator_reconstructs_final_text_under_retention_pressure`,
  `tests/server/chat_router_unit_test.py::ChatRouterUnitTest::test_create_chat_completion_streams_canonical_items`,
  `tests/server/create_response_sse_test.py::CreateResponseSSEEventsTestCase::test_streaming_emits_canonical_items`,
  and
  `tests/server/streaming_acceptance_inventory_test.py::StreamingAcceptanceInventoryTestCase::test_streaming_hardening_inventory_rejects_missing_dimension`
  to the relevant hardening inventory buckets. The negative/e2e
  changed-hardening Exit Condition remains checked because the inventory
  now covers the missing A2A retention, Chat SSE, Responses SSE, and
  hardening-negative dimensions.
- Files changed: `tests/server/streaming_acceptance_inventory_test.py`
  and this run log.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` (8 passed);
  mapped hardening inventory nodes (41 passed, 41 subtests passed);
  `git diff --check` (passed); `make lint` (passed; ruff format, black,
  ruff check --fix, mypy); post-lint inventory test (8 passed);
  post-lint mapped hardening inventory nodes (41 passed, 41 subtests
  passed); `poetry run pytest --verbose -s` (6281 passed, 42 skipped,
  3285 subtests passed); `make test-coverage` (all source files 100%).
- Boxes checked: unchanged; only the negative/e2e changed-hardening Exit
  Condition remains checked.
- Boxes left unchecked: phase-7 benchmark recording exit; `src/` coverage
  exit. Final Acceptance Gates remain unchecked.
- Follow-up review: No blocking findings. Verified the added A2A
  retention bounds/final reconstruction, Chat SSE route projection,
  Responses SSE route projection, and hardening-specific missing-dimension
  negative nodes are exact, collectable, and semantically aligned. Reviewer
  verification ran the tightened inventory test (8 passed) and the five
  newly added mapped nodes (5 passed). Residual risk: exact collection and
  skip/xfail marker checks cannot guarantee future semantic adequacy if a
  mapped test body is weakened while keeping the same node ID.
- Next slice: Continue Phase 7.4 Exit Conditions with phase-7 benchmark
  recording or the `src/` coverage exit, whichever is requested next.

### Run 25: 7.4 Phase-7 Benchmark Recording Exit

- Status: Complete; only the phase-7 benchmark recording Exit Condition
  is checked. The `src/` coverage Exit Condition and Final Acceptance
  Gates remain unchecked.
- Worker: xhigh Run25.
- Task: Audit ignored streaming benchmark artifacts and complete only
  this Phase 7.4 Exit Condition if truly satisfied: `Streaming benchmark
  results are recorded as phase-7 in specs/streaming/BENCHMARKS.md and
  compared against the main baseline and prior phase rows, with any
  performance regression explained.`
- Notes: Verified `specs/streaming/latest.json` is a completed
  `phase-7` run from `2026-06-16T06:30:51.716877+00:00` covering
  `openai`, `hermes`, and `gpt-oss`, and verified
  `specs/streaming/BENCHMARKS.md` already contained the corresponding
  phase-7 result table. The benchmark had already been run by Run 18, so
  no rerun was needed. Added the missing comparison text against the
  `main` baseline and prior phase rows, including explicit regression
  notes: OpenAI TTFT/total/throughput regressed versus most comparison
  rows and is documented as hosted provider variance with shorter output
  and no reasoning tokens; Hermes stayed flat or improved; GPT-OSS
  improved TTFT and throughput while the small total-time delta against
  the fastest local rows is explained by longer output/event count and
  single-run noise. No tracked code parses or validates
  `BENCHMARKS.md`, so no new benchmark parsing tests were applicable.
- Files changed: `specs/streaming-agents.md`, the ignored workspace
  checklist `specs/streaming-agenda.md`, and ignored benchmark notes
  `specs/streaming/BENCHMARKS.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_performance_budget_reconciliation_validation`
  (10 passed, 8 subtests passed); `git diff --check` (passed);
  `make lint` (passed; ruff format, black, ruff check --fix, mypy);
  `poetry run pytest --verbose -s` (6281 passed, 42 skipped, 3285
  subtests passed); `make test-coverage` (all source files 100%).
- Boxes checked: The phase-7 benchmark recording Exit Condition in the
  ignored workspace checklist.
- Boxes left unchecked: `src/` coverage Exit Condition. Final Acceptance
  Gates remain unchecked.
- Follow-up review: No blocking findings. Verified the `phase-7`
  benchmark row matches `specs/streaming/latest.json`, explicitly compares
  against `main` and prior phase rows, explains the OpenAI hosted
  regression and local variance, and leaves only the `src/` coverage Exit
  Condition plus Final Acceptance Gates unchecked. Tracked diff for this
  slice is limited to the Run 25 log; ignored benchmark/checklist changes
  remain workspace-only. Review was read-only plus `git diff --check`.
- Next slice: Continue Phase 7.4 Exit Conditions with the `src/`
  coverage exit if requested.

### Run 26: 7.4 Source Coverage Exit

- Status: Complete; all Phase 7.4 Exit Conditions are now checked.
  Final Acceptance Gates remain unchecked and were not started.
- Worker: xhigh Run26.
- Task: Complete only this Phase 7.4 Exit Condition if truly satisfied:
  `src/ coverage remains at 100%.`
- Notes: Ran full `make test-coverage` and inspected the report. Every
  listed `src/avalan/...` file reported `100%`, so no focused test
  additions were needed. Checked only the 7.4 source coverage Exit
  Condition in the ignored workspace checklist.
- Files changed: `specs/streaming-agents.md` and the ignored workspace
  checklist `specs/streaming-agenda.md`.
- Tests and checks run: `make test-coverage` before the checklist edit
  (all source files 100%); `make test-coverage` after the checklist edit
  (all source files 100%); `git diff --check` (passed); `make lint`
  (passed; ruff format, black, ruff check --fix, mypy); `poetry run
  pytest --verbose -s` (6281 passed, 42 skipped, 3285 subtests passed).
- Boxes checked: The `src/` coverage Exit Condition in the ignored
  workspace checklist.
- Boxes left unchecked: Final Acceptance Gates remain unchecked.
- Follow-up review: No blocking findings. Verified the Run 26 diff only
  adds this run log, the ignored checklist has the `src/` coverage Exit
  Condition checked, and Final Acceptance Gates remain unchecked.
  Read-only corroboration from the local coverage artifact showed
  `54941/54941` covered statements, `100.0%`, `0` missing lines, and no
  `src/avalan/...` file below 100%. Residual risk: the reviewer did not
  rerun `make test-coverage` because it writes ignored artifacts; the
  coverage artifact corroborates the worker's two coverage runs but is not
  committed evidence.
- Next slice: Phase 7.4 Exit Conditions are complete. Next work should
  start Final Acceptance Gates only when explicitly requested.

### Run 27: Final Acceptance Stream Equivalence Gate

- Status: Complete; checked only the first Final Acceptance Gate.
- Worker: xhigh Run27.
- Task: Complete only this Final Acceptance Gate if truly satisfied:
  `Streaming accumulation and to_str() produce the same final answer
  semantics.`
- Notes: Audited the canonical accumulation, `TextGenerationResponse.to_str()`,
  async iteration, `OrchestratorResponse`, provider/local normalization,
  usage/tool/reasoning non-answer channels, and the golden/conformance
  traces. Existing coverage already exercised the shared accumulator path,
  wrapper projections, provider/local normalization, usage precedence,
  terminal failures, and public conformance traces. Added focused response
  tests proving final-answer semantics across legacy token streams,
  canonical items, and consumer projections, including exclusion of
  reasoning, tool-call, tool-output, and usage channels from the answer
  text. Added a negative terminal-failure parity test proving errored
  semantic streams are not returned as partial answers by `to_str()`.
  Added the positive parity test to the synthetic acceptance inventory's
  `stream equivalence` dimension.
- Files changed: `tests/model/text_generation_response_more_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`,
  `specs/streaming-agents.md`, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_stream_accumulation_and_to_str_match_answer_semantics
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_to_str_preserves_stream_terminal_failure_semantics
  tests/server/streaming_acceptance_inventory_test.py::StreamingAcceptanceInventoryTestCase::test_streaming_acceptance_inventory_collects_unskipped_tests
  -q` (3 passed, 5 subtests passed); `poetry run pytest
  tests/model/text_generation_response_more_test.py
  tests/server/streaming_acceptance_inventory_test.py -q` (61 passed,
  23 subtests passed); `poetry run pytest
  tests/model/text_generation_response_golden_trace_test.py::TextGenerationResponseGoldenTraceTestCase::test_to_str_matches_accumulated_golden_answer
  tests/model/text_generation_response_golden_trace_test.py::TextGenerationResponseGoldenTraceTestCase::test_canonical_stream_matches_to_str_for_local_response
  tests/model/text_generation_response_golden_trace_test.py::TextGenerationResponseGoldenTraceTestCase::test_protocol_output_traces_capture_current_projections
  tests/server/streaming_conformance_test.py::test_canonical_trace_conforms_across_public_stream_surfaces
  tests/server/streaming_conformance_test.py::test_canonical_trace_rejects_content_after_terminal
  tests/agent/orchestrator_response_test.py::OrchestratorResponseIterationTestCase::test_to_str_emits_streamed_token_events
  tests/agent/orchestrator_response_test.py::OrchestratorResponseIterationTestCase::test_consumer_projections_align_canonical_answer_event_sequence
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_provider_stream_normalizer_assigns_identity_and_metadata
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_maps_legacy_tokens_losslessly
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_legacy_token_canonical_projection_separates_channels
  -q` (10 passed); `git diff --check` (passed); `make lint`
  (passed; ruff format, black, ruff check --fix, mypy); `poetry run
  pytest --verbose -s` (6283 passed, 42 skipped, 3290 subtests passed);
  `make test-coverage` (all listed `src/avalan/...` files 100%).
- Gate checked: `Streaming accumulation and to_str() produce the same
  final answer semantics.`
- Gates left unchecked: All remaining Final Acceptance Gates.
- Follow-up review: Approved with no findings. The reviewer verified the
  Run 27 diff from `17a2a63c3f8f316d57330144e1e6971cad27ebc1` to
  `d0fc10f9462af32283b9ad0a1c46294780d92cbd` is limited to the run log,
  focused `TextGenerationResponse` parity tests, and acceptance inventory
  entry. The added tests cover legacy, canonical, and projected stream
  shapes, exclude reasoning/tool/usage channels from final answer text,
  and verify errored semantic streams preserve partial accumulation while
  `to_str()` raises. The ignored checklist has only the first Final
  Acceptance Gate checked. Reviewer commands included `git diff --check`
  and focused pytest coverage for the new gate tests, inventory, golden
  traces, conformance paths, orchestrator projection paths, and stream
  contract paths; all passed.
- Next gate: `SDK lossless consumers receive every semantic item in
  sequence order.`

### Run 28: Final Acceptance SDK Losslessness Gate

- Status: Complete; checked only the second Final Acceptance Gate.
- Worker: Run28.
- Task: Complete only this Final Acceptance Gate if truly satisfied:
  `SDK lossless consumers receive every semantic item in sequence order.`
- Notes: Audited SDK/lossless consumption through
  `iter_stream_consumer_projections()`, `stream_consumer_iterator()`,
  direct `TextGenerationResponse` async iteration,
  `TextGenerationResponse.consumer_projections()`, orchestrator response
  projection wrapping, and the shared protocol/collector helpers used by
  SDK-facing consumers. Added lossless sequence-gap validation at the SDK
  projection iterator and public consumer-projection wrapper, while
  preserving canonical monotonic validation for protocol projections that
  intentionally expose provider sequence numbers. Added the same gap
  guard to direct semantic async iteration through
  `TextGenerationResponse`. Strengthened tests for every semantic item
  reaching consumers in order, terminal/control delivery, dropped
  sequence-gap rejection, out-of-order rejection, and late semantic item
  rejection. Added the new SDK losslessness regressions to the acceptance
  inventory.
- Files changed: `src/avalan/model/stream.py`,
  `src/avalan/model/response/text.py`,
  `tests/model/model_stream_contract_test.py`,
  `tests/model/text_generation_response_more_test.py`,
  `tests/server/router_streaming_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`,
  `specs/streaming-agents.md`, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_consumer_projection_iterator_preserves_order_losslessly
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_consumer_projection_iterator_rejects_sequence_gap
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_consumer_projections_stream_lossless_items
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_async_iteration_rejects_semantic_sequence_discontinuity
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_async_iteration_closes_on_late_semantic_output
  tests/server/router_streaming_test.py::RouterStreamingTestCase::test_stream_consumer_iterator_prefers_projection_api
  tests/server/router_streaming_test.py::RouterStreamingTestCase::test_stream_consumer_iterator_rejects_projection_sequence_gap
  tests/server/router_streaming_test.py::RouterStreamingTestCase::test_stream_consumer_iterator_rejects_out_of_order_projection
  tests/server/router_streaming_test.py::RouterStreamingTestCase::test_stream_consumer_iterator_rejects_late_projection_item`
  (9 passed, 6 subtests passed); `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py
  tests/model/text_generation_response_more_test.py
  tests/server/router_streaming_test.py
  tests/server/streaming_conformance_test.py
  tests/server/primary_consumer_projection_golden_test.py
  tests/server/protocol_streaming_e2e_test.py` (224 passed, 215 subtests
  passed); `poetry run pytest --verbose -s
  tests/server/chat_router_unit_test.py tests/server/create_response_sse_test.py
  tests/server/mcp_router_test.py tests/server/test_a2a.py
  tests/server/a2a_router_unit_extra_test.py tests/cli/model_test.py`
  (405 passed, 22 subtests passed); `poetry run pytest --verbose -s
  tests/agent/orchestrator_response_test.py
  tests/agent/orchestrator_response_additional_test.py` (118 passed, 12
  subtests passed); `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py::StreamingAcceptanceInventoryTestCase::test_streaming_acceptance_inventory_collects_unskipped_tests`
  (1 passed); `git diff --check` before lint (passed); `make lint`
  (passed; ruff format, black, ruff check --fix, mypy);
  post-lint focused pytest command including the nine focused gate tests
  plus the acceptance inventory test (10 passed, 6 subtests passed);
  `poetry run pytest --verbose -s` (6288 passed, 42 skipped, 3294
  subtests passed); `make test-coverage` (all listed
  `src/avalan/...` files 100%).
- Gate checked: `SDK lossless consumers receive every semantic item in
  sequence order.`
- Gates left unchecked: All later Final Acceptance Gates.
- Follow-up review: Approved with no findings. The reviewer verified the
  Run 28 diff from `46a7bda25c7fefdc710572253a38b1a3e1af6c71` to
  `7badc8368a2fcf25b0d480c8cd737b59351e9b67` is scoped to SDK/lossless
  stream guardrails, focused tests, the acceptance inventory, and the run
  log. Lossless SDK projection paths now reject sequence gaps while
  preserving existing monotonic, terminal, late-item, and legacy fallback
  behavior. Direct `TextGenerationResponse` semantic iteration and public
  `consumer_projections` wrappers are covered for canonical and projected
  semantic items, with negative coverage for dropped/gapped, out-of-order,
  late semantic, and terminal/control cases. The ignored checklist has
  only the first two Final Acceptance Gates checked. Reviewer commands
  included `git diff --check`, the focused gate pytest command, and a
  protocol/conformance pytest command; all passed.
- Next gate: `FancyTheme may coalesce frames without dropping lossless
  subscriber items.`

### Run 29: Final Acceptance FancyTheme Lossy Isolation Gate

- Status: Complete; checked only the third Final Acceptance Gate.
- Worker: Run29.
- Task: Complete only this Final Acceptance Gate if truly satisfied:
  `FancyTheme may coalesce frames without dropping lossless subscriber
  items.`
- Notes: Audited the FancyTheme frame production path,
  `_FrameRateRenderer` latest-frame coalescing, CLI streaming projection
  path, EventManager subscriber classes/defaults, per-subscriber queue
  policies, and canonical stream-item event delivery. Added a FancyTheme
  frame-renderer regression proving multiple real FancyTheme frames may
  coalesce to the latest frame. Added a pressure regression where the
  default UI token subscriber blocks and coalesces under queue pressure
  while an independent lossless token subscriber receives every canonical
  stream item in sequence order. Added both tests to the UI lossy
  isolation acceptance inventory.
- Files changed: `tests/cli/model_test.py`,
  `tests/event/event_manager_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`,
  `specs/streaming-agents.md`, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/cli/model_test.py::CliRenderFrameTestCase::test_frame_rate_renderer_coalesces_fancy_theme_frames
  tests/event/event_manager_test.py::EventManagerTestCase::test_coalescing_ui_stream_subscriber_keeps_lossless_items
  tests/server/streaming_acceptance_inventory_test.py::StreamingAcceptanceInventoryTestCase::test_streaming_acceptance_inventory_collects_unskipped_tests`
  (3 passed); `poetry run pytest --verbose -s
  tests/cli/model_test.py::CliRenderFrameTestCase
  tests/event/event_manager_test.py
  tests/server/protocol_streaming_e2e_test.py
  tests/server/streaming_acceptance_inventory_test.py` (102 passed, 10
  subtests passed); `git diff --check` (passed); `make lint` (passed;
  ruff format, black, ruff check --fix, mypy); post-lint focused pytest
  command for the two new tests plus acceptance inventory (3 passed);
  `poetry run pytest --verbose -s` (6290 passed, 42 skipped, 3294
  subtests passed); `make test-coverage` (all listed `src/avalan/...`
  files 100%).
- Gate checked: `FancyTheme may coalesce frames without dropping lossless
  subscriber items.`
- Gates left unchecked: All later Final Acceptance Gates.
- Follow-up review: Approved with no findings. The reviewer verified the
  Run 29 diff from `6e5e060a7fa250edf2096975ccc9e065058a17ca` to
  `a41f486f02ccaf2c70cc06aa2171548a9c1a41b4` is scoped to gate tests,
  acceptance inventory, and the run log. Real `FancyTheme.tokens()`
  frames exercise `_FrameRateRenderer` latest-frame coalescing, and the
  EventManager pressure regression shows per-subscriber UI coalescing
  does not drop the independent lossless subscriber path. The ignored
  checklist has only the first three Final Acceptance Gates checked.
  Reviewer commands included `git diff --check`, the three focused gate
  tests, and the existing lossy CLI frame e2e test; all passed.
- Next gate: `Local generation blocks or cancels under slow downstream
  pressure rather than losing tokens or growing unbounded queues.`

### Run 30: Final Acceptance Local Generation Backpressure Gate

- Status: Complete; checked only the fourth Final Acceptance Gate.
- Worker: Run30.
- Task: Complete only this Final Acceptance Gate if truly satisfied:
  `Local generation blocks or cancels under slow downstream pressure
  rather than losing tokens or growing unbounded queues.`
- Notes: Audited transformer local streaming, the bounded
  `AsyncTextIteratorStreamer` handoff, SDK-facing local stream
  normalization, DS4 async worker streaming, MLX, and vLLM local stream
  iteration. Added a transformer regression that fills the bounded
  handoff queue, proves the generation worker remains blocked while the
  downstream consumer is paused, and then drains every chunk to prove no
  token loss. Strengthened DS4 streaming coverage to assert token
  generation does not advance during downstream consumer pauses. Added the
  new local-generation checks to the provider/local backpressure
  acceptance inventory.
- Files changed: `tests/model/nlp/text_generation_methods_more_test.py`,
  `tests/model/ds4_model_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`,
  `specs/streaming-agents.md`, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/model/nlp/text_generation_methods_more_test.py::StreamGeneratorTestCase::test_stream_generator_blocks_under_slow_consumer_pressure`
  (1 passed); `poetry run pytest --verbose -s
  tests/model/ds4_model_test.py::test_ds4_stream_queue_backpressure_does_not_deadlock_event_loop`
  (1 passed); `poetry run pytest --verbose -s
  tests/model/nlp/text_generation_methods_more_test.py::StreamGeneratorTestCase::test_lossless_streamer_handoff_blocks_until_consumed
  tests/model/nlp/text_generation_methods_more_test.py::StreamGeneratorTestCase::test_stream_generator_blocks_under_slow_consumer_pressure
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_provider_and_local_streams_wait_for_slow_consumers
  tests/model/ds4_model_test.py::test_ds4_stream_queue_backpressure_does_not_deadlock_event_loop
  tests/server/streaming_acceptance_inventory_test.py::StreamingAcceptanceInventoryTestCase::test_streaming_acceptance_inventory_collects_unskipped_tests`
  (5 passed); `git diff --check` (passed); `make lint` (passed;
  ruff format, black, ruff check --fix, mypy); post-lint focused pytest
  command for the same five gate/inventory tests (5 passed);
  `poetry run pytest --verbose -s` (6291 passed, 42 skipped, 3294
  subtests passed); `make test-coverage` (all listed `src/avalan/...`
  files 100%).
- Gate checked: `Local generation blocks or cancels under slow downstream
  pressure rather than losing tokens or growing unbounded queues.`
- Gates left unchecked: All later Final Acceptance Gates.
- Follow-up review: Approved with no findings. The reviewer verified the
  Run 30 diff from `20343064711b8afa62da01ab73174a1140defda3` to
  `768ca4685e061fe1091577bae6202218823dc580` is scoped to
  transformer/local backpressure coverage, DS4 coverage, acceptance
  inventory, and Run 30 documentation. The transformer regression fills
  the bounded handoff queue and proves generation blocks without dropping
  chunks; the DS4 regression proves pull-based local generation does not
  advance while the downstream consumer is paused. Existing transformer
  cancellation coverage for a blocked handoff still passes. The ignored
  checklist has only the fourth Final Acceptance Gate newly checked.
  Reviewer commands included `git diff --check`, focused gate tests, and
  transformer cancellation tests; all passed.
- Next gate: `Hosted streams avoid Avalan-owned unbounded buffering and
  close promptly on disconnect or cancellation.`

### Run 31: Final Acceptance Hosted Stream Cleanup Gate

- Status: Complete; checked only the fifth Final Acceptance Gate.
- Worker: Run31.
- Task: Complete only this Final Acceptance Gate if truly satisfied:
  `Hosted streams avoid Avalan-owned unbounded buffering and close
  promptly on disconnect or cancellation.`
- Notes: Audited hosted/provider stream normalization and the server-facing
  projections that consume hosted streams. `TextGenerationResponse` now
  propagates `aclose()` and `cancel()` to the upstream hosted stream object
  even when a protocol adapter closes before the first provider token is
  pulled. `TextGenerationVendorStream` now tracks provider stream sources,
  forwards idempotent close/cancel calls, deduplicates cleanup sources, and
  closes provider streams on canonical stream exit. OpenAI-like, Anthropic,
  Bedrock, LiteLLM, Google, Hugging Face, and Ollama stream wrappers pass
  their provider stream sources through this cleanup path. Added negative
  coverage for pre-open SSE/projection disconnect, downstream cancellation
  during a pending provider pull, and no Avalan-owned read-ahead under hosted
  streams.
- Files changed: `src/avalan/model/vendor.py`,
  `src/avalan/model/response/text.py`,
  `src/avalan/model/nlp/text/vendor/openai.py`,
  `src/avalan/model/nlp/text/vendor/anthropic.py`,
  `src/avalan/model/nlp/text/vendor/bedrock.py`,
  `src/avalan/model/nlp/text/vendor/litellm.py`,
  `src/avalan/model/nlp/text/vendor/google.py`,
  `src/avalan/model/nlp/text/vendor/huggingface.py`,
  `src/avalan/model/nlp/text/vendor/ollama.py`,
  `tests/model/text_generation_response_more_test.py`,
  `tests/model/nlp/vendor_openai_test.py`,
  `tests/model/vendor_tool_call_token_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`,
  `specs/streaming-agents.md`, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_hosted_response_preopen_close_closes_provider
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_hosted_response_preopen_cancel_cancels_provider
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_hosted_projection_disconnect_closes_no_read_ahead
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_hosted_projection_cancel_closes_pending_read
  tests/model/nlp/vendor_openai_test.py::OpenAITestCase::test_canonical_disconnect_closes_provider_no_read_ahead
  tests/model/nlp/vendor_openai_test.py::OpenAITestCase::test_canonical_cancel_closes_pending_provider_pull
  tests/model/vendor_tool_call_token_test.py::VendorBuildToolCallTokenTestCase::test_stream_cancel_is_idempotent
  tests/model/vendor_tool_call_token_test.py::VendorBuildToolCallTokenTestCase::test_stream_close_accepts_sync_sources_and_dedupes
  tests/model/vendor_tool_call_token_test.py::VendorBuildToolCallTokenTestCase::test_stream_close_rejects_bad_sync_result
  tests/model/vendor_tool_call_token_test.py::VendorBuildToolCallTokenTestCase::test_stream_close_reports_single_source_error
  tests/model/vendor_tool_call_token_test.py::VendorBuildToolCallTokenTestCase::test_stream_close_reports_multiple_source_errors
  tests/server/streaming_acceptance_inventory_test.py::StreamingAcceptanceInventoryTestCase::test_streaming_acceptance_inventory_collects_unskipped_tests`
  (12 passed); `make lint` (passed; ruff format, black, ruff check --fix,
  mypy); post-lint focused pytest command for the same twelve
  gate/inventory tests (12 passed); `poetry run pytest --verbose -s`
  (6302 passed, 42 skipped, 3294 subtests passed); `make test-coverage`
  (all listed `src/avalan/...` files 100%).
- Gate checked: `Hosted streams avoid Avalan-owned unbounded buffering and
  close promptly on disconnect or cancellation.`
- Gates left unchecked: All later Final Acceptance Gates.
- Follow-up review: Approved with no findings. The reviewer verified the
  Run 31 diff from `9c84b2cad29ebb9466d0c52e30e39cee656cf14e` to
  `f5930991253cf20514beb4e6ff13b093f4d6c092` is scoped to hosted stream
  cleanup guardrails, focused tests, the acceptance inventory, and Run 31
  documentation. Hosted/vendor wrappers pass provider stream sources
  through cleanup and close on canonical stream exit;
  `TextGenerationResponse` propagates `aclose()` and `cancel()` to hosted
  stream objects, including pre-open cleanup. No new Avalan-owned
  unbounded buffering pattern was introduced in the reviewed hosted
  paths, and existing Chat/Responses cleanup still closes or cancels
  sources. The ignored checklist has only the fifth Final Acceptance Gate
  checked. Reviewer commands included `git diff --check`, Run 31 focused
  tests, Chat/Responses cleanup plus latency budget tests,
  `tests/model/vendor_tool_call_token_test.py`, and targeted OpenAI
  stream/tool-call tests; valid reruns passed.
- Next gate: `Live tool stdout, stderr, logs, and progress arrive before
  tool completion.`

### Run 32: Final Acceptance Live Tool Output Gate

- Status: Complete; checked only the sixth Final Acceptance Gate.
- Worker: Run32.
- Task: Complete only this Final Acceptance Gate if truly satisfied:
  `Live tool stdout, stderr, logs, and progress arrive before tool
  completion.`
- Notes: Audited the live tool execution stream paths across canonical
  lifecycle validation, orchestrator tool callbacks, Responses SSE, MCP,
  A2A, SDK-facing conformance fixtures, and e2e protocol projections. The
  strengthened fixtures now carry stdout, stderr, log, and progress items
  before tool completion, and the projections assert those live items are
  emitted before the terminal completion event. Added negative lifecycle
  coverage that rejects live output and progress emitted after tool
  completion.
- Files changed: `tests/agent/orchestrator_response_test.py`,
  `tests/model/model_stream_contract_test.py`,
  `tests/server/create_response_sse_test.py`,
  `tests/server/protocol_streaming_e2e_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`,
  `tests/server/streaming_conformance_test.py`,
  `tests/server/streaming_trace_fixtures.py`,
  `specs/streaming-agents.md`, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/agent/orchestrator_response_test.py::OrchestratorResponseCanonicalLifecycleTestCase::test_iteration_records_live_tool_output_before_completion
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_tool_lifecycle_rejects_live_items_after_completion
  tests/server/create_response_sse_test.py::CreateResponseSSEEventsTestCase::test_streaming_preserves_consecutive_tool_output_metadata
  tests/server/streaming_conformance_test.py::test_canonical_trace_conforms_across_public_stream_surfaces
  tests/server/protocol_streaming_e2e_test.py::test_same_canonical_stream_projects_through_protocols
  tests/server/streaming_acceptance_inventory_test.py::StreamingAcceptanceInventoryTestCase::test_streaming_acceptance_inventory_collects_unskipped_tests`
  (6 passed, 2 subtests passed); `poetry run pytest --verbose -s
  tests/agent/orchestrator_response_test.py
  tests/model/model_stream_contract_test.py
  tests/server/create_response_sse_test.py
  tests/server/streaming_conformance_test.py
  tests/server/protocol_streaming_e2e_test.py
  tests/server/mcp_router_test.py tests/server/test_a2a.py
  tests/server/a2a_router_unit_extra_test.py
  tests/server/streaming_acceptance_inventory_test.py` (458 passed, 194
  subtests passed); `git diff --check` (passed); `make lint` (passed;
  ruff format, black, ruff check --fix, mypy); post-lint focused pytest
  command for the same six gate/inventory tests (6 passed, 2 subtests
  passed); `poetry run pytest --verbose -s` (6303 passed, 42 skipped,
  3297 subtests passed); `make test-coverage` (all listed
  `src/avalan/...` files 100%).
- Gate checked: `Live tool stdout, stderr, logs, and progress arrive
  before tool completion.`
- Gates left unchecked: All later Final Acceptance Gates.
- Follow-up review: Approved with no findings. The reviewer verified the
  Run 32 diff from `458337bc4a5c90938b602340e46b1a364f424145` to
  `4f576f654ec2101951ea34ec57a75df91b82ee1b` is scoped to tests and the
  run log. Live stdout, stderr, log, and progress items are asserted
  before tool completion across canonical/SDK, Responses SSE, MCP, and
  A2A protocol paths. Negative lifecycle coverage rejects live output and
  progress after tool completion. The ignored checklist has only the
  sixth Final Acceptance Gate checked, and later gates remain unchecked.
  Reviewer commands included `git diff --check`, commit subject
  verification, and a focused pytest command with cache disabled; all
  valid checks passed.
- Next gate: `Parallel tools preserve deterministic final model-facing
  observation order by planned call order.`

### Run 33: Final Acceptance Parallel Tool Ordering Gate

- Status: Complete; checked only the seventh Final Acceptance Gate.
- Worker: Run33.
- Task: Complete only this Final Acceptance Gate if truly satisfied:
  `Parallel tools preserve deterministic final model-facing observation
  order by planned call order.`
- Notes: Audited the parallel tool execution and observation assembly
  paths across planned tool-call indexing, concurrent completion order,
  canonical lifecycle collection, streaming iteration result staging,
  model continuation prompt construction, and acceptance inventory
  coverage. The existing orchestration code executes parallel-safe tools
  concurrently while sorting final outcomes by `planned_index` before
  assembling model-facing observation messages. Strengthened tests now
  assert that canonical terminal completion order may differ from planned
  order, while final canonical observations and continuation messages
  remain planned-order deterministic. Added negative observation assembly
  coverage for missing and unexpected planned tool-call IDs.
- Files changed: `tests/agent/orchestrator_response_test.py`,
  `tests/model/model_stream_contract_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`,
  `specs/streaming-agents.md`, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_tool_lifecycle_assembles_planned_order_observations
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_tool_lifecycle_observation_assembly_rejects_planned_mismatches
  tests/agent/orchestrator_response_test.py::OrchestratorResponseCanonicalLifecycleTestCase::test_iteration_parallel_results_emit_before_continuation
  tests/server/streaming_acceptance_inventory_test.py::StreamingAcceptanceInventoryTestCase::test_streaming_acceptance_inventory_collects_unskipped_tests`
  (4 passed, 2 subtests passed); `poetry run pytest --verbose -s
  tests/agent/orchestrator_response_test.py
  tests/model/model_stream_contract_test.py
  tests/server/streaming_acceptance_inventory_test.py
  tests/server/test_a2a.py::test_canonical_tool_completion_keeps_task_working_until_terminal`
  (227 passed, 186 subtests passed); `git diff --check` (passed);
  `make lint` (passed; ruff format, black, ruff check --fix, and mypy);
  post-lint focused pytest command for the same four gate/inventory tests
  (4 passed, 2 subtests passed); `poetry run pytest --verbose -s` (6304
  passed, 42 skipped, 3299 subtests passed); `make test-coverage` (all
  listed `src/avalan/...` files 100%).
- Gate checked: `Parallel tools preserve deterministic final
  model-facing observation order by planned call order.`
- Gates left unchecked: All later Final Acceptance Gates.
- Follow-up review: Approved with no findings. The reviewer verified the
  Run 33 diff from `f6c5b6c9d25feba87547a9756e71e4a50fad8a0b` to
  `18f72524a33cc6895184639c6ec5f4a6f209a97d` is scoped to tests and the
  run log. Completion order differs from planned order while planned-order
  observations remain deterministic in the stream contract tests, and the
  orchestrator continuation message assembly is covered directly. The
  orchestrator test also proves canonical completion order differs from
  planned order, while continuation tool messages are planned-order
  deterministic. Missing and unexpected planned tool-call ID negatives
  are covered, with duplicate planned ID coverage still nearby. The
  ignored checklist has only the seventh Final Acceptance Gate checked.
  Reviewer commands included `git diff --check`, commit subject
  verification, and focused pytest with cache disabled; all passed.
- Next gate: `Responses SSE, Chat SSE, MCP, A2A, flow, stdout, CLI, and
  SDK expose protocol-appropriate projections of the same underlying
  stream.`

### Run 34: Final Acceptance Cross-Protocol Projection Gate

- Status: Complete; checked only the eighth Final Acceptance Gate.
- Worker: Run34.
- Task: Complete only this Final Acceptance Gate if truly satisfied:
  `Responses SSE, Chat SSE, MCP, A2A, flow, stdout, CLI, and SDK expose
  protocol-appropriate projections of the same underlying stream.`
- Notes: Audited the shared canonical trace coverage for Responses SSE,
  Chat SSE, MCP, A2A, flow, stdout, CLI, and SDK projections. Strengthened
  `tests/server/streaming_conformance_test.py` so the same golden trace now
  asserts SDK identity/sequence preservation, stdout answer-only output,
  CLI lossless render projection over every canonical item plus CLI
  rendering buckets, Chat SSE answer/usage/control behavior, Responses SSE
  reasoning/tool/usage/completion events, MCP structured final output and
  live resource ordering, A2A answer/reasoning/tool artifacts and terminal
  status, and flow payload/metadata projection. Added negative coverage for
  missing stream terminals and unsupported Chat SSE, Responses SSE, and CLI
  projection inputs.
- Files changed: `tests/server/streaming_conformance_test.py`,
  `specs/streaming-agents.md`, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/server/streaming_conformance_test.py
  tests/server/protocol_streaming_e2e_test.py
  tests/server/primary_consumer_projection_golden_test.py` (8 passed);
  `git diff --check` (passed); `make lint` (passed; `ruff format`,
  `black`, `ruff check --fix`, and `mypy` completed successfully);
  post-lint focused pytest command for the same three files (8 passed);
  post-lint `git diff --check` (passed); `poetry run pytest --verbose -s`
  (6306 passed, 42 skipped, 3299 subtests passed); `make test-coverage`
  (all listed `src/avalan/...` files 100%).
- Gate checked: `Responses SSE, Chat SSE, MCP, A2A, flow, stdout, CLI,
  and SDK expose protocol-appropriate projections of the same underlying
  stream.`
- Gates left unchecked: All later Final Acceptance Gates.
- Follow-up review: Approved with no findings. The reviewer verified the
  Run 34 diff from `5e5d11e8e542d0a803290ad5982e012cd1897166` to
  `868a7e64913214a32aae570ec90183004c127dbd` is scoped to
  `tests/server/streaming_conformance_test.py` and the run log. The
  conformance trace covers answer, reasoning, tool call/execution, flow,
  usage, completion, and closed semantics, and projects the same canonical
  stream through SDK, stdout, CLI, Chat SSE, Responses SSE, MCP, and A2A
  paths. Negative coverage for late content, missing terminal, and
  unsupported Chat/Responses/CLI projection inputs is present. The ignored
  checklist has only the eighth Final Acceptance Gate checked. Reviewer
  commands included `git diff --check` and focused pytest over the
  conformance, protocol e2e, and primary projection suites; all passed.
- Next gate: `Channel done items are emitted at deterministic native-end,
  parsed-end, channel-transition, or stream-end boundaries.`

### Run 35: Final Acceptance Channel Done Boundary Gate

- Status: Complete; checked only the ninth Final Acceptance Gate.
- Worker: Run35.
- Task: Complete only this Final Acceptance Gate if truly satisfied:
  `Channel done items are emitted at deterministic native-end, parsed-end,
  channel-transition, or stream-end boundaries.`
- Notes: Audited channel closure behavior for answer, reasoning, tool-call,
  tool execution, model continuation, usage, terminal, closed, parsed local
  tool-call/reasoning markers, provider-native done items, legacy
  TextGenerationResponse finalization, orchestrator canonical wrapping,
  Responses SSE, MCP, A2A, and OpenAI Responses function-call item IDs.
  Strengthened canonical validation so final usage, stream terminal, and
  completion validation reject open answer/reasoning/tool-call/tool
  execution/model-continuation channels; duplicate done, content after done,
  terminal before start, missing terminal, and post-final-usage content
  negatives are covered. Normalizers and adapters now emit close items at
  deterministic native-end, parsed-end, channel-transition, final-usage,
  stream-end, or terminal/error boundaries, including malformed/error
  tool-call closes marked with close metadata.
- Files changed: `src/avalan/model/stream.py`,
  `src/avalan/model/response/text.py`,
  `src/avalan/model/nlp/text/vendor/openai.py`,
  `src/avalan/agent/orchestrator/response/orchestrator_response.py`,
  `src/avalan/server/routers/mcp.py`,
  `tests/model/model_stream_contract_test.py`,
  `tests/model/nlp/vendor_openai_test.py`,
  `tests/server/create_response_sse_test.py`,
  `tests/server/mcp_router_test.py`, `tests/server/test_a2a.py`,
  `specs/streaming-agents.md`, and the ignored workspace checklist
  `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest -q
  tests/model/model_stream_contract_test.py
  tests/agent/orchestrator_response_test.py::OrchestratorResponseCanonicalLifecycleTestCase
  tests/agent/orchestrator_response_additional_test.py
  tests/server/create_response_sse_test.py tests/server/mcp_router_test.py
  tests/server/test_a2a.py tests/model/nlp/vendor_openai_test.py
  tests/model/nlp/vendor_anthropic_test.py tests/model/nlp/vendor_bedrock_test.py
  tests/model/nlp/vendor_extra_test.py
  tests/server/streaming_conformance_test.py
  tests/server/protocol_streaming_e2e_test.py
  tests/server/router_streaming_test.py tests/server/responses_utils_test.py`
  (661 passed, 261 subtests passed). Required full-check results are
  recorded after completion below.
- Review feedback loop 1: Addressed the P1 finding that
  `validate_canonical_stream_items()` accepted `ANSWER_DONE` or
  `REASONING_DONE` before matching channel content. Canonical validation now
  rejects premature answer/reasoning done items, provider normalization maps
  that invalid native event directly to a stream error without emitting the
  invalid done item, and legacy response finalization only emits
  answer/reasoning done items for channels that produced content.
- Feedback loop 1 focused checks: `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py` (127 passed, 209 subtests
  passed) and `poetry run pytest --verbose -s
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_async_iteration_finalizes_legacy_canonical_accumulator`
  (1 passed). After including the remaining CLI canonical fixture updates,
  `poetry run pytest --verbose -s tests/cli/model_test.py` passed (129
  passed, 6 subtests passed). `poetry run pytest --verbose -s
  tests/server/chat_router_unit_test.py` passed (48 passed, 6 subtests
  passed). `poetry run pytest --verbose -s
  tests/model/text_generation_response_more_test.py` passed (58 passed, 27
  subtests passed).
- Review feedback loop 2: Addressed the re-review finding that golden trace
  fixture updates were still unstaged and absent from signed HEAD. Reviewed
  and included the minimal golden trace boundary fixture changes for
  tool-call ready/done, reasoning done, answer done sequence shifts, and MCP
  tool execution start-before-output ordering, plus the pending CLI
  assertion formatting cleanup.
- Feedback loop 2 focused checks: `poetry run pytest -q
  tests/model/text_generation_response_golden_trace_test.py` passed (12
  passed), and `poetry run pytest -q
  tests/cli/model_test.py::CliTokenGenerationTestCase::test_stream_render_items_projects_consumer_projections`
  passed (1 passed).
- Final reviewer approval: Franklin re-reviewed signed HEAD
  `b17d1ac543d8f44ae5215eb06c28dea74d504099`, found no remaining issues,
  verified premature answer/reasoning done items reject while valid opened
  closures pass, confirmed golden trace fixture updates are committed, and
  confirmed the tracked worktree was clean. Reviewer checks included
  `git diff --check`, `git diff --check
  d44a298c6b3cf363d21d124c4ae17de7ebe6a82e..HEAD`,
  `poetry run pytest -q tests/model/text_generation_response_golden_trace_test.py`
  (12 passed), and `poetry run pytest -q
  tests/model/model_stream_contract_test.py
  tests/model/text_generation_response_more_test.py
  tests/cli/model_test.py::CliTokenGenerationTestCase::test_stream_render_items_projects_consumer_projections`
  (186 passed, 236 subtests passed).
- Review feedback loop 3: Addressed Bacon's P2 finding that
  `CanonicalStreamAccumulator._validate_next()` set `_terminal_outcome`
  for stream terminal items before checking open channel boundaries. The
  accumulator incremental path now rejects completed, errored, and
  cancelled stream terminals while answer, reasoning, tool-call, tool
  execution, or model-continuation channels remain open, matching the tuple
  validator before mutating terminal state. Projection-state accumulation
  now fails at the same boundary, and legacy mapped projection fixtures now
  include explicit answer done boundaries before stream completion.
- Feedback loop 3 focused checks: `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_accumulator_rejects_stream_terminal_with_open_channels
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_stream_projection_state_rejects_terminal_with_open_channel`
  passed (2 passed, 18 subtests passed). `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py` passed (132 passed, 230
  subtests passed).
- Final reviewer approval after loop 3: Bacon re-reviewed signed HEAD
  `3edb3aa5fddedcae70105f261c773cdfba1a847e`, found no remaining
  issues, confirmed the accumulator now checks open channel boundaries
  before setting terminal outcome state, and confirmed the projection
  accumulator path is covered. Reviewer checks included `git diff --check
  d44a298c6b3cf363d21d124c4ae17de7ebe6a82e..HEAD` (passed), the focused
  accumulator/projection regressions (2 passed, 18 subtests passed),
  `tests/model/model_stream_contract_test.py` (132 passed, 230 subtests
  passed), signature verification, and clean tracked status.
- Gate checked: `Channel done items are emitted at deterministic
  native-end, parsed-end, channel-transition, or stream-end boundaries.`
- Gates left unchecked: All later Final Acceptance Gates.
- Next gate: `Event stats and histories are instance-scoped and bounded.`

### Run 36: Final Acceptance Event Stats And History Scope Gate

- Status: Complete; checked only the tenth Final Acceptance Gate.
- Worker: Run36.
- Task: Complete only this Final Acceptance Gate if truly satisfied:
  `Event stats and histories are instance-scoped and bounded.`
- Notes: Audited event stats/history behavior across `EventManager`,
  server request defaults, CLI/server loader modes, and canonical stream
  accumulator retention. `EventManager` stores subscribers, listen queues,
  history, history bytes, and stats on each instance; SDK/CLI/test modes
  use bounded histories and queues, and server mode disables history,
  listen queues, and stats by default while keeping explicit opt-in
  bounded. Histories are bounded by count, bytes, TTL, or disabled
  settings; listener queues and listen queues are bounded by queue limits
  and drop/coalesce/fail-closed policy. Stream retained views are
  accumulator-instance state and bounded by `StreamRetentionPolicy`.
  Server and engine loaders pass `EventManagerMode.SERVER` for server
  orchestrators, while CLI paths use `EventManagerMode.CLI`.
- Tests strengthened: added EventManager regressions for cross-manager
  counter/history/listener/queue isolation and listener-less burst
  retention; added a stream accumulator instance-isolation regression for
  retained histories; strengthened repeated listener-less Responses server
  request tests to assert no hidden history bytes or delivery queue
  entries remain.
- Late included cleanup present in the current signed HEAD:
  `src/avalan/model/stream.py` and
  `tests/model/model_stream_contract_test.py` contain channel-boundary and
  final-usage validation cleanup. That work is outside the Run36 event
  stats/history gate, but it is present in the current amended HEAD and is
  no longer an unstaged leftover.
- Files changed by Run36: `tests/event/event_manager_test.py`,
  `tests/model/model_stream_retention_isolation_test.py`,
  `tests/server/responses_test.py`, `specs/streaming-agents.md`, and the
  ignored checklist `specs/streaming-agenda.md`.
- Tests and checks run: `poetry run pytest --verbose -s
  tests/event/event_manager_test.py tests/model/model_stream_contract_test.py
  tests/server/responses_test.py` (222 passed, 222 subtests passed);
  `git diff --check` (passed); `make lint` (passed; `ruff format`,
  `black`, `ruff check --fix`, and `mypy` completed successfully);
  post-lint `poetry run pytest --verbose -s
  tests/event/event_manager_test.py
  tests/model/model_stream_retention_isolation_test.py
  tests/server/responses_test.py` (92 passed, 10 subtests passed);
  post-lint `git diff --check` (passed); `poetry run pytest --verbose -s`
  (6314 passed, 42 skipped, 3325 subtests passed); `make test-coverage`
  (all listed `src/avalan/...` files 100%).
- Gate checked: `Event stats and histories are instance-scoped and
  bounded.`
- Gates left unchecked: All later Final Acceptance Gates.
- Next gate: `Reasoning parsing preserves whitespace and detects markers
  across token boundaries.`
- Review feedback loop 1: Reviewer found the Run36 gate complete but
  requested documentation cleanup because current signed HEAD
  `68648c5531d214b7e2d9bdc37c36ab85da797719` includes
  `src/avalan/model/stream.py` and
  `tests/model/model_stream_contract_test.py`, while the Run36 log still
  described them as unstaged leftovers from
  `25fabe8f918cdbebaf4cc8ce28ba25148c41d805`. Resolved by documenting
  those files honestly as late included channel-boundary/final-usage
  validation cleanup outside the Run36 event stats/history gate. No next
  gate work was done.
- Review feedback loop 1 checks: `git diff --check` (passed);
  `poetry run pytest --verbose -s
  tests/model/model_stream_contract_test.py
  tests/event/event_manager_test.py
  tests/model/model_stream_retention_isolation_test.py
  tests/server/responses_test.py` (222 passed, 222 subtests passed).
- Final reviewer approval: Epicurus re-reviewed signed HEAD
  `3b6514cd5fe26a5155f7593c5d39c8dc09fd8079`, found no remaining
  issues, confirmed the documentation/scope cleanup is accurate, and
  confirmed the event stats/history gate is complete with later final
  gates unchecked. Reviewer checks included `git diff --check` (passed),
  `PYTHONDONTWRITEBYTECODE=1 poetry run pytest -p no:cacheprovider
  --verbose -s tests/event/event_manager_test.py
  tests/model/model_stream_retention_isolation_test.py
  tests/server/responses_test.py tests/model/model_stream_contract_test.py`
  (222 passed, 222 subtests passed), and `git status --short` (clean).

### Run 37: Final Acceptance Reasoning Parser Boundary Gate

- Status: Complete; checked only the eleventh Final Acceptance Gate.
- Worker: Anscombe (`019ed0fc-72fb-7dc0-9a1e-7783c725c877`).
- Task: Complete only this Final Acceptance Gate if truly satisfied:
  `Reasoning parsing preserves whitespace and detects markers across token
  boundaries.`
- Notes: Audited canonical local reasoning parsing, legacy
  `ReasoningParser` token parsing, `TextGenerationResponse` canonical and
  projection paths, legacy token-to-canonical normalization, and hosted
  OpenAI reasoning deltas. Fixed legacy marker parsing so raw tokens are
  split into answer-side prefix/suffix and private marker/content fragments
  without trimming whitespace or accepting stripped false positives. Fixed
  canonical local parsing for empty reasoning markers by emitting an empty
  reasoning delta before the parsed done boundary, preserving the
  done-before-content validator for arbitrary streams. Legacy
  `ReasoningToken` streams now close reasoning before transitioning to
  answer/tool channels.
- Tests strengthened: added split-marker, split-marker-with-empty-chunk,
  one-character marker, whitespace-preservation, empty/whitespace-only
  reasoning, malformed marker, adjacent/repeated reasoning sections,
  legacy parser,
  `TextGenerationResponse.to_str()`, consumer projection, hosted OpenAI
  native reasoning, and acceptance-inventory coverage.
- Files changed: `src/avalan/model/response/parsers/reasoning.py`,
  `src/avalan/model/response/text.py`,
  `src/avalan/model/stream.py`,
  `tests/agent/reasoning_parser_edge_test.py`,
  `tests/agent/reasoning_parser_split_tag_test.py`,
  `tests/model/model_stream_contract_test.py`,
  `tests/model/nlp/vendor_openai_test.py`,
  `tests/model/response_parsers_additional_test.py`,
  `tests/model/text_generation_response_golden_trace_test.py`,
  `tests/model/text_generation_response_more_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`,
  `specs/streaming-agents.md`, and the ignored checklist
  `specs/streaming-agenda.md`.
- Tests and checks run:
  `poetry run pytest --verbose -s
  tests/model/response_parsers_additional_test.py::ReasoningParserAdditionalTestCase
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_parses_split_reasoning_tags
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_preserves_split_marker_whitespace
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_detects_character_split_reasoning_tags
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_handles_empty_reasoning_markers
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_closes_unterminated_reasoning
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_keeps_partial_end_marker_reasoning
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_keeps_malformed_markers_as_answer
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_to_str_preserves_split_reasoning_answer_whitespace
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_consumer_projections_preserve_split_reasoning_whitespace`
  (18 passed, 8 subtests passed);
  `poetry run pytest --verbose -s
  tests/model/text_generation_response_golden_trace_test.py
  tests/model/text_generation_response_more_test.py
  tests/model/text_generation_response_full_test.py
  tests/model/text_generation_response_additional_test.py
  tests/model/text_generation_response_extra_test.py
  tests/model/response_parsers_additional_test.py
  tests/model/model_stream_contract_test.py` (237 passed, 267 subtests
  passed); `poetry run pytest --verbose -s
  tests/agent/reasoning_parser_edge_test.py
  tests/agent/reasoning_parser_split_tag_test.py
  tests/model/nlp/vendor_openai_test.py::OpenAITestCase::test_canonical_stream_preserves_hosted_reasoning_whitespace
  tests/server/streaming_acceptance_inventory_test.py` (22 passed).
  Final focused rerun: `poetry run pytest --verbose -s
  tests/agent/reasoning_parser_split_tag_test.py
  tests/agent/reasoning_parser_edge_test.py
  tests/model/model_stream_contract_test.py
  tests/model/nlp/vendor_openai_test.py
  tests/model/response_parsers_additional_test.py
  tests/model/text_generation_response_golden_trace_test.py
  tests/model/text_generation_response_more_test.py
  tests/server/streaming_acceptance_inventory_test.py` (334 passed,
  279 subtests passed); `git diff --check` passed; `make lint` passed
  (`ruff format`, `black`, `ruff check --fix`, `mypy`); final full
  `poetry run pytest --verbose -s` passed (6332 passed, 42 skipped,
  3351 subtests passed); final `make test-coverage` passed with every
  listed `src/avalan/...` file at 100%.
- Review feedback loop 1: fixed empty legacy parser chunks so `""` is a
  no-op while a pending split marker remains pending. Added direct parser
  regression coverage for empty chunks between marker pieces, a
  whitespace-chunk negative that keeps malformed marker text visible, and a
  `TextGenerationResponse.to_str()` regression for
  `alpha <thi`, `""`, `nk>hidden</think> omega`. Follow-up tests:
  targeted empty-chunk regressions (3 passed); current changed-file focused
  suite recorded above.
- Review feedback loop 2: Einstein found adjacent repeated reasoning
  sections still leaked markers in legacy parsing and produced invalid
  canonical local streams. Removed the post-budget bypass from the legacy
  marker parser and `TextGenerationResponse` so explicit later markers are
  still parsed, and changed local text normalization to hold
  `REASONING_DONE` while an immediately adjacent reasoning section may
  continue the same private channel. Whitespace-only gaps between adjacent
  private sections remain private; false repeated markers still close the
  reasoning channel before public answer text. Added direct parser,
  `TextGenerationResponse.to_str()`, consumer projection, raw local
  normalization, split-token, whitespace-gap, and false-positive regression
  coverage for `x<think>a</think><think>b</think>y` and related cases. Checks:
  focused `poetry run pytest --verbose -s
  tests/model/response_parsers_additional_test.py::ReasoningParserAdditionalTestCase
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_handles_adjacent_reasoning_sections
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_handles_split_adjacent_sections
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_keeps_adjacent_gap_private
  tests/model/model_stream_contract_test.py::StreamContractTestCase::test_local_stream_normalizer_closes_before_false_repeated_marker
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_to_str_handles_adjacent_reasoning_sections
  tests/model/text_generation_response_more_test.py::TextGenerationResponseMoreTestCase::test_consumer_projections_handle_adjacent_reasoning_sections
  tests/server/streaming_acceptance_inventory_test.py` (28 passed);
  post-lint broader parser/model response suite (272 passed, 267 subtests
  passed); current changed-file focused suite
  `poetry run pytest --verbose -s
  tests/agent/reasoning_parser_split_tag_test.py
  tests/agent/reasoning_parser_edge_test.py
  tests/model/model_stream_contract_test.py
  tests/model/nlp/vendor_openai_test.py
  tests/model/response_parsers_additional_test.py
  tests/model/text_generation_response_golden_trace_test.py
  tests/model/text_generation_response_more_test.py
  tests/server/streaming_acceptance_inventory_test.py` (334 passed,
  279 subtests passed); `make lint` passed; `git diff --check` passed;
  `make test-coverage` passed with every listed `src/avalan/...` file at
  100%.
- Final reviewer approval after loop 2: Einstein re-reviewed signed HEAD
  `56ad517056e98146c79686b670aec6584c6d7118`, found no remaining
  issues, and approved the reasoning parsing gate. Manual probes confirmed
  adjacent sections, split-adjacent sections, public whitespace after
  reasoning, false repeated markers, and empty-adjacent reasoning keep
  markers out of answer text while preserving answer whitespace. Reviewer
  checks included the focused gate suite (28 passed),
  `git diff --check
  dbbe910b815b22143c6af8bba4c015fffe651131..HEAD` (passed), and clean
  tracked status.
- Final reviewer approval: Maxwell re-reviewed signed HEAD
  `ffc111456691d868e878b49a04eb6c1753592aaf`, found no remaining issues,
  and confirmed the reasoning parser gate is complete. Reviewer checks
  included `git diff --check`, `git diff --check
  d137211c788dcfdb4462fac5e365400a5cbb9da0..HEAD`, the focused changed-file
  suite (334 passed, 279 subtests passed), and direct probes for empty
  chunk split markers, adjacent reasoning sections, whitespace-only
  adjacent gaps, and false repeated markers.
- Gate checked: `Reasoning parsing preserves whitespace and detects
  markers across token boundaries.`
- Gates left unchecked: All later Final Acceptance Gates.
- Next gate: `Memory stays bounded across repeated server requests with no
  UI listener attached.`

### Run 38: Final Acceptance Server No-Listener Memory Gate

- Status: Complete; checked only the next Final Acceptance Gate.
- Worker: Run38.
- Task: Complete only
  `Memory stays bounded across repeated server requests with no UI listener
  attached.`
- Notes: Audited Chat SSE, Responses SSE, MCP, A2A, server
  orchestrator/event-manager defaults, request-local protocol projection
  state, canonical accumulators, resource/task retention, and
  cancellation/disconnect cleanup. Fixed successful MCP stream cleanup so
  completed `_stream_mcp_response()` closes its response and iterator before
  syncing final messages, matching the cancellation and error paths.
  Strengthened route-level repeated-request coverage for Chat SSE and A2A,
  Responses source release coverage, shared MCP resource-store bounds and
  normal source close, and the acceptance inventory dimension for
  no-listener server memory bounds, including server-mode DI defaults.
- Review feedback loop 1: Reviewer found that successful MCP cleanup could
  be skipped when a client closed the async generator after the terminal
  closed-resource notification but before the final JSON-RPC result.
  Follow-up pre-collects terminal MCP close notifications/result messages,
  closes the successful response source before yielding terminal payloads,
  and keeps terminal branch sync in `finally` so mid-terminal `aclose()`
  still runs cleanup/sync semantics. Added regressions that consume
  through the terminal closed-resource notification before the final result
  payload and through the final result before normal exhaustion, then close
  the stream and assert source close, no unintended cancellation, resource
  closure/bounds, and exactly-once sync.
- Feedback loop 1 files changed: `src/avalan/server/routers/mcp.py`,
  `tests/server/mcp_router_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`, and
  `specs/streaming-agents.md`.
- Files changed: `src/avalan/server/routers/mcp.py`,
  `tests/server/chat_router_unit_test.py`,
  `tests/server/create_response_sse_test.py`,
  `tests/server/mcp_router_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`,
  `tests/server/test_a2a.py`, `specs/streaming-agents.md`, and ignored
  checklist `specs/streaming-agenda.md`.
- Tests and checks run:
  - Focused new gate tests:
    `poetry run pytest --verbose -s
    tests/server/chat_router_unit_test.py::ChatRouterUnitTest::test_repeated_chat_stream_requests_bound_no_listener_state
    tests/server/test_a2a.py::test_repeated_create_task_stream_requests_bound_state_without_ui_listener
    tests/server/streaming_acceptance_inventory_test.py::StreamingAcceptanceInventoryTestCase::test_streaming_acceptance_inventory_covers_required_dimensions
    tests/server/streaming_acceptance_inventory_test.py::StreamingAcceptanceInventoryTestCase::test_streaming_acceptance_inventory_collects_unskipped_tests`
    (5 passed).
  - Focused gate suite:
    `poetry run pytest --verbose -s
    tests/server/mcp_router_test.py::MCPRouterAsyncTestCase::test_repeated_stream_responses_bound_shared_resource_store
    tests/server/mcp_router_test.py::MCPRouterAsyncTestCase::test_repeated_streams_bound_resource_state_and_keep_result
    tests/server/chat_router_unit_test.py::ChatRouterUnitTest::test_repeated_chat_stream_requests_bound_no_listener_state
    tests/server/responses_test.py::ResponsesEndpointTestCase::test_repeated_requests_without_ui_listener_do_not_retain_events
    tests/server/responses_test.py::ResponsesEndpointTestCase::test_repeated_streaming_requests_do_not_retain_events
    tests/server/create_response_sse_test.py::CreateResponseSSEEventsTestCase::test_repeated_response_stream_requests_release_sources
    tests/server/test_a2a.py::test_repeated_create_task_stream_requests_bound_state_without_ui_listener
    tests/server/router_streaming_test.py::RouterStreamingTestCase::test_default_server_stream_retention_surfaces_are_bounded
    tests/server/streaming_acceptance_inventory_test.py::StreamingAcceptanceInventoryTestCase::test_streaming_acceptance_inventory_collects_unskipped_tests`
    (9 passed, 5 subtests passed).
  - Broader touched server suite:
    `poetry run pytest --verbose -s tests/server/chat_router_unit_test.py
    tests/server/test_a2a.py tests/server/create_response_sse_test.py
    tests/server/responses_test.py
    tests/server/streaming_acceptance_inventory_test.py
    tests/server/mcp_router_test.py tests/server/router_streaming_test.py`
    (301 passed, 23 subtests passed).
  - `git diff --check` passed.
  - `make lint` passed (`ruff format`, `black`, `ruff check --fix`,
    `mypy`).
  - Full `poetry run pytest --verbose -s` passed (6347 passed, 42 skipped,
    3351 subtests passed).
  - `make test-coverage` passed with every listed `src/avalan/...` file at
    100%.
  - Feedback loop 1 focused regression:
    `poetry run pytest
    tests/server/mcp_router_test.py::MCPRouterAsyncTestCase::test_stream_response_cleans_up_when_terminal_emit_closes
    -q` (1 passed).
  - Feedback loop 1 focused MCP suite:
    `poetry run pytest tests/server/mcp_router_test.py -q` (138 passed,
    6 subtests passed).
  - Feedback loop 1 touched gate inventory:
    `poetry run pytest tests/server/streaming_acceptance_inventory_test.py
    -q` (8 passed).
  - Feedback loop 1 `git diff --check` passed.
  - Feedback loop 1 `make lint` passed (`ruff format`, `black`,
    `ruff check --fix`, `mypy`).
- Feedback loop 1 agenda verification: `specs/streaming-agenda.md` still
  has the memory-bounded gate checked and all later Final Acceptance Gates
  unchecked.
- Final reviewer approval after loop 1: Carson re-reviewed signed HEAD
  `9cd6e70b5b66047d1b9afac282477a4abc77e0c7`, found no remaining
  issues, and approved the no-listener memory gate. The reviewer confirmed
  successful MCP streams now close response sources before terminal
  payload emission, terminal sync is protected by `finally`, final
  JSON-RPC reconstruction and close notifications still work for fully
  drained requests, and terminal-boundary close cases are covered.
  Reviewer checks included the focused MCP boundary/inventory tests (4
  passed), `git diff --check
  0c324f5f0046f4e868ee6903186a2624b48d6dbd..HEAD` (passed), and clean
  tracked status.
- Final reviewer approval: Descartes re-reviewed signed HEAD
  `9cd6e70b5b66047d1b9afac282477a4abc77e0c7`, found no remaining issues,
  and confirmed the memory-bounded repeated server request gate is
  complete. Reviewer checks included `git diff --check`, `git diff --check
  27af1601ae35d9c95611f922df1e2ea0f1bc0dba..HEAD`, `git diff --check
  9c71f004154c17913d09461780128230cc1b3f21..HEAD`, focused MCP
  regressions/inventory (5 passed), full touched MCP and inventory suites
  (146 passed, 6 subtests passed), and a direct mid-terminal close probe
  confirming source close and sync occur once without cancellation.
- Gate checked: `Memory stays bounded across repeated server requests with
  no UI listener attached.`
- Gates left unchecked: All later Final Acceptance Gates.
- Next gate: `Final negative and e2e suites pass.`

### Run 39: Final Acceptance Negative And E2E Suites Gate

- Status: Complete; checked only the next Final Acceptance Gate.
- Worker: Run39.
- Task: Complete only `Final negative and e2e suites pass.`
- Audit: covered canonical validation, producers, local parser,
  orchestration/tools, eventing, SDK, stdout/CLI/FancyTheme, Chat SSE,
  Responses SSE, MCP, A2A, flow, cancellation/disconnect, backpressure,
  bounded memory, reasoning parsing, and acceptance inventory.
- Notes: Added a final negative/e2e inventory that maps each required
  final audit dimension to concrete collectable, unskipped pytest nodes.
  Tightened the hardening inventory buckets for conformance, protocol
  route, parser, and CLI/SDK projection negatives. While running
  coverage, `src/avalan/server/routers/mcp.py` first reported 99%;
  added terminal MCP message coverage so the final coverage run returned
  every listed source file to 100%. Also kept the local lossless-streamer
  timeout regression that prevents duplicate queue puts after a blocked
  handoff timeout.
- Files changed: `src/avalan/model/nlp/text/generation.py`,
  `tests/model/nlp/text_generation_methods_more_test.py`,
  `tests/server/mcp_router_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`,
  `specs/streaming-agents.md`, and ignored checklist
  `specs/streaming-agenda.md`.
- Tests and checks run:
  `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` (11 passed);
  exact final inventory nodes via `poetry run python -c ...` (63 passed,
  45 subtests passed);
  `poetry run pytest --verbose -s
  tests/model/nlp/text_generation_methods_more_test.py::StreamGeneratorTestCase::test_lossless_streamer_handoff_timeout_does_not_duplicate
  tests/server/mcp_router_test.py::MCPResourceStoreTestCase::test_terminal_mcp_messages_yields_closed_resources_first
  tests/server/streaming_acceptance_inventory_test.py` (13 passed);
  `poetry run pytest --cov=avalan.server.routers.mcp
  --cov-report=term-missing --verbose -s tests/server/mcp_router_test.py`
  (139 passed, 6 subtests passed, MCP router 100%); `git diff --check`
  (passed); `make lint` (passed; ruff format, black, ruff check --fix,
  mypy); `poetry run pytest --verbose -s` (6354 passed, 42 skipped,
  3351 subtests passed); `make test-coverage` (all listed source files
  100%).
- Gate checked: `Final negative and e2e suites pass.`
- Gates left unchecked: All later Final Acceptance Gates.
- Next gate: `Synthetic and integration acceptance harnesses demonstrate
  every final gate.`
- Final reviewer approval: Hubble re-reviewed signed HEAD
  `6c7ea7da8ebc0be11efbb0248b251fb8dc7ba950`, found no issues, and
  confirmed the final negative/e2e suites gate is complete. Reviewer checks
  included `git diff --check 55c44a9..HEAD`, final inventory nodes (63
  passed, 45 subtests passed), touched tests (181 passed, 6 subtests
  passed), and targeted inventory/new regression checks (13 passed).
- Coordinator review loop approval: Linnaeus re-reviewed signed HEAD
  `6c7ea7da8ebc0be11efbb0248b251fb8dc7ba950`, found no issues, and
  confirmed the final negative/e2e suites gate is complete. Reviewer
  checks included `poetry run pytest -q
  tests/server/streaming_acceptance_inventory_test.py` (11 passed), exact
  final inventory nodes (63 passed, 45 subtests passed), and streamer
  timeout/stop focused tests (2 passed).

### Run 40: Final Acceptance Synthetic And Integration Harness Gate

- Status: Complete; checked only the next Final Acceptance Gate.
- Worker: Run40.
- Task: Complete only `Synthetic and integration acceptance harnesses
  demonstrate every final gate.`
- Audit: Inspected the existing conformance, protocol/e2e, trace fixture,
  latency, benchmark-regression, and acceptance-inventory harnesses. The
  behavior was already covered by collectable tests, but the proof was
  implicit across several inventories. Added an explicit final-gate
  harness inventory that maps each completed functional final gate plus
  the final negative/e2e gate to both synthetic and integration evidence.
- Harness selection: synthetic nodes cover contract validators,
  accumulators, local/hosted normalizers, reasoning/tool parsers,
  latency/retention budgets, event stats, and CLI/FancyTheme lossless
  isolation. Integration nodes cover public SDK responses, Chat SSE,
  Responses SSE, MCP, A2A, flow, protocol conformance, and cross-protocol
  e2e projection. Event stats/history, channel-boundary validation, and
  reasoning parser correctness are unit-centric contracts; their
  synthetic tests are authoritative, while the surrounding route/protocol
  tests prove the server and protocol adapters retain those guarantees.
- Files changed: `tests/server/streaming_acceptance_inventory_test.py`,
  `specs/streaming-agents.md`, and ignored checklist
  `specs/streaming-agenda.md`.
- Tests and checks run:
  `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` (16 passed);
  exact final-gate harness nodes via `poetry run python -c ...`
  (58 passed, 72 subtests passed);
  `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py
  tests/server/streaming_conformance_test.py
  tests/server/protocol_streaming_e2e_test.py
  tests/server/streaming_latency_budget_test.py
  tests/model/model_stream_benchmark_regression_test.py` (33 passed, 26
  subtests passed); `git diff --check` (passed); `make lint` (passed;
  ruff format, black, ruff check --fix, mypy); post-lint exact
  final-gate harness nodes via `poetry run python -c ...` (58 passed, 72
  subtests passed); `poetry run pytest --verbose -s` (6359 passed, 42
  skipped, 3351 subtests passed); `make test-coverage` (all listed
  source files 100%).
- Gate checked: `Synthetic and integration acceptance harnesses
  demonstrate every final gate.`
- Gates left unchecked: Final benchmark comparison and final `src/`
  coverage gates.
- Next gate: `Final streaming benchmark results are compared against the
  main baseline and all phase rows, with regressions resolved or
  explicitly accepted.`
- Final reviewer approval: Chandrasekhar re-reviewed signed HEAD
  `029ac03c3e59ca615b5a11f3bb003028dc3ccf17`, found no issues, and
  confirmed the synthetic/integration harness gate is complete. Reviewer
  checks included `git diff --check
  bc932ebb1ec52bff7856ab16af6e56fcbaf394e4..HEAD`,
  `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` (16 passed), exact
  final-gate harness evidence nodes (58 passed, 72 subtests passed), and
  clean tracked status.
- Coordinator review loop approval: Singer re-reviewed signed HEAD
  `58b9fbe01abff79442c520b9962d395441777127`, found no issues, and
  confirmed the synthetic/integration harness gate is complete. Reviewer
  checks included `poetry run pytest -q
  tests/server/streaming_acceptance_inventory_test.py` (16 passed), exact
  final-gate evidence nodes (58 passed, 72 subtests passed), and
  `git diff --check
  d8849cf878dea7c1361e1d062acdb9a2451bec8a..HEAD` (passed).

### Run 41: Final Acceptance Benchmark Comparison Gate

- Status: Complete; checked only the final benchmark comparison gate.
- Worker: Run41.
- Task: Complete only `Final streaming benchmark results are compared
  against the main baseline and all phase rows, with regressions resolved
  or explicitly accepted.`
- Benchmark inputs: Audited ignored `specs/streaming/latest.json` from
  `2026-06-16T06:30:51.716877+00:00`, the `phase-7` row in ignored
  `specs/streaming/BENCHMARKS.md`, the `main` baseline from
  `2026-06-13T14:36:57.608061+00:00`, and every recorded comparison row:
  both `phase-1` rows, `phase-2`, `phase-3`, `phase-4`, `phase-5`, and
  both `phase-6` rows. The benchmark artifacts were current and complete,
  so no live benchmark rerun was needed.
- Comparison outcome: OpenAI finished with TTFT 5.429s, total 7.730s, and
  31.95 estimated tokens/s, which regressed against `main` and most/all
  phase rows depending on metric. The artifact explicitly accepts that as
  hosted provider/service variance because the response was shorter and
  used no reasoning tokens. Hermes stayed flat or improved: TTFT 0.353s,
  total 11.122s, 29.67 estimated tokens/s. GPT-OSS improved TTFT to
  0.488s and throughput to 92.32 estimated tokens/s; its 4.138s total time
  is slightly slower than the fastest `phase-2`, `phase-4`, and `phase-5`
  rows, documented as longer output/event-count plus single-run noise while
  throughput improved.
- Files changed: `tests/model/model_stream_benchmark_regression_test.py`,
  `tests/server/streaming_acceptance_inventory_test.py`,
  `specs/streaming-agents.md`, ignored checklist
  `specs/streaming-agenda.md`, and ignored benchmark notes
  `specs/streaming/BENCHMARKS.md`.
- Ignored artifacts: `specs/streaming/BENCHMARKS.md` was updated with an
  explicit final acceptance comparison and accepted-regression rationale.
  `specs/streaming/latest.json` was inspected but not changed. The ignored
  checklist was updated; the final `src/` coverage gate remains unchecked.
- Tests and checks run:
  `poetry run python specs/streaming/benchmark.py --help` (passed);
  `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py` (15 passed, 8
  subtests passed);
  `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` (16 passed);
  `git diff --check` (passed); `make lint` (passed; ruff format, black,
  ruff check --fix, mypy); `poetry run pytest --verbose -s` (6365 passed,
  42 skipped, 3357 subtests passed); `make test-coverage` (all listed
  source files 100%).
- Gate checked: `Final streaming benchmark results are compared against
  the main baseline and all phase rows, with regressions resolved or
  explicitly accepted.`
- Gates left unchecked: Final `src/` coverage gate.
- Next gate: `src/` coverage remains at 100%.

#### Review feedback loop 1

- Reviewer: Erdos.
- Finding addressed: The benchmark guard tests validated only an inline
  fixture and used section-wide acceptance terms, so they could miss an
  incomplete real benchmark artifact, missing duplicate `phase-1` /
  `phase-6` comparison rows, or a missing per-model accepted-regression
  rationale.
- Fix: Strengthened
  `tests/model/model_stream_benchmark_regression_test.py` so the final
  comparison helper validates the actual ignored
  `specs/streaming/BENCHMARKS.md` artifact, requires both recorded
  `phase-1` rows and both recorded `phase-6` rows by timestamp, and
  requires model-specific rationale for the OpenAI hosted regression and
  GPT-OSS total-time variance. Added negative fixture coverage for missing
  duplicate phase references and missing OpenAI/GPT-OSS acceptance text.
  Updated the hardening inventory so those negative nodes are collectable.
- Checks run:
  `poetry run pytest --verbose -s
  tests/model/model_stream_benchmark_regression_test.py` (19 passed, 8
  subtests passed);
  `poetry run pytest --verbose -s
  tests/server/streaming_acceptance_inventory_test.py` (16 passed);
  `git diff --check` (passed); `make lint` (passed; formatted the final
  benchmark regression test assertion layout).
- Follow-up verification: The post-review test strengthening is
  gate-related and correct. It is included in the signed HEAD update, and
  Run41 is ready for re-review.
- Gate state: Benchmark comparison gate remains checked. Final `src/`
  coverage gate remains unchecked.

#### Review feedback loop 2

- Reviewer verdict: Request changes.
- Finding addressed: The ignored `specs/streaming-agenda.md` checklist had
  the final `src/` coverage gate checked even though Run41 was only
  supposed to complete the benchmark comparison gate. This conflicted with
  the Run41 log, which already stated that the final coverage gate remained
  unchecked.
- Fix: Updated the ignored agenda checklist so the benchmark comparison
  gate remains checked and the final `src/` coverage gate is unchecked.
- Checks run: `git diff --check` (passed).
- Gate state: Benchmark comparison gate remains checked. Final `src/`
  coverage gate remains unchecked.
- Final reviewer approval after loop 2: Schrodinger re-reviewed signed HEAD
  `2567a2fca1e899ec9c6585fccea1e55acda7e5f8`, found no Run41 issues,
  and approved the benchmark comparison gate as complete. Reviewer checks
  included `git diff --check`, `git diff --check
  3dc53ddf4455ebb4672ee849cc097830e725f17a..2567a2fca1e899ec9c6585fccea1e55acda7e5f8`,
  `git diff --check
  530a6c1a4cae87420907494e6471ad8c8939a9b9..2567a2fca1e899ec9c6585fccea1e55acda7e5f8`,
  `poetry run pytest --verbose -s -p no:cacheprovider
  tests/model/model_stream_benchmark_regression_test.py` (19 passed, 8
  subtests passed), and `poetry run pytest --verbose -s -p
  no:cacheprovider tests/server/streaming_acceptance_inventory_test.py`
  (16 passed).

### Run 42: Final Acceptance Source Coverage Gate

- Status: Complete; checked the final `src/` coverage gate.
- Worker: Run42.
- Task: Complete only `src/ coverage remains at 100%.`
- Coverage audit: Ran `make test-coverage` before making any source or
  test changes. Every listed `src/avalan/...` file reported 100%, so no
  coverage gaps were present and no tests were added.
- Files changed by Run42: tracked coordination log
  `specs/streaming-agents.md` and ignored checklist
  `specs/streaming-agenda.md`. Run42 did not intentionally change `src/`
  or `tests/` files for coverage.
- Signed range note: The current signed HEAD/range also contains the
  previously approved Run41 benchmark regression guard-test cleanup,
  including assertion formatting in
  `tests/model/model_stream_benchmark_regression_test.py`. The full range
  is therefore not documentation/checklist only.
- Tests and checks run:
  `make test-coverage` (all listed source files 100%);
  `make lint` (passed; ruff format, black, ruff check --fix, mypy);
  `poetry run pytest --verbose -s` (6369 passed, 42 skipped, 3357
  subtests passed);
  post-verification `make test-coverage` (all listed source files 100%).
- Gate checked: `src/` coverage remains at 100%.
- Final gate state: All Final Acceptance Gates are checked.
- Next gate: none; there is no next unchecked gate.
- Final reviewer approval after loop 1: Aristotle re-reviewed signed HEAD
  `fcb721138e46fe55e27c48ce520bb374470487cb`, found no remaining
  issues, and approved the final coverage gate and all Final Acceptance
  Gates as complete. Reviewer checks confirmed the follow-up diff from
  `7d2d9c80cb17478475ba677a9db4ffedbbd10fc1` changed only
  `specs/streaming-agents.md`, base-to-HEAD `git diff --check` passed,
  tracked status was clean, every final checklist item was checked, and
  the recorded pre/post `make test-coverage` runs justify the final
  coverage gate.
- Final reviewer approval: Einstein re-reviewed signed HEAD
  `fcb721138e46fe55e27c48ce520bb374470487cb`, found no issues, and
  confirmed the final coverage gate and all Final Acceptance Gates are
  complete. Reviewer checks included `git diff --name-status
  2567a2fca1e899ec9c6585fccea1e55acda7e5f8..HEAD`, no `src`/`tests`
  changes in that range, `make test-coverage` with every listed
  `src/avalan/...` file at 100%, `git diff --check`, and clean tracked
  status.
