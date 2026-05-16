# DS4 Native Backend Implementation Plan

This plan tracks implementation of the native DS4 backend described in
`specs/avalan_ds4_native_backend_spec.md`. The backend name is `ds4`.
It is intentionally scoped to DS4-supported DeepSeek V4 Flash GGUF files;
it is not a generic GGUF backend.

## Progress Legend

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[!]` Blocked

Every slice has its own positive and negative tests. A phase is complete
only when all slices in that phase meet their exit conditions and the
phase-level verification commands pass.

## Global Exit Conditions

- [x] `Backend.DS4` is available and accepted from CLI args and URI params.
- [x] DS4 native dependency checks are lazy, safe, and produce actionable
  install/platform errors.
- [x] A DS4-supported GGUF can be loaded from a local path without loading a
  Hugging Face tokenizer.
- [x] Streaming and non-streaming text generation return
  `TextGenerationResponse`.
- [x] Generation settings map to DS4 for length, sampling, stop strings, and
  reasoning mode.
- [x] Native calls never block Avalan's async event loop.
- [x] Cancellation leaves the DS4 session in a known safe state.
- [x] Unit tests pass without DS4 installed by using a fake `pyds4` module.
- [x] Integration tests are skipped by default and pass when
  `AVALAN_DS4_MODEL` is configured.
- [x] Documentation and CLI examples warn that DS4 is model-specific and that
  CPU mode is debug/reference only.

## Phase 0: Source Verification And Decisions

Goal: confirm DS4's current public C API and make the few implementation
decisions that should not be guessed from the spec.

### Slice 0.1: DS4 API Verification

- [x] Inspect current DS4 sources: `ds4.h`, `ds4.c`, `ds4_server.c`,
  `AGENT.md`, `tests/`, and `tests/test-vectors/README.md`.
- [x] Record exact DS4 commit, public symbols, build targets, and any API
  version metadata.
- [x] Resolve whether `ds4_session_sample()` and `ds4_session_argmax()`
  advance session state or require a following `ds4_session_eval()`.
- [x] Resolve session and engine thread-affinity constraints.
- [x] Resolve whether reasoning boundaries, top logprobs, snapshots, payload
  helpers, MTP, and directional steering are exposed through stable public C
  APIs.

Positive tests:

- [x] Add an implementation note or fixture asserting the known DS4 API commit
  and expected public symbol names.
- [x] If a local DS4 checkout is used in CI/dev, add a skipped-by-default
  smoke test that verifies required symbols are present.

Negative tests:

- [x] Simulate a missing required symbol in the binding test double and assert
  that startup fails with a DS4 API/version error.
- [x] Simulate an unsupported backend/platform report and assert that Avalan
  returns a clear unavailable-backend error.

Exit conditions:

- [x] `sample`/`argmax` session semantics are documented in code comments or
  tests before generation loop work starts.
- [x] The initial binding scope is frozen to the symbols needed by phase 1.
- [x] Any upstream DS4 API gaps are tracked as explicit blockers or deferred
  phase-2 items.

### Slice 0.2: Binding Packaging Decision

- [x] Decide whether phase 1 consumes an external `pyds4`/`dwarfstar4`
  package or temporarily keeps an isolated in-tree binding under
  `src/avalan/backends/ds4_native/`.
- [x] Define the import name Avalan will use for the backend.
- [x] Define fallback metadata when DS4 has no native API version symbol:
  binding version, pinned DS4 commit, and native backend name.

Decision: Avalan consumes an external binding at the stable import path
`pyds4`. The in-tree `avalan.backends.ds4_native` package remains a narrow
compatibility/availability layer, not a native binding implementation.

Positive tests:

- [x] Import availability test succeeds without opening a model.
- [x] Metadata test returns binding version and DS4 commit/API data.

Negative tests:

- [x] Import failure, unsafe import, and platform-unavailable cases all return
  the same actionable Avalan error class/message family.
- [x] Availability checks must not call engine-open or touch a model path.

Exit conditions:

- [x] Avalan code can depend on one stable Python import path.
- [x] Packaging choice does not leak native pointers or build internals into
  Avalan's public model API.

## Phase 1: Native Binding Minimum Viable Surface

Goal: expose a narrow, safe Python wrapper over DS4's C API that Avalan can
use without knowing C details.

### Slice 1.1: Engine API

- [x] Add `Backend`, `ThinkMode`, `EngineOptions`, and native exception types.
- [x] Implement `Engine.__init__`, `close`, context-manager methods, and
  metadata properties.
- [x] Implement model-path validation before native open.
- [x] Map native open errors into `Ds4LoadError`, `Ds4BackendUnavailable`, or
  `Ds4InvalidModel`.
- [x] Release the GIL around engine open if using CFFI out-of-line or a C
  extension.

Note: Avalan's in-tree DS4 engine layer is an adapter over the external
`pyds4` binding. Engine open is delegated to `pyds4.Engine`; the compiled
binding remains responsible for releasing the GIL around the native open call.

Positive tests:

- [x] Engine opens with valid options through a fake native layer.
- [x] Engine close is idempotent.
- [x] Context manager closes the engine on exit.
- [x] Metadata properties return routed quant bits, MTP state, and EOS token
  when provided by the fake native layer.

Negative tests:

- [x] Missing model path raises a validation error before native open.
- [x] Directory path raises a validation error.
- [x] Native load failure raises `Ds4LoadError` with the native message.
- [x] Unsupported backend raises `Ds4BackendUnavailable`.
- [x] Closing an engine with live sessions either closes them deterministically
  or raises the documented lifetime error.

Exit conditions:

- [x] Engine lifetime behavior is deterministic and covered by tests.
- [x] No model bytes are copied into Python memory by the wrapper.

### Slice 1.2: Tokenization And Chat Rendering

- [x] Implement `token_text`, `tokenize_text`,
  `tokenize_rendered_chat`, `chat_begin`, `chat_append_message`,
  `chat_append_assistant_prefix`, and `encode_chat_prompt`.
- [x] Implement DS4 thinking-mode mapping in the binding when DS4 exposes a
  context-aware helper.
- [x] Decode token bytes in Avalan with `errors="replace"` only at the backend
  layer, not in the binding.

Positive tests:

- [x] Single-turn system/user prompt renders to the expected token list.
- [x] Multi-turn system/user/assistant prompt renders through chat append
  calls in order.
- [x] Assistant prefix reflects `none`, `high`, and `max` thinking modes.
- [x] `token_text` returns bytes for valid token IDs.

Negative tests:

- [x] Unsupported role raises a clear value error.
- [x] Invalid token ID maps native error to `Ds4GenerationError` or the chosen
  token error.
- [x] Thinking mode unavailable in the current native API is downgraded or
  rejected according to the documented policy.

Exit conditions:

- [x] Avalan can render prompts without Hugging Face tokenizers or chat
  templates.
- [x] Tool-role rendering is explicitly unsupported until the tool-call phase.

### Slice 1.3: Session API

- [x] Implement `Session` creation, close, context-manager methods, `pos`,
  `ctx`, and `tokens`.
- [x] Implement `sync`, `eval`, `argmax`, `argmax_excluding`, `sample`,
  `rewind`, and `invalidate`.
- [x] Implement `SamplingOptions` and validate its ranges before native calls.
- [x] Serialize native session calls or document single-thread-only behavior.
- [x] Release the GIL around `sync`, `eval`, and long native calls.

Note: Avalan's in-tree DS4 session layer serializes calls with an engine-level
lock and delegates GIL release for long native operations to the compiled
`pyds4` binding, matching the engine-open policy from Slice 1.1.

Positive tests:

- [x] Session sync accepts rendered prompt tokens and updates `pos`.
- [x] Greedy generation path calls `argmax`.
- [x] Sampling path calls `sample` with mapped options.
- [x] `eval` advances the session when required by the verified DS4 contract.
- [x] `invalidate` marks the session unusable or reset according to the chosen
  binding behavior.

Negative tests:

- [x] Non-positive context size raises `Ds4ContextError`.
- [x] Sampling options reject negative `top_k`, out-of-range `top_p`/`min_p`,
  and negative temperatures.
- [x] Native eval/sample errors propagate as `Ds4GenerationError`.
- [x] Calls after close fail with a clear closed-session error.

Exit conditions:

- [x] Session state transitions are fully specified and test-covered.
- [x] The binding does not promise concurrent sessions unless DS4 documents
  that it is safe.

Verification:

- [x] `poetry run pytest tests/model/ds4_engine_api_test.py tests/model/ds4_binding_packaging_test.py tests/model/ds4_api_verification_test.py --verbose -s`
- [x] `make lint`

## Phase 2: Avalan Backend Skeleton

Goal: wire `ds4` into Avalan as a selectable local text-generation backend
without full generation behavior yet.

### Slice 2.1: Backend Enum, Settings, And URI Config

- [x] Add `DS4 = "ds4"` to `Backend` in `src/avalan/entities.py`.
- [x] Add `backend_config: dict[str, object] | None = None` to
  `TransformerEngineSettings`.
- [x] Parse `ds4_` URI params in `ModelManager.load()` into normalized
  `backend_config` keys.
- [x] Add common CLI options:
  `--ds4-ctx`, `--ds4-native-backend`, `--ds4-mtp`,
  `--ds4-mtp-draft`, `--ds4-mtp-margin`, `--ds4-warm-weights`, and
  `--ds4-quality`.
- [x] Keep advanced options available through URI params if they would clutter
  generic CLI usage.

Policy: when the resolved backend is `ds4`, unknown `ds4_` URI keys are
rejected with a validation error. Explicit CLI `--ds4-*` values override URI
defaults for the same normalized key; boolean CLI flags override only when the
flag is present. DS4 options are ignored for non-DS4 backends and are never
stored as top-level transformer settings.

Positive tests:

- [x] `Backend("ds4") == Backend.DS4`.
- [x] `ai://local/./model.gguf?backend=ds4&ds4_ctx=4096` produces backend
  `DS4` and `backend_config["ctx_size"] == 4096`.
- [x] CLI parser accepts `--backend ds4` and common DS4 options.
- [x] Explicit CLI DS4 options override URI defaults only where documented.

Negative tests:

- [x] Unknown DS4 URI key is rejected or ignored according to documented
  policy, with a test locking that behavior.
- [x] Invalid `ds4_ctx`, `ds4_mtp_draft_tokens`, or disk-cache sizes fail with
  a clear validation error.
- [x] DS4 options used with a non-DS4 backend do not silently affect
  transformer, MLX, or vLLM behavior.

Exit conditions:

- [x] Backend-specific options are stored only under `backend_config`.
- [x] Existing backend enum tests and CLI choices continue to pass.

### Slice 2.2: Engine Acceptance Hook

- [x] Add `Engine._accepts_loaded_model(self, model: object) -> bool` with a
  default `False` implementation.
- [x] Include the hook in `_load()` model-type validation.
- [x] Do not hard-code `Backend.DS4` in base `Engine._load()`.

Positive tests:

- [x] A test subclass accepts a fake native model through the hook.
- [x] Existing transformer/vendor/diffusion/MLX/sentence-transformer
  acceptance remains unchanged.

Negative tests:

- [x] Unknown native object is still rejected when the hook returns `False`.
- [x] Hook is not consulted for tokenizer validation.

Exit conditions:

- [x] Base engine supports future native backends without DS4-specific logic.
- [x] Model validation errors remain actionable.

### Slice 2.3: Modality Dispatch

- [x] Add `_get_ds4_model()` lazy resolver in
  `src/avalan/model/modalities/text.py`.
- [x] Dispatch local `Backend.DS4` to `Ds4Model`.
- [x] Treat DS4 like MLX in `TextGenerationModality.__call__`: do not pass
  Hugging Face-only `stopping_criterias`, `manual_sampling`, or token-picking
  kwargs into `Ds4Model.__call__`.

Positive tests:

- [x] DS4 loader is returned when available.
- [x] Missing DS4 loader raises `ModuleNotFoundError` mentioning
  `avalan[ds4]`.
- [x] Local DS4 model call receives input, system prompt, developer prompt,
  generation settings, and tool manager only.

Negative tests:

- [x] Resolver returns `None` when the binding is missing or unsafe to import.
- [x] A local non-DS4 model still receives stopping criteria and manual
  sampling kwargs.
- [x] vLLM and MLX dispatch behavior is unchanged.

Exit conditions:

- [x] DS4 can be selected without importing the binding at module import time.
- [x] DS4 call path avoids tokenizer-dependent stopping criteria.

### Slice 2.4: `Ds4Model` Construction And Availability

- [x] Add `src/avalan/model/nlp/text/ds4.py`.
- [x] Implement safe `is_available()` without opening a model.
- [x] Force `auto_load_tokenizer=False` and `enable_eval=False`.
- [x] Implement `_load_model()` to create binding `EngineOptions` and open
  `pyds4.Engine`.
- [x] Override `uses_tokenizer` as `False`.
- [x] Override `_accepts_loaded_model()` for the configured binding module.
- [x] Validate model and MTP paths before opening the native engine.
- [x] Warn when CPU backend is explicitly selected.

Note: `Ds4Model` opens DS4 through Avalan's in-tree
`avalan.backends.ds4_native.Engine` compatibility layer, which validates
availability and constructs the external `pyds4.Engine`.

Positive tests:

- [x] `Ds4Model.is_available()` returns true with a fake safe `pyds4`.
- [x] Constructing `Ds4Model` does not load a tokenizer.
- [x] `_load_model()` passes normalized `EngineOptions` to fake `pyds4`.
- [x] `_accepts_loaded_model()` accepts fake `pyds4.Engine`.
- [x] CPU backend emits the expected warning.

Negative tests:

- [x] Missing binding returns false from `is_available()`.
- [x] Unsafe import returns false and later load raises the install/platform
  message.
- [x] Missing model path or MTP path raises before engine open.
- [x] Fake native load failure propagates as an Avalan load error with context.

Exit conditions:

- [x] Avalan can instantiate `Ds4Model` with a fake binding in unit tests.
- [x] No Hugging Face tokenizer/model APIs are touched by DS4 construction.

Verification:

- [x] `poetry run pytest tests/model/ds4_model_test.py tests/model/ds4_engine_api_test.py tests/model/ds4_binding_packaging_test.py tests/model/ds4_api_verification_test.py --verbose -s`
- [x] `poetry run pytest tests/model/text_modalities_full_test.py --verbose -s`
- [x] `make lint`

## Phase 3: Prompt Rendering And Generation

Goal: produce correct streaming and non-streaming `TextGenerationResponse`
objects through DS4-native prompt rendering and generation.

### Slice 3.1: Avalan Message Mapping

- [x] Convert Avalan input through existing message helpers where possible.
- [x] Merge `system_prompt` and `developer_prompt` into DS4-compatible system
  content with a documented separator.
- [x] Map user and assistant messages to DS4 chat roles.
- [x] Reject tool messages and active `ToolManager` in phase 1 with a clear
  `NotImplementedError`.
- [x] Count prompt tokens from DS4-rendered token IDs.

Positive tests:

- [x] Plain string input renders as a single user message.
- [x] System and developer prompts are both included in the rendered prompt.
- [x] Multi-turn input preserves user/assistant order.
- [x] `TextGenerationResponse.input_token_count` counts a `list[int]`.

Negative tests:

- [x] Tool manager raises the phase-2 unsupported-tool error.
- [x] Tool-role history raises the phase-2 unsupported-tool error.
- [x] Empty prompt or unsupported input shape fails with the same error style
  used by existing text-generation models.

Exit conditions:

- [x] All DS4 prompts are rendered through binding chat/token helpers.
- [x] No DS4 code path calls a Hugging Face chat template.

Verification:

- [x] `poetry run pytest tests/model/ds4_model_test.py --verbose -s`

### Slice 3.2: Generation Settings Mapping

- [x] Map `max_new_tokens`; otherwise use `max_length - prompt_len`; otherwise
  use Avalan's safe default.
- [x] Map `temperature`, `top_k`, `top_p`, `min_p`, and seed.
- [x] Use greedy `argmax()` when `temperature == 0` or sampling is disabled.
- [x] Use `sample()` for nonzero-temperature sampling.
- [x] Map Avalan reasoning settings to DS4 `ThinkMode.NONE`, `HIGH`, or `MAX`.
- [x] Apply stop strings with a bounded recent-text buffer.

Note: Slice 3.2 uses a direct per-call DS4 session loop so settings behavior
can be locked with fake-native tests. Moving native calls off Avalan's event
loop and sharing the generation core between streaming/non-streaming remains
tracked in Slice 3.3.

Positive tests:

- [x] `temperature=0` calls `argmax`.
- [x] `do_sample=True` with nonzero temperature calls `sample`.
- [x] `None` sampling values map to DS4 defaults.
- [x] `max_new_tokens` limits generated tokens.
- [x] `max_length` fallback accounts for prompt length.
- [x] Stop string terminates generation.
- [x] Reasoning disabled maps to `ThinkMode.NONE`; max effort maps to
  `ThinkMode.MAX`.

Negative tests:

- [x] Invalid negative length or sampling values are rejected before native
  calls.
- [x] Unsupported beam-search or multi-return settings fail clearly instead of
  being silently ignored.
- [x] Stop strings that span token boundaries are detected.

Exit conditions:

- [x] DS4 generation behavior is deterministic under fake session scripts.
- [x] Unsupported generation features are explicit errors or documented no-ops.

Verification:

- [x] `poetry run pytest tests/model/ds4_model_test.py --verbose -s`

### Slice 3.3: Worker Thread And Streaming

- [x] Implement one `Ds4Worker` per `Ds4Model`.
- [x] Own the binding `Engine` and one `Session` inside the worker lifecycle.
- [x] Execute all native calls serially on one worker thread.
- [x] Bridge worker output to an async generator via `asyncio.Queue`.
- [x] Propagate exceptions from worker thread to async consumers.
- [x] Close worker resources from model context exit.

Note: `Ds4Worker` now starts a dedicated daemon thread per `Ds4Model`,
opens the DS4 engine and one reusable session inside that thread, and
serializes prompt rendering and generation jobs through a worker queue.
`Ds4Model.__call__` offloads the prompt-render worker round-trip with
`asyncio.to_thread` so streaming publication cannot deadlock the event loop.
Streaming chunks are bridged through a bounded `asyncio.Queue`; cancellation
sets a worker-visible flag, invalidates the active session, and recreates a
fresh session before later jobs run. Non-streaming generation uses the same
worker generation core and returns the complete string through
`TextGenerationResponse`.

Positive tests:

- [x] Streaming response yields fake token text in order.
- [x] Non-streaming response returns the complete string.
- [x] Worker serializes concurrent generation attempts according to the chosen
  policy.
- [x] Exceptions raised by fake native calls surface when the response is
  consumed.
- [x] Model context exit closes session, engine, and worker.

Negative tests:

- [x] Async consumer cancellation sets the cancellation flag.
- [x] Cancellation invalidates or rewinds the session according to the
  documented policy.
- [x] Queue overflow/backpressure does not deadlock the event loop.
- [x] Worker startup failure does not leak a thread.

Exit conditions:

- [x] Native calls are never executed directly on Avalan's event loop.
- [x] Streaming and non-streaming paths share the same generation core.

Verification:

- [x] `poetry run pytest tests/model/ds4_model_test.py --verbose -s`

### Slice 3.4: EOS, Stop Strings, And Token Details

- [x] Stop on DS4 EOS token without emitting it.
- [x] Decode token bytes with UTF-8 replacement at the backend boundary.
- [x] Return plain strings in phase 1.
- [x] Add `TokenDetail` output only when top-logprobs are implemented.
- [x] Ensure generated token is evaluated after candidate selection according
  to verified DS4 session semantics.

Note: DS4 API verification records that `sample`/`argmax` return a candidate
token without advancing the session. Avalan decodes the candidate, evaluates
it with `session.eval()`, then publishes accepted text; fake-session tests lock
the native call order.

Positive tests:

- [x] EOS stops generation and is not emitted.
- [x] Invalid UTF-8 token bytes are replaced, not fatal.
- [x] Stop string stops after expected text.
- [x] Evaluation call order matches verified DS4 session semantics.

Negative tests:

- [x] Empty token text does not corrupt output or counters.
- [x] Native `token_text` failure propagates and invalidates the session if
  needed.
- [x] Stop text handling does not drop previously emitted unrelated content.

Exit conditions:

- [x] Token loop behavior is locked by fake-session tests.
- [x] Real integration smoke tests can reuse the same response API.
- [x] The CLI smoke path for plain DS4 generation works through Avalan's
  `model run` command with the DS4 backend selected, for example:
  `echo "What's your name?" | avalan model run "ai://local/./ds4flash.gguf?backend=ds4&ds4_ctx=4096" --max-new-tokens 64`.
- [x] The plain generation smoke path streams at least one text chunk, returns
  a non-empty non-streaming response, and does not load a Hugging Face
  tokenizer or chat template.

Note: Phase 3 real-model smoke coverage is env-gated in
`tests/model/ds4_phase3_integration_test.py`. It skips unless
`AVALAN_DS4_MODEL` is configured, and exercises the DS4 streaming response
API, non-streaming response API, and `model run` CLI path while guarding
against Hugging Face tokenizer loading.

Verification:

- [x] `poetry run pytest tests/model/ds4_model_test.py --verbose -s`
- [x] `poetry run pytest tests/model/ds4_model_test.py tests/model/ds4_engine_api_test.py tests/model/ds4_binding_packaging_test.py tests/model/ds4_api_verification_test.py --verbose -s`
- [x] `poetry run pytest tests/model/ds4_phase3_integration_test.py --verbose -s`
- [x] `make lint`

## Phase 4: Integration Tests, Documentation, And Packaging

Goal: make the feature usable and maintainable without requiring DS4 hardware
in default CI.

### Slice 4.1: Unit Test Suite

- [x] Add `tests/model/ds4_model_test.py`.
- [x] Add DS4 modality coverage in
  `tests/model/text_modalities_full_test.py`.
- [x] Extend existing CLI tests in `tests/cli/main_test.py` and
  `tests/cli/get_model_settings_test.py`.
- [x] Add fake `pyds4` fixtures covering engine, session, tokenization,
  sampling, cancellation, and failures.

Note: DS4 unit coverage is consolidated in the existing model, modality, and
CLI test files to reuse the fake binding fixtures and avoid duplicate
parser/modality tests.

Positive tests:

- [x] All phase 2 and phase 3 positive cases run without DS4 installed.
- [x] Coverage includes streaming, non-streaming, settings mapping, prompt
  rendering, and lifecycle cleanup.

Negative tests:

- [x] Missing dependency, invalid config, unsupported tools, native failures,
  cancellation, and invalid paths are covered without a real model.
- [x] Tests assert error messages include install or remediation guidance.

Exit conditions:

- [x] DS4 unit tests do not depend on GPU, Metal, CUDA, or a GGUF file.
- [x] `make test-coverage -- -95 src/avalan/model/nlp/text` keeps DS4 files
  above the project threshold.

Verification:

- [x] `poetry run pytest tests/model/ds4_model_test.py --verbose -s`
- [x] `poetry run pytest tests/model/ds4_model_test.py tests/model/ds4_engine_api_test.py tests/model/ds4_binding_packaging_test.py tests/model/ds4_api_verification_test.py --verbose -s`
- [x] `poetry run pytest tests/model/text_modalities_full_test.py tests/cli/get_model_settings_test.py tests/cli/main_test.py --verbose -s`
- [x] `make test-coverage -- -95 src/avalan/model/nlp/text`

### Slice 4.2: Real DS4 Integration Tests

- [x] Add env-gated integration tests skipped unless `AVALAN_DS4_MODEL` is
  set.
- [x] Support optional `AVALAN_DS4_BACKEND=metal|cuda|cpu`.
- [x] Support optional `AVALAN_DS4_CTX=4096`.
- [x] Keep generated token counts small for smoke tests.

Note: Real-model Avalan coverage lives in
`tests/model/ds4_integration_test.py` with shared env-gated helpers in
`tests/model/_ds4_integration.py`. The tests skip silently by default, require
`AVALAN_DS4_MODEL` to point at a DS4-supported GGUF, and accept optional
`AVALAN_DS4_BACKEND` and `AVALAN_DS4_CTX` values. macOS Metal real-model runs
may need to execute outside the filesystem sandbox so the native binding can
access the Metal device.

Positive tests:

- [x] Load a real DS4-supported GGUF.
- [x] Tokenize and count a short prompt.
- [x] Greedy generation produces at least one token.
- [x] Streaming yields at least one chunk.
- [x] Non-streaming returns a string.
- [x] Second call reuses/syncs prefix where DS4 exposes observable progress.

Negative tests:

- [x] Invalid real model path is skipped or fails with validation error as
  appropriate for the fixture.
- [x] CPU backend test warns that CPU is debug/reference only.
- [x] Context too small for the prompt raises `Ds4ContextError`.

Exit conditions:

- [x] Integration tests are silent skips in default CI.
- [x] Integration test instructions document required model path and hardware.

Verification:

- [x] `poetry run pytest tests/model/ds4_model_test.py::test_ds4_context_overflow_surfaces_as_context_error tests/model/ds4_integration_test.py tests/model/ds4_phase3_integration_test.py --verbose -s`
- [x] `AVALAN_DS4_MODEL=/Users/mariano/Code/ai/pyds4/.local/ds4/ds4flash.gguf AVALAN_DS4_BACKEND=metal AVALAN_DS4_CTX=4096 poetry run pytest tests/model/ds4_integration_test.py --verbose -s`

### Slice 4.3: Optional Dependency And Build Docs

- [x] Add `ds4` extra to `pyproject.toml` once package name and version are
  known.
- [x] Document supported production targets: macOS arm64 Metal and Linux CUDA.
- [x] Document CPU as debug/reference only.
- [x] Document how to install/build the binding if it is not published.

Note: Avalan's `ds4` extra depends on `pyds4>=0.1.0,<0.2.0` only for
macOS arm64 and Linux installs. The README and installation docs describe the
supported DS4 production targets, call out CPU as debug/reference only, and
document the local `pyds4` source-build fallback.

Positive tests:

- [x] Packaging metadata includes the `ds4` extra.
- [x] Dependency markers limit install attempts to supported platforms.

Negative tests:

- [x] Unsupported platform installation path has clear docs and runtime error.
- [x] Missing optional dependency does not affect importing Avalan core.

Exit conditions:

- [x] `pip install "avalan[ds4]"` or the documented equivalent is the single
  install path in user docs.
- [x] Core Avalan installation remains DS4-free.

Verification:

- [x] `poetry run pytest tests/project_metadata_test.py tests/model/ds4_binding_packaging_test.py --verbose -s`
- [x] `make lint`

### Slice 4.4: User Documentation And Examples

- [x] Add DS4 backend docs to the relevant README/docs page.
- [x] Add CLI examples for basic generation, disabled reasoning, and URI
  backend config.
- [x] Document unsupported phase-1 behavior: generic GGUF, tool calls, and
  production CPU inference.
- [x] Document integration-test environment variables.

Note: DS4 user documentation now lives in `docs/DS4.md`, with links from
the main README, docs index, examples index, and AI URI reference. The CLI
reference includes `ds4` in backend choices and documents the DS4-specific
`model run` options. A follow-up fix also makes DS4 local file runs bypass
Hugging Face token prompts and auth checks, supports `--backend ds4` outside
the URI, and documents relative, absolute, and percent-encoded local GGUF
paths.

Positive tests:

- [x] Documentation examples match actual CLI option names.
- [x] Help output includes `ds4` as a backend choice.

Negative tests:

- [x] Docs do not claim generic GGUF compatibility.
- [x] Docs do not claim native tool-call support until phase 5 completes.

Exit conditions:

- [x] A user with a supported DS4 GGUF can run the documented command.
- [x] Known limitations are visible before the examples.

Verification:

- [x] `poetry run pytest tests/cli/main_test.py::CliModelRunOptionTestCase --verbose -s`
- [x] `make lint`

## Phase 5: Advanced DS4 Features

Goal: add DS4-native features after the first backend is correct and stable.
These slices may be separate PRs.

### Slice 5.1: Snapshots And Cancellation Recovery

- [x] Bind `save_snapshot` and `load_snapshot`.
- [x] Use snapshots for generation rollback and test isolation.
- [x] Keep snapshots in memory only in this slice.

Note: Avalan exposes in-memory session snapshot save/load through the
compatibility session adapter and uses them opportunistically in
`Ds4Worker`. Snapshot support is capability-detected: bindings without
`save_snapshot`/`load_snapshot` continue to use the existing
invalidate-and-recreate recovery path, while bindings that expose the methods
restore the prompt-aligned snapshot after generation failures or async
consumer cancellation.

Positive tests:

- [x] Snapshot save/load restores fake session position and tokens.
- [x] Cancelled generation restores a prior snapshot when available.

Negative tests:

- [x] Snapshot from a mismatched context/model is rejected.
- [x] Corrupt snapshot bytes raise a binding error without crashing.

Exit conditions:

- [x] Cancellation can recover without full prompt rebuild when snapshots are
  available.
- [x] Snapshot feature is optional and capability-detected.

Verification:

- [x] `poetry run pytest tests/model/ds4_engine_api_test.py tests/model/ds4_model_test.py --verbose -s`
- [x] `make lint`

### Slice 5.2: Disk KV Cache

- [x] Bind DS4 payload helpers or consume them through the binding.
- [x] Implement token-prefix-based cache keys.
- [x] Enforce cache budget by size and LRU/hit count.
- [x] Treat rendered text as observability only, never as key material.

Note: Avalan now has an optional DS4 disk KV cache enabled by
`backend_config["kv_disk_dir"]` plus a positive
`backend_config["kv_disk_space_mb"]`. The cache is capability-detected and
uses binding-level `save_payload`/`load_payload` methods when available;
current bindings without payload helpers keep the live-session path unchanged.
Cache keys are derived from model namespace, context size, token count, and
the rendered prompt token IDs, not rendered text. Metadata records rendered
state only for validation and observability. Cache writes, corrupt payloads,
and restore failures fall back to live `session.sync()` generation and are
logged.

Positive tests:

- [x] Cache hit restores the fake session for an identical token prefix.
- [x] Cache miss occurs for different token prefix or context size.
- [x] Budget eviction removes least-useful entries.

Negative tests:

- [x] Corrupt payload is skipped and logged.
- [x] Context-size mismatch invalidates the payload.
- [x] Disk write failure falls back to live session generation.

Exit conditions:

- [x] Disk cache correctness is tested before performance claims are made.
- [x] Cache can be disabled and leaves no files behind in disabled mode.

Verification:

- [x] `poetry run pytest tests/model/ds4_model_test.py --verbose -s`
- [x] `make lint`

### Slice 5.3: Top Logprobs And `TokenDetail`

- [x] Bind `top_logprobs` and token logprob helpers.
- [x] Emit `TokenDetail` when Avalan requests token display/manual sampling.
- [x] Keep plain-string output as the default.

Note: DS4 token details are capability-detected and used only for manual
token display. Default DS4 streaming still yields plain strings. When manual
sampling is requested, Avalan selects the candidate with DS4-native
`argmax`/`sample`, reads native `top_logprobs`/`token_logprob` before
`eval`, converts log probabilities with `exp`, and emits `TokenDetail` using
Avalan's existing token-detail contract. Bindings that do not expose logprob
helpers fail with an explicit unsupported-feature error.

Positive tests:

- [x] Top-k token details include chosen token and probability distribution.
- [x] Probabilities are computed from logprobs consistently.
- [x] Existing plain streaming output remains unchanged by default.

Negative tests:

- [x] Invalid top-k is rejected.
- [x] Missing native logprob support returns a clear unsupported-feature error.

Exit conditions:

- [x] Token detail output matches Avalan's existing response contracts.
- [x] Manual token display no longer uses Hugging Face-only paths for DS4.

Verification:

- [x] `poetry run pytest tests/model/ds4_model_test.py tests/model/ds4_engine_api_test.py tests/model/text_modalities_full_test.py --verbose -s`

### Slice 5.4: MTP And Directional Steering

- [x] Expose MTP options through `backend_config`.
- [x] Validate optional MTP and steering paths.
- [x] Pass MTP draft token count, margin, quality, warm-weights, and steering
  coefficients to `EngineOptions`.

Note: MTP options are available through DS4 backend config from CLI and URI
settings. Directional steering remains URI/backend-config only. Avalan
preflights MTP and steering file paths, requires a steering file when steering
coefficients are non-zero, rejects negative MTP draft counts and margins, and
capability-detects optional `pyds4.EngineOptions` fields so default runs still
work with a minimal binding while requested advanced options fail with an
actionable backend-unavailable error.

Positive tests:

- [x] MTP options reach fake `EngineOptions`.
- [x] Directional steering options reach fake `EngineOptions`.
- [x] Missing optional features are capability-detected.

Negative tests:

- [x] Missing MTP/steering files raise validation errors.
- [x] Invalid MTP draft count or margin is rejected.
- [x] Unsupported MTP on current DS4 build raises an actionable error.

Exit conditions:

- [x] Advanced options are documented as DS4-specific.
- [x] Defaults preserve phase-1 behavior.

Verification:

- [x] `poetry run pytest tests/model/ds4_engine_api_test.py tests/model/ds4_model_test.py tests/model/model_manager_extra_test.py --verbose -s`

### Slice 5.5: Native Tool Calls And DSML Replay

- [ ] Promote or bind DS4 server DSML rendering/parsing/replay helpers.
- [ ] Map Avalan tool schemas to DS4-native tool schema rendering.
- [ ] Parse generated DSML blocks into Avalan `ToolCallToken`.
- [ ] Replay tool results exactly enough to preserve KV-prefix alignment.

Positive tests:

- [ ] Tool schema rendering matches DS4 server canonical output.
- [ ] Generated tool call round-trips through Avalan `ToolManager`.
- [ ] Tool-call streaming emits argument deltas without polluting final text.
- [ ] Two-turn tool-call conversation reuses the live session prefix.
- [ ] At least one README-style tooling example, such as
  `avalan agent run --tool "math.calculator"`, succeeds with the model tied to
  the DS4 backend.

Negative tests:

- [ ] Malformed DSML raises or streams a structured parse error according to
  Avalan's tool contract.
- [ ] Unknown tool ID falls back to documented canonical rendering or fails
  clearly.
- [ ] Tool results cannot corrupt session prefix replay.

Exit conditions:

- [ ] Native DS4 tool calls are correct enough to remove the phase-1
  `NotImplementedError`.
- [ ] Exact replay behavior is covered by both unit and integration tests.
- [ ] README tooling examples that advertise Avalan agent tools can be
  replicated with a DS4-backed engine URI without falling back to generic chat
  rendering or corrupting DS4 DSML/KV-prefix replay.

## Verification Commands

Run these before merging each implementation PR:

```bash
make lint
poetry run pytest --verbose -s
make test-coverage -- -95 src/avalan/model/nlp/text
```

Run these when native DS4 code or the binding is changed:

```bash
make
make test
```

Run real-model Avalan integration tests only when the environment is
configured:

```bash
AVALAN_DS4_MODEL=/path/to/ds4flash.gguf \
AVALAN_DS4_BACKEND=metal \
AVALAN_DS4_CTX=4096 \
poetry run pytest --verbose -s tests/model/ds4_phase3_integration_test.py

AVALAN_DS4_MODEL=/path/to/ds4flash.gguf \
AVALAN_DS4_BACKEND=metal \
AVALAN_DS4_CTX=4096 \
poetry run pytest --verbose -s tests/model/ds4_integration_test.py
```

## Tracking Summary

| Phase | Status | Exit Signal |
| --- | --- | --- |
| Phase 0: Source verification | [x] | DS4 API semantics and packaging choice are documented. |
| Phase 1: Native binding | [x] | Engine/session/chat APIs pass fake-native tests. |
| Phase 2: Avalan skeleton | [x] | `--backend ds4` loads `Ds4Model` with fake binding. |
| Phase 3: Generation | [x] | Streaming and non-streaming responses pass fake-session tests and env-gated smoke tests exist for real DS4 models. |
| Phase 4: Tests/docs/packaging | [x] | Unit, integration, docs, and extras exist. |
| Phase 5: Advanced features | [~] | Snapshot recovery is complete; remaining advanced DS4 features land in isolated slices. |
