# Flow Fixture Inventory

This inventory assigns durable fixture categories for strict flow behavior.
The fixture roots are intentionally data-only: parser, binding, validation,
runtime, privacy, and e2e tests should load these cases without allowing the
fixtures to become execution authority.

The machine-readable inventory lives at
`tests/fixtures/flow/manifest.json`.

## Fixture Roots

| Category | Root | Case types | Intended tests |
| --- | --- | --- | --- |
| Mermaid positive | `tests/fixtures/flow/mermaid/positive` | positive, round-trip | `tests/flow/mermaid_parser_test.py`, `tests/flow/mermaid_round_trip_test.py` |
| Mermaid negative | `tests/fixtures/flow/mermaid/negative` | negative | `tests/flow/mermaid_parser_test.py` |
| Mermaid executable-import security | `tests/fixtures/flow/mermaid/security/executable_import` | negative, security | `tests/flow/mermaid_security_test.py` |
| Flow View binding | `tests/fixtures/flow/binding` | positive, negative, round-trip | `tests/flow/view_binding_test.py` |
| Definition validation | `tests/fixtures/flow/definition` | positive, negative, privacy | `tests/flow/definition_validation_test.py` |
| Runtime semantics | `tests/fixtures/flow/runtime` | positive, negative, e2e | `tests/flow/runtime_test.py`, `tests/flow/runtime_e2e_test.py` |
| Privacy projection | `tests/fixtures/flow/privacy` | negative, privacy | `tests/flow/privacy_test.py`, `tests/flow/trace_export_test.py` |
| CLI and SDK parity | `tests/fixtures/flow/e2e` | positive, negative, e2e | `tests/cli/flow_command_test.py`, `tests/flow/sdk_e2e_test.py` |

## Required Mermaid Security Buckets

Executable import security fixtures are permanent regression cases. They must
cover unsupported diagram types, init directives, frontmatter, click and link
directives, HTML labels, script-like labels, unknown directives, malformed
subgraphs, ambiguous shorthand, and unsafe external links.

These fixtures must remain in the Mermaid security root even if the parser
later accepts a presentation-only form of the same syntax. Executable import
must keep failing closed unless structured semantics explicitly authorize the
behavior through a separate validated definition.

## Acceptance Map

| ID | Criterion | Owning category | Intended tests |
| --- | --- | --- | --- |
| AC01 | Mermaid import rejects unsupported executable-mode syntax. | Mermaid executable-import security | `tests/flow/mermaid_security_test.py` |
| AC02 | Mermaid parsing preserves source spans and emits structured diagnostics. | Mermaid negative | `tests/flow/mermaid_parser_test.py` |
| AC03 | Mermaid import supports explicit presentation and executable import modes. | Mermaid executable-import security | `tests/flow/mermaid_security_test.py` |
| AC04 | Mermaid normalization expands chains, multi-source shorthand, multi-target shorthand, and nested subgraphs deterministically. | Mermaid positive | `tests/flow/mermaid_parser_test.py` |
| AC05 | Mermaid-derived Flow View remains execution-inert. | Flow View binding | `tests/flow/view_binding_test.py` |
| AC06 | Mermaid binding never infers runtime semantics from labels, shapes, classes, styles, or visual edge decorations. | Flow View binding | `tests/flow/view_binding_test.py` |
| AC07 | Mermaid render preserves presentation metadata. | Mermaid positive | `tests/flow/mermaid_round_trip_test.py` |
| AC08 | Flow definitions validate node types, edges, selectors, mappings, conditions, joins, retries, loops, and outputs before execution. | Definition validation | `tests/flow/definition_validation_test.py` |
| AC09 | Conditional routes can express practical business decisions without Python callbacks. | Definition validation | `tests/flow/definition_validation_test.py` |
| AC10 | Fan-out and fan-in are deterministic and observable. | Runtime semantics | `tests/flow/runtime_test.py` |
| AC11 | Tool, agent, file conversion, human review, and subflow nodes can coexist in one flow. | Runtime semantics | `tests/flow/runtime_e2e_test.py` |
| AC12 | Errors, retries, and fallback paths preserve safe diagnostics. | Privacy projection | `tests/flow/privacy_test.py` |
| AC13 | Paused human review can resume without losing execution state. | Runtime semantics | `tests/flow/runtime_e2e_test.py` |
| AC14 | Unbounded cycles fail validation. | Definition validation | `tests/flow/definition_validation_test.py` |
| AC15 | Parallel execution is bounded and deterministic. | Runtime semantics | `tests/flow/runtime_test.py` |
| AC16 | CLI and SDK users can validate, parse, render, compare, skeletonize, run, inspect, and resume flows. | CLI and SDK parity | `tests/cli/flow_command_test.py`, `tests/flow/sdk_e2e_test.py` |
| AC17 | Public flow execution is async-only and never depends on sync `Flow.execute` behavior. | Runtime semantics | `tests/flow/runtime_test.py` |
| AC18 | Tool nodes resolve and execute only through enabled `ToolManager` descriptors, explicit argument mappings, and privacy-safe outcome projection. | Runtime semantics | `tests/flow/runtime_test.py` |
| AC19 | Mermaid conformance tests cover positive, negative, round-trip, and security fixtures. | Mermaid positive | `tests/flow/mermaid_parser_test.py`, `tests/flow/mermaid_round_trip_test.py`, `tests/flow/mermaid_security_test.py` |
| AC20 | Telemetry identifies skipped, executed, failed, retried, paused, and resumed nodes without leaking sensitive payloads. | Privacy projection | `tests/flow/privacy_test.py`, `tests/flow/trace_export_test.py` |
| AC21 | Diagnostic projection is privacy-safe across SDK, CLI, task storage, server SSE, A2A, MCP, and trace export. | Privacy projection | `tests/flow/privacy_test.py`, `tests/flow/trace_export_test.py`, `tests/server/flow_trace_test.py` |

## Coverage Expectations

Each fixture-backed feature should add positive and negative unit coverage.
Cross-module behavior should add e2e coverage. Privacy-sensitive behavior must
include a negative fixture that proves public diagnostics and traces exclude
private definitions, prompts, raw file bytes, secrets, provider bodies, token
text, unsafe paths, and unsanitized model output.
