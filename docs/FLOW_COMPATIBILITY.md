# Flow Compatibility And Runtime Boundaries

This guide records the supported strict flow surface and the remaining native
compatibility inventory. New flow definitions should use the strict contract:
declared inputs, declared outputs, explicit entry behavior, explicit output
selection, typed nodes, declarative mappings, and declarative routing.

The hosted extraction path can still be expressed as a task (`avalan task run`)
or as a flow-backed task. The direct `avalan flow run` command is useful for
local strict flow runs and for compatible native flow definitions, but durable
inspection and resume behavior belongs to the task-backed flow path.

## Strict Runtime Surface

Strict flow definitions are loaded into `FlowDefinition` and validated before
runtime construction. Validation can run without building node factories, and
diagnostics are projected through public, privacy-safe fields.

The strict surface includes:

- `[[inputs]]` and `[[outputs]]` contracts.
- `[entry]` with a declared entry node.
- `[output_behavior]` with selected output selectors.
- Node kinds for native data movement, validation, decision, join, agent,
  tool, file conversion, human review, notification, and subflows.
- Mapping objects for select, rename, object, array, merge, coalesce, file,
  and file array transfer.
- Edge kinds for success, error, timeout, finally, cancellation, pause, and
  resume.
- Join, retry, timeout, loop, privacy, observability, ownership, and runtime
  metadata sections.

Strict execution is async-only. Invalid definitions fail before any node
factory is invoked.

## CLI Surface

`avalan flow validate FLOW.toml` validates a strict definition without node
execution. `--json` prints compact public diagnostics.

`avalan flow mermaid parse DIAGRAM.mmd --mode presentation|executable` parses
Mermaid into an inert Flow View. `render`, `compare`, and `skeleton` provide
safe rendering, topology comparison against a structured definition, and
non-executable skeleton creation. Mermaid output and JSON diagnostics are
privacy-safe by default.

`avalan flow run FLOW.toml` runs a local flow. It accepts `--input`,
`--input-json`, `--file`, `--file-mime`, `--pdf`, `--json`, `--output`,
`--tool`, and `--tools`. Tool nodes are available only when explicitly enabled
through `--tool` or `--tools`.

`avalan flow inspect`, `trace`, `cancel`, and `resume` operate on task run ids
stored in a durable PostgreSQL task store. They accept `--store-dsn`,
`--store-schema`, and JSON output where applicable. `resume` also requires
`--decision-json` with a review-node-to-decision object.

## Mermaid Boundary

Mermaid is an authoring and presentation frontend. It compiles to `FlowView`,
not to an executable graph. Labels, shapes, classes, styles, subgraphs, and
edge labels never infer node types, tool permissions, prompts, mappings,
conditions, retries, joins, loops, or human review behavior.

`Flow.parse_mermaid(...)` remains a legacy topology importer for
Python-constructed native graphs. Strict Mermaid APIs should use
`parse_mermaid_view`, `render_flow_view`, `compare_flow_topology`, and
`skeleton_from_mermaid_view`.

Executable Mermaid import rejects unsafe or unsupported constructs including
frontmatter, init directives, click/link/href/callback directives, HTML
labels, script-like labels, unsafe links, ambiguous shorthand, malformed
directives, malformed subgraphs, and unsupported diagram types.

## Native Compatibility Inventory

Native flows are Python-constructed directed graphs:

- `Flow.add_node(Node(...))` registers a node by name.
- `Flow.add_connection(src, dest, label=None, conditions=None, filters=None)`
  connects existing nodes.
- `Flow.execute_async(...)` runs flows with cancellation checkpoints,
  `initial_data`, and `initial_inputs`.
- `Node` can run a Python callable, pass through input values, or invoke a
  subgraph. Node input and output schemas are Python `type` checks.
- `FlowManager` wraps async execution with timeout support and sanitized
  before/after events.

Compatibility execution can infer start nodes from graph topology and return
terminal node values. Cycles and unknown node references remain runtime errors
in this compatibility surface.

Callable `Connection.conditions` and `Connection.filters` are accepted only on
Python-constructed `Flow` objects. Structured TOML definitions must use
declarative conditions, edge routing, mappings, retries, joins, and loops, and
must not import callables, functions, modules, or Python snippets.

## Task-Backed Compatibility

Flow-backed task execution reuses task input/output validation, file
materialization, artifact retention, event sanitization, usage accounting,
agent execution, file converters, and tool policy. Task-backed flow state is
JSON and artifact-reference based, not Python object based.

Durable flow runs can persist node attempts, edge states, selected outputs,
loop counters, pause tokens, safe diagnostics, and artifact references. Human
review requires durable pause support and cannot run as an unsupported direct
no-store node.

## Definition Field Matrix

| TOML field | Strict status | Compatibility status | Notes |
| --- | --- | --- | --- |
| `flow.name` | Required | Required | Stable public name for diagnostics and task metadata. |
| `flow.version` | Accepted | Accepted | Version identity. Use either version or revision when needed. |
| `flow.revision` | Accepted | Ignored by legacy graph execution | Marks strict definitions and records revision identity. |
| `flow.description` | Accepted | Accepted | Human-readable metadata. It is not provider prompt text. |
| `flow.tags` | Accepted | Ignored by legacy graph execution | Public metadata only. |
| `flow.entrypoint` | Compatibility only | Accepted | Strict definitions use `[entry]`; legacy TOML uses this single start node. |
| `flow.output_node` | Compatibility only | Accepted | Strict definitions use `[output_behavior]`; legacy TOML uses this single terminal node. |
| `flow.input` / `[input]` | Compatibility only | Accepted | Strict definitions should use `[[inputs]]`. |
| `flow.input.type` | Compatibility only | Accepted | Use `[[inputs]].type` in strict definitions. |
| `flow.input.delivery` | Rejected | Rejected | Delivery is selected by task and provider file-delivery policy. |
| `flow.input.memory` | Rejected | Rejected | File inputs are not memory ingestion toggles. |
| `flow.output` / `[output]` | Compatibility only | Accepted | Strict definitions should use `[[outputs]]`. |
| `flow.output.schema_ref` | Compatibility only | Accepted | Use `[[outputs]].schema_ref` in strict definitions. |
| `[[inputs]]` | Accepted | Not used by legacy graph execution | Declares one or more strict input contracts. |
| `[[outputs]]` | Accepted | Not used by legacy graph execution | Declares one or more strict output contracts. |
| `[entry]` | Accepted | Not used by legacy graph execution | Declares the strict entry behavior. |
| `[output_behavior]` | Accepted | Not used by legacy graph execution | Selects final outputs explicitly. |
| `[runtime_limits]` | Accepted | Not used by legacy graph execution | Public runtime limits. |
| `[privacy]` | Accepted | Not used by legacy graph execution | Public privacy policy metadata. |
| `[observability]` | Accepted | Not used by legacy graph execution | Public observability metadata. |
| `[ownership]` | Accepted | Not used by legacy graph execution | Public ownership metadata. |
| `[variables]` | Accepted | Not used by legacy graph execution | JSON-compatible authoring metadata. |
| `nodes.<name>.type` | Accepted | Accepted | Unknown node types fail validation. |
| `nodes.<name>.ref` | Accepted for trusted ref-backed nodes | Accepted for agent nodes | Path escapes, remote refs, URI-like tool refs, and callable imports are rejected. |
| `nodes.<name>.input` | Compatibility shorthand | Accepted | Strict definitions should prefer explicit mappings. |
| `nodes.<name>.output` | Compatibility shorthand | Accepted | Strict output selection comes from `[output_behavior]`. |
| `nodes.<name>.mapping` | Accepted | Not used by legacy graph execution | Declares explicit node input mappings. |
| `nodes.<name>.join_policy` | Accepted | Not used by legacy graph execution | Required for multi-inbound strict nodes unless optional by contract. |
| `nodes.<name>.retry_policy` | Accepted | Not used by legacy graph execution | Bounded attempts only. |
| `nodes.<name>.timeout_policy` | Accepted | Not used by legacy graph execution | Per-attempt timeout. |
| `nodes.<name>.loop_policy` | Accepted | Not used by legacy graph execution | Requires bounded iteration or elapsed time and an explicit limit route. |
| `nodes.<name>.user_prompt_ref` | Rejected | Rejected | Prompts stay in the agent definition. |
| `nodes.<name>.response_format_ref` | Rejected | Rejected | Response format stays in the agent and task output contract. |
| `[[edges]]` | Accepted | Accepted | Strict edges also declare kind, routing policy, priority, default, and conditions. |
| `edges.condition` | Accepted | Accepted for declarative conditions | Python callables are not accepted from TOML. |
| `cli.runner` | Rejected | Rejected | CLI dispatch is not configured from flow TOML. |
| `cli.example_pdf` | Rejected | Rejected | Examples belong in docs or operator commands, not runtime flow TOML. |

## Unsupported Behaviors

Structured flow definitions reject these behaviors unless a dedicated,
validated contract exists:

- Dynamic Python callable imports.
- Arbitrary module or function references.
- Prompt text, prompt references, or prompt template overrides in flow TOML.
- Flow-specific raw path channels or file hashing.
- Provider configuration in flow TOML.
- Memory ingestion toggles for file inputs.
- Remote schema refs, remote flow refs, path escapes, and symlink escapes.
- Unbounded retries, unbounded loops, and raw graph cycles.
- Mermaid labels or node ids as executable node references.
- Tool refs that are disabled, ambiguous, provider-originated, path-like, or
  URI-like.

## Verification Map

| Behavior | Verification |
| --- | --- |
| Strict validation | Tests assert invalid definitions fail without building node factories. |
| Mermaid boundary | Tests assert compiler-style Flow View APIs do not call `Flow.parse_mermaid(...)`. |
| Safe executable Mermaid import | Security fixtures reject executable directives, HTML, script-like labels, unsafe links, ambiguous shorthand, and unsupported diagrams. |
| Task-backed flow state | Tests assert durable flow records are JSON and artifact-reference based. |
| Human review | Tests assert pause/resume paths, decision validation, timeout routing, cancellation, and audit sanitization. |
| Tool nodes | Tests assert explicit enablement, safe ref resolution, argument validation, cancellation, diagnostics, and privacy-safe envelopes. |
