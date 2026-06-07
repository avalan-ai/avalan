# Native Flow Compatibility

This guide records the current native flow surface and the minimal TOML
contract needed for an optional hosted extraction flow.

The required hosted extraction path remains `avalan task run`. Native flow
support is available for compatible definitions, but the direct task command
remains the primary hosted extraction path.

## Current Runtime Inventory

Native flows are Python-constructed directed graphs:

- `Flow.add_node(Node(...))` registers a node by name.
- `Flow.add_connection(src, dest, label=None, conditions=None, filters=None)`
  connects existing nodes.
- `Flow.parse_mermaid(...)` imports a Mermaid edge diagram into plain nodes and
  unlabeled or labeled edges.
- `Flow.execute_async(...)` runs flows with cancellation checkpoints,
  `initial_data`, and `initial_inputs`.
- `Node` can run a Python callable, pass through input values, or invoke a
  subgraph. Node input and output schemas are Python `type` checks.
- `FlowManager` wraps async execution with timeout support and sanitized
  before/after events.

Flow execution infers start nodes from graph topology and returns the single
terminal node value when exactly one terminal node exists. When multiple
terminal nodes exist, execution returns a mapping keyed by terminal node name.
Cycles and unknown node references are runtime errors.

## Target Boundary

The current native flow surface is compatibility inventory. Future strict flow
APIs must not treat these compatibility helpers as semantic authority:

- `Flow.parse_mermaid(...)` is a legacy topology importer only. Mermaid
  compiler, render, compare, binding, and executable-import APIs must construct
  inert view data and diagnostics without calling it.
- Callable `Connection.conditions` and `Connection.filters` are accepted only
  on Python-constructed `Flow` objects. Structured TOML definitions must use
  declarative routing and must not import callables, functions, modules, or
  Python snippets.
- `avalan flow run` currently executes compatible native TOML flows. Future
  strict flow execution may replace that behavior; any compatibility runner
  that remains must be explicitly named and documented as compatibility
  behavior.

## Current CLI Inventory

The CLI exposes `avalan flow run <flow>` for native TOML flows built from
registered built-in nodes. It supports `--input`, `--input-json`,
field-addressed `--input-name` style flags, `--file`, `--file-mime`, `--pdf`,
`--json`, and `--output`.

## Current Task-Target Inventory

`FlowTaskTargetRunner` can execute a resolver-supplied `Flow` instance for
compatible task definitions. It validates safe relative references, requires
exactly one start node at run time, binds JSON-compatible and file-backed task
input into the flow, validates task output, emits sanitized flow events, and
preserves cancellation through `Flow.execute_async`.

Current flow-backed task support is intentionally narrow:

- File and file-array task inputs are passed through task materialization and
  provider delivery descriptors.
- Object and array task outputs require a declared schema.
- The CLI supplies a TOML resolver for direct task runs; SDK callers can still
  supply their own resolver.
- Python callables are accepted only in already-constructed `Flow` objects.
- Agent nodes reuse the task agent runner; prompts, reasoning settings,
  response format, and provider configuration remain in the agent file.

## Minimal Native TOML Subset

A native TOML flow for hosted extraction is limited to one entry node, one
terminal output, one PDF input, and one agent node. The minimal target shape is:

```toml
[flow]
name = "document_extraction"
version = "1"
entrypoint = "extract"
output_node = "extraction_json"

[flow.input]
name = "document"
type = "file"
mime_types = ["application/pdf"]

[flow.output]
name = "extraction_json"
type = "object"
schema_ref = "invoice.schema.json"

[nodes.extract]
type = "agent"
ref = "agent.toml"
input = "document"
output = "extraction_json"
```

The agent spec remains the authoritative source for system prompts, user
prefixes, reasoning settings, response format, provider configuration, and file
delivery behavior. Flow TOML should not introduce prompt mutation fields or a
second response-format contract.

## Compatibility Matrix

| TOML field | Current status | Native subset status | Notes |
| --- | --- | --- | --- |
| `flow.name` | Accepted | Accepted | Metadata only. Must not affect task idempotency or provider payloads. |
| `flow.version` | Accepted | Accepted | Metadata only until native flow schema versioning is defined. |
| `flow.description` | Accepted | Accepted | Human-readable metadata. Must not be sent to providers. |
| `flow.entrypoint` | Accepted | Accepted | Maps to the single start node. Missing or unknown values fail before execution. |
| `flow.output_node` | Accepted | Accepted | Maps to the single terminal output. Multiple terminal outputs are rejected. |
| `flow.input.name` | Accepted | Accepted | Names the input exposed to the entry node. |
| `flow.input.type` | Accepted | Accepted | File-backed agent nodes use task file materialization. |
| `flow.input.mime_types` | Accepted | Accepted for `application/pdf` | MIME mismatch fails before execution. |
| `flow.input.delivery` | Rejected | Rejected | Delivery is selected by the task and agent provider profile. |
| `flow.input.memory` | Rejected | Rejected | File inputs must not be routed into memory ingestion by native flow TOML. |
| `flow.output.name` | Accepted | Accepted | Names the value returned by `flow.output_node`. |
| `flow.output.type` | Accepted | Accepted | Artifact output mapping remains unsupported. |
| `flow.output.schema_ref` | Accepted | Accepted | Schema refs are recorded for the flow contract. |
| `nodes.<name>.type` | Accepted for built-ins and agent nodes | Accepted | Unknown node types fail with stable diagnostics. |
| `nodes.<name>.ref` | Rejected for built-ins, accepted for agent nodes | Accepted | Built-in nodes cannot load external refs; path escapes and remote refs are rejected. |
| `nodes.<name>.input` | Accepted | Accepted | Must name the declared flow input or a previous supported node output. |
| `nodes.<name>.output` | Accepted | Accepted | Names the node output for compatibility with single-output flow contracts. |
| `nodes.<name>.user_prompt_ref` | Rejected | Rejected | User prompts come from the agent spec. |
| `nodes.<name>.response_format_ref` | Rejected | Rejected | Response format comes from the agent spec and task output schema. |
| `cli.runner` | Rejected | Rejected | Native CLI dispatch must replace local runner shims. |
| `cli.example_pdf` | Rejected | Rejected | Examples belong in docs or operator commands, not runtime flow TOML. |

## Unsupported Behaviors

Native TOML should reject these behaviors until they have explicit validation,
privacy, and trust contracts:

- Dynamic Python callable imports.
- Arbitrary module or function references.
- Prompt text, prompt references, or prompt template overrides in flow TOML.
- Flow-specific file descriptors, raw path channels, or file hashing.
- Flow-specific schema expansion or output validation copies.
- Multiple entry nodes or multiple terminal outputs.
- Remote schema refs, remote flow refs, path escapes, and symlink escapes.
- Provider configuration in flow TOML.
- Memory ingestion toggles for file inputs.
- Subgraphs loaded from TOML.

## Verification Map

| Behavior | Verification |
| --- | --- |
| File-backed agent nodes | Tests assert PDF descriptors become provider file blocks without memory ingestion. |
| Agent node reuse | Tests assert flow-backed tasks call the same agent runner as direct tasks. |
| Output contract mapping | Tests assert flow task output is validated against the declared task schema before success. |
| Direct and flow parity | Tests compare fake-provider prompt blocks, file blocks, response schema, reasoning effort, output objects, and sanitized telemetry. |
