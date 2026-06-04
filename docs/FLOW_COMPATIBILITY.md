# Native Flow Compatibility

This guide records the current native flow surface and the minimal TOML
contract needed for an optional hosted extraction flow.

The required hosted extraction path remains `avalan task run`. Native flow
support is an optional path and does not block task-run acceptance.

## Current Runtime Inventory

Native flows are Python-constructed directed graphs:

- `Flow.add_node(Node(...))` registers a node by name.
- `Flow.add_connection(src, dest, label=None, conditions=None, filters=None)`
  connects existing nodes.
- `Flow.parse_mermaid(...)` imports a Mermaid edge diagram into plain nodes and
  unlabeled or labeled edges.
- `Flow.execute(...)` runs synchronously from explicit or inferred start nodes.
- `Flow.execute_async(...)` runs asynchronously with cancellation checkpoints,
  `initial_data`, and `initial_inputs`.
- `Node` can run a Python callable, pass through input values, or invoke a
  subgraph. Node input and output schemas are Python `type` checks.
- `FlowManager` wraps async execution with timeout support and sanitized
  before/after events.

Current flow execution infers start nodes from graph topology and returns the
single terminal node value when exactly one terminal node exists. When multiple
terminal nodes exist, execution returns a mapping keyed by terminal node name.
Cycles and unknown node references are runtime errors.

## Current CLI Inventory

The CLI parser exposes `avalan flow run <flow>`, but no command dispatch branch
currently executes a flow definition. The parser also has no native flow input
or output flags, such as `--input-json`, `--file`, `--pdf`, `--json`, or
`--output`.

## Current Task-Target Inventory

`FlowTaskTargetRunner` can execute a resolver-supplied `Flow` instance for
compatible task definitions. It already validates safe relative references,
requires exactly one start node at run time, binds JSON-compatible task input
into the flow, validates task output, emits sanitized flow events, and preserves
cancellation through `Flow.execute_async`.

Current flow-backed task support is intentionally narrow:

- File and file-array task inputs are rejected.
- Object and array task outputs require a declared schema.
- A resolver must be supplied by the caller; TOML flow references are not loaded
  by the runner.
- Python callables are accepted only in already-constructed `Flow` objects.

## Minimal Native TOML Subset

A native TOML flow for hosted extraction should be limited to one entry node,
one terminal output, one PDF input, and one agent node until a broader trust
model exists. The minimal supported shape should be:

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
| `flow.name` | Rejected | Accepted | Metadata only. Must not affect task idempotency or provider payloads. |
| `flow.version` | Rejected | Accepted | Metadata only until native flow schema versioning is defined. |
| `flow.description` | Rejected | Accepted | Human-readable metadata. Must not be sent to providers. |
| `flow.entrypoint` | Rejected | Accepted | Maps to the single start node. Missing or unknown values must fail before execution. |
| `flow.output_node` | Rejected | Accepted | Maps to the single terminal output. Multiple terminal outputs remain unsupported for the hosted extraction path. |
| `flow.input.name` | Rejected | Accepted | Names the file input exposed to the entry node. |
| `flow.input.type` | Rejected | Accepted for `file` only | `file[]`, object fan-in, and scalar flow inputs stay unsupported for the hosted extraction path. |
| `flow.input.mime_types` | Rejected | Accepted for `application/pdf` | MIME mismatch must fail before provider execution. |
| `flow.input.delivery` | Rejected | Rejected | Delivery is selected by the task and agent provider profile. |
| `flow.input.memory` | Rejected | Rejected | File inputs must not be routed into memory ingestion by native flow TOML. |
| `flow.output.name` | Rejected | Accepted | Names the value returned by `flow.output_node`. |
| `flow.output.type` | Rejected | Accepted for `object` | Array and artifact outputs need separate mapping rules. |
| `flow.output.schema_ref` | Rejected | Accepted | Must reuse task and agent schema-reference expansion. |
| `nodes.<name>.type` | Rejected | Accepted for `agent` only | Unknown node types must fail with stable diagnostics. |
| `nodes.<name>.ref` | Rejected | Accepted | Relative agent refs resolve from the flow file directory. Path escapes and remote refs are rejected. |
| `nodes.<name>.input` | Rejected | Accepted | Must name the declared flow input or a previous supported node output. |
| `nodes.<name>.output` | Rejected | Accepted | Must match `flow.output_node` output mapping for the single-node hosted extraction path. |
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

## Gap Map

| Gap | Required work |
| --- | --- |
| No TOML flow loader | Add a native flow loader with typed diagnostics for flow, input, output, node, edge, and variable sections. |
| No `flow run` dispatch | Add a CLI command implementation with JSON input, file input, PDF sugar, JSON output, exit-code semantics, and sanitized stderr. |
| No target-runner registry | Add optional dispatch only when a task execution target requires a non-default runner. |
| File inputs rejected by flow tasks | Reuse task file materialization and provider delivery descriptors for compatible flow nodes. |
| No agent node type | Add an agent node that reuses agent task prompt composition, file delivery, schema refs, cancellation, and event handling. |
| No output-node contract | Map node outputs to task output contracts and schema refs before reporting success. |
| No parity fixture | Add fake-provider parity coverage comparing direct task and native flow payloads. |

These gaps should be implemented in order: TOML loading and CLI dispatch first,
file-backed agent nodes second, and parity documentation and tests last.
