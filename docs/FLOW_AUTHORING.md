# Flow Authoring Guide

Flows compose validated definitions, inert topology views, and async runtime
execution. A flow definition is the only execution authority. Mermaid diagrams
are useful for authoring and review, but their labels, shapes, styles, and
edge labels do not create runtime behavior.

## Definition And View

A structured flow definition declares:

- Public flow identity: `name`, optional `version` or `revision`, tags, and
  ownership metadata.
- One or more `[[inputs]]`.
- One or more `[[outputs]]`.
- `[entry]` with the node that starts execution.
- `[output_behavior]` with explicit selectors for final outputs.
- `[nodes.<name>]` tables with supported node types and required config.
- `[[edges]]` tables with explicit edge kinds and routing details.
- Optional runtime, privacy, and observability metadata.

A Flow View records topology and presentation metadata. It can be parsed from
Mermaid, rendered back to safe Mermaid, compared with a definition, or used to
create a non-executable skeleton. It does not create `Flow`, `Node`, or
`Connection` objects.

## Minimal Strict Definition

```toml
[flow]
name = "profile_projection"
version = "1"
tags = ["docs"]

[[inputs]]
name = "payload"
type = "object"

[[outputs]]
name = "profile"
type = "object"

[entry]
type = "node"
node = "start"

[output_behavior]
type = "map"

[output_behavior.outputs]
profile = "pick.value"

[nodes.start]
type = "input"

[nodes.pick]
type = "select"

[nodes.pick.mapping.value]
type = "object"

[nodes.pick.mapping.value.fields]
name = "input.payload.name"
account_id = "input.payload.account.id"

[[edges]]
source = "start"
target = "pick"
kind = "success"
```

Validate before running:

```bash
avalan flow validate profile.flow.toml
avalan flow validate profile.flow.toml --json
```

Run with a JSON input:

```bash
avalan flow run profile.flow.toml \
  --input-json '{"name":"Ada","account":{"id":"acct_123"}}' \
  --json
```

## Node Types

Native data nodes include `input`, `constant`, `pass-through`, `select`,
`validation`, `decision`, `join`, and `notification`. These nodes run through
the strict executor and use declared contracts and mappings.

Task-backed node families include `agent`, `file_convert`, `pdf_to_images`,
`tool`, `human_review`, and `subflow` where the registry and runtime provide
the required support. Ref-backed nodes must use trusted local refs or enabled
tool names. Remote refs, path escapes, callable imports, and provider-originated
tool names are rejected.

Tool nodes require explicit enablement in the CLI:

```bash
avalan flow run tool.flow.toml \
  --tool math.add \
  --input-json '{"a":2,"b":3}' \
  --json
```

Use `--tools namespace` to enable a namespace. Tool arguments must be mapped
explicitly unless the enabled tool has no required user-provided parameters.

## Selectors And Mappings

Selectors address flow inputs, node outputs, and nested object or array values:

| Selector | Meaning |
| --- | --- |
| `input.payload` | The flow input named `payload`. |
| `input.payload.customer.id` | Nested object member. |
| `start.value.items[0]` | First item from a node output array. |
| `check.result.status` | Named node output field. |

Selectors cannot access environment variables, filesystems, network locations,
runtime internals, private task keys, arbitrary object attributes, templates,
or Python code.

Mappings make node inputs explicit:

```toml
[nodes.check.mapping.arguments]
type = "object"

[nodes.check.mapping.arguments.fields]
account_id = "input.payload.account.id"
score = "risk.value.score"
```

Supported mapping kinds are `select`, `rename`, `object`, `array`, `merge`,
`coalesce`, `file`, and `file[]`. A `coalesce` mapping uses `sources` and
returns the first selector that exists at runtime.

## Conditions And Routing

Edges default to `success` kind. Use explicit kinds for error, timeout,
finally, cancellation, pause, and resume routes. Declarative conditions support
equality, inequality, existence, type checks, membership, numeric comparison,
string prefix/suffix/contains checks, null checks, and boolean `all`, `any`,
and `not` composition.

```toml
[[edges]]
source = "risk"
target = "review"
kind = "success"
priority = 10

[edges.condition]
op = "gte"
selector = "risk.value.score"
value = 0.7

[[edges]]
source = "risk"
target = "approve"
kind = "success"
default = true
```

When more than one outgoing route can match, set priorities, exclusive routing,
or a default route so routing remains deterministic.

## Graph-Authored Strict Flows

Use `[graph]` when Mermaid should author the route topology while TOML remains
the source of execution semantics. The loader compiles the graph to ordinary
strict `[[edges]]`, validates the resulting definition, and never runs Mermaid
directly.

An inline graph keeps the diagram in the flow definition:

```toml
[graph]
format = "mermaid"
source = "inline"
mode = "executable"
diagram = '''
flowchart LR
start route_profile@-->|diagram label ignored| pick
start -.-> note["review-only note"]
note -.-> pick
'''

[graph.edges.route_profile]
kind = "success"
label = "profile_ready"
```

A file-backed graph keeps Mermaid beside the flow definition:

```toml
[graph]
format = "mermaid"
source = "file"
mode = "executable"
path = "profile_topology.mmd"
```

Graph paths are resolved relative to the flow file. URL schemes, absolute path
escapes, parent traversal, symlink escapes, and unreadable files are rejected
with privacy-safe diagnostics.

Mermaid node IDs become actual only when they exactly match declared strict
node names such as `[nodes.start]` and `[nodes.pick]`. Decorative nodes and
edges can appear in the diagram for review context, but they do not create
runtime nodes, routes, joins, retries, mappings, prompts, tools, or outputs.

Executable graph edges must use explicit Mermaid edge IDs, for example
`route_profile@-->`. Metadata binds only through `[graph.edges.<edge_id>]`.
Labels, classes, styles, shapes, subgraphs, generated view IDs, link-style
indexes, and source-target pairs do not bind metadata and do not infer runtime
behavior. Mermaid edge labels are inert unless the matching graph edge
metadata explicitly sets a strict `label`.

Compile graph-authored flows to canonical strict TOML for review, CI, storage,
or older consumers:

```bash
avalan flow compile docs/examples/flows/graph_inline.flow.toml \
  --output strict.flow.toml
avalan flow compile docs/examples/flows/graph_file.flow.toml --check
```

Canonical strict TOML contains normal `[[edges]]` records and omits executable
`[graph]` source.

## Joins, Retries, Timeouts, And Loops

Nodes with more than one inbound path need a join policy unless their contract
allows otherwise:

```toml
[nodes.merge]
type = "join"

[nodes.merge.join_policy]
type = "all_success"
```

Retry and loop policies must be bounded:

```toml
[nodes.call_provider.retry_policy]
max_attempts = 3
backoff = "exponential"
initial_delay_seconds = 1
max_delay_seconds = 8
retryable_categories = ["timeout", "provider"]
exhausted_route = "manual_check"

[nodes.repair.loop_policy]
max_iterations = 3
limit_route = "manual_check"
output_selector = "repair.value"
```

Unbounded graph cycles, unbounded retries, and unbounded loops fail validation.

## Mermaid Modes

Use presentation mode when a diagram is only visual:

```bash
avalan flow mermaid parse diagram.mmd --mode presentation --json
avalan flow mermaid render diagram.mmd --mode presentation
```

Use executable mode when the diagram is meant to bind to structured execution
metadata:

```bash
avalan flow mermaid compare diagram.mmd profile.flow.toml \
  --mode executable
```

Executable mode fails closed on unsupported or unsafe Mermaid features,
including frontmatter, init directives, click/link/href/callback directives,
HTML labels, script-like labels, unsafe links, ambiguous shorthand, malformed
directives, malformed subgraphs, and unsupported diagram types.

Create a non-executable skeleton from safe topology:

```bash
avalan flow mermaid skeleton diagram.mmd \
  --mode presentation \
  --name profile_projection \
  --flow-version 1
```

Skeletons need explicit node contracts, mappings, entry behavior, and output
behavior before they can execute.

## Task-Backed Execution

Use task-backed flow execution when the flow needs durable state, file
materialization, artifact retention, provider file delivery, queued workers,
usage accounting, or human review.

Task-backed flow definitions reuse task input and output validation. File
inputs remain descriptors or artifact references; raw file bytes and private
paths are not emitted in diagnostics, events, inspection output, or trace
exports.

Durable inspection commands operate on task run ids:

```bash
avalan flow inspect RUN_ID --store-dsn "$AVALAN_TASK_STORE_DSN" --json
avalan flow trace RUN_ID --store-dsn "$AVALAN_TASK_STORE_DSN" --json
avalan flow cancel RUN_ID --store-dsn "$AVALAN_TASK_STORE_DSN" --json
```

## Human Review

Human review is a durable pause node. Its config declares the review payload
schema, allowed decisions, decision schema, timeout, and optional public audit
metadata.

```toml
[nodes.review]
type = "human_review"

[nodes.review.mapping.payload]
type = "select"
source = "risk.value"

[nodes.review.config]
allowed_decisions = ["approved", "rejected", "needs_correction"]
timeout_seconds = 3600

[nodes.review.config.payload_schema]
type = "object"
required = ["summary"]

[nodes.review.config.payload_schema.properties.summary]
type = "string"

[nodes.review.config.decision_schema]
type = "object"
required = ["decision"]

[nodes.review.config.decision_schema.properties.decision]
enum = ["approved", "rejected", "needs_correction"]

[nodes.review.config.decision_schema.properties.comment]
type = "string"

[nodes.review.config.audit_metadata]
queue = "risk"

[[edges]]
source = "review"
target = "approve"
kind = "resume"
label = "approved"

[[edges]]
source = "review"
target = "reject"
kind = "resume"
label = "rejected"

[[edges]]
source = "review"
target = "repair"
kind = "resume"
label = "needs_correction"

[[edges]]
source = "review"
target = "expired"
kind = "timeout"
label = "expired"
```

Resume with a JSON object keyed by review node:

```bash
avalan flow resume review.flow.toml RUN_ID \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --decision-json '{"review":{"decision":"approved","comment":"ok"}}' \
  --json
```

## Privacy Defaults

Public diagnostics and trace export do not include raw prompts, raw file bytes,
private filenames, secrets, provider payload bodies, token text, raw model
outputs, stack traces, or unsanitized tool diagnostic payloads. Keep flow
metadata public-safe; private instructions and provider settings belong in
agent and task configuration surfaces that already enforce their own privacy
contracts.

## Compatibility Boundary

Older Python-constructed native flows can still use `Flow`, `Node`,
`Connection`, callable conditions, callable filters, and `Flow.execute_async`.
That surface is compatibility inventory. New structured TOML should avoid
topology-inferred entrypoints, terminal-output inference, callable imports,
prompt mutation fields, provider configuration, and Mermaid-derived execution
semantics.
