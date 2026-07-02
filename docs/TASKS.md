# Tasks

Avalan tasks package repeatable work behind explicit input and output
contracts. A task can run an agent or flow target while adding operational
concerns such as file delivery, validation, queueing, storage, privacy, and
durable execution.

Use tasks when a workflow needs to be run repeatedly and safely, especially
when files, schemas, queues, or external callers are involved.

## What a Task Adds

A task can define:

- The runtime target to execute.
- Text, JSON, and file inputs.
- File type and size limits.
- Conversion rules for files.
- Output format and JSON schema.
- Inline or queued execution.
- Durable status plus privacy-sanitized snapshots and events; raw inputs,
  outputs, or file bytes only when explicitly retained.
- Privacy and retention behavior.
- Idempotency keys and retry policy.
- Artifact storage policy.
- Metrics and trace sinks.

This lets the model or flow focus on the work while the task controls the
boundary around it.

## Run a Task

The example below runs a flow-backed invoice extraction task over a PDF and
writes JSON output. It requires the `task`, `task-pdf-images`, `agent`, and
`vendors` extras, plus provider credentials for the referenced model.

```sh
avalan task run docs/examples/tasks/poc_extraction/image_flow_task.toml \
    --ephemeral \
    --pdf docs/examples/tasks/poc_extraction/sample.pdf \
    --json \
    --output image.json
```

That task renders PDF pages to images, sends them to a vision agent, and
validates the result against a schema.

## Task Structure

Task definitions are TOML files. The exact fields depend on the target and
input type, but a task usually includes:

```toml
[task]
name = "invoice-extraction"
version = "1"

[input]
type = "file"
mime_types = ["application/pdf"]

[output]
type = "object"
schema_ref = "invoice.schema.json"

[execution]
type = "flow"
ref = "image_flow.toml"

[limits]
file_count = 1
file_bytes = 1048576
```

See [docs/examples/tasks](examples/tasks/README.md) for validated example
definitions and runnable commands; some require provider or runtime
configuration.

## Execution Targets

Task definitions have a target type under `[execution]`. The current validated
CLI path supports `agent` and `flow` targets:

```toml
[execution]
type = "agent"
ref = "agents/basic_answer.toml"
```

```toml
[execution]
type = "flow"
ref = "image_flow.toml"
```

The task type model also includes `task`, `model`, `callable`, and `tool` for
registry-backed or programmatic runners. Treat those as extension points unless
your application registers a runner for them; the built-in examples and CLI
validation use agent and flow targets today.

## Shell Pipeline Tasks

Tasks can wrap an agent or flow that uses `shell.pipeline`, but pipeline
authority remains runtime configuration. The task input, queue payload, MCP
request, or A2A request cannot enable pipelines by itself.

- Agent-backed tasks may reference an agent TOML that explicitly enables
  `shell.pipeline` and sets `[tool.shell] allow_pipelines = true`.
- Flow-backed tasks with a strict `tool` node require an SDK, host
  application, or worker runtime with a configured tool resolver that exposes
  `shell.pipeline` and trusted shell settings with `allow_pipelines = true`.
  The standalone `avalan task run` path uses the default flow validator for
  this fixture and must not be treated as the supported runtime for
  `pipeline_flow.task.toml`.
- Queued workers force implicit pipeline settings closed unless the operator
  starts the worker with explicit pipeline flags.

Validated examples live at
[pipeline_agent.task.toml](examples/tasks/pipeline_agent.task.toml) and
[pipeline_flow.task.toml](examples/tasks/pipeline_flow.task.toml). The flow
example uses structured step objects and local-only byte routing.

## Skills in Tasks

Tasks can carry `[skills]` settings for eligible targets: `agent`, `flow`,
`model`, `task`, and `tool`. The task definition narrows a trusted registry;
it does not authorize source roots by itself.

```toml
[execution]
type = "agent"
ref = "agents/skills_pdf_reader.toml"

[skills]
source_labels = ["workspace-main"]
skill_ids = ["pdf"]

[skills.read_limits]
max_bytes_per_read = 65536
max_lines_per_read = 2000
```

Direct SDK or host execution supplies `TrustedSkillSettings` while loading the
task. Queue submission records a durable skills identity containing settings
and source fingerprints, enabled skills tools, allowed skill IDs, and registry
metadata. Workers revalidate that identity before execution and fail closed if
the trusted registry is missing, stale, wider than requested, unavailable,
malformed, or policy denied.

The example at [skills_pdf.task.toml](examples/tasks/skills_pdf.task.toml)
uses the hermetic PDF skill under [docs/examples/skills](examples/skills/).
It is covered by focused loader tests with trusted settings; a plain task
payload or remote request cannot grant that authority.

## File Delivery

Tasks are the safest place to describe file behavior:

- Which MIME types are accepted.
- Maximum size and count.
- Whether files are passed directly or through storage.
- Which conversions happen before execution.
- Whether original files are retained.

File delivery rules are documented in
[task_file_delivery.md](task_file_delivery.md).

## Inline and Queued Execution

Use inline execution for local development, tests, and synchronous calls:

```sh
avalan task run task.toml --ephemeral --json
```

Use queued execution when work should be claimed by workers:

```sh
avalan task enqueue task.toml \
    --store-dsn postgresql://user:pass@localhost:5432/avalan \
    --queue default
```

Queue mode is useful for longer jobs, retries, worker pools, and production
systems that need durable task status. The task definition must set
`[run] mode = "queue"`; `--queue` selects the queue name for the submitted run.

Run a worker for a queue:

```sh
avalan task worker \
    --store-dsn "$AVALAN_TASK_STORE_DSN" \
    --queue default \
    --worker-id default-1 \
    --lease-seconds 300 \
    --heartbeat-seconds 30 \
    --limit 100
```

Inspect a durable run:

```sh
avalan task inspect RUN_ID --store-dsn "$AVALAN_TASK_STORE_DSN"
avalan task events RUN_ID --store-dsn "$AVALAN_TASK_STORE_DSN" --after-sequence 0
avalan task output RUN_ID --store-dsn "$AVALAN_TASK_STORE_DSN"
avalan task artifacts RUN_ID --store-dsn "$AVALAN_TASK_STORE_DSN"
```

See [TASK_OPERATIONS.md](TASK_OPERATIONS.md) and
[TASK_POSTGRESQL.md](TASK_POSTGRESQL.md) for storage and worker setup.

## Idempotency, Retries, and Limits

Task runs can be deduplicated with idempotency settings:

```toml
[run]
mode = "direct"
timeout_seconds = 300
idempotency = "input_and_files_hash"
```

Use `input_hash` for structured payloads, `input_and_files_hash` when files
are part of the logical input, `custom` when the caller supplies the key, and
`none` for fully independent runs.

Limits belong in the task definition, not only in prompts:

```toml
[limits]
input_bytes = 8192
file_count = 1
file_bytes = 1048576
output_bytes = 131072
total_tokens = 100000
```

Retry and timeout policy should reflect whether the target is idempotent.
Avoid retrying tasks that can trigger duplicate external side effects unless
the target and tools are designed for it.

## Artifacts and Privacy

Tasks can store output artifacts and sanitized metadata while dropping raw
inputs, prompts, token text, file bytes, or tool results.

```toml
[artifact]
storage = "local"
store_bytes = false
max_count = 24
max_bytes = 1048576

[privacy]
input = "drop"
prompt = "drop"
output = "drop"
files = "drop"
file_bytes = "drop"
token_text = "drop"
tool_arguments = "drop"
tool_results = "drop"
events = "drop"
errors = "drop"
raw_retention_days = 0
```

Use stricter privacy settings for regulated documents, private user data, or
tool calls that include credentials or customer records.

## Tasks and Flows

Many production tasks target flows:

- The task validates and stores inputs.
- The flow orchestrates the steps.
- Agents and tools do the AI and external work.
- The task validates and stores the output.

This separation is useful for document processing, support automation,
research jobs, and any workflow where callers need a stable contract.

## Tasks and Agents

Tasks can also wrap a single agent. Use this when the agent behavior is
already enough, but the caller still needs:

- File limits.
- Schema validation.
- Queueing.
- Durable status.
- A repeatable CLI or API contract.

## Output Validation

Prefer schema-validated JSON for automation. A free-form answer is fine for
humans, but downstream systems need predictable fields.

Common task outputs include:

- Extracted records.
- Classifications.
- Routing decisions.
- Summaries with source references.
- Tool or workflow status payloads.

Validate definitions and representative input before adding a task to a queue:

```sh
avalan task validate task.toml \
    --input-json '{"question":"What changed?","priority":2}'
```

## Operational Guidance

- Keep task inputs narrow and typed.
- Put file limits in the task, not only in prompts.
- Use queued mode for long-running work.
- Store outputs when callers need retries or audit.
- Treat privacy and retention settings as part of the contract.
- Use flows when the task needs multiple deterministic steps.

## Related Documentation

- [TASK_OPERATIONS.md](TASK_OPERATIONS.md) - Queue operations and runbooks.
- [TASK_POSTGRESQL.md](TASK_POSTGRESQL.md) - PostgreSQL task storage.
- [task_file_delivery.md](task_file_delivery.md) - File delivery rules.
- [FLOWS.md](FLOWS.md) - Flow-backed task execution.
- [AGENT_GUIDE.md](AGENT_GUIDE.md) - Agent-backed task execution.
- [CLI.md](CLI.md) - Complete task CLI reference.
