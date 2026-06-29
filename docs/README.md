# Avalan Documentation

Avalan is a Python SDK and CLI for building, orchestrating, and serving AI
agents and workflows across local and hosted models.

## User guides

- [Models](MODELS.md) - Choose hosted provider URIs or local backends, tune
  generation settings, stream responses, and understand where the model layer
  fits.
- [Agents](AGENT_GUIDE.md) - Configure agent TOML, prompts, tools, memory,
  structured output, multimodal inputs, and serving protocols.
- [Tools and reasoning](TOOLS.md) - Use built-in tools, custom tools,
  confirmation, observability, safety boundaries, and reasoning strategies.
- [Memories](MEMORIES.md) - Configure recent memory, permanent memory,
  namespaces, chunking, retrieval, and memory tools.
- [Flows](FLOWS.md) - Author multi-step graph workflows with branching,
  explicit nodes, review boundaries, and task-backed execution.
- [Tasks](TASKS.md) - Define durable task contracts, file delivery, queues,
  storage, schema validation, and repeatable execution.

## Setup and references

- [Install](INSTALL.md) - Platform setup, extras, backend dependencies, and
  source builds.
- [CLI reference](CLI.md) - Complete command and flag reference.
- [Model URI syntax](ai_uri.md) - Engine URI grammar, providers, tokens, and
  backend selection.
- [Modalities](MODALITIES.md) - Text, vision, and audio examples.
- [Runnable examples](examples/README.md) - Agent, model, flow, task, and
  tool examples.
- [Tutorials](tutorials/) - Longer walkthroughs and notebooks.
- [Development](DEVELOPMENT.md) - Release, test, translation, and recording
  maintenance notes.

## Operations and advanced guides

- [DS4 native backend](DS4.md) - Run DS4-supported DeepSeek V4 Flash GGUF
  files with `--backend ds4`, including CLI examples, URI configuration,
  limitations, and integration-test environment variables.
- [Memory PostgreSQL storage](MEMORY_POSTGRESQL.md) - Configure permanent
  memory PostgreSQL schemas and extension prerequisites.
- [Task PostgreSQL storage](TASK_POSTGRESQL.md) - Configure task PostgreSQL
  storage and schema migration dependencies.
- [Task queue operations](TASK_OPERATIONS.md) - Deploy task workers and follow
  failure-mode runbooks for durable queue mode.
- [Container execution](CONTAINERS.md) - Configure trusted container profiles,
  understand current runtime scope, optional live-runtime gates, platform
  limits, security posture, diagnostics, and deferred conformance.
- [Isolation execution](ISOLATION.md) - Configure local, sandbox, and
  container execution modes, trusted policy fields, approval behavior,
  fail-closed semantics, diagnostics, and optional live-runtime gates.
- [Isolation traceability](ISOLATION_TRACEABILITY.md) - Map isolation safety
  claims to concrete test files and commands.
- [Task file delivery](task_file_delivery.md) - Understand direct and queued
  file descriptors, provider references, conversion, local backends, and the
  file-delivery capability matrix.
- [Flow authoring](FLOW_AUTHORING.md) - Author strict flow definitions, bind
  Mermaid Flow Views, run task-backed flows, and handle human review safely.
- [Native flow compatibility](FLOW_COMPATIBILITY.md) - Inventory current flow
  runtime boundaries, strict CLI support, and remaining native compatibility
  behavior.
- [Flow fixture inventory](FLOW_FIXTURE_INVENTORY.md) - Track fixture
  categories, acceptance ownership, and permanent security buckets for strict
  flow behavior.
- [Terminal recording](RECORDING.md) - Capture Avalan CLI sessions as real
  terminal video with colors, live panels, and tool-call updates.
