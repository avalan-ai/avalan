# Agents

Avalan agents package a model, instructions, tools, memory, and runtime
policy into a single TOML definition. They are meant to be explicit: the
model is visible, tool access is declared, memory is configured, and serve
protocol selection can be declared in the file.

Use an agent when the same assistant behavior needs to be run from the CLI,
embedded in Python, served over HTTP, or reused by flows and tasks.

## What an Agent Defines

An agent can include:

- A model, selected with a local model id or an `ai://` vendor URI.
- System instructions and prompt templates.
- Generation settings such as temperature, token limits, and streaming.
- Built-in or custom tools.
- Recent-message memory and permanent memory.
- Structured output schemas.
- Compatibility with file inputs through model/provider capabilities; use
  tasks and flows for file limits, conversion, storage, and privacy.
- Reasoning strategy settings.
- Serving protocols such as OpenAI-compatible HTTP, MCP, A2A, and co-hosted
  flow endpoints.

The reference file is
[src/avalan/agent/templates/blueprint.toml](../src/avalan/agent/templates/blueprint.toml).

## Minimal Agent

```toml
[agent]
name = "support-writer"
role = "support assistant"
task = "Write concise, accurate customer support replies."
goal_instructions = """
Use the provided context before answering. If the answer is not supported by
the context, say what is missing.
"""

[engine]
uri = "ai://env:OPENAI_API_KEY@openai/gpt-4o"

[run]
max_new_tokens = 500
temperature = 0.2
```

Run it from the CLI:

```sh
echo "Draft a reply explaining the refund policy." \
    | avalan agent run support-writer.toml
```

The same file can be loaded by the SDK, referenced by a flow node, or wrapped
by a task contract.

## Prompt Templates

Agent prompt fields such as `role`, `task`, `goal_instructions`, `rules`,
`user`, and `user_template` can use Jinja2 expressions. Values under
`[template]` become template variables for those fields.

```toml
[agent]
name = "support-writer"
role = "You write customer support replies for {{product}}."
task = "Write a reply for the ticket."
goal_instructions = "Use the provided case details before answering."
user = "Ticket:\n{{input}}"

[template]
product = "Avalan"
```

For direct prompt agents, `system`, `developer`, and `user` can be used
instead of `role`, `task`, and `goal_instructions`; direct `system` and
`developer` fields are passed through as prompt text. `user` and
`user_template` are mutually exclusive.

## Model Control

Agents use the same model runtime as `avalan model run`. The engine can point
to hosted vendors, custom endpoints, or local backends.

```toml
[engine]
uri = "ai://env:OPENAI_API_KEY@openai/gpt-4o"

[run]
temperature = 0.1
top_p = 0.9
max_new_tokens = 800
```

OpenAI Responses pre-output `response.failed` stream handling defaults to 24
retries with a 1.0 second delay. Override them with
`openai_response_failed_retries` and
`openai_response_failed_retry_delay_seconds` under `[run]`, or with the
matching `GenerationSettings` fields in SDK code. Set retries to `0` to
disable this retry.

For local models, use the model id plus the backend options supported by the
runtime. See [MODELS.md](MODELS.md) and [ai_uri.md](ai_uri.md).

## Tools

Tools are declared in the agent file so tool access is reviewable before the
agent runs.

```toml
[tool]
enable = ["browser.open", "memory", "mcp.call", "a2a.call"]
```

Avalan can expose built-in tools for browser automation, code execution,
databases, memory, MCP servers, A2A agents, shell file/media inspection, and
custom Python tools. Tool calls are normal runtime events, so applications can
log them, display them, or require approval before execution. In the CLI, use
`--tools-confirm` when tool calls should be approved before they run.

See [TOOLS.md](TOOLS.md) for tool configuration and reasoning strategies.

## Skills

Skills are reusable instruction resources exposed as read-only tools. The
agent can enable `skills.list`, `skills.match`, `skills.read`, or
`skills.check`, while trusted source roots are supplied by SDK, CLI, or host
configuration.

```toml
[tool]
enable = ["skills.match", "skills.read"]

[tool.skills]
source_labels = ["workspace-main"]
skill_ids = ["pdf"]
bootstrap = "auto"

[tool.skills.read_limits]
max_bytes_per_read = 65536
max_lines_per_read = 2000
```

`[tool.skills]` can only narrow trusted settings. It must not define source
paths, install skills, or grant tools. Skill instructions remain subordinate
to higher-priority instructions and runtime policy.

Run the tracked PDF skill example with trusted CLI settings:

```sh
echo "Plan a PDF review." \
    | avalan agent run docs/examples/agent_skills_pdf.toml \
        --tool-skills-source workspace-main=docs/examples/skills \
        --tool-skills-source-authority workspace-main=workspace:docs \
        --display-tools
```

See [TOOLS.md](TOOLS.md#skills-tools) for response envelopes, statuses,
diagnostics, path policy, privacy defaults, and migration notes.

## Memory

Agents can keep short-term conversation state and retrieve long-term context
from permanent stores.

```toml
[tool]
enable = ["memory"]

[memory]
recent = true
permanent_message = "postgresql://user:pass@localhost:5432/avalan"

[memory.permanent]
support = "postgresql://user:pass@localhost:5432/avalan"
```

`recent` keeps the current session transcript available during the run.
`permanent_message` persists conversation history by agent, participant, and
session. `[memory.permanent]` maps retrieval namespaces to indexed document or
raw memory stores.

Short-term memory is useful for one session. Permanent message memory is
useful for cross-session conversation history. Permanent document memory is
useful for knowledge, preferences, prior work, and retrieval over stored
documents.

See [MEMORIES.md](MEMORIES.md) for memory backends and retrieval patterns.

## File Inputs and Multimodal Agents

Agents can receive files directly with `--input-file`, through an
OpenAI-compatible served endpoint, or through tasks and flows that define file
limits, conversion rules, and privacy behavior.

Common patterns include:

- Vision agents that receive images or rendered PDF pages.
- Audio agents that transcribe or analyze media.
- Document extraction agents that return JSON validated against a schema.
- Multimodal agents that combine text instructions with files.

For repeatable file workflows, wrap the agent with a task. See
[TASKS.md](TASKS.md) and [task_file_delivery.md](task_file_delivery.md).

## Structured Output

Agents can produce JSON that is validated by a schema, which is especially
useful for extraction, classification, routing, and automation.

```toml
[run.response_format]
type = "json_schema"
name = "invoice_extraction"
strict = true
schema_ref = "invoice.schema.json"
```

In production flows, prefer schema-bound outputs whenever another system will
consume the result.

## Serving Agents

The same agent can be served through multiple protocols.

```toml
[serve]
protocols = ["openai:responses,completion", "mcp", "a2a", "flow"]
```

Run the server with the CLI command for agent serving:

```sh
avalan agent serve docs/examples/agent_tool.toml --host 127.0.0.1 --port 9001
```

Then connect clients through the protocol they already support:

- OpenAI-compatible clients use `/v1` for completions and Responses.
- MCP clients use `/mcp` and call the configured MCP tool name.
- A2A clients use `/a2a` and call the configured agent skill.
- Flow clients use the co-hosted flow API surface when `flow` is enabled.

This lets one agent definition serve local tools, hosted model calls,
OpenAI-compatible clients, MCP clients, and A2A clients without duplicating
runtime logic.

## Proxy Agents

`avalan agent proxy` is a shortcut for serving a model URI as an agent with
recent memory, default PostgreSQL permanent-message memory, and the same
serving surfaces. Override the permanent-message DSN for your environment, or
disable/avoid proxy mode when you do not want durable message storage.

```sh
avalan agent proxy \
    --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
    --memory-permanent-message "postgresql://user:pass@localhost:5432/avalan" \
    --run-max-new-tokens 1024 \
    --host 127.0.0.1 \
    --port 9001
```

## Design Guidance

- Keep agent files narrow. If work has multiple deterministic steps, use a
  flow and call the agent from the relevant node.
- Put file limits, durable execution, and output contracts in tasks.
- Require confirmation for tools that can mutate data, spend money, browse
  authenticated sessions, or execute code.
- Prefer structured output when downstream automation depends on the answer.
- Treat memory as an explicit dependency. Name namespaces and retrieval
  behavior deliberately.
- Keep model settings close to the agent so inference behavior is reviewable.

## Related Documentation

- [TOOLS.md](TOOLS.md) - Tool configuration and reasoning strategies.
- [MEMORIES.md](MEMORIES.md) - Recent and permanent memory.
- [FLOWS.md](FLOWS.md) - Multi-step orchestration.
- [TASKS.md](TASKS.md) - Durable task contracts and queues.
- [MODELS.md](MODELS.md) - Model selection and backend control.
- [MODALITIES.md](MODALITIES.md) - Text, vision, and audio examples.
