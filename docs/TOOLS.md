# Tools and Reasoning

Avalan tools make external capabilities available to agents while keeping the
runtime loop explicit. A model can ask for a tool call, Avalan parses and
validates the call, your application can confirm or reject it, and the result
returns to the model as an observable event.

Reasoning strategies belong in this guide because strategy controls when the
model should think, when it should call a tool, and when the loop should stop.

## Tool Loop

The core abstractions are `Tool`, `ToolSet`, and `ToolManager`:

- `Tool` is an async callable with typed parameters and a model-facing
  docstring.
- `ToolSet` groups related tools under a namespace such as `database` or
  `shell`.
- `ToolManager` exposes selected tools, generates JSON schemas, parses tool
  calls, validates arguments, applies filters, runs tools, and emits
  diagnostics.

Tool names are namespaced. Enabling `math.calculator` exposes one tool;
enabling `database` or `database.*` exposes the database toolset. Shell tools
use explicit names such as `shell.rg` and `shell.pdftotext`.

## Enable Tools

Declare tools in an agent file:

```toml
[tool]
enable = [
    "math.calculator",
    "browser.open",
    "database",
    "memory",
]
```

Or add them at runtime:

```sh
echo "What is (4 + 6) * 5 / 2?" \
  | avalan agent run \
      --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4o-mini" \
      --tool "math.calculator" \
      --display-tools
```

Use `--display-tools` to show tool calls and results, `--display-events` for
non-tool stream events, and `--tools-confirm` when a human should approve each
call before execution.

## Schemas and Formats

Avalan derives tool schemas from Python type hints and docstrings. The first
docstring sentence becomes the model-facing description, `Args:` entries become
parameter descriptions, and return documentation describes the result. The
implicit runtime `context` parameter is never exposed to the model.

Tool names stay canonical inside Avalan, such as `database.run` or
`mcp.call`. When a provider requires function names without dots, Avalan maps
the provider-safe name back to the canonical tool before execution.

Tool-call parsing supports provider-native and text formats:

| Format | Use |
| --- | --- |
| `json` | Generic JSON tool calls. |
| `openai` | OpenAI-style function calling. |
| `react` | `Action:` and `Action Input:` text. |
| `bracket` | Compact bracket calls. |
| `harmony` | Harmony channel/tool-call format. |
| `dsml` | DSML tool syntax. |

## Current Built-in Toolsets

Built-in tool availability depends on installed extras and configuration.
Graph, code, and browser tools need their optional dependencies; database
tools need database settings; MCP and A2A tools are available only when
enabled.

| Toolset | Tools | Use |
| --- | --- | --- |
| `math` | `math.calculator` | Evaluate arithmetic and algebra through SymPy. |
| `graph` | `graph.pie`, `graph.bar`, `graph.line`, `graph.scatter`, `graph.histogram` | Render charts through Matplotlib and return image metadata. |
| `code` | `code.run`, `code.search.ast.grep` | Execute small Python snippets in RestrictedPython or search/rewrite code with `ast-grep`. |
| `database` | `count`, `inspect`, `keys`, `relationships`, `plan`, `run`, `sample`, `locks`, `tasks`, `kill`, `size`, `tables` | Inspect schemas, query data, explain queries, and inspect database operations. |
| `browser` | `browser.open` | Render a page with Playwright and return Markdown, optionally narrowed by search context. |
| `memory` | `memory.message.read`, `memory.read`, `memory.list`, `memory.stores` | Retrieve prior messages and permanent memory partitions. |
| `mcp` | `mcp.call` | Call tools exposed by an MCP server. |
| `a2a` | `a2a.call` | Call another A2A agent as a tool, including file forwarding. |
| `shell` | `rg`, `head`, `tail`, `ls`, `cat`, `nl`, `file`, `find`, `wc`, `awk`, `sed`, `jq`, `pdfinfo`, `pdftotext`, `pdftoppm`, `tesseract`, `pipeline` | Read, inspect, search, transform, and compose workspace file operations under policy limits. `shell.pipeline` also requires `allow_pipelines = true`. |

`search_engine.search` also exists as a simple SDK/demo tool. It is useful for
tests or custom toolsets, but production search should be backed by a real
provider and registered deliberately.

## MCP: Tools from HTTP Servers

MCP is the right boundary when an agent should call capabilities hosted by an
HTTP MCP service. The current client sends streamable HTTP JSON-RPC requests
and accepts JSON or SSE responses. Stdio/local MCP servers need an HTTP MCP
service in front of them before `mcp.call` can reach them.

Enable `mcp.call`, then tell the agent which MCP endpoint and tool to use.

```sh
echo "Use the MCP service to answer the request." \
  | avalan agent run \
      --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4o-mini" \
      --tool "mcp.call" \
      --developer 'Call `mcp.call` with uri `http://127.0.0.1:9001/mcp`, name `ask_avalan`, and arguments {"input_string":"<the user request>"}.' \
      --tools-confirm
```

Avalan agents can also be served as MCP tools:

```sh
avalan agent serve docs/examples/agent_tool.toml \
  --protocol mcp \
  --mcp-name ask_avalan \
  --mcp-description "Ask the avalan-hosted agent." \
  --host 127.0.0.1 \
  --port 9001
```

This makes MCP a practical agent inter-operation path: one Avalan process can
serve a specialized agent, and another agent or MCP client can call it as a
tool. MCP is best for tool-style calls where the client expects a named tool,
structured arguments, progress notifications, and a final tool result.

`mcp.call` accepts model-supplied `uri` and `name` values. In production,
pin or allowlist the endpoint and tool name in application code, a custom
wrapper, or a `ToolFilter`; do not rely on prompt text as the policy boundary.
Use `--tools-confirm` as an interactive safeguard for humans, not as the only
control.

## A2A: Agent-to-Agent Calls

A2A is the right boundary when the remote capability is itself an agent with
task, status, artifact, tool-call, and intermediate-output events. Enable
`a2a.call` in the caller agent:

```sh
echo "Ask the remote PDF agent for the invoice number and total." \
  | avalan agent run \
      --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4o-mini" \
      --tool "a2a.call" \
      --input-file docs/examples/playground/invoice.pdf \
      --developer 'Call `a2a.call` with uri `http://127.0.0.1:9017/a2a`, name `run`, and arguments {"input_string":"Return the invoice number and total."}.'
```

Serve the remote agent with A2A enabled:

```sh
avalan agent serve \
  --protocol a2a \
  --a2a-name run \
  --a2a-description "Extract facts from attached PDFs." \
  --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4o-mini" \
  --tool "shell.pdfinfo" \
  --tool "shell.pdftotext" \
  --tool-shell-allow-media-tools \
  --tool-shell-workspace-root "." \
  --host 127.0.0.1 \
  --port 9017
```

The caller's native input files can be forwarded as A2A file parts, so the
remote agent receives the prompt and the files without putting file bytes in
the tool arguments. Use A2A when you want agent-level delegation, streamed
status, artifacts, and a remote agent card.

`a2a.call` also accepts model-supplied `uri` and `name` values. Treat those as
network trust decisions: pin or allowlist remote agents in the application
layer, and confirm delegations that can trigger external actions or expose
files.

## Serving Agents over MCP and A2A

Avalan can be a protocol client and a protocol server at the same time.
`mcp.call` and `a2a.call` let one agent call remote capabilities. Serving an
agent with MCP or A2A lets external clients, or other Avalan agents, call that
agent.

```toml
[serve]
protocols = ["openai:responses,completion", "mcp", "a2a"]
mcp_prefix = "/mcp"
mcp_name = "run"
a2a_prefix = "/a2a"
a2a_name = "run"
```

```sh
avalan agent serve agent.toml \
    --protocol openai:responses,completion \
    --protocol mcp \
    --protocol a2a \
    --mcp-name run \
    --a2a-name run
```

Use MCP to expose a narrow, schema-described capability as a tool. Use A2A to
expose a full agent that can stream status, return artifacts, and keep task
state. A common topology is a coordinating Avalan agent with `mcp.call` for
services and `a2a.call` for specialist agents.

MCP and A2A requests execute the served agent's configured capabilities. They
do not grant arbitrary local shell authority to the remote caller. A served
agent can expose `shell.pipeline` only when its trusted agent configuration
enables that tool and sets `[tool.shell] allow_pipelines = true`; MCP/A2A
payloads cannot enable pipelines or alter shell runtime settings.

## Shell Tools

Shell tools are not arbitrary shell execution. They are wrappers around
specific commands, checked against `ShellToolSettings`, path policy, byte
limits, timeout limits, and output limits.

```toml
[tool]
enable = ["shell.rg", "shell.head", "shell.jq"]

[tool.shell]
workspace_root = "."
cwd = "."
materialized_input_files_dir = "avalan-input-files"
max_stdout_bytes = 65536
allow_media_tools = false
allow_pipelines = false
```

Media tools such as `shell.pdfinfo`, `shell.pdftotext`, `shell.pdftoppm`, and
`shell.tesseract` require `allow_media_tools = true` or the corresponding CLI
flag. Absolute paths, symlinks, hidden files, and executable search paths are
also opt-in.

Attached and generated files that need to be exposed to shell tools are
materialized under `workspace_root / materialized_input_files_dir`. The
directory is workspace-relative and defaults to `avalan-input-files`.

Agent TOML is trusted deployment configuration only when supplied through an
operator-controlled channel. In that mode `[tool.sandbox]` and
`[tool.container]` may define approved isolation profiles, while
`[tool.shell.sandbox]` and `[tool.shell.container]` may select one profile.
Top-level sandbox policy is valid only when `[tool.shell]` selects the
sandbox `backend`. Top-level container policy is valid when `[tool.shell]`
selects the container `backend`, or when `[runtime.container]` uses it as the
whole-agent runtime envelope without shell container execution.
Untrusted agent, flow, task, prompt, MCP, A2A, or HTTP inputs must be capped to
supported profile selectors where that surface allows selection; they must not
define roots, images, mounts, executable paths, backend flags, network policy,
or secrets.

See [Isolation execution](ISOLATION.md) for the supported mode/backend matrix,
policy fields, examples, approval behavior, diagnostics, and real-runtime
test gates.

### Structured Shell Pipelines

`shell.pipeline` composes existing shell tools through structured JSON
arguments. It is not a shell string runner: Avalan never evaluates
model-supplied `cmd | other_cmd`, `bash -c`, `/bin/sh -c`, arbitrary
executables, redirections, command substitutions, or public stdin. The tool is
default-denied and appears only when trusted configuration both selects
`shell.pipeline`, `shell`, or `shell.*` and sets `allow_pipelines = true`.

```toml
[tool]
enable = ["shell.pipeline"]

[tool.shell]
workspace_root = "."
cwd = "."
allow_pipelines = true
max_pipeline_stages = 3
max_pipeline_bytes = 1048576
max_intermediate_bytes = 262144
max_stdout_bytes = 65536
max_stderr_bytes = 32768
allowed_commands = ["cat", "rg", "wc", "sed", "awk", "jq"]
```

A model-facing call is an object with ordered `steps`. Each step has a stable
`id`, a known shell `command`, optional command-specific `options`, optional
workspace-relative `paths`, optional `cwd`, and optional `stdin_from`. Stream
references are typed objects and currently support only `stdout`.

```json
{
  "mode": "pipeline",
  "steps": [
    {
      "id": "search",
      "command": "rg",
      "options": {"pattern": "class ShellToolSettings"},
      "paths": ["src/avalan/tool/shell"]
    },
    {
      "id": "count",
      "command": "wc",
      "options": {"lines": true},
      "stdin_from": {"step_id": "search", "stream": "stdout"}
    }
  ],
  "timeout_seconds": 10,
  "max_stdout_bytes": 8192,
  "max_stderr_bytes": 8192,
  "max_intermediate_bytes": 262144
}
```

`mode = "pipeline"` is a linear byte pipeline. The first step cannot read
stdin, and each later step either omits `stdin_from` and implicitly reads the
previous step's stdout or explicitly references the previous step's stdout.
`mode = "serial"` runs shell-local steps in order and permits `stdin_from`
only from an earlier step. `mode = "parallel"` runs independent shell-local
steps and rejects `stdin_from`.

Policy normalization happens before execution:

- `allow_pipelines` must be true and stage count must fit
  `max_pipeline_stages`.
- Each command is normalized by the same shell command policy used for direct
  `shell.cat`, `shell.rg`, `shell.jq`, and other tools.
- `allowed_commands`, path policy, cwd policy, executable resolution, timeout
  limits, argument budgets, output byte caps, and media gates still apply.
- `stdin_from` references must name existing earlier stages and stream
  `stdout`.
- Text and JSON stream contracts are checked before bytes are routed.
- Generated-output commands such as page rasterization and OCR output files
  are not valid producers inside byte pipelines.
- Public `ShellCommandRequest.stdin` and the `"-"` path sentinel remain
  denied.

Only final-stage stdout is user-visible by default. Intermediate stdout is
transport data: it is capped by `max_intermediate_bytes`, included only in
stage metadata when the runtime projects safe events, and redacted or
truncated before display, task, MCP, A2A, server, or audit output. Stderr is
collected per stage and contributes to the formatted result without making
private host paths public.

Full byte-stream pipelines are local-only. When shell execution is
`backend = "sandbox"` or `backend = "container"`, `mode = "pipeline"` and any
composition with `stdin_from` fail closed with a policy-denied result. Isolated
`serial` or `parallel` compositions without stdin routing may delegate through
the existing single-command executor when the selected backend and profile can
enforce the individual commands. Avalan does not lower any isolated composition
to shell text.

See [agent_shell_pipeline.toml](examples/agent_shell_pipeline.toml),
[shell_pipeline.flow.toml](examples/flows/shell_pipeline.flow.toml), and the
pipeline task examples in [docs/examples/tasks](examples/tasks/README.md).

## Database Tools

Database tools share configured SQLAlchemy settings. Keep database agents
read-only unless the workflow explicitly requires mutation.

```toml
[tool]
enable = ["database"]

[tool.database]
dsn = "postgresql+asyncpg://user:password@localhost:5432/app"
read_only = true
allowed_commands = "select"
```

A typical data agent should inspect tables and relationships before running
SQL. Use `--tools-confirm` when connecting an agent to production data.

For production read-only agents, prefer explicit database tools such as
`database.tables`, `database.inspect`, `database.relationships`,
`database.sample`, `database.plan`, and `database.run` instead of enabling the
whole `database` namespace. Enabling `database` also exposes operational tools
such as `database.kill`; SQL `allowed_commands` constrains SQL execution but
does not make operational tools harmless.

## Browser Tools

`browser.open` renders a page and returns Markdown. It can optionally search
the rendered content before returning it to reduce context size.

```sh
echo "Summarize the latest release page." \
  | avalan agent run \
      --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4o-mini" \
      --tool "browser.open" \
      --tool-browser-search \
      --tool-browser-search-context 10
```

Browser output becomes model context. Be careful with authenticated pages,
internal systems, and large pages.

## Memory Tools

Memory tools are read/list oriented:

- `memory.message.read(search)` searches prior user messages.
- `memory.read(namespace, search)` retrieves permanent memory partitions.
- `memory.list(namespace)` lists stored memory entries.
- `memory.stores()` lists configured stores.

Use [MEMORIES.md](MEMORIES.md) for indexing, namespaces, and storage setup.

## Search Strategy

Search in Avalan is a choice of retrieval surface:

| Surface | Use |
| --- | --- |
| `browser.open` | Retrieve and search known URLs. |
| `shell.rg` | Search workspace files. |
| `memory.message.read` | Search prior user messages. |
| `memory.read` | Search indexed permanent memory. |
| `database.run` or `database.sample` | Search structured data. |
| `mcp.call` | Call an external web or enterprise search service. |
| `a2a.call` | Delegate research to another agent. |

For open-ended internet search, prefer a real search provider exposed through
MCP or a specialist A2A research agent.

## Custom Tools

Custom tools subclass `Tool`. Avalan derives tool schemas from type hints and
docstrings, excluding implicit `context` parameters.

```python
from avalan.tool import Tool


class LookupCustomerTool(Tool):
    """Look up a customer by email address.

    Args:
        email: Customer email address.

    Returns:
        Customer profile fields available to the agent.
    """

    async def __call__(self, email: str) -> dict[str, str]:
        assert email
        return {"email": email, "tier": "enterprise"}
```

Keep docstrings accurate. Agents use them when selecting tools.

## Reasoning Models

Reasoning models can emit hidden or visible thinking content. Avalan can parse
reasoning tags, display reasoning, start generation inside a thinking block,
or limit reasoning token output.

```sh
echo "What is (4 + 6) and then that result times 5, divided by 2?" \
  | avalan model run "ai://local/openai/gpt-oss-20b" \
      --backend mlx \
      --max-new-tokens 1024 \
      --display-reasoning
```

Useful controls:

- `--start-thinking` when a model assumes reasoning starts immediately.
- `--reasoning-tag think` or `--reasoning-tag channel` for parser mode.
- `--reasoning-effort none|minimal|low|medium|high|xhigh|max` for providers
  that support it.
- `--reasoning-max-new-tokens` and `--reasoning-stop-on-max-new-tokens` to
  bound long reasoning traces.
- `--no-reasoning` when reasoning parsing should be disabled.

## ReACT and Tool Formats

ReACT interleaves model reasoning and tool use. Avalan supports multiple tool
formats, including `json`, `react`, `bracket`, `openai`, `harmony`, and
`dsml`.

```sh
echo "Use the page to explain what avalan does." \
  | avalan agent run \
      --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
      --backend mlx \
      --tool-format react \
      --tool "browser.open" \
      --maximum-tool-cycles 4 \
      --display-tools
```

Use the format expected by the model family. For example, OpenAI GPT-OSS style
local runs often use `harmony`, while DS4 native tool calls use `dsml`.

## Safety and Control

- Expose only the tools needed for the task.
- Use `--tools-confirm` for tools that can mutate state, query sensitive data,
  browse authenticated sessions, call remote services, or spend money.
- Set `--maximum-tool-cycles` to avoid unbounded model/tool loops.
- Prefer read-only database settings and narrow `allowed_commands`.
- For read-only database agents, enable explicit database tools and exclude
  `database.kill`.
- Scope shell tools to a trusted `workspace_root`; opt into media tools,
  absolute paths, symlinks, hidden files, and executable paths deliberately.
- Treat MCP and A2A endpoints as network trust boundaries; pin or allowlist
  remote `uri` and `name` values in code for production agents.
- Log tool events safely. Avoid storing secrets, raw file bytes, or sensitive
  tool arguments in production logs.

## Choosing the Boundary

- Use a tool for one capability.
- Use MCP when that capability is hosted as a remote tool server.
- Use A2A when the remote capability is a full agent with status and artifacts.
- Use an agent when the model needs instructions, tools, memory, and serving.
- Use a flow when deterministic orchestration or branching matters.
- Use a task when work needs contracts, file delivery, queueing, storage, or
  durable status.

## Related Documentation

- [AGENT_GUIDE.md](AGENT_GUIDE.md) - Agent configuration and serving.
- [MEMORIES.md](MEMORIES.md) - Memory tools and retrieval.
- [FLOWS.md](FLOWS.md) - Tool calls inside multi-step workflows.
- [TASKS.md](TASKS.md) - Durable task execution.
- [CLI.md](CLI.md) - Complete CLI flags.
