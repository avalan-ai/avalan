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

Skills fit the same loop but have a narrower mental model:

- Agents think.
- Tools act.
- Skills teach.
- Registries disclose.

A skill is Markdown instruction content with model-safe metadata. It does not
execute, grant tools, install packages, mount paths, or change policy. A
trusted registry is built before the run; the model can then discover and
read bounded skill resources through read-only `skills.*` tools.

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
tools need database settings; shell media commands need their external
binaries or Python packages; MCP and A2A tools are available only when
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
| `skills` | `skills.list`, `skills.match`, `skills.read`, `skills.check` | Discover and read trusted instruction resources through a registry. |
| `shell` | `rg`, `head`, `tail`, `ls`, `cat`, `nl`, `file`, `find`, `wc`, `awk`, `sed`, `jq`, `pdfinfo`, `pdftotext`, `pdftoppm`, `reportlab`, `pdfplumber`, `pypdf`, `tesseract`, `pipeline` | Read, inspect, search, transform, and compose workspace file operations under policy limits. `shell.pipeline` also requires `allow_pipelines = true`. |

`search_engine.search` also exists as a simple SDK/demo tool. It is useful for
tests or custom toolsets, but production search should be backed by a real
provider and registered deliberately.

## Skills Tools

The `skills` namespace exposes reusable operating instructions without
stuffing every skill body into the prompt. The registry is operator or SDK
configuration; the model-facing tools are read/query-only.

Use `skills.match` or `skills.list` to discover candidates, `skills.read` to
read the selected instruction resource, and `skills.check` to inspect
availability or diagnostics without reading content. There is no
model-facing `skills.load` tool.

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

`[tool.skills]` in agent TOML is a narrowing surface. It can turn skills off,
turn bootstrap instructions off, filter trusted authority kinds, filter source
labels, filter skill IDs, and reduce read, index, source, cursor, privacy, or
observability settings. It cannot define `source` or `sources`; trusted roots
come from the SDK, CLI, or host process.

Agent CLI commands accept trusted source flags:

```sh
echo "Use the PDF skill to plan the inspection workflow." \
  | avalan agent run docs/examples/agent_skills_pdf.toml \
      --tool skills.match \
      --tool skills.read \
      --tool-skills-source workspace-main=docs/examples/skills \
      --tool-skills-source-authority workspace-main=workspace:docs \
      --tool-skills-skill pdf \
      --tool-skills-max-bytes-per-read 65536 \
      --display-tools \
      --display-events
```

For SDK construction, pass trusted settings through the tool context:

```python
from pathlib import Path

from avalan.skill import (
    SkillReadLimits,
    SkillSourceConfig,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
)
from avalan.tool.context import ToolSettingsContext

skills = TrustedSkillSettings(
    sources=(
        SkillSourceConfig(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(workspace_id="docs"),
            root_path=Path("docs/examples/skills"),
        ),
    ),
    allowed_skill_ids=("pdf",),
    read_limits=SkillReadLimits(max_bytes_per_read=65536),
)
tool_settings = ToolSettingsContext(skills=skills)
```

The current response envelope fields are:

| Field | Meaning |
| --- | --- |
| `status` | Compact result classification. |
| `registry_version` | Deterministic version visible to this run. |
| `items` | Skill metadata or match candidates for discovery tools. |
| `content` | Bounded resource content for `skills.read`. |
| `diagnostics` | Structured findings with code, status, message, path, hint, and optional candidates/details. |
| `provenance` | Model-facing source label, skill ID, resource ID, and content identity. |
| `next_cursor` | Continuation handle when a resource read is truncated. |

Statuses are `ok`, `empty`, `ambiguous`, `not_found`, `disabled`,
`unavailable`, `malformed`, `policy_denied`, `truncated`, `stale`, and
`blocked`.

Diagnostic codes use the `skills.` prefix: `skills.empty_registry`,
`skills.no_match`, `skills.not_found`, `skills.ambiguous_name`,
`skills.disabled`, `skills.source_unavailable`,
`skills.manifest_malformed`, `skills.resource_missing`,
`skills.resource_outside_root`, `skills.resource_oversized`,
`skills.resource_stale`, `skills.duplicate_id`,
`skills.runtime_source_unavailable`, `skills.resource_binary`,
`skills.policy_denied`, and `skills.syntax_unsupported`.

### Flow and Task Skills

Strict flows support a top-level `[skills]` table and node-level
`[nodes.<name>.skills]` tables for agent nodes and strict tool nodes that call
canonical skills tools. Flow settings also narrow trusted settings supplied by
the SDK, host process, or registry-backed loader.

```toml
[skills]
source_labels = ["workspace-main"]
skill_ids = ["pdf"]

[nodes.read_skill]
type = "tool"
ref = "skills.read"

[nodes.read_skill.skills]
skill_ids = ["pdf"]
```

Tasks support `[skills]` for eligible targets, including agent, flow, model,
task, and tool targets. Queued tasks capture a durable skills identity with
settings and source fingerprints, enabled skills tools, allowed skill IDs, and
registry metadata. Workers revalidate that identity before execution and fail
closed if the registry is missing, stale, wider than requested, unavailable,
malformed, or policy denied.

Flow resume uses the same principle: paused strict flows record skills
metadata, and resume rejects stale, widened, or policy-denied metadata instead
of silently substituting a different registry.

The standalone flow and task CLIs expose `--tool` for strict tool nodes, but
they do not make untrusted flow/task input a skills authority. Provide trusted
skills settings from SDK or host code for flow/task examples that contain
`[skills]`, and start workers with the same trusted settings in production.

### Source Authority, Paths, and Privacy

Source authority is explicit and logical. Supported authority kinds are
`bundled`, `workspace`, `user_local`, `plugin_provided`, and
`preinstalled_remote`. Model-visible output uses source labels, skill IDs,
resource IDs, registry versions, and provenance rather than host paths.

Skill resource reads are path-policy checked: traversal, undeclared resource
IDs, hidden paths unless allowed, resources outside the authorized root,
binary or non-UTF-8 resources, stale resources, and oversized reads are
reported as envelopes with diagnostics. Skill content is treated as
untrusted instruction text and remains subordinate to system, developer, user,
repository, runtime, and tool policy.

Privacy defaults include source labels, authority, and diagnostic paths while
redacting host paths. Operators can narrow exposure with `[tool.skills.privacy]`
or the CLI diagnostics/observability flags. Server projections redact skill
bodies and host-looking paths from remote protocol output.

### Server, MCP, and A2A Boundaries

Serving an agent over OpenAI-compatible HTTP, MCP, or A2A does not let remote
callers define skill sources, widen limits, or add model-facing `skills.*`
tool definitions. Remote request tools named `skills.*` are rejected; only the
served agent's trusted configuration can expose the skills registry.

MCP and A2A callers may trigger the served agent's configured `skills.*`
tools if the served agent enabled them, just as they may trigger other
enabled tools. Treat served skills as a disclosure surface: choose trusted
source roots deliberately, use model-safe labels, and keep diagnostic and
observability output appropriate for the protocol audience.

### Migration Note

Unsupported early skill syntax has no backward compatibility guarantee.
Unknown fields such as `source`, `sources`, or model-facing `skills.load`
must be removed or moved into trusted operator settings. Current supported
skill packages use Markdown `SKILL.md` files with front matter containing at
least `name` and `description`; `tags`, `version`, `enabled`, and `resources`
are supported optional fields.

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

Media tools such as `shell.pdfinfo`, `shell.pdftotext`, `shell.pdftoppm`,
`shell.reportlab`, `shell.pdfplumber`, `shell.pypdf`, and `shell.tesseract`
require `allow_media_tools = true` or the corresponding CLI flag. The Python
PDF tools resolve a trusted Python executable and report `command_unavailable`
when the required package cannot be imported. In container mode, the selected
image must make both `avalan` and the target PDF library importable to that
Python interpreter. Absolute paths, symlinks, hidden files, and executable
search paths are also opt-in.

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
pipeline_transport = "buffered"
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

Pipeline transport defaults to `buffered`, which reads intermediate stdout in
Avalan and writes it to the next structured subprocess stdin. Trusted runtime
configuration may set `pipeline_transport = "native"` to connect adjacent
subprocesses with OS pipes/file descriptors. Native transport still uses the
same structured command specs and per-stage argv handling; it does not lower
pipelines to shell syntax. In native transport, intermediate stdout bypasses
Avalan capture, so intermediate stdout bytes/truncation are reported
unavailable and `max_intermediate_bytes` is not enforced between stages. Final
stdout/stderr caps and per-stage stderr capture still apply.

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
