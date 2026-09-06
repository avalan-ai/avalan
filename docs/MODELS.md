# Models

Avalan runs local and hosted models behind one interface. Use model ids for
open local models and `ai://` engine URIs for vendor-hosted models, custom
endpoints, and authenticated providers.

Models are the lowest-level runtime surface. Agents, flows, and tasks build
on top of the same model layer.

Depending on the model and backend, this layer covers text generation, NLP
tasks, vision, audio, and generated media.

## Run a Hosted Model

```sh
export OPENAI_API_KEY=...

echo "Give me two facts about Leo Messi." \
    | avalan model run "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
        --system "You are Aurora, a helpful assistant" \
        --max-new-tokens 100 \
        --temperature 0.1
```

Engine URIs are documented in [ai_uri.md](ai_uri.md).

## Run an Open Model

Open models can run through backends such as `transformers`, `vllm`, `mlx`,
and DS4 where supported.

```sh
echo "Summarize the role of retrieval in agent systems." \
    | avalan model run meta-llama/Llama-3.2-1B-Instruct \
        --backend transformers \
        --max-new-tokens 200
```

Backend availability depends on the installed extras, platform, model type,
and local hardware.

## Search for Models

Use `avalan model search` to find open models by name, task, author, or
library, and cap results with `--limit`:

```sh
avalan model search --name DeepSeek-R1 \
    --library mlx \
    --task text-generation \
    --author mlx-community \
    --limit 3
```

This is useful when choosing a local model for a specific backend or task.

Install or inspect a model before a local run:

```sh
avalan model display mlx-community/DeepSeek-R1-Distill-Qwen-14B --summary
avalan model install mlx-community/DeepSeek-R1-Distill-Qwen-14B
```

Model cache commands live under `avalan cache`:

```sh
avalan cache list --summary
avalan cache download mlx-community/DeepSeek-R1-Distill-Qwen-14B
```

Cache deletion first prints matching candidates and asks for confirmation.
Use `--delete` when you want to skip the prompt and delete immediately:

```sh
avalan cache delete --model mlx-community/DeepSeek-R1-Distill-Qwen-14B
avalan cache delete --model mlx-community/DeepSeek-R1-Distill-Qwen-14B --delete
```

## Vendor URIs

Vendor models use the `ai://` URI form:

```text
ai://env:OPENAI_API_KEY@openai/gpt-4o
ai://env:ANTHROPIC_API_KEY@anthropic/claude-sonnet-4
ai://env:GOOGLE_API_KEY@google/gemini-2.5-pro
```

The `env:NAME` part reads the access token from an environment variable.
Avalan also supports `secret:NAME` for configured secret backends and direct
tokens when you explicitly want the token in the URI. Prefer `env:` or
`secret:` so API keys do not end up in shell history, process listings, logs,
or checked-in files.

Bare model ids are local/open model references when the first path segment is
not a known vendor name:

```text
meta-llama/Llama-3.2-1B-Instruct
ai://local/meta-llama/Llama-3.2-1B-Instruct
```

If a local model lives under a vendor-like namespace, make locality explicit:

```text
ai://local/openai/gpt-oss-20b
```

Query parameters can carry backend or provider options:

```text
ai://local/meta-llama/Llama-3.2-1B-Instruct?backend=mlx
ai://env:AZURE_OPENAI_API_KEY@openai/my-deployment?azure_api_version=YYYY-MM-DD
```

Avalan also supports custom endpoints for compatible providers. See
[ai_uri.md](ai_uri.md) for full URI syntax.

Use `--base-url` for OpenAI-compatible services, Azure endpoints, local
gateways, or an Avalan-served agent:

```sh
echo "Summarize this." \
    | avalan model run "ai://openai" \
        --base-url "http://127.0.0.1:9001/v1"
```

## Generation Settings

Common runtime settings include:

- `max_new_tokens`
- `temperature`
- `top_p`
- `top_k`
- Streaming on or off.
- System prompt.
- Tool and structured-output settings when used through agents.
- Reasoning settings for models or providers that expose them.

From the CLI:

```sh
echo "Write a concise release note." \
    | avalan model run "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
        --system "You write terse release notes." \
        --max-new-tokens 180 \
        --temperature 0.2 \
        --top-p 0.9
```

From Python:

```python
import asyncio
from os import environ

from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.text.vendor.openai import OpenAIModel


async def main() -> None:
    settings = TransformerEngineSettings(access_token=environ["OPENAI_API_KEY"])

    with OpenAIModel("gpt-4o", settings) as model:
        response = await model(
            "Give me two facts about Leo Messi.",
            settings=GenerationSettings(use_async_generator=False),
        )
        print(await response.to_str())


asyncio.run(main())
```

## Streaming

Models can stream canonical runtime items. Streaming is useful for chat UIs,
agent progress displays, and long-running responses. Continuing from the
Python setup above:

```python
from avalan.model.stream import CanonicalStreamItem, StreamItemKind

async for item in await model(
    "Give me two more facts about Leo Messi.",
    settings=GenerationSettings(use_async_generator=True),
):
    assert isinstance(item, CanonicalStreamItem)
    if item.kind is StreamItemKind.ANSWER_DELTA and item.text_delta is not None:
        print(item.text_delta, end="", flush=True)
```

## Reasoning Controls

Reasoning-capable models may expose native reasoning text, provider-generated
reasoning summaries, or reasoning-effort settings. These are separate from the
assistant answer:

- **Native reasoning text** is reasoning text a model or provider explicitly
  emits. Avalan marks it `native_text` on the private reasoning channel.
- A **provider reasoning summary** is a provider-generated summary of its
  internal reasoning. Avalan marks it `summary`; it is not raw chain-of-thought
  or raw OpenAI reasoning tokens.
- **Encrypted reasoning** is opaque replay data used for `store=false` tool
  continuations. Avalan neither decodes nor displays it.
- **Assistant commentary** is not a reasoning-summary feature and is not
  synthesized by Avalan. Tool calls, results, and any tool narration remain
  separate tool lifecycle data.

Requesting reasoning and displaying reasoning are independent controls.
`summary` asks a capable provider to generate a summary. `--display-reasoning`
only opts into rendering private reasoning already present in the stream; it
does not change the request. Reasoning stays hidden by default in both Basic
and Fancy themes, and `--stats` alone never reveals it. `--quiet` has higher
output precedence: it suppresses diagnostics and recording, including an
explicit `--display-reasoning`, and leaves answer-only stdout. It does not
cancel a requested provider summary.

### SDK

The supported summary modes are `auto`, `concise`, and `detailed`:

```python
from avalan.entities import (
    GenerationSettings,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningSummaryMode,
)

generation_settings = GenerationSettings(
    use_async_generator=True,
    reasoning=ReasoningSettings(
        effort=ReasoningEffort.LOW,
        summary=ReasoningSummaryMode.CONCISE,
    ),
)
```

Only adapters that explicitly advertise a requested mode can accept it.
OpenAI and Azure OpenAI use the Responses API path. Other providers remain
unchanged and reject an explicit unsupported summary before dispatch rather
than silently dropping it or substituting native reasoning.

Streaming SDK consumers receive summary text as private
`REASONING_DELTA` items whose `reasoning_representation` is `SUMMARY`.
`ANSWER_DELTA` remains answer-only. Multipart summaries preserve their typed
segment identity and order; a provider part finishing does not close the whole
canonical reasoning channel. Non-streaming calls preserve the same structured
reasoning semantics while `to_str()` remains the final answer compatibility
view.

### Agent TOML

Put request controls under `[run.reasoning]`:

```toml
[run.reasoning]
effort = "low"
summary = "concise"
```

Omitting `summary` preserves the prior request shape. A summary cannot be
requested when reasoning is disabled.

### CLI

For a direct model run:

```sh
printf '%s\n' 'Explain why the sky appears blue.' \
  | avalan model run "ai://env:OPENAI_API_KEY@openai/gpt-5-mini" \
      --reasoning-effort low \
      --reasoning-summary concise \
      --display-reasoning
```

For an agent run, `--reasoning-summary` and `--run-reasoning-summary` are
aliases:

```sh
printf '%s\n' 'Draft a short answer.' \
  | avalan agent run docs/examples/agent_support_reply.toml \
      --run-reasoning-summary concise \
      --display-reasoning
```

The model and agent flags accept exactly `auto`, `concise`, or `detailed`.
Without `--display-reasoning`, the request still asks for the summary but the
summary is not rendered. In non-interactive runs, displayed reasoning goes to
stderr so stdout remains answer-only and suitable for strict JSON or shell
pipelines. `--stats` and `--record` never enable reasoning display on their
own. `--quiet` goes further: it overrides even explicit display flags,
suppresses diagnostics and recording, and emits only the answer on stdout.

### OpenAI-compatible Responses request

An Avalan `/v1/responses` request uses the same typed shape:

```json
{
  "model": "gpt-5-mini",
  "input": "Explain why the sky appears blue.",
  "reasoning": {
    "effort": "low",
    "summary": "concise"
  },
  "stream": true
}
```

Streaming output uses `response.reasoning_summary_*` events for summaries and
`response.output_text.*` for the answer. Non-streaming output keeps reasoning
items distinct from the answer message. Invalid modes are request-validation
errors; unsupported providers return an actionable
`reasoning_summary_unsupported` error with no provider call or partial output.

### Privacy and usage

Reasoning summaries are private reasoning. They do not enter final assistant
messages, answer text, tool arguments, agent memory, generic telemetry, or
task output. Protocol projections retain only their documented bounded and
redacted reasoning views. When reasoning display and recording are explicitly
enabled outside quiet mode, terminal recordings capture the rendered private
diagnostic just as it appears on screen. Quiet mode suppresses that diagnostic
and recording entirely, leaving answer-only stdout.

Provider `reasoning_tokens` usage is a token-accounting counter. It is not a
count of visible summary chunks, parts, characters, or canonical deltas.

Native/local reasoning controls remain available independently:

```sh
echo "What is (4 + 6) * 5 / 2?" \
    | avalan model run "ai://local/openai/gpt-oss-20b" \
        --backend mlx \
        --max-new-tokens 1024 \
        --display-reasoning
```

- `--start-thinking` for models that begin inside a reasoning block.
- `--reasoning-tag think|channel` to select the parser style.
- `--reasoning-effort none|minimal|low|medium|high|xhigh|max` where the
  provider supports it.
- `--reasoning-summary auto|concise|detailed` to request a provider summary
  where explicitly supported.
- `--reasoning-max-new-tokens` and `--reasoning-stop-on-max-new-tokens` to
  bound long reasoning output.
- `--no-reasoning` to disable reasoning parsing.

See [TOOLS.md](TOOLS.md) for ReACT, tool formats, and tool-calling loops.

## Provider-native Inline Compaction for Agent Runs

Inline compaction is an opt-in, provider-managed context boundary for the
ordinary legacy `avalan agent run` tool loop. Omitting it, or explicitly
selecting `none`, preserves the prior request behavior: provider failures stay
visible, Avalan does not create a local summary, and it does not automatically
call a standalone compaction operation.

The feature is accepted only for the exact native OpenAI Responses transport
at `https://api.openai.com/v1`, or a tightly scoped Azure Responses transport.
Azure requires HTTPS, a host ending in `.openai.azure.com` or
`.cognitiveservices.azure.com`, and `/openai/v1` after trailing-slash
normalization. It permits only the default or `443` port, no userinfo, params,
query, or fragment, and `azure_api_version` omitted or exactly `preview`.
Other providers, compatible gateways, custom base URLs, and transport shapes
fail closed before dispatch. This transport rule does not establish live
capability for a particular model or deployment.

In agent TOML, configure one operation under `[run.compaction]`:

```toml
[run.compaction]
operation = "inline"
compact_threshold = 1024
```

`operation` is `inline` or `none`. `compact_threshold` is required only for
`inline` and must be a positive integer; `none` accepts no threshold.

From the CLI, set both opt-in controls for inline operation:

```sh
avalan agent run agent.toml \
    --run-compaction inline \
    --run-compact-threshold 1024
```

Use `--run-compaction none` to explicitly disable it. From the public SDK,
pass the typed policy at the agent-run boundary:

```python
from avalan.sdk import InlineCompaction, run_agent

result = await run_agent(
    orchestrator,
    agent_input,
    compaction=InlineCompaction(compact_threshold=1024),
)
```

The provider owns the compacted state. Avalan treats its encrypted compaction
item as opaque: never inspect, decode, print, or log it. After a successful
provider response, that state remains part of the provider continuation across
tool cycles; it is not a locally generated replacement for the transcript.

### Lifecycle and observability

Requesting `context_management` is only eligibility; it is not proof that a
compaction happened. Avalan emits `inline_compaction.started` only after the
native OpenAI or Azure Responses stream adds a compaction item. The basic live
display then shows `Compacting...` with an elapsed timer. It emits
`inline_compaction.committed` and shows `Compacted in ...` only after the
opaque item, completed response, and whole-response replay state have all
validated and committed. A candidate that cannot commit emits a content-free
`inline_compaction.rolled_back` outcome instead.

A completed response with no provider boundary remains silent in the normal
basic UI. It is still available to event listeners and metrics as
`inline_compaction.completed_no_boundary`. The lifecycle payloads contain only
the configured threshold, counts, and elapsed duration. Event listeners receive
those numeric values under `inline_compaction`, in addition to the ordinary
stream observability fields. The lifecycle contract rejects any extra payload
keys, provider correlations, opaque provider payload, encrypted compaction
content, prompts, raw provider items, or provider IDs.

An iterator ending without a validated provider terminal is not a successful
response when inline compaction is enabled. Avalan fails that response closed,
rolls back any candidate boundary, and preserves final usage before the error
terminal when safely received. Diagnostics count one logical request and every
actual provider dispatch separately; a retry therefore increments
`attempt_count` without increasing `request_count`.

### Replay resource limits

Direct OpenAI/Azure Responses tool loops retain opaque replay in memory only.
`StreamRetentionPolicy.openai_replay_serialized_byte_limit` bounds one complete
replay suffix (default 32 MiB). A separate
`openai_replay_client_serialized_byte_limit` bounds all live suffixes and
rollback checkpoints owned by one `OpenAIClient` (default 128 MiB). Both accept
nonnegative integer byte counts; zero forbids retention. A configured smaller
suffix limit remains a hard limit, including for a single compaction item.
These controls are independent of output tokens and `compact_threshold`.

The 32 MiB default accommodates a measured 22.49 MiB multimodal compaction
item with approximately 9.5 MiB left for subsequent replay items. It is an
operational default, not a provider payload-size guarantee. The aggregate
default accommodates two executions each holding a maximum-size committed
suffix and a maximum-size replacement. More concurrent executions share that
quota; admission fails without evicting another execution or waiting/retrying.
Applications should budget concurrency and client count together. Distinct
clients have independent quotas.

Admission checks the replacement before discarding any superseded candidate.
The last committed suffix remains available for rollback until response
validation succeeds. Providers can rotate encrypted compaction content between
`response.output_item.done` and `response.completed`. After validating every
terminal identity, Avalan replaces the candidate with the completed response's
authoritative item and rechecks both quotas, including any subsequent replay
items. This replacement does not count as another compaction boundary.

Local admission does not guarantee provider acceptance. In fixture validation,
Azure returned a terminal compaction string with 23,587,532 characters, then
rejected its unchanged replay with a 10,485,760-character input limit. Replaying
the terminal item correctly does not resolve that provider constraint. Avalan
surfaces the non-retryable rejection; it cannot shorten opaque state or omit
required images to bypass the limit.

Repeated compactions retain only the latest candidate
plus that checkpoint; commit and rollback release the unused generation.
Cancellation and stream failure roll back before releasing the execution.
Per-response compaction identities are bounded by `openai_replay_item_limit`
and retain hashes, not opaque payloads. JSON size accounting and compaction
hashing use 16,384-character chunks to bound temporary encoding allocations.

These are serialized replay storage quotas, **not process RSS limits**. Python
object overhead, caller inputs/images, provider SDK response buffers and HTTP
request serialization have their own memory costs; the SDK materializes an
incoming item before Avalan can reject it. No opaque data are spilled to disk,
truncated, or replaced with a local summary. Durable conversation-ledger
retention is a separate subsystem.

Configure the direct client explicitly when tighter bounds are required:

```python
from avalan.model.nlp.text.vendor.openai import OpenAIClient
from avalan.model.stream import StreamRetentionPolicy

client = OpenAIClient(
    api_key=api_key,
    base_url=base_url,
    stream_retention_policy=StreamRetentionPolicy(
        openai_replay_serialized_byte_limit=32 * 1024 * 1024,
        openai_replay_client_serialized_byte_limit=128 * 1024 * 1024,
    ),
)
```

`client.replay_retention_diagnostics` exposes current, peak, and allowed
serialized bytes, including rollback storage. It contains no provider IDs or
contents. Stream errors preserve safe classifications through private replay:

| Code | Meaning |
| --- | --- |
| `reasoning_replay_retention_exceeded` | Local capacity exhaustion; includes `resource`, `observed`, `allowed`, and `retryable=false`. |
| `reasoning_replay_invalid` | Invalid local replay structure; non-retryable. |
| `inline_compaction_protocol_invalid` | Invalid provider compaction protocol; non-retryable. |
| `openai_provider_request_failed` | Provider failure; consult safe `provider_failure` retry diagnostics. |
| `string_above_max_length` | Provider HTTP 400 input-size rejection; non-retryable, with no echoed input or raw provider message. |
| `openai_request_cancelled` | Cancellation, with the cancelled terminal outcome. |

An unchanged local capacity error must not be retried as an upstream server
failure. Applications can adjust their explicit resource budget before a new
attempt, or surface the error. Successful compaction on a final response alone
does not prove continuation: verify that a subsequent accepted request carries
the committed item and completes its next task step. See the official
[server-side compaction contract](https://developers.openai.com/api/docs/guides/compaction).

## Modalities

Avalan supports text, vision, and audio workloads. Model choice and backend
support vary by modality.

Use [MODALITIES.md](MODALITIES.md) for examples covering:

- Text generation.
- Vision inputs.
- Audio transcription or processing.
- Multimodal CLI and Python calls.

Attach supported files with `--input-file`:

```sh
echo "Describe the attached invoice." \
    | avalan model run "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
        --input-file docs/examples/playground/invoice.pdf
```

For repeatable file workflows with limits, conversion, storage, and output
schemas, wrap an agent or flow that uses the model in a task. See
[TASKS.md](TASKS.md) and [task_file_delivery.md](task_file_delivery.md).

## DS4

Avalan includes native support for DS4-supported DeepSeek V4 Flash GGUFs.
This backend has specific installation, model, and environment requirements.

See [DS4.md](DS4.md) for the DS4 setup guide.

## Backend Notes

| Backend | Use | Notes |
| --- | --- | --- |
| `transformers` | Broad local model and modality support. | Default local backend. |
| `mlx` | Apple Silicon text generation. | Best for MLX-compatible model families on macOS arm64. |
| `vllm` | High-throughput server-class NVIDIA GPU deployments. | Install vLLM in the runtime environment. |
| `ds4` | DS4-supported DeepSeek V4 Flash GGUFs. | Not a generic GGUF loader; see [DS4.md](DS4.md). |

- `--trust-remote-code` executes model repository code; enable it only for
  models and revisions you trust.
- Keep provider keys in environment variables and use `ai://env:KEY@...`
  rather than hard-coding secrets in scripts or docs.

## Choosing the Right Layer

- Use `avalan model run` for direct model calls and backend testing.
- Use an agent when the model needs tools, memory, instructions, or serving.
- Use a flow when several steps must be orchestrated.
- Use a task when callers need a durable contract around the work.

## Related Documentation

- [ai_uri.md](ai_uri.md) - Engine URI syntax.
- [INSTALL.md](INSTALL.md) - Extras and backend dependencies.
- [MODALITIES.md](MODALITIES.md) - Text, vision, and audio examples.
- [AGENT_GUIDE.md](AGENT_GUIDE.md) - Model use inside agents.
- [FLOWS.md](FLOWS.md) - Model and agent nodes in flows.
- [TASKS.md](TASKS.md) - Agent-backed and flow-backed tasks that use models.
