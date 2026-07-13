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
