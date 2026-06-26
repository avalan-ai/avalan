<h1 align="center">avalan</h1>
<h3 align="center">The multi-backend, multi-modal micro-framework for AI agent development, orchestration, and deployment</h3>

<p align="center">
  <a href="https://pypi.org/project/avalan/"><img src="https://img.shields.io/pypi/pyversions/avalan.svg" alt="Python versions" /></a>
  <a href="https://pypi.org/project/avalan/"><img src="https://img.shields.io/github/v/release/avalan-ai/avalan?label=Release" alt="Release" /></a>
  <a href="https://github.com/avalan-ai/avalan/actions/workflows/test.yml"><img src="https://github.com/avalan-ai/avalan/actions/workflows/test.yml/badge.svg" alt="Tests" /></a>
  <a href="https://coveralls.io/github/avalan-ai/avalan"><img src="https://coveralls.io/repos/github/avalan-ai/avalan/badge.svg" alt="Code test coverage" /></a>
  <img src="https://img.shields.io/github/last-commit/avalan-ai/avalan.svg" alt="Last commit" />
  <a href="https://github.com/avalan-ai/avalan/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/avalan.svg" alt="License" /></a>
  <a href="https://discord.gg/8Eh9TNvk"><img src="https://img.shields.io/badge/discord-community-blue" alt="Discord Community" /></a>
  <a href="https://codewiki.google/github.com/avalan-ai/avalan"><img src="https://img.shields.io/badge/Google-Code%20Wiki-4285F4?logo=google&amp;logoColor=white" alt="Google Code Wiki" /></a>
</p>

Avalan is a Python SDK and CLI for building your own intelligence harness
across local and hosted models. Prompts, model routing, tools, memory,
workflows, tasks, and serving protocols stay explicit, so the model is a
component you control rather than the whole system.

## 🧠 Build Your Intelligence Harness

Avalan is for teams that want the model to be one part of the system, not the
system itself. It gives you the runtime harness around inference, tools,
memory, orchestration, tasks, protocols, and human control.

With Avalan, you can:

- 💻 Run **local and hosted models** behind one interface, with inference-time
  control over prompts, sampling, streaming, backends, and vendor routing.
- 🛠️ **Execute tools** as first-class runtime steps, including browser automation,
  code, databases, shell media tools, MCP servers, and A2A agents.
- 🔁 **Compose flows** where each node, edge, input, output, and review boundary is
  visible.
- ✅ Package repeatable work as **tasks with typed inputs**, file handling, queues,
  durable execution, privacy settings, and schema-validated outputs.
- 🌐 **Serve your agents** through OpenAI-compatible APIs, MCP, and A2A without
  changing the agent runtime.
- 🔒 Add **human-in-the-loop** and least-privilege controls around tool execution,
  data access, files, and networked agent calls.

## 📦 Install

Avalan supports Python 3.11 through 3.14.

### 🍺 Homebrew (macOS)

```sh
brew install avalan-ai/avalan/avalan
```

### 🐧 Ubuntu (PPA)

```sh
sudo add-apt-repository -y ppa:avalan-ai/avalan
sudo apt update
sudo apt install -y avalan
```

### 🐍 Pip

```sh
python3 -m pip install -U "avalan[agent,server,tool,vendors]"
```

### 🛠️ From Source with Poetry

```sh
poetry install --extras "agent server tool vendors" --with test
```

See [docs/INSTALL.md](docs/INSTALL.md) for pip extras, backend-specific
dependencies, DS4 setup, and source build prerequisites.

## ⚡ Quickstart

### 💬 Call a hosted model from the CLI

Export a vendor key, then run:

```sh
export OPENAI_API_KEY=...
echo "Who are you, and who is Leo Messi?" \
    | avalan model run "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
        --system "You are Aurora, a helpful assistant" \
        --max-new-tokens 100
```

### 🐍 Use the Python SDK

```python
from asyncio import run
from os import environ
from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.text.vendor.openai import OpenAIModel
from avalan.model.stream import StreamItemKind

async def main() -> None:
    api_key = environ["OPENAI_API_KEY"]
    settings = TransformerEngineSettings(access_token=api_key)

    with OpenAIModel("gpt-5-mini", settings) as model:
        async for item in await model(
            "Give me five facts about Leo Messi.",
            settings=GenerationSettings(use_async_generator=True),
        ):
            if (
                item.kind is StreamItemKind.ANSWER_DELTA
                and item.text_delta is not None
            ):
                print(item.text_delta, end="", flush=True)

run(main())
```

### ✅ Run a flow-backed task

This example requires task, PDF image conversion, agent, and vendor
dependencies:

```sh
python3 -m pip install -U "avalan[agent,task,task-pdf-images,vendors]"
```

```sh
avalan task run docs/examples/tasks/poc_extraction/image_flow_task.toml \
    --ephemeral \
    --pdf docs/examples/tasks/poc_extraction/sample.pdf \
    --json \
    --output image.json
```

For task contracts, flow graphs, file conversion, queues, and schema output,
see [docs/TASKS.md](docs/TASKS.md), [docs/FLOWS.md](docs/FLOWS.md), and
[docs/examples/tasks/poc_extraction/README.md](docs/examples/tasks/poc_extraction/README.md).

## Documentation & Resources

- [docs/README.md](docs/README.md) - Complete documentation index.
- [docs/INSTALL.md](docs/INSTALL.md) - Platform setup, extras, backends, and
  source builds.
- [docs/MODELS.md](docs/MODELS.md) - Model URIs, local and hosted backends,
  generation settings, and streaming.
- [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) - Agent TOML, prompts, tools, memory,
  structured output, multimodal inputs, and serving.
- [docs/TOOLS.md](docs/TOOLS.md) - Built-in tools, MCP, A2A, shell tools,
  confirmation, safety controls, and reasoning strategies.
- [docs/MEMORIES.md](docs/MEMORIES.md) - Recent and permanent memory,
  namespaces, indexing, retrieval, and storage backends.
- [docs/FLOWS.md](docs/FLOWS.md) - Explicit graph workflows, strict
  definitions, Mermaid views, branching, and review boundaries.
- [docs/TASKS.md](docs/TASKS.md) - Durable task contracts, file delivery,
  queues, storage, privacy, and schema validation.
- [docs/MODALITIES.md](docs/MODALITIES.md) - Text, vision, and audio CLI and
  Python examples.
- [docs/CLI.md](docs/CLI.md) - Command and flag reference.
- [docs/examples](docs/examples/README.md) - Runnable examples.

## Community & Support

- Join the [Avalan Discord](https://discord.gg/8Eh9TNvk) to ask questions,
  share workflows, and follow release announcements.
- Browse generated documentation or ask follow-up questions in
  [Google Code Wiki](https://codewiki.google/github.com/avalan-ai/avalan).
- For commercial support, email
  [avalan@avalan.ai](mailto:avalan@avalan.ai).

## Contributing

We welcome pull requests, issue reports, docs improvements, and new examples.

1. Read the [Code of Conduct](CODE_OF_CONDUCT.md) before you start.
2. Install the development environment with
   `poetry install --all-extras --with test`.
3. Run `make lint`.
4. Run `poetry run pytest --verbose -s`.

Open a [GitHub issue](https://github.com/avalan-ai/avalan/issues) if you
discover bugs or want to propose larger changes.
