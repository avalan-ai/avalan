# Installation

Avalan supports Python 3.11 through 3.14.

## Installation options

Pip extras are additive, so install the smallest capability set you need.
Homebrew and Ubuntu packages install the same dependency profile as
`avalan[agent,server,tool,vendors]`.

```bash
python3 -m pip install -U avalan
python3 -m pip install -U "avalan[agent,server,tool,vendors]"
python3 -m pip install -U "avalan[agent,task,task-pdf-images,vendors]"
```

The first command is the lean SDK and CLI install. The second is the common
agent, serving, tool, and hosted-vendor profile used by the main examples. The
third adds task contracts and PDF-to-image conversion for flow-backed task
examples.

Add hardware or backend extras when needed:

- `mlx` or `apple` - Apple Silicon acceleration via MLX / MLX-LM.
- `nvidia` - Linux and NVIDIA quantization support.
- `vllm` - reserved for the vLLM backend.
- `quantization` - 4-bit and 8-bit model loading.
- `ds4` - native DS4 inference for DS4-supported DeepSeek V4 Flash GGUFs
  through [pyds4](https://github.com/avalan-ai/pyds4). Production targets are
  macOS arm64 with Metal and Linux with CUDA; CPU mode is only a
  debug/reference path.

Task-specific extras include `task` for structured validation,
`task-pdf-images` for PDF rasterization, `task-documents` for document
conversion, `task-pgsql` for durable stores and workers, `task-prometheus` for
metrics, and `task-otel` for traces.

> [!NOTE]
> The `vllm` extra is intentionally empty while vLLM depends on vulnerable
> `diskcache` releases without an upstream fix. Install vLLM separately if you
> accept that dependency. `markitdown` document conversion in the `memory`
> extra is currently limited to Python 3.11 through 3.13 by upstream
> dependencies. The pinned `torchvision` release used by the `vision` extra
> also excludes Python 3.14.1, and the `memory` extra omits `psycopg-binary`
> on Python 3.14 until compatible wheels are published.

### macOS

Install avalan using [Homebrew](https://brew.sh):

```bash
brew install avalan-ai/avalan/avalan
```

### Ubuntu

Install avalan from the PPA:

```bash
sudo add-apt-repository -y ppa:avalan-ai/avalan
sudo apt update
sudo apt install -y avalan
```

To use Poetry in a local Ubuntu test project, install Python prerequisites:

```bash
sudo apt update -y
sudo apt install -y python3 python3-venv python3-dev python3-pip curl
```

Install [poetry](https://python-poetry.org) and add it to `$PATH`:

```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Start a new poetry project, specify the python version, and add avalan with
`all` extras:

```bash
mkdir avalan-test/ && cd avalan-test/
poetry init --no-interaction --python=">=3.11,<3.15"
poetry add "avalan[all]" --no-cache
```

### DS4 native backend

DS4 is available as a local text-generation backend for DS4-supported
DeepSeek V4 Flash GGUF files via [pyds4](https://github.com/avalan-ai/pyds4).
It is not a generic GGUF backend. DS4 opens the GGUF directly from the
filesystem and does not require `HF_TOKEN`.

Avalan's `ds4` extra installs the `pyds4` bridge used by the native backend:

```bash
python3 -m pip install -U "avalan[ds4]"
```

The extra is scoped to platforms that can run DS4's native targets: macOS
arm64 with Metal and Linux with CUDA. The DS4 CPU backend is a
debug/reference path for correctness checks only and should not be treated as
a production inference target.

The DS4 backend is intentionally model-specific. It only supports
DS4-supported DeepSeek V4 Flash GGUF files and is not a generic GGUF loader.

If a published `pyds4` wheel is not available for your platform, build the
binding from a local `pyds4` source checkout before installing Avalan:

```bash
git clone https://github.com/antirez/ds4.git /path/to/ds4

DS4_SOURCE_DIR=/path/to/ds4 \
PYDS4_BACKEND=metal \
python3 -m pip install -e /path/to/pyds4

python3 -m pip install -U "avalan[ds4]"
```

Use `PYDS4_BACKEND=cuda` for Linux CUDA builds. Local CPU builds are possible
with `PYDS4_BACKEND=cpu`, but keep them limited to diagnostics.

Use `--ds4-native-backend cuda` on Linux CUDA builds. CPU mode is only a
debug/reference path. Use `ai://local//absolute/path.gguf` for absolute paths,
or a normal `ai://local/relative/path.gguf` URI for paths relative to the
current directory.

```bash
export DS4_MODEL=/path/to/ds4flash.gguf

printf '%s\n' 'Write a short greeting.' \
    | avalan model run "ai://local/${DS4_MODEL}?backend=ds4" \
        --ds4-ctx 4096 \
        --ds4-native-backend metal \
        --max-new-tokens 64 \
        --temperature 0
```

See [DS4.md](DS4.md) for CLI examples and current limitations.

#### DS4 tool use

DS4-backed agents can use normal Avalan tools. Basic output shows the answer
text and hides DSML/protocol/tool-call markup from the final answer while
preserving tool results; with `--display-tools`, it also shows tool lifecycle
details and results:

```bash
printf '%s\n' 'What is (4 + 6) and then that result times 5, divided by 2?' \
  | avalan agent run \
      --engine-uri "ai://local/${DS4_MODEL}" \
      --backend ds4 \
      --ds4-ctx 4096 \
      --ds4-native-backend metal \
      --tool "math.calculator" \
      --memory-recent \
      --run-max-new-tokens 8192 \
      --run-temperature 0 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats \
      --display-events \
      --display-tools
```

Internally, native DS4 tool calls use DSML: Avalan renders tool schemas,
parses completed DSML tool blocks, streams argument deltas, and preserves
exact raw DSML replay metadata for session alignment.

### Source build tips

On macOS, ensure the Xcode command line tools are present and install the build
dependencies before compiling extras that rely on `sentencepiece`:

```bash
xcode-select --install
brew install cmake pkg-config protobuf sentencepiece
```

## Task PostgreSQL migrations

Durable task storage uses PostgreSQL through the `task-pgsql` extra:

```bash
python3 -m pip install -U "avalan[task-pgsql]"
```

Task schema migration commands additionally require Alembic and SQLAlchemy in
the environment that runs them:

```bash
python3 -m pip install -U alembic "SQLAlchemy>=2.0.43,<3.0.0"
```

Set `AVALAN_TASK_PGSQL_DSN` before running migration diagnostics, and set
`AVALAN_TASK_PGSQL_SCHEMA` when using an isolated schema.

Set `AVALAN_TASK_TEST_POSTGRESQL_DSN` to run the env-gated PostgreSQL
verification tests against an existing test database. Run `make test-pgsql` to
start a throwaway PostgreSQL container with Docker, create and drop a temporary
test database, and run the full suite including migration and queue e2e tests.
Set `AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN` before `make test` to run the same
temporary database flow against an existing PostgreSQL server. Set
`AVALAN_TASK_BENCHMARK_POSTGRESQL_DSN` only when running the opt-in EXPLAIN
benchmark checks.

For durable queue deployments, also configure `AVALAN_TASK_HMAC_KEY_ID`,
`AVALAN_TASK_HMAC_KEY_B64`, and `AVALAN_TASK_ARTIFACT_ROOT`, then run workers
under a process supervisor or container restart policy. See
[Task queue operations](TASK_OPERATIONS.md) for deployment profiles and
failure-mode runbooks.

## Container execution

Container-backed execution is disabled by default and requires trusted
operator configuration. Current release readiness centers on the core
container contract, shell container policy, fake-backend tests, and injected
runtime backends; real Docker and Apple `container` jobs are optional
environment-gated checks. See
[Container execution](CONTAINERS.md) for supported scope, runtime setup,
test gates, platform limits, fail-closed behavior, diagnostics, and known
deferred conformance.
