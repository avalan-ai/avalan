# Installation

## MacOS

Install avalan using [Homebrew](https://brew.sh):

```bash
brew install avalan
```

## Ubuntu

Update package index and install python prerequisites:

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

## DS4 native backend

Avalan's `ds4` extra installs the `pyds4` bridge used by the native DS4
backend:

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
