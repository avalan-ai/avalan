# Agent Instructions

This repository contains the **avalan** framework: a multi-backend,
multi-modal micro-framework for building, orchestrating, and deploying AI
agents. It ships as a Python SDK and a CLI (`avl` / `avalan`), bridging
local and hosted models behind a single runtime. Dependency management uses
[Poetry](https://python-poetry.org/); testing uses `pytest`.

## Codebase tour

Source lives under `src/avalan/`. Tests under `tests/` mirror this layout,
with files named `*_test.py`.

- `agent/` — Agent definition and execution. Agents are TOML + Jinja2
  templates loaded by `AgentLoader` (`agent/loader.py`) and run by
  `EngineAgent` (`agent/engine.py`); multi-agent coordination lives in
  `agent/orchestrator/`. The reference template is
  `agent/templates/blueprint.toml`.
- `model/` — Model abstraction across vendors and backends. Vendor URIs
  follow `ai://env:KEY@vendor/model-id`. Backends include `transformers`,
  vLLM, MLX-LM, and native DS4. See `model/manager.py` and `model/vendor.py`.
- `tool/` — Tool system. `Tool` (ABC, async) and `ToolSet` live in
  `tool/__init__.py`; tools generate JSON schemas automatically and are
  loaded by `tool/manager.py`. Built-ins: browser, code, databases, graph,
  MCP, memory, search, YouTube.
- `memory/` — `MemoryStore[T]` with transient (`RecentMessageMemory`) and
  persistent (`memory/permanent/`: PostgreSQL, S3, FAISS, Elasticsearch)
  backends; chunking strategies in `memory/partitioner/`.
- `flow/` — DAG-based orchestration (`flow.py`, `parser.py`, `node.py`)
  with per-domain node types (browser, code, db, MCP, etc.).
- `event/` — Agent lifecycle events (`Event`, `EventType`).
- `server/` — FastAPI app exposing OpenAI-compatible endpoints, MCP, and
  the A2A protocol (`server/a2a/`).
- `backends/` — Native inference backends (e.g., `backends/ds4_native/`).
- `cli/` — Entry points `avl` and `avalan` resolve to
  `cli/__main__.py:main`; subcommands live under `cli/commands/`
  (`agent`, `model`, `memory`, `tool`, `tokenizer`, `deploy`, …).
- `deploy/`, `secrets/`, `logic/`, `compat.py`, `entities.py`, `filters.py`,
  `utils.py` — supporting modules. `entities.py` holds the shared data
  models (`Message`, `ToolCall`, `GenerationSettings`, …).

Additional documentation: `docs/CLI.md`, `docs/DS4.md`, `docs/INSTALL.md`,
`docs/ai_uri.md`, plus `docs/examples/` and `docs/tutorials/`.

## Formatting & style

- 4-space indentation (enforced by `.editorconfig`).
- Target Python 3.11; avalan supports 3.11–3.13. Do not use
  `from __future__ import annotations`.
- Strict type hints throughout. Prefer `type | None` over `Optional[type]`.
- `from X import Y` imports only — no inline imports. Sort by package name
  alphabetically; relative imports at the top, also sorted alphabetically.
- Use assertions to validate arguments.
- Do not ignore exceptions unless instructed.
- Do not declare `__all__` lists.
- Avoid code duplication.

### PEP standards

1. **PEP 8** — `snake_case` for modules/functions/variables, `PascalCase`
   for classes, `UPPER_SNAKE` for constants; single space around binary
   operators; no extra spaces inside brackets.
2. **PEP 257** — triple-quoted docstrings immediately under
   `def`/`class`/`module`; imperative voice ("Return", not "Returns");
   wrap at ~72 chars; use `Args:` / `Returns:` / `Raises:` sections.
3. **PEP 585** — built-in generics: `list[int]`, `dict[str, float]`,
   `tuple[int, ...]`. Don't import `List` / `Dict` / `Tuple` from `typing`.
4. **PEP 604** — union syntax: `str | float`, not `Union[str, float]`.
5. **PEP 634/635/636** — use structural pattern matching (`match`/`case`)
   where appropriate.

Run `make lint` before committing — it runs `ruff format`, `black`,
`ruff check --fix`, and `mypy` over `src/` and `tests/`:

```bash
make lint
```

## Tool docstrings

All `Tool` subclasses must include descriptive docstrings (agents rely on
these when selecting tools):

- Imperative summary sentence ending with a period.
- Blank line, then `Args:` and `Returns:` sections covering every
  user-provided parameter and the return value. **Do not document the
  implicit `context` parameter.**
- Update `tests/tool/tool_docstring_format_test.py` when introducing new
  tools (it validates format and the absence of `context`).

## Testing & coverage

Aim for **100% test coverage**. Add unit tests alongside any code change,
including negative cases and varied parameter values. Tests mirror the
source layout under `tests/` and use the `*_test.py` naming convention.

Run the full suite before committing:

```bash
poetry run pytest --verbose -s
```

Use `make test-coverage` to audit:

```bash
make test-coverage                          # report all files in src/
make test-coverage -- -95                   # files below 95%
make test-coverage -- -95 src/avalan/tool   # below 95% under that path
```

Do not fake coverage with `exec` / `compile` tricks.

## Make targets

- `make install` — `poetry sync --all-extras`.
- `make lint` — format and type-check (`ruff`, `black`, `mypy`).
- `make test` / `make tests` — run pytest with verbose output.
- `make test-coverage [-- THRESHOLD [PATH]]` — coverage report; a negative
  threshold filters to files below it.
- `make version X.Y.Z` — bump `pyproject.toml` and push `release/vX.Y.Z`.
- `make release X.Y.Z` — tag, publish to PyPI, and cut a GitHub release.

## Submitting changes

Tests must pass before every commit.

### Commit messages

- Short and descriptive (e.g. `Fix memory tests`).
- Do not amend or rewrite previous commits.

### Pull request body

Include two sections:

1. **Summary** — overview of what changed.
2. **Testing** — commands executed and their output. If tests could not be
   run (missing deps, network restrictions, etc.), say so.
