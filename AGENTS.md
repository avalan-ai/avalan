# Codex Agent Instructions for avalan

This repository contains the **avalan** framework, a Python project that orchestrates multiple models and provides a CLI for AI agents.  The repository uses [Poetry](https://python-poetry.org/) for dependency management and `pytest` for testing.

## Formatting & Style

- Python files use 4 spaces per indentation as enforced by `.editorconfig`.
- Adhere to standard PEP8 style.  Type hints and docstrings are encouraged throughout the codebase.
- Avoid code duplication wherever possible.
- Code must target Python **3.11** or newer with fully strict type hints.
- Prefer `type | None` instead of `Optional[type]` for optional values.
- Use assertions to ensure argument validity.

Run `make lint` to perform syntax checks and formatting via
[ruff](https://docs.astral.sh/ruff/):

```bash
make lint
```

## Running Tests

Run the full test suite before every commit and add unit test coverage for all code additions or modifications:

```bash
poetry run pytest --verbose -s
```

Tests must pass before you commit.

## Commit Messages

- Keep commit messages short and descriptive (e.g. `Fix memory tests`).
- Do not amend or rewrite previous commits.

## Pull Request Message

When you open a PR, include two sections in the body:

1. **Summary** – A short overview describing what was changed.
2. **Testing** – Commands you executed and their output.  If tests could not be run due to missing dependencies or network restrictions, mention this.

