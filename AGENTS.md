# Agent Instructions

This repository contains the **avalan** framework, a Python project that
orchestrates multiple models and provides a CLI for AI agents. The repository
uses [Poetry](https://python-poetry.org/) for dependency management
and `pytest` for testing.

## Formatting & Style

- Python files use 4 spaces per indentation as enforced by `.editorconfig`.
- Avoid code duplication wherever possible.
- Code must target Python **3.11** or newer with fully strict type hints.
- Prefer `type | None` instead of `Optional[type]` for optional values.
- Use assertions to ensure argument validity.
- Do not ignore exceptions unless instructed.
- Don't use `from __future__ import annotations`.
- Don't use inline imports unless instructed, imports should always be at the top.
- Do not declare `__all__` lists.

### Coding Standards

Code must adhere to the following PEP standards:

1. **PEP 8**: Style guide for Python code: 4 spaces per indentation level;
with naming conventions `lowercase_with_underscores` for modules/packages,
`PascalCase` for classes, `lowercase_with_underscores` for
functions/variables/parameters, `ALL_CAPS_WITH_UNDERSCORES` for constants;
no extra spaces inside parentheses/brackets/braces, surround binary operators
with a single space.

2. **PEP 257**: write docstrings with triple-quoted strings immediately under
`def`/`class`/`module` for one-line docstrings. For multiline docstrings
write the summary, then a blank line, the extended description, and include
optional and relevant sections like `Args:`, `Returns:`, `Raises:`. Follow
imperative tone ("Return", not "Returns"), and wrap at ~72 chars.

3. **PEP 585**: Write `list[int]`, `dict[str, float]`, `tuple[int, ...]`, etc.,
using the built-in collection classes directly, instead of importing `List`,
`Dict`, `Tuple` from `typing`.

4. **PEP 604**: Write `str | float` instead of `Union[str, float]`.

5. **PEP 634**, **PEP 635** and **PEP 636**: use structural pattern matching
(`match` / `case`) when appropriate.

Type hints encouraged throughout the codebase.

Before committing, run `make lint` to perform syntax checks and formatting
fixes with [black](https://black.readthedocs.io/en/stable/) and
[ruff](https://docs.astral.sh/ruff/):

```bash
make lint
```

## Testing

When adding or modifying code, make sure you add unit tests for it, aiming
for full coverage. You can get information about test coverage by running
the `test-coverage` target from the `Makefile`. If you run it without
arguments:

```bash
make test-coverage
```

You'll get test coverage information for all files in `src/`, in the form:
`path: percentage`. If you want to get the list of files where test coverage
is less than 95%, do:

```bash
make test coverage -- -95
```

You can also add a specific path. For example, if you're looking for files
that have less than 95% coverage on folder `src/avalan/tool`:

```bash
make test-coverage -- -95 src/avalan/tool
```

## Submitting changes

Run the full test suite before every commit:

```bash
poetry run pytest --verbose -s
```

Tests must pass before you commit.

### Commit Messages

- Keep commit messages short and descriptive (e.g. `Fix memory tests`).
- Do not amend or rewrite previous commits.

### Pull Request Message

When you open a Pull Request, include two sections in the body:

1. **Summary** – A short overview describing what was changed.
2. **Testing** – Commands you executed and their output. If tests could not be
run due to missing dependencies or network restrictions, mention this.

