# Codex Agent Instructions for avalan

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

Right before committing, run `make lint` to perform syntax checks and
formatting fixes with [black](https://black.readthedocs.io/en/stable/) and
 [ruff](https://docs.astral.sh/ruff/):

```bash
make lint
```

## Running Tests

Run the full test suite before every commit and add unit test coverage for all
code additions or modifications:

```bash
poetry run pytest --verbose -s
```

Tests must pass before you commit.

## Commit Messages

- Keep commit messages short and descriptive (e.g. `Fix memory tests`).
- Do not amend or rewrite previous commits.

## Pull Request Message

When you open a Pull Request, include two sections in the body:

1. **Summary** – A short overview describing what was changed.
2. **Testing** – Commands you executed and their output. If tests could not be
run due to missing dependencies or network restrictions, mention this.

