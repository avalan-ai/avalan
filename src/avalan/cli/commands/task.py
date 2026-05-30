from ...cli.theme import Theme
from ...task import (
    TaskDefinitionLoader,
    TaskLoadIssue,
    TaskValidationIssue,
    validate_task_definition,
)

from argparse import Namespace
from collections.abc import Iterable
from os import strerror
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table


def task_validate(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Validate a task definition without executing it."""
    definition_path = Path(args.definition)
    try:
        load_result = TaskDefinitionLoader().load_result(definition_path)
    except OSError as exc:
        message = strerror(exc.errno) if exc.errno else "Unable to read file."
        console.print("Task definition could not be read.", markup=False)
        console.print(f"error file.read {message}", markup=False)
        return False

    if load_result.definition is None:
        _print_issues(
            console,
            "Task definition could not be loaded.",
            load_result.issues,
        )
        return False

    issues = validate_task_definition(
        load_result.definition,
        execution_roots=(definition_path.parent,),
    )
    if issues:
        _print_issues(console, "Task definition is invalid.", issues)
        return False

    console.print(
        "Task definition is valid: "
        f"{load_result.definition.task.name} "
        f"{load_result.definition.task.version}",
        markup=False,
    )
    return True


def _print_issues(
    console: Console,
    title: str,
    issues: Iterable[TaskLoadIssue | TaskValidationIssue],
) -> None:
    rows = tuple(issue.as_dict() for issue in issues)
    console.print(title, markup=False)
    console.print(
        f"{len(rows)} issue{'s' if len(rows) != 1 else ''} found.",
        markup=False,
    )
    table = Table(show_header=True)
    table.add_column("Severity")
    table.add_column("Code")
    table.add_column("Path")
    table.add_column("Message")
    table.add_column("Hint")
    for row in rows:
        table.add_row(
            escape(row["severity"]),
            escape(row["code"]),
            escape(row["path"]),
            escape(row["message"]),
            escape(row["hint"]),
        )
    console.print(table)
