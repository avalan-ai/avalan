from ...cli.theme import Theme
from ...task import (
    TaskDefinitionLoader,
    TaskLoadIssue,
    TaskValidationIssue,
    validate_task_definition,
)
from ...task.stores import (
    TASK_PGSQL_ALEMBIC_VERSION_TABLE,
    TASK_PGSQL_HEAD_REVISION,
    PgsqlTaskMigrationError,
    PgsqlTaskMigrationSettings,
    task_pgsql_script_location,
)
from ...task.stores import (
    task_pgsql_check as run_task_pgsql_check,
)
from ...task.stores import (
    task_pgsql_current as run_task_pgsql_current,
)
from ...task.stores import (
    task_pgsql_stamp as run_task_pgsql_stamp,
)
from ...task.stores import (
    task_pgsql_upgrade as run_task_pgsql_upgrade,
)

from argparse import Namespace
from collections.abc import Iterable
from os import environ, strerror
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


def task_run(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Print a diagnostic for task run execution."""
    return _print_command_unavailable(console, "run")


def task_enqueue(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Print a diagnostic for task enqueue execution."""
    return _print_command_unavailable(console, "enqueue")


def task_inspect(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Print a diagnostic for task inspection."""
    return _print_command_unavailable(console, "inspect")


def task_output(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Print a diagnostic for task output inspection."""
    return _print_command_unavailable(console, "output")


def task_events(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Print a diagnostic for task event inspection."""
    return _print_command_unavailable(console, "events")


def task_artifacts(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Print a diagnostic for task artifact inspection."""
    return _print_command_unavailable(console, "artifacts")


def task_worker(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Print a diagnostic for task worker startup."""
    return _print_command_unavailable(console, "worker")


def task_pgsql_status(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Print the current PostgreSQL task schema revision."""
    settings = _task_pgsql_settings(args)
    if settings is None:
        _print_pgsql_missing_dsn(console)
        return False
    try:
        run_task_pgsql_current(settings, verbose=bool(args.verbose))
    except PgsqlTaskMigrationError as exc:
        _print_pgsql_error(console, exc)
        return False
    console.print("Task PostgreSQL migration status checked.", markup=False)
    return True


def task_pgsql_migrate(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Apply PostgreSQL task schema migrations."""
    settings = _task_pgsql_settings(args)
    if settings is None:
        _print_pgsql_missing_dsn(console)
        return False
    revision = _task_pgsql_revision(args)
    try:
        run_task_pgsql_upgrade(settings, revision=revision)
    except (AssertionError, PgsqlTaskMigrationError) as exc:
        _print_pgsql_error(console, exc)
        return False
    console.print(
        f"Task PostgreSQL migrations applied to {revision}.",
        markup=False,
    )
    return True


def task_pgsql_check(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Check whether PostgreSQL task migrations are current."""
    settings = _task_pgsql_settings(args)
    if settings is None:
        _print_pgsql_missing_dsn(console)
        return False
    try:
        run_task_pgsql_check(settings)
    except PgsqlTaskMigrationError as exc:
        _print_pgsql_error(console, exc)
        return False
    console.print("Task PostgreSQL migrations are current.", markup=False)
    return True


def task_pgsql_stamp(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Stamp the PostgreSQL task schema revision."""
    settings = _task_pgsql_settings(args)
    if settings is None:
        _print_pgsql_missing_dsn(console)
        return False
    revision = _task_pgsql_revision(args)
    try:
        run_task_pgsql_stamp(settings, revision=revision)
    except (AssertionError, PgsqlTaskMigrationError) as exc:
        _print_pgsql_error(console, exc)
        return False
    console.print(
        f"Task PostgreSQL schema stamped at {revision}.",
        markup=False,
    )
    return True


def task_pgsql_diagnose(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Print safe PostgreSQL task migration diagnostics."""
    settings = _task_pgsql_settings(args)
    table = Table(show_header=True)
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("dsn", "configured" if settings is not None else "missing")
    table.add_row("schema", settings.schema if settings else "default")
    table.add_row("head_revision", TASK_PGSQL_HEAD_REVISION)
    table.add_row("version_table", TASK_PGSQL_ALEMBIC_VERSION_TABLE)
    table.add_row("script_location", task_pgsql_script_location())
    console.print("Task PostgreSQL diagnostics.", markup=False)
    console.print(table)
    return settings is not None


def _print_command_unavailable(console: Console, command: str) -> bool:
    console.print(
        f"Task {command} command is not available in this build.",
        markup=False,
    )
    return False


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


def _task_pgsql_settings(
    args: Namespace,
) -> PgsqlTaskMigrationSettings | None:
    dsn = args.dsn or environ.get("AVALAN_TASK_PGSQL_DSN")
    if not dsn:
        return None
    schema = args.schema or environ.get("AVALAN_TASK_PGSQL_SCHEMA")
    return PgsqlTaskMigrationSettings(url=dsn, schema=schema)


def _task_pgsql_revision(args: Namespace) -> str:
    revision = args.migration_revision
    assert isinstance(revision, str)
    return revision


def _print_pgsql_missing_dsn(console: Console) -> None:
    console.print("Task PostgreSQL DSN is not configured.", markup=False)
    console.print(
        "Set AVALAN_TASK_PGSQL_DSN or pass --dsn.",
        markup=False,
    )


def _print_pgsql_error(
    console: Console,
    error: AssertionError | PgsqlTaskMigrationError,
) -> None:
    if isinstance(error, PgsqlTaskMigrationError):
        message = str(error)
    else:
        message = "Invalid PostgreSQL migration argument."
    console.print("Task PostgreSQL migration command failed.", markup=False)
    console.print(message, markup=False)
