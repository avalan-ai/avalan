from ...cli.theme import Theme
from ...task import (
    REDACTED_MARKER,
    PrivacyField,
    PrivacySanitizationError,
    PrivacySanitizer,
    TaskDefinition,
    TaskDefinitionLoader,
    TaskInputType,
    TaskLoadIssue,
    TaskValidationIssue,
    validate_task_definition,
    validate_task_input,
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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from json import dumps as json_dumps
from json import loads as json_loads
from math import isfinite
from os import environ, strerror
from pathlib import Path
from re import fullmatch

from rich.console import Console
from rich.markup import escape
from rich.table import Table


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskCliInput:
    value: object = None
    provided: bool = False


class TaskCliInputError(ValueError):
    code: str
    message: str
    hint: str

    def __init__(self, *, code: str, message: str, hint: str) -> None:
        self.code = code
        self.message = message
        self.hint = hint
        super().__init__(message)


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

    task_input: TaskCliInput | None = None
    if _task_cli_input_provided(args):
        try:
            task_input = task_cli_input(args, load_result.definition)
        except TaskCliInputError as exc:
            _print_task_cli_input_error(console, exc)
            return False
        input_issues = validate_task_input(
            load_result.definition,
            task_input.value,
        )
        if input_issues:
            _print_issues(console, "Task input is invalid.", input_issues)
            return False

    console.print(
        "Task definition is valid: "
        f"{load_result.definition.task.name} "
        f"{load_result.definition.task.version}",
        markup=False,
    )
    if task_input is not None:
        console.print("Task input is valid.", markup=False)
        _print_task_cli_input_summary(
            console,
            load_result.definition,
            task_input.value,
        )
    return True


def task_run(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Print a diagnostic for task run execution."""
    if not _validate_task_cli_input_for_command(args, console):
        return False
    return _print_command_unavailable(console, "run")


def task_enqueue(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Print a diagnostic for task enqueue execution."""
    if not _validate_task_cli_input_for_command(args, console):
        return False
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


def task_cli_input(
    args: Namespace,
    definition: TaskDefinition,
) -> TaskCliInput:
    """Return a validated CLI input shape for task commands."""
    assert isinstance(definition, TaskDefinition)
    raw_input = getattr(args, "task_input", None)
    raw_json = getattr(args, "task_input_json", None)
    input_fields = _task_cli_input_fields(args)
    task_files = _task_cli_file_fields(args)
    provided = (
        raw_input is not None
        or raw_json is not None
        or bool(input_fields)
        or bool(task_files)
    )
    if not provided:
        return TaskCliInput()
    if raw_input is not None and raw_json is not None:
        raise _task_cli_input_error(
            "Pass either --input or --input-json, not both."
        )
    if raw_input is not None and (input_fields or task_files):
        raise _task_cli_input_error(
            "Pass --input by itself or use field-addressed input flags."
        )

    if raw_input is not None:
        return TaskCliInput(
            value=_parse_task_cli_scalar(definition, raw_input),
            provided=True,
        )

    if raw_json is not None:
        json_value = _load_task_cli_json(raw_json)
        if input_fields or task_files:
            if not isinstance(json_value, Mapping):
                raise _task_cli_input_error(
                    "Field-addressed input requires a JSON object."
                )
            object_value = _mutable_json_object(json_value)
            _merge_task_cli_fields(object_value, input_fields)
            _merge_task_cli_files(object_value, task_files)
            json_value = object_value
        return TaskCliInput(value=json_value, provided=True)

    if definition.input.type in {TaskInputType.FILE, TaskInputType.FILE_ARRAY}:
        if input_fields:
            raise _task_cli_input_error(
                "File input contracts only accept --file values."
            )
        descriptors = tuple(
            _task_cli_file_descriptor(reference)
            for _field, reference in task_files
        )
        if definition.input.type == TaskInputType.FILE:
            if len(descriptors) != 1:
                raise _task_cli_input_error(
                    "Single-file input requires exactly one --file value."
                )
            file_value: object = descriptors[0]
        else:
            file_value = list(descriptors)
        return TaskCliInput(value=file_value, provided=True)

    mapped_value: dict[str, object] = {}
    _merge_task_cli_fields(mapped_value, input_fields)
    _merge_task_cli_files(mapped_value, task_files)
    return TaskCliInput(value=mapped_value, provided=True)


def _validate_task_cli_input_for_command(
    args: Namespace,
    console: Console,
) -> bool:
    if not _task_cli_input_provided(args):
        return True
    definition_path_value = getattr(args, "definition", None)
    if not isinstance(definition_path_value, str):
        return True
    definition_path = Path(definition_path_value)
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
    try:
        task_input = task_cli_input(args, load_result.definition)
    except TaskCliInputError as exc:
        _print_task_cli_input_error(console, exc)
        return False
    issues = validate_task_input(load_result.definition, task_input.value)
    if issues:
        _print_issues(console, "Task input is invalid.", issues)
        return False
    console.print("Task input is valid.", markup=False)
    _print_task_cli_input_summary(
        console,
        load_result.definition,
        task_input.value,
    )
    return True


def _task_cli_input_provided(args: Namespace) -> bool:
    return (
        getattr(args, "task_input", None) is not None
        or getattr(args, "task_input_json", None) is not None
        or bool(getattr(args, "task_input_fields", ()))
        or bool(getattr(args, "task_files", ()))
    )


def _task_cli_input_fields(args: Namespace) -> tuple[tuple[str, str], ...]:
    values = getattr(args, "task_input_fields", ()) or ()
    assert isinstance(values, list | tuple)
    fields: list[tuple[str, str]] = []
    for value in values:
        assert isinstance(value, str)
        field, separator, raw_value = value.partition("=")
        if separator != "=" or not _is_task_cli_field_path(field):
            raise _task_cli_input_error("Task input field is invalid.")
        fields.append((field, raw_value))
    return tuple(fields)


def _task_cli_file_fields(args: Namespace) -> tuple[tuple[str, str], ...]:
    values = getattr(args, "task_files", ()) or ()
    assert isinstance(values, list | tuple)
    fields: list[tuple[str, str]] = []
    for value in values:
        assert isinstance(value, str)
        field, separator, reference = value.partition("=")
        if (
            separator != "="
            or not _is_task_cli_field_path(field)
            or not reference.strip()
        ):
            raise _task_cli_input_error("Task file input is invalid.")
        fields.append((field, reference))
    return tuple(fields)


def _parse_task_cli_scalar(
    definition: TaskDefinition,
    value: str,
) -> object:
    assert isinstance(value, str)
    match definition.input.type:
        case TaskInputType.STRING:
            return value
        case TaskInputType.INTEGER:
            parsed = _parse_task_cli_scalar_json(
                value,
                "Task input must be an integer.",
            )
            if isinstance(parsed, int) and not isinstance(parsed, bool):
                return parsed
            raise _task_cli_input_error("Task input must be an integer.")
        case TaskInputType.NUMBER:
            parsed = _parse_task_cli_scalar_json(
                value,
                "Task input must be a finite number.",
            )
            if (
                isinstance(parsed, int | float)
                and not isinstance(parsed, bool)
                and isfinite(parsed)
            ):
                return parsed
            raise _task_cli_input_error("Task input must be a finite number.")
        case TaskInputType.BOOLEAN:
            parsed = _parse_task_cli_scalar_json(
                value.lower(),
                "Task input must be a boolean.",
            )
            if isinstance(parsed, bool):
                return parsed
            raise _task_cli_input_error("Task input must be a boolean.")
        case TaskInputType.OBJECT | TaskInputType.ARRAY:
            return _parse_task_cli_json_value(value)
        case TaskInputType.FILE:
            return _task_cli_file_descriptor(value)
        case TaskInputType.FILE_ARRAY:
            return [_task_cli_file_descriptor(value)]
    raise _task_cli_input_error("Task input type is not supported.")


def _load_task_cli_json(value: str) -> object:
    assert isinstance(value, str)
    if value.startswith("@"):
        path_value = value[1:]
        if not path_value:
            raise _task_cli_input_error("Task input JSON file is invalid.")
        try:
            value = Path(path_value).read_text(encoding="utf-8")
        except OSError as exc:
            message = strerror(exc.errno) if exc.errno else "Unable to read."
            raise TaskCliInputError(
                code="input.read",
                message=f"Task input JSON could not be read: {message}",
                hint="Pass a readable JSON file after @.",
            ) from exc
    return _parse_task_cli_json_value(value)


def _parse_task_cli_json_value(value: str) -> object:
    try:
        return json_loads(value)
    except JSONDecodeError as exc:
        raise TaskCliInputError(
            code="input.json",
            message="Task input JSON is invalid.",
            hint="Pass valid JSON or use --input for plain string input.",
        ) from exc


def _parse_task_cli_scalar_json(value: str, message: str) -> object:
    try:
        return json_loads(value)
    except JSONDecodeError as exc:
        raise _task_cli_input_error(message) from exc


def _parse_task_cli_field_value(value: str) -> object:
    try:
        return json_loads(value)
    except JSONDecodeError:
        return value


def _mutable_json_object(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: _mutable_json_value(item)
        for key, item in value.items()
        if isinstance(key, str)
    }


def _mutable_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _mutable_json_object(value)
    if isinstance(value, list | tuple):
        return [_mutable_json_value(item) for item in value]
    return value


def _merge_task_cli_fields(
    value: dict[str, object],
    fields: tuple[tuple[str, str], ...],
) -> None:
    for field, raw_value in fields:
        _set_task_cli_field(
            value,
            field,
            _parse_task_cli_field_value(raw_value),
            append=False,
        )


def _merge_task_cli_files(
    value: dict[str, object],
    fields: tuple[tuple[str, str], ...],
) -> None:
    for field, reference in fields:
        _set_task_cli_field(
            value,
            field,
            _task_cli_file_descriptor(reference),
            append=True,
        )


def _set_task_cli_field(
    value: dict[str, object],
    field: str,
    item: object,
    *,
    append: bool,
) -> None:
    target = value
    parts = field.split(".")
    for part in parts[:-1]:
        existing = target.get(part)
        if existing is None:
            child: dict[str, object] = {}
            target[part] = child
            target = child
            continue
        if not isinstance(existing, dict):
            raise _task_cli_input_error("Task input field conflicts.")
        target = existing
    leaf = parts[-1]
    existing = target.get(leaf)
    if append:
        if existing is None:
            target[leaf] = item
        elif isinstance(existing, list):
            existing.append(item)
        else:
            target[leaf] = [existing, item]
        return
    if existing is not None:
        raise _task_cli_input_error("Task input field is duplicated.")
    target[leaf] = item


def _task_cli_file_descriptor(reference: str) -> dict[str, object]:
    return {
        "source_kind": "local_path",
        "reference": reference,
    }


def _is_task_cli_field_path(value: str) -> bool:
    return bool(
        fullmatch(
            r"[A-Za-z][A-Za-z0-9_-]{0,63}" r"(\.[A-Za-z][A-Za-z0-9_-]{0,63})*",
            value,
        )
    )


def _task_cli_input_error(message: str) -> TaskCliInputError:
    return TaskCliInputError(
        code="input.parse",
        message=message,
        hint="Use --input, --input-json, --input-name, or --file name=path.",
    )


def _print_task_cli_input_error(
    console: Console,
    error: TaskCliInputError,
) -> None:
    console.print("Task input could not be parsed.", markup=False)
    console.print(f"error {error.code} {error.message}", markup=False)
    console.print(error.hint, markup=False)


def _print_task_cli_input_summary(
    console: Console,
    definition: TaskDefinition,
    value: object,
) -> None:
    sanitizer = PrivacySanitizer(definition.privacy)
    table = Table(show_header=True)
    table.add_column("Kind")
    table.add_column("Summary")
    table.add_row(
        "input",
        _format_task_cli_value(
            _safe_task_cli_summary(sanitizer, PrivacyField.INPUT, value)
        ),
    )
    for index, file_summary in enumerate(_task_cli_file_summaries(value)):
        table.add_row(
            f"file[{index}]",
            _format_task_cli_value(
                _safe_task_cli_summary(
                    sanitizer,
                    PrivacyField.FILES,
                    file_summary,
                )
            ),
        )
    console.print(table)


def _safe_task_cli_summary(
    sanitizer: PrivacySanitizer,
    field: PrivacyField,
    value: object,
) -> object:
    try:
        return sanitizer.sanitize(field, value)
    except PrivacySanitizationError:
        return {"privacy": REDACTED_MARKER}


def _task_cli_file_summaries(
    value: object,
) -> tuple[Mapping[str, object], ...]:
    summaries: list[Mapping[str, object]] = []
    _collect_task_cli_file_summaries(value, summaries)
    return tuple(summaries)


def _collect_task_cli_file_summaries(
    value: object,
    summaries: list[Mapping[str, object]],
) -> None:
    if isinstance(value, Mapping):
        if _is_task_cli_file_descriptor(value):
            summaries.append(value)
            return
        for item in value.values():
            _collect_task_cli_file_summaries(item, summaries)
    elif isinstance(value, list | tuple):
        for item in value:
            _collect_task_cli_file_summaries(item, summaries)


def _is_task_cli_file_descriptor(value: Mapping[str, object]) -> bool:
    return value.get("source_kind") == "local_path" and isinstance(
        value.get("reference"), str
    )


def _format_task_cli_value(value: object) -> str:
    return json_dumps(
        _plain_task_cli_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _plain_task_cli_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_task_cli_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_plain_task_cli_value(item) for item in value]
    return value


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
