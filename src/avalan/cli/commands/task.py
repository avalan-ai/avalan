from ...agent.loader import OrchestratorLoader
from ...cli.theme import Theme
from ...pgsql import PsycopgAsyncDatabase, PsycopgPoolSettings
from ...task import (
    REDACTED_MARKER,
    PrivacyField,
    PrivacySanitizationError,
    PrivacySanitizer,
    RunMode,
    TaskClient,
    TaskClientUnsupportedOperationError,
    TaskClientWaitTimeoutError,
    TaskDefinition,
    TaskDefinitionLoader,
    TaskInputType,
    TaskLoadIssue,
    TaskRunState,
    TaskStoreNotFoundError,
    TaskTargetContext,
    TaskValidationError,
    TaskValidationIssue,
    TaskWorker,
    validate_task_definition,
    validate_task_input,
)
from ...task.artifacts import LocalArtifactStore
from ...task.queues import PgsqlTaskQueue
from ...task.stores import (
    TASK_PGSQL_ALEMBIC_VERSION_TABLE,
    TASK_PGSQL_HEAD_REVISION,
    InMemoryTaskStore,
    PgsqlTaskMigrationError,
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
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
from ...task.targets.agent import (
    AgentOrchestratorLoader,
    AgentTaskTargetRunner,
)

from argparse import Namespace
from asyncio import run as asyncio_run
from collections.abc import Coroutine, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass
from json import JSONDecodeError
from json import dumps as json_dumps
from json import loads as json_loads
from logging import Logger, getLogger
from math import isfinite
from os import environ, strerror
from pathlib import Path
from re import fullmatch
from types import TracebackType
from typing import Any, cast
from uuid import uuid4

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


@dataclass(slots=True)
class _TaskCliClientContext:
    client: TaskClient
    database: PsycopgAsyncDatabase | None = None
    stack: AsyncExitStack | None = None

    async def __aenter__(self) -> TaskClient:
        if self.stack is not None:
            await self.stack.__aenter__()
        if self.database is not None:
            await self.database.open()
        return self.client

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self.database is not None:
            await self.database.aclose()
        if self.stack is not None:
            await self.stack.__aexit__(exc_type, exc, traceback)
        return None


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
    hub: object | None = None,
    logger: Logger | None = None,
) -> bool:
    """Run a task definition directly."""
    return _run_awaitable(_task_run(args, console, hub=hub, logger=logger))


def task_enqueue(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: object | None = None,
    logger: Logger | None = None,
) -> bool:
    """Enqueue a task definition."""
    return _run_awaitable(_task_enqueue(args, console, hub=hub, logger=logger))


def task_inspect(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Inspect a task run."""
    return _run_awaitable(_task_inspect(args, console))


def task_output(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Inspect a task run output."""
    return _run_awaitable(_task_output(args, console))


def task_events(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Inspect task run events."""
    return _run_awaitable(_task_events(args, console))


def task_artifacts(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Inspect task run artifacts."""
    return _run_awaitable(_task_artifacts(args, console))


def task_worker(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: object | None = None,
    logger: Logger | None = None,
) -> bool:
    """Run a bounded task queue worker."""
    return _run_awaitable(_task_worker(args, console, hub=hub, logger=logger))


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


async def _task_run(
    args: Namespace,
    console: Console,
    *,
    hub: object | None,
    logger: Logger | None,
) -> bool:
    loaded = _load_definition_for_execution(args, console)
    if loaded is None:
        return False
    definition_path, definition, task_input = loaded
    if definition.run.mode != RunMode.DIRECT:
        _print_task_command_error(
            console,
            "Task run requires a direct-mode definition.",
            "run.mode",
            "Use task enqueue for queued definitions.",
        )
        return False
    dsn = _task_store_dsn(args)
    ephemeral = bool(getattr(args, "ephemeral", False))
    if dsn is None and not ephemeral:
        _print_missing_store(console)
        return False
    client_context = _task_cli_client_context(
        definition_path,
        dsn=dsn,
        schema=_task_store_schema(args),
        queue=False,
        ephemeral=ephemeral,
        hub=hub,
        logger=logger,
    )
    try:
        async with client_context as client:
            result = await client.run(
                definition,
                input_value=task_input.value,
                metadata=_task_command_metadata(ephemeral=ephemeral),
            )
    except (
        AssertionError,
        ImportError,
        OSError,
        TaskClientUnsupportedOperationError,
        TaskValidationError,
    ) as exc:
        _print_task_execution_error(console, exc)
        return False
    console.print(
        (
            "Task run completed (non-durable): "
            if ephemeral
            else "Task run completed: "
        )
        + result.run.run_id,
        markup=False,
    )
    console.print(f"state {result.run.state.value}", markup=False)
    if result.run.result is not None:
        _print_task_result(console, result.run.result.output_summary)
    return result.run.state == TaskRunState.SUCCEEDED


async def _task_enqueue(
    args: Namespace,
    console: Console,
    *,
    hub: object | None,
    logger: Logger | None,
) -> bool:
    loaded = _load_definition_for_execution(args, console)
    if loaded is None:
        return False
    definition_path, definition, task_input = loaded
    if bool(getattr(args, "ephemeral", False)):
        _print_task_command_error(
            console,
            "Ephemeral storage is not supported for queued tasks.",
            "store.ephemeral_unsupported",
            "Configure a durable task store for enqueue.",
        )
        return False
    if definition.run.mode != RunMode.QUEUE:
        _print_task_command_error(
            console,
            "Task enqueue requires a queued-mode definition.",
            "run.mode",
            "Use task run for direct definitions.",
        )
        return False
    dsn = _task_store_dsn(args)
    if dsn is None:
        _print_missing_store(console)
        return False
    client_context = _task_cli_client_context(
        definition_path,
        dsn=dsn,
        schema=_task_store_schema(args),
        queue=True,
        ephemeral=False,
        hub=hub,
        logger=logger,
    )
    try:
        async with client_context as client:
            submission = await client.enqueue(
                definition,
                input_value=task_input.value,
                queue_metadata=_safe_queue_metadata(args),
            )
            console.print(
                f"Task enqueued: {submission.run.run_id}",
                markup=False,
            )
            console.print(f"state {submission.run.state.value}", markup=False)
            if bool(getattr(args, "wait", False)):
                output = await client.wait(
                    submission.run.run_id,
                    timeout_seconds=getattr(args, "wait_timeout", None),
                    poll_interval_seconds=getattr(
                        args,
                        "poll_interval",
                        1.0,
                    ),
                )
                console.print(
                    f"Task finished: {output.run_id}",
                    markup=False,
                )
                console.print(f"state {output.state.value}", markup=False)
                _print_task_result(console, output.output_summary)
                return output.ready
    except (
        AssertionError,
        ImportError,
        OSError,
        TaskClientUnsupportedOperationError,
        TaskClientWaitTimeoutError,
        TaskValidationError,
    ) as exc:
        _print_task_execution_error(console, exc)
        return False
    return True


async def _task_inspect(args: Namespace, console: Console) -> bool:
    client_context = _task_cli_inspection_client_context(args, console)
    if client_context is None:
        return False
    try:
        async with client_context as client:
            inspection = await client.inspect(
                args.run_id,
                after_sequence=_task_cli_after_sequence(args),
            )
    except (
        AssertionError,
        ImportError,
        OSError,
        TaskStoreNotFoundError,
    ) as exc:
        _print_task_inspection_error(console, exc)
        return False
    console.print(
        f"inspect {_format_task_cli_value(inspection.as_dict())}",
        markup=False,
        soft_wrap=True,
    )
    return True


async def _task_output(args: Namespace, console: Console) -> bool:
    client_context = _task_cli_inspection_client_context(args, console)
    if client_context is None:
        return False
    try:
        async with client_context as client:
            output = await client.output(args.run_id)
    except (
        AssertionError,
        ImportError,
        OSError,
        TaskStoreNotFoundError,
    ) as exc:
        _print_task_inspection_error(console, exc)
        return False
    console.print(
        f"output {_format_task_cli_value(output.as_dict())}",
        markup=False,
        soft_wrap=True,
    )
    return True


async def _task_events(args: Namespace, console: Console) -> bool:
    client_context = _task_cli_inspection_client_context(args, console)
    if client_context is None:
        return False
    try:
        async with client_context as client:
            events = await client.events(
                args.run_id,
                attempt_id=getattr(args, "attempt_id", None),
                after_sequence=_task_cli_after_sequence(args),
            )
    except (
        AssertionError,
        ImportError,
        OSError,
        TaskStoreNotFoundError,
    ) as exc:
        _print_task_inspection_error(console, exc)
        return False
    console.print(
        f"events {_format_task_cli_value(_task_event_cli_values(events))}",
        markup=False,
        soft_wrap=True,
    )
    return True


async def _task_artifacts(args: Namespace, console: Console) -> bool:
    client_context = _task_cli_inspection_client_context(args, console)
    if client_context is None:
        return False
    try:
        async with client_context as client:
            artifacts = await client.artifacts(args.run_id)
    except (
        AssertionError,
        ImportError,
        OSError,
        TaskStoreNotFoundError,
    ) as exc:
        _print_task_inspection_error(console, exc)
        return False
    console.print(
        f"artifacts {_format_task_cli_value(artifacts)}",
        markup=False,
        soft_wrap=True,
    )
    return True


async def _task_worker(
    args: Namespace,
    console: Console,
    *,
    hub: object | None,
    logger: Logger | None,
) -> bool:
    if bool(getattr(args, "ephemeral", False)):
        _print_task_command_error(
            console,
            "Ephemeral storage is not supported for task workers.",
            "store.ephemeral_unsupported",
            "Configure a durable task store for workers.",
        )
        return False
    dsn = _task_store_dsn(args)
    if dsn is None:
        _print_missing_store(console)
        return False
    processed = 0
    limit = (
        1
        if bool(getattr(args, "once", False))
        else getattr(
            args,
            "limit",
            1,
        )
    )
    try:
        database = _task_pgsql_database(dsn, _task_store_schema(args))
        store = PgsqlTaskStore(database)
        queue = PgsqlTaskQueue(database)
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(database)
            worker = TaskWorker(
                store,
                queue,
                target=_agent_task_target(
                    Path.cwd(),
                    hub=hub,
                    logger=logger,
                    stack=stack,
                ),
                worker_id=getattr(args, "worker_id", None),
                queue_name=getattr(args, "queue", None) or "default",
                lease_seconds=getattr(args, "lease_seconds", 300),
                artifact_store=_task_artifact_store(),
            )
            for _index in range(limit):
                result = await worker.process_once()
                if not result.processed:
                    break
                processed += 1
                run = (
                    result.completion.run
                    if result.completion is not None
                    else result.retry.run if result.retry is not None else None
                )
                if run is not None:
                    console.print(
                        f"Task processed: {run.run_id} {run.state.value}",
                        markup=False,
                    )
    except (AssertionError, ImportError, OSError, TaskValidationError) as exc:
        _print_task_execution_error(console, exc)
        return False
    console.print(
        f"Task worker processed {processed} run"
        f"{'s' if processed != 1 else ''}.",
        markup=False,
    )
    return True


def _load_definition_for_execution(
    args: Namespace,
    console: Console,
) -> tuple[Path, TaskDefinition, TaskCliInput] | None:
    definition_path = Path(args.definition)
    try:
        load_result = TaskDefinitionLoader().load_result(definition_path)
    except OSError as exc:
        message = strerror(exc.errno) if exc.errno else "Unable to read file."
        console.print("Task definition could not be read.", markup=False)
        console.print(f"error file.read {message}", markup=False)
        return None
    if load_result.definition is None:
        _print_issues(
            console,
            "Task definition could not be loaded.",
            load_result.issues,
        )
        return None
    definition = load_result.definition
    try:
        task_input = task_cli_input(args, definition)
    except TaskCliInputError as exc:
        _print_task_cli_input_error(console, exc)
        return None
    input_issues = validate_task_input(definition, task_input.value)
    if input_issues:
        _print_issues(console, "Task input is invalid.", input_issues)
        return None
    return definition_path, definition, task_input


def _task_cli_client_context(
    definition_path: Path,
    *,
    dsn: str | None,
    schema: str | None,
    queue: bool,
    ephemeral: bool,
    hub: object | None,
    logger: Logger | None,
) -> _TaskCliClientContext:
    stack = AsyncExitStack()
    target = _agent_task_target(
        definition_path.parent,
        hub=hub,
        logger=logger,
        stack=stack,
    )
    artifact_store = _task_artifact_store()
    if ephemeral:
        return _TaskCliClientContext(
            TaskClient(
                InMemoryTaskStore(),
                target=target,
                artifact_store=artifact_store,
                execution_roots=(definition_path.parent,),
            ),
            stack=stack,
        )
    assert dsn is not None
    database = _task_pgsql_database(dsn, schema)
    store = PgsqlTaskStore(database)
    task_queue = PgsqlTaskQueue(database) if queue else None
    return _TaskCliClientContext(
        TaskClient(
            store,
            target=target,
            queue=task_queue,
            artifact_store=artifact_store,
            execution_roots=(definition_path.parent,),
        ),
        database=database,
        stack=stack,
    )


def _task_cli_inspection_client_context(
    args: Namespace,
    console: Console,
) -> _TaskCliClientContext | None:
    dsn = _task_store_dsn(args)
    if dsn is None:
        _print_missing_inspection_store(console)
        return None
    database = _task_pgsql_database(dsn, _task_store_schema(args))
    return _TaskCliClientContext(
        TaskClient(
            PgsqlTaskStore(database),
            target=_task_cli_inspection_target,
        ),
        database=database,
    )


async def _task_cli_inspection_target(
    context: TaskTargetContext,
) -> object:
    _ = context
    raise TaskClientUnsupportedOperationError(
        code="task.inspect_only",
        operation="run",
        message="Inspection clients cannot execute task runs.",
    )


def _agent_task_target(
    ref_base: Path,
    *,
    hub: object | None,
    logger: Logger | None,
    stack: AsyncExitStack,
) -> AgentTaskTargetRunner:
    return AgentTaskTargetRunner(
        cast(
            AgentOrchestratorLoader,
            OrchestratorLoader(
                hub=cast(Any, hub),
                logger=logger or getLogger("avalan.task"),
                participant_id=uuid4(),
                stack=stack,
            ),
        ),
        ref_base=ref_base,
    )


def _task_pgsql_database(
    dsn: str,
    schema: str | None,
) -> PsycopgAsyncDatabase:
    return PsycopgAsyncDatabase(
        PsycopgPoolSettings(
            dsn=dsn,
            schema=schema,
            application_name="avalan-task",
        )
    )


def _task_artifact_store() -> LocalArtifactStore | None:
    root = environ.get("AVALAN_TASK_ARTIFACT_ROOT")
    if not root:
        return None
    return LocalArtifactStore(root, raw_storage_allowed=True)


def _task_store_dsn(args: Namespace) -> str | None:
    value = (
        getattr(args, "store_dsn", None)
        or getattr(args, "dsn", None)
        or environ.get("AVALAN_TASK_STORE_DSN")
        or environ.get("AVALAN_TASK_PGSQL_DSN")
    )
    if isinstance(value, str) and value.strip():
        return value
    return None


def _task_store_schema(args: Namespace) -> str | None:
    value = (
        getattr(args, "store_schema", None)
        or getattr(args, "schema", None)
        or environ.get("AVALAN_TASK_STORE_SCHEMA")
        or environ.get("AVALAN_TASK_PGSQL_SCHEMA")
    )
    if isinstance(value, str) and value.strip():
        return value
    return None


def _safe_queue_metadata(args: Namespace) -> Mapping[str, object]:
    queue_name = getattr(args, "queue", None)
    if isinstance(queue_name, str) and queue_name.strip():
        return {"cli_queue": queue_name}
    return {}


def _task_command_metadata(*, ephemeral: bool) -> Mapping[str, object]:
    return {"store_mode": "ephemeral-memory" if ephemeral else "durable"}


def _print_missing_store(console: Console) -> None:
    console.print("Task store is not configured.", markup=False)
    console.print(
        "error store.missing task command requires a configured durable task"
        " store.",
        markup=False,
    )
    console.print(
        "Set AVALAN_TASK_STORE_DSN, pass --store-dsn, or pass --ephemeral for"
        " a direct local run.",
        markup=False,
    )


def _print_missing_inspection_store(console: Console) -> None:
    console.print("Task store is not configured.", markup=False)
    console.print(
        "error store.missing task inspection requires a configured durable"
        " task store.",
        markup=False,
    )
    console.print(
        "Set AVALAN_TASK_STORE_DSN or pass --store-dsn.",
        markup=False,
    )


def _print_task_command_error(
    console: Console,
    message: str,
    code: str,
    hint: str,
) -> None:
    console.print(message, markup=False)
    console.print(f"error {code}", markup=False)
    console.print(hint, markup=False)


def _print_task_inspection_error(
    console: Console,
    error: BaseException,
) -> None:
    console.print("Task inspection failed.", markup=False)
    if isinstance(error, TaskStoreNotFoundError):
        console.print("error task.not_found", markup=False)
        return
    if isinstance(error, ImportError):
        console.print("error dependency.missing", markup=False)
        return
    if isinstance(error, OSError):
        console.print("error io.failure", markup=False)
        return
    console.print("error task.inspection", markup=False)


def _print_task_execution_error(
    console: Console,
    error: BaseException,
) -> None:
    console.print("Task command failed.", markup=False)
    if isinstance(error, TaskValidationError):
        _print_issues(
            console, "Task definition or input is invalid.", error.issues
        )
        return
    if isinstance(error, TaskClientWaitTimeoutError):
        console.print("error task.wait_timeout", markup=False)
        return
    if isinstance(error, TaskClientUnsupportedOperationError):
        console.print(f"error {error.code}", markup=False)
        return
    if isinstance(error, ImportError):
        console.print("error dependency.missing", markup=False)
        return
    if isinstance(error, OSError):
        console.print("error io.failure", markup=False)
        return
    console.print("error task.execution", markup=False)


def _print_task_result(
    console: Console,
    value: object,
) -> None:
    if value is None:
        return
    console.print(f"output {_format_task_cli_value(value)}", markup=False)


def _task_cli_after_sequence(args: Namespace) -> int | None:
    value = getattr(args, "after_sequence", None)
    if value is None:
        return None
    assert isinstance(value, int)
    assert not isinstance(value, bool)
    assert value >= 0
    return value


def _task_event_cli_value(event: object) -> Mapping[str, object]:
    value: dict[str, object] = {
        "event_id": getattr(event, "event_id"),
        "run_id": getattr(event, "run_id"),
        "sequence": getattr(event, "sequence"),
        "event_type": getattr(event, "event_type"),
        "category": getattr(getattr(event, "category"), "value"),
        "created_at": getattr(event, "created_at").isoformat(),
    }
    attempt_id = getattr(event, "attempt_id", None)
    if attempt_id is not None:
        value["attempt_id"] = attempt_id
    payload = getattr(event, "payload", None)
    if payload is not None:
        value["payload"] = payload
    return value


def _task_event_cli_values(events: Iterable[object]) -> tuple[object, ...]:
    return tuple(_task_event_cli_value(event) for event in events)


def _run_awaitable(awaitable: Coroutine[object, object, bool]) -> bool:
    result: bool | BaseException | None = None

    def target() -> None:
        nonlocal result
        try:
            result = asyncio_run(awaitable)
        except BaseException as exc:
            result = exc

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(target)
        future.result()
    if isinstance(result, BaseException):
        raise result
    assert isinstance(result, bool)
    return result


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
