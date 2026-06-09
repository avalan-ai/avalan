from ...agent.loader import OrchestratorLoader
from ...cli.theme import Theme
from ...flow import (
    Flow,
    FlowDefinition,
    FlowStateStore,
    FlowToolResolver,
    InMemoryFlowStateStore,
    PgsqlFlowStateStore,
)
from ...flow.loader import FlowDefinitionLoader
from ...pgsql import PsycopgAsyncDatabase, PsycopgPoolSettings
from ...task import (
    REDACTED_MARKER,
    ArtifactStoreError,
    FeatureGateDiagnostic,
    HmacProvider,
    PrivacyField,
    PrivacySanitizationError,
    PrivacySanitizer,
    RunMode,
    TaskArtifactPurpose,
    TaskClient,
    TaskClientUnsupportedOperationError,
    TaskClientWaitTimeoutError,
    TaskDefinition,
    TaskDefinitionLoader,
    TaskExecutionContext,
    TaskFeature,
    TaskInputType,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskLoadIssue,
    TaskOutputType,
    TaskRetentionAction,
    TaskRetentionError,
    TaskRetentionService,
    TaskRetentionStoreNotFoundError,
    TaskRunResult,
    TaskRunState,
    TaskStoreNotFoundError,
    TaskTargetContext,
    TaskTargetRunner,
    TaskTargetRunnerRegistry,
    TaskTargetType,
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    TaskWorker,
    TaskWorkerShutdown,
    UsageSource,
    require_feature,
    validate_task_definition,
    validate_task_input,
)
from ...task.artifacts import LocalArtifactStore
from ...task.converters.registry import default_file_converters
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
from ...task.targets.flow import FlowTaskTargetRunner, task_flow_node_registry

from argparse import Namespace
from asyncio import run as asyncio_run
from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Callable, Coroutine, Iterable, Mapping
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
from sys import exc_info
from tempfile import NamedTemporaryFile, TemporaryDirectory
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


@dataclass(frozen=True, slots=True, kw_only=True)
class _TaskCliFileSpec:
    field: str
    descriptor: Mapping[str, object]


_AVALAN_INPUT_TYPE_SCHEMA_KEY = "x-avalan-input-type"
_AVALAN_MIME_TYPES_SCHEMA_KEY = "x-avalan-mime-types"
_PDF_MIME_TYPE = "application/pdf"


class TaskCliInputError(ValueError):
    code: str
    message: str
    hint: str

    def __init__(self, *, code: str, message: str, hint: str) -> None:
        self.code = code
        self.message = message
        self.hint = hint
        super().__init__(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class _TaskCliHmacProvider:
    key_id: str
    secret: bytes
    algorithm: str = "hmac-sha256"

    def __post_init__(self) -> None:
        assert isinstance(self.key_id, str) and self.key_id.strip()
        assert isinstance(self.secret, bytes) and self.secret
        assert isinstance(self.algorithm, str) and self.algorithm.strip()

    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        assert isinstance(purpose, TaskKeyPurpose)
        if key_id is not None:
            assert isinstance(key_id, str) and key_id.strip()
        return TaskKeyMaterial(
            key_id=key_id or self.key_id,
            algorithm=self.algorithm,
            secret=self.secret,
        )


@dataclass(slots=True)
class _TaskCliClientContext:
    client: TaskClient
    database: PsycopgAsyncDatabase | None = None
    stack: AsyncExitStack | None = None

    async def __aenter__(self) -> TaskClient:
        stack_entered = False
        if self.stack is not None:
            await self.stack.__aenter__()
            stack_entered = True
        try:
            if self.database is not None:
                await self.database.open()
        except BaseException:
            if stack_entered and self.stack is not None:
                exc_type, exc, traceback = exc_info()
                await self.stack.__aexit__(exc_type, exc, traceback)
            raise
        return self.client

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            if self.database is not None:
                await self.database.aclose()
        finally:
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
        hmac_provider=_task_hmac_provider(),
        require_configured_keys=True,
        execution_roots=(definition_path.parent,),
    )
    if issues:
        _print_issues(console, "Task definition is invalid.", issues)
        return False
    flow_issues = _validate_task_flow_reference(
        definition_path,
        load_result.definition,
    )
    if flow_issues:
        _print_issues(console, "Task definition is invalid.", flow_issues)
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


def _validate_task_flow_reference(
    definition_path: Path,
    definition: TaskDefinition,
) -> tuple[TaskValidationIssue, ...]:
    assert isinstance(definition_path, Path)
    assert isinstance(definition, TaskDefinition)
    if definition.execution.type != TaskTargetType.FLOW:
        return ()
    flow_ref = definition.execution.ref
    path = Path(flow_ref)
    if not path.is_absolute():
        path = definition_path.parent / path
    context = TaskTargetContext(
        definition=definition,
        execution=TaskExecutionContext(
            run_id="validation-run",
            attempt_id="validation-attempt",
            attempt_number=1,
        ),
        file_converters=default_file_converters(),
    )
    agent_target = _agent_task_target(
        definition_path.parent,
        hub=None,
        logger=getLogger("avalan.task"),
        stack=AsyncExitStack(),
    )
    loader = FlowDefinitionLoader(
        registry=task_flow_node_registry(
            context,
            agent_runner=agent_target,
            execution_roots=(definition_path.parent,),
        )
    )
    try:
        result = loader.load_validation_result(path)
        if result.definition is not None and not result.authoring_graph:
            result = loader.load_result(path)
    except OSError:
        return (
            TaskValidationIssue(
                code="flow.read_failed",
                path="execution.ref",
                message="Flow definition could not be read.",
                hint="Use a readable flow TOML file.",
                category=TaskValidationCategory.UNSUPPORTED,
            ),
        )
    if result.definition is not None:
        return ()
    return tuple(
        TaskValidationIssue(
            code=issue.code,
            path=issue.path,
            message=issue.message,
            hint=issue.hint,
            category=TaskValidationCategory.UNSUPPORTED,
        )
        for issue in result.issues
    )


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


def task_usage(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Inspect task run usage."""
    return _run_awaitable(_task_usage(args, console))


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
    shutdown = TaskWorkerShutdown()
    return _run_awaitable(
        _task_worker(
            args,
            console,
            hub=hub,
            logger=logger,
            shutdown=shutdown,
        ),
        on_interrupt=shutdown.request,
    )


def task_retention_sweep(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Delete expired task artifact bytes."""
    return _run_awaitable(_task_retention_sweep(args, console))


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


async def _task_retention_sweep(
    args: Namespace,
    console: Console,
) -> bool:
    dsn = _task_store_dsn(args)
    if dsn is None:
        _print_missing_inspection_store(console)
        return False
    artifact_store = _task_artifact_store()
    if artifact_store is None:
        _print_missing_artifact_store(console)
        return False
    try:
        database = _task_pgsql_database(dsn, _task_store_schema(args))
        service = TaskRetentionService(
            PgsqlTaskStore(database),
            {"local": artifact_store},
        )
        async with database:
            sweep = await service.sweep_expired(
                purposes=_task_retention_purposes(args),
                limit=_task_retention_limit(args),
            )
    except (
        AssertionError,
        ArtifactStoreError,
        ImportError,
        OSError,
        TaskRetentionError,
    ) as exc:
        _print_task_retention_error(console, exc)
        return False
    counts = _task_retention_counts(sweep.results)
    console.print(
        f"Task retention sweep processed {len(sweep.results)} artifact"
        f"{'s' if len(sweep.results) != 1 else ''}.",
        markup=False,
    )
    console.print(
        "retention "
        + _format_task_cli_value(
            {
                "deleted": counts[TaskRetentionAction.DELETED],
                "limit": sweep.limit,
                "lost": counts[TaskRetentionAction.LOST],
                "total": len(sweep.results),
            }
        ),
        markup=False,
    )
    return True


async def _task_run(
    args: Namespace,
    console: Console,
    *,
    hub: object | None,
    logger: Logger | None,
) -> bool:
    diagnostic_console = _task_diagnostic_console(args, console)
    loaded = _load_definition_for_execution(args, diagnostic_console)
    if loaded is None:
        return False
    definition_path, definition, task_input = loaded
    if definition.run.mode != RunMode.DIRECT:
        _print_task_command_error(
            diagnostic_console,
            "Task run requires a direct-mode definition.",
            "run.mode",
            "Use task enqueue for queued definitions.",
        )
        return False
    structured_output = _task_run_structured_output_requested(args)
    if structured_output and not _task_output_is_structured(definition):
        _print_task_command_error(
            diagnostic_console,
            "Task run output is not structured.",
            "output.unsupported",
            "Use --json or --output with json, object, or array output tasks.",
        )
        return False
    if not _validate_task_run_output_path(args, diagnostic_console):
        return False
    dsn = _task_store_dsn(args)
    ephemeral = bool(getattr(args, "ephemeral", False))
    if dsn is None and not ephemeral:
        _print_missing_store(diagnostic_console)
        return False
    client_context = _task_cli_client_context(
        definition_path,
        dsn=dsn,
        schema=_task_store_schema(args),
        queue=False,
        ephemeral=ephemeral,
        hub=hub,
        logger=logger,
        input_value=task_input.value,
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
        _print_task_execution_error(diagnostic_console, exc)
        return False
    if result.run.state != TaskRunState.SUCCEEDED:
        if not _task_run_json_output(args) and not _task_run_quiet(args):
            _print_task_run_summary(console, result, ephemeral=ephemeral)
        _print_task_run_failure(diagnostic_console, result)
        return False
    if structured_output:
        output_written = _write_task_run_structured_output(
            args,
            console,
            diagnostic_console,
            result.output,
        )
        if not output_written:
            return False
    if not _task_run_json_output(args) and not _task_run_quiet(args):
        _print_task_run_summary(console, result, ephemeral=ephemeral)
    return True


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
                queue_name=_task_cli_queue_name(args),
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


async def _task_usage(args: Namespace, console: Console) -> bool:
    client_context = _task_cli_inspection_client_context(args, console)
    if client_context is None:
        return False
    try:
        source = _task_cli_usage_source(args)
        async with client_context as client:
            inspection = await client.usage_inspection(
                args.run_id,
                attempt_id=getattr(args, "attempt_id", None),
                source=source,
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
        f"usage {_format_task_cli_value(inspection.as_dict())}",
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
    shutdown: TaskWorkerShutdown | None = None,
) -> bool:
    if bool(getattr(args, "ephemeral", False)):
        _print_task_command_error(
            console,
            "Ephemeral storage is not supported for task workers.",
            "store.ephemeral_unsupported",
            "Configure a durable task store for workers.",
        )
        return False
    lease_seconds = getattr(args, "lease_seconds", 300)
    heartbeat_seconds = getattr(args, "heartbeat_seconds", None)
    if heartbeat_seconds is not None and heartbeat_seconds >= lease_seconds:
        _print_task_command_error(
            console,
            "Task worker heartbeat must be shorter than the lease.",
            "worker.heartbeat_interval",
            "Set --heartbeat-seconds lower than --lease-seconds.",
        )
        return False
    dsn = _task_store_dsn(args)
    if dsn is None:
        _print_missing_store(console)
        return False
    diagnostics = require_feature(TaskFeature.QUEUE_WORKERS)
    if diagnostics:
        _print_feature_gate_diagnostics(console, diagnostics)
        return False
    processed = 0
    lease_lost = False
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
        shutdown = shutdown or TaskWorkerShutdown()
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
                hmac_provider=_task_hmac_provider(),
                worker_id=getattr(args, "worker_id", None),
                queue_name=getattr(args, "queue", None) or "default",
                lease_seconds=lease_seconds,
                artifact_store=_task_artifact_store(),
                shutdown=shutdown,
                heartbeat_seconds=heartbeat_seconds,
            )
            for _index in range(limit):
                if shutdown.requested:
                    break
                result = await worker.process_once()
                if not result.processed:
                    break
                processed += 1
                abandonment = getattr(result, "abandonment", None)
                run = (
                    result.completion.run
                    if result.completion is not None
                    else (
                        result.retry.run
                        if result.retry is not None
                        else (
                            abandonment.run
                            if abandonment is not None
                            else None
                        )
                    )
                )
                if run is not None:
                    console.print(
                        f"Task processed: {run.run_id} {run.state.value}",
                        markup=False,
                    )
                if bool(getattr(result, "lease_lost", False)):
                    lease_lost = True
                    claim = result.claimed
                    message = "Task claim lost."
                    if claim is not None:
                        message = f"Task claim lost: {claim.run.run_id}"
                    console.print(message, markup=False)
                    break
                if bool(getattr(result, "shutdown_requested", False)):
                    console.print(
                        "Task worker shutdown requested.",
                        markup=False,
                    )
                    break
    except (AssertionError, ImportError, OSError, TaskValidationError) as exc:
        _print_task_execution_error(console, exc)
        return False
    console.print(
        f"Task worker processed {processed} run"
        f"{'s' if processed != 1 else ''}.",
        markup=False,
    )
    return not lease_lost


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
    input_value: object = None,
    flow_tool_resolver: FlowToolResolver | None = None,
) -> _TaskCliClientContext:
    stack = AsyncExitStack()
    artifact_store = _task_artifact_store()
    if (
        artifact_store is None
        and ephemeral
        and _task_cli_contains_local_file(input_value)
    ):
        artifact_root = stack.enter_context(TemporaryDirectory())
        artifact_store = LocalArtifactStore(
            artifact_root,
            raw_storage_allowed=True,
        )
    hmac_provider = _task_hmac_provider()
    agent_target = _agent_task_target(
        definition_path.parent,
        hub=hub,
        logger=logger,
        stack=stack,
    )
    if ephemeral:
        memory_store = InMemoryTaskStore()
        target = _task_cli_target_runner(
            definition_path,
            agent_target=agent_target,
            flow_state_store=InMemoryFlowStateStore(task_store=memory_store),
            flow_tool_resolver=flow_tool_resolver,
        )
        return _TaskCliClientContext(
            TaskClient(
                memory_store,
                target=target,
                artifact_store=artifact_store,
                hmac_provider=hmac_provider,
                execution_roots=(definition_path.parent,),
                input_roots=(Path.cwd(),),
            ),
            stack=stack,
        )
    assert dsn is not None
    database = _task_pgsql_database(dsn, schema)
    pgsql_store = PgsqlTaskStore(database)
    task_queue = PgsqlTaskQueue(database) if queue else None
    target = _task_cli_target_runner(
        definition_path,
        agent_target=agent_target,
        flow_state_store=PgsqlFlowStateStore(database),
        flow_tool_resolver=flow_tool_resolver,
    )
    return _TaskCliClientContext(
        TaskClient(
            pgsql_store,
            target=target,
            queue=task_queue,
            artifact_store=artifact_store,
            hmac_provider=hmac_provider,
            execution_roots=(definition_path.parent,),
            input_roots=(Path.cwd(),),
        ),
        database=database,
        stack=stack,
    )


def _task_cli_target_runner(
    definition_path: Path,
    *,
    agent_target: TaskTargetRunner,
    flow_state_store: FlowStateStore,
    flow_tool_resolver: FlowToolResolver | None = None,
) -> TaskTargetRunner:
    return TaskTargetRunnerRegistry(
        agent_target,
        {
            TaskTargetType.FLOW: FlowTaskTargetRunner(
                ref_base=definition_path.parent,
                strict_resolver=_task_strict_flow_resolver(
                    definition_path.parent,
                    agent_runner=agent_target,
                    tool_resolver=flow_tool_resolver,
                ),
                flow_state_store=flow_state_store,
                agent_runner=agent_target,
                execution_roots=(definition_path.parent,),
                tool_resolver=flow_tool_resolver,
            )
        },
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


def _task_flow_resolver(
    ref_base: Path,
    *,
    agent_runner: TaskTargetRunner | None = None,
    tool_resolver: FlowToolResolver | None = None,
) -> Callable[[TaskTargetContext], Flow]:
    def resolve(context: TaskTargetContext) -> Flow:
        flow_ref = context.definition.execution.ref
        path = Path(flow_ref)
        if not path.is_absolute():
            path = ref_base / path
        result = FlowDefinitionLoader(
            registry=task_flow_node_registry(
                context,
                agent_runner=agent_runner,
                execution_roots=(ref_base,),
                tool_resolver=tool_resolver,
            )
        ).load_result(path)
        if result.flow is None:
            raise TaskValidationError(
                tuple(
                    TaskValidationIssue(
                        code=issue.code,
                        path=issue.path,
                        message=issue.message,
                        hint=issue.hint,
                        category=TaskValidationCategory.UNSUPPORTED,
                    )
                    for issue in result.issues
                )
            )
        return result.flow

    return resolve


def _task_strict_flow_resolver(
    ref_base: Path,
    *,
    agent_runner: TaskTargetRunner | None = None,
    tool_resolver: FlowToolResolver | None = None,
) -> Callable[[TaskTargetContext], FlowDefinition]:
    def resolve(context: TaskTargetContext) -> FlowDefinition:
        flow_ref = context.definition.execution.ref
        path = Path(flow_ref)
        if not path.is_absolute():
            path = ref_base / path
        result = FlowDefinitionLoader(
            registry=task_flow_node_registry(
                context,
                agent_runner=agent_runner,
                execution_roots=(ref_base,),
                tool_resolver=tool_resolver,
            )
        ).load_validation_result(path)
        if result.definition is None:
            raise TaskValidationError(
                tuple(
                    TaskValidationIssue(
                        code=issue.code,
                        path=issue.path,
                        message=issue.message,
                        hint=issue.hint,
                        category=TaskValidationCategory.UNSUPPORTED,
                    )
                    for issue in result.issues
                )
            )
        return result.definition

    return resolve


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


def _task_hmac_provider() -> HmacProvider | None:
    key_id = environ.get("AVALAN_TASK_HMAC_KEY_ID")
    key_b64 = environ.get("AVALAN_TASK_HMAC_KEY_B64")
    if not (
        isinstance(key_id, str)
        and key_id.strip()
        and isinstance(key_b64, str)
        and key_b64.strip()
    ):
        return None
    try:
        secret = b64decode(key_b64.strip(), validate=True)
    except (BinasciiError, ValueError):
        return None
    return _TaskCliHmacProvider(key_id=key_id, secret=secret)


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
    queue_name = _task_cli_queue_name(args)
    if queue_name is not None:
        return {"cli_queue": queue_name}
    return {}


def _task_retention_purposes(
    args: Namespace,
) -> tuple[TaskArtifactPurpose, ...] | None:
    values = getattr(args, "purpose", ()) or ()
    assert isinstance(values, list | tuple)
    purposes: list[TaskArtifactPurpose] = []
    for value in values:
        assert isinstance(value, str)
        try:
            purposes.append(TaskArtifactPurpose(value))
        except ValueError as exc:
            raise AssertionError(
                "purpose must be a task artifact purpose"
            ) from exc
    return tuple(purposes) or None


def _task_retention_limit(args: Namespace) -> int:
    value = getattr(args, "limit", 100)
    assert isinstance(value, int), "limit must be an integer"
    assert not isinstance(value, bool), "limit must be an integer"
    assert value > 0, "limit must be positive"
    return value


def _task_retention_counts(
    results: Iterable[object],
) -> Mapping[TaskRetentionAction, int]:
    counts = {
        TaskRetentionAction.DELETED: 0,
        TaskRetentionAction.LOST: 0,
    }
    for result in results:
        action = getattr(result, "action")
        assert isinstance(action, TaskRetentionAction)
        counts[action] += 1
    return counts


def _task_cli_queue_name(args: Namespace) -> str | None:
    value = getattr(args, "queue", None)
    if isinstance(value, str) and value.strip():
        return value
    return None


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


def _print_missing_artifact_store(console: Console) -> None:
    console.print("Task artifact store is not configured.", markup=False)
    console.print(
        "error artifact_store.missing retention sweep requires a configured"
        " artifact store.",
        markup=False,
    )
    console.print(
        "Set AVALAN_TASK_ARTIFACT_ROOT before running retention sweep.",
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


def _print_task_retention_error(
    console: Console,
    error: BaseException,
) -> None:
    console.print("Task retention sweep failed.", markup=False)
    if isinstance(error, TaskRetentionStoreNotFoundError):
        console.print("error artifact_store.missing", markup=False)
        return
    if isinstance(error, ArtifactStoreError):
        console.print("error artifact_store.failure", markup=False)
        return
    if isinstance(error, ImportError):
        console.print("error dependency.missing", markup=False)
        return
    if isinstance(error, OSError):
        console.print("error io.failure", markup=False)
        return
    console.print("error retention.sweep", markup=False)


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


def _print_feature_gate_diagnostics(
    console: Console,
    diagnostics: Iterable[FeatureGateDiagnostic],
) -> None:
    rows = tuple(diagnostics)
    console.print("Task feature is unavailable.", markup=False)
    for diagnostic in rows:
        console.print(
            f"error {diagnostic.code} {diagnostic.message}",
            markup=False,
        )
        console.print(diagnostic.hint, markup=False)


def _print_task_result(
    console: Console,
    value: object,
) -> None:
    if value is None:
        return
    console.print(f"output {_format_task_cli_value(value)}", markup=False)


def _print_task_run_summary(
    console: Console,
    result: TaskRunResult,
    *,
    ephemeral: bool,
) -> None:
    run = result.run
    console.print(
        (
            "Task run completed (non-durable): "
            if ephemeral
            else "Task run completed: "
        )
        + run.run_id,
        markup=False,
    )
    console.print(f"state {run.state.value}", markup=False)
    if run.result is not None:
        _print_task_result(console, run.result.output_summary)


def _print_task_run_failure(
    console: Console,
    result: TaskRunResult,
) -> None:
    run = result.run
    console.print("Task run did not succeed.", markup=False)
    console.print(f"error task.run_failed {run.state.value}", markup=False)
    if run.result is not None and run.result.error is not None:
        console.print(
            f"failure {_format_task_cli_value(run.result.error)}",
            markup=False,
        )


def _task_diagnostic_console(args: Namespace, console: Console) -> Console:
    if not _task_run_json_output(args):
        return console
    return Console(stderr=True, highlight=False)


def _task_run_json_output(args: Namespace) -> bool:
    return bool(getattr(args, "task_run_json", False))


def _task_run_output_path(args: Namespace) -> str | None:
    value = getattr(args, "task_output_path", None)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _task_run_quiet(args: Namespace) -> bool:
    return bool(getattr(args, "quiet", False))


def _task_run_structured_output_requested(args: Namespace) -> bool:
    return (
        _task_run_json_output(args) or _task_run_output_path(args) is not None
    )


def _task_output_is_structured(definition: TaskDefinition) -> bool:
    return definition.output.type in {
        TaskOutputType.JSON,
        TaskOutputType.OBJECT,
        TaskOutputType.ARRAY,
    }


def _write_task_run_structured_output(
    args: Namespace,
    console: Console,
    diagnostic_console: Console,
    value: object,
) -> bool:
    serialized = _format_task_cli_value(value) + "\n"
    output_path = _task_run_output_path(args)
    if output_path is not None and not _write_task_run_output_file(
        output_path,
        serialized,
        diagnostic_console,
    ):
        return False
    if _task_run_json_output(args):
        console.file.write(serialized)
        console.file.flush()
    return True


def _write_task_run_output_file(
    value: str,
    serialized: str,
    console: Console,
) -> bool:
    path = Path(value)
    parent = path.parent if path.parent != Path("") else Path(".")
    if not parent.exists() or not parent.is_dir():
        _print_task_command_error(
            console,
            "Task output file could not be written.",
            "output.write",
            "Create the parent directory before using --output.",
        )
        return False
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            delete=False,
            dir=parent,
            encoding="utf-8",
        ) as file:
            temp_path = Path(file.name)
            file.write(serialized)
            file.flush()
        temp_path.replace(path)
    except OSError:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass
        _print_task_command_error(
            console,
            "Task output file could not be written.",
            "output.write",
            "Use a writable output path.",
        )
        return False
    return True


def _validate_task_run_output_path(
    args: Namespace,
    console: Console,
) -> bool:
    output_path = _task_run_output_path(args)
    if output_path is None:
        return True
    path = Path(output_path)
    parent = path.parent if path.parent != Path("") else Path(".")
    if parent.exists() and parent.is_dir():
        return True
    _print_task_command_error(
        console,
        "Task output file could not be written.",
        "output.write",
        "Create the parent directory before using --output.",
    )
    return False


def _task_cli_after_sequence(args: Namespace) -> int | None:
    value = getattr(args, "after_sequence", None)
    if value is None:
        return None
    assert isinstance(value, int)
    assert not isinstance(value, bool)
    assert value >= 0
    return value


def _task_cli_usage_source(args: Namespace) -> UsageSource | None:
    value = getattr(args, "source", None)
    if value is None:
        return None
    assert isinstance(value, str)
    try:
        return UsageSource(value)
    except ValueError as exc:
        raise AssertionError("source must be a task usage source") from exc


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


def _run_awaitable(
    awaitable: Coroutine[object, object, bool],
    *,
    on_interrupt: Callable[[], None] | None = None,
) -> bool:
    result: bool | BaseException | None = None

    def target() -> None:
        nonlocal result
        try:
            result = asyncio_run(awaitable)
        except BaseException as exc:
            result = exc

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(target)
        try:
            future.result()
        except KeyboardInterrupt:
            if on_interrupt is not None:
                on_interrupt()
                future.result()
            raise
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
    raw_pdf = getattr(args, "task_pdf", None)
    input_fields = _task_cli_input_fields(args)
    file_options_provided = _task_cli_file_options_provided(args)
    if raw_input is not None and raw_json is not None:
        raise _task_cli_input_error(
            "Pass either --input or --input-json, not both."
        )
    if raw_pdf is not None:
        if raw_input is not None or raw_json is not None:
            raise _task_cli_input_error(
                "Pass --pdf by itself for single-file PDF input."
            )
        descriptor = _task_cli_pdf_descriptor(raw_pdf)
        if definition.input.type == TaskInputType.FILE:
            if input_fields or bool(getattr(args, "task_files", ()) or ()):
                raise _task_cli_input_error(
                    "Pass --pdf by itself for single-file PDF input."
                )
            if file_options_provided:
                raise _task_cli_input_error(
                    "Pass --pdf by itself for single-file PDF input."
                )
            value: object = descriptor
        elif definition.input.type == TaskInputType.FILE_ARRAY:
            if input_fields or bool(getattr(args, "task_files", ()) or ()):
                raise _task_cli_input_error(
                    "Pass --pdf by itself for single-file PDF input."
                )
            if file_options_provided:
                raise _task_cli_input_error(
                    "Pass --pdf by itself for single-file PDF input."
                )
            value = [descriptor]
        elif definition.input.type == TaskInputType.OBJECT:
            field = _task_cli_pdf_object_field(definition)
            if field is None:
                raise _task_cli_input_error(
                    "--pdf requires a single top-level file input."
                )
            mapped_value: dict[str, object] = {}
            _merge_task_cli_fields(mapped_value, input_fields)
            _merge_task_cli_files(
                mapped_value,
                (
                    _TaskCliFileSpec(
                        field=field,
                        descriptor=descriptor,
                    ),
                    *_task_cli_file_specs(args),
                ),
                definition=definition,
            )
            value = mapped_value
        else:
            raise _task_cli_input_error(
                "--pdf requires a single top-level file input."
            )
        return TaskCliInput(value=value, provided=True)
    task_files = _task_cli_file_specs(args)
    provided = (
        raw_input is not None
        or raw_json is not None
        or bool(input_fields)
        or bool(task_files)
        or file_options_provided
    )
    if not provided:
        return TaskCliInput()
    if raw_input is not None and (
        input_fields or task_files or file_options_provided
    ):
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
        if input_fields or task_files or file_options_provided:
            if not isinstance(json_value, Mapping):
                raise _task_cli_input_error(
                    "Field-addressed input requires a JSON object."
                )
            object_value = _mutable_json_object(json_value)
            _merge_task_cli_fields(object_value, input_fields)
            _merge_task_cli_files(
                object_value,
                task_files,
                definition=definition,
            )
            json_value = object_value
        return TaskCliInput(value=json_value, provided=True)

    if definition.input.type in {TaskInputType.FILE, TaskInputType.FILE_ARRAY}:
        if input_fields:
            raise _task_cli_input_error(
                "File input contracts only accept --file values."
            )
        descriptors = tuple(spec.descriptor for spec in task_files)
        if definition.input.type == TaskInputType.FILE:
            if len(descriptors) != 1:
                raise _task_cli_input_error(
                    "Single-file input requires exactly one --file value."
                )
            file_value: object = descriptors[0]
        else:
            file_value = list(descriptors)
        return TaskCliInput(value=file_value, provided=True)

    object_input: dict[str, object] = {}
    _merge_task_cli_fields(object_input, input_fields)
    _merge_task_cli_files(object_input, task_files, definition=definition)
    return TaskCliInput(value=object_input, provided=True)


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
        or getattr(args, "task_pdf", None) is not None
        or bool(getattr(args, "task_input_fields", ()))
        or bool(getattr(args, "task_files", ()))
        or _task_cli_file_options_provided(args)
    )


def _task_cli_file_options_provided(args: Namespace) -> bool:
    return any(
        bool(getattr(args, name, ()))
        for name in (
            "task_file_descriptors",
            "task_provider_file_ids",
            "task_hosted_urls",
            "task_object_store_uris",
            "task_file_mime_types",
            "task_file_roles",
            "task_file_sizes",
            "task_file_sha256",
            "task_file_conversions",
        )
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


def _task_cli_file_specs(args: Namespace) -> tuple[_TaskCliFileSpec, ...]:
    specs: list[_TaskCliFileSpec] = []
    specs.extend(
        _TaskCliFileSpec(
            field=field,
            descriptor=_task_cli_file_descriptor(reference),
        )
        for field, reference in _task_cli_file_fields(args)
    )
    specs.extend(_task_cli_descriptor_specs(args))
    specs.extend(
        _task_cli_provider_reference_specs(
            args,
            attr="task_provider_file_ids",
            kind="provider_file_id",
        )
    )
    specs.extend(
        _task_cli_provider_reference_specs(
            args,
            attr="task_hosted_urls",
            kind="hosted_url",
        )
    )
    specs.extend(
        _task_cli_provider_reference_specs(
            args,
            attr="task_object_store_uris",
            kind="object_store_uri",
        )
    )
    return _apply_task_cli_file_hints(tuple(specs), args)


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


def _task_cli_descriptor_specs(
    args: Namespace,
) -> tuple[_TaskCliFileSpec, ...]:
    values = getattr(args, "task_file_descriptors", ()) or ()
    assert isinstance(values, list | tuple)
    specs: list[_TaskCliFileSpec] = []
    for value in values:
        assert isinstance(value, str)
        field, separator, raw_descriptor = value.partition("=")
        if (
            separator != "="
            or not _is_task_cli_field_path(field)
            or not raw_descriptor.strip()
        ):
            raise _task_cli_input_error("Task file descriptor is invalid.")
        descriptor = _load_task_cli_json(raw_descriptor)
        if not isinstance(descriptor, Mapping):
            raise _task_cli_input_error("Task file descriptor is invalid.")
        specs.append(
            _TaskCliFileSpec(
                field=field,
                descriptor=cast(
                    Mapping[str, object],
                    _mutable_json_object(descriptor),
                ),
            )
        )
    return tuple(specs)


def _task_cli_provider_reference_specs(
    args: Namespace,
    *,
    attr: str,
    kind: str,
) -> tuple[_TaskCliFileSpec, ...]:
    values = getattr(args, attr, ()) or ()
    assert isinstance(values, list | tuple)
    specs: list[_TaskCliFileSpec] = []
    for value in values:
        assert isinstance(value, str)
        field, separator, raw_reference = value.partition("=")
        provider, provider_separator, reference = raw_reference.partition(":")
        if (
            separator != "="
            or provider_separator != ":"
            or not _is_task_cli_field_path(field)
            or not provider.strip()
            or not reference.strip()
        ):
            raise _task_cli_input_error(
                "Task provider file reference is invalid."
            )
        specs.append(
            _TaskCliFileSpec(
                field=field,
                descriptor=_task_cli_provider_reference_descriptor(
                    kind=kind,
                    provider=provider,
                    reference=reference,
                ),
            )
        )
    return tuple(specs)


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
    fields: tuple[_TaskCliFileSpec, ...],
    *,
    definition: TaskDefinition | None = None,
) -> None:
    for spec in fields:
        _set_task_cli_field(
            value,
            spec.field,
            spec.descriptor,
            append=True,
            array=_task_cli_schema_field_input_type(
                definition,
                spec.field,
            )
            == TaskInputType.FILE_ARRAY.value,
        )


def _set_task_cli_field(
    value: dict[str, object],
    field: str,
    item: object,
    *,
    append: bool,
    array: bool = False,
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
            target[leaf] = [item] if array else item
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


def _task_cli_pdf_descriptor(reference: str) -> dict[str, object]:
    descriptor = _task_cli_file_descriptor(reference)
    descriptor["mime_type"] = _PDF_MIME_TYPE
    return descriptor


def _task_cli_pdf_object_field(definition: TaskDefinition) -> str | None:
    schema = definition.input.schema
    if not isinstance(schema, Mapping):
        return None
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None
    fields = tuple(
        key
        for key, value in properties.items()
        if isinstance(key, str) and _task_cli_schema_accepts_pdf(value)
    )
    if len(fields) != 1:
        return None
    return fields[0]


def _task_cli_schema_field_input_type(
    definition: TaskDefinition | None,
    field: str,
) -> str | None:
    if definition is None or definition.input.type != TaskInputType.OBJECT:
        return None
    schema = definition.input.schema
    if not isinstance(schema, Mapping):
        return None
    field_schema: object = schema
    for part in field.split("."):
        if not isinstance(field_schema, Mapping):
            return None
        properties = field_schema.get("properties")
        if not isinstance(properties, Mapping):
            return None
        field_schema = properties.get(part)
    if not isinstance(field_schema, Mapping):
        return None
    input_type = field_schema.get(_AVALAN_INPUT_TYPE_SCHEMA_KEY)
    if not isinstance(input_type, str):
        return None
    return input_type


def _task_cli_schema_accepts_pdf(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    input_type = value.get(_AVALAN_INPUT_TYPE_SCHEMA_KEY)
    if input_type not in {
        TaskInputType.FILE.value,
        TaskInputType.FILE_ARRAY.value,
    }:
        return False
    mime_types = value.get(_AVALAN_MIME_TYPES_SCHEMA_KEY)
    if mime_types is None:
        return True
    if not isinstance(mime_types, list | tuple):
        return False
    return _PDF_MIME_TYPE in mime_types


def _task_cli_provider_reference_descriptor(
    *,
    kind: str,
    provider: str,
    reference: str,
) -> dict[str, object]:
    return {
        "source_kind": "provider_reference",
        "reference": reference,
        "provider_reference": {
            "kind": kind,
            "provider": provider,
            "reference": reference,
        },
    }


def _apply_task_cli_file_hints(
    specs: tuple[_TaskCliFileSpec, ...],
    args: Namespace,
) -> tuple[_TaskCliFileSpec, ...]:
    hints = _task_cli_file_hints(args)
    if not hints:
        return specs
    spec_fields = {spec.field for spec in specs}
    if not spec_fields.issuperset(hints):
        raise _task_cli_input_error(
            "Task file options require a matching file descriptor."
        )
    hinted_specs: list[_TaskCliFileSpec] = []
    for spec in specs:
        descriptor = _mutable_json_object(spec.descriptor)
        hint = hints.get(spec.field, {})
        for key in ("mime_type", "role", "size_bytes", "sha256"):
            if key not in hint:
                continue
            _set_task_cli_file_hint(descriptor, key, hint[key])
        conversions = hint.get("conversions")
        if conversions is not None:
            existing = descriptor.get("conversions", ())
            if not isinstance(existing, list | tuple):
                raise _task_cli_input_error(
                    "Task file descriptor conversions are invalid."
                )
            existing_values = cast(
                list[object],
                _plain_task_cli_value(existing),
            )
            descriptor["conversions"] = [
                *existing_values,
                *cast(list[object], conversions),
            ]
        hinted_specs.append(
            _TaskCliFileSpec(field=spec.field, descriptor=descriptor)
        )
    return tuple(hinted_specs)


def _task_cli_file_hints(
    args: Namespace,
) -> dict[str, dict[str, object]]:
    hints: dict[str, dict[str, object]] = {}
    for attr, key in (
        ("task_file_mime_types", "mime_type"),
        ("task_file_roles", "role"),
        ("task_file_sha256", "sha256"),
    ):
        for field, value in _task_cli_field_assignments(
            args,
            attr=attr,
            error_message="Task file option is invalid.",
        ):
            _set_task_cli_hint(hints, field, key, value)
    for field, value in _task_cli_field_assignments(
        args,
        attr="task_file_sizes",
        error_message="Task file size hint is invalid.",
    ):
        try:
            parsed_size = int(value, 10)
        except ValueError as exc:
            raise _task_cli_input_error(
                "Task file size hint is invalid."
            ) from exc
        _set_task_cli_hint(hints, field, "size_bytes", parsed_size)
    for field, value in _task_cli_field_assignments(
        args,
        attr="task_file_conversions",
        error_message="Task file conversion hint is invalid.",
    ):
        conversions = cast(
            list[object],
            hints.setdefault(field, {}).setdefault("conversions", []),
        )
        conversions.append(_task_cli_conversion_hint(value))
    return hints


def _task_cli_field_assignments(
    args: Namespace,
    *,
    attr: str,
    error_message: str,
) -> tuple[tuple[str, str], ...]:
    values = getattr(args, attr, ()) or ()
    assert isinstance(values, list | tuple)
    fields: list[tuple[str, str]] = []
    for value in values:
        assert isinstance(value, str)
        field, separator, raw_value = value.partition("=")
        if (
            separator != "="
            or not _is_task_cli_field_path(field)
            or not raw_value.strip()
        ):
            raise _task_cli_input_error(error_message)
        fields.append((field, raw_value))
    return tuple(fields)


def _set_task_cli_hint(
    hints: dict[str, dict[str, object]],
    field: str,
    key: str,
    value: object,
) -> None:
    field_hints = hints.setdefault(field, {})
    if key in field_hints:
        raise _task_cli_input_error("Task file option is duplicated.")
    field_hints[key] = value


def _set_task_cli_file_hint(
    descriptor: dict[str, object],
    key: str,
    value: object,
) -> None:
    existing = descriptor.get(key)
    if existing is not None and existing != value:
        raise _task_cli_input_error("Task file descriptor option conflicts.")
    descriptor[key] = value


def _task_cli_conversion_hint(value: str) -> Mapping[str, object]:
    name, separator, raw_options = value.partition(":")
    if not name.strip():
        raise _task_cli_input_error("Task file conversion hint is invalid.")
    if separator != ":":
        return {"name": name}
    options = _parse_task_cli_json_value(raw_options)
    if not isinstance(options, Mapping):
        raise _task_cli_input_error(
            "Task file conversion options are invalid."
        )
    return {"name": name, "options": _mutable_json_object(options)}


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
        hint=(
            "Use --input, --input-json, --pdf, --input-name, "
            "--file name=path, or --file-descriptor name=json."
        ),
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
    return value.get("source_kind") in {
        "artifact",
        "inline_bytes",
        "local_path",
        "provider_reference",
        "remote_url",
    } and isinstance(value.get("reference"), str)


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


def _task_cli_contains_local_file(value: object) -> bool:
    if isinstance(value, Mapping):
        if value.get("source_kind") == "local_path" and isinstance(
            value.get("reference"), str
        ):
            return True
        return any(
            _task_cli_contains_local_file(item) for item in value.values()
        )
    if isinstance(value, list | tuple):
        return any(_task_cli_contains_local_file(item) for item in value)
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
