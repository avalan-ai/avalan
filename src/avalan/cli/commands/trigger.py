"""Compose trigger commands with the canonical durable SDK boundaries."""

from ...agent.loader import OrchestratorLoader
from ...model.hubs.huggingface import HuggingfaceHub
from ...pgsql import PsycopgAsyncDatabase, PsycopgPoolSettings
from ...task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from ...task.client import TaskClient
from ...task.deployment import ExecutionDeploymentError
from ...task.encryption import (
    TaskEncryptionError,
)
from ...task.queues.pgsql import PgsqlTaskQueue
from ...task.stores.pgsql import PgsqlTaskStore
from ...task.validation import TaskValidationError
from ...trigger.admission import TriggerCommitOutcome
from ...trigger.apply import (
    TriggerApplyCancelledError,
    TriggerRegistrationService,
)
from ...trigger.codec import policy_payload, record_payload, schedule_payload
from ...trigger.definition import TriggerConfiguration, timestamp
from ...trigger.error import TriggerError
from ...trigger.loader import TriggerLoader
from ...trigger.observability import (
    ObservedTriggerScheduler,
    TriggerMetrics,
    replay_events,
)
from ...trigger.preparation import (
    TriggerPreparationCancelledError,
    TriggerPreparationFailure,
    TriggerPreparationService,
)
from ...trigger.records import OwnerScopeId, TriggerSnapshot
from ...trigger.schedule import preview
from ...trigger.scheduler import TriggerSchedulerCancelledError
from ...trigger.scheduler_types import TriggerSchedulerSettings
from ...trigger.store import HistoryCursor
from ...trigger.stores.pgsql import PgsqlTriggerStore
from ...trigger.stores.pgsql_admission import PgsqlTriggerAdmissionStore
from ..task_privacy import task_hmac_provider
from ..task_store import TaskStoreConfigurationError, task_store_configuration
from ..trigger_host import TriggerCliHost
from ..trigger_runtime import encrypted_input_store

from argparse import Namespace
from asyncio import Event, get_running_loop
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from json import dumps
from logging import Logger
from os import environ
from pathlib import Path
from signal import SIGINT, SIGTERM
from uuid import uuid4

from rich.console import Console


def _snapshot(value: TriggerSnapshot) -> dict[str, object]:
    definition, state = value.definition, value.state
    return {
        "name": definition.name,
        "trigger_id": definition.trigger_id,
        "revision": state.revision,
        "generation": state.generation,
        "status": state.status.value,
        "next_at": state.next_at.isoformat() if state.next_at else None,
        "last_processed_at": state.last_processed_at.isoformat(),
        "retry_after": (
            state.retry_after.isoformat() if state.retry_after else None
        ),
        "failure_count": state.failure_count,
        "error_code": (
            state.last_error_code.value if state.last_error_code else None
        ),
        "task_definition_id": definition.task_definition_id,
        "execution_deployment_id": definition.execution_deployment_id,
        "schedule": schedule_payload(definition.schedule),
        "policy": policy_payload(definition.policy),
    }


def _owner(args: Namespace) -> OwnerScopeId:
    value = getattr(args, "owner_scope", None) or environ.get(
        "AVALAN_TASK_OWNER_SCOPE"
    )
    if not isinstance(value, str) or not value.strip():
        raise TaskStoreConfigurationError("owner_scope")
    return OwnerScopeId(value=value)


def _scheduler_settings(args: Namespace) -> TriggerSchedulerSettings:
    return TriggerSchedulerSettings(
        **{
            name: getattr(args, name)
            for name in (
                "discovery_limit",
                "decisions_per_tick",
                "decisions_per_trigger",
                "admissions_per_tick",
                "admissions_per_trigger",
                "candidate_evaluations",
                "poll_interval_seconds",
                "tick_timeout_seconds",
                "shutdown_timeout_seconds",
                "orphan_grace_seconds",
            )
        }
    )


async def _configuration(
    args: Namespace, host: TriggerCliHost
) -> tuple[TriggerConfiguration, Path]:
    path = Path(args.configuration).resolve(strict=True)
    root = Path(args.root).resolve(strict=True) if args.root else path.parent

    async def validate(
        task_path: Path, configuration: TriggerConfiguration
    ) -> None:
        await TriggerCliHost(host.root, host.loader).binding(
            path.parent, configuration.task_ref
        )

    configuration = await TriggerLoader(
        roots=(root,), task_validator=validate
    ).load(path)
    return configuration, path.parent


def _preparation_error(error: TriggerPreparationFailure) -> dict[str, object]:
    staging = error.staging or (
        error.registration.staging if error.registration else None
    )
    return {
        "ok": False,
        "code": error.code.value,
        "path": error.path,
        "outcome": error.admission_outcome.value,
        "resources_uncertain": error.resources_uncertain,
        "recovery_id": staging.recovery_id if staging else None,
        "submission_ids": [value.submission_id for value in error.submissions],
    }


def _emit(console: Console, result: dict[str, object]) -> None:
    console.print(
        dumps(result, sort_keys=True, separators=(",", ":")),
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


async def run_trigger_command(
    args: Namespace, console: Console, hub: HuggingfaceHub, logger: Logger
) -> bool:
    """Emit only safe bounded results; never include registered input."""
    try:
        result = await _run(args, hub, logger)
    except TriggerApplyCancelledError as error:
        _emit(
            console,
            {
                "ok": False,
                "code": "trigger.cancelled",
                "outcome": error.result.outcome.value,
                "recovery_id": error.result.prepared.staging.recovery_id,
                "cleanup_pending": error.result.cleanup_pending,
                "trigger": (
                    _snapshot(error.result.snapshot)
                    if error.result.snapshot is not None
                    else None
                ),
            },
        )
        raise
    except TriggerPreparationCancelledError as error:
        _emit(console, _preparation_error(error.preparation))
        raise
    except TriggerSchedulerCancelledError as error:
        _emit(
            console,
            {
                "ok": False,
                "code": "trigger.cancelled",
                "shutdown_settled": error.result.settled,
                "pending_operations": error.result.pending_operations,
            },
        )
        raise
    except TriggerPreparationFailure as error:
        result = _preparation_error(error)
    except TriggerError as error:
        result = {"ok": False, "code": error.code.value, "path": error.path}
    except TaskStoreConfigurationError as error:
        result = {
            "ok": False,
            "code": "task.store_configuration",
            "path": error.field_name,
        }
    except ExecutionDeploymentError:
        result = {
            "ok": False,
            "code": "trigger.deployment_mismatch",
            "path": "deployment",
        }
    except TaskValidationError as error:
        result = {
            "ok": False,
            "code": "trigger.invalid_task",
            "issues": [
                {"code": issue.code, "path": issue.path}
                for issue in error.issues
            ],
        }
    except TaskEncryptionError:
        result = {
            "ok": False,
            "code": "trigger.encryption_unavailable",
            "path": "encryption",
        }
    except Exception:
        result = {
            "ok": False,
            "code": "trigger.operation_failed",
            "path": "operation",
        }
    _emit(console, result)
    return result.get("ok") is True


async def _run(
    args: Namespace, hub: HuggingfaceHub, logger: Logger
) -> dict[str, object]:
    action = args.trigger_command
    stack = AsyncExitStack()
    scheduler: ObservedTriggerScheduler | None = None
    try:
        if action in {"validate", "preview", "apply", "serve"}:
            loader = OrchestratorLoader(
                hub=hub, logger=logger, participant_id=uuid4(), stack=stack
            )
            host = TriggerCliHost(
                Path(getattr(args, "deployment_root", ".")), loader
            )
        if action in {"validate", "preview", "apply"}:
            configuration, source = await _configuration(args, host)
            if action == "validate":
                return {
                    "ok": True,
                    "name": configuration.name,
                    "registered": False,
                }
            if action == "preview":
                reference = (
                    timestamp(
                        args.reference_time,
                        "reference_time",
                    )
                    if args.reference_time
                    else datetime.now(UTC)
                )
                value = preview(
                    configuration.schedule,
                    reference_time=reference,
                    count=args.count,
                )
                return {
                    "ok": True,
                    "name": configuration.name,
                    "registered": False,
                    "timezone": value.timezone,
                    "reference_time": reference.isoformat(),
                    "effective_anchor": (
                        value.effective_anchor.isoformat()
                        if value.effective_anchor
                        else None
                    ),
                    "assumed_anchor": value.assumed_anchor,
                    "occurrences": [
                        {"utc": instant.isoformat(), "local": local}
                        for instant, local in zip(
                            value.occurrences, value.local_times, strict=True
                        )
                    ],
                    "schedule": schedule_payload(configuration.schedule),
                    "policy": policy_payload(configuration.policy),
                }
        settings = task_store_configuration(args)
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=settings.require_dsn(), schema=settings.schema
            )
        )
        await database.open()
        stack.push_async_callback(database.aclose)
        store = PgsqlTriggerStore(database, _owner(args))
        await store.preflight()
        if action in {"apply", "serve"}:
            cipher, artifacts = encrypted_input_store(args, database)
            queue = PgsqlTaskQueue(database)
            tasks = PgsqlTaskStore(database)
            if action == "apply":
                binding = await host.retain(source, configuration.task_ref)
            else:
                await host.restore()
                bindings = tuple(host.bindings.values())
                if not bindings:
                    raise ExecutionDeploymentError("cli.catalog_empty")
                binding = bindings[0]
            client = TaskClient(
                tasks,
                target=binding.target,
                queue=queue,
                owner_scope=store.owner.value,
                execution_deployment_id=binding.manifest.execution_deployment_id,
                execution_deployments=host.catalog,
                hmac_provider=task_hmac_provider(),
                encryption_provider=cipher,
                raw_storage_allowed=True,
                artifact_store=artifacts,
                execution_roots=(binding.application_root,),
                input_roots=(source,) if action == "apply" else (),
            )
            preparation = TriggerPreparationService(
                client,
                PgsqlTriggerAdmissionStore(store, queue),
                PgsqlArtifactOwnership(database),
                cipher,
            )
            if action == "apply":
                prepared = await preparation.prepare_registration(
                    configuration, binding.definition
                )
                result = await TriggerRegistrationService(preparation).apply(
                    prepared, expected_generation=args.expected_generation
                )
                return {
                    "ok": result.outcome is TriggerCommitOutcome.COMMITTED,
                    "outcome": result.outcome.value,
                    "request_id": prepared.staging.recovery_id,
                    "code": (
                        result.error_code.value if result.error_code else None
                    ),
                    "trigger": (
                        _snapshot(result.snapshot) if result.snapshot else None
                    ),
                }
            scheduler = ObservedTriggerScheduler(
                preparation,
                settings=_scheduler_settings(args),
                owned_resources=(stack,),
            )
            if args.metrics_file:
                scheduler.sink = TriggerMetrics(Path(args.metrics_file))
            if args.once:
                tick = await scheduler.process_once()
                shutdown = await scheduler.shutdown()
                return {
                    "ok": shutdown.settled,
                    "admitted": tick.admitted,
                    "admission_retries": tick.admission_retries,
                    "skipped": tick.skipped,
                    "remaining_work": tick.remaining_work,
                    "pending_operations": tick.pending_operations,
                    "conflicts": tick.conflicts,
                    "errors": [
                        {"code": e.code.value, "name": e.trigger_name}
                        for e in tick.errors
                    ],
                    "occurrences": [
                        record_payload(item) for item in tick.occurrences
                    ],
                    "coverage": [record_payload(item) for item in tick.ranges],
                    "sink_failures": scheduler.sink_failures,
                    "shutdown_settled": shutdown.settled,
                }
            stop = Event()
            loop = get_running_loop()
            for signal in (SIGINT, SIGTERM):
                loop.add_signal_handler(signal, stop.set)
            try:
                shutdown = await scheduler.serve(stop=stop)
            finally:
                for signal in (SIGINT, SIGTERM):
                    loop.remove_signal_handler(signal)
            return {
                "ok": shutdown.settled,
                "shutdown_settled": shutdown.settled,
                "pending_operations": shutdown.pending_operations,
                "sink_failures": scheduler.sink_failures,
            }
        if action == "list":
            page = await store.list(
                cursor=HistoryCursor(offset=args.cursor), limit=args.limit
            )
            return {
                "ok": True,
                "items": [_snapshot(item) for item in page.items],
                "next_cursor": (
                    page.next_cursor.offset if page.next_cursor else None
                ),
            }
        if action == "inspect":
            snapshot = await store.inspect(args.name)
            recent = (
                await store.occurrences(args.name, limit=5, newest_first=True)
                if snapshot
                else None
            )
            runs = []
            for occurrence in recent.items if recent else ():
                run = (
                    await PgsqlTaskStore(database).get_run(occurrence.run_id)
                    if occurrence.run_id
                    else None
                )
                runs.append(
                    {
                        "occurrence": record_payload(occurrence),
                        "task_state": run.state.value if run else None,
                    }
                )
            return {
                "ok": snapshot is not None,
                "trigger": _snapshot(snapshot) if snapshot else None,
                "recent_occurrences": runs,
            }
        if action in {"pause", "resume"}:
            snapshot = await store.set_enabled(
                args.name,
                enabled=action == "resume",
                expected_generation=args.expected_generation,
            )
            return {"ok": True, "trigger": _snapshot(snapshot)}
        cursor = HistoryCursor(offset=args.cursor)
        if action == "events":
            events = await store.events(
                args.name, cursor=cursor, limit=args.limit
            )
            sink = (
                await replay_events(events.items, Path(args.event_file))
                if args.event_file
                else None
            )
            return {
                "ok": True,
                "items": [record_payload(item) for item in events.items],
                "next_cursor": (
                    events.next_cursor.offset if events.next_cursor else None
                ),
                "sink_failed": sink.failed if sink else 0,
            }
        if action == "occurrences":
            decisions = (
                await store.coverage(
                    args.name, cursor=cursor, limit=args.limit
                )
                if args.coverage
                else await store.occurrences(
                    args.name, cursor=cursor, limit=args.limit
                )
            )
            return {
                "ok": True,
                "items": [record_payload(item) for item in decisions.items],
                "next_cursor": (
                    decisions.next_cursor.offset
                    if decisions.next_cursor
                    else None
                ),
            }
        raise TaskStoreConfigurationError("command")
    finally:
        if scheduler is None:
            await stack.aclose()
        else:
            await scheduler.shutdown()
