"""Run bounded acceptance roles in independently started interpreters."""

from asyncio import run
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from io import StringIO
from json import dumps, loads
from logging import getLogger
from os import _exit, environ
from pathlib import Path
from sys import argv
from unittest.mock import AsyncMock, patch

from rich.console import Console
from task_deployment_helpers import DeploymentProviderLoader
from trigger_acceptance_resume import execute_resume
from trigger_acceptance_uniqueness import execute_competitor, execute_failure
from trigger_cli_helpers import write_application, write_flow_application

from avalan.cli.__main__ import CLI
from avalan.cli.commands.task import _task_usage, _task_worker
from avalan.cli.commands.trigger import run_trigger_command
from avalan.cli.trigger_runtime import encrypted_input_store
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.pgsql import (
    PgsqlUnitOfWork,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
)
from avalan.task.queues.pgsql import PgsqlTaskQueue
from avalan.task.stores.pgsql import PgsqlTaskStore
from avalan.trigger.admission import (
    PreparedTriggerAdmission,
    TriggerAdmissionResult,
)
from avalan.trigger.codec import decode_record, encode_record
from avalan.trigger.observability import ObservedTriggerScheduler
from avalan.trigger.plan import plan_admission
from avalan.trigger.records import (
    OwnerScopeId,
    TriggerDefinition,
    TriggerSnapshot,
    TriggerState,
)
from avalan.trigger.scheduler_types import TriggerProcessResult
from avalan.trigger.stores.pgsql import PgsqlTriggerStore, decision_time
from avalan.trigger.stores.pgsql_admission import PgsqlTriggerAdmissionStore


async def execute(action: str, root: Path) -> None:
    """Execute a CLI role with deterministic provider transport."""
    if action.startswith("resume-") or action in ("compete", "failure-tick"):
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=environ["AVALAN_TASK_STORE_DSN"],
                schema=environ["AVALAN_TASK_STORE_SCHEMA"],
            )
        )
        await database.open()
        try:
            if action == "compete":
                await execute_competitor(root, database)
            elif action == "failure-tick":
                await execute_failure(root, database)
            else:
                await execute_resume(action, root, database)
        finally:
            await database.aclose()
        return
    cli = CLI(getLogger(__name__))
    hub = HuggingfaceHub("unused", str(root / "cache"), getLogger(__name__))
    host = [
        "--deployment-root",
        str(root / "catalog"),
        "--raw-storage-allowed",
    ]

    async def command(arguments: list[str]) -> None:
        output = StringIO()
        await run_trigger_command(
            cli._parser.parse_args(["trigger", *arguments]),
            Console(file=output, width=10000),
            hub,
            getLogger(__name__),
        )
        result = loads(output.getvalue())
        assert result["ok"], result
        print(dumps(result), flush=True)

    if action == "register":
        application = root / "application"
        write_application(application)
        task = application / "task.toml"
        task.write_text(
            task.read_text().replace(
                'input="encrypt"', 'input="encrypt"\noutput="encrypt"'
            )
            + '\n[retry]\nmax_attempts=2\nbackoff="linear"\n'
        )
        (application / "hmac.toml").write_text(
            task.read_text().replace(
                'idempotency="none"', 'idempotency="input_hash"'
            )
        )
        due = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(
            minutes=5
        )
        schedules = {
            "interval": (
                f'type="interval"\nevery_seconds=86400\nstart_at="{due.isoformat()}"'
            ),
            "cron": (
                f'type="cron"\nexpression="{due.minute} {due.hour} * *'
                ' *"\ntimezone="UTC"'
            ),
            "at": f'type="at"\nat="{due.isoformat()}"',
        }
        for name, schedule in schedules.items():
            path = application / f"{name}.toml"
            task_ref = "hmac.toml" if name == "interval" else "task.toml"
            path.write_text(
                f'[trigger]\nschema_version=1\nname="{name}"\n[task]\nref="{task_ref}"\n[input]\nvalue="frozen'
                f' private input"\n[schedule]\n{schedule}\n'
            )
            # Only initial registration is historical. Every child scheduler
            # uses the real database decision clock inside its admission lock.
            with patch(
                "avalan.trigger.apply.decision_time",
                AsyncMock(return_value=due - timedelta(minutes=1)),
            ):
                await command(["apply", str(path), *host])
    elif action == "failure-register":
        due = datetime.now(UTC) - timedelta(minutes=5)
        for name in ("failure-bad", "failure-good"):
            path = root / "application" / f"{name}.toml"
            path.write_text(
                f'[trigger]\nschema_version=1\nname="{name}"\n[task]\nref="task.toml"\n[input]\nvalue="frozen'
                " private"
                f' input"\n[schedule]\ntype="at"\nat="{due.isoformat()}"\n'
            )
            with patch(
                "avalan.trigger.apply.decision_time",
                AsyncMock(return_value=due - timedelta(minutes=1)),
            ):
                await command(["apply", str(path), *host])
    elif action == "catchup-register":
        for policy in ("skip", "latest", "all"):
            path = root / "application" / f"backlog-{policy}.toml"
            path.write_text(
                f'[trigger]\nschema_version=1\nname="backlog-{policy}"\n[task]\nref="task.toml"\n[input]\nvalue="frozen'
                " private"
                ' input"\n[schedule]\ntype="interval"\n'
                f'every_seconds=60\n[policy]\nmisfire="{policy}"\noverlap="allow"\n'
            )
            with patch(
                "avalan.trigger.apply.decision_time",
                AsyncMock(
                    return_value=datetime.now(UTC)
                    - timedelta(days=1, minutes=5)
                ),
            ):
                await command(["apply", str(path), *host])
    elif action == "catchup-tick":
        await command(
            [
                "serve",
                "--once",
                "--admissions-per-trigger",
                "2",
                "--decisions-per-trigger",
                "3",
                *host,
            ]
        )
    elif action == "catchup-inspect":
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=environ["AVALAN_TASK_STORE_DSN"],
                schema=environ["AVALAN_TASK_STORE_SCHEMA"],
            )
        )
        await database.open()
        try:
            store = PgsqlTriggerStore(
                database, OwnerScopeId(value="acceptance-owner")
            )
            admitted = []
            for policy in ("skip", "latest", "all"):
                name = f"backlog-{policy}"
                occurrences = await store.occurrences(name)
                admitted.append(
                    sum(
                        item.disposition.value == "admitted"
                        for item in occurrences.items
                    )
                )
                coverage = await store.coverage(name)
                dispositions = {
                    item.disposition.value for item in coverage.items
                }
                if policy == "all":
                    assert "expired" in dispositions
                if policy == "skip":
                    assert "skipped_misfire" in dispositions
                if policy == "latest":
                    assert "coalesced" in dispositions
                current = await store.inspect(name)
                assert current is not None
                if policy == "all":
                    assert (
                        current.state.next_at is not None
                        and current.state.next_at <= datetime.now(UTC)
                    )
                    anchor, cursor, revision = (
                        current.definition.schedule,
                        current.state.next_at,
                        current.state.revision,
                    )
                    paused = await store.set_enabled(
                        name,
                        enabled=False,
                        expected_generation=current.state.generation,
                    )
                    resumed = await store.set_enabled(
                        name,
                        enabled=True,
                        expected_generation=paused.state.generation,
                    )
                    assert (
                        resumed.definition.schedule == anchor
                        and resumed.state.next_at == cursor
                        and resumed.state.revision == revision
                    )
                current = await store.inspect(name)
                assert current is not None
                await store.set_enabled(
                    name,
                    enabled=False,
                    expected_generation=current.state.generation,
                )
            assert admitted == [0, 1, 2], admitted
            print(
                dumps(
                    {
                        "skip": admitted[0],
                        "latest": admitted[1],
                        "all": admitted[2],
                        "compressed": True,
                        "cursor_retained": True,
                    }
                ),
                flush=True,
            )
        finally:
            await database.aclose()
    elif action in ("control-prepare", "control-pause"):
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=environ["AVALAN_TASK_STORE_DSN"],
                schema=environ["AVALAN_TASK_STORE_SCHEMA"],
            )
        )
        await database.open()
        try:
            store = PgsqlTriggerStore(
                database, OwnerScopeId(value="acceptance-owner")
            )
            run = await PgsqlTaskStore(database).get_run(
                (root / "control-run.txt").read_text()
            )
            assert run.request.trigger is not None
            for name in ("interval", "cron", "at", "cancel"):
                snapshot = await store.inspect(name)
                if (
                    snapshot is not None
                    and snapshot.definition.trigger_id
                    == run.request.trigger.trigger_id
                ):
                    break
            else:
                raise AssertionError("control run has no registered trigger")
            if action == "control-prepare":
                path = root / "application" / f"{name}.toml"
                path.write_text(
                    path.read_text().split("[schedule]")[0]
                    + '[schedule]\ntype="interval"\nevery_seconds=1\n'
                    '[policy]\nmisfire="all"\noverlap="skip"\n'
                )
                await command(
                    [
                        "apply",
                        str(path),
                        "--expected-generation",
                        str(snapshot.state.generation),
                        *host,
                    ]
                )
                current = await store.inspect(name)
                assert (
                    current is not None and current.state.next_at is not None
                )
                for _ in range(10000):
                    async with store.transaction() as unit:
                        now = await decision_time(unit)
                    if now >= current.state.next_at:
                        break
                else:
                    raise AssertionError("control schedule did not become due")
            else:
                await command(
                    [
                        "pause",
                        name,
                        "--expected-generation",
                        str(snapshot.state.generation),
                    ]
                )
        finally:
            await database.aclose()
    elif action == "cancel-register":
        path = root / "application/cancel.toml"
        due = datetime.now(UTC) - timedelta(minutes=5)
        path.write_text(
            '[trigger]\nschema_version=1\nname="cancel"\n[task]\nref="task.toml"\n[input]\nvalue="frozen'
            f' private input"\n[schedule]\ntype="at"\nat="{due.isoformat()}"\n'
        )
        with patch(
            "avalan.trigger.apply.decision_time",
            AsyncMock(return_value=due - timedelta(minutes=1)),
        ):
            await command(["apply", str(path), *host])
    elif action == "files-register":
        application = root / "files"
        write_flow_application(application)
        task = application / "task.toml"
        task.write_text(
            task.read_text()
            .replace('type="object"\nschema={type="object"}', 'type="file"')
            .replace('files="drop"', 'files="encrypt"')
        )
        flow = application / "flow.toml"
        flow.write_text(
            flow.read_text().replace(
                'name="scheduled"\ntype="string"',
                'name="scheduled"\ntype="file"',
            )
        )
        source = application / "input.txt"
        source.write_bytes(b"durable process artifact bytes")
        path = application / "trigger.toml"
        path.write_text(
            '[trigger]\nschema_version=1\nname="durable-file"\n[task]\nref="task.toml"\n[input]\nvalue={source_kind="local_path",'
            ' reference="input.txt",'
            ' mime_type="text/plain"}\n[schedule]\ntype="interval"\n'
            "every_seconds=86400\n"
        )
        with patch(
            "avalan.trigger.apply.decision_time",
            AsyncMock(
                return_value=datetime.now(UTC) - timedelta(days=1, minutes=5)
            ),
        ):
            await command(["apply", str(path), *host])
        source.unlink()
    elif action == "files-crash":
        original = PgsqlTriggerAdmissionStore.admit

        async def crash(
            admission: PgsqlTriggerAdmissionStore,
            prepared: PreparedTriggerAdmission,
        ) -> TriggerAdmissionResult:
            (root / "file-recovery.json").write_text(
                dumps(
                    {
                        "definition": encode_record(
                            prepared.plan.snapshot.definition
                        ),
                        "state": encode_record(prepared.plan.snapshot.state),
                        "prepared_at": prepared.plan.prepared_at.isoformat(),
                    }
                )
            )
            transaction = admission.store.transaction

            @asynccontextmanager
            async def lost_ack() -> AsyncIterator[PgsqlUnitOfWork]:
                async with transaction() as unit:
                    yield unit
                _exit(23)

            with patch.object(admission.store, "transaction", lost_ack):
                return await original(admission, prepared)

        with patch.object(PgsqlTriggerAdmissionStore, "admit", crash):
            await command(["serve", "--once", *host])
        raise AssertionError("commit termination hook was not reached")
    elif action == "files-recover":
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=environ["AVALAN_TASK_STORE_DSN"],
                schema=environ["AVALAN_TASK_STORE_SCHEMA"],
            )
        )
        await database.open()
        try:
            evidence = loads((root / "file-recovery.json").read_text())
            definition, state = (
                decode_record(evidence["definition"]),
                decode_record(evidence["state"]),
            )
            assert isinstance(definition, TriggerDefinition) and isinstance(
                state, TriggerState
            )
            store = PgsqlTriggerStore(
                database, OwnerScopeId(value="acceptance-owner")
            )
            recovered = await PgsqlTriggerAdmissionStore(
                store, PgsqlTaskQueue(database)
            ).recover(
                plan_admission(
                    TriggerSnapshot(definition=definition, state=state),
                    datetime.fromisoformat(evidence["prepared_at"]),
                )
            )
            assert recovered.outcome.value == "committed"
            occurrences = await store.occurrences("durable-file")
            assert len(occurrences.items) == 1
            run_id = occurrences.items[0].run_id
            assert run_id is not None
            records = await PgsqlTaskStore(database).list_artifacts(run_id)
            assert len(records) == 1
            _, artifacts = encrypted_input_store(
                cli._parser.parse_args(["trigger", "serve", "--once", *host]),
                database,
            )
            with await artifacts.open(records[0].ref) as stream:
                assert stream.read() == b"durable process artifact bytes"
            current = await store.inspect("durable-file")
            assert (
                current is not None and current.state.next_at != state.next_at
            )
            print(
                dumps(
                    {
                        "recovered": "committed",
                        "artifacts": 1,
                        "cursor_advanced": True,
                    }
                ),
                flush=True,
            )
        finally:
            await database.aclose()
    elif action == "usage":
        for run_id in (root / "run-ids.txt").read_text().splitlines():
            output = StringIO()
            assert await _task_usage(
                cli._parser.parse_args(["task", "usage", run_id]),
                Console(file=output, width=10000),
            )
            line = output.getvalue().strip()
            assert line.startswith("usage ")
            print(dumps(loads(line.removeprefix("usage "))), flush=True)
    elif action == "overlap-tick":
        original_tick = ObservedTriggerScheduler.process_once

        async def clean_tick(
            scheduler: ObservedTriggerScheduler,
        ) -> TriggerProcessResult:
            tick = await original_tick(scheduler)
            assert not tick.errors
            assert not tick.preparation_failures
            assert not tick.unresolved
            assert tick.pending_operations == 0
            return tick

        with patch.object(
            ObservedTriggerScheduler, "process_once", clean_tick
        ):
            await command(["serve", "--once", *host])
    elif action == "tick":
        await command(["serve", "--once", *host])
    elif action in ("worker", "worker-retry"):
        provider = DeploymentProviderLoader(
            response_text="durable native output",
            responses=(
                (OSError("deterministic provider retry"),)
                if action == "worker-retry"
                else ()
            ),
        )
        output = StringIO()
        with patch(
            "avalan.agent.loader.OrchestratorLoader.from_file",
            provider.from_file,
        ):
            assert await _task_worker(
                cli._parser.parse_args(["task", "worker", "--once", *host]),
                Console(file=output),
                hub=hub,
                logger=getLogger(__name__),
            )
        assert provider.inputs == ["frozen private input"]
        assert all(str(root / "catalog") in path for path in provider.paths)
        print(dumps({"executed": 1, "retained": True}), flush=True)
    else:
        assert action == "inspect"
        for name in ("interval", "cron", "at"):
            await command(["inspect", name])
            await command(["events", name])


if __name__ == "__main__":
    run(execute(argv[1], Path(argv[2])))
