"""Exercise CLI composition over real PostgreSQL and production encryption."""

from asyncio import Event, get_running_loop
from base64 import b64encode
from datetime import UTC, datetime, timedelta
from io import StringIO
from json import loads
from logging import getLogger
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from pgsql_harness import (
    drop_task_pgsql_schema,
    isolated_task_pgsql_schema,
    real_task_pgsql_dsn,
    task_pgsql_psycopg_dsn,
)
from rich.console import Console
from task_deployment_helpers import (
    DeploymentProviderLoader,
    DeploymentProviderResponse,
)
from trigger_cli_helpers import write_application, write_flow_application

from avalan.cli.__main__ import CLI
from avalan.cli.commands.task import _task_retention_sweep, _task_worker
from avalan.cli.commands.trigger import run_trigger_command
from avalan.cli.trigger_runtime import encrypted_input_store
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.pgsql import PsycopgAsyncDatabase, PsycopgPoolSettings
from avalan.task.state import TaskRunState
from avalan.task.stores.pgsql import (
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
    task_pgsql_upgrade,
)
from avalan.trigger.observability import ObservedTriggerScheduler
from avalan.trigger.scheduler_types import TriggerShutdownResult


class TriggerCliPgsqlTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        dsn = real_task_pgsql_dsn()
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        self.dsn = dsn
        self.schema = isolated_task_pgsql_schema("avalan_trigger_cli")
        task_pgsql_upgrade(
            PgsqlTaskMigrationSettings(url=dsn, schema=self.schema)
        )
        self.database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=task_pgsql_psycopg_dsn(dsn), schema=self.schema
            )
        )
        await self.database.open()
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        write_application(self.root / "application")
        self.trigger = self.root / "application/daily.toml"
        self.trigger.write_text("""[trigger]
schema_version=1
name="daily"
[task]
ref="task.toml"
[input]
value="private scheduled input"
[schedule]
type="interval"
every_seconds=86400
""")
        self.cli = CLI(getLogger(__name__))
        self.hub = HuggingfaceHub(
            "unused", str(self.root / "cache"), getLogger(__name__)
        )
        self.environment = patch.dict(
            environ,
            {
                "AVALAN_TASK_STORE_DSN": task_pgsql_psycopg_dsn(dsn),
                "AVALAN_TASK_STORE_SCHEMA": self.schema,
                "AVALAN_TASK_OWNER_SCOPE": "cli-owner",
                "AVALAN_TASK_ENCRYPTION_KEY_ID": "cli-test",
                "AVALAN_TASK_ENCRYPTION_KEY_B64": (
                    b64encode(b"x" * 32).decode()
                ),
            },
        )
        self.environment.start()

    async def asyncTearDown(self) -> None:
        if hasattr(self, "database"):
            self.environment.stop()
            self.temporary.cleanup()
            await self.database.aclose()
            await drop_task_pgsql_schema(self.dsn, self.schema)

    async def command(self, action: str, *options: str) -> dict[str, object]:
        args = self.cli._parser.parse_args(["trigger", action, *options])
        output = StringIO()
        await run_trigger_command(
            args,
            Console(file=output, width=10000),
            self.hub,
            getLogger(__name__),
        )
        text = output.getvalue()
        self.assertNotIn("private scheduled input", text)
        self.assertNotIn("cli-test", text)
        result = loads(text)
        assert isinstance(result, dict)
        return result

    def host_options(self) -> list[str]:
        return [
            "--deployment-root",
            str(self.root / "catalog"),
            "--raw-storage-allowed",
        ]

    async def test_apply_once_restart_worker_and_management(self) -> None:
        # Controlled historical registration gives a due daily slot without
        # sleeps. Scheduler admission retains its actual locked DB clock.
        with patch(
            "avalan.trigger.apply.decision_time",
            AsyncMock(
                return_value=datetime.now(UTC) - timedelta(days=1, minutes=5)
            ),
        ):
            applied = await self.command(
                "apply", str(self.trigger), *self.host_options()
            )
        self.assertTrue(applied["ok"], applied)
        initial = await self.command("inspect", "daily")
        self.assertTrue(initial["ok"])
        metrics = self.root / "metrics.prom"
        tick = await self.command(
            "serve",
            "--once",
            *self.host_options(),
            "--metrics-file",
            str(metrics),
        )
        self.assertTrue(tick["ok"], tick)
        self.assertEqual(tick["admitted"], 1)
        self.assertIn(
            "avalan_trigger_dispatch_lag_seconds_count 1", metrics.read_text()
        )
        decisions = await self.command("occurrences", "daily")
        assert isinstance(decisions["items"], list)
        self.assertEqual(len(decisions["items"]), 1)
        occurrence = decisions["items"][0]
        assert isinstance(occurrence, dict)
        payload = occurrence["payload"]
        assert isinstance(payload, dict)
        run_id = payload["run_id"]
        assert isinstance(run_id, str)
        before = await PgsqlTaskStore(self.database).get_run(run_id)
        assert before is not None
        self.assertEqual(before.state, TaskRunState.QUEUED)
        self.assertIsNotNone(before.request.input_payload)
        self.assertNotIn(
            "private scheduled input", repr(before.request.input_payload)
        )
        self.assertIn("aes-256-gcm.v1", repr(before.request.input_payload))
        # A new worker reconstructs the catalog; provider response is local,
        # while TaskWorker/native runner, claim, context and SQL are actual.
        provider = DeploymentProviderLoader(
            responses=(DeploymentProviderResponse("completed"),)
        )
        args = self.cli._parser.parse_args(
            ["task", "worker", "--once", *self.host_options()]
        )
        output = StringIO()
        with patch(
            "avalan.agent.loader.OrchestratorLoader.from_file",
            provider.from_file,
        ):
            self.assertTrue(
                await _task_worker(
                    args,
                    Console(file=output),
                    hub=self.hub,
                    logger=getLogger(__name__),
                )
            )
        completed = await PgsqlTaskStore(self.database).get_run(run_id)
        assert completed is not None
        self.assertEqual(completed.state, TaskRunState.SUCCEEDED)
        self.assertEqual(provider.inputs, ["private scheduled input"])
        self.assertIn(str(self.root / "catalog"), provider.paths[0])
        event_file = self.root / "events.jsonl"
        events = await self.command(
            "events", "daily", "--event-file", str(event_file)
        )
        self.assertTrue(events["ok"])
        self.assertEqual(events["sink_failed"], 0)
        self.assertIn('"event_id"', event_file.read_text())
        current = await self.command("inspect", "daily")
        snapshot = current["trigger"]
        assert isinstance(snapshot, dict)
        generation = str(snapshot["generation"])
        paused = await self.command(
            "pause", "daily", "--expected-generation", generation
        )
        self.assertTrue(paused["ok"])
        stale = await self.command(
            "resume", "daily", "--expected-generation", generation
        )
        self.assertFalse(stale["ok"])
        self.assertEqual(stale["code"], "trigger.conflict")
        assert isinstance(paused["trigger"], dict)
        resumed = await self.command(
            "resume",
            "daily",
            "--expected-generation",
            str(paused["trigger"]["generation"]),
        )
        self.assertTrue(resumed["ok"])
        self.assertTrue((await self.command("list"))["ok"])
        self.assertTrue(
            (await self.command("occurrences", "daily", "--coverage"))["ok"]
        )

    async def test_strict_flow_worker_executes_frozen_application_input(
        self,
    ) -> None:
        write_flow_application(self.root / "application")
        task = self.root / "application/task.toml"
        task.write_text(
            task.read_text().replace(
                'type="text"',
                'type="json"\nschema={type="string", const="frozen flow'
                ' value"}',
            )
        )
        self.trigger.write_text(
            self.trigger.read_text().replace(
                'value="private scheduled input"',
                'value={scheduled="frozen flow value"}',
            )
        )
        with patch(
            "avalan.trigger.apply.decision_time",
            AsyncMock(
                return_value=datetime.now(UTC) - timedelta(days=1, minutes=5)
            ),
        ):
            applied = await self.command(
                "apply", str(self.trigger), *self.host_options()
            )
        self.assertTrue(applied["ok"], applied)
        tick = await self.command("serve", "--once", *self.host_options())
        self.assertEqual(tick.get("admitted"), 1, tick)
        args = self.cli._parser.parse_args(
            ["task", "worker", "--once", *self.host_options()]
        )
        output = StringIO()
        self.assertTrue(
            await _task_worker(
                args,
                Console(file=output),
                hub=self.hub,
                logger=getLogger(__name__),
            )
        )
        inspected = await self.command("inspect", "daily")
        recent = inspected["recent_occurrences"]
        assert isinstance(recent, list)
        run_id = recent[0]["occurrence"]["payload"]["run_id"]
        run = await PgsqlTaskStore(self.database).get_run(run_id)
        assert run is not None and run.result is not None
        self.assertEqual(run.state, TaskRunState.SUCCEEDED, repr(run.result))
        self.assertNotIn("frozen flow value", repr(run.result.output_summary))
        self.assertIn("redacted", repr(run.result.output_summary))

    async def test_encrypted_file_registration_survives_source_removal(
        self,
    ) -> None:
        write_flow_application(self.root / "application")
        task = self.root / "application/task.toml"
        task.write_text(
            task.read_text()
            .replace('type="object"\nschema={type="object"}', 'type="file"')
            .replace('files="drop"', 'files="encrypt"')
        )
        flow = self.root / "application/flow.toml"
        flow.write_text(
            flow.read_text().replace(
                'name="scheduled"\ntype="string"',
                'name="scheduled"\ntype="file"',
            )
        )
        source = self.root / "application/input.txt"
        source.write_text("durable private bytes")
        self.trigger.write_text(
            self.trigger.read_text().replace(
                'value="private scheduled input"',
                'value={source_kind="local_path", reference="input.txt",'
                ' mime_type="text/plain"}',
            )
        )
        with patch(
            "avalan.trigger.apply.decision_time",
            AsyncMock(
                return_value=datetime.now(UTC) - timedelta(days=1, minutes=5)
            ),
        ):
            applied = await self.command(
                "apply", str(self.trigger), *self.host_options()
            )
        self.assertTrue(applied["ok"], applied)
        source.unlink()
        tick = await self.command("serve", "--once", *self.host_options())
        self.assertEqual(tick.get("admitted"), 1, tick)
        args = self.cli._parser.parse_args(
            [
                "task",
                "retention-sweep",
                "--encrypted-artifacts",
                "--raw-storage-allowed",
            ]
        )
        output = StringIO()
        self.assertTrue(
            await _task_retention_sweep(args, Console(file=output))
        )
        _, artifacts = encrypted_input_store(args, self.database)
        inspected = await self.command("inspect", "daily")
        assert isinstance(inspected["recent_occurrences"], list)
        run_id = inspected["recent_occurrences"][0]["occurrence"]["payload"][
            "run_id"
        ]
        rows = await PgsqlTaskStore(self.database).list_artifacts(run_id)
        self.assertEqual(len(rows), 1)
        with await artifacts.open(rows[0].ref) as stream:
            self.assertEqual(stream.read(), b"durable private bytes")

    async def test_safe_preflight_sink_failure_and_stale_apply(self) -> None:
        rejected = await self.command(
            "apply",
            str(self.trigger),
            "--deployment-root",
            str(self.root / "catalog"),
        )
        self.assertFalse(rejected["ok"])
        self.assertFalse((self.root / "catalog").exists())
        with patch.dict(
            environ, {"AVALAN_TASK_ENCRYPTION_KEY_B64": "not-base64-secret"}
        ):
            rejected = await self.command(
                "apply", str(self.trigger), *self.host_options()
            )
        self.assertEqual(rejected["code"], "trigger.encryption_unavailable")
        with patch(
            "avalan.trigger.apply.decision_time",
            AsyncMock(
                return_value=datetime.now(UTC) - timedelta(days=1, minutes=5)
            ),
        ):
            applied = await self.command(
                "apply", str(self.trigger), *self.host_options()
            )
        self.assertTrue(applied["ok"])
        tick = await self.command(
            "serve",
            "--once",
            *self.host_options(),
            "--metrics-file",
            str(self.root / "missing/metrics"),
        )
        self.assertTrue(tick["ok"], tick)
        self.assertEqual(tick["admitted"], 1)
        self.assertEqual(tick["sink_failures"], 1)
        bad_sink = await self.command(
            "events",
            "daily",
            "--event-file",
            str(self.root / "missing/events"),
        )
        self.assertTrue(bad_sink["ok"])
        failed = bad_sink["sink_failed"]
        assert isinstance(failed, int)
        self.assertGreater(failed, 0)
        stale = await self.command(
            "apply",
            str(self.trigger),
            *self.host_options(),
            "--expected-generation",
            "1",
        )
        self.assertFalse(stale["ok"])
        self.assertEqual(stale["code"], "trigger.conflict")
        with patch.dict(environ, {"AVALAN_TASK_OWNER_SCOPE": "another-owner"}):
            self.assertFalse((await self.command("inspect", "daily"))["ok"])
            self.assertEqual((await self.command("list"))["items"], [])
        self.assertFalse((await self.command("list", "--limit", "201"))["ok"])

    async def test_continuous_serve_stops_and_invalid_management_is_safe(
        self,
    ) -> None:
        (self.root / "catalog").mkdir()
        empty = await self.command("serve", "--once", *self.host_options())
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["code"], "trigger.deployment_mismatch")
        with patch.dict(environ, {"AVALAN_TASK_OWNER_SCOPE": ""}):
            missing = await self.command("list")
        self.assertEqual(missing["path"], "owner_scope")
        args = self.cli._parser.parse_args(["trigger", "list"])
        args.trigger_command = "unsupported"
        output = StringIO()
        self.assertFalse(
            await run_trigger_command(
                args, Console(file=output), self.hub, getLogger(__name__)
            )
        )
        self.assertEqual(loads(output.getvalue())["path"], "command")
        self.assertTrue(
            (
                await self.command(
                    "apply", str(self.trigger), *self.host_options()
                )
            )["ok"]
        )
        original = ObservedTriggerScheduler.serve

        async def stop_owned_service(
            scheduler: ObservedTriggerScheduler, *, stop: Event | None = None
        ) -> TriggerShutdownResult:
            assert stop is not None
            get_running_loop().call_soon(stop.set)
            return await original(scheduler, stop=stop)

        with patch.object(
            ObservedTriggerScheduler, "serve", stop_owned_service
        ):
            result = await self.command("serve", *self.host_options())
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["shutdown_settled"])
        self.assertEqual(result["pending_operations"], 0)
