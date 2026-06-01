from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rich.console import Console

from avalan.cli.commands import task as task_cmds
from avalan.task import TaskRunState

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "task" / "fixtures"


class _FakeTaskClient:
    def __init__(
        self,
        *,
        run_result: object | None = None,
        enqueue_result: object | None = None,
        wait_result: object | None = None,
    ) -> None:
        self.run_result = run_result
        self.enqueue_result = enqueue_result
        self.wait_result = wait_result
        self.input_value: object = None
        self.wait_timeout: float | None = None
        self.poll_interval: float | None = None

    async def run(
        self,
        definition: object,
        *,
        input_value: object = None,
        metadata: object | None = None,
    ) -> object:
        self.input_value = input_value
        return self.run_result

    async def enqueue(
        self,
        definition: object,
        *,
        input_value: object = None,
        queue_metadata: object | None = None,
    ) -> object:
        self.input_value = input_value
        return self.enqueue_result

    async def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> object:
        self.wait_timeout = timeout_seconds
        self.poll_interval = poll_interval_seconds
        return self.wait_result


class _FakeTaskClientContext:
    def __init__(self, client: _FakeTaskClient) -> None:
        self.client = client

    async def __aenter__(self) -> _FakeTaskClient:
        return self.client

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        return None


class CliTaskValidateTestCase(TestCase):
    def setUp(self) -> None:
        self.theme = MagicMock()

    def test_validate_prints_success_for_valid_definition(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_validate(
            Namespace(definition=str(FIXTURE_ROOT / "minimal.task.toml")),
            console,
            self.theme,
        )

        self.assertTrue(result)
        self.assertIn(
            "Task definition is valid: person_explainer 1",
            console.export_text(),
        )

    def test_validate_prints_load_issues(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_validate(
            Namespace(
                definition=str(FIXTURE_ROOT / "missing_sections.task.toml")
            ),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition could not be loaded.", output)
        self.assertIn("task.missing_section", output)
        self.assertIn("input", output)

    def test_validate_prints_validation_issues(self) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "bad.task.toml"
            definition.write_text(
                """
                [task]
                name = "bad"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "object"
                schema = {type = "array"}

                [execution]
                type = "flow"
                ref = "flows/private.toml"
                """,
                encoding="utf-8",
            )

            result = task_cmds.task_validate(
                Namespace(definition=str(definition)),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition is invalid.", output)
        self.assertIn("output.invalid_schema", output)
        self.assertIn("execution.unsupported_flow", output)
        self.assertNotIn("flows/private.toml", output)

    def test_validate_missing_file_prints_safe_diagnostic(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_validate(
            Namespace(definition="/tmp/private/missing.task.toml"),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition could not be read.", output)
        self.assertIn("file.read", output)
        self.assertNotIn("/tmp/private/missing.task.toml", output)


class CliTaskCommandShellTestCase(TestCase):
    def setUp(self) -> None:
        self.theme = MagicMock()

    def test_shell_commands_report_unavailable_diagnostic(self) -> None:
        commands = [
            ("artifacts", task_cmds.task_artifacts),
            ("events", task_cmds.task_events),
            ("inspect", task_cmds.task_inspect),
            ("output", task_cmds.task_output),
        ]

        for name, command in commands:
            console = Console(record=True, width=160)
            with self.subTest(command=name):
                result = command(Namespace(), console, self.theme)

            output = console.export_text()
            self.assertFalse(result)
            self.assertIn(f"Task {name} command is not available", output)

    def test_run_requires_store_without_ephemeral(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_run(
                Namespace(
                    definition=str(FIXTURE_ROOT / "minimal.task.toml"),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task store is not configured.", output)
        self.assertIn("store.missing", output)

    def test_worker_requires_durable_store(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_worker(
                Namespace(
                    queue="default",
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("store.missing", console.export_text())

    def test_run_uses_client_and_prints_sanitized_output(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.SUCCEEDED,
                    result=SimpleNamespace(
                        output_summary={"privacy": "<redacted>"}
                    ),
                )
            )
        )

        with patch.object(
            task_cmds,
            "_task_cli_client_context",
            return_value=_FakeTaskClientContext(client),
        ):
            result = task_cmds.task_run(
                Namespace(
                    definition=str(FIXTURE_ROOT / "minimal.task.toml"),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertEqual(client.input_value, "Ada Lovelace")
        self.assertIn("Task run completed (non-durable): run-1", output)
        self.assertIn('"privacy":"<redacted>"', output)

    def test_enqueue_waits_for_terminal_output(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            enqueue_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-queued",
                    state=TaskRunState.QUEUED,
                )
            ),
            wait_result=SimpleNamespace(
                run_id="run-queued",
                state=TaskRunState.SUCCEEDED,
                output_summary={"privacy": "<redacted>"},
                ready=True,
            ),
        )

        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "queued.task.toml"
            definition.write_text(
                """
                [task]
                name = "queued"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "text"

                [execution]
                type = "agent"
                ref = "agent.toml"

                [run]
                mode = "queue"
                queue = "documents"
                """,
                encoding="utf-8",
            )
            with patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ):
                result = task_cmds.task_enqueue(
                    Namespace(
                        definition=str(definition),
                        task_input="Ada",
                        task_input_json=None,
                        task_input_fields=(),
                        task_files=(),
                        store_dsn="postgresql://user:secret@db/tasks",
                        store_schema=None,
                        wait=True,
                        wait_timeout=5.0,
                        poll_interval=0.1,
                        ephemeral=False,
                        queue="documents",
                    ),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertTrue(result)
        self.assertEqual(client.wait_timeout, 5.0)
        self.assertEqual(client.poll_interval, 0.1)
        self.assertIn("Task enqueued: run-queued", output)
        self.assertIn("Task finished: run-queued", output)
        self.assertNotIn("secret", output)


class CliTaskPgsqlTestCase(TestCase):
    def setUp(self) -> None:
        self.theme = MagicMock()

    def test_pgsql_status_dispatches_current_with_safe_success(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(task_cmds, "run_task_pgsql_current") as current:
            result = task_cmds.task_pgsql_status(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema="tenant_tasks",
                    verbose=True,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        current.assert_called_once()
        settings = current.call_args.args[0]
        self.assertEqual(
            settings.url, "postgresql://user:secret@db.example.com/tasks"
        )
        self.assertEqual(settings.schema, "tenant_tasks")
        self.assertEqual(current.call_args.kwargs["verbose"], True)
        self.assertIn("migration status checked", output)
        self.assertNotIn("secret", output)
        self.assertNotIn("db.example.com", output)

    def test_pgsql_migrate_dispatches_upgrade_revision(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(task_cmds, "run_task_pgsql_upgrade") as upgrade:
            result = task_cmds.task_pgsql_migrate(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                    migration_revision="head",
                ),
                console,
                self.theme,
            )

        self.assertTrue(result)
        upgrade.assert_called_once()
        self.assertEqual(upgrade.call_args.kwargs["revision"], "head")
        self.assertNotIn("secret", console.export_text())

    def test_pgsql_check_dispatches_check(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(task_cmds, "run_task_pgsql_check") as check:
            result = task_cmds.task_pgsql_check(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                ),
                console,
                self.theme,
            )

        self.assertTrue(result)
        check.assert_called_once()
        self.assertIn("migrations are current", console.export_text())

    def test_pgsql_stamp_dispatches_stamp_revision(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(task_cmds, "run_task_pgsql_stamp") as stamp:
            result = task_cmds.task_pgsql_stamp(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                    migration_revision="20260530_0001",
                ),
                console,
                self.theme,
            )

        self.assertTrue(result)
        stamp.assert_called_once()
        self.assertEqual(stamp.call_args.kwargs["revision"], "20260530_0001")
        self.assertIn("20260530_0001", console.export_text())
        self.assertNotIn("secret", console.export_text())

    def test_pgsql_diagnose_uses_env_without_printing_dsn(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(
            task_cmds.environ,
            {
                "AVALAN_TASK_PGSQL_DSN": (
                    "postgresql://user:secret@db.example.com/tasks"
                ),
                "AVALAN_TASK_PGSQL_SCHEMA": "tenant_tasks",
            },
        ):
            result = task_cmds.task_pgsql_diagnose(
                Namespace(dsn=None, schema=None),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("configured", output)
        self.assertIn("tenant_tasks", output)
        self.assertIn("20260530_0001", output)
        self.assertNotIn("secret", output)
        self.assertNotIn("db.example.com", output)

    def test_pgsql_commands_require_configured_dsn(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_pgsql_check(
                Namespace(dsn=None, schema=None),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("DSN is not configured", output)
        self.assertIn("AVALAN_TASK_PGSQL_DSN", output)

    def test_pgsql_status_requires_configured_dsn(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_pgsql_status(
                Namespace(dsn=None, schema=None, verbose=False),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("DSN is not configured", console.export_text())

    def test_pgsql_migrate_requires_configured_dsn(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_pgsql_migrate(
                Namespace(
                    dsn=None,
                    schema=None,
                    migration_revision="head",
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("DSN is not configured", console.export_text())

    def test_pgsql_stamp_requires_configured_dsn(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_pgsql_stamp(
                Namespace(
                    dsn=None,
                    schema=None,
                    migration_revision="head",
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("DSN is not configured", console.export_text())

    def test_pgsql_status_errors_are_sanitized(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "run_task_pgsql_current",
            side_effect=task_cmds.PgsqlTaskMigrationError(
                "dependency.task_pgsql_migrations_missing: install extras"
            ),
        ):
            result = task_cmds.task_pgsql_status(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                    verbose=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("dependency.task_pgsql_migrations_missing", output)
        self.assertNotIn("secret", output)

    def test_pgsql_check_errors_are_sanitized(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "run_task_pgsql_check",
            side_effect=task_cmds.PgsqlTaskMigrationError(
                "dependency.task_pgsql_migrations_missing: install extras"
            ),
        ):
            result = task_cmds.task_pgsql_check(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("dependency.task_pgsql_migrations_missing", output)
        self.assertNotIn("secret", output)

    def test_pgsql_errors_are_sanitized(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "run_task_pgsql_upgrade",
            side_effect=task_cmds.PgsqlTaskMigrationError(
                "dependency.task_pgsql_migrations_missing: install extras"
            ),
        ):
            result = task_cmds.task_pgsql_migrate(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                    migration_revision="head",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("dependency.task_pgsql_migrations_missing", output)
        self.assertNotIn("secret", output)
        self.assertNotIn("db.example.com", output)

    def test_pgsql_stamp_errors_are_sanitized(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "run_task_pgsql_stamp",
            side_effect=task_cmds.PgsqlTaskMigrationError(
                "dependency.task_pgsql_migrations_missing: install extras"
            ),
        ):
            result = task_cmds.task_pgsql_stamp(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                    migration_revision="head",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("dependency.task_pgsql_migrations_missing", output)
        self.assertNotIn("secret", output)

    def test_invalid_revision_prints_safe_diagnostic(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_pgsql_migrate(
            Namespace(
                dsn="postgresql://user:secret@db.example.com/tasks",
                schema=None,
                migration_revision="head;drop",
            ),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Invalid PostgreSQL migration argument", output)
        self.assertNotIn("head;drop", output)
        self.assertNotIn("secret", output)


if __name__ == "__main__":
    main()
