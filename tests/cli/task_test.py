from argparse import Namespace
from base64 import b64decode
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from io import StringIO
from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rich.console import Console

from avalan.cli.commands import task as task_cmds
from avalan.entities import Message, MessageContentFile, MessageContentText
from avalan.task import (
    ArtifactStoreError,
    CallableTaskTargetRunner,
    SanitizedTaskEvent,
    TaskClientUnsupportedOperationError,
    TaskClientWaitTimeoutError,
    TaskDefinition,
    TaskEventCategory,
    TaskExecutionContext,
    TaskExecutionTarget,
    TaskInputContract,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskRetentionAction,
    TaskRetentionStoreNotFoundError,
    TaskRunState,
    TaskStoreNotFoundError,
    TaskTargetContext,
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    canonical_schema_json,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "task" / "fixtures"
TASK_HMAC_ENV = {
    "AVALAN_TASK_HMAC_KEY_ID": "cli-test-v1",
    "AVALAN_TASK_HMAC_KEY_B64": "dGFzay1obWFjLXRlc3Qta2V5",
}


class _FakeTaskClient:
    def __init__(
        self,
        *,
        run_result: object | None = None,
        enqueue_result: object | None = None,
        wait_result: object | None = None,
        inspect_result: object | None = None,
        output_result: object | None = None,
        events_result: tuple[object, ...] = (),
        artifacts_result: tuple[object, ...] = (),
        run_error: BaseException | None = None,
        enqueue_error: BaseException | None = None,
        wait_error: BaseException | None = None,
        inspect_error: BaseException | None = None,
        output_error: BaseException | None = None,
        events_error: BaseException | None = None,
        artifacts_error: BaseException | None = None,
    ) -> None:
        self.run_result = run_result
        self.enqueue_result = enqueue_result
        self.wait_result = wait_result
        self.inspect_result = inspect_result
        self.output_result = output_result
        self.events_result = events_result
        self.artifacts_result = artifacts_result
        self.run_error = run_error
        self.enqueue_error = enqueue_error
        self.wait_error = wait_error
        self.inspect_error = inspect_error
        self.output_error = output_error
        self.events_error = events_error
        self.artifacts_error = artifacts_error
        self.input_value: object = None
        self.queue_name: str | None = None
        self.queue_metadata: object = None
        self.wait_timeout: float | None = None
        self.poll_interval: float | None = None
        self.after_sequence: int | None = None
        self.attempt_id: str | None = None

    async def run(
        self,
        definition: object,
        *,
        input_value: object = None,
        metadata: object | None = None,
    ) -> object:
        if self.run_error is not None:
            raise self.run_error
        self.input_value = input_value
        return self.run_result

    async def enqueue(
        self,
        definition: object,
        *,
        input_value: object = None,
        queue_name: str | None = None,
        queue_metadata: object | None = None,
    ) -> object:
        if self.enqueue_error is not None:
            raise self.enqueue_error
        self.input_value = input_value
        self.queue_name = queue_name
        self.queue_metadata = queue_metadata
        return self.enqueue_result

    async def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> object:
        if self.wait_error is not None:
            raise self.wait_error
        self.wait_timeout = timeout_seconds
        self.poll_interval = poll_interval_seconds
        return self.wait_result

    async def inspect(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> object:
        _ = run_id
        if self.inspect_error is not None:
            raise self.inspect_error
        self.after_sequence = after_sequence
        return self.inspect_result

    async def output(self, run_id: str) -> object:
        _ = run_id
        if self.output_error is not None:
            raise self.output_error
        return self.output_result

    async def events(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        after_sequence: int | None = None,
    ) -> tuple[object, ...]:
        _ = run_id
        if self.events_error is not None:
            raise self.events_error
        self.attempt_id = attempt_id
        self.after_sequence = after_sequence
        return self.events_result

    async def artifacts(self, run_id: str) -> tuple[object, ...]:
        _ = run_id
        if self.artifacts_error is not None:
            raise self.artifacts_error
        return self.artifacts_result


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


class _Snapshot:
    def __init__(self, value: object) -> None:
        self.value = value

    def as_dict(self) -> object:
        return self.value


class _FakeResource:
    def __init__(
        self,
        *,
        open_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.open_error = open_error
        self.close_error = close_error
        self.entered = False
        self.exited = False
        self.opened = False
        self.closed = False

    async def __aenter__(self) -> "_FakeResource":
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        self.exited = True
        return None

    async def open(self) -> None:
        self.opened = True
        if self.open_error is not None:
            raise self.open_error

    async def aclose(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _FakeTaskWorker:
    instances: list["_FakeTaskWorker"] = []
    results: list[object] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls = 0
        self.instances.append(self)

    async def process_once(self) -> object:
        self.calls += 1
        if self.results:
            result = self.results.pop(0)
            if callable(result):
                return result(self)
            return result
        return SimpleNamespace(processed=False)


class _FakeRetentionService:
    instances: list["_FakeRetentionService"] = []
    results: tuple[object, ...] = ()
    error: BaseException | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.purposes: object = None
        self.limit: int | None = None
        self.instances.append(self)

    async def sweep_expired(
        self,
        *,
        purposes: object = None,
        limit: int = 100,
    ) -> object:
        if self.error is not None:
            raise self.error
        self.purposes = purposes
        self.limit = limit
        return SimpleNamespace(limit=limit, results=self.results)


class _ExtractionCliResponse:
    input_token_count = 19
    output_token_count = 23
    total_token_count = 42

    def __init__(self, output: Mapping[str, object]) -> None:
        self.output = output

    async def to_json(self) -> str:
        return dumps(self.output, sort_keys=True, separators=(",", ":"))

    async def to_str(self) -> str:
        return await self.to_json()


class _ExtractionCliOrchestrator:
    def __init__(self, output: Mapping[str, object]) -> None:
        self.output = output
        self.inputs: list[object] = []
        self.text_formats: list[Mapping[str, object]] = []
        self.reasoning_options: list[Mapping[str, object]] = []

    async def __aenter__(self) -> "_ExtractionCliOrchestrator":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        return None

    async def __call__(self, input: object) -> _ExtractionCliResponse:
        self.inputs.append(input)
        return _ExtractionCliResponse(self.output)


def _extraction_cli_output() -> dict[str, object]:
    return {
        "line_items": [
            {
                "line_number": 1,
                "vendor_name": "Northwind Office Supplies",
                "vendor_address": "42 Market St, Denver, CO 80202",
                "customer_name": "Contoso Research Lab",
                "customer_address": "100 Example Ave, Suite 1, Denver, CO 80202",
                "invoice_number": "INV-1001",
                "invoice_date": "01/15/2026",
                "due_date": "02/14/2026",
                "purchase_order": "PO-555100",
                "description": "Document processing services",
                "quantity": "5",
                "unit_price": "25.00",
                "line_amount": "125.00",
                "tax_amount": "0.00",
                "total_amount": "125.00",
                "currency": "USD",
                "notes": "Synthetic invoice fixture",
            }
        ]
    }


class CliTaskValidateTestCase(TestCase):
    def setUp(self) -> None:
        self.theme = MagicMock()

    def test_validate_prints_success_for_valid_definition(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
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

    def test_validate_reports_missing_hmac_key(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_validate(
                Namespace(definition=str(FIXTURE_ROOT / "minimal.task.toml")),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("privacy.hmac_key_missing", output)
        self.assertNotIn("Ada Lovelace", output)

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

            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                result = task_cmds.task_validate(
                    Namespace(definition=str(definition)),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition is invalid.", output)
        self.assertIn("output.invalid_schema", output)
        self.assertNotIn("feature.flow_backed_tasks_disabled", output)
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

    def test_inspection_commands_require_durable_store(self) -> None:
        commands = [
            task_cmds.task_artifacts,
            task_cmds.task_events,
            task_cmds.task_inspect,
            task_cmds.task_output,
        ]

        for command in commands:
            console = Console(record=True, width=160)
            with (
                self.subTest(command=command.__name__),
                patch.dict(task_cmds.environ, {}, clear=True),
            ):
                result = command(
                    Namespace(
                        run_id="run-private",
                        store_dsn=None,
                        store_schema=None,
                        attempt_id=None,
                        after_sequence=None,
                    ),
                    console,
                    self.theme,
                )

            output = console.export_text()
            self.assertFalse(result)
            self.assertIn("store.missing", output)
            self.assertNotIn("run-private", output)

    def test_inspection_commands_print_stable_snapshots(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        client = _FakeTaskClient(
            inspect_result=_Snapshot(
                {
                    "run": {
                        "run_id": "run-1",
                        "state": "succeeded",
                        "input_summary": {"privacy": "<redacted>"},
                    },
                    "events": (),
                }
            ),
            output_result=_Snapshot(
                {
                    "run_id": "run-1",
                    "state": "failed",
                    "ready": False,
                    "error": {"code": "runnable.failed"},
                }
            ),
            events_result=(
                SanitizedTaskEvent(
                    event_id="event-1",
                    run_id="run-1",
                    sequence=2,
                    event_type="token_generated",
                    category=TaskEventCategory.TOKEN,
                    created_at=now,
                    attempt_id="attempt-1",
                    payload={"privacy": "<redacted>"},
                ),
            ),
            artifacts_result=(
                {
                    "artifact_id": "artifact-1",
                    "state": "ready",
                    "ref": {
                        "artifact_id": "artifact-1",
                        "store": "local",
                    },
                },
            ),
        )

        with patch.object(
            task_cmds,
            "_task_cli_inspection_client_context",
            return_value=_FakeTaskClientContext(client),
        ):
            inspect_console = Console(record=True, width=200)
            inspect_result = task_cmds.task_inspect(
                Namespace(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    after_sequence=1,
                ),
                inspect_console,
                self.theme,
            )
            output_console = Console(record=True, width=200)
            output_result = task_cmds.task_output(
                Namespace(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                ),
                output_console,
                self.theme,
            )
            events_console = Console(record=True, width=200)
            events_result = task_cmds.task_events(
                Namespace(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    attempt_id="attempt-1",
                    after_sequence=1,
                ),
                events_console,
                self.theme,
            )
            artifacts_console = Console(record=True, width=200)
            artifacts_result = task_cmds.task_artifacts(
                Namespace(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                ),
                artifacts_console,
                self.theme,
            )

        self.assertTrue(inspect_result)
        self.assertTrue(output_result)
        self.assertTrue(events_result)
        self.assertTrue(artifacts_result)
        self.assertEqual(client.after_sequence, 1)
        self.assertEqual(client.attempt_id, "attempt-1")
        rendered = (
            inspect_console.export_text()
            + output_console.export_text()
            + events_console.export_text()
            + artifacts_console.export_text()
        )
        self.assertIn("inspect", rendered)
        self.assertIn('"input_summary":{"privacy":"<redacted>"}', rendered)
        self.assertIn('"error":{"code":"runnable.failed"}', rendered)
        self.assertIn('"sequence":2', rendered)
        self.assertIn('"artifact_id":"artifact-1"', rendered)
        self.assertNotIn("secret", rendered)
        self.assertNotIn("token_id", rendered)

    def test_inspection_commands_report_not_found_safely(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            output_error=TaskStoreNotFoundError("private run secret")
        )

        with patch.object(
            task_cmds,
            "_task_cli_inspection_client_context",
            return_value=_FakeTaskClientContext(client),
        ):
            result = task_cmds.task_output(
                Namespace(
                    run_id="run-private",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("task.not_found", output)
        self.assertNotIn("private run secret", output)
        self.assertNotIn("run-private", output)

    def test_inspection_commands_report_safe_errors(self) -> None:
        cases = (
            (
                task_cmds.task_inspect,
                _FakeTaskClient(inspect_error=ImportError("private")),
                "dependency.missing",
                Namespace(
                    run_id="run-private",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    after_sequence=None,
                ),
            ),
            (
                task_cmds.task_events,
                _FakeTaskClient(events_error=OSError("private")),
                "io.failure",
                Namespace(
                    run_id="run-private",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    attempt_id=None,
                    after_sequence=None,
                ),
            ),
            (
                task_cmds.task_artifacts,
                _FakeTaskClient(artifacts_error=AssertionError("private")),
                "task.inspection",
                Namespace(
                    run_id="run-private",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                ),
            ),
        )
        for command, client, expected, args in cases:
            console = Console(record=True, width=160)
            with (
                self.subTest(command=command.__name__),
                patch.object(
                    task_cmds,
                    "_task_cli_inspection_client_context",
                    return_value=_FakeTaskClientContext(client),
                ),
            ):
                result = command(args, console, self.theme)

            output = console.export_text()
            self.assertFalse(result)
            self.assertIn(expected, output)
            self.assertNotIn("private", output)

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

    def test_worker_reports_missing_dependency_gate(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "require_feature",
            return_value=(
                SimpleNamespace(
                    code="dependency.task_worker_pgsql_missing",
                    message="Task queue workers require the task-pgsql extra.",
                    hint="Install avalan[task-pgsql] before starting workers.",
                ),
            ),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="default",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("dependency.task_worker_pgsql_missing", output)
        self.assertIn("avalan[task-pgsql]", output)
        self.assertNotIn("postgresql://db/tasks", output)

    def test_run_rejects_queued_definition(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "queued.task.toml"
            _write_queued_definition(definition)

            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada",
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

        self.assertFalse(result)
        self.assertIn(
            "Task run requires a direct-mode definition.",
            console.export_text(),
        )

    def test_run_reports_client_error(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(run_error=ImportError("missing private dep"))

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
        self.assertFalse(result)
        self.assertIn("dependency.missing", output)

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

    def test_run_json_prints_only_structured_output(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.SUCCEEDED,
                    result=SimpleNamespace(output_summary={"ignored": True}),
                ),
                output={"b": 2, "a": [1]},
            )
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
        ):
            definition = _write_direct_object_definition(Path(tmpdir))
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=True,
                    task_output_path=None,
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), '{"a":[1],"b":2}\n')

    def test_run_json_and_output_write_same_structured_value(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.SUCCEEDED,
                    result=SimpleNamespace(output_summary={"ignored": True}),
                ),
                output={"answer": "ok"},
            )
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
        ):
            root = Path(tmpdir)
            definition = _write_direct_object_definition(root)
            output_path = root / "result.json"
            output_path.write_text("old\n", encoding="utf-8")
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=True,
                    task_output_path=str(output_path),
                    task_pdf=None,
                    quiet=True,
                ),
                console,
                self.theme,
            )
            written = output_path.read_text(encoding="utf-8")

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), '{"answer":"ok"}\n')
        self.assertEqual(written, '{"answer":"ok"}\n')

    def test_run_output_parent_failure_skips_client(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient()

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
        ):
            root = Path(tmpdir)
            definition = _write_direct_object_definition(root)
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=False,
                    task_output_path=str(root / "missing" / "result.json"),
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIsNone(client.input_value)
        self.assertIn("output.write", output)
        self.assertNotIn("Ada Lovelace", output)

    def test_run_returns_false_when_output_write_fails(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.SUCCEEDED,
                    result=SimpleNamespace(output_summary={"ignored": True}),
                ),
                output={"answer": "ok"},
            )
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
            patch.object(
                task_cmds,
                "_write_task_run_structured_output",
                return_value=False,
            ),
        ):
            definition = _write_direct_object_definition(Path(tmpdir))
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=False,
                    task_output_path="result.json",
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertEqual(client.input_value, "Ada Lovelace")

    def test_run_json_failure_does_not_write_stdout(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        diagnostic_console = Console(record=True, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.FAILED,
                    result=SimpleNamespace(
                        output_summary=None,
                        error={"code": "output_contract.failed"},
                    ),
                ),
                output={"private": "partial"},
            )
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
            patch.object(
                task_cmds,
                "_task_diagnostic_console",
                return_value=diagnostic_console,
            ),
        ):
            definition = _write_direct_object_definition(Path(tmpdir))
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=True,
                    task_output_path=None,
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )

        diagnostics = diagnostic_console.export_text()
        self.assertFalse(result)
        self.assertEqual(stream.getvalue(), "")
        self.assertIn("task.run_failed", diagnostics)
        self.assertIn("output_contract.failed", diagnostics)
        self.assertNotIn("partial", diagnostics)

    def test_run_json_failure_codes_are_sanitized(self) -> None:
        cases = (
            ("provider.structured_output_failed", "provider"),
            ("output.parse_failed", "output_contract"),
            ("output_contract.failed", "output_contract"),
        )

        for code, category in cases:
            with self.subTest(code=code):
                stream = StringIO()
                console = Console(file=stream, width=160)
                diagnostic_console = Console(record=True, width=160)
                client = _FakeTaskClient(
                    run_result=SimpleNamespace(
                        run=SimpleNamespace(
                            run_id="run-1",
                            state=TaskRunState.FAILED,
                            result=SimpleNamespace(
                                output_summary=None,
                                error={
                                    "category": category,
                                    "code": code,
                                    "message": "safe failure summary",
                                },
                            ),
                        ),
                        output={"private": "partial provider body"},
                    )
                )

                with (
                    TemporaryDirectory() as tmpdir,
                    patch.object(
                        task_cmds,
                        "_task_cli_client_context",
                        return_value=_FakeTaskClientContext(client),
                    ),
                    patch.object(
                        task_cmds,
                        "_task_diagnostic_console",
                        return_value=diagnostic_console,
                    ),
                ):
                    definition = _write_direct_object_definition(Path(tmpdir))
                    result = task_cmds.task_run(
                        Namespace(
                            definition=str(definition),
                            task_input="Ada Lovelace",
                            task_input_json=None,
                            task_input_fields=(),
                            task_files=(),
                            store_dsn=None,
                            store_schema=None,
                            ephemeral=True,
                            task_run_json=True,
                            task_output_path=None,
                            task_pdf=None,
                            quiet=False,
                        ),
                        console,
                        self.theme,
                    )

                diagnostics = diagnostic_console.export_text()
                self.assertFalse(result)
                self.assertEqual(stream.getvalue(), "")
                self.assertIn(code, diagnostics)
                self.assertNotIn("partial provider body", diagnostics)
                self.assertNotIn("Ada Lovelace", diagnostics)

    def test_run_quiet_failure_suppresses_summary(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.FAILED,
                    result=SimpleNamespace(
                        output_summary=None,
                        error={
                            "category": "output_contract",
                            "code": "output.parse_failed",
                            "message": "safe failure summary",
                        },
                    ),
                ),
                output={"private": "partial provider body"},
            )
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
        ):
            definition = _write_direct_object_definition(Path(tmpdir))
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=False,
                    task_output_path=None,
                    task_pdf=None,
                    quiet=True,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("output.parse_failed", output)
        self.assertNotIn("Task run completed", output)
        self.assertNotIn("partial provider body", output)
        self.assertNotIn("Ada Lovelace", output)

    def test_run_pdf_missing_file_reports_safe_diagnostic(self) -> None:
        console = Console(record=True, width=160)

        async def target(context: TaskTargetContext) -> object:
            _ = context
            raise AssertionError("target should not run")

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_agent_task_target",
                return_value=CallableTaskTargetRunner(target),
            ),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            root = Path(tmpdir)
            definition = root / "missing_pdf.task.toml"
            definition.write_text(
                """
                [task]
                name = "missing_pdf"
                version = "1"

                [input]
                type = "file"
                mime_types = ["application/pdf"]

                [output]
                type = "text"

                [execution]
                type = "agent"
                ref = "agent.toml"
                """,
                encoding="utf-8",
            )
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=False,
                    task_output_path=None,
                    task_pdf="/tmp/private/customer-secret.pdf",
                    quiet=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task run did not succeed.", output)
        self.assertIn("input_contract.failed", output)
        self.assertNotIn("customer-secret.pdf", output)
        self.assertNotIn("/tmp/private", output)

    def test_run_json_rejects_text_output_contract(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        diagnostic_console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "_task_diagnostic_console",
            return_value=diagnostic_console,
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
                    task_run_json=True,
                    task_output_path=None,
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertEqual(stream.getvalue(), "")
        self.assertIn("output.unsupported", diagnostic_console.export_text())

    def test_run_json_uses_real_ephemeral_client(self) -> None:
        async def target(context: TaskTargetContext) -> object:
            return {"answer": "ok", "input": context.input_value}

        stream = StringIO()
        console = Console(file=stream, width=160)

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_agent_task_target",
                return_value=CallableTaskTargetRunner(target),
            ),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            root = Path(tmpdir)
            definition = _write_direct_object_definition(root)
            output_path = root / "result.json"
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=True,
                    task_output_path=str(output_path),
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )
            written = output_path.read_text(encoding="utf-8")

        expected = '{"answer":"ok","input":"Ada Lovelace"}\n'
        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), expected)
        self.assertEqual(written, expected)

    def test_run_poc_extraction_fixture_reaches_fake_provider(self) -> None:
        fixture = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
        )
        pdf_bytes = (fixture / "sample.pdf").read_bytes()
        output = _extraction_cli_output()
        expected = dumps(output, sort_keys=True, separators=(",", ":")) + "\n"
        cases = (
            (
                "pdf",
                {
                    "task_pdf": "./sample.pdf",
                    "task_files": (),
                    "task_file_mime_types": (),
                },
            ),
            (
                "file",
                {
                    "task_pdf": None,
                    "task_files": ("input=./sample.pdf",),
                    "task_file_mime_types": ("input=application/pdf",),
                },
            ),
        )

        for name, input_args in cases:
            with self.subTest(command=name):
                stream = StringIO()
                console = Console(file=stream, width=160)
                orchestrator = _ExtractionCliOrchestrator(output)
                settings_values: list[Any] = []

                async def from_settings(
                    loader: object,
                    settings: object,
                    *,
                    tool_settings: object | None = None,
                    tool_format: object | None = None,
                ) -> _ExtractionCliOrchestrator:
                    _ = loader, tool_settings, tool_format
                    call_options = cast(Any, settings).call_options
                    assert isinstance(call_options, Mapping)
                    response_format = cast(
                        Mapping[str, object],
                        call_options["response_format"],
                    )
                    orchestrator.text_formats.append(
                        {
                            "type": response_format["type"],
                            "name": response_format["name"],
                            "schema": response_format["schema"],
                            "strict": response_format["strict"],
                        }
                    )
                    orchestrator.reasoning_options.append(
                        cast(Mapping[str, object], call_options["reasoning"])
                    )
                    settings_values.append(settings)
                    return orchestrator

                with (
                    TemporaryDirectory() as tmpdir,
                    patch.object(
                        task_cmds.OrchestratorLoader,
                        "from_settings",
                        new=from_settings,
                    ),
                    patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
                ):
                    output_path = Path(tmpdir) / "extraction.json"
                    result = task_cmds.task_run(
                        Namespace(
                            definition=str(fixture / "task.toml"),
                            task_input=None,
                            task_input_json=None,
                            task_input_fields=(),
                            task_file_descriptors=(),
                            task_provider_file_ids=(),
                            task_hosted_urls=(),
                            task_object_store_uris=(),
                            task_file_roles=(),
                            task_file_sizes=(),
                            task_file_sha256=(),
                            task_file_conversions=(),
                            store_dsn=None,
                            store_schema=None,
                            ephemeral=True,
                            task_run_json=True,
                            task_output_path=str(output_path),
                            quiet=True,
                            **input_args,
                        ),
                        console,
                        self.theme,
                    )
                    written = output_path.read_text(encoding="utf-8")

                self.assertTrue(result)
                self.assertEqual(stream.getvalue(), expected)
                self.assertEqual(written, expected)
                self.assertEqual(len(settings_values), 1)
                settings = settings_values[0]
                agent_config = settings.agent_config
                self.assertIsInstance(agent_config, Mapping)
                self.assertIn("instructions", agent_config)
                self.assertNotIn("system", agent_config)
                self.assertNotIn("task", agent_config)
                self.assertEqual(settings.tools, [])
                call_options = settings.call_options
                self.assertIsInstance(call_options, Mapping)
                self.assertNotIn("tools", call_options)
                self.assertNotIn("tool_choice", call_options)
                self.assertEqual(
                    orchestrator.reasoning_options,
                    [{"effort": "high"}],
                )
                self.assertEqual(
                    canonical_schema_json(
                        orchestrator.text_formats[0]["schema"]
                    ),
                    canonical_schema_json(
                        task_cmds.TaskDefinitionLoader()
                        .load(fixture / "task.toml")
                        .output.schema
                    ),
                )
                self.assertEqual(len(orchestrator.inputs), 1)
                agent_input = orchestrator.inputs[0]
                self.assertIsInstance(agent_input, Message)
                content = cast(Message, agent_input).content
                self.assertIsInstance(content, list)
                blocks = cast(list[Any], content)
                text_blocks = [
                    block
                    for block in blocks
                    if isinstance(block, MessageContentText)
                ]
                file_blocks = [
                    block
                    for block in blocks
                    if isinstance(block, MessageContentFile)
                ]
                self.assertEqual(len(text_blocks), 1)
                self.assertIn(
                    "Analyze the attached synthetic invoice PDF",
                    text_blocks[0].text,
                )
                self.assertEqual(len(file_blocks), 1)
                self.assertEqual(
                    file_blocks[0].file["mime_type"],
                    "application/pdf",
                )
                self.assertEqual(
                    b64decode(cast(str, file_blocks[0].file["file_data"])),
                    pdf_bytes,
                )

    def test_structured_output_writer_reports_safe_failures(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        diagnostic_console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "_write_task_run_output_file",
            return_value=False,
        ):
            self.assertFalse(
                task_cmds._write_task_run_structured_output(
                    Namespace(task_run_json=True, task_output_path="out.json"),
                    console,
                    diagnostic_console,
                    {"answer": "ok"},
                )
            )

        self.assertEqual(stream.getvalue(), "")

    def test_output_file_writer_reports_safe_failures(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing_parent = root / "missing" / "result.json"
            self.assertFalse(
                task_cmds._write_task_run_output_file(
                    str(missing_parent),
                    "{}\n",
                    console,
                )
            )

            with (
                patch.object(
                    task_cmds.Path,
                    "replace",
                    side_effect=OSError("private replace failure"),
                ),
                patch.object(
                    task_cmds.Path,
                    "unlink",
                    side_effect=OSError("private cleanup failure"),
                ),
            ):
                self.assertFalse(
                    task_cmds._write_task_run_output_file(
                        str(root / "result.json"),
                        "{}\n",
                        console,
                    )
                )

        output = console.export_text()
        self.assertIn("output.write", output)
        self.assertNotIn("private", output)

    def test_run_without_result_skips_output_line(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.FAILED,
                    result=None,
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
        self.assertFalse(result)
        self.assertIn("Task run completed (non-durable): run-1", output)
        self.assertIn("state failed", output)
        self.assertNotIn("output ", output)

    def test_enqueue_rejects_ephemeral_storage(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "queued.task.toml"
            _write_queued_definition(definition)

            result = task_cmds.task_enqueue(
                Namespace(
                    definition=str(definition),
                    task_input="Ada",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    wait=False,
                    wait_timeout=None,
                    poll_interval=1.0,
                    ephemeral=True,
                    queue="default",
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("store.ephemeral_unsupported", console.export_text())

    def test_enqueue_rejects_direct_definition(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_enqueue(
            Namespace(
                definition=str(FIXTURE_ROOT / "minimal.task.toml"),
                task_input="Ada",
                task_input_json=None,
                task_input_fields=(),
                task_files=(),
                store_dsn="postgresql://db/tasks",
                store_schema=None,
                wait=False,
                wait_timeout=None,
                poll_interval=1.0,
                ephemeral=False,
                queue="default",
            ),
            console,
            self.theme,
        )

        self.assertFalse(result)
        self.assertIn(
            "Task enqueue requires a queued-mode definition.",
            console.export_text(),
        )

    def test_enqueue_requires_store(self) -> None:
        console = Console(record=True, width=160)

        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(task_cmds.environ, {}, clear=True),
        ):
            definition = Path(tmpdir) / "queued.task.toml"
            _write_queued_definition(definition)

            result = task_cmds.task_enqueue(
                Namespace(
                    definition=str(definition),
                    task_input="Ada",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    wait=False,
                    wait_timeout=None,
                    poll_interval=1.0,
                    ephemeral=False,
                    queue="default",
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("store.missing", console.export_text())

    def test_enqueue_without_wait_returns_after_submission(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            enqueue_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-queued",
                    state=TaskRunState.QUEUED,
                )
            )
        )

        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "queued.task.toml"
            _write_queued_definition(definition)
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
                        store_dsn="postgresql://db/tasks",
                        store_schema=None,
                        wait=False,
                        wait_timeout=None,
                        poll_interval=1.0,
                        ephemeral=False,
                        queue="priority-documents",
                    ),
                    console,
                    self.theme,
                )

        self.assertTrue(result)
        self.assertEqual(client.queue_name, "priority-documents")
        self.assertEqual(
            client.queue_metadata,
            {"cli_queue": "priority-documents"},
        )
        self.assertIn("Task enqueued: run-queued", console.export_text())

    def test_enqueue_reports_wait_timeout(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            enqueue_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-queued",
                    state=TaskRunState.QUEUED,
                )
            ),
            wait_error=TaskClientWaitTimeoutError(run_id="run-queued"),
        )

        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "queued.task.toml"
            _write_queued_definition(definition)
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
                        store_dsn="postgresql://db/tasks",
                        store_schema=None,
                        wait=True,
                        wait_timeout=0.01,
                        poll_interval=0.01,
                        ephemeral=False,
                        queue="default",
                    ),
                    console,
                    self.theme,
                )

        self.assertFalse(result)
        self.assertIn("task.wait_timeout", console.export_text())

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

    def test_worker_rejects_ephemeral_storage(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_worker(
            Namespace(
                queue="default",
                store_dsn="postgresql://db/tasks",
                store_schema=None,
                ephemeral=True,
            ),
            console,
            self.theme,
        )

        self.assertFalse(result)
        self.assertIn("store.ephemeral_unsupported", console.export_text())

    def test_worker_rejects_heartbeat_interval_not_shorter_than_lease(
        self,
    ) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_worker(
            Namespace(
                queue="default",
                store_dsn="postgresql://user:secret@db/tasks",
                store_schema=None,
                worker_id="worker-a",
                once=True,
                limit=10,
                lease_seconds=30,
                heartbeat_seconds=30,
                ephemeral=False,
            ),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("worker.heartbeat_interval", output)
        self.assertNotIn("secret", output)

    def test_worker_processes_until_no_work(self) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeTaskWorker.instances = []
        _FakeTaskWorker.results = [
            SimpleNamespace(
                processed=True,
                completion=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-1",
                        state=TaskRunState.SUCCEEDED,
                    )
                ),
                retry=None,
            ),
            SimpleNamespace(processed=False, completion=None, retry=None),
        ]

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(task_cmds, "PgsqlTaskQueue", return_value=object()),
            patch.object(
                task_cmds, "_agent_task_target", return_value=object()
            ),
            patch.object(task_cmds, "TaskWorker", _FakeTaskWorker),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="documents",
                    store_dsn="postgresql://db/tasks",
                    store_schema="tasks",
                    worker_id="worker-a",
                    once=False,
                    limit=2,
                    lease_seconds=30,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertTrue(database.entered)
        self.assertTrue(database.exited)
        self.assertIsNotNone(
            _FakeTaskWorker.instances[0].kwargs["hmac_provider"]
        )
        self.assertIn("Task processed: run-1 succeeded", output)
        self.assertIn("Task worker processed 1 run.", output)

    def test_worker_stops_after_shutdown_request(self) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeTaskWorker.instances = []

        def stop_after_first(worker: _FakeTaskWorker) -> object:
            worker.kwargs["shutdown"].request()
            return SimpleNamespace(
                processed=True,
                completion=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-1",
                        state=TaskRunState.SUCCEEDED,
                    )
                ),
                retry=None,
            )

        _FakeTaskWorker.results = [
            stop_after_first,
            SimpleNamespace(
                processed=True,
                completion=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-2",
                        state=TaskRunState.SUCCEEDED,
                    )
                ),
                retry=None,
            ),
        ]

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(task_cmds, "PgsqlTaskQueue", return_value=object()),
            patch.object(
                task_cmds, "_agent_task_target", return_value=object()
            ),
            patch.object(task_cmds, "TaskWorker", _FakeTaskWorker),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="documents",
                    store_dsn="postgresql://db/tasks",
                    store_schema="tasks",
                    worker_id="worker-a",
                    once=False,
                    limit=2,
                    lease_seconds=30,
                    heartbeat_seconds=0.25,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertEqual(_FakeTaskWorker.instances[0].calls, 1)
        self.assertEqual(
            _FakeTaskWorker.instances[0].kwargs["heartbeat_seconds"],
            0.25,
        )
        self.assertIn("Task processed: run-1 succeeded", output)
        self.assertNotIn("run-2", output)
        self.assertIn("Task worker processed 1 run.", output)

    def test_worker_reports_shutdown_abandonment(self) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeTaskWorker.instances = []
        _FakeTaskWorker.results = [
            SimpleNamespace(
                processed=True,
                completion=None,
                retry=None,
                abandonment=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-abandoned",
                        state=TaskRunState.QUEUED,
                    )
                ),
                shutdown_requested=True,
                lease_lost=False,
            ),
            SimpleNamespace(
                processed=True,
                completion=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-2",
                        state=TaskRunState.SUCCEEDED,
                    )
                ),
                retry=None,
            ),
        ]

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(task_cmds, "PgsqlTaskQueue", return_value=object()),
            patch.object(
                task_cmds, "_agent_task_target", return_value=object()
            ),
            patch.object(task_cmds, "TaskWorker", _FakeTaskWorker),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="documents",
                    store_dsn="postgresql://db/tasks",
                    store_schema="tasks",
                    worker_id="worker-a",
                    once=False,
                    limit=2,
                    lease_seconds=30,
                    heartbeat_seconds=0.25,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertEqual(_FakeTaskWorker.instances[0].calls, 1)
        self.assertIn("Task processed: run-abandoned queued", output)
        self.assertIn("Task worker shutdown requested.", output)
        self.assertNotIn("run-2", output)
        self.assertIn("Task worker processed 1 run.", output)

    def test_worker_reports_claim_loss_as_failure(self) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeTaskWorker.instances = []
        _FakeTaskWorker.results = [
            SimpleNamespace(
                processed=True,
                completion=None,
                retry=None,
                abandonment=None,
                shutdown_requested=False,
                lease_lost=True,
                claimed=SimpleNamespace(
                    run=SimpleNamespace(run_id="run-lost")
                ),
                private_detail="private heartbeat outage",
            ),
            SimpleNamespace(
                processed=True,
                completion=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-2",
                        state=TaskRunState.SUCCEEDED,
                    )
                ),
                retry=None,
            ),
        ]

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(task_cmds, "PgsqlTaskQueue", return_value=object()),
            patch.object(
                task_cmds, "_agent_task_target", return_value=object()
            ),
            patch.object(task_cmds, "TaskWorker", _FakeTaskWorker),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="documents",
                    store_dsn="postgresql://db/tasks",
                    store_schema="tasks",
                    worker_id="worker-a",
                    once=False,
                    limit=2,
                    lease_seconds=30,
                    heartbeat_seconds=0.25,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertEqual(_FakeTaskWorker.instances[0].calls, 1)
        self.assertIn("Task claim lost: run-lost", output)
        self.assertNotIn("run-2", output)
        self.assertNotIn("private heartbeat outage", output)
        self.assertIn("Task worker processed 1 run.", output)

    def test_worker_reports_retry_and_counts_missing_run_results(
        self,
    ) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeTaskWorker.instances = []
        _FakeTaskWorker.results = [
            SimpleNamespace(
                processed=True,
                completion=None,
                retry=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-retry",
                        state=TaskRunState.QUEUED,
                    )
                ),
            ),
            SimpleNamespace(processed=True, completion=None, retry=None),
        ]

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(task_cmds, "PgsqlTaskQueue", return_value=object()),
            patch.object(
                task_cmds, "_agent_task_target", return_value=object()
            ),
            patch.object(task_cmds, "TaskWorker", _FakeTaskWorker),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="documents",
                    store_dsn="postgresql://db/tasks",
                    store_schema="tasks",
                    worker_id="worker-a",
                    once=False,
                    limit=2,
                    lease_seconds=30,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("Task processed: run-retry queued", output)
        self.assertIn("Task worker processed 2 runs.", output)
        self.assertNotIn("None", output)

    def test_worker_reports_startup_error(self) -> None:
        console = Console(record=True, width=160)

        with (
            patch.object(
                task_cmds,
                "_task_pgsql_database",
                side_effect=OSError("private dsn"),
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="default",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    worker_id=None,
                    once=True,
                    limit=10,
                    lease_seconds=30,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("io.failure", console.export_text())

    def test_retention_sweep_processes_expired_artifacts(self) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeRetentionService.instances = []
        _FakeRetentionService.error = None
        _FakeRetentionService.results = (
            SimpleNamespace(action=TaskRetentionAction.DELETED),
            SimpleNamespace(action=TaskRetentionAction.LOST),
        )

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(
                task_cmds, "_task_artifact_store", return_value=object()
            ),
            patch.object(
                task_cmds,
                "TaskRetentionService",
                _FakeRetentionService,
            ),
        ):
            result = task_cmds.task_retention_sweep(
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema="tasks",
                    purpose=("input", "output"),
                    limit=2,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertTrue(database.entered)
        self.assertEqual(
            _FakeRetentionService.instances[0].purposes,
            (
                task_cmds.TaskArtifactPurpose.INPUT,
                task_cmds.TaskArtifactPurpose.OUTPUT,
            ),
        )
        self.assertEqual(_FakeRetentionService.instances[0].limit, 2)
        self.assertIn("Task retention sweep processed 2 artifacts.", output)
        self.assertIn('"deleted":1', output)
        self.assertIn('"lost":1', output)
        self.assertNotIn("secret", output)

    def test_retention_sweep_requires_store_and_artifact_root(self) -> None:
        missing_store_console = Console(record=True, width=160)
        with patch.dict(task_cmds.environ, {}, clear=True):
            missing_store = task_cmds.task_retention_sweep(
                Namespace(
                    store_dsn=None, store_schema=None, purpose=(), limit=100
                ),
                missing_store_console,
                self.theme,
            )

        missing_root_console = Console(record=True, width=160)
        with (
            patch.object(task_cmds, "_task_artifact_store", return_value=None),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            missing_root = task_cmds.task_retention_sweep(
                Namespace(
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=100,
                ),
                missing_root_console,
                self.theme,
            )

        self.assertFalse(missing_store)
        self.assertIn("store.missing", missing_store_console.export_text())
        self.assertFalse(missing_root)
        self.assertIn(
            "artifact_store.missing",
            missing_root_console.export_text(),
        )

    def test_retention_sweep_reports_safe_errors(self) -> None:
        cases = (
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=("bad",),
                    limit=100,
                ),
                None,
                "retention.sweep",
            ),
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=0,
                ),
                None,
                "retention.sweep",
            ),
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=100,
                ),
                TaskRetentionStoreNotFoundError("private local path"),
                "artifact_store.missing",
            ),
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=100,
                ),
                ArtifactStoreError("private artifact path"),
                "artifact_store.failure",
            ),
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=100,
                ),
                ImportError("private dependency"),
                "dependency.missing",
            ),
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=100,
                ),
                OSError("private dsn"),
                "io.failure",
            ),
        )
        for args, error, expected in cases:
            console = Console(record=True, width=160)
            database = _FakeResource()
            _FakeRetentionService.instances = []
            _FakeRetentionService.error = error
            _FakeRetentionService.results = ()
            with (
                self.subTest(expected=expected),
                patch.object(
                    task_cmds, "_task_pgsql_database", return_value=database
                ),
                patch.object(
                    task_cmds, "PgsqlTaskStore", return_value=object()
                ),
                patch.object(
                    task_cmds, "_task_artifact_store", return_value=object()
                ),
                patch.object(
                    task_cmds,
                    "TaskRetentionService",
                    _FakeRetentionService,
                ),
            ):
                result = task_cmds.task_retention_sweep(
                    args,
                    console,
                    self.theme,
                )

            output = console.export_text()
            self.assertFalse(result)
            self.assertIn(expected, output)
            self.assertNotIn("secret", output)
            self.assertNotIn("private", output)

    def test_run_awaitable_requests_interrupt_callback(self) -> None:
        interrupted: list[bool] = []

        class _Future:
            def __init__(self, target: Callable[[], None]) -> None:
                self.target = target
                self.calls = 0

            def result(self) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise KeyboardInterrupt()
                self.target()

        class _Executor:
            def __init__(self, max_workers: int) -> None:
                self.max_workers = max_workers

            def __enter__(self) -> "_Executor":
                return self

            def __exit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                return None

            def submit(self, target: Callable[[], None]) -> _Future:
                return _Future(target)

        async def complete() -> bool:
            return True

        with (
            patch.object(task_cmds, "ThreadPoolExecutor", _Executor),
            self.assertRaises(KeyboardInterrupt),
        ):
            task_cmds._run_awaitable(
                complete(),
                on_interrupt=lambda: interrupted.append(True),
            )

        self.assertEqual(interrupted, [True])

    def test_client_context_enters_database_and_stack(self) -> None:
        database = _FakeResource()
        stack_resource = _FakeResource()
        stack = AsyncExitStack()

        async def exercise() -> bool:
            await stack.enter_async_context(stack_resource)
            context = task_cmds._TaskCliClientContext(
                client=object(),
                database=database,
                stack=stack,
            )
            async with context as client:
                self.assertIsNotNone(client)
            return True

        task_cmds._run_awaitable(exercise())

        self.assertTrue(database.opened)
        self.assertTrue(database.closed)
        self.assertTrue(stack_resource.entered)
        self.assertTrue(stack_resource.exited)

    def test_client_context_closes_stack_when_database_open_fails(
        self,
    ) -> None:
        database = _FakeResource(open_error=OSError("private open"))
        stack_resource = _FakeResource()
        stack = AsyncExitStack()

        async def exercise() -> bool:
            await stack.enter_async_context(stack_resource)
            context = task_cmds._TaskCliClientContext(
                client=object(),
                database=database,
                stack=stack,
            )
            with self.assertRaises(OSError):
                async with context:
                    pass
            return True

        task_cmds._run_awaitable(exercise())

        self.assertTrue(database.opened)
        self.assertFalse(database.closed)
        self.assertTrue(stack_resource.entered)
        self.assertTrue(stack_resource.exited)

    def test_client_context_closes_stack_when_database_close_fails(
        self,
    ) -> None:
        database = _FakeResource(close_error=OSError("private close"))
        stack_resource = _FakeResource()
        stack = AsyncExitStack()

        async def exercise() -> bool:
            await stack.enter_async_context(stack_resource)
            context = task_cmds._TaskCliClientContext(
                client=object(),
                database=database,
                stack=stack,
            )
            with self.assertRaises(OSError):
                async with context:
                    pass
            return True

        task_cmds._run_awaitable(exercise())

        self.assertTrue(database.opened)
        self.assertTrue(database.closed)
        self.assertTrue(stack_resource.entered)
        self.assertTrue(stack_resource.exited)

    def test_client_context_handles_optional_resources(self) -> None:
        client = object()
        database = _FakeResource()
        stack_resource = _FakeResource()
        stack = AsyncExitStack()
        failing_database = _FakeResource(open_error=OSError("private open"))

        async def exercise() -> bool:
            async with task_cmds._TaskCliClientContext(
                client=client,
            ) as returned:
                self.assertIs(returned, client)
            async with task_cmds._TaskCliClientContext(
                client=client,
                database=database,
            ):
                pass
            await stack.enter_async_context(stack_resource)
            async with task_cmds._TaskCliClientContext(
                client=client,
                stack=stack,
            ):
                pass
            with self.assertRaises(OSError):
                async with task_cmds._TaskCliClientContext(
                    client=client,
                    database=failing_database,
                ):
                    pass
            return True

        task_cmds._run_awaitable(exercise())

        self.assertTrue(database.opened)
        self.assertTrue(database.closed)
        self.assertTrue(stack_resource.entered)
        self.assertTrue(stack_resource.exited)
        self.assertTrue(failing_database.opened)
        self.assertFalse(failing_database.closed)

    def test_client_context_factories_and_helpers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            definition_path = Path(tmpdir) / "task.toml"
            definition_path.write_text("", encoding="utf-8")
            with (
                patch.object(
                    task_cmds, "_agent_task_target", return_value=object()
                ),
                patch.object(
                    task_cmds, "_task_artifact_store", return_value=None
                ),
                patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
            ):
                ephemeral_context = task_cmds._task_cli_client_context(
                    definition_path,
                    dsn=None,
                    schema=None,
                    queue=False,
                    ephemeral=True,
                    hub=None,
                    logger=None,
                )
            database = _FakeResource()
            with (
                patch.object(
                    task_cmds, "_agent_task_target", return_value=object()
                ),
                patch.object(
                    task_cmds, "_task_pgsql_database", return_value=database
                ),
                patch.object(
                    task_cmds, "PgsqlTaskStore", return_value=object()
                ),
                patch.object(
                    task_cmds, "PgsqlTaskQueue", return_value=object()
                ),
                patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
            ):
                durable_context = task_cmds._task_cli_client_context(
                    definition_path,
                    dsn="postgresql://db/tasks",
                    schema="tasks",
                    queue=True,
                    ephemeral=False,
                    hub=None,
                    logger=None,
                )
            inspection_database = _FakeResource()
            with (
                patch.object(
                    task_cmds,
                    "_task_pgsql_database",
                    return_value=inspection_database,
                ),
                patch.object(
                    task_cmds, "PgsqlTaskStore", return_value=object()
                ),
            ):
                inspection_context = (
                    task_cmds._task_cli_inspection_client_context(
                        Namespace(
                            store_dsn="postgresql://db/tasks",
                            store_schema="tasks",
                        ),
                        Console(record=True, width=160),
                    )
                )

        self.assertIsNone(ephemeral_context.database)
        self.assertIsNotNone(ephemeral_context.client._hmac_provider)
        self.assertIs(database, durable_context.database)
        self.assertIsNotNone(durable_context.client._hmac_provider)
        self.assertIsNotNone(inspection_context)
        assert inspection_context is not None
        self.assertIs(inspection_database, inspection_context.database)

    def test_ephemeral_client_context_cleans_temporary_artifacts(self) -> None:
        async def exercise() -> bool:
            with TemporaryDirectory() as tmpdir:
                definition_path = Path(tmpdir) / "task.toml"
                definition_path.write_text("", encoding="utf-8")
                with (
                    patch.object(
                        task_cmds,
                        "_agent_task_target",
                        return_value=object(),
                    ),
                    patch.object(
                        task_cmds,
                        "_task_artifact_store",
                        return_value=None,
                    ),
                    patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
                ):
                    context = task_cmds._task_cli_client_context(
                        definition_path,
                        dsn=None,
                        schema=None,
                        queue=False,
                        ephemeral=True,
                        hub=None,
                        logger=None,
                        input_value={
                            "source_kind": "local_path",
                            "reference": "sample.pdf",
                        },
                    )
                async with context as client:
                    artifact_store = client._artifact_store
                    self.assertIsNotNone(artifact_store)
                    assert artifact_store is not None
                    artifact_root = artifact_store._root
                    self.assertTrue(artifact_root.exists())
                self.assertFalse(artifact_root.exists())
            return True

        self.assertTrue(task_cmds._run_awaitable(exercise()))

    def test_hmac_provider_uses_environment_key(self) -> None:
        with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
            provider = task_cmds._task_hmac_provider()

        self.assertIsNotNone(provider)
        assert provider is not None
        default_key = provider.hmac_key(purpose=TaskKeyPurpose.PRIVACY_HASH)
        override_key = provider.hmac_key(
            purpose=TaskKeyPurpose.IDEMPOTENCY,
            key_id="cli-test-override",
        )
        self.assertEqual(default_key.key_id, "cli-test-v1")
        self.assertEqual(default_key.algorithm, "hmac-sha256")
        self.assertEqual(default_key.secret, b"task-hmac-test-key")
        self.assertEqual(override_key.key_id, "cli-test-override")

        with self.assertRaises(AssertionError):
            provider.hmac_key(
                purpose=TaskKeyPurpose.PRIVACY_HASH,
                key_id=" ",
            )

    def test_hmac_provider_rejects_incomplete_environment(self) -> None:
        cases = (
            {},
            {"AVALAN_TASK_HMAC_KEY_ID": "cli-test-v1"},
            {
                "AVALAN_TASK_HMAC_KEY_B64": TASK_HMAC_ENV[
                    "AVALAN_TASK_HMAC_KEY_B64"
                ]
            },
            {
                "AVALAN_TASK_HMAC_KEY_ID": "cli-test-v1",
                "AVALAN_TASK_HMAC_KEY_B64": "not base64",
            },
            {
                "AVALAN_TASK_HMAC_KEY_ID": "cli-test-v1",
                "AVALAN_TASK_HMAC_KEY_B64": "",
            },
        )

        for env in cases:
            with (
                self.subTest(env=env),
                patch.dict(
                    task_cmds.environ,
                    env,
                    clear=True,
                ),
            ):
                self.assertIsNone(task_cmds._task_hmac_provider())

        with self.assertRaises(AssertionError):
            task_cmds._TaskCliHmacProvider(key_id="", secret=b"secret")
        with self.assertRaises(AssertionError):
            task_cmds._TaskCliHmacProvider(key_id="cli-test-v1", secret=b"")

    def test_agent_target_and_database_helpers_construct(self) -> None:
        stack = AsyncExitStack()

        target = task_cmds._agent_task_target(
            Path("."),
            hub=None,
            logger=None,
            stack=stack,
        )
        database = task_cmds._task_pgsql_database(
            "postgresql://user:secret@db/tasks",
            "tasks",
        )

        self.assertIsNotNone(target)
        self.assertIsNotNone(database)

    def test_flow_resolver_loads_flow_and_reports_load_issues(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "constant"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "constant"
                value = "ok"
                """,
                encoding="utf-8",
            )
            broken_path = root / "broken.toml"
            broken_path.write_text("[flow]\nname = 'broken'", encoding="utf-8")
            resolver = task_cmds._task_flow_resolver(root)

            flow = resolver(
                _flow_task_context(TaskExecutionTarget.flow("flow.toml"))
            )
            with self.assertRaises(TaskValidationError) as context:
                resolver(
                    _flow_task_context(TaskExecutionTarget.flow("broken.toml"))
                )

        self.assertIsNotNone(flow)
        self.assertEqual(
            context.exception.issues[0].category,
            TaskValidationCategory.UNSUPPORTED,
        )
        self.assertEqual(
            context.exception.issues[0].code,
            "flow.missing_section",
        )

    def test_low_level_helpers_cover_safe_branches(self) -> None:
        console = Console(record=True, width=160)
        task_cmds._print_task_command_error(
            console,
            "message",
            "code.value",
            "hint",
        )
        task_cmds._print_task_result(console, None)
        task_cmds._print_task_execution_error(
            console,
            TaskClientUnsupportedOperationError(
                code="task.unsupported",
                operation="run",
                message="unsupported",
            ),
        )
        task_cmds._print_task_execution_error(console, OSError("private"))
        task_cmds._print_task_execution_error(console, RuntimeError("private"))
        task_cmds._print_task_inspection_error(
            console,
            ImportError("private"),
        )
        task_cmds._print_task_inspection_error(console, OSError("private"))
        task_cmds._print_task_inspection_error(
            console,
            AssertionError("private"),
        )
        task_cmds._print_task_execution_error(
            console,
            TaskValidationError(
                (
                    TaskValidationIssue(
                        code="bad",
                        path="input",
                        message="bad input",
                        hint="fix it",
                        category=TaskValidationCategory.VALUE,
                    ),
                )
            ),
        )
        with self.assertRaises(RuntimeError):
            task_cmds._run_awaitable(_raise_runtime_error())
        with self.assertRaises(TaskClientUnsupportedOperationError):
            task_cmds._run_awaitable(
                task_cmds._task_cli_inspection_target(
                    cast(TaskTargetContext, object())
                )
            )
        self.assertIsNone(
            task_cmds._task_cli_after_sequence(Namespace(after_sequence=None))
        )
        with self.assertRaises(AssertionError):
            task_cmds._task_cli_after_sequence(Namespace(after_sequence=-1))
        with patch.dict(
            task_cmds.environ,
            {
                "AVALAN_TASK_STORE_SCHEMA": "tasks",
                "AVALAN_TASK_ARTIFACT_ROOT": "/tmp/task-artifacts",
            },
            clear=True,
        ):
            self.assertEqual(
                task_cmds._task_store_schema(Namespace()), "tasks"
            )
            self.assertIsNotNone(task_cmds._task_artifact_store())
        with patch.dict(task_cmds.environ, {}, clear=True):
            self.assertIsNone(task_cmds._task_artifact_store())
        self.assertEqual(
            task_cmds._safe_queue_metadata(Namespace(queue="")), {}
        )
        self.assertEqual(
            task_cmds._safe_queue_metadata(Namespace(queue="q")),
            {"cli_queue": "q"},
        )
        self.assertIsNone(task_cmds._task_cli_queue_name(Namespace(queue="")))
        self.assertEqual(
            task_cmds._task_command_metadata(ephemeral=True)["store_mode"],
            "ephemeral-memory",
        )
        self.assertTrue(
            task_cmds._task_cli_contains_local_file(
                {
                    "nested": {
                        "source_kind": "local_path",
                        "reference": "sample.pdf",
                    }
                }
            )
        )
        self.assertTrue(
            task_cmds._task_cli_contains_local_file(
                [{"source_kind": "local_path", "reference": "sample.pdf"}]
            )
        )
        self.assertFalse(
            task_cmds._task_cli_contains_local_file(
                {"nested": {"source_kind": "remote_url", "reference": "url"}}
            )
        )
        event_value = task_cmds._task_event_cli_value(
            SimpleNamespace(
                event_id="event-1",
                run_id="run-1",
                sequence=1,
                event_type="start",
                category=TaskEventCategory.ENGINE,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                attempt_id=None,
                payload=None,
            )
        )
        self.assertNotIn("attempt_id", event_value)
        self.assertNotIn("payload", event_value)

        output = console.export_text()
        self.assertIn("task.unsupported", output)
        self.assertIn("dependency.missing", output)
        self.assertIn("task.inspection", output)
        self.assertIn("task.execution", output)

    def test_validate_task_cli_input_for_command_success_path(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds._validate_task_cli_input_for_command(
            Namespace(
                definition=str(FIXTURE_ROOT / "minimal.task.toml"),
                task_input="Ada Lovelace",
                task_input_json=None,
                task_input_fields=(),
                task_files=(),
            ),
            console,
        )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("Task input is valid.", output)
        self.assertIn("<redacted>", output)

    def test_validate_task_cli_input_for_command_failure_paths(self) -> None:
        cases = (
            (
                str(FIXTURE_ROOT / "missing_sections.task.toml"),
                "Ada",
                None,
                "Task definition could not be loaded.",
            ),
            (
                str(FIXTURE_ROOT / "minimal.task.toml"),
                None,
                "{not json",
                "Task input could not be parsed.",
            ),
            (
                str(FIXTURE_ROOT / "minimal.task.toml"),
                None,
                '{"name":"Ada"}',
                "Task input is invalid.",
            ),
        )
        for definition, task_input, task_input_json, expected in cases:
            console = Console(record=True, width=160)
            with self.subTest(expected=expected):
                result = task_cmds._validate_task_cli_input_for_command(
                    Namespace(
                        definition=definition,
                        task_input=task_input,
                        task_input_json=task_input_json,
                        task_input_fields=(),
                        task_files=(),
                    ),
                    console,
                )

            self.assertFalse(result)
            self.assertIn(expected, console.export_text())

    def test_validate_task_cli_input_for_command_non_input_paths(self) -> None:
        console = Console(record=True, width=160)
        self.assertTrue(
            task_cmds._validate_task_cli_input_for_command(
                Namespace(
                    definition=None,
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                ),
                console,
            )
        )
        self.assertTrue(
            task_cmds._validate_task_cli_input_for_command(
                Namespace(
                    definition=123,
                    task_input="Ada",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                ),
                console,
            )
        )
        self.assertFalse(
            task_cmds._validate_task_cli_input_for_command(
                Namespace(
                    definition="/tmp/private/missing.task.toml",
                    task_input="Ada",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                ),
                console,
            )
        )
        self.assertIn("file.read", console.export_text())


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


def _write_queued_definition(path: Path) -> None:
    path.write_text(
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


def _write_direct_object_definition(path: Path) -> Path:
    definition = path / "direct_object.task.toml"
    definition.write_text(
        """
        [task]
        name = "direct_object"
        version = "1"

        [input]
        type = "string"

        [output]
        type = "object"

        [output.schema]
        type = "object"
        required = ["answer", "input"]

        [output.schema.properties.answer]
        type = "string"

        [output.schema.properties.input]
        type = "string"

        [execution]
        type = "agent"
        ref = "agents/direct_object.toml"
        """,
        encoding="utf-8",
    )
    return definition


def _flow_task_context(execution: TaskExecutionTarget) -> TaskTargetContext:
    return TaskTargetContext(
        definition=TaskDefinition(
            task=TaskMetadata(name="flow", version="1"),
            input=TaskInputContract.object(),
            output=TaskOutputContract.json({}),
            execution=execution,
        ),
        execution=TaskExecutionContext(
            run_id="run-1",
            attempt_id="attempt-1",
            attempt_number=1,
        ),
        input_value={},
        files=(),
        metadata={},
    )


async def _raise_runtime_error() -> bool:
    raise RuntimeError("private")


if __name__ == "__main__":
    main()
