from argparse import Namespace
from asyncio import CancelledError
from io import StringIO
from json import loads
from logging import Logger, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from rich.console import Console
from trigger.preparation_e2e_test import file_task
from trigger.preparation_fault_test import configuration, services
from trigger_cli_helpers import write_application

from avalan.cli.__main__ import CLI
from avalan.cli.commands.trigger import run_trigger_command
from avalan.cli.task_store import TaskStoreConfigurationError
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.task.artifact_ownership import ArtifactStagingOwner
from avalan.task.deployment import ExecutionDeploymentError
from avalan.task.encryption import TaskEncryptionError
from avalan.task.validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
)
from avalan.trigger.admission import TriggerCommitOutcome
from avalan.trigger.apply import (
    TriggerApplyCancelledError,
    TriggerApplyResult,
    TriggerRegistrationService,
)
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.preparation import (
    PreparedTriggerRegistration,
    TriggerPreparationCancelledError,
    TriggerPreparationFailure,
)
from avalan.trigger.records import TriggerSnapshot
from avalan.trigger.scheduler import TriggerSchedulerCancelledError
from avalan.trigger.scheduler_types import TriggerShutdownResult
from avalan.trigger.stores.memory import InMemoryTriggerStore


class TriggerCommandTest(IsolatedAsyncioTestCase):
    async def test_preview_and_validate_are_read_only_and_safe(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            write_application(root)
            trigger = root / "timer.toml"
            trigger.write_text("""[trigger]
schema_version=1
name="timer"
[task]
ref="task.toml"
[input]
value="private input must not be emitted"
[schedule]
type="interval"
every_seconds=60
""")
            cli = CLI(getLogger(__name__))
            hub = HuggingfaceHub(
                "unused", str(root / "cache"), getLogger(__name__)
            )
            for action in ("validate", "preview"):
                argv = ["trigger", action, str(trigger), "--json"]
                if action == "preview":
                    argv.extend(
                        [
                            "--reference-time",
                            "2026-09-07T12:00:00+00:00",
                            "--count",
                            "2",
                        ]
                    )
                args = cli._parser.parse_args(argv)
                output = StringIO()
                with patch(
                    "avalan.cli.commands.trigger.PsycopgAsyncDatabase",
                    side_effect=AssertionError(
                        "read-only command opened database"
                    ),
                ):
                    self.assertTrue(
                        await run_trigger_command(
                            args,
                            Console(file=output, width=10000),
                            hub,
                            getLogger(__name__),
                        )
                    )
                value = loads(output.getvalue())
                self.assertTrue(value["ok"])
                self.assertNotIn("private input", output.getvalue())
                self.assertFalse((root / "manifest.json").exists())
                if action == "preview":
                    self.assertTrue(value["assumed_anchor"])
                    self.assertEqual(len(value["occurrences"]), 2)
                    self.assertEqual(
                        value["occurrences"][0]["utc"],
                        "2026-09-07T12:01:00+00:00",
                    )
            for reference, valid in (
                ("2026-09-07T09:00:00.123456-03:00", True),
                ("2026-09-07T12:00:00.1234567Z", False),
                ("2026-09-07 12:00:00Z", False),
            ):
                args = cli._parser.parse_args(
                    [
                        "trigger",
                        "preview",
                        str(trigger),
                        "--reference-time",
                        reference,
                    ]
                )
                output = StringIO()
                self.assertEqual(
                    await run_trigger_command(
                        args, Console(file=output), hub, getLogger(__name__)
                    ),
                    valid,
                )
                preview = loads(output.getvalue())
                if valid:
                    self.assertEqual(
                        preview["reference_time"],
                        "2026-09-07T12:00:00.123456+00:00",
                    )
                    self.assertEqual(
                        preview["occurrences"][0]["utc"],
                        "2026-09-07T12:01:00.123456+00:00",
                    )
                else:
                    self.assertFalse(preview["ok"])
                    self.assertEqual(preview["path"], "reference_time")
            self.assertFalse(
                await CLI._needs_hf_token(Namespace(command="trigger"))
            )
            self.assertTrue(
                CLI._can_use_anonymous_hub(Namespace(command="trigger"))
            )

    async def test_configuration_error_never_prints_connection_secret(
        self,
    ) -> None:
        cli = CLI(getLogger(__name__))
        args = cli._parser.parse_args(
            [
                "trigger",
                "list",
                "--store-dsn",
                "postgres://secret@localhost/db",
                "--store-schema",
                "bad schema",
            ]
        )
        output = StringIO()
        with TemporaryDirectory() as directory:
            hub = HuggingfaceHub("unused", directory, getLogger(__name__))
            self.assertFalse(
                await run_trigger_command(
                    args, Console(file=output), hub, getLogger(__name__)
                )
            )
        self.assertNotIn("secret", output.getvalue())
        self.assertEqual(loads(output.getvalue())["path"], "store_schema")

    async def test_runnable_examples_validate_and_preview(self) -> None:
        cli = CLI(getLogger(__name__))
        root = Path(__file__).resolve().parents[2] / "docs/examples/triggers"
        with TemporaryDirectory() as directory:
            hub = HuggingfaceHub("unused", directory, getLogger(__name__))
            for path in sorted(root.glob("*.trigger.toml")):
                for action in ("validate", "preview"):
                    args = cli._parser.parse_args(
                        ["trigger", action, str(path)]
                    )
                    output = StringIO()
                    self.assertTrue(
                        await run_trigger_command(
                            args,
                            Console(file=output, width=20),
                            hub,
                            getLogger(__name__),
                        ),
                        output.getvalue(),
                    )
                    self.assertTrue(loads(output.getvalue())["ok"])

    async def test_safe_failure_and_cancellation_records(self) -> None:
        args = Namespace(trigger_command="unused")
        failures = (
            TriggerError(TriggerErrorCode.INVALID_CONFIG, "safe.path"),
            ExecutionDeploymentError("private-path"),
            TaskEncryptionError("private-key"),
            OSError("private-password"),
            TaskValidationError(
                (
                    TaskValidationIssue(
                        category=TaskValidationCategory.VALUE,
                        code="safe.code",
                        path="input",
                        message="private-input",
                        hint="private-hint",
                    ),
                )
            ),
            TriggerPreparationFailure(
                staging=ArtifactStagingOwner(
                    staging_id=str(uuid4()),
                    owner_scope="owner",
                    recovery_id=str(uuid4()),
                )
            ),
        )
        with TemporaryDirectory() as directory:
            hub = HuggingfaceHub("unused", directory, getLogger(__name__))
            for error in failures:
                output = StringIO()
                with patch(
                    "avalan.cli.commands.trigger._run",
                    AsyncMock(side_effect=error),
                ):
                    self.assertFalse(
                        await run_trigger_command(
                            args,
                            Console(file=output, width=20),
                            hub,
                            getLogger(__name__),
                        )
                    )
                result = loads(output.getvalue())
                self.assertFalse(result["ok"])
                self.assertNotIn("private", output.getvalue())
            for cancellation in (
                TriggerPreparationCancelledError(TriggerPreparationFailure()),
                TriggerSchedulerCancelledError(
                    TriggerShutdownResult(pending_operations=1)
                ),
            ):
                output = StringIO()
                with patch(
                    "avalan.cli.commands.trigger._run",
                    AsyncMock(side_effect=cancellation),
                ):
                    with self.assertRaises(type(cancellation)):
                        await run_trigger_command(
                            args,
                            Console(file=output),
                            hub,
                            getLogger(__name__),
                        )
                self.assertFalse(loads(output.getvalue())["ok"])

    async def test_actual_entrypoint_emits_one_json_document(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.dict("os.environ", {"HF_TOKEN": ""}),
        ):
            root = Path(directory)
            write_application(root)
            configuration = root / "trigger.toml"
            configuration.write_text(
                '[trigger]\nschema_version=1\nname="entry"\n[task]\nref="task.toml"\n[input]\nvalue="private"\n[schedule]\ntype="interval"\nevery_seconds=60\n'
            )
            for action in ("preview", "validate"):
                output = StringIO()
                with (
                    patch(
                        "sys.argv",
                        ["avalan", "trigger", action, str(configuration)],
                    ),
                    patch(
                        "avalan.cli.__main__.Console",
                        return_value=Console(file=output, width=20),
                    ),
                ):
                    await CLI(getLogger(__name__))()
                self.assertTrue(loads(output.getvalue())["ok"])
            for failure in (
                TaskStoreConfigurationError("store_schema"),
                TriggerSchedulerCancelledError(
                    TriggerShutdownResult(pending_operations=1)
                ),
            ):
                output = StringIO()
                with (
                    patch("sys.argv", ["avalan", "trigger", "list"]),
                    patch(
                        "avalan.cli.__main__.Console",
                        return_value=Console(file=output, width=20),
                    ),
                    patch(
                        "avalan.cli.commands.trigger._run",
                        AsyncMock(side_effect=failure),
                    ),
                ):
                    with self.assertRaises(
                        (SystemExit, TriggerSchedulerCancelledError)
                    ):
                        await CLI(getLogger(__name__))()
                self.assertFalse(loads(output.getvalue())["ok"])
            # The shared outer boundary also sanitizes failures from the
            # existing task/Flow commands, which do not use trigger._run.
            output = StringIO()
            with (
                patch("sys.argv", ["avalan", "trigger", "list"]),
                patch(
                    "avalan.cli.__main__.Console",
                    return_value=Console(file=output, width=20),
                ),
                patch.object(
                    CLI,
                    "_main",
                    AsyncMock(
                        side_effect=TaskStoreConfigurationError("store_schema")
                    ),
                ),
            ):
                with self.assertRaises(SystemExit):
                    await CLI(getLogger(__name__))()
            self.assertEqual(loads(output.getvalue())["path"], "store_schema")

    async def test_interrupted_apply_reports_recovered_and_unknown_outcomes(
        self,
    ) -> None:
        for unknown in (False, True):
            with TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "input.txt").write_text("private input")
                preparation = await services(root)
                prepared = await preparation.prepare_registration(
                    configuration(), file_task()
                )
                registration = TriggerRegistrationService(preparation)
                original = registration._apply_memory

                async def commit_then_cancel(
                    store: InMemoryTriggerStore,
                    value: PreparedTriggerRegistration,
                    fingerprint: str,
                    expected_generation: int | None,
                ) -> TriggerSnapshot:
                    await original(
                        store, value, fingerprint, expected_generation
                    )
                    raise CancelledError("private transport credential")

                async def apply(
                    _args: Namespace, _hub: HuggingfaceHub, _logger: Logger
                ) -> TriggerApplyResult:
                    return await registration.apply(
                        prepared, expected_generation=None
                    )

                recovery = (
                    AsyncMock(
                        side_effect=CancelledError(
                            "private recovery credential"
                        )
                    )
                    if unknown
                    else registration.recover
                )
                output = StringIO()
                with (
                    patch.object(
                        registration, "_apply_memory", commit_then_cancel
                    ),
                    patch.object(registration, "recover", recovery),
                    patch(
                        "avalan.cli.commands.trigger._run",
                        AsyncMock(side_effect=apply),
                    ),
                ):
                    with self.assertRaises(
                        TriggerApplyCancelledError
                    ) as interrupted:
                        await run_trigger_command(
                            Namespace(trigger_command="apply"),
                            Console(file=output, width=20),
                            HuggingfaceHub(
                                "unused",
                                str(root / "cache"),
                                getLogger(__name__),
                            ),
                            getLogger(__name__),
                        )
                expected = (
                    TriggerCommitOutcome.UNKNOWN
                    if unknown
                    else TriggerCommitOutcome.COMMITTED
                )
                self.assertEqual(
                    interrupted.exception.result.outcome, expected
                )
                result = loads(output.getvalue())
                self.assertEqual(len(output.getvalue().splitlines()), 1)
                self.assertFalse(result["ok"])
                self.assertEqual(result["outcome"], expected.value)
                self.assertEqual(
                    result["recovery_id"], prepared.staging.recovery_id
                )
                self.assertEqual(
                    result["cleanup_pending"],
                    interrupted.exception.result.cleanup_pending,
                )
                self.assertNotIn("private", output.getvalue())
                self.assertNotIn("credential", output.getvalue())
                self.assertNotIn("ciphertext", output.getvalue())
                self.assertNotIn(str(root), output.getvalue())
                self.assertIsNotNone(
                    await preparation.admission.store.inspect("daily")
                )
                if unknown:
                    self.assertIsNone(result["trigger"])
                else:
                    self.assertEqual(result["trigger"]["name"], "daily")
