from argparse import Namespace
from base64 import b64encode
from contextlib import AsyncExitStack
from io import StringIO
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from rich.console import Console

from avalan.cli.__main__ import CLI
from avalan.cli.commands.task import _task_retention_sweep, _task_worker
from avalan.cli.trigger_runtime import encrypted_input_store
from avalan.cli.trigger_worker import deployment_worker
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.pgsql import PsycopgAsyncDatabase, PsycopgPoolSettings
from avalan.task.deployment import ExecutionDeploymentError
from avalan.task.encryption import TaskEncryptionError
from avalan.task.worker import TaskWorkerShutdown


class TriggerRuntimeTest(IsolatedAsyncioTestCase):
    async def test_invalid_storage_policy_and_empty_worker_are_rejected(
        self,
    ) -> None:
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(dsn="postgresql://unused/db")
        )
        with (
            TemporaryDirectory() as directory,
            patch.dict(
                "os.environ",
                {
                    "AVALAN_TASK_ENCRYPTION_KEY_ID": "test",
                    "AVALAN_TASK_ENCRYPTION_KEY_B64": (
                        b64encode(b"x" * 32).decode()
                    ),
                },
            ),
        ):
            args = Namespace(
                raw_storage_allowed=True,
                retention_days=30,
                max_artifact_bytes=1024,
                deployment_root=directory,
            )
            for name, values in (
                ("retention_days", (True, "30", 0, 3651)),
                ("max_artifact_bytes", (False, "100", 0, 1073741825)),
            ):
                original = getattr(args, name)
                for value in values:
                    setattr(args, name, value)
                    with self.assertRaisesRegex(TaskEncryptionError, name):
                        encrypted_input_store(args, database)
                setattr(args, name, original)
            async with AsyncExitStack() as stack:
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "hub_required"
                ):
                    await deployment_worker(
                        args,
                        database=database,
                        stack=stack,
                        hub=None,
                        logger=None,
                        shutdown=TaskWorkerShutdown(),
                        hmac_provider=None,
                    )
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "catalog_empty"
                ):
                    await deployment_worker(
                        args,
                        database=database,
                        stack=stack,
                        hub=HuggingfaceHub(
                            "unused",
                            str(Path(directory) / "cache"),
                            getLogger(__name__),
                        ),
                        logger=None,
                        shutdown=TaskWorkerShutdown(),
                        hmac_provider=None,
                    )
        self.assertIsNone(database._pool)

    async def test_existing_task_boundaries_report_host_policy_failures(
        self,
    ) -> None:
        cli = CLI(getLogger(__name__))
        with patch.dict(
            "os.environ", {"AVALAN_TASK_STORE_DSN": "postgresql://unused/db"}
        ):
            args = cli._parser.parse_args(
                ["task", "retention-sweep", "--encrypted-artifacts"]
            )
            output = StringIO()
            self.assertFalse(
                await _task_retention_sweep(args, Console(file=output))
            )
            self.assertIn("artifact.encryption", output.getvalue())
            worker = cli._parser.parse_args(
                [
                    "task",
                    "worker",
                    "--deployment-root",
                    "/unused",
                    "--tool",
                    "search.*",
                ]
            )
            output = StringIO()
            with patch(
                "avalan.cli.commands.task.require_feature", return_value=()
            ):
                self.assertFalse(
                    await _task_worker(
                        worker, Console(file=output), hub=None, logger=None
                    )
                )
            self.assertIn("worker.deployment_host", output.getvalue())
