"""Prove scheduler restart and shutdown across fresh OS processes."""

from asyncio import to_thread
from json import loads
from os import environ
from pathlib import Path
from selectors import EVENT_READ, DefaultSelector
from signal import SIGTERM
from subprocess import PIPE, Popen
from sys import executable
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from pgsql_harness import (
    drop_task_pgsql_schema,
    isolated_task_pgsql_schema,
    real_task_pgsql_dsn,
)
from trigger_scheduler_process import execute

from avalan.task.stores.pgsql import (
    PgsqlTaskMigrationSettings,
    task_pgsql_upgrade,
)


class TriggerSchedulerProcessTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        dsn = real_task_pgsql_dsn()
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        self.dsn = dsn
        self.schema = isolated_task_pgsql_schema("scheduler_process")
        task_pgsql_upgrade(
            PgsqlTaskMigrationSettings(url=dsn, schema=self.schema)
        )

    async def asyncTearDown(self) -> None:
        if hasattr(self, "schema"):
            await drop_task_pgsql_schema(self.dsn, self.schema)

    def start(self, action: str, root: Path) -> Popen[str]:
        environment = dict(environ)
        environment["PYTHONPATH"] = (
            str(Path("src").resolve()) + ":" + str(Path("tests").resolve())
        )
        return Popen(
            [
                executable,
                str(Path(__file__).with_name("trigger_scheduler_process.py")),
                action,
                self.schema,
                str(root),
            ],
            env=environment,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
        )

    async def finish(self, process: Popen[str], expected: int = 0) -> str:
        try:
            out, error = await to_thread(process.communicate, timeout=20)
        except BaseException:
            process.kill()
            await to_thread(process.communicate, timeout=5)
            raise
        assert process.returncode == expected, error
        return out

    async def test_lost_ack_restart_recovers_one_durable_run(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_bytes(
                b"durable scheduler process input"
            )
            registration = await self.finish(self.start("register", root))
            assert loads(registration)["registered"] == "committed"
            (root / "input.txt").unlink()
            await self.finish(self.start("crash", root), expected=23)
            recovered = loads(await self.finish(self.start("recover", root)))
            assert recovered == {
                "recovered": "committed",
                "admitted": 0,
                "occurrences": 1,
            }
            # The in-process caller observes the same durable recovery and
            # decrypted bytes after both fresh-process clients have exited.
            await execute("recover", self.schema, root)

    async def test_signal_shutdown_settles_prepared_work_before_exit(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_bytes(
                b"durable scheduler process input"
            )
            await self.finish(self.start("register", root))
            process = self.start("serve", root)
            try:
                assert process.stdout is not None
                with DefaultSelector() as selector:
                    selector.register(process.stdout, EVENT_READ)
                    assert await to_thread(
                        selector.select, 10
                    ), "child did not prepare"
                    assert loads(process.stdout.readline()) == {"prepared": 1}
                process.send_signal(SIGTERM)
                stopped = loads(await self.finish(process))
                assert stopped == {"settled": True, "pending": 0}
            finally:
                if process.poll() is None:
                    process.kill()
                    await to_thread(process.communicate, timeout=5)
            restarted = loads(await self.finish(self.start("tick", root)))
            assert restarted == {"admitted": 1, "pending": 0, "errors": []}

    async def test_global_budget_rolls_back_sql_recheck_before_restart(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_bytes(
                b"durable scheduler process input"
            )
            await self.finish(self.start("register", root))
            await execute("budget", self.schema, root)
            restarted = loads(await self.finish(self.start("tick", root)))
            assert restarted == {"admitted": 1, "pending": 0, "errors": []}

    async def test_advancing_sql_clock_revisits_initial_conflict(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_bytes(
                b"durable scheduler process input"
            )
            await execute("fairness", self.schema, root)
