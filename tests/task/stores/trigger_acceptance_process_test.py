"""Reconcile three schedule kinds across independent durable CLI processes."""

from asyncio import gather, to_thread
from base64 import b64encode
from datetime import UTC, datetime
from json import dumps, loads
from os import environ
from pathlib import Path
from selectors import EVENT_READ, DefaultSelector
from subprocess import PIPE, Popen
from sys import executable
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from pgsql_harness import (
    drop_task_pgsql_schema,
    isolated_task_pgsql_schema,
    real_task_pgsql_dsn,
    task_pgsql_psycopg_dsn,
)

from avalan.pgsql import PsycopgAsyncDatabase, PsycopgPoolSettings
from avalan.task.encryption import AesGcmTaskCipher
from avalan.task.privacy import decrypt_encrypted_privacy_value
from avalan.task.state import TaskRunState
from avalan.task.stores.pgsql import (
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
    task_pgsql_upgrade,
)
from avalan.trigger.codec import encode_record
from avalan.trigger.records import (
    OccurrenceDisposition,
    OwnerScopeId,
    TriggerSnapshot,
    TriggerStatus,
)
from avalan.trigger.stores.pgsql import PgsqlTriggerStore, decision_time


class TriggerAcceptanceProcessTest(IsolatedAsyncioTestCase):
    async def test_three_schedules_two_processes_and_native_workers(
        self,
    ) -> None:
        dsn = real_task_pgsql_dsn()
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        schema = isolated_task_pgsql_schema("trigger_acceptance")
        task_pgsql_upgrade(PgsqlTaskMigrationSettings(url=dsn, schema=schema))
        processes: list[Popen[str]] = []
        try:
            with TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                environment = dict(environ)
                environment.update(
                    {
                        "PYTHONPATH": ":".join(
                            str(Path(path).resolve())
                            for path in ("src", "tests", "tests/task/stores")
                        ),
                        "AVALAN_TASK_STORE_DSN": task_pgsql_psycopg_dsn(dsn),
                        "AVALAN_TASK_STORE_SCHEMA": schema,
                        "AVALAN_TASK_OWNER_SCOPE": "acceptance-owner",
                        "AVALAN_TASK_ENCRYPTION_KEY_ID": "acceptance",
                        "AVALAN_TASK_HMAC_KEY_ID": "first",
                        "AVALAN_TASK_HMAC_KEY_B64": (
                            b64encode(b"h" * 32).decode()
                        ),
                        "AVALAN_TASK_ENCRYPTION_KEY_B64": (
                            b64encode(b"a" * 32).decode()
                        ),
                    }
                )

                def start(action: str) -> Popen[str]:
                    child = Popen(
                        [
                            executable,
                            str(
                                Path(__file__).with_name(
                                    "trigger_acceptance_process.py"
                                )
                            ),
                            action,
                            str(root),
                        ],
                        env=environment,
                        stdin=PIPE,
                        stdout=PIPE,
                        stderr=PIPE,
                        text=True,
                    )
                    processes.append(child)
                    return child

                async def finish(child: Popen[str], expected: int = 0) -> str:
                    output, error = await to_thread(
                        child.communicate, timeout=60
                    )
                    self.assertEqual(child.returncode, expected, error)
                    self.assertNotIn("frozen private input", output)
                    self.assertNotIn(
                        environment["AVALAN_TASK_ENCRYPTION_KEY_B64"], output
                    )
                    return output

                registration = await finish(start("register"))
                self.assertEqual(len(registration.splitlines()), 3)
                preparation_database = PsycopgAsyncDatabase(
                    PsycopgPoolSettings(
                        dsn=task_pgsql_psycopg_dsn(dsn), schema=schema
                    )
                )
                await preparation_database.open()
                try:
                    trigger_store = PgsqlTriggerStore(
                        preparation_database,
                        OwnerScopeId(value="acceptance-owner"),
                    )
                    plans = []
                    for name in ("interval", "cron", "at"):
                        snapshot = await trigger_store.inspect(name)
                        assert snapshot is not None
                        plans.append(
                            {
                                "definition": encode_record(
                                    snapshot.definition
                                ),
                                "state": encode_record(snapshot.state),
                                "prepared_at": datetime.now(UTC).isoformat(),
                            }
                        )
                    (root / "plans.json").write_text(dumps(plans))
                finally:
                    await preparation_database.aclose()
                # Both interpreters exist concurrently; no in-process scheduler
                # or shared Python catalog participates in this race.
                first, second = start("tick"), start("tick")
                self.assertNotEqual(first.pid, second.pid)
                ticks = await gather(finish(first), finish(second))
                self.assertTrue(
                    all(0 <= loads(value)["admitted"] <= 3 for value in ticks)
                )
                # A competing tick may recover the same committed decisions;
                # counters are observations, not distinct database admissions.
                database = PsycopgAsyncDatabase(
                    PsycopgPoolSettings(
                        dsn=task_pgsql_psycopg_dsn(dsn), schema=schema
                    )
                )
                await database.open()
                try:
                    async with database.connection() as connection:
                        async with connection.cursor() as cursor:
                            await cursor.execute(
                                "SELECT (SELECT count(*) FROM"
                                " trigger_occurrences) AS occurrences, (SELECT"
                                " count(*) FROM task_runs) AS runs, (SELECT"
                                " count(*) FROM task_queue_items) AS queued"
                            )
                            self.assertEqual(
                                await cursor.fetchone(),
                                {"occurrences": 3, "runs": 3, "queued": 3},
                            )
                            await cursor.execute(
                                "SELECT run_id FROM task_runs ORDER BY run_id"
                            )
                            rows = await cursor.fetchall()
                    # Optional expiry is controlled SQL fixture state;
                    # the mandatory occurrence rows remain immutable.
                    async with database.connection() as connection:
                        async with connection.transaction():
                            async with connection.cursor() as cursor:
                                await cursor.execute(
                                    "UPDATE task_idempotency_keys SET"
                                    " created_at = CURRENT_TIMESTAMP -"
                                    " INTERVAL '2 seconds', expires_at ="
                                    " CURRENT_TIMESTAMP - INTERVAL '1 second'"
                                )
                    async with database.connection() as connection:
                        async with connection.cursor() as cursor:
                            await cursor.execute(
                                "SELECT count(*) AS expired FROM"
                                " task_idempotency_keys WHERE expires_at <="
                                " clock_timestamp()"
                            )
                            self.assertEqual(
                                await cursor.fetchone(), {"expired": 1}
                            )
                    competitors = []
                    for key_id, key in (
                        ("rotated-one", b"b" * 32),
                        ("rotated-two", b"c" * 32),
                    ):
                        environment["AVALAN_TASK_HMAC_KEY_ID"] = key_id
                        environment["AVALAN_TASK_HMAC_KEY_B64"] = b64encode(
                            key
                        ).decode()
                        competitor = start("compete")
                        competitors.append(competitor)
                    for competitor in competitors:
                        assert competitor.stdout is not None
                        with DefaultSelector() as selector:
                            selector.register(competitor.stdout, EVENT_READ)
                            assert await to_thread(
                                selector.select, 30
                            ), "competitor did not prepare"
                            self.assertEqual(
                                loads(competitor.stdout.readline()),
                                {"prepared": 3},
                            )
                    for competitor in competitors:
                        assert competitor.stdin is not None
                        competitor.stdin.write("admit\n")
                        competitor.stdin.flush()
                    recovered = [
                        loads(value)
                        for value in await gather(
                            *(finish(child) for child in competitors)
                        )
                    ]
                    self.assertEqual(recovered[0], recovered[1])
                    self.assertEqual(
                        recovered[0]["outcomes"], ["committed"] * 3
                    )
                    self.assertEqual(len(set(recovered[0]["runs"])), 3)
                    self.assertEqual(
                        set(recovered[0]["runs"]),
                        {str(row["run_id"]) for row in rows},
                    )
                    async with database.connection() as connection:
                        async with connection.cursor() as cursor:
                            await cursor.execute(
                                "SELECT count(*) AS runs FROM task_runs"
                            )
                            self.assertEqual(
                                await cursor.fetchone(), {"runs": 3}
                            )
                            await cursor.execute(
                                "SELECT count(*) AS reservations FROM"
                                " task_idempotency_keys"
                            )
                            self.assertEqual(
                                await cursor.fetchone(), {"reservations": 1}
                            )
                    tasks = PgsqlTaskStore(database)
                    identities = []
                    for row in rows:
                        run_id = str(row["run_id"])
                        before = await tasks.get_run(run_id)
                        self.assertEqual(before.state, TaskRunState.QUEUED)
                        self.assertIsNotNone(before.request.trigger)
                        self.assertNotIn(
                            "frozen private input",
                            repr(before.request.input_payload),
                        )
                        identities.append((run_id, before.request.trigger))

                    overlap_store = PgsqlTriggerStore(
                        database, OwnerScopeId(value="acceptance-owner")
                    )

                    async def due_replacement(
                        name: str,
                    ) -> tuple[TriggerSnapshot, set[str]]:
                        current = await overlap_store.inspect(name)
                        assert (
                            current is not None
                            and current.state.next_at is not None
                        )
                        self.assertEqual(
                            current.state.status, TriggerStatus.ACTIVE
                        )
                        for _ in range(10000):
                            async with overlap_store.transaction() as unit:
                                now = await decision_time(unit)
                            if now >= current.state.next_at:
                                break
                        else:
                            self.fail("intended replacement was not due")
                        prior = await overlap_store.occurrences(
                            name, limit=200, newest_first=True
                        )
                        return current, {
                            item.occurrence_id for item in prior.items
                        }

                    async def new_overlap(
                        before: TriggerSnapshot,
                        prior: set[str],
                        blocker: str,
                        state: TaskRunState,
                    ) -> None:
                        self.assertEqual(
                            (await tasks.get_run(blocker)).state, state
                        )
                        current = await overlap_store.inspect(
                            before.definition.name
                        )
                        assert current is not None
                        self.assertEqual(
                            current.state.status, TriggerStatus.ACTIVE
                        )
                        self.assertEqual(
                            current.state.revision, before.state.revision
                        )
                        assert before.state.next_at is not None
                        history = await overlap_store.occurrences(
                            before.definition.name,
                            limit=200,
                            newest_first=True,
                        )
                        fresh = tuple(
                            item
                            for item in history.items
                            if item.occurrence_id not in prior
                            and item.revision == before.state.revision
                            and item.scheduled_at >= before.state.next_at
                            and item.disposition
                            == OccurrenceDisposition.SKIPPED_OVERLAP
                        )
                        self.assertTrue(
                            fresh, "no new current-revision overlap decision"
                        )
                        self.assertTrue(
                            all(item.run_id is None for item in fresh)
                        )

                    async def retry_future(run_id: str) -> None:
                        async with database.connection() as connection:
                            async with connection.cursor() as cursor:
                                await cursor.execute(
                                    "SELECT available_at > clock_timestamp()"
                                    " AS future FROM task_queue_items WHERE"
                                    " run_id = %s AND state = 'available'",
                                    (run_id,),
                                )
                                self.assertEqual(
                                    await cursor.fetchone(), {"future": True}
                                )
                        self.assertEqual(
                            (await tasks.get_run(run_id)).state,
                            TaskRunState.QUEUED,
                        )

                    async def prove_overlap(
                        run_id: str, *, held_retry: bool = False
                    ) -> None:
                        blocker_state = (await tasks.get_run(run_id)).state
                        (root / "control-run.txt").write_text(run_id)
                        applied = loads(await finish(start("control-prepare")))
                        self.assertTrue(applied["ok"])
                        before, prior = await due_replacement(
                            applied["trigger"]["name"]
                        )
                        self.assertEqual(
                            before.state.revision,
                            applied["trigger"]["revision"],
                        )
                        if held_retry:
                            await retry_future(run_id)
                        first_control, second_control = (
                            start("overlap-tick"),
                            start("overlap-tick"),
                        )
                        observations = await gather(
                            finish(first_control), finish(second_control)
                        )
                        for value in observations:
                            tick = loads(value)
                            self.assertEqual(tick["admitted"], 0)
                            self.assertEqual(tick["errors"], [])
                            self.assertEqual(tick["pending_operations"], 0)
                            self.assertTrue(tick["shutdown_settled"])
                        if held_retry:
                            await retry_future(run_id)
                        await new_overlap(before, prior, run_id, blocker_state)
                        self.assertTrue(
                            loads(await finish(start("control-pause")))["ok"]
                        )

                    # Actual queued state on the original admitted revision
                    # blocks a new revision in both independent schedulers.
                    await prove_overlap(identities[0][0])
                    await finish(start("worker-retry"))
                    failed_runs = []
                    for run_id, _ in identities:
                        attempts = await tasks.list_attempts(run_id)
                        if attempts:
                            self.assertEqual(len(attempts), 1)
                            self.assertEqual(attempts[0].state.value, "failed")
                            failed_runs.append(run_id)
                    self.assertEqual(len(failed_runs), 1)
                    async with database.connection() as connection:
                        async with connection.cursor() as cursor:
                            await cursor.execute(
                                "SELECT available_at > updated_at AS delayed"
                                " FROM task_queue_items WHERE run_id = %s",
                                (failed_runs[0],),
                            )
                            self.assertEqual(
                                await cursor.fetchone(), {"delayed": True}
                            )
                    # Bounded test-only hold of the real retry queue entry.
                    # Preserve the actual failed attempt and runtime backoff;
                    # extend only availability across the process race.
                    failed_attempts = await tasks.list_attempts(failed_runs[0])
                    async with database.connection() as connection:
                        async with connection.transaction():
                            async with connection.cursor() as cursor:
                                await cursor.execute(
                                    "UPDATE task_queue_items SET available_at"
                                    " = clock_timestamp() + INTERVAL '3"
                                    " minutes' WHERE run_id = %s AND state ="
                                    " 'available'",
                                    (failed_runs[0],),
                                )
                    await prove_overlap(failed_runs[0], held_retry=True)
                    self.assertEqual(
                        await tasks.list_attempts(failed_runs[0]),
                        failed_attempts,
                    )
                    # Release only after independent before/after SQL-clock
                    # proofs, preserved attempt and new skip evidence.
                    async with database.connection() as connection:
                        async with connection.transaction():
                            async with connection.cursor() as cursor:
                                await cursor.execute(
                                    "UPDATE task_queue_items SET available_at"
                                    " = clock_timestamp() WHERE run_id = %s"
                                    " AND state = 'available'",
                                    (failed_runs[0],),
                                )
                    # The failed attempt is settled before a new interpreter
                    # claims its retry; the run and frozen request stay put.
                    for _ in range(3):
                        self.assertEqual(
                            loads(await finish(start("worker"))),
                            {"executed": 1, "retained": True},
                        )
                    attempt_count = 0
                    for run_id, provenance in identities:
                        after = await tasks.get_run(run_id)
                        self.assertEqual(after.state, TaskRunState.SUCCEEDED)
                        self.assertEqual(after.request.trigger, provenance)
                        assert after.result is not None
                        self.assertEqual(
                            decrypt_encrypted_privacy_value(
                                after.result.output_summary,
                                decryption_provider=AesGcmTaskCipher(
                                    key_id="acceptance", key=b"a" * 32
                                ),
                            ),
                            "durable native output",
                        )
                        totals = await tasks.usage_totals(run_id)
                        self.assertFalse(totals.has_observations)
                        attempts = await tasks.list_attempts(run_id)
                        attempt_count += len(attempts)
                        self.assertIn(len(attempts), (1, 2))
                        self.assertEqual(
                            attempts[0].context.trigger, provenance
                        )
                    self.assertEqual(attempt_count, 4)
                    (root / "run-ids.txt").write_text(
                        "\n".join(run_id for run_id, _ in identities)
                    )
                    usage = await finish(start("usage"))
                    self.assertEqual(len(usage.splitlines()), 3)
                    for line in usage.splitlines():
                        public_usage = loads(line)
                        self.assertEqual(public_usage["usage"], [])
                        self.assertTrue(
                            all(
                                value is None
                                for value in public_usage[
                                    "usage_totals"
                                ].values()
                            )
                        )
                    self.assertEqual(
                        loads(await finish(start("tick")))["admitted"], 0
                    )
                    inspected = await finish(start("inspect"))
                    self.assertEqual(len(inspected.splitlines()), 6)
                    for line in inspected.splitlines()[::2]:
                        snapshot_json = loads(line)
                        self.assertTrue(snapshot_json["ok"])
                        for recent in snapshot_json["recent_occurrences"]:
                            run_id = recent["occurrence"]["payload"]["run_id"]
                            self.assertEqual(
                                recent["task_state"],
                                (
                                    (await tasks.get_run(run_id)).state.value
                                    if run_id
                                    else None
                                ),
                            )
                    self.assertEqual(
                        loads(await finish(start("resume-register"))),
                        {"registered": True},
                    )
                    suspended_worker = start("resume-suspend")
                    assert suspended_worker.stdout is not None
                    with DefaultSelector() as selector:
                        selector.register(suspended_worker.stdout, EVENT_READ)
                        assert await to_thread(selector.select, 30)
                        self.assertEqual(
                            loads(suspended_worker.stdout.readline()),
                            {"input_required": True, "replacement": True},
                        )
                    resume_before, resume_prior = await due_replacement(
                        "durable-resume"
                    )
                    resumed_history = await overlap_store.occurrences(
                        "durable-resume"
                    )
                    blockers = tuple(
                        item.run_id
                        for item in resumed_history.items
                        if item.run_id is not None
                    )
                    self.assertEqual(len(blockers), 1)
                    self.assertEqual(
                        (await tasks.get_run(blockers[0])).state,
                        TaskRunState.INPUT_REQUIRED,
                    )
                    overlap_first, overlap_second = (
                        start("resume-overlap"),
                        start("resume-overlap"),
                    )
                    overlap = await gather(
                        finish(overlap_first), finish(overlap_second)
                    )
                    self.assertTrue(
                        all(
                            loads(value)
                            == {"admitted": 0, "shutdown_settled": True}
                            for value in overlap
                        )
                    )
                    await new_overlap(
                        resume_before,
                        resume_prior,
                        blockers[0],
                        TaskRunState.INPUT_REQUIRED,
                    )
                    assert suspended_worker.stdin is not None
                    suspended_worker.stdin.write("resume\n")
                    suspended_worker.stdin.flush()
                    self.assertEqual(
                        loads(await finish(suspended_worker)),
                        {
                            "suspended": True,
                            "answered": True,
                            "replaced": True,
                        },
                    )
                    self.assertEqual(
                        loads(await finish(start("resume-finish"))),
                        {
                            "resumed": True,
                            "old_deployment": True,
                            "attempts": 1,
                        },
                    )
                    await finish(start("cancel-register"))
                    self.assertEqual(
                        loads(await finish(start("tick")))["admitted"], 1
                    )
                    cancel_store = PgsqlTriggerStore(
                        database, OwnerScopeId(value="acceptance-owner")
                    )
                    cancel_occurrences = await cancel_store.occurrences(
                        "cancel"
                    )
                    self.assertEqual(len(cancel_occurrences.items), 1)
                    cancel_run_id = cancel_occurrences.items[0].run_id
                    assert cancel_run_id is not None
                    cancelled = await tasks.transition_run(
                        cancel_run_id,
                        from_states=(TaskRunState.QUEUED,),
                        to_state=TaskRunState.CANCEL_REQUESTED,
                        reason="acceptance_cancel_request",
                    )
                    self.assertEqual(
                        cancelled.state, TaskRunState.CANCEL_REQUESTED
                    )
                    await prove_overlap(cancel_run_id)
                    self.assertEqual(
                        (await tasks.get_run(cancel_run_id)).state,
                        TaskRunState.CANCEL_REQUESTED,
                    )
                    await finish(start("files-register"))
                    await finish(start("files-crash"), expected=23)
                    self.assertEqual(
                        loads(await finish(start("files-recover"))),
                        {
                            "recovered": "committed",
                            "artifacts": 1,
                            "cursor_advanced": True,
                        },
                    )
                    async with database.connection() as connection:
                        async with connection.cursor() as cursor:
                            await cursor.execute(
                                "SELECT (SELECT count(*) FROM task_runs) AS"
                                " runs, (SELECT count(*) FROM task_runs WHERE"
                                " state='succeeded') AS succeeded, (SELECT"
                                " count(*) FROM task_runs WHERE"
                                " state='queued') AS queued, (SELECT count(*)"
                                " FROM task_runs WHERE"
                                " state='cancel_requested') AS"
                                " cancel_requested, (SELECT count(*) FROM"
                                " task_attempts) AS attempts, (SELECT count(*)"
                                " FROM task_queue_items WHERE"
                                " state='available') AS available, (SELECT"
                                " count(*) FROM task_artifacts) AS artifacts"
                            )
                            final_counts = await cursor.fetchone()
                    self.assertEqual(
                        final_counts,
                        {
                            "runs": 6,
                            "succeeded": 4,
                            "queued": 1,
                            "cancel_requested": 1,
                            "attempts": 5,
                            "available": 2,
                            "artifacts": 1,
                        },
                    )
                    await finish(start("catchup-register"))
                    catchup = loads(await finish(start("catchup-tick")))
                    self.assertEqual(catchup["admitted"], 3)
                    self.assertEqual(
                        loads(await finish(start("catchup-inspect"))),
                        {
                            "skip": 0,
                            "latest": 1,
                            "all": 2,
                            "compressed": True,
                            "cursor_retained": True,
                        },
                    )
                    async with database.connection() as connection:
                        async with connection.cursor() as cursor:
                            await cursor.execute(
                                "SELECT count(*) AS runs FROM task_runs"
                            )
                            self.assertEqual(
                                await cursor.fetchone(), {"runs": 9}
                            )
                    await finish(start("failure-register"))
                    self.assertEqual(
                        loads(await finish(start("failure-tick"))),
                        {
                            "failures": 1,
                            "status": "active",
                            "admitted": 1,
                            "admission_retries": 0,
                            "good_runs": 1,
                        },
                    )
                    self.assertEqual(
                        loads(await finish(start("failure-tick"))),
                        {
                            "failures": 2,
                            "status": "error",
                            "admitted": 0,
                            "admission_retries": 1,
                            "good_runs": 1,
                        },
                    )
                    print(
                        dumps(
                            {
                                "processes": len(processes),
                                "exit_codes": [
                                    child.returncode for child in processes
                                ],
                                "pre_catchup_counts": final_counts,
                            }
                        ),
                        flush=True,
                    )
                finally:
                    await database.aclose()
        finally:
            for child in processes:
                if child.poll() is None:
                    child.kill()
                await to_thread(child.communicate, timeout=10)
            await drop_task_pgsql_schema(dsn, schema)
