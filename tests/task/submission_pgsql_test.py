from asyncio import Event, create_task, sleep
from dataclasses import replace
from json import loads
from typing import cast
from unittest import IsolatedAsyncioTestCase

from avalan.pgsql import (
    PgsqlConnection,
    PgsqlCursor,
    PgsqlParameters,
    PgsqlRow,
    PgsqlUnitOfWork,
)
from avalan.task.queues.submission import (
    TaskSubmissionEvidence,
    insert_task_submission,
    lock_task_submission,
    read_task_submission,
    submission_evidence_from_envelope,
)
from avalan.task.store import TaskExecutionRequest
from avalan.task.submission import (
    _PREPARATION_AUTHORITY,
    PreparedTaskSubmission,
)


def _prepared() -> PreparedTaskSubmission:
    return PreparedTaskSubmission(
        submission_id="submission",
        run_id="proposed-run",
        owner_scope="owner",
        execution=TaskExecutionRequest(definition_id="definition", queue="q"),
        priority=0,
        available_at=None,
        idempotency=None,
        idempotency_expires_at=None,
        artifacts=(),
        temporary_artifacts=(),
        run_metadata={},
        queue_metadata={},
        _authority=_PREPARATION_AUTHORITY,
    )


def _evidence() -> TaskSubmissionEvidence:
    return TaskSubmissionEvidence(
        owner_scope="owner",
        submission_id="submission",
        run_id="existing-run",
        definition_id="definition",
        fingerprint="a" * 64,
        created=False,
    )


class SuppliedCursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, PgsqlParameters]] = []
        self.row: PgsqlRow | None = None
        self.isolation = "read committed"
        self.writer_finished = Event()
        self.writer_finished.set()
        self.barrier_entered = Event()

    async def execute(
        self, query: str, parameters: PgsqlParameters = None
    ) -> None:
        self.statements.append((query, parameters))
        if "pg_advisory_xact_lock" in query:
            self.barrier_entered.set()
            await self.writer_finished.wait()

    async def fetchone(self) -> PgsqlRow | None:
        if "transaction_isolation" in self.statements[-1][0]:
            return {"isolation": self.isolation}
        return self.row


class UnusedConnection:
    def cursor(self) -> None:
        raise AssertionError("participant must use the supplied cursor")


def _unit(cursor: SuppliedCursor) -> PgsqlUnitOfWork:
    return PgsqlUnitOfWork(
        connection=cast(PgsqlConnection, UnusedConnection()),
        cursor=cast(PgsqlCursor, cursor),
    )


class TaskSubmissionPgsqlTest(IsolatedAsyncioTestCase):
    def test_evidence_round_trip_is_closed_and_versioned(self) -> None:
        evidence = _evidence()
        self.assertEqual(
            submission_evidence_from_envelope(evidence.envelope()), evidence
        )
        for change in (
            {"format": "other"},
            {"version": True},
            {"version": 2},
            {"extra": None},
            {"payload": None},
        ):
            with self.subTest(change=change):
                with self.assertRaises(AssertionError):
                    submission_evidence_from_envelope(
                        {
                            **evidence.envelope(),
                            **change,
                        }
                    )
        payload = cast(dict[str, object], evidence.envelope()["payload"])
        for change in (
            {"owner_scope": 1},
            {"created": 1},
            {"run_id": ""},
            {"fingerprint": "x" * 64},
            {"extra": None},
        ):
            with self.subTest(change=change):
                with self.assertRaises(AssertionError):
                    submission_evidence_from_envelope(
                        {
                            **evidence.envelope(),
                            "payload": {**payload, **change},
                        }
                    )
        with self.assertRaises(AssertionError):
            submission_evidence_from_envelope(None)

    async def test_fresh_absence_waits_for_original_transaction_to_finish(
        self,
    ) -> None:
        cursor = SuppliedCursor()
        cursor.writer_finished.clear()
        read = create_task(
            read_task_submission(
                _unit(cursor), _prepared(), fingerprint="a" * 64
            )
        )
        await cursor.barrier_entered.wait()
        await sleep(0)
        self.assertFalse(read.done())
        self.assertEqual(len(cursor.statements), 2)
        # The original writer commits while the fresh reader waits.
        cursor.row = {
            "run_id": "existing-run",
            "payload": _evidence().envelope(),
        }
        cursor.writer_finished.set()
        self.assertEqual(await read, _evidence())
        self.assertEqual(len(cursor.statements), 3)
        self.assertIn('FROM "task_submissions"', cursor.statements[2][0])

    async def test_absence_after_barrier_and_scoped_lock_keys(self) -> None:
        cursor = SuppliedCursor()
        unit = _unit(cursor)
        self.assertIsNone(
            await read_task_submission(unit, _prepared(), fingerprint="a" * 64)
        )
        first = cursor.statements[1][1]
        await lock_task_submission(unit, _prepared())
        self.assertEqual(cursor.statements[-1][1], first)
        await lock_task_submission(
            unit, replace(_prepared(), owner_scope="other")
        )
        self.assertNotEqual(cursor.statements[-1][1], first)
        await lock_task_submission(
            unit, replace(_prepared(), submission_id="other")
        )
        self.assertNotEqual(cursor.statements[-1][1], first)

    async def test_read_rejects_mismatched_persisted_associations(
        self,
    ) -> None:
        for evidence in (
            replace(_evidence(), owner_scope="other"),
            replace(_evidence(), submission_id="other"),
            replace(_evidence(), run_id="other"),
            replace(_evidence(), definition_id="other"),
            replace(_evidence(), fingerprint="b" * 64),
        ):
            with self.subTest(evidence=evidence):
                cursor = SuppliedCursor()
                cursor.row = {
                    "run_id": "existing-run",
                    "payload": evidence.envelope(),
                }
                with self.assertRaises(AssertionError):
                    await read_task_submission(
                        _unit(cursor), _prepared(), fingerprint="a" * 64
                    )

    async def test_old_snapshot_isolation_cannot_prove_absence(self) -> None:
        for isolation in ("repeatable read", "serializable"):
            with self.subTest(isolation=isolation):
                cursor = SuppliedCursor()
                cursor.isolation = isolation
                with self.assertRaisesRegex(AssertionError, "read committed"):
                    await read_task_submission(
                        _unit(cursor), _prepared(), fingerprint="a" * 64
                    )
                self.assertEqual(len(cursor.statements), 1)

    async def test_insert_uses_supplied_unit_and_parameterized_envelope(
        self,
    ) -> None:
        cursor = SuppliedCursor()
        await insert_task_submission(_unit(cursor), _evidence())
        self.assertEqual(len(cursor.statements), 1)
        query, parameters = cursor.statements[0]
        self.assertIn('INSERT INTO "task_submissions"', query)
        assert isinstance(parameters, tuple)
        self.assertEqual(
            parameters[:3], ("owner", "submission", "existing-run")
        )
        assert isinstance(parameters[3], str)
        self.assertEqual(loads(parameters[3]), _evidence().envelope())
