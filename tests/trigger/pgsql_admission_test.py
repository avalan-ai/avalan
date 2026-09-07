"""Exercise admission protocol faults without claiming PostgreSQL proof."""

from .admission_test import prepared_plan
from .pgsql_store_test import Connection, Cursor, Database, snapshot_row
from .plan_test import snapshot
from .records_test import NOW, OWNER

from asyncio import CancelledError
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from json import loads
from unittest import IsolatedAsyncioTestCase

from pytest import raises

from avalan.pgsql import PgsqlParameters, PgsqlRow, PgsqlUnitOfWork
from avalan.task.queues.pgsql import PgsqlTaskQueue
from avalan.task.state import TaskRunState
from avalan.task.store import TaskRun
from avalan.task.submission import PreparedTaskSubmission, TaskSubmissionWrite
from avalan.trigger.admission import (
    PreparedTriggerAdmission,
    TriggerAdmissionCancelledError,
    TriggerCommitOutcome,
)
from avalan.trigger.codec import record_payload
from avalan.trigger.error import TriggerError
from avalan.trigger.plan import TriggerAdmissionPlan, plan_admission
from avalan.trigger.records import TriggerOccurrence
from avalan.trigger.stores.pgsql import PgsqlTriggerStore
from avalan.trigger.stores.pgsql_admission import PgsqlTriggerAdmissionStore


class AdmissionCursor(Cursor):
    def __init__(self) -> None:
        super().__init__()
        self.row = snapshot_row()
        self.row = {
            "definition": record_payload(snapshot().definition),
            "state": record_payload(snapshot().state),
        }
        self.occurrences: list[PgsqlRow] = []
        self.spans: list[PgsqlRow] = []
        self.acquired: object = True
        self.outstanding: PgsqlRow | None = None
        self.fail_write = False
        self.recovery_error: BaseException | None = None
        self.connections = 0

    async def execute(
        self, query: str, parameters: PgsqlParameters = None
    ) -> None:
        if self.connections > 1 and self.recovery_error is not None:
            raise self.recovery_error
        await super().execute(query, parameters)
        if (
            "INSERT INTO trigger_occurrences" in query
            or "INSERT INTO trigger_coverage_spans" in query
        ):
            if self.fail_write:
                raise RuntimeError("injected write failure")
            assert isinstance(parameters, tuple)
            encoded = parameters[-1]
            assert isinstance(encoded, str)
            row = {"payload": loads(encoded)}
            if "trigger_occurrences" in query:
                self.occurrences.append(row)
            else:
                self.spans.append(row)

    async def fetchone(self) -> PgsqlRow | None:
        if "SELECT run_id FROM task_runs WHERE" in self.query:
            return None
        if "pg_try_advisory" in self.query:
            return {"acquired": self.acquired}
        if "SELECT r.state" in self.query:
            return self.outstanding
        return await super().fetchone()

    async def fetchall(self) -> list[PgsqlRow]:
        if "SELECT payload FROM trigger_occurrences" in self.query:
            return list(self.occurrences)
        if "SELECT payload FROM trigger_coverage_spans" in self.query:
            return list(self.spans)
        return await super().fetchall()


class AdmissionConnection(Connection):
    def __init__(self, cursor: AdmissionCursor) -> None:
        super().__init__(cursor)
        self.cursor_handle = cursor
        self.ack_error: BaseException | None = None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["AdmissionConnection"]:
        self.transactions += 1
        previous = (
            list(self.cursor_handle.occurrences),
            list(self.cursor_handle.spans),
        )
        try:
            yield self
        except BaseException:
            self.cursor_handle.occurrences, self.cursor_handle.spans = previous
            raise
        if self.ack_error is not None:
            error, self.ack_error = self.ack_error, None
            raise error


class AdmissionDatabase(Database):
    def __init__(self) -> None:
        self.cursor: AdmissionCursor = AdmissionCursor()
        self.handle: AdmissionConnection = AdmissionConnection(self.cursor)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AdmissionConnection]:
        self.cursor.connections += 1
        yield self.handle


class SubmissionQueue(PgsqlTaskQueue):
    def __init__(self, database: AdmissionDatabase) -> None:
        super().__init__(database)
        self.units: list[PgsqlUnitOfWork] = []
        self.created = True

    async def _assert_submission_unit(self, unit: PgsqlUnitOfWork) -> None:
        assert unit.database is self._database

    async def submit_prepared(
        self,
        prepared: PreparedTaskSubmission,
        *,
        unit_of_work: PgsqlUnitOfWork,
    ) -> TaskSubmissionWrite:
        self.units.append(unit_of_work)
        return TaskSubmissionWrite(
            submission_id=prepared.submission_id,
            created=self.created,
            run=TaskRun(
                run_id=prepared.run_id,
                definition_id=prepared.execution.definition_id,
                state=TaskRunState.QUEUED,
                request=prepared.execution,
                created_at=NOW,
                updated_at=NOW,
            ),
        )


def prepared(
    queue: SubmissionQueue, plan: TriggerAdmissionPlan | None = None
) -> PreparedTriggerAdmission:
    candidate = prepared_plan(
        plan or plan_admission(snapshot(), NOW + timedelta(minutes=5))
    )
    return replace(
        candidate,
        submissions=tuple(
            replace(item, _participant=queue) for item in candidate.submissions
        ),
    )


class PgsqlAdmissionTestCase(IsolatedAsyncioTestCase):
    async def test_new_admission_uses_one_unit_and_duplicate_recovers(
        self,
    ) -> None:
        database = AdmissionDatabase()
        queue = SubmissionQueue(database)
        store = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(database, OWNER), queue
        )
        candidate = prepared(queue)
        result = await store.admit(candidate)
        assert result.outcome is TriggerCommitOutcome.COMMITTED
        assert result.unresolved_ids == ()
        assert len(queue.units) == 1
        assert queue.units[0].cursor is database.cursor
        assert database.handle.transactions == 1
        database.cursor.row = None
        again = await store.admit(candidate)
        assert again.outcome is TriggerCommitOutcome.COMMITTED
        assert len(queue.units) == 1
        recovered = await store.recover(candidate.plan)
        assert recovered.outcome is TriggerCommitOutcome.COMMITTED
        assert any(
            "pg_advisory_xact_lock" in query
            for query, _ in database.cursor.calls
        )

    async def test_preflight_and_contention_do_not_submit(self) -> None:
        database = AdmissionDatabase()
        queue = SubmissionQueue(database)
        store = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(database, OWNER), queue
        )
        await store.preflight()
        database.cursor.acquired = False
        result = await store.admit(prepared(queue))
        assert (
            result.contended
            and result.outcome is TriggerCommitOutcome.NOT_COMMITTED
        )
        database.cursor.acquired = True
        database.cursor.row = None
        result = await store.admit(prepared(queue))
        assert result.contended
        assert queue.units == []

    async def test_mismatched_capabilities_are_rejected_before_writes(
        self,
    ) -> None:
        database = AdmissionDatabase()
        queue = SubmissionQueue(database)
        with raises(TriggerError):
            PgsqlTriggerAdmissionStore(
                PgsqlTriggerStore(AdmissionDatabase(), OWNER), queue
            )
        store = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(database, OWNER), queue
        )
        with raises(TriggerError):
            await store.admit(prepared_plan(plan_admission(snapshot(), NOW)))
        other = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(database, replace(OWNER, value="other")), queue
        )
        with raises(TriggerError):
            await other.admit(prepared(queue))
        assert database.cursor.calls == []

    async def test_failure_rolls_back_and_fenced_absence_settles(self) -> None:
        database = AdmissionDatabase()
        queue = SubmissionQueue(database)
        store = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(database, OWNER), queue
        )
        database.cursor.fail_write = True
        result = await store.admit(prepared(queue))
        assert result.outcome is TriggerCommitOutcome.NOT_COMMITTED
        assert database.cursor.occurrences == database.cursor.spans == []
        assert database.cursor.connections == 2
        queue.created = False
        database.cursor.fail_write = False
        result = await store.admit(prepared(queue))
        assert result.outcome is TriggerCommitOutcome.NOT_COMMITTED

    async def test_lost_ack_recovers_commit_and_unknown_retains_preparation(
        self,
    ) -> None:
        database = AdmissionDatabase()
        queue = SubmissionQueue(database)
        store = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(database, OWNER), queue
        )
        candidate = prepared(queue)
        database.handle.ack_error = RuntimeError("lost acknowledgment")
        result = await store.admit(candidate)
        assert result.outcome is TriggerCommitOutcome.COMMITTED
        assert len(database.cursor.occurrences) == 1
        fresh = AdmissionDatabase()
        fresh_queue = SubmissionQueue(fresh)
        uncertain = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(fresh, OWNER), fresh_queue
        )
        fresh.handle.ack_error = RuntimeError("lost acknowledgment")
        fresh.cursor.recovery_error = RuntimeError("recovery unavailable")
        candidate = prepared(fresh_queue)
        result = await uncertain.admit(candidate)
        assert result.outcome is TriggerCommitOutcome.UNKNOWN
        assert result.prepared is candidate
        assert result.unresolved_ids == candidate.plan.request_ids

    async def test_cancellation_retains_latest_commit_evidence(self) -> None:
        database = AdmissionDatabase()
        queue = SubmissionQueue(database)
        store = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(database, OWNER), queue
        )
        candidate = prepared(queue)
        database.handle.ack_error = CancelledError()
        with raises(TriggerAdmissionCancelledError) as cancelled:
            await store.admit(candidate)
        assert cancelled.value.result.outcome is TriggerCommitOutcome.COMMITTED
        database.handle.ack_error = CancelledError()
        database.cursor.recovery_error = CancelledError()
        # Permit the duplicate read, interrupt its fresh reconciliation.
        database.cursor.connections = 0
        with raises(TriggerAdmissionCancelledError) as repeated:
            await store.admit(candidate)
        assert repeated.value.result.outcome is TriggerCommitOutcome.COMMITTED

    async def test_authoritative_state_replans_or_skips_overlap(self) -> None:
        database = AdmissionDatabase()
        queue = SubmissionQueue(database)
        store = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(database, OWNER), queue
        )
        stale = prepared(queue, plan_admission(snapshot(), NOW))
        assert (await store.admit(stale)).replan_required
        database.cursor.outstanding = {"state": "input_required"}
        result = await store.admit(prepared(queue))
        assert result.outcome is TriggerCommitOutcome.COMMITTED
        assert queue.units == []
        decision = result.resolved[-1].decisions[0]
        assert isinstance(decision, TriggerOccurrence)
        assert decision.run_id is None

    async def test_invalid_store_records_and_recovery_budget_fail_safely(
        self,
    ) -> None:
        database = AdmissionDatabase()
        queue = SubmissionQueue(database)
        store = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(database, OWNER), queue
        )
        candidate = prepared(queue)
        database.cursor.acquired = None
        assert (
            await store.admit(candidate)
        ).outcome is TriggerCommitOutcome.NOT_COMMITTED
        database.cursor.acquired = True
        for state in (None, "unrecognized"):
            database.cursor.outstanding = {"state": state}
            assert (
                await store.admit(candidate)
            ).outcome is TriggerCommitOutcome.NOT_COMMITTED
        database.cursor.occurrences = [
            {"payload": record_payload(snapshot().state)}
        ]
        assert (
            await store.recover(candidate.plan)
        ).outcome is TriggerCommitOutcome.UNKNOWN
        database.cursor.occurrences *= 1001
        assert (
            await store.recover(candidate.plan)
        ).outcome is TriggerCommitOutcome.UNKNOWN
        empty = plan_admission(snapshot(), NOW - timedelta(seconds=1))
        assert (
            await store.recover(empty)
        ).outcome is TriggerCommitOutcome.NOT_COMMITTED

    async def test_secondary_interruptions_preserve_termination_categories(
        self,
    ) -> None:
        for original, secondary, expected in (
            (
                RuntimeError("ack"),
                CancelledError(),
                TriggerAdmissionCancelledError,
            ),
            (RuntimeError("ack"), KeyboardInterrupt(), KeyboardInterrupt),
            (SystemExit("stop"), RuntimeError("unavailable"), SystemExit),
        ):
            database = AdmissionDatabase()
            queue = SubmissionQueue(database)
            store = PgsqlTriggerAdmissionStore(
                PgsqlTriggerStore(database, OWNER), queue
            )
            database.handle.ack_error = original
            database.cursor.recovery_error = secondary
            with raises(expected) as interrupted:
                await store.admit(prepared(queue))
            if isinstance(interrupted.value, SystemExit):
                assert interrupted.value.code == "stop"
