"""Compose task and trigger admission in one fenced PostgreSQL transaction."""

from ...pgsql import PgsqlUnitOfWork
from ...task.artifact_ownership import ArtifactObject, ArtifactStagingOwner
from ...task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from ...task.queues.pgsql import PgsqlTaskQueue
from ...task.queues.submission import lock_task_submission
from ...task.state import TASK_RUN_TERMINAL_STATES, TaskRunState
from ...task.submission import TaskSubmissionOutcome
from ..admission import (
    PreparedTriggerAdmission,
    TriggerAdmissionCancelledError,
    TriggerAdmissionResult,
    TriggerAdmissionWrite,
    TriggerCommitOutcome,
    evaluate_admission,
    recover_decisions,
)
from ..codec import encode_record, record_from_payload
from ..coverage import TriggerDecision
from ..error import TriggerError, TriggerErrorCode
from ..plan import PlannedOccurrence, TriggerAdmissionPlan
from ..records import TriggerCoverageSpan, TriggerOccurrence
from ..search_budget import SearchWorkExhausted
from .pgsql import PgsqlTriggerStore, _snapshot, decision_time, lock_identity

from asyncio import CancelledError
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from json import dumps


class PgsqlTriggerAdmissionStore:
    """Require the exact task database participant before any mutation."""

    def __init__(
        self, store: PgsqlTriggerStore, queue: PgsqlTaskQueue
    ) -> None:
        if (
            not isinstance(store, PgsqlTriggerStore)
            or not isinstance(queue, PgsqlTaskQueue)
            or queue._database is not store.database
        ):
            raise TriggerError(
                TriggerErrorCode.STORE_INCOMPATIBLE, "store.database"
            )
        self.store = store
        self.queue = queue
        self.ownership = PgsqlArtifactOwnership(store.database)

    def _validate_plan(self, plan: TriggerAdmissionPlan) -> None:
        if plan.snapshot.state.owner_scope_id != self.store.owner:
            raise TriggerError(
                TriggerErrorCode.STORE_INCOMPATIBLE, "owner_scope_id"
            )

    async def preflight(self) -> None:
        """Validate both participants before external task preparation."""
        async with self.store.transaction() as unit:
            await self.queue._assert_submission_unit(unit)

    async def _try_fence(
        self, unit: PgsqlUnitOfWork, plan: TriggerAdmissionPlan
    ) -> bool:
        digest = sha256(
            dumps(
                [
                    "avalan.trigger",
                    self.store.owner.value,
                    "trigger:" + plan.snapshot.state.trigger_id,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest()
        await unit.cursor.execute(
            "SELECT pg_try_advisory_xact_lock(%s) AS acquired",
            (int.from_bytes(digest[:8], "big", signed=True),),
        )
        row = await unit.cursor.fetchone()
        if row is None or type(row["acquired"]) is not bool:
            raise TriggerError(
                TriggerErrorCode.STORE_INCOMPATIBLE, "store.lock"
            )
        return row["acquired"]

    async def _history(
        self, unit: PgsqlUnitOfWork, plan: TriggerAdmissionPlan
    ) -> tuple[TriggerDecision, ...]:
        if not plan.decisions:
            return ()
        starts = tuple(
            (
                item.scheduled_at
                if isinstance(item, PlannedOccurrence)
                else item.first_at
            )
            for item in plan.decisions
        )
        ends = tuple(
            (
                item.scheduled_at
                if isinstance(item, PlannedOccurrence)
                else item.until_at - timedelta(microseconds=1)
            )
            for item in plan.decisions
        )
        state = plan.snapshot.state
        parameters = (
            self.store.owner.value,
            state.trigger_id,
            state.revision,
            min(starts),
            max(ends),
        )
        await unit.cursor.execute(
            """
SELECT payload FROM trigger_occurrences
WHERE owner_scope_id = %s AND trigger_id = %s AND revision = %s
    AND scheduled_at >= %s AND scheduled_at <= %s
ORDER BY scheduled_at LIMIT 1001
""",
            parameters,
        )
        occurrences = await unit.cursor.fetchall()
        await unit.cursor.execute(
            """
SELECT payload FROM trigger_coverage_spans
WHERE owner_scope_id = %s AND trigger_id = %s AND revision = %s
    AND until_at > %s AND first_at <= %s
ORDER BY first_at LIMIT 1001
""",
            parameters,
        )
        spans = await unit.cursor.fetchall()
        if len(occurrences) + len(spans) > 1000:
            raise TriggerError(
                TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED, "recovery.history"
            )
        records: list[TriggerDecision] = []
        for rows, expected in (
            (occurrences, TriggerOccurrence),
            (spans, TriggerCoverageSpan),
        ):
            for row in rows:
                record = record_from_payload(row["payload"])
                if not isinstance(record, expected):
                    raise TriggerError(
                        TriggerErrorCode.UNSUPPORTED_VERSION, "store.history"
                    )
                assert isinstance(
                    record, TriggerOccurrence | TriggerCoverageSpan
                )
                records.append(record)
        return tuple(records)

    async def _outstanding(
        self, unit: PgsqlUnitOfWork, plan: TriggerAdmissionPlan
    ) -> tuple[TaskRunState, ...]:
        await unit.cursor.execute(
            """
SELECT r.state FROM trigger_occurrences o
JOIN task_runs r ON r.run_id = o.run_id
WHERE o.owner_scope_id = %s AND o.trigger_id = %s
    AND r.state <> ALL(%s) LIMIT 1
""",
            (
                self.store.owner.value,
                plan.snapshot.state.trigger_id,
                sorted(state.value for state in TASK_RUN_TERMINAL_STATES),
            ),
        )
        row = await unit.cursor.fetchone()
        if row is None:
            return ()
        value = row["state"]
        if not isinstance(value, str):
            raise TriggerError(
                TriggerErrorCode.UNSUPPORTED_VERSION, "store.run_state"
            )
        try:
            return (TaskRunState(value),)
        except ValueError:
            raise TriggerError(
                TriggerErrorCode.UNSUPPORTED_VERSION, "store.run_state"
            ) from None

    async def _persist(
        self, unit: PgsqlUnitOfWork, write: TriggerAdmissionWrite
    ) -> None:
        submissions = {item.occurrence_id: item for item in write.submissions}
        for prepared in write.submissions:
            await lock_task_submission(unit, prepared)
            await unit.cursor.execute(
                "SELECT run_id FROM task_runs WHERE run_id = %s",
                (prepared.run_id,),
            )
            if await unit.cursor.fetchone() is not None:
                raise TriggerError(
                    TriggerErrorCode.CONFLICT, "admission.new_run"
                )
            result = await self.queue.submit_prepared(
                prepared, unit_of_work=unit
            )
            if not result.created or result.run.run_id != prepared.run_id:
                raise TriggerError(
                    TriggerErrorCode.CONFLICT, "admission.new_run"
                )
            staging = ArtifactStagingOwner(
                staging_id=prepared.submission_id,
                owner_scope=prepared.owner_scope,
                recovery_id=prepared.submission_id,
            )
            for artifact in sorted(
                prepared.artifacts,
                key=lambda item: (item.ref.store, item.ref.storage_key),
            ):
                physical = ArtifactObject.from_ref(artifact.ref)
                await self.ownership.attach_run(
                    physical.object_id,
                    staging,
                    artifact.ref.artifact_id,
                    unit_of_work=unit,
                )
            if prepared.artifacts:
                await self.ownership.release_staging(
                    staging,
                    outcome=TaskSubmissionOutcome.COMMITTED,
                    unit_of_work=unit,
                )
        for decision in write.decisions:
            if isinstance(decision, TriggerCoverageSpan):
                await self.store._write_span(unit, decision)
                continue
            submission = submissions.get(decision.occurrence_id)
            await unit.cursor.execute(
                """
INSERT INTO trigger_occurrences
    (owner_scope_id, trigger_id, revision, occurrence_id, scheduled_at,
     decided_at, disposition, run_id, submission_id,
     task_definition_id, payload)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
""",
                (
                    self.store.owner.value,
                    decision.trigger_id,
                    decision.revision,
                    decision.occurrence_id,
                    decision.scheduled_at,
                    decision.decided_at,
                    decision.disposition.value,
                    decision.run_id,
                    (
                        submission.submission_id
                        if submission is not None
                        else None
                    ),
                    write.mutation.snapshot.definition.task_definition_id,
                    encode_record(decision),
                ),
            )
        await self.store._write(unit, write.mutation)

    async def recover(
        self, plan: TriggerAdmissionPlan
    ) -> TriggerAdmissionResult:
        """Fence the original writer before checking for absence."""
        self._validate_plan(plan)
        try:
            async with self.store.transaction() as unit:
                await lock_identity(
                    unit,
                    self.store.owner,
                    "trigger:" + plan.snapshot.state.trigger_id,
                )
                resolved = recover_decisions(
                    plan, await self._history(unit, plan)
                )
            return TriggerAdmissionResult(
                plan=plan,
                outcome=(
                    TriggerCommitOutcome.COMMITTED
                    if plan.decisions and len(resolved) == len(plan.decisions)
                    else TriggerCommitOutcome.NOT_COMMITTED
                ),
                resolved=resolved,
            )
        except SearchWorkExhausted:
            raise
        except Exception:
            return TriggerAdmissionResult(
                plan=plan,
                outcome=TriggerCommitOutcome.UNKNOWN,
                error_code=TriggerErrorCode.COMMIT_UNKNOWN,
            )

    async def admit(
        self, prepared: PreparedTriggerAdmission
    ) -> TriggerAdmissionResult:
        """Expose commit evidence only after transaction exit or recovery."""
        plan = prepared.plan
        self._validate_plan(plan)
        if any(
            item._participant is not self.queue
            for item in prepared.submissions
        ):
            raise TriggerError(
                TriggerErrorCode.STORE_INCOMPATIBLE, "prepared.participant"
            )
        result = TriggerAdmissionResult(
            plan=plan, outcome=TriggerCommitOutcome.UNKNOWN, prepared=prepared
        )
        write = None
        history: tuple[TriggerDecision, ...] = ()
        try:
            async with self.store.transaction() as unit:
                await self.queue._assert_submission_unit(unit)
                if not await self._try_fence(unit, plan):
                    result = replace(
                        result,
                        outcome=TriggerCommitOutcome.NOT_COMMITTED,
                        contended=True,
                    )
                else:
                    history = await self._history(unit, plan)
                    resolved = recover_decisions(plan, history)
                    if plan.decisions and len(resolved) == len(plan.decisions):
                        result = replace(
                            result,
                            outcome=TriggerCommitOutcome.COMMITTED,
                            resolved=resolved,
                        )
                    else:
                        await unit.cursor.execute(
                            """
SELECT d.payload AS definition, t.payload AS state
FROM triggers t JOIN trigger_definitions d
    ON (d.owner_scope_id, d.trigger_id, d.revision)
     = (t.owner_scope_id, t.trigger_id, t.revision)
WHERE t.owner_scope_id = %s AND t.trigger_id = %s
FOR UPDATE OF t SKIP LOCKED
""",
                            (
                                self.store.owner.value,
                                plan.snapshot.state.trigger_id,
                            ),
                        )
                        row = await unit.cursor.fetchone()
                        if row is None:
                            result = replace(
                                result,
                                outcome=TriggerCommitOutcome.NOT_COMMITTED,
                                resolved=resolved,
                                contended=True,
                            )
                        else:
                            current = _snapshot(row)
                            now = await decision_time(unit)
                            evaluated = evaluate_admission(
                                self.store.owner,
                                prepared,
                                current,
                                decision_time=now,
                                existing=history,
                                outstanding_states=await self._outstanding(
                                    unit, plan
                                ),
                            )
                            if isinstance(evaluated, TriggerAdmissionResult):
                                result = evaluated
                            else:
                                write = evaluated
                                await self._persist(unit, write)
            if write is not None:
                result = TriggerAdmissionResult(
                    plan=plan,
                    outcome=TriggerCommitOutcome.COMMITTED,
                    resolved=recover_decisions(
                        plan, history + write.decisions
                    ),
                    snapshot=write.mutation.snapshot,
                    prepared=prepared,
                )
            return result
        except BaseException as error:
            # A failed acknowledgment is not evidence of rollback. Keep an
            # already-known recovered commit if reconciliation is interrupted.
            try:
                recovered = await self.recover(plan)
                if result.outcome is not TriggerCommitOutcome.COMMITTED:
                    result = replace(recovered, prepared=prepared)
            except BaseException as interrupted:
                if isinstance(interrupted, CancelledError):
                    if isinstance(error, Exception):
                        raise TriggerAdmissionCancelledError(result) from None
                elif not isinstance(interrupted, Exception):
                    raise
            if isinstance(error, CancelledError):
                raise TriggerAdmissionCancelledError(result) from None
            if (
                isinstance(error, SearchWorkExhausted)
                and result.outcome is TriggerCommitOutcome.NOT_COMMITTED
            ):
                raise error
            if not isinstance(error, Exception):
                raise
            return replace(
                result,
                error_code=(
                    (
                        error.code
                        if isinstance(error, TriggerError)
                        else TriggerErrorCode.ADMISSION_RETRYABLE
                    )
                    if result.outcome is TriggerCommitOutcome.NOT_COMMITTED
                    else result.error_code
                ),
            )
