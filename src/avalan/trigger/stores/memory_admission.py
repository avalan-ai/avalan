"""Commit trigger and task memory snapshots under their actual shared locks."""

from ...task.artifact_ownership import ArtifactObject, ArtifactStagingOwner
from ...task.artifacts.ownership_memory import (
    MemoryArtifactOwnership,
    MemoryArtifactUnit,
)
from ...task.queues.memory_submission import (
    MemoryTaskSubmissionParticipant,
    MemoryTaskSubmissionUnit,
    _commit,
    _snapshot,
)
from ...task.submission import TaskSubmissionOutcome
from ..admission import (
    PreparedTriggerAdmission,
    ResolvedTriggerDecision,
    TriggerAdmissionCancelledError,
    TriggerAdmissionResult,
    TriggerAdmissionWrite,
    TriggerCommitOutcome,
    evaluate_admission,
    recover_decisions,
)
from ..coverage import TriggerDecision
from ..error import TriggerError, TriggerErrorCode
from ..plan import PlannedOccurrence, TriggerAdmissionPlan
from ..records import TriggerOccurrence
from .memory import (
    InMemoryTriggerStore,
    copy_trigger_state,
    publish_trigger_state,
)

from asyncio import CancelledError
from dataclasses import replace
from datetime import timedelta


class MemoryTriggerAdmissionStore:
    """Publish all three stores together, without a pretend SQL transaction."""

    def __init__(
        self,
        store: InMemoryTriggerStore,
        participant: MemoryTaskSubmissionParticipant,
        ownership: MemoryArtifactOwnership,
    ) -> None:
        assert isinstance(store, InMemoryTriggerStore)
        assert isinstance(participant, MemoryTaskSubmissionParticipant)
        assert isinstance(ownership, MemoryArtifactOwnership)
        assert ownership is participant.store._artifact_ownership
        assert (
            store._artifact_ownership is None
            or store._artifact_ownership is ownership
        )
        store._artifact_ownership = ownership
        self.store = store
        self.participant = participant
        self.ownership = ownership

    async def preflight(self) -> None:
        await self.participant.preflight_submission()

    def _validate(self, plan: TriggerAdmissionPlan) -> None:
        if plan.snapshot.definition.owner_scope_id != self.store.owner:
            raise TriggerError(
                TriggerErrorCode.STORE_INCOMPATIBLE, "owner_scope_id"
            )

    def _history(
        self, plan: TriggerAdmissionPlan
    ) -> tuple[TriggerDecision, ...]:
        if not plan.decisions:
            return ()
        first = min(
            (
                item.scheduled_at
                if isinstance(item, PlannedOccurrence)
                else item.first_at
            )
            for item in plan.decisions
        )
        last = max(
            (
                item.scheduled_at
                if isinstance(item, PlannedOccurrence)
                else item.until_at - timedelta(microseconds=1)
            )
            for item in plan.decisions
        )
        name = plan.snapshot.definition.name
        values: tuple[TriggerDecision, ...] = (
            *self.store._occurrences.get(name, ()),
            *self.store._spans.get(name, ()),
        )
        return tuple(
            value
            for value in values
            if value.revision == plan.snapshot.definition.revision
            and (
                first <= value.scheduled_at <= last
                if isinstance(value, TriggerOccurrence)
                else value.until_at > first and value.first_at <= last
            )
        )

    async def recover(
        self, plan: TriggerAdmissionPlan
    ) -> TriggerAdmissionResult:
        self._validate(plan)
        async with self.store._lock:
            resolved = recover_decisions(plan, self._history(plan))
            return TriggerAdmissionResult(
                plan=plan,
                outcome=(
                    TriggerCommitOutcome.COMMITTED
                    if len(resolved) == len(plan.request_ids) and resolved
                    else TriggerCommitOutcome.NOT_COMMITTED
                ),
                resolved=resolved,
                snapshot=self.store._current.get(
                    plan.snapshot.definition.name
                ),
            )

    async def admit(
        self, prepared: PreparedTriggerAdmission
    ) -> TriggerAdmissionResult:
        self._validate(prepared.plan)
        try:
            return await self._admit(prepared)
        except CancelledError as error:
            try:
                result = await self.recover(prepared.plan)
            except (Exception, CancelledError):
                result = TriggerAdmissionResult(
                    plan=prepared.plan, outcome=TriggerCommitOutcome.UNKNOWN
                )
            raise TriggerAdmissionCancelledError(
                replace(result, prepared=prepared)
            ) from error

    async def _admit(
        self, prepared: PreparedTriggerAdmission
    ) -> TriggerAdmissionResult:
        plan = prepared.plan
        store = self.store
        participant = self.participant
        ownership = self.ownership
        async with store._lock:
            async with participant.store._lock:
                async with ownership._lock:
                    history = self._history(plan)
                    resolved = recover_decisions(plan, history)
                    if len(resolved) == len(plan.request_ids) and resolved:
                        return TriggerAdmissionResult(
                            plan=plan,
                            outcome=TriggerCommitOutcome.COMMITTED,
                            resolved=resolved,
                            prepared=prepared,
                        )
                    objects = MemoryArtifactUnit(
                        store=ownership,
                        objects=dict(ownership._objects),
                        staging=dict(ownership._staging),
                        revisions=set(ownership._revisions),
                        runs=dict(ownership._runs),
                    )
                    unit = MemoryTaskSubmissionUnit(
                        participant=participant,
                        ownership=objects,
                        tasks=_snapshot(participant.store),
                        items=dict(participant._items),
                        ledger=dict(participant._ledger),
                    )
                    try:
                        states = tuple(
                            unit.tasks._runs[value.run_id].state
                            for values in store._occurrences.values()
                            for value in values
                            if value.trigger_id
                            == plan.snapshot.state.trigger_id
                            and value.run_id is not None
                        )
                        decision = evaluate_admission(
                            store.owner,
                            prepared,
                            store._current.get(plan.snapshot.definition.name),
                            decision_time=store._clock(),
                            existing=history,
                            outstanding_states=states,
                        )
                        if isinstance(decision, TriggerAdmissionResult):
                            return replace(decision, prepared=prepared)
                        await self._persist(decision, unit, objects)
                        trigger = copy_trigger_state(store)
                        for value in decision.decisions:
                            name = plan.snapshot.definition.name
                            if isinstance(value, TriggerOccurrence):
                                trigger._occurrences[name] = (
                                    *trigger._occurrences.get(name, ()),
                                    value,
                                )
                            else:
                                trigger._spans[name] = (
                                    *trigger._spans.get(name, ()),
                                    value,
                                )
                        trigger._write(decision.mutation)
                        result = TriggerAdmissionResult(
                            plan=plan,
                            outcome=TriggerCommitOutcome.COMMITTED,
                            resolved=tuple(
                                ResolvedTriggerDecision(
                                    request_id=identity, decisions=(value,)
                                )
                                for identity, value in zip(
                                    plan.request_ids,
                                    decision.decisions,
                                    strict=True,
                                )
                            ),
                            snapshot=decision.mutation.snapshot,
                            prepared=prepared,
                        )
                        # No await or user callback occurs while publishing.
                        # All readers remain behind these same three locks.
                        _commit(unit.tasks, participant.store)
                        participant._items = unit.items
                        participant._ledger = unit.ledger
                        ownership._objects = objects.objects
                        ownership._staging = objects.staging
                        ownership._revisions = objects.revisions
                        ownership._runs = objects.runs
                        publish_trigger_state(trigger, store)
                        return result
                    finally:
                        unit.active = False
                        objects.active = False

    async def _persist(
        self,
        decision: TriggerAdmissionWrite,
        unit: MemoryTaskSubmissionUnit,
        objects: MemoryArtifactUnit,
    ) -> None:
        for submission in decision.submissions:
            if submission.run_id in unit.tasks._runs:
                raise TriggerError(TriggerErrorCode.CONFLICT, "admission.run")
            write = await self.participant.submit_prepared(
                submission, unit_of_work=unit
            )
            if not write.created or write.run.run_id != submission.run_id:
                raise TriggerError(TriggerErrorCode.CONFLICT, "admission.run")
            for artifact in submission.artifacts:
                value = ArtifactObject.from_ref(artifact.ref)
                staging = ArtifactStagingOwner(
                    staging_id=submission.submission_id,
                    owner_scope=submission.owner_scope,
                    recovery_id=submission.submission_id,
                )
                objects.attach_run(
                    value.object_id, staging, artifact.ref.artifact_id
                )

            if submission.artifacts:
                objects.release_staging(
                    staging, outcome=TaskSubmissionOutcome.COMMITTED
                )
