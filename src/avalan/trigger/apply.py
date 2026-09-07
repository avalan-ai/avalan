"""Register validated input and revision owners as one atomic mutation."""

from ..pgsql import PgsqlUnitOfWork
from ..task.artifact_ownership import ArtifactRevisionOwner
from ..task.artifacts.ownership_memory import (
    MemoryArtifactOwnership,
    MemoryArtifactUnit,
)
from ..task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from ..task.deployment import ExecutionDeploymentError
from ..task.submission import TaskSubmissionOutcome
from .admission import TriggerCommitOutcome
from .codec import encode_record
from .error import TriggerError, TriggerErrorCode
from .preparation import PreparedTriggerRegistration, TriggerPreparationService
from .records import TriggerSnapshot
from .registration import apply_registration
from .stores.memory import (
    InMemoryTriggerStore,
    copy_trigger_state,
    publish_trigger_state,
)
from .stores.pgsql import (
    PgsqlTriggerStore,
    _snapshot,
    decision_time,
    lock_identity,
)

from asyncio import CancelledError
from dataclasses import dataclass, field, replace
from hashlib import sha256
from json import dumps


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerApplyResult:
    outcome: TriggerCommitOutcome
    prepared: PreparedTriggerRegistration = field(repr=False)
    snapshot: TriggerSnapshot | None = None
    error_code: TriggerErrorCode | None = None
    error_path: str | None = None
    cleanup_pending: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.outcome, TriggerCommitOutcome)
        assert (self.outcome == TriggerCommitOutcome.COMMITTED) == (
            self.snapshot is not None
        )


class TriggerApplyCancelledError(CancelledError):
    def __init__(self, result: TriggerApplyResult) -> None:
        self.result = result
        super().__init__("trigger registration interrupted")


def _fingerprint(
    prepared: PreparedTriggerRegistration, expected_generation: int | None
) -> str:
    request = prepared.registration
    return sha256(
        dumps(
            [
                request.name,
                request.semantic_hash,
                request.desired_enabled,
                expected_generation,
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class TriggerRegistrationService:
    """Keep validated registration and physical ownership on the same store."""

    def __init__(self, preparation: TriggerPreparationService) -> None:
        self.preparation = preparation
        self.store = preparation.admission.store
        self.ownership = preparation.ownership

    async def _recover_pgsql(
        self,
        unit: PgsqlUnitOfWork,
        prepared: PreparedTriggerRegistration,
        fingerprint: str,
    ) -> TriggerSnapshot | None:
        await unit.cursor.execute(
            """
SELECT fingerprint, definition, state FROM trigger_registration_requests
WHERE owner_scope_id = %s AND request_id = %s::uuid
""",
            (prepared.staging.owner_scope, prepared.staging.recovery_id),
        )
        row = await unit.cursor.fetchone()
        if row is None:
            return None
        if row["fingerprint"] != fingerprint:
            raise TriggerError(
                TriggerErrorCode.CONFLICT, "registration.identity"
            )
        return _snapshot(row)

    async def recover(
        self,
        prepared: PreparedTriggerRegistration,
        *,
        expected_generation: int | None,
    ) -> TriggerApplyResult:
        fingerprint = _fingerprint(prepared, expected_generation)
        try:
            if isinstance(self.store, PgsqlTriggerStore):
                async with self.store.transaction() as unit:
                    await lock_identity(
                        unit,
                        self.store.owner,
                        "registration:" + prepared.staging.recovery_id,
                    )
                    snapshot = await self._recover_pgsql(
                        unit, prepared, fingerprint
                    )
            else:
                async with self.store._lock:
                    prior = self.store._registrations.get(
                        prepared.staging.recovery_id
                    )
                    if prior is not None and prior[0] != fingerprint:
                        raise TriggerError(
                            TriggerErrorCode.CONFLICT, "registration.identity"
                        )
                    snapshot = prior[1] if prior else None
            return TriggerApplyResult(
                prepared=prepared,
                snapshot=snapshot,
                outcome=(
                    TriggerCommitOutcome.COMMITTED
                    if snapshot is not None
                    else TriggerCommitOutcome.NOT_COMMITTED
                ),
            )
        except Exception:
            return TriggerApplyResult(
                prepared=prepared, outcome=TriggerCommitOutcome.UNKNOWN
            )

    async def release_unused(
        self,
        prepared: PreparedTriggerRegistration,
        *,
        expected_generation: int | None,
    ) -> TriggerApplyResult:
        """Settle unused staging only after completion-fenced reconciliation.

        Revision and run references remain intact. Byte cleanup is a separate
        grace-checked operation; an unknown result never releases staging.
        """
        if (
            prepared.task_input._client._preparation_authority
            is not self.preparation.client._preparation_authority
        ):
            raise TriggerError(
                TriggerErrorCode.STORE_INCOMPATIBLE, "registration.owner"
            )
        result = await self.recover(
            prepared, expected_generation=expected_generation
        )
        if result.outcome == TriggerCommitOutcome.UNKNOWN:
            return result
        outcome = (
            TaskSubmissionOutcome.COMMITTED
            if result.outcome == TriggerCommitOutcome.COMMITTED
            else TaskSubmissionOutcome.NOT_COMMITTED
        )
        ownership = self.preparation.ownership
        try:
            if isinstance(ownership, PgsqlArtifactOwnership):
                async with ownership.transaction() as unit:
                    await ownership.release_staging(
                        prepared.staging, outcome=outcome, unit_of_work=unit
                    )
            else:
                async with ownership.transaction() as memory:
                    memory.release_staging(prepared.staging, outcome=outcome)
        except CancelledError as error:
            raise TriggerApplyCancelledError(
                replace(result, cleanup_pending=True)
            ) from error
        except Exception:
            return replace(result, cleanup_pending=True)
        return result

    async def apply(
        self,
        prepared: PreparedTriggerRegistration,
        *,
        expected_generation: int | None,
    ) -> TriggerApplyResult:
        if (
            prepared.task_input._client._preparation_authority
            is not self.preparation.client._preparation_authority
            or prepared.staging.owner_scope != self.store.owner.value
        ):
            raise TriggerError(
                TriggerErrorCode.STORE_INCOMPATIBLE, "registration.owner"
            )
        fingerprint = _fingerprint(prepared, expected_generation)
        try:
            await prepared.task_input._client._deployment_client(
                prepared.registration.execution_deployment_id,
                file_delivery=bool(prepared.task_input.materialized_files),
            )
            if isinstance(self.store, PgsqlTriggerStore):
                snapshot = await self._apply_pgsql(
                    self.store, prepared, fingerprint, expected_generation
                )
            else:
                snapshot = await self._apply_memory(
                    self.store, prepared, fingerprint, expected_generation
                )
            return TriggerApplyResult(
                prepared=prepared,
                snapshot=snapshot,
                outcome=TriggerCommitOutcome.COMMITTED,
            )
        except CancelledError as error:
            try:
                result = await self.recover(
                    prepared, expected_generation=expected_generation
                )
            except CancelledError:
                result = TriggerApplyResult(
                    prepared=prepared, outcome=TriggerCommitOutcome.UNKNOWN
                )
            raise TriggerApplyCancelledError(result) from error
        except ExecutionDeploymentError:
            recovered = await self.recover(
                prepared, expected_generation=expected_generation
            )
            if recovered.outcome != TriggerCommitOutcome.NOT_COMMITTED:
                return recovered
            return replace(
                recovered,
                error_code=TriggerErrorCode.DEPLOYMENT_MISMATCH,
                error_path="deployment",
            )
        except TriggerError as error:
            return TriggerApplyResult(
                prepared=prepared,
                outcome=TriggerCommitOutcome.NOT_COMMITTED,
                error_code=error.code,
                error_path=error.path,
            )
        except Exception:
            return await self.recover(
                prepared, expected_generation=expected_generation
            )

    async def _apply_pgsql(
        self,
        store: PgsqlTriggerStore,
        prepared: PreparedTriggerRegistration,
        fingerprint: str,
        expected_generation: int | None,
    ) -> TriggerSnapshot:
        assert isinstance(self.ownership, PgsqlArtifactOwnership)
        async with store.transaction() as unit:
            await lock_identity(
                unit,
                store.owner,
                "registration:" + prepared.staging.recovery_id,
            )
            prior = await self._recover_pgsql(unit, prepared, fingerprint)
            if prior is not None:
                return prior
            await lock_identity(
                unit, store.owner, "name:" + prepared.registration.name
            )
            current = await store._read(
                unit, prepared.registration.name, lock=True
            )
            mutation = apply_registration(
                store.owner,
                prepared.registration,
                current,
                expected_generation=expected_generation,
                decision_time=await decision_time(unit),
                trigger_id=(
                    current.state.trigger_id
                    if current
                    else prepared.staging.recovery_id
                ),
                limits=store._limits,
            )
            await store._write(unit, mutation)
            snapshot = mutation.snapshot
            if mutation.definition_created:
                owner = ArtifactRevisionOwner(
                    owner_scope=store.owner.value,
                    resource_id=snapshot.state.trigger_id,
                    revision=snapshot.state.revision,
                )
                for value in sorted(
                    prepared.objects, key=lambda item: item.object_id
                ):
                    await self.ownership.attach_revision(
                        value, prepared.staging, owner, unit_of_work=unit
                    )
            await self.ownership.release_staging(
                prepared.staging,
                outcome=TaskSubmissionOutcome.COMMITTED,
                unit_of_work=unit,
            )
            await unit.cursor.execute(
                """
INSERT INTO trigger_registration_requests
    (owner_scope_id, request_id, fingerprint, trigger_id, revision,
     definition, state)
VALUES (%s, %s::uuid, %s, %s, %s, %s::jsonb, %s::jsonb)
""",
                (
                    store.owner.value,
                    prepared.staging.recovery_id,
                    fingerprint,
                    snapshot.state.trigger_id,
                    snapshot.state.revision,
                    encode_record(snapshot.definition),
                    encode_record(snapshot.state),
                ),
            )
            return snapshot

    async def _apply_memory(
        self,
        store: InMemoryTriggerStore,
        prepared: PreparedTriggerRegistration,
        fingerprint: str,
        expected_generation: int | None,
    ) -> TriggerSnapshot:
        assert isinstance(self.ownership, MemoryArtifactOwnership)
        ownership = self.ownership
        async with store._lock:
            async with ownership._lock:
                prior = store._registrations.get(prepared.staging.recovery_id)
                if prior is not None:
                    if prior[0] != fingerprint:
                        raise TriggerError(
                            TriggerErrorCode.CONFLICT, "registration.identity"
                        )
                    return prior[1]
                current = store._current.get(prepared.registration.name)
                mutation = apply_registration(
                    store.owner,
                    prepared.registration,
                    current,
                    expected_generation=expected_generation,
                    decision_time=store._clock(),
                    trigger_id=(
                        current.state.trigger_id
                        if current
                        else prepared.staging.recovery_id
                    ),
                    limits=store._limits,
                )
                snapshot = mutation.snapshot
                unit = MemoryArtifactUnit(
                    store=ownership,
                    objects=dict(ownership._objects),
                    staging=dict(ownership._staging),
                    revisions=set(ownership._revisions),
                    runs=dict(ownership._runs),
                )
                try:
                    if mutation.definition_created:
                        owner = ArtifactRevisionOwner(
                            owner_scope=store.owner.value,
                            resource_id=snapshot.state.trigger_id,
                            revision=snapshot.state.revision,
                        )
                        for value in prepared.objects:
                            unit.attach_revision(
                                value.object_id, prepared.staging, owner
                            )
                    unit.release_staging(
                        prepared.staging,
                        outcome=TaskSubmissionOutcome.COMMITTED,
                    )
                    trigger = copy_trigger_state(store)
                    trigger._write(mutation)
                    trigger._registrations[prepared.staging.recovery_id] = (
                        fingerprint,
                        snapshot,
                    )
                    publish_trigger_state(trigger, store)
                    ownership._objects = unit.objects
                    ownership._staging = unit.staging
                    ownership._revisions = unit.revisions
                    ownership._runs = unit.runs
                    return snapshot
                finally:
                    unit.active = False
