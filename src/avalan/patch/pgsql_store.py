"""Persist the durable patch semantic contract through PostgreSQL rows.

The store is intentionally dormant: callers must supply an already migrated,
authenticated test database.  It does not start patch workers or expose a
mutation route.
"""

from asyncio import CancelledError, sleep
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TypeVar

from avalan.patch.codec import decode_result, encode_result
from avalan.patch.coordinator import RetransmissionKey
from avalan.patch.domain import (
    AlgorithmDigest,
    ArtifactState,
    Audience,
    ByteSize,
    CommitStepState,
    DurationTicks,
    ExpiryTick,
    LifecyclePhase,
    MutationState,
    PatchArtifactId,
    PatchCommitOwnerId,
    PatchContextId,
    PatchDomainId,
    PatchEventId,
    PatchExecutionId,
    PatchLineageId,
    PatchObserverCorrelationId,
    PatchPendingOperationId,
    PatchPlanId,
    PatchRequestId,
    PatchResult,
    PatchRetentionKeyId,
    PatchRetentionRecordId,
    PatchStepId,
    PatchWorkspaceId,
    SequenceNumber,
)
from avalan.patch.durable_store import (
    DenyDurableApprovalVerifier,
    DenyDurableRetentionAuthorizer,
    DenyDurableRetentionEnvelopeValidator,
    DurableApproval,
    DurableApprovalVerifier,
    DurableArtifactJournalEntry,
    DurableArtifactState,
    DurableCommitClaim,
    DurableCommitClaimState,
    DurableCommitLease,
    DurableJournal,
    DurableJournalCursor,
    DurableOutboxRecord,
    DurablePatchStore,
    DurablePendingAccess,
    DurablePendingRecord,
    DurablePendingRequest,
    DurablePlanReference,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableRequestSnapshot,
    DurableReservation,
    DurableRetentionAccess,
    DurableRetentionAuthorizer,
    DurableRetentionCleanup,
    DurableRetentionEnvelopeValidator,
    DurableRetentionKind,
    DurableRetentionPolicy,
    DurableRetentionRecord,
    DurableStepBinding,
    DurableStepJournalEntry,
    DurableStoreError,
    DurableStoreErrorCode,
    DurableTerminalRecord,
    EncryptedRetentionValue,
    derive_artifact_state,
)
from avalan.patch.policy import (
    PatchPrincipalId,
    PatchTenantId,
    PolicyRouteId,
)
from avalan.pgsql import (
    PgsqlCursor,
    PgsqlDatabase,
    PgsqlFailureCategory,
    PgsqlRow,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    classify_pgsql_error,
)

_T = TypeVar("_T")
_PLAN_TAG = "patch-durable-plan-v1"
_PLAN_SEPARATOR = "\x1f"
_STEP_SEPARATOR = "\x1e"
_DOMAIN_ADVISORY_LOCK = 7_556_109_174_025_592_683


@dataclass(frozen=True, slots=True)
class PgsqlDurablePatchStoreSettings:
    """Configure one bounded async PostgreSQL durable patch store pool."""

    dsn: str
    schema: str | None = None
    pool_minimum: int = 1
    pool_maximum: int = 10

    def __post_init__(self) -> None:
        """Require one nonempty DSN and bounded pool capacity."""
        if (
            not self.dsn
            or len(self.dsn) > 8192
            or (
                self.schema is not None
                and (not self.schema or len(self.schema) > 63)
            )
            or type(self.pool_minimum) is not int
            or type(self.pool_maximum) is not int
            or not 1 <= self.pool_minimum <= self.pool_maximum <= 64
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

    def database(self) -> PsycopgAsyncDatabase:
        """Create one closed bounded database wrapper for this store."""
        return PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=self.dsn,
                schema=self.schema,
                pool_minimum=self.pool_minimum,
                pool_maximum=self.pool_maximum,
                application_name="avalan-patch-durable-store",
            )
        )


class PgsqlDurablePatchStore(DurablePatchStore):
    """Implement strict durable patch semantics through SQL transactions."""

    def __init__(
        self,
        database: PgsqlDatabase,
        *,
        owns_database: bool = False,
        approval_verifier: DurableApprovalVerifier = (
            DenyDurableApprovalVerifier()
        ),
        retention_authorizer: DurableRetentionAuthorizer = (
            DenyDurableRetentionAuthorizer()
        ),
        retention_validator: DurableRetentionEnvelopeValidator = (
            DenyDurableRetentionEnvelopeValidator()
        ),
    ) -> None:
        """Bind an already migrated database without opening a transaction."""
        if (
            not hasattr(database, "connection")
            or type(owns_database) is not bool
            or not callable(getattr(approval_verifier, "verify", None))
            or not callable(
                getattr(retention_authorizer, "audiences_for", None)
            )
            or not callable(getattr(retention_validator, "validate", None))
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        self._database = database
        self._owns_database = owns_database
        self._approval_verifier = approval_verifier
        self._retention_authorizer = retention_authorizer
        self._retention_validator = retention_validator
        self._opened = False
        self._closed = False

    @classmethod
    def from_settings(
        cls,
        settings: PgsqlDurablePatchStoreSettings,
        *,
        approval_verifier: DurableApprovalVerifier = (
            DenyDurableApprovalVerifier()
        ),
        retention_authorizer: DurableRetentionAuthorizer = (
            DenyDurableRetentionAuthorizer()
        ),
        retention_validator: DurableRetentionEnvelopeValidator = (
            DenyDurableRetentionEnvelopeValidator()
        ),
    ) -> "PgsqlDurablePatchStore":
        """Create a store owning the bounded PostgreSQL pool from settings."""
        if type(settings) is not PgsqlDurablePatchStoreSettings:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        return cls(
            settings.database(),
            owns_database=True,
            approval_verifier=approval_verifier,
            retention_authorizer=retention_authorizer,
            retention_validator=retention_validator,
        )

    @property
    def database(self) -> PgsqlDatabase:
        """Return the database for migration and test-harness ownership."""
        return self._database

    async def open(self) -> None:
        """Open an owned database pool only after PostgreSQL support exists."""
        if self._closed or find_spec("psycopg") is None:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        if self._opened:
            return
        opener = getattr(self._database, "open", None)
        if opener is not None:
            result = opener()
            if isinstance(result, Awaitable):
                await result
        self._opened = True

    async def aclose(self) -> None:
        """Close only the database pool owned by this durable store."""
        if self._closed:
            return
        self._closed = True
        if not self._owns_database:
            return
        closer = getattr(self._database, "aclose", None)
        result = closer() if closer is not None else None
        if isinstance(result, Awaitable):
            await result

    async def __aenter__(self) -> "PgsqlDurablePatchStore":
        """Open the durable store for one async context."""
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        """Close owned database resources after one async context."""
        await self.aclose()
        return None

    async def reserve(
        self,
        identity: DurableRequestIdentity,
        canonical_digest: AlgorithmDigest,
    ) -> DurableReservation:
        """Reserve one retransmission identity before SQL inspection."""
        _require_exact(identity, DurableRequestIdentity)
        _require_exact(canonical_digest, AlgorithmDigest)

        async def execute(cursor: PgsqlCursor) -> DurableReservation:
            request_id = PatchRequestId.new()
            await cursor.execute(
                _INSERT_RESERVATION_SQL,
                (
                    request_id.value,
                    identity.tenant_id.value,
                    identity.principal_id.value,
                    identity.execution_id.value,
                    identity.route_id.value,
                    identity.retransmission_key.value,
                    canonical_digest.value,
                ),
            )
            inserted = await cursor.fetchone()
            if inserted is not None:
                return DurableReservation(
                    request_id, identity, canonical_digest, False
                )
            row = await _select_identity_for_update(cursor, identity)
            if _row_str(row, "canonical_digest") != canonical_digest.value:
                raise DurableStoreError(
                    DurableStoreErrorCode.IDEMPOTENCY_CONFLICT
                )
            return DurableReservation(
                PatchRequestId(_row_str(row, "request_id")),
                identity,
                canonical_digest,
                True,
            )

        return await self._transaction("patch_durable_reserve", execute)

    async def persist_plan(
        self,
        reservation: DurableReservation,
        plan: DurablePlanReference,
    ) -> DurableRequestSnapshot:
        """Persist one immutable plan reference with a locked comparison."""
        _require_exact(reservation, DurableReservation)
        _require_exact(plan, DurablePlanReference)

        async def execute(cursor: PgsqlCursor) -> DurableRequestSnapshot:
            row = await _select_reservation_for_update(cursor, reservation)
            if plan.canonical_digest != reservation.canonical_digest:
                raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)
            stored = _plan_from_row(row)
            if stored is not None and stored != plan:
                raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)
            if stored is None:
                if _row_str(row, "lifecycle") != LifecyclePhase.RECEIVED.value:
                    raise DurableStoreError(
                        DurableStoreErrorCode.LIFECYCLE_CONFLICT
                    )
                await cursor.execute(
                    _PERSIST_PLAN_SQL,
                    (_encode_plan(plan), reservation.request_id.value),
                )
                row = await _select_request_for_update(
                    cursor, reservation.request_id
                )
            return await _snapshot(cursor, row)

        return await self._transaction("patch_durable_persist_plan", execute)

    async def claim_commit(
        self,
        reservation: DurableReservation,
        plan: DurablePlanReference,
        approval: DurableApproval,
        owner_id: PatchCommitOwnerId,
        now: ExpiryTick,
        lease_duration: DurationTicks,
        artifact_ids: tuple[PatchArtifactId, ...],
    ) -> DurableCommitClaim:
        """Consume approval with one owner/fence commit-start transaction."""
        _require_exact(reservation, DurableReservation)
        _require_exact(plan, DurablePlanReference)
        _require_exact(approval, DurableApproval)
        _require_exact(owner_id, PatchCommitOwnerId)
        _require_exact(now, ExpiryTick)
        _require_exact(lease_duration, DurationTicks)
        _require_artifacts(artifact_ids)
        self._approval_verifier.verify(approval)

        async def execute(cursor: PgsqlCursor) -> DurableCommitClaim:
            row = await _select_reservation_for_update(cursor, reservation)
            if (
                _row_str(row, "lifecycle")
                == LifecyclePhase.REQUEST_COMPLETED.value
            ):
                return DurableCommitClaim(
                    DurableCommitClaimState.TERMINAL,
                    None,
                    await _terminal(cursor, row),
                )
            if _plan_from_row(row) != plan:
                raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)
            if _row_optional_str(row, "owner_id") is not None:
                return DurableCommitClaim(
                    DurableCommitClaimState.ATTACHED, None, None
                )
            if _row_str(row, "lifecycle") != LifecyclePhase.PLANNED.value:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            _validate_approval(reservation, plan, approval, now)
            await _lock_domain(cursor, plan.domain_id)
            await cursor.execute(
                _INSERT_GRANT_CONSUMPTION_SQL,
                (
                    approval.grant_id.value,
                    approval.approval_id.value,
                    reservation.request_id.value,
                ),
            )
            consumed = await cursor.fetchone()
            if consumed is None:
                raise DurableStoreError(
                    DurableStoreErrorCode.APPROVAL_CONSUMED
                )
            fence = await _advance_domain_fence(cursor, plan.domain_id)
            expiry = _lease_expiry(now, lease_duration)
            await cursor.execute(
                _CLAIM_COMMIT_SQL,
                (
                    owner_id.value,
                    plan.domain_id.value,
                    fence.value,
                    expiry.value,
                    reservation.request_id.value,
                ),
            )
            claimed = await cursor.fetchone()
            if claimed is None:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            journal_cursor = DurableJournalCursor(
                reservation.request_id,
                SequenceNumber(_row_int(claimed, "journal_revision")),
            )
            for artifact_id in artifact_ids:
                journal_cursor = await _advance_journal(
                    cursor, claimed, journal_cursor
                )
                await cursor.execute(
                    _INSERT_ARTIFACT_JOURNAL_SQL,
                    (
                        reservation.request_id.value,
                        journal_cursor.revision.value,
                        artifact_id.value,
                        DurableArtifactState.INTENDED.value,
                    ),
                )
            return DurableCommitClaim(
                DurableCommitClaimState.OWNER,
                DurableCommitLease(
                    reservation.request_id,
                    plan.domain_id,
                    owner_id,
                    fence,
                    expiry,
                ),
                None,
            )

        return await self._transaction("patch_durable_claim_commit", execute)

    async def renew_lease(
        self,
        lease: DurableCommitLease,
        now: ExpiryTick,
        lease_duration: DurationTicks,
    ) -> DurableCommitLease:
        """Renew a current fenced SQL lease through exact CAS."""
        _require_exact(lease, DurableCommitLease)
        _require_exact(now, ExpiryTick)
        _require_exact(lease_duration, DurationTicks)
        expiry = _lease_expiry(now, lease_duration)
        if expiry.value <= lease.expires_at.value:
            raise DurableStoreError(DurableStoreErrorCode.LEASE_EXPIRED)

        async def execute(cursor: PgsqlCursor) -> DurableCommitLease:
            await _select_lease_for_update(cursor, lease, now)
            await cursor.execute(
                _RENEW_LEASE_SQL,
                (
                    expiry.value,
                    lease.request_id.value,
                    lease.domain_id.value,
                    lease.owner_id.value,
                    lease.fence.value,
                    lease.expires_at.value,
                    now.value,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            return DurableCommitLease(
                lease.request_id,
                lease.domain_id,
                lease.owner_id,
                lease.fence,
                expiry,
            )

        return await self._transaction("patch_durable_renew_lease", execute)

    async def replace_expired_owner(
        self,
        reservation: DurableReservation,
        expired_lease: DurableCommitLease,
        owner_id: PatchCommitOwnerId,
        now: ExpiryTick,
        lease_duration: DurationTicks,
    ) -> DurableCommitLease:
        """Fence an expired SQL lease before assigning a replacement owner."""
        _require_exact(reservation, DurableReservation)
        _require_exact(expired_lease, DurableCommitLease)
        _require_exact(owner_id, PatchCommitOwnerId)
        _require_exact(now, ExpiryTick)
        _require_exact(lease_duration, DurationTicks)
        if owner_id == expired_lease.owner_id:
            raise DurableStoreError(DurableStoreErrorCode.FENCED)

        async def execute(cursor: PgsqlCursor) -> DurableCommitLease:
            row = await _select_reservation_for_update(cursor, reservation)
            if _lease_from_row(row) != expired_lease:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            if now.value < expired_lease.expires_at.value:
                raise DurableStoreError(DurableStoreErrorCode.LEASE_EXPIRED)
            if _row_str(row, "lifecycle") not in {
                LifecyclePhase.COMMIT_STARTED.value,
                LifecyclePhase.SETTLEMENT_PENDING.value,
            }:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            await _lock_domain(cursor, expired_lease.domain_id)
            fence = await _advance_domain_fence(
                cursor, expired_lease.domain_id
            )
            expiry = _lease_expiry(now, lease_duration)
            await cursor.execute(
                _REPLACE_EXPIRED_LEASE_SQL,
                (
                    owner_id.value,
                    fence.value,
                    expiry.value,
                    reservation.request_id.value,
                    expired_lease.owner_id.value,
                    expired_lease.fence.value,
                    expired_lease.expires_at.value,
                    now.value,
                ),
            )
            replaced = await cursor.fetchone()
            if replaced is None:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            return DurableCommitLease(
                reservation.request_id,
                expired_lease.domain_id,
                owner_id,
                fence,
                expiry,
            )

        return await self._transaction("patch_durable_replace_owner", execute)

    async def is_current_fence(
        self, lease: DurableCommitLease, now: ExpiryTick
    ) -> bool:
        """Read whether an exact owner/fence remains current and unexpired."""
        _require_exact(lease, DurableCommitLease)
        _require_exact(now, ExpiryTick)

        async def execute(cursor: PgsqlCursor) -> bool:
            await cursor.execute(
                _CURRENT_FENCE_SQL,
                (
                    lease.request_id.value,
                    lease.domain_id.value,
                    lease.owner_id.value,
                    lease.fence.value,
                    lease.expires_at.value,
                    now.value,
                ),
            )
            return await cursor.fetchone() is not None

        return await self._transaction("patch_durable_current_fence", execute)

    async def append_step(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        step_id: PatchStepId,
        state: CommitStepState,
        now: ExpiryTick,
    ) -> DurableJournal:
        """Append one monotonic requested-effect transition through SQL CAS."""
        _require_exact(lease, DurableCommitLease)
        _require_exact(expected, DurableJournalCursor)
        _require_exact(step_id, PatchStepId)
        _require_exact(state, CommitStepState)
        _require_exact(now, ExpiryTick)

        async def execute(cursor: PgsqlCursor) -> DurableJournal:
            row = await _select_lease_for_update(cursor, lease, now)
            _require_cursor(row, expected)
            plan = _require_plan(_plan_from_row(row))
            binding = next(
                (item for item in plan.steps if item.step_id == step_id), None
            )
            if binding is None:
                raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
            current = await _step_state(cursor, lease.request_id, step_id)
            if not _step_transition(current, state):
                raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
            revision = await _advance_journal(cursor, row, expected)
            await cursor.execute(
                _INSERT_STEP_JOURNAL_SQL,
                (
                    lease.request_id.value,
                    revision.revision.value,
                    step_id.value,
                    binding.lineage_id.value,
                    state.value,
                ),
            )
            return await _journal(cursor, lease.request_id, revision.revision)

        return await self._transaction("patch_durable_append_step", execute)

    async def append_artifact(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        artifact_id: PatchArtifactId,
        state: DurableArtifactState,
        now: ExpiryTick,
    ) -> DurableJournal:
        """Append one monotonic visible-artifact transition through SQL CAS."""
        _require_exact(lease, DurableCommitLease)
        _require_exact(expected, DurableJournalCursor)
        _require_exact(artifact_id, PatchArtifactId)
        _require_exact(state, DurableArtifactState)
        _require_exact(now, ExpiryTick)

        async def execute(cursor: PgsqlCursor) -> DurableJournal:
            row = await _select_lease_for_update(cursor, lease, now)
            _require_cursor(row, expected)
            previous = await _artifact_state(
                cursor, lease.request_id, artifact_id
            )
            if previous is None or not _artifact_transition(previous, state):
                raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
            revision = await _advance_journal(cursor, row, expected)
            await cursor.execute(
                _INSERT_ARTIFACT_JOURNAL_SQL,
                (
                    lease.request_id.value,
                    revision.revision.value,
                    artifact_id.value,
                    state.value,
                ),
            )
            return await _journal(cursor, lease.request_id, revision.revision)

        return await self._transaction(
            "patch_durable_append_artifact", execute
        )

    async def suspend(
        self,
        lease: DurableCommitLease,
        pending: DurablePendingRequest,
        now: ExpiryTick,
    ) -> DurablePendingRecord:
        """Persist a fenced pending branch and durable outbox event."""
        _require_exact(lease, DurableCommitLease)
        _require_exact(pending, DurablePendingRequest)
        _require_exact(now, ExpiryTick)

        async def execute(cursor: PgsqlCursor) -> DurablePendingRecord:
            row = await _select_lease_for_update(cursor, lease, now)
            current = _pending_from_row(row)
            if current is not None:
                requested = DurablePendingRecord(
                    lease.request_id,
                    _identity_from_row(row).execution_id,
                    pending.pending_operation_id,
                    pending.correlation_id,
                    lease.fence,
                    current.event_cursor,
                    _row_bool(row, "cancellation_requested"),
                    pending.next_check_after,
                )
                if current == requested:
                    return current
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            if (
                _row_str(row, "lifecycle")
                != LifecyclePhase.COMMIT_STARTED.value
            ):
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            sequence = SequenceNumber(_row_int(row, "event_cursor") + 1)
            await cursor.execute(
                _SUSPEND_SQL,
                (
                    pending.pending_operation_id.value,
                    pending.correlation_id.value,
                    pending.next_check_after.value,
                    sequence.value,
                    sequence.value,
                    lease.request_id.value,
                    lease.owner_id.value,
                    lease.fence.value,
                    lease.expires_at.value,
                    now.value,
                ),
            )
            changed = await cursor.fetchone()
            if changed is None:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            await _insert_outbox(
                cursor,
                lease.request_id,
                sequence,
                LifecyclePhase.SETTLEMENT_PENDING,
                pending.correlation_id,
            )
            return DurablePendingRecord(
                lease.request_id,
                _identity_from_row(row).execution_id,
                pending.pending_operation_id,
                pending.correlation_id,
                lease.fence,
                sequence,
                _row_bool(row, "cancellation_requested"),
                pending.next_check_after,
            )

        return await self._transaction("patch_durable_suspend", execute)

    async def request_cancellation(
        self, access: DurableRequestAccess
    ) -> DurableRequestSnapshot:
        """Persist cancellation intent without releasing the commit fence."""
        _require_exact(access, DurableRequestAccess)

        async def execute(cursor: PgsqlCursor) -> DurableRequestSnapshot:
            row = await _select_access_for_update(cursor, access)
            if _row_str(row, "lifecycle") not in {
                LifecyclePhase.COMMIT_STARTED.value,
                LifecyclePhase.SETTLEMENT_PENDING.value,
            }:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            lease = _lease_from_row(row)
            if lease is None:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            await _require_current_domain_fence(cursor, lease)
            await cursor.execute(
                _REQUEST_CANCELLATION_SQL,
                (access.request_id.value,),
            )
            changed = await _require_row(cursor)
            return await _snapshot(cursor, changed)

        return await self._transaction(
            "patch_durable_request_cancellation", execute
        )

    async def settle(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        result: PatchResult,
        correlation_id: PatchObserverCorrelationId,
        now: ExpiryTick,
    ) -> DurableTerminalRecord:
        """Atomically settle result and terminal outbox record under CAS."""
        _require_exact(lease, DurableCommitLease)
        _require_exact(expected, DurableJournalCursor)
        _require_exact(result, PatchResult)
        _require_exact(correlation_id, PatchObserverCorrelationId)
        _require_exact(now, ExpiryTick)

        async def execute(cursor: PgsqlCursor) -> DurableTerminalRecord:
            row = await _select_request_for_update(cursor, lease.request_id)
            if (
                _row_str(row, "lifecycle")
                == LifecyclePhase.REQUEST_COMPLETED.value
            ):
                terminal = await _terminal(cursor, row)
                if terminal.result != result:
                    raise DurableStoreError(
                        DurableStoreErrorCode.TERMINAL_CONFLICT
                    )
                return terminal
            _require_lease(row, lease, now)
            await _require_current_domain_fence(cursor, lease)
            _require_cursor(row, expected)
            plan = _require_plan(_plan_from_row(row))
            if (
                result.request_id != lease.request_id
                or result.plan_id != plan.plan_id
            ):
                raise DurableStoreError(
                    DurableStoreErrorCode.TERMINAL_CONFLICT
                )
            pending = _pending_from_row(row)
            if (
                pending is not None
                and pending.correlation_id != correlation_id
            ):
                raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)
            journal = await _journal(
                cursor, lease.request_id, expected.revision
            )
            mutation = _journal_mutation_state(journal, plan)
            artifact = await _journal_artifact_state(
                cursor, lease.request_id, expected.revision
            )
            if (
                result.truth.mutation_state is not mutation
                or result.truth.artifact_state is not artifact
            ):
                raise DurableStoreError(
                    DurableStoreErrorCode.TERMINAL_CONFLICT
                )
            sequence = SequenceNumber(_row_int(row, "event_cursor") + 1)
            await cursor.execute(
                _SETTLE_SQL,
                (
                    encode_result(result),
                    correlation_id.value,
                    (
                        None
                        if pending is None
                        else pending.pending_operation_id.value
                    ),
                    sequence.value,
                    lease.request_id.value,
                    lease.owner_id.value,
                    lease.fence.value,
                    lease.expires_at.value,
                    now.value,
                    expected.revision.value,
                ),
            )
            changed = await cursor.fetchone()
            if changed is None:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            outbox = await _insert_outbox(
                cursor,
                lease.request_id,
                sequence,
                LifecyclePhase.REQUEST_COMPLETED,
                correlation_id,
            )
            await cursor.execute(
                _DELETE_TERMINAL_RETENTION_SQL,
                (lease.request_id.value,),
            )
            return DurableTerminalRecord(
                result,
                outbox,
                (None if pending is None else pending.pending_operation_id),
            )

        return await self._transaction("patch_durable_settle", execute)

    async def inspect(
        self, access: DurableRequestAccess
    ) -> DurableRequestSnapshot:
        """Read one authenticated content-free durable record snapshot."""
        _require_exact(access, DurableRequestAccess)

        async def execute(cursor: PgsqlCursor) -> DurableRequestSnapshot:
            return await _snapshot(
                cursor, await _select_access_for_update(cursor, access)
            )

        return await self._transaction("patch_durable_inspect", execute)

    async def inspect_pending(
        self, access: DurablePendingAccess
    ) -> DurablePendingRecord | DurableTerminalRecord:
        """Read the original pending correlation or terminal result."""
        _require_exact(access, DurablePendingAccess)

        async def execute(
            cursor: PgsqlCursor,
        ) -> DurablePendingRecord | DurableTerminalRecord:
            row = await _select_access_for_update(cursor, access.request)
            pending = _pending_from_row(row)
            if pending is not None:
                if not _pending_access_matches(access, pending):
                    raise DurableStoreError(
                        DurableStoreErrorCode.ACCESS_DENIED
                    )
                return pending
            terminal = await _terminal(cursor, row)
            if not _terminal_access_matches(access, terminal):
                raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)
            return terminal

        return await self._transaction(
            "patch_durable_inspect_pending", execute
        )

    async def await_terminal(
        self, access: DurablePendingAccess
    ) -> DurableTerminalRecord:
        """Await terminal reconciliation without retaining SQL state."""
        while True:
            outcome = await self.inspect_pending(access)
            if type(outcome) is DurableTerminalRecord:
                return outcome
            await sleep(0.05)

    async def outbox(
        self,
        access: DurableRequestAccess,
        after: SequenceNumber,
        limit: int,
    ) -> tuple[DurableOutboxRecord, ...]:
        """Read stable ordered outbox records without acknowledgement."""
        _require_exact(access, DurableRequestAccess)
        _require_exact(after, SequenceNumber)
        if type(limit) is not int or not 1 <= limit <= 1024:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

        async def execute(
            cursor: PgsqlCursor,
        ) -> tuple[DurableOutboxRecord, ...]:
            await _select_access_for_update(cursor, access)
            await cursor.execute(
                _SELECT_OUTBOX_SQL,
                (access.request_id.value, after.value, limit),
            )
            return tuple(
                _outbox_from_row(item) for item in await cursor.fetchall()
            )

        return await self._transaction("patch_durable_outbox", execute)

    async def put_retention(
        self,
        reservation: DurableReservation,
        record: DurableRetentionRecord,
    ) -> None:
        """Persist one bounded versioned-key ciphertext record idempotently."""
        _require_exact(reservation, DurableReservation)
        _require_exact(record, DurableRetentionRecord)

        async def validate_reservation(cursor: PgsqlCursor) -> None:
            await _select_reservation_for_update(cursor, reservation)

        await self._transaction(
            "patch_durable_validate_retention_reservation",
            validate_reservation,
        )
        await self._retention_validator.validate(
            reservation.request_id, record
        )

        async def execute(cursor: PgsqlCursor) -> None:
            request_row = await _select_reservation_for_update(
                cursor, reservation
            )
            await cursor.execute(
                _COUNT_RETENTION_SQL,
                (reservation.request_id.value,),
            )
            if _row_int(await _require_row(cursor), "record_count") >= 128:
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_LIMIT)
            await cursor.execute(
                _SUM_RETENTION_SQL,
                (reservation.request_id.value,),
            )
            total = _row_int(await _require_row(cursor), "byte_count")
            if total + record.value.size().value > 4_194_304:
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_LIMIT)
            if (
                _row_str(request_row, "lifecycle")
                == LifecyclePhase.REQUEST_COMPLETED.value
            ):
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
            await cursor.execute(
                _INSERT_RETENTION_SQL,
                (
                    record.retention_id.value,
                    reservation.request_id.value,
                    record.kind.value,
                    record.key_id.value,
                    record.value._ciphertext,
                    record.value.digest().value,
                    record.policy.expires_at.value,
                    record.policy.delete_on_terminal,
                ),
            )
            existing = await cursor.fetchone()
            if existing is None:
                await cursor.execute(
                    _SELECT_RETENTION_SQL,
                    (record.retention_id.value, reservation.request_id.value),
                )
                stored = _retention_from_row(await _require_row(cursor))
                if stored != record:
                    raise DurableStoreError(
                        DurableStoreErrorCode.RETENTION_CONFLICT
                    )

        await self._transaction("patch_durable_put_retention", execute)

    async def get_retention(
        self,
        access: DurableRetentionAccess,
        retention_id: PatchRetentionRecordId,
        now: ExpiryTick,
    ) -> DurableRetentionRecord:
        """Read one authorized unexpired versioned-key ciphertext record."""
        _require_exact(access, DurableRetentionAccess)
        _require_exact(retention_id, PatchRetentionRecordId)
        _require_exact(now, ExpiryTick)

        async def execute(
            cursor: PgsqlCursor,
        ) -> tuple[DurableRequestIdentity, DurableRetentionRecord]:
            request_row = await _select_access_for_update(
                cursor, access.request
            )
            await cursor.execute(
                _SELECT_RETENTION_SQL,
                (retention_id.value, access.request.request_id.value),
            )
            retention_row = await cursor.fetchone()
            if retention_row is None:
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
            record = _retention_from_row(retention_row)
            if now.value >= record.policy.expires_at.value:
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
            return _identity_from_row(request_row), record

        identity, record = await self._transaction(
            "patch_durable_get_retention", execute
        )
        audiences = await self._retention_authorizer.audiences_for(
            identity, record.kind
        )
        if (
            type(audiences) is not frozenset
            or not audiences
            or any(type(item) is not Audience for item in audiences)
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        await self._retention_validator.validate(
            access.request.request_id, record
        )
        return record

    async def cleanup_retention(
        self, now: ExpiryTick
    ) -> DurableRetentionCleanup:
        """Delete expired selected ciphertext without mutation changes."""
        _require_exact(now, ExpiryTick)

        async def execute(cursor: PgsqlCursor) -> DurableRetentionCleanup:
            await cursor.execute(_DELETE_EXPIRED_RETENTION_SQL, (now.value,))
            row = await _require_row(cursor)
            return DurableRetentionCleanup(
                _row_int(row, "record_count"),
                ByteSize(_row_int(row, "byte_count")),
            )

        return await self._transaction(
            "patch_durable_cleanup_retention", execute
        )

    async def _transaction(
        self,
        operation: str,
        callback: Callable[[PgsqlCursor], Awaitable[_T]],
    ) -> _T:
        """Run one owned short transaction with no target or worker await."""
        if not self._opened or self._closed:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        try:
            async with self._database.connection() as connection:
                async with connection.transaction():
                    async with connection.cursor() as cursor:
                        return await callback(cursor)
        except DurableStoreError:
            raise
        except (KeyboardInterrupt, SystemExit, CancelledError):
            raise
        except BaseException as error:
            failure = classify_pgsql_error(error, operation=operation)
            if failure.category == PgsqlFailureCategory.UNIQUE_CONFLICT:
                raise DurableStoreError(
                    DurableStoreErrorCode.JOURNAL_CONFLICT
                ) from None
            raise DurableStoreError(
                DurableStoreErrorCode.LIFECYCLE_CONFLICT
            ) from None


def _require_exact(value: object, expected: type[object]) -> None:
    """Reject values outside the closed durable-store semantic boundary."""
    if type(value) is not expected:
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


def _require_artifacts(artifact_ids: tuple[PatchArtifactId, ...]) -> None:
    """Require a bounded exact artifact-intent vector before SQL mutation."""
    if (
        type(artifact_ids) is not tuple
        or len(artifact_ids) > 1024
        or any(type(item) is not PatchArtifactId for item in artifact_ids)
        or len(set(artifact_ids)) != len(artifact_ids)
    ):
        raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)


def _lease_expiry(now: ExpiryTick, duration: DurationTicks) -> ExpiryTick:
    """Return one bounded finite lease expiry from the trusted tick clock."""
    value = now.value + duration.value
    if value > 2**63 - 1:
        raise DurableStoreError(DurableStoreErrorCode.LEASE_EXPIRED)
    return ExpiryTick(value)


def _validate_approval(
    reservation: DurableReservation,
    plan: DurablePlanReference,
    approval: DurableApproval,
    now: ExpiryTick,
) -> None:
    """Require exact unexpired approval binding before SQL consumption."""
    if (
        approval.identity != reservation.identity
        or approval.canonical_digest != reservation.canonical_digest
        or approval.plan_id != plan.plan_id
        or approval.fingerprint_digest != plan.fingerprint_digest
        or approval.review_digest != plan.review_digest
        or approval.context_id != plan.context_id
        or approval.workspace_id != plan.workspace_id
        or approval.domain_id != plan.domain_id
    ):
        raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)
    if now.value >= approval.expires_at.value:
        raise DurableStoreError(DurableStoreErrorCode.APPROVAL_EXPIRED)


def _encode_plan(plan: DurablePlanReference) -> bytes:
    """Encode non-content plan evidence with an exact closed field grammar."""
    steps = _STEP_SEPARATOR.join(
        f"{item.step_id.value}:{item.lineage_id.value}" for item in plan.steps
    )
    fields = (
        _PLAN_TAG,
        plan.plan_id.value,
        plan.canonical_digest.value,
        plan.fingerprint_digest.value,
        plan.review_digest.value,
        plan.context_id.value,
        plan.workspace_id.value,
        plan.domain_id.value,
        steps,
    )
    if any(_PLAN_SEPARATOR in item or "\n" in item for item in fields):
        raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)
    return _PLAN_SEPARATOR.join(fields).encode("ascii")


def _decode_plan(payload: bytes) -> DurablePlanReference:
    """Decode one exact sealed non-content plan reference."""
    try:
        fields = payload.decode("ascii").split(_PLAN_SEPARATOR)
        if len(fields) != 9 or fields[0] != _PLAN_TAG or not fields[8]:
            raise ValueError
        steps = tuple(
            DurableStepBinding(
                PatchStepId(item.split(":", 1)[0]),
                PatchLineageId(item.split(":", 1)[1]),
            )
            for item in fields[8].split(_STEP_SEPARATOR)
        )
        return DurablePlanReference(
            PatchPlanId(fields[1]),
            AlgorithmDigest("sha256", fields[2]),
            AlgorithmDigest("sha256", fields[3]),
            AlgorithmDigest("sha256", fields[4]),
            PatchContextId(fields[5]),
            PatchWorkspaceId(fields[6]),
            PatchDomainId(fields[7]),
            steps,
        )
    except (UnicodeDecodeError, ValueError, IndexError) as error:
        raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH) from error


async def _select_identity_for_update(
    cursor: PgsqlCursor, identity: DurableRequestIdentity
) -> PgsqlRow:
    """Lock one retransmission identity after insert-or-attach selection."""
    await cursor.execute(
        _SELECT_IDENTITY_FOR_UPDATE_SQL,
        (
            identity.tenant_id.value,
            identity.principal_id.value,
            identity.execution_id.value,
            identity.route_id.value,
            identity.retransmission_key.value,
        ),
    )
    return await _require_row(cursor)


async def _select_reservation_for_update(
    cursor: PgsqlCursor, reservation: DurableReservation
) -> PgsqlRow:
    """Lock and authenticate reservation without exposing other rows."""
    row = await _select_identity_for_update(cursor, reservation.identity)
    if (
        _row_str(row, "request_id") != reservation.request_id.value
        or _row_str(row, "canonical_digest")
        != reservation.canonical_digest.value
    ):
        raise DurableStoreError(DurableStoreErrorCode.INVALID_RESERVATION)
    return row


async def _select_request_for_update(
    cursor: PgsqlCursor, request_id: PatchRequestId
) -> PgsqlRow:
    """Lock one known request row without a cross-identity projection."""
    await cursor.execute(_SELECT_REQUEST_FOR_UPDATE_SQL, (request_id.value,))
    return await _require_row(cursor)


async def _select_access_for_update(
    cursor: PgsqlCursor, access: DurableRequestAccess
) -> PgsqlRow:
    """Lock one authenticated request while denying mismatches uniformly."""
    try:
        row = await _select_identity_for_update(cursor, access.identity)
    except DurableStoreError as error:
        if error.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT:
            raise DurableStoreError(
                DurableStoreErrorCode.ACCESS_DENIED
            ) from None
        raise
    if _row_str(row, "request_id") != access.request_id.value:
        raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)
    return row


async def _select_lease_for_update(
    cursor: PgsqlCursor, lease: DurableCommitLease, now: ExpiryTick
) -> PgsqlRow:
    """Lock one exact current lease and reject stale owners before mutation."""
    row = await _select_request_for_update(cursor, lease.request_id)
    _require_lease(row, lease, now)
    await _require_current_domain_fence(cursor, lease)
    return row


def _require_lease(
    row: PgsqlRow, lease: DurableCommitLease, now: ExpiryTick
) -> None:
    """Fail closed unless a row retains the exact live owner/fence witness."""
    if _lease_from_row(row) != lease or _row_str(row, "lifecycle") not in {
        LifecyclePhase.COMMIT_STARTED.value,
        LifecyclePhase.SETTLEMENT_PENDING.value,
    }:
        raise DurableStoreError(DurableStoreErrorCode.FENCED)
    if now.value >= lease.expires_at.value:
        raise DurableStoreError(DurableStoreErrorCode.LEASE_EXPIRED)


def _require_cursor(row: PgsqlRow, expected: DurableJournalCursor) -> None:
    """Require a request-bound compare-and-set journal cursor."""
    if (
        _row_str(row, "request_id") != expected.request_id.value
        or _row_int(row, "journal_revision") != expected.revision.value
    ):
        raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)


async def _advance_journal(
    cursor: PgsqlCursor,
    row: PgsqlRow,
    expected: DurableJournalCursor,
) -> DurableJournalCursor:
    """Advance one bounded journal revision through row-level CAS."""
    if expected.revision.value >= 8192:
        raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
    revision = SequenceNumber(expected.revision.value + 1)
    await cursor.execute(
        _ADVANCE_JOURNAL_SQL,
        (revision.value, expected.request_id.value, expected.revision.value),
    )
    if await cursor.fetchone() is None:
        raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
    return DurableJournalCursor(
        PatchRequestId(_row_str(row, "request_id")), revision
    )


async def _step_state(
    cursor: PgsqlCursor, request_id: PatchRequestId, step_id: PatchStepId
) -> CommitStepState | None:
    """Read the latest step state under the caller's locked request row."""
    await cursor.execute(
        _SELECT_STEP_STATE_SQL, (request_id.value, step_id.value)
    )
    row = await cursor.fetchone()
    return None if row is None else CommitStepState(_row_str(row, "state"))


async def _artifact_state(
    cursor: PgsqlCursor,
    request_id: PatchRequestId,
    artifact_id: PatchArtifactId,
) -> DurableArtifactState | None:
    """Read one journal-derived artifact state under the locked request row."""
    await cursor.execute(
        _SELECT_ARTIFACT_STATE_SQL, (request_id.value, artifact_id.value)
    )
    row = await cursor.fetchone()
    return (
        None if row is None else DurableArtifactState(_row_str(row, "state"))
    )


async def _journal(
    cursor: PgsqlCursor,
    request_id: PatchRequestId,
    revision: SequenceNumber,
) -> DurableJournal:
    """Read one complete immutable journal history under a request row lock."""
    await cursor.execute(_SELECT_STEP_JOURNAL_SQL, (request_id.value,))
    steps = tuple(
        DurableStepJournalEntry(
            DurableJournalCursor(
                request_id, SequenceNumber(_row_int(row, "revision"))
            ),
            PatchStepId(_row_str(row, "step_id")),
            PatchLineageId(_row_str(row, "lineage_id")),
            CommitStepState(_row_str(row, "state")),
        )
        for row in await cursor.fetchall()
    )
    await cursor.execute(_SELECT_ARTIFACT_JOURNAL_SQL, (request_id.value,))
    artifacts = tuple(
        DurableArtifactJournalEntry(
            DurableJournalCursor(
                request_id, SequenceNumber(_row_int(row, "revision"))
            ),
            PatchArtifactId(_row_str(row, "artifact_id")),
            DurableArtifactState(_row_str(row, "state")),
        )
        for row in await cursor.fetchall()
    )
    return DurableJournal(
        DurableJournalCursor(request_id, revision), steps, artifacts
    )


async def _snapshot(
    cursor: PgsqlCursor, row: PgsqlRow
) -> DurableRequestSnapshot:
    """Build one coherent content-free snapshot under a held row lock."""
    reservation = _reservation_from_row(row)
    return DurableRequestSnapshot(
        reservation,
        _plan_from_row(row),
        LifecyclePhase(_row_str(row, "lifecycle")),
        _lease_from_row(row),
        await _journal(
            cursor,
            reservation.request_id,
            SequenceNumber(_row_int(row, "journal_revision")),
        ),
        _pending_from_row(row),
        (
            await _terminal(cursor, row)
            if _row_str(row, "lifecycle")
            == LifecyclePhase.REQUEST_COMPLETED.value
            else None
        ),
        _row_bool(row, "cancellation_requested"),
        SequenceNumber(_row_int(row, "event_cursor")),
    )


async def _terminal(
    cursor: PgsqlCursor, row: PgsqlRow
) -> DurableTerminalRecord:
    """Read exact terminal result and uniquely keyed terminal outbox record."""
    payload = _row_bytes(row, "terminal_result")
    if payload is None:
        raise DurableStoreError(DurableStoreErrorCode.TERMINAL_CONFLICT)
    await cursor.execute(
        _SELECT_TERMINAL_OUTBOX_SQL, (_row_str(row, "request_id"),)
    )
    return DurableTerminalRecord(
        decode_result(payload),
        _outbox_from_row(await _require_row(cursor)),
        (
            None
            if _row_optional_str(row, "terminal_pending_operation_id") is None
            else PatchPendingOperationId(
                _row_str(row, "terminal_pending_operation_id")
            )
        ),
    )


def _reservation_from_row(row: PgsqlRow) -> DurableReservation:
    """Decode one stored reservation as durable identity evidence."""
    identity = _identity_from_row(row)
    return DurableReservation(
        PatchRequestId(_row_str(row, "request_id")),
        identity,
        AlgorithmDigest("sha256", _row_str(row, "canonical_digest")),
        False,
    )


def _identity_from_row(row: PgsqlRow) -> DurableRequestIdentity:
    """Decode exact stored authenticated retransmission tuple fields."""
    return DurableRequestIdentity(
        PatchTenantId(_row_str(row, "tenant_id")),
        PatchPrincipalId(_row_str(row, "principal_id")),
        PatchExecutionId(_row_str(row, "execution_id")),
        PolicyRouteId(_row_str(row, "route_id")),
        RetransmissionKey(_row_str(row, "retransmission_key")),
    )


def _plan_from_row(row: PgsqlRow) -> DurablePlanReference | None:
    """Decode optional plan bytes and fail closed on invalid records."""
    value = _row_bytes(row, "plan_payload")
    return None if value is None else _decode_plan(value)


def _lease_from_row(row: PgsqlRow) -> DurableCommitLease | None:
    """Decode optional current fence witness from one request row."""
    owner = _row_optional_str(row, "owner_id")
    if owner is None:
        return None
    domain = _row_optional_str(row, "domain_id")
    expires = _row_optional_int(row, "lease_expires_at")
    if domain is None or expires is None:
        raise DurableStoreError(DurableStoreErrorCode.FENCED)
    return DurableCommitLease(
        PatchRequestId(_row_str(row, "request_id")),
        PatchDomainId(domain),
        PatchCommitOwnerId(owner),
        SequenceNumber(_row_int(row, "fence")),
        ExpiryTick(expires),
    )


def _pending_from_row(row: PgsqlRow) -> DurablePendingRecord | None:
    """Decode optional exact pending continuation state from one row."""
    identifier = _row_optional_str(row, "pending_operation_id")
    if identifier is None:
        return None
    correlation = _row_optional_str(row, "pending_correlation_id")
    next_check = _row_optional_int(row, "pending_next_check_after")
    cursor = _row_optional_int(row, "pending_event_cursor")
    lease = _lease_from_row(row)
    if (
        correlation is None
        or next_check is None
        or cursor is None
        or lease is None
    ):
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
    return DurablePendingRecord(
        PatchRequestId(_row_str(row, "request_id")),
        _identity_from_row(row).execution_id,
        PatchPendingOperationId(identifier),
        PatchObserverCorrelationId(correlation),
        lease.fence,
        SequenceNumber(cursor),
        _row_bool(row, "cancellation_requested"),
        DurationTicks(next_check),
    )


def _outbox_from_row(row: PgsqlRow) -> DurableOutboxRecord:
    """Decode one content-free ordered durable outbox record."""
    return DurableOutboxRecord(
        PatchEventId(_row_str(row, "event_id")),
        PatchRequestId(_row_str(row, "request_id")),
        SequenceNumber(_row_int(row, "sequence")),
        LifecyclePhase(_row_str(row, "lifecycle")),
        PatchObserverCorrelationId(_row_str(row, "correlation_id")),
    )


async def _insert_outbox(
    cursor: PgsqlCursor,
    request_id: PatchRequestId,
    sequence: SequenceNumber,
    lifecycle: LifecyclePhase,
    correlation_id: PatchObserverCorrelationId,
) -> DurableOutboxRecord:
    """Insert exactly one stable at-least-once lifecycle outbox record."""
    event_id = PatchEventId.new()
    await cursor.execute(
        _INSERT_OUTBOX_SQL,
        (
            event_id.value,
            request_id.value,
            sequence.value,
            lifecycle.value,
            correlation_id.value,
        ),
    )
    return _outbox_from_row(await _require_row(cursor))


async def _journal_artifact_state(
    cursor: PgsqlCursor,
    request_id: PatchRequestId,
    revision: SequenceNumber,
) -> ArtifactState:
    """Derive terminal artifact truth only from ordered journal history."""
    return derive_artifact_state(
        (await _journal(cursor, request_id, revision)).artifacts
    )


def _journal_mutation_state(
    journal: DurableJournal, plan: DurablePlanReference
) -> MutationState:
    """Derive exact mutation truth only from complete durable step history."""
    states = {item.step_id: item.state for item in journal.steps}
    values: list[CommitStepState] = []
    for binding in plan.steps:
        state = states.get(binding.step_id)
        if state is None or state is CommitStepState.PLANNED:
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_INCOMPLETE)
        values.append(state)
    if any(item is CommitStepState.UNKNOWN for item in values):
        return MutationState.INDETERMINATE
    committed = sum(item is CommitStepState.COMMITTED for item in values)
    if committed == 0:
        return MutationState.NOT_COMMITTED
    if committed == len(values):
        return MutationState.COMMITTED
    return MutationState.PARTIALLY_COMMITTED


def _step_transition(
    previous: CommitStepState | None, next_state: CommitStepState
) -> bool:
    """Return whether one requested-effect transition remains monotonic."""
    if previous is None:
        return next_state is CommitStepState.PLANNED
    return previous is CommitStepState.PLANNED and next_state in {
        CommitStepState.COMMITTED,
        CommitStepState.NOT_COMMITTED,
        CommitStepState.UNKNOWN,
    }


def _artifact_transition(
    previous: DurableArtifactState, next_state: DurableArtifactState
) -> bool:
    """Return whether one target artifact transition remains monotonic."""
    allowed = {
        DurableArtifactState.INTENDED: {
            DurableArtifactState.NOT_CREATED,
            DurableArtifactState.PRESENT,
            DurableArtifactState.UNKNOWN,
        },
        DurableArtifactState.PRESENT: {
            DurableArtifactState.REMOVED,
            DurableArtifactState.LEAKED,
            DurableArtifactState.UNKNOWN,
        },
        DurableArtifactState.NOT_CREATED: set[DurableArtifactState](),
        DurableArtifactState.REMOVED: set[DurableArtifactState](),
        DurableArtifactState.LEAKED: set[DurableArtifactState](),
        DurableArtifactState.UNKNOWN: set[DurableArtifactState](),
    }
    return next_state in allowed[previous]


def _require_plan(plan: DurablePlanReference | None) -> DurablePlanReference:
    """Return persisted plan evidence or fail closed before mutation."""
    if plan is None:
        raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)
    return plan


def _pending_access_matches(
    access: DurablePendingAccess, pending: DurablePendingRecord
) -> bool:
    """Return whether exact original pending correlation values match."""
    return (
        access.pending_operation_id == pending.pending_operation_id
        and access.correlation_id == pending.correlation_id
    )


def _terminal_access_matches(
    access: DurablePendingAccess, terminal: DurableTerminalRecord
) -> bool:
    """Return whether terminal access preserves the original pending handle."""
    return (
        access.pending_operation_id == terminal.pending_operation_id
        and access.correlation_id == terminal.outbox.correlation_id
    )


def _retention_from_row(row: PgsqlRow) -> DurableRetentionRecord:
    """Decode one encrypted retention row without plaintext conversion."""
    ciphertext = _row_bytes(row, "ciphertext")
    if ciphertext is None:
        raise DurableStoreError(DurableStoreErrorCode.RETENTION_CONFLICT)
    value = EncryptedRetentionValue(ciphertext)
    if value.digest().value != _row_str(row, "ciphertext_digest"):
        raise DurableStoreError(DurableStoreErrorCode.RETENTION_CONFLICT)
    return DurableRetentionRecord(
        PatchRetentionRecordId(_row_str(row, "retention_id")),
        DurableRetentionKind(_row_str(row, "kind")),
        PatchRetentionKeyId(_row_str(row, "key_id")),
        value,
        DurableRetentionPolicy(
            ExpiryTick(_row_int(row, "expires_at")),
            _row_bool(row, "delete_on_terminal"),
        ),
    )


async def _lock_domain(cursor: PgsqlCursor, domain_id: PatchDomainId) -> None:
    """Acquire one transaction-scoped per-domain advisory lock in SQL."""
    await cursor.execute(
        _LOCK_DOMAIN_SQL,
        (domain_id.value, _DOMAIN_ADVISORY_LOCK),
    )


async def _advance_domain_fence(
    cursor: PgsqlCursor, domain_id: PatchDomainId
) -> SequenceNumber:
    """Persist and return the next domain-wide fencing epoch atomically."""
    await cursor.execute(_INSERT_DOMAIN_SQL, (domain_id.value,))
    await cursor.execute(_ADVANCE_DOMAIN_FENCE_SQL, (domain_id.value,))
    row = await _require_row(cursor)
    return SequenceNumber(_row_int(row, "current_fence"))


async def _require_current_domain_fence(
    cursor: PgsqlCursor, lease: DurableCommitLease
) -> None:
    """Lock and require the authoritative domain epoch for one lease."""
    await cursor.execute(
        _SELECT_DOMAIN_FENCE_FOR_SHARE_SQL,
        (lease.domain_id.value,),
    )
    row = await cursor.fetchone()
    if row is None or _row_int(row, "current_fence") != lease.fence.value:
        raise DurableStoreError(DurableStoreErrorCode.FENCED)


async def _require_row(cursor: PgsqlCursor) -> PgsqlRow:
    """Return one database row or fail closed on missing durable evidence."""
    row = await cursor.fetchone()
    if row is None:
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
    return row


def _row_value_string(value: object) -> str:
    """Require one stored nonempty string without coercion."""
    if type(value) is not str or not value:
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
    return value


def _row_str(row: PgsqlRow, key: str) -> str:
    """Read one required nonempty string column from a durable row."""
    return _row_value_string(row.get(key))


def _row_optional_str(row: PgsqlRow, key: str) -> str | None:
    """Read one optional nonempty string column from a durable row."""
    value = row.get(key)
    if value is None:
        return None
    return _row_value_string(value)


def _row_int(row: PgsqlRow, key: str) -> int:
    """Read one required nonnegative integer column from a durable row."""
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
    return value


def _row_optional_int(row: PgsqlRow, key: str) -> int | None:
    """Read one optional positive integer column from a durable row."""
    value = row.get(key)
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
    return value


def _row_bool(row: PgsqlRow, key: str) -> bool:
    """Read one exact Boolean column from a durable row."""
    value = row.get(key)
    if type(value) is not bool:
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
    return value


def _row_bytes(row: PgsqlRow, key: str) -> bytes | None:
    """Read one optional immutable bytea column without content decoding."""
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, memoryview):
        value = value.tobytes()
    if type(value) is not bytes:
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
    return value


_INSERT_RESERVATION_SQL = """
INSERT INTO "patch_durable_requests" (
    "request_id", "tenant_id", "principal_id", "execution_id", "route_id",
    "retransmission_key", "canonical_digest"
) VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT ON CONSTRAINT "uq_patch_durable_requests_retransmission"
DO NOTHING
RETURNING "request_id"
"""

_SELECT_IDENTITY_FOR_UPDATE_SQL = """
SELECT *
FROM "patch_durable_requests"
WHERE "tenant_id" = %s
  AND "principal_id" = %s
  AND "execution_id" = %s
  AND "route_id" = %s
  AND "retransmission_key" = %s
FOR UPDATE
"""

_SELECT_REQUEST_FOR_UPDATE_SQL = """
SELECT *
FROM "patch_durable_requests"
WHERE "request_id" = %s
FOR UPDATE
"""

_PERSIST_PLAN_SQL = """
UPDATE "patch_durable_requests"
SET "plan_payload" = %s, "lifecycle" = 'planned',
    "updated_at" = CURRENT_TIMESTAMP
WHERE "request_id" = %s
  AND "plan_payload" IS NULL
  AND "lifecycle" = 'received'
RETURNING *
"""

_LOCK_DOMAIN_SQL = """
SELECT pg_advisory_xact_lock(hashtextextended(%s, %s))
"""

_INSERT_DOMAIN_SQL = """
INSERT INTO "patch_durable_domains" ("domain_id")
VALUES (%s)
ON CONFLICT ("domain_id") DO NOTHING
"""

_ADVANCE_DOMAIN_FENCE_SQL = """
UPDATE "patch_durable_domains"
SET "current_fence" = "current_fence" + 1,
    "updated_at" = CURRENT_TIMESTAMP
WHERE "domain_id" = %s
  AND "current_fence" < 9223372036854775807
RETURNING "current_fence"
"""

_SELECT_DOMAIN_FENCE_FOR_SHARE_SQL = """
SELECT "current_fence"
FROM "patch_durable_domains"
WHERE "domain_id" = %s
FOR SHARE
"""

_INSERT_GRANT_CONSUMPTION_SQL = """
INSERT INTO "patch_durable_grant_consumptions" (
    "grant_id", "approval_id", "request_id"
) VALUES (%s, %s, %s)
ON CONFLICT DO NOTHING
RETURNING "grant_id"
"""

_CLAIM_COMMIT_SQL = """
UPDATE "patch_durable_requests"
SET "owner_id" = %s, "domain_id" = %s, "fence" = %s,
    "lease_expires_at" = %s, "lifecycle" = 'commit_started',
    "updated_at" = CURRENT_TIMESTAMP
WHERE "request_id" = %s
  AND "lifecycle" = 'planned'
  AND "owner_id" IS NULL
RETURNING *
"""

_INSERT_ARTIFACT_INTENT_SQL = """
INSERT INTO "patch_durable_artifact_intents" (
    "request_id", "artifact_id"
) VALUES (%s, %s)
"""

_RENEW_LEASE_SQL = """
UPDATE "patch_durable_requests" AS "request"
SET "lease_expires_at" = %s, "updated_at" = CURRENT_TIMESTAMP
FROM "patch_durable_domains" AS "domain"
WHERE "request"."request_id" = %s
  AND "request"."domain_id" = %s
  AND "request"."owner_id" = %s
  AND "request"."fence" = %s
  AND "request"."lease_expires_at" = %s
  AND "request"."lease_expires_at" > %s
  AND "request"."lifecycle" IN ('commit_started', 'settlement_pending')
  AND "domain"."domain_id" = "request"."domain_id"
  AND "domain"."current_fence" = "request"."fence"
RETURNING *
"""

_REPLACE_EXPIRED_LEASE_SQL = """
UPDATE "patch_durable_requests"
SET "owner_id" = %s, "fence" = %s, "lease_expires_at" = %s,
    "updated_at" = CURRENT_TIMESTAMP
WHERE "request_id" = %s
  AND "owner_id" = %s
  AND "fence" = %s
  AND "lease_expires_at" = %s
  AND "lease_expires_at" <= %s
  AND "lifecycle" IN ('commit_started', 'settlement_pending')
RETURNING *
"""

_CURRENT_FENCE_SQL = """
SELECT "request"."request_id"
FROM "patch_durable_requests" AS "request"
JOIN "patch_durable_domains" AS "domain"
  ON "domain"."domain_id" = "request"."domain_id"
WHERE "request"."request_id" = %s
  AND "request"."domain_id" = %s
  AND "request"."owner_id" = %s
  AND "request"."fence" = %s
  AND "request"."lease_expires_at" = %s
  AND "request"."lease_expires_at" > %s
  AND "request"."lifecycle" IN ('commit_started', 'settlement_pending')
  AND "domain"."current_fence" = "request"."fence"
"""

_SELECT_STEP_STATE_SQL = """
SELECT "state"
FROM "patch_durable_step_journal"
WHERE "request_id" = %s AND "step_id" = %s
ORDER BY "revision" DESC
LIMIT 1
"""

_ADVANCE_JOURNAL_SQL = """
UPDATE "patch_durable_requests"
SET "journal_revision" = %s, "updated_at" = CURRENT_TIMESTAMP
WHERE "request_id" = %s AND "journal_revision" = %s
RETURNING "journal_revision"
"""

_INSERT_STEP_JOURNAL_SQL = """
INSERT INTO "patch_durable_step_journal" (
    "request_id", "revision", "step_id", "lineage_id", "state"
) VALUES (%s, %s, %s, %s, %s)
"""

_SELECT_ARTIFACT_STATE_SQL = """
SELECT "state"
FROM "patch_durable_artifact_journal"
WHERE "request_id" = %s AND "artifact_id" = %s
ORDER BY "revision" DESC
LIMIT 1
"""

_INSERT_ARTIFACT_JOURNAL_SQL = """
INSERT INTO "patch_durable_artifact_journal" (
    "request_id", "revision", "artifact_id", "state"
) VALUES (%s, %s, %s, %s)
"""

_SELECT_STEP_JOURNAL_SQL = """
SELECT "revision", "step_id", "lineage_id", "state"
FROM "patch_durable_step_journal"
WHERE "request_id" = %s
ORDER BY "revision"
"""

_SELECT_ARTIFACT_JOURNAL_SQL = """
SELECT "revision", "artifact_id", "state"
FROM "patch_durable_artifact_journal"
WHERE "request_id" = %s
ORDER BY "revision"
"""

_SUSPEND_SQL = """
UPDATE "patch_durable_requests" AS "request"
SET "lifecycle" = 'settlement_pending',
    "pending_operation_id" = %s,
    "pending_correlation_id" = %s,
    "pending_next_check_after" = %s,
    "pending_event_cursor" = %s,
    "event_cursor" = %s,
    "updated_at" = CURRENT_TIMESTAMP
FROM "patch_durable_domains" AS "domain"
WHERE "request"."request_id" = %s
  AND "request"."owner_id" = %s
  AND "request"."fence" = %s
  AND "request"."lease_expires_at" = %s
  AND "request"."lease_expires_at" > %s
  AND "request"."lifecycle" = 'commit_started'
  AND "request"."pending_operation_id" IS NULL
  AND "domain"."domain_id" = "request"."domain_id"
  AND "domain"."current_fence" = "request"."fence"
RETURNING *
"""

_REQUEST_CANCELLATION_SQL = """
UPDATE "patch_durable_requests" AS "request"
SET "cancellation_requested" = TRUE, "updated_at" = CURRENT_TIMESTAMP
FROM "patch_durable_domains" AS "domain"
WHERE "request"."request_id" = %s
  AND "request"."lifecycle" IN ('commit_started', 'settlement_pending')
  AND "domain"."domain_id" = "request"."domain_id"
  AND "domain"."current_fence" = "request"."fence"
RETURNING *
"""

_SETTLE_SQL = """
UPDATE "patch_durable_requests" AS "request"
SET "terminal_result" = %s,
    "terminal_correlation_id" = %s,
    "terminal_pending_operation_id" = %s,
    "event_cursor" = %s,
    "pending_operation_id" = NULL,
    "pending_correlation_id" = NULL,
    "pending_next_check_after" = NULL,
    "pending_event_cursor" = NULL,
    "lifecycle" = 'request_completed',
    "updated_at" = CURRENT_TIMESTAMP
FROM "patch_durable_domains" AS "domain"
WHERE "request"."request_id" = %s
  AND "request"."owner_id" = %s
  AND "request"."fence" = %s
  AND "request"."lease_expires_at" = %s
  AND "request"."lease_expires_at" > %s
  AND "request"."journal_revision" = %s
  AND "request"."lifecycle" IN ('commit_started', 'settlement_pending')
  AND "domain"."domain_id" = "request"."domain_id"
  AND "domain"."current_fence" = "request"."fence"
RETURNING *
"""

_INSERT_OUTBOX_SQL = """
INSERT INTO "patch_durable_outbox" (
    "event_id", "request_id", "sequence", "lifecycle", "correlation_id"
) VALUES (%s, %s, %s, %s, %s)
RETURNING "event_id", "request_id", "sequence", "lifecycle", "correlation_id"
"""

_SELECT_OUTBOX_SQL = """
SELECT "event_id", "request_id", "sequence", "lifecycle", "correlation_id"
FROM "patch_durable_outbox"
WHERE "request_id" = %s AND "sequence" > %s
ORDER BY "sequence"
LIMIT %s
"""

_SELECT_TERMINAL_OUTBOX_SQL = """
SELECT "event_id", "request_id", "sequence", "lifecycle", "correlation_id"
FROM "patch_durable_outbox"
WHERE "request_id" = %s AND "lifecycle" = 'request_completed'
"""

_COUNT_RETENTION_SQL = """
SELECT COUNT(*)::BIGINT AS "record_count"
FROM "patch_durable_retention"
WHERE "request_id" = %s
"""

_SUM_RETENTION_SQL = """
SELECT COALESCE(SUM(OCTET_LENGTH("ciphertext")), 0)::BIGINT AS "byte_count"
FROM "patch_durable_retention"
WHERE "request_id" = %s
"""

_INSERT_RETENTION_SQL = """
INSERT INTO "patch_durable_retention" (
    "retention_id", "request_id", "kind", "key_id", "ciphertext",
    "ciphertext_digest", "expires_at", "delete_on_terminal"
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT ("retention_id") DO NOTHING
RETURNING "retention_id"
"""

_SELECT_RETENTION_SQL = """
SELECT "retention_id", "kind", "key_id", "ciphertext", "ciphertext_digest",
       "expires_at", "delete_on_terminal"
FROM "patch_durable_retention"
WHERE "retention_id" = %s AND "request_id" = %s
"""

_DELETE_RETENTION_SQL = """
DELETE FROM "patch_durable_retention"
WHERE "retention_id" = %s AND "request_id" = %s
"""

_DELETE_TERMINAL_RETENTION_SQL = """
DELETE FROM "patch_durable_retention"
WHERE "request_id" = %s AND "delete_on_terminal" = TRUE
"""

_DELETE_EXPIRED_RETENTION_SQL = """
WITH "deleted" AS (
    DELETE FROM "patch_durable_retention"
    WHERE "expires_at" <= %s
       OR (
           "delete_on_terminal" = TRUE
           AND EXISTS (
               SELECT 1 FROM "patch_durable_requests"
               WHERE "patch_durable_requests"."request_id"
                     = "patch_durable_retention"."request_id"
                 AND "lifecycle" = 'request_completed'
           )
       )
    RETURNING OCTET_LENGTH("ciphertext") AS "byte_count"
)
SELECT COUNT(*)::BIGINT AS "record_count",
       COALESCE(SUM("byte_count"), 0)::BIGINT AS "byte_count"
FROM "deleted"
"""
