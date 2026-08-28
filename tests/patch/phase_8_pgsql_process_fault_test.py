"""Kill isolated PostgreSQL workers after each Phase 8 durable barrier."""

from asyncio import create_task, gather, run, sleep, to_thread
from collections.abc import Callable, Iterator
from multiprocessing import get_context
from os import environ, umask
from pathlib import Path
from queue import Empty
from runpy import run_path
from typing import TypeGuard
from uuid import uuid4

import pytest
from phase_8_store_test import (
    _APPROVAL_AUTHORITY,
    _approval,
    _artifact,
    _correlation,
    _digest,
    _identity,
    _owner,
    _plan,
    _result,
)

from avalan.patch.coordinator import RetransmissionKey, _sealed_journal_steps
from avalan.patch.domain import (
    AlgorithmDigest,
    ArtifactState,
    CommitStepState,
    ContextKind,
    DurationTicks,
    ExpiryTick,
    LifecyclePhase,
    MutationState,
    PatchExecutionId,
    PatchPending,
    PatchPendingOperationId,
    SequenceNumber,
)
from avalan.patch.durable_approval import PhaseFiveDurableApprovalIssuer
from avalan.patch.durable_coordinator import (
    DurablePatchTestHost,
    DurablePatchTestHostProfile,
)
from avalan.patch.durable_store import (
    DurableArtifactState,
    DurableCommitClaimState,
    DurableJournal,
    DurableJournalCursor,
    DurablePendingAccess,
    DurablePendingRequest,
    DurablePlanReference,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableStepBinding,
    DurableStoreError,
    DurableStoreErrorCode,
)
from avalan.patch.local_commit import LocalCommitTarget
from avalan.patch.pgsql_store import (
    PgsqlDurablePatchStore,
    PgsqlDurablePatchStoreSettings,
)
from avalan.patch.policy import (
    PatchPrincipalId,
    PatchTenantId,
    PolicyRouteId,
)
from avalan.patch.target import LocalScopeResolver, ScopeSelection
from avalan.pgsql import (
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.task.store import TaskStoreNotFoundError
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
    task_pgsql_stamp,
    task_pgsql_upgrade,
)

_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")
_BARRIERS = (
    "reservation_commit",
    "plan_commit",
    "private_preparation",
    "commit_claim",
    "artifact_effect_before_journal",
    "artifact_present_journal",
    "requested_effect_before_journal",
    "requested_effect_planned_journal",
    "requested_effect_committed_journal",
    "verification",
    "cleanup_effect_before_journal",
    "cleanup_removed_journal",
    "pending",
    "terminal_settlement",
)

_ClaimRaceOutcome = tuple[str, str, int | None]
_RenewalRaceOutcome = tuple[str, str, int | str]
_JournalRaceOutcome = tuple[str, int | str]
_SettlementRaceOutcome = tuple[str, str, int]
_RaceTarget = (
    Callable[[str, str, str, object, object], None]
    | Callable[[str, str, object, object], None]
)


def _is_claim_race_outcome(value: object) -> TypeGuard[_ClaimRaceOutcome]:
    """Return whether a spawned claim worker returned its closed receipt."""
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(value[index], str) for index in (0, 1))
        and (type(value[2]) is int or value[2] is None)
    )


def _is_renewal_race_outcome(value: object) -> TypeGuard[_RenewalRaceOutcome]:
    """Return whether a spawned renewal worker returned its closed receipt."""
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(value[index], str) for index in (0, 1))
        and isinstance(value[2], (int, str))
    )


def _is_journal_race_outcome(value: object) -> TypeGuard[_JournalRaceOutcome]:
    """Return whether a spawned journal worker returned its closed receipt."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], (int, str))
    )


def _is_settlement_race_outcome(
    value: object,
) -> TypeGuard[_SettlementRaceOutcome]:
    """Return whether a spawned settlement worker returned its receipt."""
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(value[index], str) for index in (0, 1))
        and type(value[2]) is int
    )


def _claim_race_outcomes(
    values: tuple[object, ...],
) -> tuple[_ClaimRaceOutcome, ...]:
    """Validate and type the closed claim-race result vector."""
    if not all(_is_claim_race_outcome(value) for value in values):
        raise AssertionError("invalid spawned claim receipt")
    return tuple(value for value in values if _is_claim_race_outcome(value))


def _renewal_race_outcomes(
    values: tuple[object, ...],
) -> tuple[_RenewalRaceOutcome, ...]:
    """Validate and type the closed renewal-race result vector."""
    if not all(_is_renewal_race_outcome(value) for value in values):
        raise AssertionError("invalid spawned renewal receipt")
    return tuple(value for value in values if _is_renewal_race_outcome(value))


def _journal_race_outcomes(
    values: tuple[object, ...],
) -> tuple[_JournalRaceOutcome, ...]:
    """Validate and type the closed journal-race result vector."""
    if not all(_is_journal_race_outcome(value) for value in values):
        raise AssertionError("invalid spawned journal receipt")
    return tuple(value for value in values if _is_journal_race_outcome(value))


def _settlement_race_outcomes(
    values: tuple[object, ...],
) -> tuple[_SettlementRaceOutcome, ...]:
    """Validate and type the closed settlement-race result vector."""
    if not all(_is_settlement_race_outcome(value) for value in values):
        raise AssertionError("invalid spawned settlement receipt")
    return tuple(
        value for value in values if _is_settlement_race_outcome(value)
    )


def _real_local_identity() -> DurableRequestIdentity:
    """Reconstruct the fixed Phase 7 sealed-plan approval subject."""
    return DurableRequestIdentity(
        PatchTenantId("tenant-seven"),
        PatchPrincipalId("principal-seven"),
        PatchExecutionId("execution_" + "c" * 16),
        PolicyRouteId("route-seven"),
        RetransmissionKey("phase-eight-real-local"),
    )


pytestmark = pytest.mark.skipif(
    _DSN is None,
    reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
)


@pytest.fixture(autouse=True)
def _phase_8_file_creation_umask() -> Iterator[None]:
    """Create ordinary Phase 8 fixture files with their sealed 0644 mode."""
    previous = umask(0o022)
    try:
        yield
    finally:
        umask(previous)


async def _store(dsn: str, schema: str) -> PgsqlDurablePatchStore:
    """Open one fresh independent PostgreSQL durable-store client."""
    store = PgsqlDurablePatchStore.from_settings(
        PgsqlDurablePatchStoreSettings(
            dsn=dsn,
            schema=schema,
            pool_minimum=1,
            pool_maximum=2,
        ),
        approval_verifier=_APPROVAL_AUTHORITY,
    )
    await store.open()
    return store


async def _drop_schema(dsn: str, schema: str) -> None:
    """Drop one owned isolated durable-test schema after all pools close."""
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


async def _table_exists(dsn: str, schema: str, table: str) -> bool:
    """Return whether one exact isolated schema table remains visible."""
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_name = %s"
                    ") AS exists",
                    (schema, table),
                )
                row = await cursor.fetchone()
                assert row is not None
                return bool(row["exists"])


async def _execute(
    dsn: str,
    schema: str,
    statement: str,
    parameters: tuple[object, ...] = (),
) -> None:
    """Execute one fixed test-only SQL mutation in the isolated schema."""
    database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(dsn=dsn, schema=schema)
    )
    async with database:
        async with database.connection() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    await cursor.execute(statement, parameters)


async def _target_effect_count(dsn: str, schema: str) -> int:
    """Return the number of controlled target effects for one crash test."""
    database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(dsn=dsn, schema=schema)
    )
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    'SELECT COUNT(*) AS "effect_count" '
                    'FROM "patch_phase8_target_effects"'
                )
                row = await cursor.fetchone()
                assert row is not None
                count = row["effect_count"]
                if type(count) is not int:
                    raise AssertionError("target effect count is malformed")
                return count


async def _record_target_effect(
    dsn: str,
    schema: str,
    request_id: str,
) -> None:
    """Apply one controlled target side effect once per request."""
    await _execute(
        dsn,
        schema,
        'INSERT INTO "patch_phase8_target_effects" ("request_id") VALUES (%s)',
        (request_id,),
    )


async def _artifact_exists(dsn: str, schema: str, request_id: str) -> bool:
    """Return whether one controlled visible staging artifact remains."""
    database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(dsn=dsn, schema=schema)
    )
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT EXISTS (SELECT 1 FROM "
                    '"patch_phase8_target_artifacts" '
                    'WHERE "request_id" = %s) AS exists',
                    (request_id,),
                )
                row = await cursor.fetchone()
                assert row is not None
                return bool(row["exists"])


async def _create_artifact(dsn: str, schema: str, request_id: str) -> None:
    """Create one controlled durable staging artifact for recovery evidence."""
    await _execute(
        dsn,
        schema,
        'INSERT INTO "patch_phase8_target_artifacts" ("request_id") '
        "VALUES (%s)",
        (request_id,),
    )


async def _remove_artifact(dsn: str, schema: str, request_id: str) -> None:
    """Remove one controlled staging artifact during fenced reconciliation."""
    await _execute(
        dsn,
        schema,
        'DELETE FROM "patch_phase8_target_artifacts" WHERE "request_id" = %s',
        (request_id,),
    )


async def _pause_at_barrier(
    ready: object, release: object, barrier: str
) -> None:
    """Stop the child at one actual completed durable lifecycle boundary."""
    put = getattr(ready, "put", None)
    wait = getattr(release, "wait", None)
    if not callable(put) or not callable(wait):
        raise AssertionError("test process synchronization is unavailable")
    await to_thread(put, barrier)
    await to_thread(wait, 30)


def _child_blocked_at_barrier(
    dsn: str,
    schema: str,
    barrier: str,
    ready: object,
    release: object,
) -> None:
    """Block inside one real durable lifecycle boundary for process loss."""

    async def execute() -> None:
        store = await _store(dsn, schema)
        try:
            token = "a"
            identity = _identity(token)
            digest = _digest(token)
            reservation = await store.reserve(identity, digest)
            if barrier == "reservation_commit":
                await _pause_at_barrier(ready, release, barrier)
                return
            plan = _plan(digest, token, step_count=1)
            await store.persist_plan(reservation, plan)
            if barrier == "plan_commit":
                await _pause_at_barrier(ready, release, barrier)
                return
            artifact_id = _artifact(token)
            if barrier == "private_preparation":
                await _pause_at_barrier(ready, release, barrier)
                return
            claim = await store.claim_commit(
                reservation,
                plan,
                _approval(identity, digest, plan, token),
                _owner(token),
                ExpiryTick(10),
                DurationTicks(30),
                (artifact_id,),
            )
            assert claim.state is DurableCommitClaimState.OWNER
            assert claim.lease is not None
            if barrier == "commit_claim":
                await _pause_at_barrier(ready, release, barrier)
                return
            snapshot = await store.inspect(
                DurableRequestAccess(reservation.request_id, identity)
            )
            journal = snapshot.journal
            await _create_artifact(dsn, schema, reservation.request_id.value)
            if barrier == "artifact_effect_before_journal":
                await _pause_at_barrier(ready, release, barrier)
                return
            journal = await store.append_artifact(
                claim.lease,
                journal.cursor,
                artifact_id,
                DurableArtifactState.PRESENT,
                ExpiryTick(11),
            )
            if barrier == "artifact_present_journal":
                await _pause_at_barrier(ready, release, barrier)
                return
            await _record_target_effect(
                dsn, schema, reservation.request_id.value
            )
            if barrier == "requested_effect_before_journal":
                await _pause_at_barrier(ready, release, barrier)
                return
            journal = await store.append_step(
                claim.lease,
                journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(12),
            )
            if barrier == "requested_effect_planned_journal":
                await _pause_at_barrier(ready, release, barrier)
                return
            journal = await store.append_step(
                claim.lease,
                journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.COMMITTED,
                ExpiryTick(12),
            )
            if barrier == "requested_effect_committed_journal":
                await _pause_at_barrier(ready, release, barrier)
                return
            assert await _target_effect_count(dsn, schema) == 1
            assert await _artifact_exists(
                dsn, schema, reservation.request_id.value
            )
            if barrier == "verification":
                await _pause_at_barrier(ready, release, barrier)
                return
            await _remove_artifact(dsn, schema, reservation.request_id.value)
            if barrier == "cleanup_effect_before_journal":
                await _pause_at_barrier(ready, release, barrier)
                return
            journal = await store.append_artifact(
                claim.lease,
                journal.cursor,
                artifact_id,
                DurableArtifactState.REMOVED,
                ExpiryTick(13),
            )
            if barrier == "cleanup_removed_journal":
                await _pause_at_barrier(ready, release, barrier)
                return
            pending = DurablePendingRequest(
                PatchPendingOperationId("pending_" + token * 16),
                _correlation(token),
                DurationTicks(5),
            )
            await store.suspend(claim.lease, pending, ExpiryTick(14))
            if barrier == "pending":
                await _pause_at_barrier(ready, release, barrier)
                return
            terminal = await store.settle(
                claim.lease,
                journal.cursor,
                _result(
                    reservation.request_id,
                    plan,
                    MutationState.COMMITTED,
                    ArtifactState.CLEANED,
                ),
                pending.correlation_id,
                ExpiryTick(15),
            )
            assert (
                terminal.pending_operation_id == pending.pending_operation_id
            )
            if barrier == "terminal_settlement":
                await _pause_at_barrier(ready, release, barrier)
                return
            raise AssertionError(barrier)
        finally:
            await store.aclose()

    run(execute())


def _child_contend_claim_append_and_settle(
    dsn: str,
    schema: str,
    token: str,
    start: object,
    ready: object,
) -> None:
    """Race one fresh pool through claim, effect, journal, and settle."""

    async def execute() -> DurableCommitClaimState:
        store = await _store(dsn, schema)
        try:
            identity = _identity("b")
            digest = _digest("b")
            reservation = await store.reserve(identity, digest)
            plan = _plan(digest, "b", step_count=1)
            await store.persist_plan(reservation, plan)
            wait = getattr(start, "wait", None)
            if not callable(wait):
                raise AssertionError("test contention barrier is unavailable")
            await to_thread(wait, 20)
            claim = await store.claim_commit(
                reservation,
                plan,
                _approval(identity, digest, plan, "b"),
                _owner(token),
                ExpiryTick(10),
                DurationTicks(30),
                (),
            )
            if claim.state is DurableCommitClaimState.ATTACHED:
                return claim.state
            assert claim.state is DurableCommitClaimState.OWNER
            assert claim.lease is not None
            await _record_target_effect(
                dsn, schema, reservation.request_id.value
            )
            journal = await store.append_step(
                claim.lease,
                DurableJournalCursor(
                    reservation.request_id, SequenceNumber(0)
                ),
                plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(11),
            )
            journal = await store.append_step(
                claim.lease,
                journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.COMMITTED,
                ExpiryTick(11),
            )
            await store.settle(
                claim.lease,
                journal.cursor,
                _result(reservation.request_id, plan, MutationState.COMMITTED),
                _correlation("b"),
                ExpiryTick(12),
            )
            return claim.state
        finally:
            await store.aclose()

    put = getattr(ready, "put", None)
    if not callable(put):
        raise AssertionError("test process result queue is unavailable")
    try:
        put(("ok", run(execute()).value))
    except Exception as error:
        put(("error", type(error).__name__))


async def _recover_after_kill(
    dsn: str,
    schema: str,
    barrier: str,
) -> None:
    """Fence, reconcile, and settle without invoking a second target effect."""
    store = await _store(dsn, schema)
    try:
        identity = _identity("a")
        digest = _digest("a")
        replay = await store.reserve(identity, digest)
        assert replay.replayed
        access = DurableRequestAccess(replay.request_id, identity)
        snapshot = await store.inspect(access)
        expected_plan = _plan(digest, "a", step_count=1)
        continue_requested_effect = barrier in {
            "reservation_commit",
            "plan_commit",
        }
        initial_effects = (
            0
            if barrier
            in {
                "reservation_commit",
                "plan_commit",
                "private_preparation",
                "commit_claim",
                "artifact_effect_before_journal",
                "artifact_present_journal",
            }
            else 1
        )
        assert await _target_effect_count(dsn, schema) == initial_effects
        if barrier == "reservation_commit":
            assert snapshot.reservation.request_id == replay.request_id
            assert snapshot.reservation.identity == replay.identity
            assert (
                snapshot.reservation.canonical_digest
                == replay.canonical_digest
            )
            assert not snapshot.reservation.replayed
            assert snapshot.plan is None
            assert snapshot.lifecycle is LifecyclePhase.RECEIVED
            assert snapshot.lease is None
            assert snapshot.journal == DurableJournal(
                DurableJournalCursor(replay.request_id, SequenceNumber(0)),
                (),
                (),
            )
            assert snapshot.pending is None
            assert snapshot.terminal is None
            assert not snapshot.cancellation_requested
            assert snapshot.event_cursor == SequenceNumber(0)
            snapshot = await store.persist_plan(replay, expected_plan)
        if barrier in {"reservation_commit", "plan_commit"}:
            assert snapshot.reservation.request_id == replay.request_id
            assert snapshot.reservation.identity == replay.identity
            assert (
                snapshot.reservation.canonical_digest
                == replay.canonical_digest
            )
            assert not snapshot.reservation.replayed
            assert snapshot.plan == expected_plan
            assert snapshot.lifecycle is LifecyclePhase.PLANNED
            assert snapshot.lease is None
            assert snapshot.journal == DurableJournal(
                DurableJournalCursor(replay.request_id, SequenceNumber(0)),
                (),
                (),
            )
            assert snapshot.pending is None
            assert snapshot.terminal is None
            assert not snapshot.cancellation_requested
            assert snapshot.event_cursor == SequenceNumber(0)
        assert snapshot.plan is not None
        plan = snapshot.plan
        expected_effects = 1 if continue_requested_effect else initial_effects
        artifact_id = _artifact("a")
        if snapshot.terminal is not None:
            assert barrier == "terminal_settlement"
            assert await _target_effect_count(dsn, schema) == 1
            assert not await _artifact_exists(
                dsn, schema, replay.request_id.value
            )
            events = await store.outbox(access, SequenceNumber(0), 10)
            assert events[-1] == snapshot.terminal.outbox
            assert (
                sum(
                    event.lifecycle is LifecyclePhase.REQUEST_COMPLETED
                    for event in events
                )
                == 1
            )
            return
        if snapshot.lease is None:
            claim = await store.claim_commit(
                replay,
                plan,
                _approval(identity, digest, plan, "a"),
                _owner("b"),
                ExpiryTick(40),
                DurationTicks(30),
                (artifact_id,),
            )
            assert claim.state is DurableCommitClaimState.OWNER
            assert claim.lease is not None
            lease = claim.lease
        else:
            lease = await store.replace_expired_owner(
                replay,
                snapshot.lease,
                _owner("b"),
                ExpiryTick(40),
                DurationTicks(30),
            )
        snapshot = await store.inspect(access)
        journal = snapshot.journal
        states = {item.artifact_id: item.state for item in journal.artifacts}
        artifact_state = states.get(artifact_id)
        assert artifact_state is not None
        artifact_exists = await _artifact_exists(
            dsn, schema, replay.request_id.value
        )
        if artifact_state is DurableArtifactState.INTENDED:
            if artifact_exists:
                journal = await store.append_artifact(
                    lease,
                    journal.cursor,
                    artifact_id,
                    DurableArtifactState.PRESENT,
                    ExpiryTick(41),
                )
                await _remove_artifact(dsn, schema, replay.request_id.value)
                journal = await store.append_artifact(
                    lease,
                    journal.cursor,
                    artifact_id,
                    DurableArtifactState.REMOVED,
                    ExpiryTick(41),
                )
                terminal_artifact = ArtifactState.CLEANED
            else:
                journal = await store.append_artifact(
                    lease,
                    journal.cursor,
                    artifact_id,
                    DurableArtifactState.NOT_CREATED,
                    ExpiryTick(41),
                )
                terminal_artifact = ArtifactState.ABSENT
        elif artifact_state is DurableArtifactState.PRESENT:
            if artifact_exists:
                await _remove_artifact(dsn, schema, replay.request_id.value)
            journal = await store.append_artifact(
                lease,
                journal.cursor,
                artifact_id,
                DurableArtifactState.REMOVED,
                ExpiryTick(41),
            )
            terminal_artifact = ArtifactState.CLEANED
        elif artifact_state is DurableArtifactState.REMOVED:
            assert not artifact_exists
            terminal_artifact = ArtifactState.CLEANED
        else:
            raise AssertionError(artifact_state)
        step_state = next(
            (
                entry.state
                for entry in reversed(journal.steps)
                if entry.step_id == plan.steps[0].step_id
            ),
            None,
        )
        expected_step = (
            CommitStepState.COMMITTED
            if expected_effects == 1
            else CommitStepState.NOT_COMMITTED
        )
        if step_state is None:
            if continue_requested_effect:
                await _record_target_effect(
                    dsn, schema, replay.request_id.value
                )
            journal = await store.append_step(
                lease,
                journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(42),
            )
            step_state = CommitStepState.PLANNED
        if step_state is CommitStepState.PLANNED:
            journal = await store.append_step(
                lease,
                journal.cursor,
                plan.steps[0].step_id,
                expected_step,
                ExpiryTick(42),
            )
        else:
            assert step_state is expected_step
        pending = snapshot.pending
        correlation = (
            _correlation("a") if pending is None else pending.correlation_id
        )
        terminal = await store.settle(
            lease,
            journal.cursor,
            _result(
                replay.request_id,
                plan,
                (
                    MutationState.COMMITTED
                    if expected_effects == 1
                    else MutationState.NOT_COMMITTED
                ),
                terminal_artifact,
            ),
            correlation,
            ExpiryTick(43),
        )
        assert terminal.pending_operation_id == (
            None if pending is None else pending.pending_operation_id
        )
        assert await _target_effect_count(dsn, schema) == expected_effects
        assert not await _artifact_exists(dsn, schema, replay.request_id.value)
        settled = await store.inspect(access)
        assert settled.terminal is not None
        assert settled.terminal == terminal
        events = await store.outbox(access, SequenceNumber(0), 10)
        assert events[-1] == terminal.outbox
        assert (
            sum(
                event.lifecycle is LifecyclePhase.REQUEST_COMPLETED
                for event in events
            )
            == 1
        )
        final_replay = await store.reserve(identity, digest)
        assert final_replay.replayed
        assert final_replay.request_id == replay.request_id
        assert await _target_effect_count(dsn, schema) == expected_effects
    finally:
        await store.aclose()


def _child_reconcile_after_kill(
    dsn: str, schema: str, barrier: str, result: object
) -> None:
    """Run fresh-process fenced reconciliation and report only its outcome."""
    put = getattr(result, "put", None)
    if not callable(put):
        raise AssertionError("test process result queue is unavailable")
    try:
        run(_recover_after_kill(dsn, schema, barrier))
    except Exception as error:
        put(("error", type(error).__name__))
        return
    put(("ok", barrier))


def _child_commit_real_local_effect_then_pause(
    dsn: str,
    schema: str,
    root: str,
    ready: object,
    release: object,
) -> None:
    """Commit through the Phase 7 rooted worker after durable claim."""

    async def execute() -> None:
        helpers = run_path("tests/patch/phase_7_contract_test.py")
        workspace = Path(root)
        profile_factory = helpers["_profile"]
        if not callable(profile_factory):
            raise AssertionError("Phase 7 local profile is unavailable")
        profile = profile_factory(workspace)
        scope = await LocalScopeResolver(profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        target = LocalCommitTarget(profile)
        sealed = await helpers["_sealed"](
            profile,
            target,
            scope,
            "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: note.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            ),
            {"note.txt": b"before\n"},
        )
        plan = DurablePlanReference(
            sealed.plan_id,
            sealed.binding.request_digest,
            sealed.fingerprint.digest(),
            sealed.review.diff.digest,
            sealed.binding.target.context_id,
            sealed.binding.target.workspace_id,
            sealed.binding.target.domain_id,
            tuple(
                DurableStepBinding(step_id, lineage_id)
                for step_id, lineage_id in _sealed_journal_steps(sealed)
            ),
        )
        phase_six = helpers["_PHASE6"]
        approvals = helpers["ApprovalService"](
            phase_six["_Broker"](),
            phase_six["_Clock"](),
            phase_six["RuntimeGrantStore"](),
        )
        grant = await phase_six["_issue_grant"](sealed, approvals)
        identity = DurableRequestIdentity(
            sealed.binding.subject.tenant,
            sealed.binding.subject.principal,
            sealed.binding.request.execution_id,
            sealed.binding.final.approval.route,
            RetransmissionKey("phase-eight-real-local"),
        )
        approval = await PhaseFiveDurableApprovalIssuer(
            approvals, _APPROVAL_AUTHORITY
        ).issue(identity, plan, grant, sealed, sealed.binding.subject)
        store = await _store(dsn, schema)
        try:
            reservation = await store.reserve(
                identity, sealed.binding.request_digest
            )
            await store.persist_plan(reservation, plan)
            claim = await store.claim_commit(
                reservation,
                plan,
                approval,
                _owner("f"),
                ExpiryTick(10),
                DurationTicks(30),
                (),
            )
            assert claim.state is DurableCommitClaimState.OWNER
            assert claim.lease is not None
            assert (workspace / "note.txt").read_bytes() == b"before\n"
            coordinator_store = helpers["InMemoryCoordinatorStore"](approvals)
            coordinator = helpers["InMemoryPatchCoordinator"](
                coordinator_store,
                helpers["InMemoryLeaseManager"](coordinator_store),
                helpers["ScriptedReconciler"](phase_six["_snapshot"]()),
            )
            coordinator_reservation = await coordinator.reserve(
                helpers["RuntimeIdentity"](
                    sealed.binding.subject,
                    sealed.binding.final.approval.route,
                    RetransmissionKey("phase-eight-real-local"),
                ),
                sealed.binding.request_digest,
            )
            await coordinator.execute(
                coordinator_reservation,
                sealed,
                grant,
                phase_six["_snapshot"](),
                await target.worker(scope),
                "phase-eight-real-local-owner",
            )
            coordinator_record = await coordinator_store.record(
                coordinator_reservation
            )
            assert coordinator_record.journal is not None
            assert tuple(
                (item.identifier, item.lineage)
                for item in coordinator_record.journal.steps
            ) == tuple((item.step_id, item.lineage_id) for item in plan.steps)
            assert (workspace / "note.txt").read_bytes() == b"after\n"
            put = getattr(ready, "put", None)
            wait = getattr(release, "wait", None)
            if not callable(put) or not callable(wait):
                raise AssertionError(
                    "test process synchronization is unavailable"
                )
            await to_thread(
                put,
                (
                    "effect",
                    reservation.request_id.value,
                    sealed.binding.request_digest.value,
                ),
            )
            await to_thread(wait, 30)
        finally:
            await store.aclose()

    put = getattr(ready, "put", None)
    if not callable(put):
        raise AssertionError("test process result queue is unavailable")
    try:
        run(execute())
    except Exception as error:
        put(("error", type(error).__name__))


def _child_reconcile_real_local_effect(
    dsn: str,
    schema: str,
    root: str,
    digest: str,
    result: object,
) -> None:
    """Settle an observed real local effect without a target worker."""

    async def execute() -> str:
        store = await _store(dsn, schema)
        try:
            identity = _real_local_identity()
            replay = await store.reserve(
                identity, AlgorithmDigest("sha256", digest)
            )
            assert replay.replayed
            access = DurableRequestAccess(replay.request_id, identity)
            snapshot = await store.inspect(access)
            assert snapshot.lifecycle is LifecyclePhase.COMMIT_STARTED
            assert snapshot.lease is not None
            assert snapshot.plan is not None
            assert snapshot.terminal is None
            assert (Path(root) / "note.txt").read_bytes() == b"after\n"
            plan = snapshot.plan
            lease = snapshot.lease
            journal = await store.append_step(
                lease,
                snapshot.journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(11),
            )
            journal = await store.append_step(
                lease,
                journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.COMMITTED,
                ExpiryTick(11),
            )
            terminal = await store.settle(
                lease,
                journal.cursor,
                _result(
                    replay.request_id,
                    plan,
                    MutationState.COMMITTED,
                    ArtifactState.ABSENT,
                ),
                _correlation("f"),
                ExpiryTick(12),
            )
            events = await store.outbox(access, SequenceNumber(0), 10)
            assert events == (terminal.outbox,)
            assert (Path(root) / "note.txt").read_bytes() == b"after\n"
            return replay.request_id.value
        finally:
            await store.aclose()

    put = getattr(result, "put", None)
    if not callable(put):
        raise AssertionError("test process result queue is unavailable")
    try:
        put(("ok", run(execute())))
    except Exception as error:
        put(("error", type(error).__name__))


def _child_race_claim(
    dsn: str,
    schema: str,
    owner_token: str,
    start: object,
    result: object,
) -> None:
    """Race the exact approval consumption and initial fence assignment."""

    async def execute() -> tuple[str, int | None]:
        store = await _store(dsn, schema)
        try:
            identity = _identity("e")
            digest = _digest("e")
            reservation = await store.reserve(identity, digest)
            snapshot = await store.inspect(
                DurableRequestAccess(reservation.request_id, identity)
            )
            assert snapshot.plan is not None
            wait = getattr(start, "wait", None)
            if not callable(wait):
                raise AssertionError("test contention barrier is unavailable")
            await to_thread(wait, 30)
            claim = await store.claim_commit(
                reservation,
                snapshot.plan,
                _approval(identity, digest, snapshot.plan, "e"),
                _owner(owner_token),
                ExpiryTick(10),
                DurationTicks(10),
                (),
            )
            return (
                claim.state.value,
                None if claim.lease is None else claim.lease.fence.value,
            )
        finally:
            await store.aclose()

    put = getattr(result, "put", None)
    if not callable(put):
        raise AssertionError("test process result queue is unavailable")
    try:
        put(("ok", *run(execute())))
    except DurableStoreError as error:
        put(("error", error.code.value, None))
    except Exception as error:
        put(("error", type(error).__name__, None))


def _child_race_renew_or_expire(
    dsn: str,
    schema: str,
    operation: str,
    start: object,
    result: object,
) -> None:
    """Race renewal against expiry replacement through separate SQL pools."""

    async def execute() -> tuple[str, int]:
        store = await _store(dsn, schema)
        try:
            identity = _identity("e")
            replay = await store.reserve(identity, _digest("e"))
            snapshot = await store.inspect(
                DurableRequestAccess(replay.request_id, identity)
            )
            assert snapshot.lease is not None
            wait = getattr(start, "wait", None)
            if not callable(wait):
                raise AssertionError("test contention barrier is unavailable")
            await to_thread(wait, 30)
            match operation:
                case "renew":
                    lease = await store.renew_lease(
                        snapshot.lease,
                        ExpiryTick(19),
                        DurationTicks(10),
                    )
                case "expire":
                    lease = await store.replace_expired_owner(
                        replay,
                        snapshot.lease,
                        _owner("f"),
                        ExpiryTick(20),
                        DurationTicks(10),
                    )
                case _:
                    raise AssertionError(operation)
            return operation, lease.fence.value
        finally:
            await store.aclose()

    put = getattr(result, "put", None)
    if not callable(put):
        raise AssertionError("test process result queue is unavailable")
    try:
        put(("ok", *run(execute())))
    except DurableStoreError as error:
        put(("error", operation, error.code.value))
    except Exception as error:
        put(("error", operation, type(error).__name__))


def _child_race_journal_append(
    dsn: str,
    schema: str,
    start: object,
    result: object,
) -> None:
    """Race one journal CAS append through independently opened pools."""

    async def execute() -> int:
        store = await _store(dsn, schema)
        try:
            identity = _identity("e")
            replay = await store.reserve(identity, _digest("e"))
            snapshot = await store.inspect(
                DurableRequestAccess(replay.request_id, identity)
            )
            assert snapshot.lease is not None
            assert snapshot.plan is not None
            wait = getattr(start, "wait", None)
            if not callable(wait):
                raise AssertionError("test contention barrier is unavailable")
            await to_thread(wait, 30)
            journal = await store.append_step(
                snapshot.lease,
                snapshot.journal.cursor,
                snapshot.plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(21),
            )
            return journal.cursor.revision.value
        finally:
            await store.aclose()

    put = getattr(result, "put", None)
    if not callable(put):
        raise AssertionError("test process result queue is unavailable")
    try:
        put(("ok", run(execute())))
    except DurableStoreError as error:
        put(("error", error.code.value))
    except Exception as error:
        put(("error", type(error).__name__))


def _child_race_terminal_settlement(
    dsn: str,
    schema: str,
    start: object,
    result: object,
) -> None:
    """Race terminal settlement and its uniquely keyed outbox publication."""

    async def execute() -> tuple[str, int]:
        store = await _store(dsn, schema)
        try:
            identity = _identity("e")
            replay = await store.reserve(identity, _digest("e"))
            snapshot = await store.inspect(
                DurableRequestAccess(replay.request_id, identity)
            )
            assert snapshot.lease is not None
            assert snapshot.plan is not None
            wait = getattr(start, "wait", None)
            if not callable(wait):
                raise AssertionError("test contention barrier is unavailable")
            await to_thread(wait, 30)
            terminal = await store.settle(
                snapshot.lease,
                snapshot.journal.cursor,
                _result(
                    replay.request_id,
                    snapshot.plan,
                    MutationState.COMMITTED,
                ),
                _correlation("e"),
                ExpiryTick(22),
            )
            return (
                terminal.outbox.event_id.value,
                terminal.outbox.sequence.value,
            )
        finally:
            await store.aclose()

    put = getattr(result, "put", None)
    if not callable(put):
        raise AssertionError("test process result queue is unavailable")
    try:
        put(("ok", *run(execute())))
    except DurableStoreError as error:
        put(("error", error.code.value, 0))
    except Exception as error:
        put(("error", type(error).__name__, 0))


def test_pgsql_pending_restart_authenticates_original_branch() -> None:
    """Replay one PostgreSQL pending branch through authenticated await."""
    assert _DSN is not None
    assert _DSN.startswith("postgresql")

    async def scenario() -> None:
        schema = "patch_phase8_pending_" + uuid4().hex
        await to_thread(
            task_pgsql_upgrade,
            PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
        )
        first = await _store(_DSN, schema)
        try:
            identity = _identity("c")
            digest = _digest("c")
            reservation = await first.reserve(identity, digest)
            plan = _plan(digest, "c", step_count=1)
            await first.persist_plan(reservation, plan)
            claim = await first.claim_commit(
                reservation,
                plan,
                _approval(identity, digest, plan, "c"),
                _owner("c"),
                ExpiryTick(10),
                DurationTicks(30),
                (),
            )
            assert claim.state is DurableCommitClaimState.OWNER
            assert claim.lease is not None
            journal = await first.append_step(
                claim.lease,
                DurableJournalCursor(
                    reservation.request_id, SequenceNumber(0)
                ),
                plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(11),
            )
            journal = await first.append_step(
                claim.lease,
                journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.COMMITTED,
                ExpiryTick(11),
            )
            pending = DurablePendingRequest(
                PatchPendingOperationId("pending_" + "c" * 16),
                _correlation("c"),
                DurationTicks(5),
            )
            await first.suspend(claim.lease, pending, ExpiryTick(12))
        finally:
            await first.aclose()

        fresh = await _store(_DSN, schema)
        try:
            replay = await fresh.reserve(identity, digest)
            assert replay.replayed
            assert replay.request_id == reservation.request_id
            access = DurablePendingAccess(
                DurableRequestAccess(replay.request_id, identity),
                pending.pending_operation_id,
                pending.correlation_id,
            )
            host = DurablePatchTestHost(
                fresh, DurablePatchTestHostProfile(True, True)
            )
            projected = await host.inspect(access)
            assert isinstance(projected, PatchPending)
            assert projected.request_id == replay.request_id
            assert projected.correlation_id == pending.correlation_id
            with pytest.raises(DurableStoreError) as denied:
                await host.inspect(
                    DurablePendingAccess(
                        DurableRequestAccess(
                            replay.request_id, _identity("d")
                        ),
                        pending.pending_operation_id,
                        pending.correlation_id,
                    )
                )
            assert denied.value.code is DurableStoreErrorCode.ACCESS_DENIED
            awaiting = create_task(host.await_resume(access))
            await sleep(0)
            result = _result(
                replay.request_id,
                plan,
                MutationState.COMMITTED,
                ArtifactState.ABSENT,
            )
            terminal = await fresh.settle(
                claim.lease,
                journal.cursor,
                result,
                pending.correlation_id,
                ExpiryTick(13),
            )
            assert terminal.result == result
            assert await awaiting == result
            assert await host.inspect(access) == result
            assert await host.resume(access) == result
        finally:
            await fresh.aclose()
            await _drop_schema(_DSN, schema)

    run(scenario())


@pytest.mark.parametrize("barrier", _BARRIERS)
def test_killed_pgsql_child_reconciles_each_durable_boundary(
    barrier: str,
) -> None:
    """Kill blocked workers and settle each target-aware durable boundary."""
    assert _DSN is not None

    async def scenario() -> None:
        schema = "patch_phase8_kill_" + uuid4().hex
        await to_thread(
            task_pgsql_upgrade,
            PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
        )
        await _execute(
            _DSN,
            schema,
            'CREATE TABLE "patch_phase8_target_effects" ('
            '"request_id" TEXT PRIMARY KEY)',
        )
        await _execute(
            _DSN,
            schema,
            'CREATE TABLE "patch_phase8_target_artifacts" ('
            '"request_id" TEXT PRIMARY KEY)',
        )
        context = get_context("spawn")
        ready = context.Queue()
        release = context.Event()
        process = context.Process(
            target=_child_blocked_at_barrier,
            args=(_DSN, schema, barrier, ready, release),
        )
        process.start()
        try:
            try:
                observed = ready.get(timeout=20)
            except Empty as error:
                raise AssertionError(
                    "child failed before its durable barrier"
                ) from error
            assert observed == barrier
            process.terminate()
            process.join(20)
            assert process.exitcode is not None
            assert process.exitcode != 0
            recovered = context.Queue()
            recovery = context.Process(
                target=_child_reconcile_after_kill,
                args=(_DSN, schema, barrier, recovered),
            )
            recovery.start()
            assert recovered.get(timeout=30) == ("ok", barrier)
            recovery.join(30)
            assert recovery.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(20)
            await _drop_schema(_DSN, schema)

    run(scenario())


def test_patch_e2e_006_real_local_commit_survives_durable_process_crash(
    tmp_path: Path,
) -> None:
    """Reconcile one approved rooted local effect after commit-start crash."""
    assert _DSN is not None

    async def scenario() -> None:
        schema = "patch_phase8_real_local_" + uuid4().hex
        await to_thread(
            task_pgsql_upgrade,
            PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
        )
        workspace = tmp_path / "workspace"
        workspace.mkdir(mode=0o700)
        target_file = workspace / "note.txt"
        target_file.write_bytes(b"before\n")
        context = get_context("spawn")
        ready = context.Queue()
        release = context.Event()
        worker = context.Process(
            target=_child_commit_real_local_effect_then_pause,
            args=(_DSN, schema, str(workspace), ready, release),
        )
        recovery: object | None = None
        try:
            worker.start()
            try:
                observed = ready.get(timeout=45)
            except Empty as error:
                raise AssertionError(
                    "real local worker failed before the crash boundary"
                ) from error
            assert observed[0] == "effect"
            request_id = observed[1]
            assert type(request_id) is str
            digest = observed[2]
            assert type(digest) is str
            assert target_file.read_bytes() == b"after\n"
            worker.terminate()
            worker.join(30)
            assert worker.exitcode is not None
            assert worker.exitcode != 0

            results = context.Queue()
            recovery = context.Process(
                target=_child_reconcile_real_local_effect,
                args=(_DSN, schema, str(workspace), digest, results),
            )
            recovery.start()
            assert results.get(timeout=45) == ("ok", request_id)
            recovery.join(45)
            assert recovery.exitcode == 0

            store = await _store(_DSN, schema)
            try:
                identity = _real_local_identity()
                replay = await store.reserve(
                    identity, AlgorithmDigest("sha256", digest)
                )
                assert replay.replayed
                assert replay.request_id.value == request_id
                snapshot = await store.inspect(
                    DurableRequestAccess(replay.request_id, identity)
                )
                assert snapshot.terminal is not None
                events = await store.outbox(
                    DurableRequestAccess(replay.request_id, identity),
                    SequenceNumber(0),
                    10,
                )
                assert events == (snapshot.terminal.outbox,)
                assert target_file.read_bytes() == b"after\n"
                assert tuple(workspace.iterdir()) == (target_file,)
            finally:
                await store.aclose()
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(30)
            if (
                recovery is not None
                and getattr(recovery, "is_alive", lambda: False)()
            ):
                getattr(recovery, "terminate")()
                getattr(recovery, "join")(30)
            await _drop_schema(_DSN, schema)

    run(scenario())


def test_pgsql_cross_process_approval_fence_journal_and_settlement_races() -> (
    None
):
    """Require one legal PostgreSQL winner across independent store clients."""
    assert _DSN is not None

    async def scenario() -> None:
        schema = "patch_phase8_race_" + uuid4().hex
        await to_thread(
            task_pgsql_upgrade,
            PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
        )
        first = await _store(_DSN, schema)
        second = await _store(_DSN, schema)
        try:
            identity = _identity("b")
            digest = _digest("b")
            reservation = await first.reserve(identity, digest)
            plan = _plan(digest, "b", step_count=1)
            await first.persist_plan(reservation, plan)
            first_claim = await first.claim_commit(
                reservation,
                plan,
                _approval(identity, digest, plan, "b"),
                _owner("b"),
                ExpiryTick(10),
                DurationTicks(10),
                (),
            )
            assert first_claim.lease is not None
            with pytest.raises(DurableStoreError) as premature:
                await second.replace_expired_owner(
                    reservation,
                    first_claim.lease,
                    _owner("c"),
                    ExpiryTick(19),
                    DurationTicks(10),
                )
            assert premature.value.code is DurableStoreErrorCode.LEASE_EXPIRED
            replacement = await second.replace_expired_owner(
                reservation,
                first_claim.lease,
                _owner("c"),
                ExpiryTick(20),
                DurationTicks(10),
            )
            assert replacement.fence.value == first_claim.lease.fence.value + 1
            with pytest.raises(DurableStoreError) as fenced:
                await first.append_step(
                    first_claim.lease,
                    DurableJournalCursor(
                        reservation.request_id, SequenceNumber(0)
                    ),
                    plan.steps[0].step_id,
                    CommitStepState.PLANNED,
                    ExpiryTick(21),
                )
            assert fenced.value.code is DurableStoreErrorCode.FENCED
        finally:
            await first.aclose()
            await second.aclose()
            await _drop_schema(_DSN, schema)

    run(scenario())


def test_spawned_pools_race_every_durable_ownership_boundary() -> None:
    """Prove one legal state transition at every fenced PostgreSQL race."""
    assert _DSN is not None

    async def scenario() -> None:
        schema = "patch_phase8_all_races_" + uuid4().hex
        await to_thread(
            task_pgsql_upgrade,
            PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
        )
        context = get_context("spawn")
        processes: list[object] = []

        def run_race(
            target: _RaceTarget,
            arguments: tuple[tuple[object, ...], ...],
        ) -> tuple[object, ...]:
            """Start exact independent pools at one shared process barrier."""
            start = context.Barrier(len(arguments))
            results = context.Queue()
            started = tuple(
                context.Process(
                    target=target,
                    args=arguments[index] + (start, results),
                )
                for index in range(len(arguments))
            )
            processes.extend(started)
            for process in started:
                process.start()
            outcomes = tuple(results.get(timeout=45) for _ in started)
            for process in started:
                process.join(45)
                assert process.exitcode == 0
            return outcomes

        try:
            first = await _store(_DSN, schema)
            try:
                identity = _identity("e")
                digest = _digest("e")
                reservation = await first.reserve(identity, digest)
                await first.persist_plan(
                    reservation, _plan(digest, "e", step_count=1)
                )
            finally:
                await first.aclose()

            claims = _claim_race_outcomes(
                run_race(
                    _child_race_claim,
                    (
                        (_DSN, schema, "e"),
                        (_DSN, schema, "f"),
                    ),
                )
            )
            assert set(claims) == {
                ("ok", DurableCommitClaimState.ATTACHED.value, None),
                ("ok", DurableCommitClaimState.OWNER.value, 1),
            }

            renewal = _renewal_race_outcomes(
                run_race(
                    _child_race_renew_or_expire,
                    (
                        (_DSN, schema, "renew"),
                        (_DSN, schema, "expire"),
                    ),
                )
            )
            successful_renewal = tuple(
                outcome for outcome in renewal if outcome[0] == "ok"
            )
            failed_renewal = tuple(
                outcome for outcome in renewal if outcome[0] == "error"
            )
            assert len(successful_renewal) == 1
            assert len(failed_renewal) == 1
            assert failed_renewal[0][-1] in {
                DurableStoreErrorCode.FENCED.value,
                DurableStoreErrorCode.LEASE_EXPIRED.value,
            }

            journals = _journal_race_outcomes(
                run_race(
                    _child_race_journal_append,
                    ((_DSN, schema), (_DSN, schema)),
                )
            )
            assert set(journals) == {
                ("error", DurableStoreErrorCode.JOURNAL_CONFLICT.value),
                ("ok", 1),
            }

            current = await _store(_DSN, schema)
            try:
                replay = await current.reserve(_identity("e"), _digest("e"))
                snapshot = await current.inspect(
                    DurableRequestAccess(replay.request_id, _identity("e"))
                )
                assert snapshot.lease is not None
                assert snapshot.plan is not None
                journal = await current.append_step(
                    snapshot.lease,
                    snapshot.journal.cursor,
                    snapshot.plan.steps[0].step_id,
                    CommitStepState.COMMITTED,
                    ExpiryTick(21),
                )
                assert journal.cursor.revision == SequenceNumber(2)
            finally:
                await current.aclose()

            settlements = _settlement_race_outcomes(
                run_race(
                    _child_race_terminal_settlement,
                    ((_DSN, schema), (_DSN, schema)),
                )
            )
            assert len(settlements) == 2
            assert all(outcome[0] == "ok" for outcome in settlements)
            assert settlements[0][1:] == settlements[1][1:]

            final = await _store(_DSN, schema)
            try:
                replay = await final.reserve(_identity("e"), _digest("e"))
                access = DurableRequestAccess(
                    replay.request_id, _identity("e")
                )
                snapshot = await final.inspect(access)
                assert snapshot.terminal is not None
                events = await final.outbox(access, SequenceNumber(0), 10)
                assert events == (snapshot.terminal.outbox,)
                assert events[0].event_id.value == settlements[0][1]
                assert events[0].sequence.value == settlements[0][2]
            finally:
                await final.aclose()
        finally:
            for process in processes:
                if getattr(process, "is_alive", lambda: False)():
                    getattr(process, "terminate")()
                    getattr(process, "join")(30)
            await _drop_schema(_DSN, schema)

    run(scenario())


def test_spawned_pools_have_one_claim_effect_and_terminal_publication() -> (
    None
):
    """Coordinate spawned pools and preserve one target effect and terminal."""
    assert _DSN is not None

    async def scenario() -> None:
        schema = "patch_phase8_spawn_" + uuid4().hex
        await to_thread(
            task_pgsql_upgrade,
            PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
        )
        await _execute(
            _DSN,
            schema,
            'CREATE TABLE "patch_phase8_target_effects" ('
            '"request_id" TEXT PRIMARY KEY)',
        )
        context = get_context("spawn")
        start = context.Barrier(2)
        ready = context.Queue()
        processes = tuple(
            context.Process(
                target=_child_contend_claim_append_and_settle,
                args=(_DSN, schema, token, start, ready),
            )
            for token in ("b", "c")
        )
        for process in processes:
            process.start()
        try:
            outcomes = tuple(ready.get(timeout=30) for _ in processes)
            for process in processes:
                process.join(30)
                assert process.exitcode == 0
            assert tuple(sorted(outcomes)) == (
                ("ok", DurableCommitClaimState.ATTACHED.value),
                ("ok", DurableCommitClaimState.OWNER.value),
            )
            assert await _target_effect_count(_DSN, schema) == 1
            store = await _store(_DSN, schema)
            try:
                identity = _identity("b")
                replay = await store.reserve(identity, _digest("b"))
                snapshot = await store.inspect(
                    DurableRequestAccess(replay.request_id, identity)
                )
                assert snapshot.terminal is not None
                events = await store.outbox(
                    DurableRequestAccess(replay.request_id, identity),
                    SequenceNumber(0),
                    10,
                )
                assert (
                    sum(
                        event.lifecycle is LifecyclePhase.REQUEST_COMPLETED
                        for event in events
                    )
                    == 1
                )
            finally:
                await store.aclose()
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(20)
            await _drop_schema(_DSN, schema)

    run(scenario())


def test_pgsql_prior_migration_rollback_and_unknown_records() -> None:
    """Migrate from the prior head, roll back failure, and fail closed."""
    assert _DSN is not None

    async def scenario() -> None:
        schema = "patch_phase8_migration_" + uuid4().hex
        settings = PgsqlTaskMigrationSettings(url=_DSN, schema=schema)
        try:
            await to_thread(
                task_pgsql_upgrade,
                settings,
                revision="20260801_0003",
            )
            assert not await _table_exists(
                _DSN, schema, "patch_durable_requests"
            )
            await _execute(
                _DSN,
                schema,
                'CREATE TABLE "patch_durable_retention" ("id" INT)',
            )
            with pytest.raises(Exception):
                await to_thread(task_pgsql_upgrade, settings)
            assert not await _table_exists(
                _DSN, schema, "patch_durable_requests"
            )
            assert await _table_exists(_DSN, schema, "patch_durable_retention")
            await _execute(
                _DSN,
                schema,
                'DROP TABLE "patch_durable_retention"',
            )
            await to_thread(task_pgsql_upgrade, settings)
            assert await _table_exists(_DSN, schema, "patch_durable_requests")
            await to_thread(task_pgsql_stamp, settings)

            first = await _store(_DSN, schema)
            second = await _store(_DSN, schema)
            try:
                identity = _identity("c")
                digest = _digest("c")
                reservation = await first.reserve(identity, digest)
                plan = _plan(digest, "c", step_count=1)
                await first.persist_plan(reservation, plan)
                assert (
                    await second.inspect(
                        DurableRequestAccess(reservation.request_id, identity)
                    )
                ).plan == plan
                await _execute(
                    _DSN,
                    schema,
                    'UPDATE "patch_durable_requests" '
                    'SET "plan_payload" = %s '
                    'WHERE "request_id" = %s',
                    (b"patch-durable-plan-v999", reservation.request_id.value),
                )
                with pytest.raises(DurableStoreError) as unknown:
                    await second.inspect(
                        DurableRequestAccess(reservation.request_id, identity)
                    )
                assert (
                    unknown.value.code is DurableStoreErrorCode.PLAN_MISMATCH
                )
            finally:
                await first.aclose()
                await second.aclose()
        finally:
            await _drop_schema(_DSN, schema)

    run(scenario())


def test_pgsql_rolling_task_client_and_durable_client_compatibility() -> None:
    """Keep the actual prior task-store client live during durable upgrade."""
    assert _DSN is not None

    async def legacy_read(client: PgsqlTaskStore) -> None:
        """Exercise the pre-Phase-8 task-store read contract unchanged."""
        with pytest.raises(TaskStoreNotFoundError):
            await client.get_run("prior-client-missing-run")

    async def scenario() -> None:
        schema = "patch_phase8_rolling_" + uuid4().hex
        settings = PgsqlTaskMigrationSettings(url=_DSN, schema=schema)
        legacy_database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=_DSN,
                schema=schema,
                pool_minimum=1,
                pool_maximum=1,
            )
        )
        legacy = PgsqlTaskStore(legacy_database)
        durable: PgsqlDurablePatchStore | None = None
        try:
            await to_thread(
                task_pgsql_upgrade,
                settings,
                revision="20260801_0003",
            )
            await legacy.open()
            await legacy_read(legacy)

            await to_thread(task_pgsql_upgrade, settings)
            durable = await _store(_DSN, schema)
            identity = _identity("d")
            reservation, _ = await gather(
                durable.reserve(identity, _digest("d")),
                legacy_read(legacy),
            )
            plan = _plan(_digest("d"), "d", step_count=1)
            snapshot, _ = await gather(
                durable.persist_plan(reservation, plan),
                legacy_read(legacy),
            )
            assert snapshot.plan == plan
            assert (
                await durable.inspect(
                    DurableRequestAccess(reservation.request_id, identity)
                )
            ).plan == plan
        finally:
            if durable is not None:
                await durable.aclose()
            await legacy.aclose()
            await _drop_schema(_DSN, schema)

    run(scenario())
