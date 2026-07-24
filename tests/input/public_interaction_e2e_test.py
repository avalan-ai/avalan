"""Exercise public durable interaction behavior."""

from asyncio import gather, run
from datetime import timedelta
from pathlib import Path
from sys import path as sys_path

sys_path.append(str(Path(__file__).parents[1] / "interaction" / "stores"))

import interaction_pgsql_store_test as durable_support  # noqa: E402
from pgsql_support import FakePgsqlDatabase  # noqa: E402

from avalan.interaction import (  # noqa: E402
    ContinuationClaimOwnerId,
    ContinuationDispatchId,
    InputErrorCode,
    ProviderIdempotencyKey,
    ResolveInteractionRejected,
)
from avalan.interaction.stores.pgsql import (  # noqa: E402
    ContinuationStoreConflictError,
    PgsqlDurableTaskCoordinator,
    PgsqlInteractionStoreError,
)
from avalan.task import TaskInteractionEventType  # noqa: E402
from avalan.task.stores import PgsqlTaskStore  # noqa: E402


def test_idempotency_and_staleness() -> None:
    """Prove replay, conflict, expiry, and duplicate-claim behavior."""

    async def exercise() -> tuple[object, ...]:
        database = FakePgsqlDatabase()
        durable_support._seed_running_task(database, "run")
        interaction_store = await durable_support._store(database)
        identifiers = durable_support._Ids()
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: durable_support._NOW,
            id_factory=lambda: identifiers.next("task"),
        )
        coordinator = PgsqlDurableTaskCoordinator(
            interaction_store,
            task_store,
        )
        request = durable_support._request()
        staged = await coordinator.create_and_suspend(
            durable_support._create_command(request),
            durable_support._portable(request),
            queue_item_id="queue-item",
            claim_token="claim-token",
            segment_id="segment",
            task_run_id="run",
            checkpoint_id="checkpoint",
        )
        command = durable_support._answer(staged.interaction)
        first = await coordinator.resolve_and_requeue(
            command,
            task_run_id="run",
        )
        replay = await coordinator.resolve_and_requeue(
            command,
            task_run_id="run",
        )
        conflict = await interaction_store.resolve(
            durable_support._answer(
                staged.interaction,
                value=False,
            )
        )
        assert isinstance(conflict, ResolveInteractionRejected)
        assert conflict.error.code is InputErrorCode.IDEMPOTENCY_CONFLICT
        assert not conflict.store_mutation_applied

        ready = await interaction_store.get_continuation(
            request.continuation_id
        )
        claims = await gather(
            interaction_store.claim(
                request.continuation_id,
                expected_store_revision=ready.store_revision,
                owner_id=ContinuationClaimOwnerId("worker-a"),
                lease_expires_at=durable_support._NOW + timedelta(minutes=2),
                dispatch_id=ContinuationDispatchId("dispatch-a"),
                provider_idempotency_key=ProviderIdempotencyKey(
                    "provider-key"
                ),
                now=durable_support._NOW + timedelta(seconds=2),
            ),
            interaction_store.claim(
                request.continuation_id,
                expected_store_revision=ready.store_revision,
                owner_id=ContinuationClaimOwnerId("worker-b"),
                lease_expires_at=durable_support._NOW + timedelta(minutes=2),
                dispatch_id=ContinuationDispatchId("dispatch-b"),
                provider_idempotency_key=ProviderIdempotencyKey(
                    "provider-key"
                ),
                now=durable_support._NOW + timedelta(seconds=2),
            ),
            return_exceptions=True,
        )
        claim_successes = tuple(
            value for value in claims if not isinstance(value, BaseException)
        )
        claim_failures = tuple(
            value for value in claims if isinstance(value, BaseException)
        )
        assert len(claim_successes) == 1
        assert len(claim_failures) == 1
        assert isinstance(
            claim_failures[0],
            ContinuationStoreConflictError,
        )

        expired_request = durable_support._request("expired")
        expired_created = await interaction_store.create_durable(
            durable_support._create_command(expired_request),
            durable_support._portable(expired_request),
        )
        expired_resolution = await interaction_store.resolve(
            durable_support._answer(expired_created)
        )
        assert expired_resolution.record.request.resolution is not None
        expired_ready = await interaction_store.get_continuation(
            expired_request.continuation_id
        )
        expired_code = None
        try:
            await interaction_store.claim(
                expired_request.continuation_id,
                expected_store_revision=expired_ready.store_revision,
                owner_id=ContinuationClaimOwnerId("expired-worker"),
                lease_expires_at=durable_support._NOW + timedelta(minutes=12),
                dispatch_id=ContinuationDispatchId("expired-dispatch"),
                provider_idempotency_key=ProviderIdempotencyKey(
                    "provider-key"
                ),
                now=durable_support._NOW + timedelta(minutes=11),
            )
        except PgsqlInteractionStoreError as error:
            expired_code = error.code

        return (
            first.resolution.record,
            replay.resolution.record,
            conflict.error.code,
            len(claim_successes),
            type(claim_failures[0]),
            expired_code,
            tuple(row["event_type"] for row in database.events.values()),
        )

    (
        first_record,
        replay_record,
        conflict_code,
        claim_count,
        claim_error,
        expired_code,
        event_types,
    ) = run(exercise())
    assert replay_record == first_record
    assert conflict_code is InputErrorCode.IDEMPOTENCY_CONFLICT
    assert claim_count == 1
    assert claim_error is ContinuationStoreConflictError
    assert expired_code is InputErrorCode.EXPIRED
    assert event_types == (
        TaskInteractionEventType.INPUT_REQUIRED.value,
        TaskInteractionEventType.INPUT_RESUMED.value,
    )
