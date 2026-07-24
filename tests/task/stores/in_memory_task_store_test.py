from datetime import UTC, datetime, timedelta
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.task import (
    IdempotencyMode,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskIdempotencyDigest,
    TaskIdempotencyIdentity,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskRunState,
    TaskStoreConflictError,
    TaskStoreNotFoundError,
    UsageSource,
    UsageTotals,
)
from avalan.task.stores import InMemoryTaskStore


class FixedClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.value = self.value + timedelta(seconds=1)
        return self.value


class InMemoryTaskStoreTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.clock = FixedClock()
        self.store = InMemoryTaskStore(clock=self.clock)
        self.definition = TaskDefinition(
            task=TaskMetadata(name="classify", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/classify.toml"),
        )
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-classify",
        )

    async def test_generated_ids_are_non_empty(self) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        attempt = await self.store.create_attempt(run.run_id)

        self.assertTrue(run.run_id)
        self.assertTrue(attempt.attempt_id)

    async def test_default_clock_creates_timestamped_runs(self) -> None:
        store = InMemoryTaskStore()
        await store.register_definition(
            self.definition,
            definition_hash="hash-default-clock",
        )
        run = await store.create_run(
            TaskExecutionRequest(
                definition_id="hash-default-clock",
                idempotency_key="request-1",
                queue="default",
            )
        )

        self.assertIsNotNone(run.created_at.tzinfo)
        self.assertEqual(run.request.idempotency_key, "request-1")
        self.assertEqual(run.request.queue, "default")

    async def test_lookup_methods_raise_stable_not_found_errors(self) -> None:
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.get_definition("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.get_run("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.get_attempt("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_attempts("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_run_transitions("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_attempt_transitions("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_events("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.get_artifact("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_artifacts("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.get_attempt_segment("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_attempt_segment_transitions("missing")

    async def test_attempt_segments_enforce_lineage_and_correlation(
        self,
    ) -> None:
        first_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        first_attempt = await self.store.create_attempt(first_run.run_id)

        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "initial segment cannot resume another segment",
        ):
            await self.store.create_attempt_segment(
                first_attempt.attempt_id,
                resumed_from_segment_id="missing-segment",
            )
        first_segment = await self.store.create_attempt_segment(
            first_attempt.attempt_id,
            metadata={"phase": "initial"},
        )
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "task attempt already has an active segment",
        ):
            await self.store.create_attempt_segment(first_attempt.attempt_id)

        running_segment = await self.store.transition_attempt_segment(
            first_segment.segment_id,
            from_states={TaskAttemptSegmentState.CREATED},
            to_state=TaskAttemptSegmentState.RUNNING,
            reason="started",
        )
        with self.assertRaisesRegex(
            AssertionError,
            "only suspended segments have request correlation",
        ):
            await self.store.transition_attempt_segment(
                running_segment.segment_id,
                from_states={TaskAttemptSegmentState.RUNNING},
                to_state=TaskAttemptSegmentState.SUCCEEDED,
                reason="invalid-correlation",
                request_id="request-1",
                continuation_id="continuation-1",
            )
        suspended_segment = await self.store.transition_attempt_segment(
            running_segment.segment_id,
            from_states={TaskAttemptSegmentState.RUNNING},
            to_state=TaskAttemptSegmentState.SUSPENDED,
            reason="input-required",
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "resumed segment must link the prior segment",
        ):
            await self.store.create_attempt_segment(
                first_attempt.attempt_id,
                resumed_from_segment_id="wrong-segment",
            )
        resumed_segment = await self.store.create_attempt_segment(
            first_attempt.attempt_id,
            resumed_from_segment_id=suspended_segment.segment_id,
        )
        self.assertEqual(
            await self.store.get_attempt_segment(resumed_segment.segment_id),
            resumed_segment,
        )
        self.assertEqual(
            await self.store.list_attempt_segments(first_attempt.attempt_id),
            (suspended_segment, resumed_segment),
        )
        transitions = await self.store.list_attempt_segment_transitions(
            first_segment.segment_id
        )
        self.assertEqual(
            [transition.to_state for transition in transitions],
            [
                TaskAttemptSegmentState.RUNNING,
                TaskAttemptSegmentState.SUSPENDED,
            ],
        )

        await self.store.transition_attempt(
            first_attempt.attempt_id,
            from_states={TaskAttemptState.CREATED},
            to_state=TaskAttemptState.ABANDONED,
            reason="test-terminal",
        )
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "task attempt cannot start a segment",
        ):
            await self.store.create_attempt_segment(first_attempt.attempt_id)

        second_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        second_attempt = await self.store.create_attempt(second_run.run_id)
        second_segment = await self.store.create_attempt_segment(
            second_attempt.attempt_id
        )
        with self.assertRaisesRegex(
            TaskStoreNotFoundError,
            "task attempt segment was not found for run",
        ):
            await self.store.append_usage(
                first_run.run_id,
                attempt_id=first_attempt.attempt_id,
                segment_id=second_segment.segment_id,
                source=UsageSource.EXACT,
                totals=UsageTotals(input_tokens=1),
            )

    async def test_usage_list_and_totals_filter_by_source(self) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        attempt = await self.store.create_attempt(run.run_id)
        exact = await self.store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=1, cached_input_tokens=0),
        )
        await self.store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            source=UsageSource.UNAVAILABLE,
            totals=UsageTotals(),
        )
        estimated = await self.store.append_usage(
            run.run_id,
            source=UsageSource.ESTIMATED,
            totals=UsageTotals(output_tokens=5),
        )

        self.assertEqual(
            await self.store.list_usage(
                run.run_id,
                attempt_id=attempt.attempt_id,
                source=UsageSource.EXACT,
            ),
            (exact,),
        )
        self.assertEqual(
            await self.store.list_usage(
                run.run_id,
                source=UsageSource.ESTIMATED,
            ),
            (estimated,),
        )
        self.assertEqual(
            await self.store.usage_totals(
                run.run_id,
                source=UsageSource.EXACT,
            ),
            UsageTotals(input_tokens=1, cached_input_tokens=0),
        )

    async def test_usage_filters_reject_invalid_source(self) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        invalid_source = cast(UsageSource, "exact")

        with self.assertRaises(AssertionError):
            await self.store.list_usage(
                run.run_id,
                source=invalid_source,
            )
        with self.assertRaises(AssertionError):
            await self.store.usage_totals(
                run.run_id,
                source=invalid_source,
            )

    async def test_run_and_attempt_metadata_rejects_unsafe_values(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            await self.store.create_run(
                TaskExecutionRequest(definition_id="hash-classify"),
                metadata={"raw": b"bytes"},
            )

        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        with self.assertRaises(AssertionError):
            await self.store.create_attempt(
                run.run_id,
                metadata={"raw": object()},
            )
        with self.assertRaises(AssertionError):
            await self.store.append_artifact(
                run.run_id,
                ref=TaskArtifactRef(
                    artifact_id="artifact-1",
                    store="local",
                    storage_key="ar/artifact-1",
                ),
                purpose=TaskArtifactPurpose.INPUT,
                metadata={"raw": object()},
            )

    async def test_terminal_run_cannot_create_attempt(self) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        failed = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.FAILED,
            reason="validation_failed",
        )

        self.assertEqual(failed.state, TaskRunState.FAILED)
        with self.assertRaises(TaskStoreConflictError):
            await self.store.create_attempt(run.run_id)

    async def test_assign_claim_rejects_duplicate_claims(self) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        validated = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        queued = await self.store.transition_run(
            validated.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        claimed = await self.store.assign_claim(
            queued.run_id,
            from_states={TaskRunState.QUEUED},
            worker_id="worker-1",
            lease_expires_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            reason="claimed",
        )

        self.assertIsNotNone(claimed.claim)
        with self.assertRaises(TaskStoreConflictError):
            await self.store.assign_claim(
                queued.run_id,
                from_states={TaskRunState.CLAIMED},
                worker_id="worker-2",
                lease_expires_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
                reason="claimed_again",
            )

    async def test_claimed_run_can_be_marked_cancel_requested_without_token(
        self,
    ) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        validated = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        queued = await self.store.transition_run(
            validated.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        claimed = await self.store.assign_claim(
            queued.run_id,
            from_states={TaskRunState.QUEUED},
            worker_id="worker-1",
            lease_expires_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            reason="claimed",
        )
        assert claimed.claim is not None
        running = await self.store.transition_run(
            claimed.run_id,
            from_states={TaskRunState.CLAIMED},
            to_state=TaskRunState.RUNNING,
            reason="started",
            claim_token=claimed.claim.claim_token,
        )

        cancel_requested = await self.store.transition_run(
            running.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.CANCEL_REQUESTED,
            reason="cancel_requested",
        )

        self.assertEqual(cancel_requested.state, TaskRunState.CANCEL_REQUESTED)
        self.assertIsNotNone(cancel_requested.claim)
        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_run(
                cancel_requested.run_id,
                from_states={TaskRunState.CANCEL_REQUESTED},
                to_state=TaskRunState.FAILED,
                reason="failed",
            )

    async def test_attempt_transition_rejects_stale_expected_state(
        self,
    ) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        attempt = await self.store.create_attempt(run.run_id)

        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_attempt(
                attempt.attempt_id,
                from_states={TaskAttemptState.RUNNING},
                to_state=TaskAttemptState.SUCCEEDED,
                reason="stale",
            )

    async def test_idempotency_reservation_returns_existing_run(
        self,
    ) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        duplicate = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        identity = _identity("identity-1")

        first = await self.store.reserve_idempotency_key(
            identity,
            run_id=run.run_id,
            metadata={"source": "sdk"},
        )
        second = await self.store.reserve_idempotency_key(
            identity,
            run_id=duplicate.run_id,
        )
        found = await self.store.lookup_idempotency_key(identity)

        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(second.reservation.run_id, run.run_id)
        self.assertEqual(found, first.reservation)
        self.assertEqual(first.reservation.metadata["source"], "sdk")

    async def test_expired_idempotency_reservation_can_be_replaced(
        self,
    ) -> None:
        first_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        second_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        identity = _identity("identity-expiring")

        first = await self.store.reserve_idempotency_key(
            identity,
            run_id=first_run.run_id,
            expires_at=datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC),
        )
        self.clock.value = datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC)
        self.assertIsNone(await self.store.lookup_idempotency_key(identity))
        second = await self.store.reserve_idempotency_key(
            identity,
            run_id=second_run.run_id,
        )

        self.assertTrue(first.created)
        self.assertTrue(second.created)
        self.assertEqual(second.reservation.run_id, second_run.run_id)

    async def test_active_idempotency_key_rejects_identity_mismatch(
        self,
    ) -> None:
        first_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        second_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        identity = _identity("identity-collision")
        mismatched = _identity(
            "identity-collision",
            spec_hash="hash-other",
        )

        reserved = await self.store.reserve_idempotency_key(
            identity,
            run_id=first_run.run_id,
        )

        with self.assertRaises(TaskStoreConflictError):
            await self.store.reserve_idempotency_key(
                mismatched,
                run_id=second_run.run_id,
            )
        with self.assertRaises(TaskStoreConflictError):
            await self.store.lookup_idempotency_key(mismatched)

        found = await self.store.lookup_idempotency_key(identity)
        self.assertEqual(found, reserved.reservation)

    async def test_idempotency_reservation_requires_existing_run(self) -> None:
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.reserve_idempotency_key(
                _identity("identity-missing-run"),
                run_id="missing",
            )

    async def test_claim_token_rejected_for_unclaimed_run(self) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )

        with self.assertRaises(TaskStoreConflictError):
            await self.store.create_attempt(
                run.run_id,
                claim_token="unexpected",
            )


if __name__ == "__main__":
    main()


def _identity(
    identity_key: str,
    *,
    spec_hash: str = "hash-classify",
) -> TaskIdempotencyIdentity:
    digest = TaskIdempotencyDigest(
        algorithm="hmac-sha256",
        digest="a" * 64,
        key_id="idempotency",
    )
    return TaskIdempotencyIdentity(
        identity_key=identity_key,
        task_name="classify",
        task_version="1",
        spec_hash=spec_hash,
        owner_scope=digest,
        strategy=IdempotencyMode.INPUT_HASH,
        input=digest,
    )
