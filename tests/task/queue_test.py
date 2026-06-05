from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import cast
from unittest import TestCase, main

from avalan.task import (
    TaskAttempt,
    TaskAttemptState,
    TaskClaim,
    TaskExecutionContext,
    TaskExecutionRequest,
    TaskQueueClaim,
    TaskQueueDepth,
    TaskQueueHealth,
    TaskQueueItem,
    TaskQueueItemState,
    TaskRun,
    TaskRunState,
)


class TaskQueueModelTest(TestCase):
    def test_queue_item_exposes_cancellation_visibility(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)

        item = TaskQueueItem(
            queue_item_id="queue-item-1",
            run_id="run-1",
            queue_name="default",
            state=TaskQueueItemState.AVAILABLE,
            priority=5,
            available_at=now,
            attempts=0,
            created_at=now,
            updated_at=now,
            run_state=TaskRunState.CANCEL_REQUESTED,
            metadata={"labels": ["safe"]},
        )

        self.assertTrue(item.cancel_requested)
        self.assertEqual(item.metadata["labels"], ("safe",))
        self.assertIsInstance(item.metadata, MappingProxyType)
        with self.assertRaises(TypeError):
            cast(dict[str, object], item.metadata)["raw"] = "value"

    def test_queue_models_validate_invariants(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)

        with self.assertRaises(AssertionError):
            TaskQueueItem(
                queue_item_id="queue-item-1",
                run_id="run-1",
                queue_name="default",
                state=TaskQueueItemState.CLAIMED,
                priority=0,
                available_at=now,
                attempts=0,
                created_at=now,
                updated_at=now,
                run_state=TaskRunState.QUEUED,
            )
        with self.assertRaises(AssertionError):
            TaskQueueItem(
                queue_item_id="queue-item-1",
                run_id="run-1",
                queue_name="default",
                state=TaskQueueItemState.AVAILABLE,
                priority=0,
                available_at=now,
                attempts=0,
                created_at=now,
                updated_at=now,
                run_state=TaskRunState.QUEUED,
                metadata={"raw": object()},
            )

        claimed = TaskQueueItem(
            queue_item_id="queue-item-2",
            run_id="run-2",
            queue_name="default",
            state=TaskQueueItemState.CLAIMED,
            priority=0,
            available_at=now,
            attempts=1,
            created_at=now,
            updated_at=now,
            run_state=TaskRunState.CLAIMED,
            claimed_at=now,
            lease_expires_at=now + timedelta(minutes=1),
            worker_id="worker-1",
            claim_token="claim-token",
            heartbeat_at=now,
        )
        depth = TaskQueueDepth(
            queue_name="default",
            available=1,
            scheduled=2,
            claimed=3,
            dead=4,
            cancel_requested=5,
        )
        health = TaskQueueHealth(
            queue_name="default",
            depth=depth,
            checked_at=now,
            oldest_available_at=now - timedelta(minutes=1),
            expired_claims=1,
        )

        self.assertEqual(claimed.worker_id, "worker-1")
        self.assertEqual(depth.active, 6)
        self.assertEqual(health.expired_claims, 1)
        with self.assertRaises(AssertionError):
            TaskQueueDepth(
                queue_name="default",
                available=-1,
                scheduled=0,
                claimed=0,
                dead=0,
                cancel_requested=0,
            )

    def test_queue_claim_validates_consistent_claim_context(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        claim = TaskClaim(
            worker_id="worker-1",
            claim_token="claim-token",
            claimed_at=now,
            lease_expires_at=now + timedelta(minutes=5),
            heartbeat_at=now,
        )
        queue_item = TaskQueueItem(
            queue_item_id="queue-item-1",
            run_id="run-1",
            queue_name="default",
            state=TaskQueueItemState.CLAIMED,
            priority=0,
            available_at=now,
            attempts=1,
            created_at=now,
            updated_at=now,
            run_state=TaskRunState.CLAIMED,
            claimed_at=claim.claimed_at,
            lease_expires_at=claim.lease_expires_at,
            worker_id=claim.worker_id,
            claim_token=claim.claim_token,
            heartbeat_at=claim.heartbeat_at,
        )
        run = TaskRun(
            run_id="run-1",
            definition_id="definition-1",
            state=TaskRunState.CLAIMED,
            request=TaskExecutionRequest(definition_id="definition-1"),
            created_at=now,
            updated_at=now,
            claim=claim,
            last_attempt_id="attempt-1",
        )
        attempt = TaskAttempt(
            attempt_id="attempt-1",
            run_id="run-1",
            attempt_number=1,
            state=TaskAttemptState.CREATED,
            context=TaskExecutionContext(
                run_id="run-1",
                attempt_id="attempt-1",
                attempt_number=1,
                claim=claim,
            ),
            created_at=now,
            updated_at=now,
        )

        queue_claim = TaskQueueClaim(
            queue_item=queue_item,
            run=run,
            attempt=attempt,
        )

        self.assertEqual(queue_claim.attempt.context.claim, claim)
        with self.assertRaises(AssertionError):
            TaskQueueClaim(
                queue_item=queue_item,
                run=run,
                attempt=TaskAttempt(
                    attempt_id="attempt-2",
                    run_id="run-1",
                    attempt_number=2,
                    state=TaskAttemptState.CREATED,
                    context=TaskExecutionContext(
                        run_id="run-1",
                        attempt_id="attempt-2",
                        attempt_number=2,
                    ),
                    created_at=now,
                    updated_at=now,
                ),
            )


if __name__ == "__main__":
    main()
