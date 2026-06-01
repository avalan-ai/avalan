from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.task import (
    TaskAttemptState,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskQueueAbandonment,
    TaskQueueArtifact,
    TaskQueueClaim,
    TaskQueueCompletion,
    TaskQueueDepth,
    TaskQueueHealth,
    TaskQueueItem,
    TaskQueueItemState,
    TaskQueueRetry,
    TaskQueueSubmission,
    TaskRetryPolicy,
    TaskRunPolicy,
    TaskRunState,
    TaskTargetContext,
    TaskTargetRunner,
    TaskValidationContext,
    TaskValidationIssue,
    TaskWorker,
)
from avalan.task.idempotency import TaskIdempotencyIdentity
from avalan.task.stores import InMemoryTaskStore


class FakeTarget(TaskTargetRunner):
    def __init__(self, output: object = "done") -> None:
        self.output = output
        self.contexts: list[TaskTargetContext] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        await context.check_cancelled()
        return self.output


class FakeQueue:
    def __init__(self, store: InMemoryTaskStore, now: datetime) -> None:
        self.store = store
        self.now = now
        self.item: TaskQueueItem | None = None
        self.completed: TaskQueueCompletion | None = None
        self.retried: TaskQueueRetry | None = None

    async def enqueue_run(
        self,
        request: TaskExecutionRequest,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        idempotency: TaskIdempotencyIdentity | None = None,
        idempotency_expires_at: datetime | None = None,
        artifacts: tuple[TaskQueueArtifact, ...] = (),
        run_metadata: Mapping[str, object] | None = None,
        queue_metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueSubmission:
        raise AssertionError("enqueue_run should not be used")

    async def enqueue(
        self,
        run_id: str,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueItem:
        raise AssertionError("enqueue should not be used")

    async def claim(
        self,
        queue_name: str,
        *,
        worker_id: str,
        lease_expires_at: datetime,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueClaim | None:
        if (
            self.item is None
            or self.item.state != TaskQueueItemState.AVAILABLE
        ):
            return None
        run = await self.store.assign_claim(
            self.item.run_id,
            from_states={TaskRunState.QUEUED},
            worker_id=worker_id,
            lease_expires_at=lease_expires_at,
            reason="claimed",
            metadata=metadata,
        )
        claim_token = run.claim.claim_token if run.claim else ""
        attempt = await self.store.create_attempt(
            run.run_id,
            claim_token=claim_token,
            metadata=metadata,
        )
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=run.run_id,
            queue_name=queue_name,
            state=TaskQueueItemState.CLAIMED,
            priority=self.item.priority,
            available_at=self.item.available_at,
            attempts=self.item.attempts,
            created_at=self.item.created_at,
            updated_at=now or self.now,
            run_state=run.state,
            claimed_at=run.claim.claimed_at if run.claim else None,
            lease_expires_at=run.claim.lease_expires_at if run.claim else None,
            worker_id=worker_id,
            claim_token=claim_token,
            heartbeat_at=run.claim.heartbeat_at if run.claim else None,
            metadata=metadata or {},
        )
        return TaskQueueClaim(queue_item=self.item, run=run, attempt=attempt)

    async def heartbeat(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        lease_expires_at: datetime,
        now: datetime | None = None,
    ) -> TaskQueueItem:
        raise AssertionError("heartbeat should not be used")

    async def complete(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        run_state: TaskRunState,
        attempt_state: TaskAttemptState,
        result: TaskExecutionResult | None = None,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueCompletion:
        assert self.item is not None
        attempt_id = (
            await self.store.get_run(self.item.run_id)
        ).last_attempt_id
        attempt = await self.store.transition_attempt(
            attempt_id or "",
            from_states={TaskAttemptState.RUNNING},
            to_state=attempt_state,
            reason="completed",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        run = await self.store.transition_run(
            self.item.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=run_state,
            reason="completed",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=run.run_id,
            queue_name=self.item.queue_name,
            state=(
                TaskQueueItemState.DONE
                if run_state == TaskRunState.SUCCEEDED
                else TaskQueueItemState.DEAD
            ),
            priority=self.item.priority,
            available_at=self.item.available_at,
            attempts=self.item.attempts,
            created_at=self.item.created_at,
            updated_at=now or self.now,
            run_state=run.state,
        )
        self.completed = TaskQueueCompletion(
            queue_item=self.item,
            run=run,
            attempt=attempt,
        )
        return self.completed

    async def retry(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        result: TaskExecutionResult,
        available_at: datetime,
        max_attempts: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueRetry:
        assert self.item is not None
        attempt_id = (
            await self.store.get_run(self.item.run_id)
        ).last_attempt_id
        attempt = await self.store.transition_attempt(
            attempt_id or "",
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.FAILED,
            reason="attempt_retry",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        run = await self.store.transition_run(
            self.item.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.QUEUED,
            reason="attempt_retry",
            claim_token=claim_token,
            metadata=metadata,
        )
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=run.run_id,
            queue_name=self.item.queue_name,
            state=TaskQueueItemState.AVAILABLE,
            priority=self.item.priority,
            available_at=available_at,
            attempts=self.item.attempts + 1,
            created_at=self.item.created_at,
            updated_at=now or self.now,
            run_state=run.state,
        )
        self.retried = TaskQueueRetry(
            queue_item=self.item,
            run=run,
            attempt=attempt,
        )
        return self.retried

    async def abandon_expired(
        self,
        queue_name: str,
        *,
        max_attempts: int,
        limit: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[TaskQueueAbandonment, ...]:
        return ()

    async def drain(
        self,
        queue_name: str,
        *,
        limit: int,
        now: datetime | None = None,
    ) -> tuple[TaskQueueItem, ...]:
        return ()

    async def depth(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueDepth:
        return TaskQueueDepth(
            queue_name=queue_name,
            available=0,
            scheduled=0,
            claimed=0,
            dead=0,
            cancel_requested=0,
        )

    async def health(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueHealth:
        return TaskQueueHealth(
            queue_name=queue_name,
            depth=await self.depth(queue_name),
            checked_at=now or self.now,
        )


class TaskWorkerTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)
        self.store = InMemoryTaskStore(clock=lambda: self.now)
        self.definition = _definition()
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
        )
        run = await self.store.create_run(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<redacted>"},
                queue="default",
            )
        )
        self.run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        self.run = await self.store.transition_run(
            self.run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        self.queue = FakeQueue(self.store, self.now)
        self.queue.item = TaskQueueItem(
            queue_item_id="queue-item-1",
            run_id=self.run.run_id,
            queue_name="default",
            state=TaskQueueItemState.AVAILABLE,
            priority=0,
            available_at=self.now,
            attempts=0,
            created_at=self.now,
            updated_at=self.now,
            run_state=self.run.state,
        )

    async def test_process_once_completes_claimed_run(self) -> None:
        target = FakeTarget("safe output")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            result.completion.attempt.state,
            TaskAttemptState.SUCCEEDED,
        )
        self.assertEqual(
            result.completion.run.result.output_summary,
            {"privacy": "<redacted>"},
        )
        self.assertEqual(
            target.contexts[0].execution.claim.worker_id, "worker-1"
        )
        self.assertEqual(
            target.contexts[0].input_value["privacy"], "<redacted>"
        )

    async def test_process_once_retries_retryable_failures(self) -> None:
        target = FakeTarget()
        target.run = _raise_os_error
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.retry)
        assert result.retry is not None
        self.assertTrue(result.retry.retryable)
        self.assertEqual(result.retry.run.state, TaskRunState.QUEUED)
        self.assertEqual(result.retry.attempt.state, TaskAttemptState.FAILED)
        self.assertNotIn("private", str(result.retry.attempt.result))

    async def test_process_once_reports_no_work(self) -> None:
        self.queue.item = None
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertFalse(result.processed)
        self.assertIsNone(result.completion)


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="worker_task", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        run=TaskRunPolicy.queued("default"),
        retry=TaskRetryPolicy(max_attempts=2),
    )


async def _raise_os_error(context: TaskTargetContext) -> object:
    raise OSError("private backend path")


if __name__ == "__main__":
    main()
