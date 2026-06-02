from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.event import Event, EventType
from avalan.task import (
    HASHED_MARKER,
    REDACTED_MARKER,
    TaskArtifactPurpose,
    TaskArtifactState,
    TaskAttemptState,
    TaskClient,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskFileDescriptor,
    TaskInputContract,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskQueue,
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
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.idempotency import TaskIdempotencyIdentity
from avalan.task.stores import InMemoryTaskStore


class StaticHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        return TaskKeyMaterial(
            key_id=key_id or purpose.value,
            algorithm="hmac-sha256",
            secret=b"queue-worker-e2e-secret",
        )


class ReadingTarget(TaskTargetRunner):
    def __init__(self) -> None:
        self.file_bodies: list[bytes] = []
        self.inputs: list[object] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        self.inputs.append(context.input_value)
        await context.check_cancelled()
        assert context.artifact_store is not None
        for file in context.files:
            assert file.artifact_ref is not None
            reader = await context.artifact_store.open(file.artifact_ref)
            try:
                self.file_bodies.append(reader.read())
            finally:
                reader.close()
        if context.event_listener is not None:
            result = context.event_listener(
                Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={
                        "status": "ok",
                        "token": "private-token",
                        "token_id": 7,
                    },
                )
            )
            if result is not None:
                await result
        await context.observe_usage(
            SimpleNamespace(
                input_token_count=11,
                output_token_count=3,
                total_token_count=14,
            )
        )
        return "public answer"


class FailingTarget(TaskTargetRunner):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        await context.check_cancelled()
        raise OSError("private backend path /tmp/customer-secret.txt")


class InMemoryTaskQueue:
    def __init__(
        self,
        store: InMemoryTaskStore,
        *,
        clock: object,
    ) -> None:
        self.store = store
        self.clock = cast("Clock", clock)
        self.items: dict[str, TaskQueueItem] = {}
        self.items_by_run_id: dict[str, str] = {}
        self.next_id = 1

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
        if idempotency is not None:
            existing = await self.store.lookup_idempotency_key(idempotency)
            if existing is not None:
                run = await self.store.get_run(existing.run_id)
                queue_item_id = self.items_by_run_id.get(run.run_id)
                return TaskQueueSubmission(
                    run=run,
                    created=False,
                    queue_item=(
                        self.items[queue_item_id]
                        if queue_item_id is not None
                        else None
                    ),
                    artifacts=await self.store.list_artifacts(run.run_id),
                )

        run = await self.store.create_run(request, metadata=run_metadata)
        artifact_records = []
        for artifact in artifacts:
            artifact_records.append(
                await self.store.append_artifact(
                    run.run_id,
                    ref=artifact.ref,
                    purpose=artifact.purpose,
                    state=artifact.state,
                    provenance=artifact.provenance,
                    retention=artifact.retention,
                    metadata=artifact.metadata,
                )
            )
        idempotency_result = None
        if idempotency is not None:
            idempotency_result = await self.store.reserve_idempotency_key(
                idempotency,
                run_id=run.run_id,
                expires_at=idempotency_expires_at,
            )
        run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        queue_item_id = f"queue-item-{self.next_id}"
        self.next_id += 1
        now = self.clock.now
        item = TaskQueueItem(
            queue_item_id=queue_item_id,
            run_id=run.run_id,
            queue_name=queue_name,
            state=TaskQueueItemState.AVAILABLE,
            priority=priority,
            available_at=available_at or now,
            attempts=0,
            created_at=now,
            updated_at=now,
            run_state=run.state,
            metadata=queue_metadata or {},
        )
        self.items[item.queue_item_id] = item
        self.items_by_run_id[run.run_id] = item.queue_item_id
        return TaskQueueSubmission(
            run=run,
            created=True,
            queue_item=item,
            idempotency=idempotency_result,
            artifacts=tuple(artifact_records),
        )

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
        current_time = now or self.clock.now
        for item in tuple(self.items.values()):
            if (
                item.queue_name != queue_name
                or item.state != TaskQueueItemState.AVAILABLE
                or item.available_at > current_time
            ):
                continue
            run = await self.store.get_run(item.run_id)
            if run.state != TaskRunState.QUEUED:
                continue
            claimed = await self.store.assign_claim(
                run.run_id,
                from_states={TaskRunState.QUEUED},
                worker_id=worker_id,
                lease_expires_at=lease_expires_at,
                reason="claimed",
                metadata=metadata,
            )
            claim_token = claimed.claim.claim_token if claimed.claim else ""
            attempt = await self.store.create_attempt(
                claimed.run_id,
                claim_token=claim_token,
                metadata=metadata,
            )
            updated = TaskQueueItem(
                queue_item_id=item.queue_item_id,
                run_id=claimed.run_id,
                queue_name=item.queue_name,
                state=TaskQueueItemState.CLAIMED,
                priority=item.priority,
                available_at=item.available_at,
                attempts=item.attempts,
                created_at=item.created_at,
                updated_at=current_time,
                run_state=claimed.state,
                claimed_at=claimed.claim.claimed_at if claimed.claim else None,
                lease_expires_at=(
                    claimed.claim.lease_expires_at if claimed.claim else None
                ),
                worker_id=worker_id,
                claim_token=claim_token,
                heartbeat_at=(
                    claimed.claim.heartbeat_at if claimed.claim else None
                ),
                metadata=metadata or {},
            )
            self.items[item.queue_item_id] = updated
            return TaskQueueClaim(
                queue_item=updated,
                run=claimed,
                attempt=attempt,
            )
        return None

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
        item = self.items[queue_item_id]
        run = await self.store.get_run(item.run_id)
        attempt = await self.store.transition_attempt(
            run.last_attempt_id or "",
            from_states={TaskAttemptState.RUNNING},
            to_state=attempt_state,
            reason="completed",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        completed_run = await self.store.transition_run(
            run.run_id,
            from_states={run.state},
            to_state=run_state,
            reason="completed",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        updated = TaskQueueItem(
            queue_item_id=item.queue_item_id,
            run_id=item.run_id,
            queue_name=item.queue_name,
            state=(
                TaskQueueItemState.DONE
                if run_state == TaskRunState.SUCCEEDED
                else TaskQueueItemState.DEAD
            ),
            priority=item.priority,
            available_at=item.available_at,
            attempts=item.attempts,
            created_at=item.created_at,
            updated_at=now or self.clock.now,
            run_state=completed_run.state,
        )
        self.items[queue_item_id] = updated
        return TaskQueueCompletion(
            queue_item=updated,
            run=completed_run,
            attempt=attempt,
        )

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
        _ = max_attempts
        item = self.items[queue_item_id]
        run = await self.store.get_run(item.run_id)
        attempt = await self.store.transition_attempt(
            run.last_attempt_id or "",
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.FAILED,
            reason="retry",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        queued_run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.QUEUED,
            reason="retry",
            claim_token=claim_token,
            metadata=metadata,
        )
        updated = TaskQueueItem(
            queue_item_id=item.queue_item_id,
            run_id=item.run_id,
            queue_name=item.queue_name,
            state=TaskQueueItemState.AVAILABLE,
            priority=item.priority,
            available_at=available_at,
            attempts=item.attempts + 1,
            created_at=item.created_at,
            updated_at=now or self.clock.now,
            run_state=queued_run.state,
        )
        self.items[queue_item_id] = updated
        return TaskQueueRetry(
            queue_item=updated,
            run=queued_run,
            attempt=attempt,
        )

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
        return tuple(
            item
            for item in self.items.values()
            if item.queue_name == queue_name
            and item.state == TaskQueueItemState.AVAILABLE
        )[:limit]

    async def depth(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueDepth:
        current_time = now or self.clock.now
        available = 0
        scheduled = 0
        claimed = 0
        dead = 0
        cancel_requested = 0
        for item in self.items.values():
            if item.queue_name != queue_name:
                continue
            run = await self.store.get_run(item.run_id)
            if run.state == TaskRunState.CANCEL_REQUESTED:
                cancel_requested += 1
            if item.state == TaskQueueItemState.AVAILABLE:
                if item.available_at <= current_time:
                    available += 1
                else:
                    scheduled += 1
            elif item.state == TaskQueueItemState.CLAIMED:
                claimed += 1
            elif item.state == TaskQueueItemState.DEAD:
                dead += 1
        return TaskQueueDepth(
            queue_name=queue_name,
            available=available,
            scheduled=scheduled,
            claimed=claimed,
            dead=dead,
            cancel_requested=cancel_requested,
        )

    async def health(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueHealth:
        current_time = now or self.clock.now
        depth = await self.depth(queue_name, now=current_time)
        oldest_available_at = min(
            (
                item.available_at
                for item in self.items.values()
                if item.queue_name == queue_name
                and item.state == TaskQueueItemState.AVAILABLE
                and item.available_at <= current_time
            ),
            default=None,
        )
        expired_claims = sum(
            1
            for item in self.items.values()
            if item.queue_name == queue_name
            and item.state == TaskQueueItemState.CLAIMED
            and item.lease_expires_at is not None
            and item.lease_expires_at <= current_time
        )
        return TaskQueueHealth(
            queue_name=queue_name,
            depth=depth,
            checked_at=current_time,
            oldest_available_at=oldest_available_at,
            expired_claims=expired_claims,
        )


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    async def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class QueueWorkerE2ETest(IsolatedAsyncioTestCase):
    async def test_file_task_runs_through_client_worker_and_inspection(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            input_path = root / "private.txt"
            input_path.write_text("private file body", encoding="utf-8")
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
            )
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            target = ReadingTarget()
            client = _client(
                store,
                queue,
                target=target,
                artifact_store=artifact_store,
                execution_roots=(root,),
                clock=clock,
            )
            worker = _worker(
                store,
                queue,
                target=target,
                artifact_store=artifact_store,
                clock=clock,
            )
            definition = _definition(
                input_contract=TaskInputContract.file(
                    mime_types=("text/plain",)
                )
            )

            submission = await client.enqueue(
                definition,
                input_value=TaskFileDescriptor.local_path(
                    "private.txt",
                    mime_type="text/plain",
                    metadata={"filename": "private.txt"},
                ),
                idempotency_key="private-idempotency-key",
                owner_scope="customer-123",
                queue_metadata={"tenant": "safe"},
            )
            processed = await worker.process_once()
            waited = await client.wait(
                submission.run.run_id,
                timeout_seconds=0,
                poll_interval_seconds=0.01,
            )
            inspection = await client.inspect(submission.run.run_id)
            after_first_event = await client.events(
                submission.run.run_id,
                after_sequence=1,
            )
            depth = await queue.depth("default")
            health = await queue.health("default")

        self.assertTrue(submission.created)
        self.assertIsNotNone(submission.idempotency)
        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertTrue(waited.ready)
        self.assertEqual(waited.state, TaskRunState.SUCCEEDED)
        self.assertEqual(waited.output_summary, {"privacy": REDACTED_MARKER})
        self.assertEqual(target.file_bodies, [b"private file body"])
        self.assertEqual(target.inputs[0]["privacy"], HASHED_MARKER)
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(
            inspection.attempts[0].state,
            TaskAttemptState.SUCCEEDED,
        )
        self.assertEqual(len(inspection.events), 1)
        self.assertEqual(inspection.events[0].sequence, 1)
        self.assertNotIn("private-token", str(inspection.as_dict()))
        self.assertNotIn("private file body", str(inspection.as_dict()))
        self.assertNotIn("private-idempotency-key", str(inspection.as_dict()))
        self.assertNotIn("customer-123", str(inspection.as_dict()))
        self.assertEqual(inspection.usage_totals.input_tokens, 11)
        self.assertEqual(inspection.usage_totals.output_tokens, 3)
        self.assertEqual(inspection.usage_totals.total_tokens, 14)
        self.assertEqual(len(inspection.artifacts), 1)
        artifact = inspection.artifacts[0]
        assert isinstance(artifact, Mapping)
        self.assertEqual(artifact["purpose"], TaskArtifactPurpose.INPUT.value)
        self.assertEqual(artifact["state"], TaskArtifactState.READY.value)
        self.assertEqual(after_first_event, ())
        self.assertEqual(depth.active, 0)
        self.assertEqual(depth.dead, 0)
        self.assertIsNone(health.oldest_available_at)

    async def test_terminal_worker_failure_is_safe_to_inspect(self) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = FailingTarget()
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(retry=TaskRetryPolicy(max_attempts=1))

        submission = await client.enqueue(
            definition,
            input_value="private prompt",
            queue_metadata={"tenant": "safe"},
        )
        result = await worker.process_once()
        output = await client.wait(
            submission.run.run_id,
            timeout_seconds=0,
            poll_interval_seconds=0.01,
        )
        inspection = await client.inspect(submission.run.run_id)
        depth = await queue.depth("default")

        self.assertTrue(result.processed)
        self.assertIsNone(result.retry)
        self.assertIsNotNone(result.claimed)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(inspection.attempts[0].state, TaskAttemptState.FAILED)
        self.assertEqual(depth.dead, 1)
        self.assertIn("infra", str(output.error))
        self.assertNotIn("private backend path", str(inspection.as_dict()))
        self.assertNotIn("customer-secret", str(inspection.as_dict()))
        self.assertNotIn("private prompt", str(inspection.as_dict()))


def _client(
    store: InMemoryTaskStore,
    queue: InMemoryTaskQueue,
    *,
    target: TaskTargetRunner,
    artifact_store: LocalArtifactStore | None = None,
    execution_roots: tuple[Path, ...] = (),
    clock: Clock,
) -> TaskClient:
    return TaskClient(
        store,
        target=target,
        queue=cast(TaskQueue, queue),
        hmac_provider=StaticHmacProvider(),
        artifact_store=artifact_store,
        definition_hash=lambda definition: "queue-worker-e2e",
        execution_roots=execution_roots,
        clock=lambda: clock.now,
        sleep=clock.sleep,
    )


def _worker(
    store: InMemoryTaskStore,
    queue: InMemoryTaskQueue,
    *,
    target: TaskTargetRunner,
    artifact_store: LocalArtifactStore | None = None,
    clock: Clock,
) -> TaskWorker:
    return TaskWorker(
        store,
        cast(TaskQueue, queue),
        target=target,
        worker_id="worker-1",
        artifact_store=artifact_store,
        clock=lambda: clock.now,
    )


def _definition(
    *,
    input_contract: TaskInputContract | None = None,
    retry: TaskRetryPolicy | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="queue_worker_e2e", version="1"),
        input=input_contract or TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        run=TaskRunPolicy.queued("default"),
        retry=retry or TaskRetryPolicy(max_attempts=2),
    )


if __name__ == "__main__":
    main()
