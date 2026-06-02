from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.event import Event, EventType
from avalan.flow.flow import Flow
from avalan.flow.node import Node
from avalan.task import (
    HASHED_MARKER,
    REDACTED_MARKER,
    STORED_MARKER,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactState,
    TaskAttemptState,
    TaskClient,
    TaskClientWaitTimeoutError,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskFileDescriptor,
    TaskInputContract,
    TaskInputFile,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskObservabilityPolicy,
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
    TaskValidationError,
    TaskValidationIssue,
    TaskWorker,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.idempotency import TaskIdempotencyIdentity
from avalan.task.stores import InMemoryTaskStore
from avalan.task.targets import FLOW_TASK_INPUT_KEY, FlowTaskTargetRunner


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


class TextTarget(TaskTargetRunner):
    def __init__(self) -> None:
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
        return "public answer"


class StructuredQueueTarget(TaskTargetRunner):
    def __init__(self, outcomes: tuple[object, ...]) -> None:
        self.outcomes = outcomes
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
        attempt_number = len(self.inputs)
        if context.event_listener is not None:
            result = context.event_listener(
                Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={
                        "status": "attempt",
                        "count": attempt_number,
                        "token": f"private-token-{attempt_number}",
                        "token_id": attempt_number,
                    },
                )
            )
            if result is not None:
                await result
        await context.observe_usage(
            SimpleNamespace(
                input_token_count=attempt_number,
                output_token_count=attempt_number + 1,
                total_token_count=(attempt_number * 2) + 1,
            )
        )
        outcome = self.outcomes[
            min(attempt_number - 1, len(self.outcomes) - 1)
        ]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class ArtifactOutputTarget(TaskTargetRunner):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        await context.check_cancelled()
        assert context.artifact_store is not None
        return await context.artifact_store.put(
            b"private generated report",
            media_type="text/plain",
            metadata={"filename": "report.txt"},
        )


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


class CancellingTarget(TaskTargetRunner):
    def __init__(
        self,
        cancel: Callable[[str], Awaitable[object]],
    ) -> None:
        self.cancel = cancel
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
        await self.cancel(context.execution.run_id)
        await context.check_cancelled()
        return "unused"


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
        queued_run = replace(queued_run, claim=None)
        self.store._runs[run.run_id] = queued_run
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

    async def test_file_array_task_materializes_all_inputs(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            first_path = root / "first.txt"
            second_path = root / "second.txt"
            first_path.write_text("first private body", encoding="utf-8")
            second_path.write_text("second private body", encoding="utf-8")
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
                input_contract=TaskInputContract.file_array(
                    mime_types=("text/plain",)
                ),
                artifact=TaskArtifactPolicy(max_count=2),
            )

            submission = await client.enqueue(
                definition,
                input_value=(
                    TaskFileDescriptor.local_path(
                        "first.txt",
                        mime_type="text/plain",
                        metadata={"filename": "first.txt"},
                    ),
                    TaskFileDescriptor.local_path(
                        "second.txt",
                        mime_type="text/plain",
                        metadata={"filename": "second.txt"},
                    ),
                ),
            )
            processed = await worker.process_once()
            inspection = await client.inspect(submission.run.run_id)
            artifacts = await store.list_artifacts(
                submission.run.run_id,
                purpose=TaskArtifactPurpose.INPUT,
            )

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertEqual(
            target.file_bodies,
            [b"first private body", b"second private body"],
        )
        self.assertEqual(len(artifacts), 2)
        self.assertEqual(
            [artifact.purpose for artifact in artifacts],
            [TaskArtifactPurpose.INPUT, TaskArtifactPurpose.INPUT],
        )
        self.assertEqual(len(inspection.artifacts), 2)
        self.assertNotIn("first private body", str(inspection.as_dict()))
        self.assertNotIn("second private body", str(inspection.as_dict()))

    async def test_structured_file_input_materializes_for_worker(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            Path(root, "document.txt").write_bytes(b"private structured body")
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
                input_contract=TaskInputContract.object(
                    schema={
                        "type": "object",
                        "required": ["prompt", "document"],
                        "additionalProperties": False,
                        "properties": {
                            "prompt": {"type": "string"},
                            "document": {"type": "object"},
                        },
                    }
                ),
                artifact=TaskArtifactPolicy(max_count=1),
            )

            submission = await client.enqueue(
                definition,
                input_value={
                    "prompt": "Review the document.",
                    "document": {
                        "source_kind": "local_path",
                        "reference": "document.txt",
                        "mime_type": "text/plain",
                    },
                },
            )
            processed = await worker.process_once()
            inspection = await client.inspect(submission.run.run_id)
            artifacts = await store.list_artifacts(
                submission.run.run_id,
                purpose=TaskArtifactPurpose.INPUT,
            )

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertEqual(target.file_bodies, [b"private structured body"])
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].purpose, TaskArtifactPurpose.INPUT)
        self.assertEqual(len(inspection.artifacts), 1)
        assert isinstance(target.inputs[0], Mapping)
        self.assertEqual(target.inputs[0]["privacy"], HASHED_MARKER)
        self.assertNotIn("private structured body", str(inspection.as_dict()))
        self.assertNotIn("document.txt", str(inspection.as_dict()))
        self.assertNotIn("Review the document.", str(inspection.as_dict()))

    async def test_duplicate_submission_reuses_queued_run(self) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = TextTarget()
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition()

        first = await client.enqueue(
            definition,
            input_value="same prompt",
            idempotency_key="same-window",
            owner_scope="same-owner",
        )
        second = await client.enqueue(
            definition,
            input_value="same prompt",
            idempotency_key="same-window",
            owner_scope="same-owner",
        )
        before_work = await queue.depth("default")
        processed = await worker.process_once()
        after_work = await queue.depth("default")
        output = await client.output(first.run.run_id)
        inspection = await client.inspect(first.run.run_id)

        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(second.run.run_id, first.run.run_id)
        self.assertIsNotNone(first.queue_item)
        self.assertIsNotNone(second.queue_item)
        assert first.queue_item is not None
        assert second.queue_item is not None
        self.assertEqual(
            second.queue_item.queue_item_id,
            first.queue_item.queue_item_id,
        )
        self.assertEqual(before_work.available, 1)
        self.assertEqual(before_work.active, 1)
        self.assertTrue(processed.processed)
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(after_work.active, 0)
        self.assertEqual(len(target.inputs), 1)
        assert isinstance(target.inputs[0], Mapping)
        self.assertEqual(target.inputs[0]["privacy"], HASHED_MARKER)
        self.assertEqual(len(inspection.attempts), 1)
        self.assertNotIn("same-window", str(inspection.as_dict()))
        self.assertNotIn("same-owner", str(inspection.as_dict()))
        self.assertNotIn("same prompt", str(inspection.as_dict()))

    async def test_explicit_queue_name_runs_on_matching_worker(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = TextTarget()
        client = _client(store, queue, target=target, clock=clock)
        default_worker = _worker(
            store,
            queue,
            target=target,
            clock=clock,
        )
        priority_worker = _worker(
            store,
            queue,
            target=target,
            queue_name="priority-documents",
            clock=clock,
        )

        submission = await client.enqueue(
            _definition(),
            input_value="private priority prompt",
            queue_name="priority-documents",
            queue_metadata={"tenant": "safe"},
        )
        default_depth = await queue.depth("default")
        priority_depth = await queue.depth("priority-documents")
        idle = await default_worker.process_once()
        processed = await priority_worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)
        final_default_depth = await queue.depth("default")
        final_priority_depth = await queue.depth("priority-documents")

        self.assertTrue(submission.created)
        self.assertIsNotNone(submission.queue_item)
        assert submission.queue_item is not None
        self.assertEqual(
            submission.queue_item.queue_name,
            "priority-documents",
        )
        self.assertEqual(
            submission.queue_item.metadata,
            {"tenant": "safe"},
        )
        self.assertEqual(
            submission.run.request.queue,
            "priority-documents",
        )
        self.assertEqual(default_depth.active, 0)
        self.assertEqual(priority_depth.available, 1)
        self.assertFalse(idle.processed)
        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(final_default_depth.active, 0)
        self.assertEqual(final_priority_depth.active, 0)
        self.assertEqual(len(target.inputs), 1)
        assert isinstance(target.inputs[0], Mapping)
        self.assertEqual(target.inputs[0]["privacy"], HASHED_MARKER)
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(
            inspection.attempts[0].state,
            TaskAttemptState.SUCCEEDED,
        )
        inspection_value = str(inspection.as_dict())
        self.assertIn("priority-documents", inspection_value)
        self.assertNotIn("private priority prompt", inspection_value)
        self.assertNotIn("tenant", inspection_value)

    async def test_duplicate_submission_after_completion_reuses_result(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = TextTarget()
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition()

        first = await client.enqueue(
            definition,
            input_value="same completed prompt",
            idempotency_key="same-completed-window",
            owner_scope="same-completed-owner",
        )
        processed = await worker.process_once()
        second = await client.enqueue(
            definition,
            input_value="same completed prompt",
            idempotency_key="same-completed-window",
            owner_scope="same-completed-owner",
        )
        output = await client.output(second.run.run_id)
        inspection = await client.inspect(second.run.run_id)
        depth = await queue.depth("default")

        self.assertTrue(first.created)
        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertFalse(second.created)
        self.assertEqual(second.run.run_id, first.run.run_id)
        self.assertEqual(second.run.state, TaskRunState.SUCCEEDED)
        self.assertIsNotNone(second.queue_item)
        assert second.queue_item is not None
        self.assertEqual(second.queue_item.state, TaskQueueItemState.DONE)
        self.assertTrue(output.ready)
        self.assertEqual(output.output_summary, {"privacy": REDACTED_MARKER})
        self.assertEqual(depth.active, 0)
        self.assertEqual(depth.dead, 0)
        self.assertEqual(len(target.inputs), 1)
        self.assertEqual(
            [attempt.state for attempt in inspection.attempts],
            [TaskAttemptState.SUCCEEDED],
        )
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("same completed prompt", inspection_value)
        self.assertNotIn("same-completed-window", inspection_value)
        self.assertNotIn("same-completed-owner", inspection_value)

    async def test_queued_attachment_requires_durable_reference(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = TextTarget()
        client = _client(store, queue, target=target, clock=clock)

        with self.assertRaises(TaskValidationError) as error:
            await client.enqueue(
                _definition(),
                input_value="safe prompt",
                files=(
                    TaskInputFile(
                        logical_path="volatile/private.txt",
                        media_type="text/plain",
                        size_bytes=7,
                        metadata={"filename": "private.txt"},
                    ),
                ),
            )
        depth = await queue.depth("default")

        self.assertEqual(len(error.exception.issues), 1)
        self.assertEqual(
            error.exception.issues[0].path,
            "files[0].artifact_ref",
        )
        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(queue.items, {})
        self.assertEqual(depth.active, 0)
        self.assertEqual(target.inputs, [])
        self.assertNotIn("private.txt", str(error.exception))
        self.assertNotIn("safe prompt", str(error.exception))

    async def test_durable_attachment_runs_through_queue_and_inspection(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=lambda: "explicit-input-e2e",
            )
            explicit_ref = await artifact_store.put(
                b"private explicit attachment",
                media_type="text/plain",
                metadata={"filename": "private.txt"},
            )
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            target = ReadingTarget()
            client = _client(
                store,
                queue,
                target=target,
                artifact_store=artifact_store,
                clock=clock,
            )
            worker = _worker(
                store,
                queue,
                target=target,
                artifact_store=artifact_store,
                clock=clock,
            )

            submission = await client.enqueue(
                _definition(),
                input_value="private prompt with attachment",
                files=(
                    TaskInputFile(
                        logical_path="provided/private.txt",
                        artifact_ref=explicit_ref,
                        media_type="text/plain",
                        size_bytes=explicit_ref.size_bytes,
                        metadata={"filename": "private.txt"},
                    ),
                ),
            )
            processed = await worker.process_once()
            output = await client.output(submission.run.run_id)
            inspection = await client.inspect(submission.run.run_id)
            artifacts = await store.list_artifacts(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(target.file_bodies, [b"private explicit attachment"])
        self.assertEqual(len(target.inputs), 1)
        assert isinstance(target.inputs[0], Mapping)
        self.assertEqual(target.inputs[0]["privacy"], HASHED_MARKER)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].artifact_id, "explicit-input-e2e")
        self.assertEqual(artifacts[0].purpose, TaskArtifactPurpose.INPUT)
        self.assertEqual(artifacts[0].state, TaskArtifactState.READY)
        self.assertIn("privacy", artifacts[0].ref.metadata)
        self.assertEqual(len(inspection.artifacts), 1)
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(
            inspection.attempts[0].state,
            TaskAttemptState.SUCCEEDED,
        )
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private explicit attachment", inspection_value)
        self.assertNotIn("private prompt with attachment", inspection_value)
        self.assertNotIn("private.txt", inspection_value)
        self.assertNotIn("private.txt", str(artifacts))

    async def test_output_artifact_task_runs_through_queue_and_inspection(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=lambda: "artifact-output-e2e",
            )
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            target = ArtifactOutputTarget()
            client = _client(
                store,
                queue,
                target=target,
                artifact_store=artifact_store,
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
                output_contract=TaskOutputContract.file(),
                artifact=TaskArtifactPolicy.references_only(
                    retention_days=5,
                ),
            )

            submission = await client.enqueue(
                definition,
                input_value="private artifact prompt",
            )
            processed = await worker.process_once()
            output = await client.wait(
                submission.run.run_id,
                timeout_seconds=0,
                poll_interval_seconds=0.01,
            )
            inspection = await client.inspect(submission.run.run_id)
            records = await store.list_artifacts(
                submission.run.run_id,
                purpose=TaskArtifactPurpose.OUTPUT,
            )
            reader = await artifact_store.open(records[0].ref)
            try:
                body = reader.read()
            finally:
                reader.close()

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertIsInstance(processed.output, TaskArtifactRef)
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        assert isinstance(output.output_summary, Mapping)
        self.assertEqual(output.output_summary["state"], "ready")
        self.assertEqual(body, b"private generated report")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].state, TaskArtifactState.READY)
        self.assertEqual(records[0].purpose, TaskArtifactPurpose.OUTPUT)
        self.assertEqual(records[0].retention.delete_after_days, 5)
        self.assertEqual(records[0].ref.metadata, {"privacy": "<redacted>"})
        self.assertEqual(len(inspection.artifacts), 1)
        artifact = inspection.artifacts[0]
        assert isinstance(artifact, Mapping)
        self.assertEqual(artifact["purpose"], TaskArtifactPurpose.OUTPUT.value)
        self.assertEqual(artifact["state"], TaskArtifactState.READY.value)
        self.assertNotIn("private generated report", str(inspection.as_dict()))
        self.assertNotIn("private artifact prompt", str(inspection.as_dict()))
        self.assertNotIn("report.txt", str(records))

    async def test_cancelled_queued_submission_is_not_claimed(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = TextTarget()
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)

        submission = await client.enqueue(
            _definition(),
            input_value="private cancelled prompt",
        )
        cancelled = await client.cancel(submission.run.run_id)
        idle = await worker.process_once()
        inspection = await client.inspect(submission.run.run_id)
        depth = await queue.depth("default")

        self.assertEqual(cancelled.state, TaskRunState.CANCEL_REQUESTED)
        self.assertFalse(idle.processed)
        self.assertEqual(inspection.run.state, TaskRunState.CANCEL_REQUESTED)
        self.assertEqual(inspection.attempts, ())
        self.assertEqual(depth.cancel_requested, 1)
        self.assertEqual(depth.available, 1)
        self.assertEqual(target.inputs, [])
        self.assertNotIn("private cancelled prompt", str(inspection.as_dict()))

    async def test_running_queue_task_cancellation_finalizes_safely(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        client_ref: list[TaskClient] = []

        async def cancel(run_id: str) -> object:
            return await client_ref[0].cancel(run_id)

        target = CancellingTarget(cancel)
        client = _client(store, queue, target=target, clock=clock)
        client_ref.append(client)
        worker = _worker(store, queue, target=target, clock=clock)

        submission = await client.enqueue(
            _definition(),
            input_value="private running prompt",
        )
        result = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)
        depth = await queue.depth("default")

        self.assertTrue(result.processed)
        self.assertIsNone(result.retry)
        self.assertIsNotNone(result.claimed)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.CANCELLED)
        self.assertEqual(inspection.run.state, TaskRunState.CANCELLED)
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(inspection.attempts[0].state, TaskAttemptState.FAILED)
        self.assertEqual(depth.active, 0)
        self.assertEqual(depth.dead, 1)
        self.assertEqual(len(target.inputs), 1)
        assert isinstance(target.inputs[0], Mapping)
        self.assertEqual(target.inputs[0]["privacy"], HASHED_MARKER)
        self.assertIn("cancellation", str(output.error))
        self.assertNotIn("private running prompt", str(inspection.as_dict()))

    async def test_scheduled_submission_waits_until_available(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = TextTarget()
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        available_at = clock.now + timedelta(seconds=30)

        submission = await client.enqueue(
            _definition(),
            input_value="scheduled prompt",
            available_at=available_at,
        )
        idle = await worker.process_once()
        scheduled_depth = await queue.depth("default")
        scheduled_health = await queue.health("default")
        with self.assertRaises(TaskClientWaitTimeoutError) as timeout:
            await client.wait(
                submission.run.run_id,
                timeout_seconds=0,
                poll_interval_seconds=0.01,
            )
        await clock.sleep(30)
        processed = await worker.process_once()
        output = await client.wait(
            submission.run.run_id,
            timeout_seconds=0,
            poll_interval_seconds=0.01,
        )
        ready_health = await queue.health("default")

        self.assertFalse(idle.processed)
        self.assertEqual(timeout.exception.run_id, submission.run.run_id)
        self.assertEqual(scheduled_depth.available, 0)
        self.assertEqual(scheduled_depth.scheduled, 1)
        self.assertEqual(scheduled_depth.active, 1)
        self.assertIsNone(scheduled_health.oldest_available_at)
        self.assertTrue(processed.processed)
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(target.inputs), 1)
        assert isinstance(target.inputs[0], Mapping)
        self.assertEqual(target.inputs[0]["privacy"], HASHED_MARKER)
        self.assertIsNone(ready_health.oldest_available_at)

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

    async def test_retry_exhaustion_records_safe_terminal_failure(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = FailingTarget()
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)

        submission = await client.enqueue(
            _definition(retry=TaskRetryPolicy(max_attempts=2)),
            input_value="private retry prompt",
        )
        retry = await worker.process_once()
        terminal = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)
        depth = await queue.depth("default")

        self.assertTrue(retry.processed)
        self.assertIsNotNone(retry.retry)
        assert retry.retry is not None
        self.assertTrue(retry.retry.retryable)
        self.assertEqual(retry.retry.run.state, TaskRunState.QUEUED)
        self.assertTrue(terminal.processed)
        self.assertIsNone(terminal.retry)
        self.assertIsNotNone(terminal.claimed)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(
            [attempt.state for attempt in inspection.attempts],
            [TaskAttemptState.FAILED, TaskAttemptState.FAILED],
        )
        self.assertEqual(depth.dead, 1)
        self.assertIn("infra", str(output.error))
        self.assertNotIn("private backend path", str(inspection.as_dict()))
        self.assertNotIn("customer-secret", str(inspection.as_dict()))
        self.assertNotIn("private retry prompt", str(inspection.as_dict()))

    async def test_invalid_structured_queue_input_is_rejected_safely(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = StructuredQueueTarget(
            (
                {
                    "status": "ready",
                    "count": 1,
                    "summary": "unused private summary",
                },
            )
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)

        with self.assertRaises(TaskValidationError) as error:
            await client.enqueue(
                _structured_definition(),
                input_value={
                    "prompt": "private structured prompt",
                    "limit": 0,
                },
                queue_metadata={"tenant": "safe"},
            )
        idle = await worker.process_once()
        depth = await queue.depth("default")

        self.assertEqual(len(error.exception.issues), 1)
        self.assertEqual(error.exception.issues[0].code, "input.invalid_type")
        self.assertEqual(error.exception.issues[0].path, "input")
        self.assertFalse(idle.processed)
        self.assertEqual(queue.items, {})
        self.assertEqual(depth.active, 0)
        self.assertEqual(target.inputs, [])
        error_value = str(error.exception)
        self.assertNotIn("private structured prompt", error_value)
        self.assertNotIn("unused private summary", error_value)
        self.assertNotIn("tenant", error_value)

    async def test_transient_structured_queue_failure_retries_to_success(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = StructuredQueueTarget(
            (
                OSError("private backend path /tmp/customer-secret.txt"),
                {
                    "status": "ready",
                    "count": 2,
                    "summary": "private final summary",
                },
            )
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)

        submission = await client.enqueue(
            _structured_definition(retry=TaskRetryPolicy(max_attempts=2)),
            input_value={
                "prompt": "private structured prompt",
                "limit": 2,
            },
        )
        retry = await worker.process_once()
        completed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)
        depth = await queue.depth("default")

        self.assertTrue(retry.processed)
        self.assertIsNotNone(retry.retry)
        assert retry.retry is not None
        self.assertTrue(retry.retry.retryable)
        self.assertEqual(retry.retry.run.state, TaskRunState.QUEUED)
        self.assertTrue(completed.processed)
        self.assertIsNotNone(completed.completion)
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            output.output_summary, {"status": "ready", "count": 2}
        )
        self.assertEqual(
            [attempt.state for attempt in inspection.attempts],
            [TaskAttemptState.FAILED, TaskAttemptState.SUCCEEDED],
        )
        first_error = cast(
            Mapping[str, object],
            inspection.attempts[0].result.error,
        )
        self.assertEqual(first_error["category"], "infra")
        self.assertEqual(first_error["code"], "infra.failure")
        self.assertEqual(len(inspection.events), 2)
        self.assertEqual(
            [event.sequence for event in inspection.events], [1, 2]
        )
        self.assertEqual(inspection.usage_totals.input_tokens, 3)
        self.assertEqual(inspection.usage_totals.output_tokens, 5)
        self.assertEqual(inspection.usage_totals.total_tokens, 8)
        self.assertEqual(depth.active, 0)
        self.assertEqual(depth.dead, 0)
        self.assertEqual(len(target.inputs), 2)
        for input_value in target.inputs:
            assert isinstance(input_value, Mapping)
            self.assertEqual(input_value["privacy"], HASHED_MARKER)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private backend path", inspection_value)
        self.assertNotIn("customer-secret", inspection_value)
        self.assertNotIn("private structured prompt", inspection_value)
        self.assertNotIn("private final summary", inspection_value)
        self.assertNotIn("private-token", inspection_value)
        self.assertNotIn("token_id", inspection_value)

    async def test_structured_queue_output_contract_failure_stays_safe(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = StructuredQueueTarget(
            (
                {
                    "status": "ready",
                    "count": "private invalid count",
                    "summary": "private invalid summary",
                },
            )
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)

        submission = await client.enqueue(
            _structured_definition(retry=TaskRetryPolicy(max_attempts=2)),
            input_value={
                "prompt": "private invalid prompt",
                "limit": 1,
            },
        )
        result = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)
        depth = await queue.depth("default")

        self.assertTrue(result.processed)
        self.assertIsNone(result.retry)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(inspection.attempts[0].state, TaskAttemptState.FAILED)
        error_summary = cast(Mapping[str, object], output.error)
        self.assertEqual(error_summary["category"], "output_contract")
        self.assertEqual(error_summary["code"], "output_contract.failed")
        details = cast(Mapping[str, object], error_summary["details"])
        issues = cast(tuple[Mapping[str, object], ...], details["issues"])
        self.assertEqual(issues[0]["code"], "output.invalid_type")
        self.assertEqual(issues[0]["path"], "output")
        self.assertEqual(len(inspection.events), 1)
        self.assertEqual(inspection.usage_totals.input_tokens, 1)
        self.assertEqual(inspection.usage_totals.output_tokens, 2)
        self.assertEqual(inspection.usage_totals.total_tokens, 3)
        self.assertEqual(depth.active, 0)
        self.assertEqual(depth.dead, 1)
        self.assertEqual(len(target.inputs), 1)
        assert isinstance(target.inputs[0], Mapping)
        self.assertEqual(target.inputs[0]["privacy"], HASHED_MARKER)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private invalid prompt", inspection_value)
        self.assertNotIn("private invalid count", inspection_value)
        self.assertNotIn("private invalid summary", inspection_value)
        self.assertNotIn("private-token", inspection_value)
        self.assertNotIn("token_id", inspection_value)

    async def test_queued_flow_scalar_input_runs_with_stored_json(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: f"{inputs[FLOW_TASK_INPUT_KEY]} done",
            )
        )
        target = FlowTaskTargetRunner(flow_resolver=lambda _: flow)
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
        )

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value="safe",
        )
        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertEqual(processed.output, "safe done")
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)

    async def test_queued_flow_object_input_validates_output_contract(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "status": "ready",
                    "count": inputs["limit"],
                    "summary": f"{inputs['prompt']} done",
                },
            )
        )
        target = FlowTaskTargetRunner(flow_resolver=lambda _: flow)
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _structured_definition(
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
        )

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value={"prompt": "safe", "limit": 2},
        )
        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            output.output_summary,
            {"status": "ready", "count": 2},
        )
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(
            inspection.attempts[0].state,
            TaskAttemptState.SUCCEEDED,
        )

    async def test_queued_flow_reserved_input_key_cannot_spoof_binding(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)

        def use_full_input(inputs: Mapping[str, object]) -> dict[str, object]:
            full_input = cast(
                Mapping[str, object],
                inputs[FLOW_TASK_INPUT_KEY],
            )
            return {
                "status": "ready",
                "limit": full_input["limit"],
                "reserved": full_input[FLOW_TASK_INPUT_KEY],
            }

        flow = Flow()
        flow.add_node(Node("A", func=use_full_input))
        target = FlowTaskTargetRunner(flow_resolver=lambda _: flow)
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(
            input_contract=TaskInputContract.object(
                schema={
                    "type": "object",
                    "required": [FLOW_TASK_INPUT_KEY, "limit"],
                    "additionalProperties": False,
                    "properties": {
                        FLOW_TASK_INPUT_KEY: {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                }
            ),
            output_contract=TaskOutputContract.object(
                schema={
                    "type": "object",
                    "required": ["status", "limit", "reserved"],
                    "additionalProperties": False,
                    "properties": {
                        "status": {"type": "string", "enum": ["ready"]},
                        "limit": {"type": "integer", "minimum": 1},
                        "reserved": {"type": "string"},
                    },
                }
            ),
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
        )

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value={
                FLOW_TASK_INPUT_KEY: "spoofed input",
                "limit": 2,
            },
        )
        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertEqual(
            processed.output,
            {
                "status": "ready",
                "limit": 2,
                "reserved": "spoofed input",
            },
        )
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(output.output_summary, {"status": "ready"})

    async def test_queued_flow_keeps_legacy_user_envelope_fields(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)

        def use_full_input(inputs: Mapping[str, object]) -> dict[str, object]:
            full_input = cast(
                Mapping[str, object],
                inputs[FLOW_TASK_INPUT_KEY],
            )
            return {
                "status": "ready",
                "limit": full_input["limit"],
                "privacy": full_input["privacy"],
                "value": full_input["value"],
            }

        flow = Flow()
        flow.add_node(Node("A", func=use_full_input))
        target = FlowTaskTargetRunner(flow_resolver=lambda _: flow)
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(
            input_contract=TaskInputContract.object(
                schema={
                    "type": "object",
                    "required": ["privacy", "value", "limit"],
                    "additionalProperties": False,
                    "properties": {
                        "privacy": {
                            "type": "string",
                            "enum": [STORED_MARKER],
                        },
                        "value": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                }
            ),
            output_contract=TaskOutputContract.object(
                schema={
                    "type": "object",
                    "required": [
                        "status",
                        "privacy",
                        "value",
                        "limit",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "status": {"type": "string", "enum": ["ready"]},
                        "privacy": {"type": "string"},
                        "value": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                }
            ),
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
        )

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value={
                "privacy": STORED_MARKER,
                "value": "safe envelope value",
                "limit": 2,
            },
        )
        processed = await worker.process_once()
        run = await store.get_run(submission.run.run_id)
        output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertEqual(
            processed.output,
            {
                "status": "ready",
                "privacy": STORED_MARKER,
                "value": "safe envelope value",
                "limit": 2,
            },
        )
        self.assertEqual(
            run.request.input_summary,
            {
                "privacy": STORED_MARKER,
                "value": "safe envelope value",
                "limit": 2,
            },
        )
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(output.output_summary, {"status": "ready"})

    async def test_queued_flow_rejects_unavailable_input_safely(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: "unused private output"))
        target = FlowTaskTargetRunner(flow_resolver=lambda _: flow)
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
            retry=TaskRetryPolicy(max_attempts=1),
        )

        submission = await client.enqueue(
            definition,
            input_value="private prompt",
        )
        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNone(processed.retry)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertIn("input_contract", str(output.error))
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private prompt", inspection_value)
        self.assertNotIn("unused private output", inspection_value)

    async def _enqueue_raw_input(
        self,
        store: InMemoryTaskStore,
        queue: InMemoryTaskQueue,
        definition: TaskDefinition,
        *,
        input_value: object,
    ) -> TaskQueueSubmission:
        await store.register_definition(
            definition,
            definition_hash="queue-worker-e2e",
        )
        return await queue.enqueue_run(
            TaskExecutionRequest(
                definition_id="queue-worker-e2e",
                input_summary=input_value,
                queue=definition.run.queue,
            ),
            queue_name=definition.run.queue or "default",
        )


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
    queue_name: str = "default",
    artifact_store: LocalArtifactStore | None = None,
    clock: Clock,
) -> TaskWorker:
    return TaskWorker(
        store,
        cast(TaskQueue, queue),
        target=target,
        worker_id="worker-1",
        queue_name=queue_name,
        artifact_store=artifact_store,
        clock=lambda: clock.now,
    )


def _structured_definition(
    *,
    retry: TaskRetryPolicy | None = None,
    execution: TaskExecutionTarget | None = None,
    observability: TaskObservabilityPolicy | None = None,
) -> TaskDefinition:
    return _definition(
        input_contract=TaskInputContract.object(
            schema={
                "type": "object",
                "required": ["prompt", "limit"],
                "additionalProperties": False,
                "properties": {
                    "prompt": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1},
                },
            }
        ),
        output_contract=TaskOutputContract.object(
            schema={
                "type": "object",
                "required": ["status", "count", "summary"],
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["ready"]},
                    "count": {"type": "integer", "minimum": 1},
                    "summary": {"type": "string", "minLength": 1},
                },
            }
        ),
        retry=retry,
        execution=execution,
        observability=observability,
    )


def _definition(
    *,
    input_contract: TaskInputContract | None = None,
    output_contract: TaskOutputContract | None = None,
    artifact: TaskArtifactPolicy | None = None,
    retry: TaskRetryPolicy | None = None,
    execution: TaskExecutionTarget | None = None,
    observability: TaskObservabilityPolicy | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="queue_worker_e2e", version="1"),
        input=input_contract or TaskInputContract.string(),
        output=output_contract or TaskOutputContract.text(),
        execution=execution or TaskExecutionTarget.agent("agent.toml"),
        run=TaskRunPolicy.queued("default"),
        artifact=artifact or TaskArtifactPolicy.references_only(),
        observability=observability or TaskObservabilityPolicy(),
        retry=retry or TaskRetryPolicy(max_attempts=2),
    )


if __name__ == "__main__":
    main()
