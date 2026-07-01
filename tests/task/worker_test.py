from asyncio import (
    CancelledError,
    Event,
    TimeoutError,
    create_task,
    sleep,
)
from asyncio import (
    Task as AsyncTask,
)
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.skill import (
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillReadLimits,
    SkillRegistry,
    SkillSourceConfig,
    SkillStatus,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
)
from avalan.task import (
    EncryptedPrivacyValue,
    ObservabilitySinkHealth,
    ObservabilitySinkType,
    PrivacyAction,
    PrivacySanitizer,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactState,
    TaskAttemptState,
    TaskDefinition,
    TaskExecutionPayload,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskFileDescriptor,
    TaskInputContract,
    TaskInputFile,
    TaskKeyPurpose,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskOutputParseError,
    TaskPrivacyPolicy,
    TaskProviderReferenceKind,
    TaskProviderStructuredOutputError,
    TaskQueueAbandonment,
    TaskQueueArtifact,
    TaskQueueClaim,
    TaskQueueCompletion,
    TaskQueueConflictError,
    TaskQueueDepth,
    TaskQueueHealth,
    TaskQueueItem,
    TaskQueueItemState,
    TaskQueueRetry,
    TaskQueueSubmission,
    TaskRetryPolicy,
    TaskRunPolicy,
    TaskRunState,
    TaskStoreConflictError,
    TaskTargetContext,
    TaskTargetRunner,
    TaskValidationCategory,
    TaskValidationContext,
    TaskValidationIssue,
    TaskWorker,
    TaskWorkerShutdown,
    UsageSource,
    UsageTotals,
)
from avalan.task.error import classify_task_error
from avalan.task.idempotency import TaskIdempotencyIdentity
from avalan.task.runner import (
    TaskExecutableInputFileEntry,
    task_execution_file_entries_value,
)
from avalan.task.skills import (
    build_task_skill_registry,
    task_definition_with_skills_identity,
)
from avalan.task.stores import InMemoryTaskStore
from avalan.task.worker import (
    _target_runner,
    _TaskWorkerShutdownRequested,
    _utc_now,
    _worker_id,
)


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


class MultiCallUsageResponse:
    def __init__(self, *responses: object) -> None:
        self.usage_responses = responses


class UsageTextOutput(str):
    _usage: object | None
    _usage_responses: tuple[object, ...]

    def __new__(
        cls,
        value: str,
        *,
        usage: object | None = None,
        usage_responses: tuple[object, ...] = (),
    ) -> "UsageTextOutput":
        output = str.__new__(cls, value)
        output._usage = usage
        output._usage_responses = usage_responses
        return output

    @property
    def usage(self) -> object | None:
        return self._usage

    @property
    def usage_responses(self) -> tuple[object, ...]:
        return self._usage_responses


class RecordingUsageSink:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.usage_totals: list[UsageTotals] = []

    async def record_event(self, event: object) -> None:
        self.events.append(event)

    async def record_usage(
        self,
        *,
        run_id: str,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.usage_totals.append(totals)

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(
            name="recording",
            event_count=len(self.events),
            usage_count=len(self.usage_totals),
        )

    @property
    def usage_event_count(self) -> int:
        return sum(
            1
            for event in self.events
            if type(event).__name__ == "SanitizedTaskUsageEvent"
        )


class WaitingTarget(FakeTarget):
    def __init__(self) -> None:
        super().__init__("done")
        self.started = Event()

    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        self.started.set()
        while True:
            await sleep(0)
            await context.check_cancelled()


class PassiveWaitingTarget(FakeTarget):
    def __init__(self) -> None:
        super().__init__("done")
        self.started = Event()

    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        self.started.set()
        while True:
            await sleep(0)


class ShutdownReturningTarget(FakeTarget):
    def __init__(self, shutdown: TaskWorkerShutdown) -> None:
        super().__init__("done")
        self.shutdown = shutdown

    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        self.shutdown.request()
        return "safe output"


class DelayedWaitShutdown(TaskWorkerShutdown):
    async def wait(self) -> None:
        await sleep(1)


class InvalidTarget(FakeTarget):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        return (
            TaskValidationIssue(
                code="target.invalid",
                path="execution",
                message="invalid target",
                hint="fix target",
                category=TaskValidationCategory.VALUE,
            ),
        )


class ArtifactOutputTarget(FakeTarget):
    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        return TaskArtifactRef(
            artifact_id="artifact-output-1",
            store="local",
            storage_key="output.txt",
            media_type="text/plain",
            size_bytes=7,
            metadata={"filename": "private-output.txt"},
        )


class UsageTarget(FakeTarget):
    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        await context.observe_usage(
            SimpleNamespace(input_token_count=3, output_token_count=5)
        )
        return self.output


class PartiallyObservedUsageWrapperTarget(FakeTarget):
    def __init__(self) -> None:
        super().__init__("done")
        self.first_response = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": 4,
                "cached_input_tokens": 1,
                "output_tokens": 6,
                "total_tokens": 10,
                "raw_response_id": "private-first-response",
            },
        )
        self.second_response = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": 5,
                "cache_creation_input_tokens": 2,
                "output_tokens": 7,
                "reasoning_tokens": 3,
                "total_tokens": 12,
                "raw_response_id": "private-second-response",
            },
        )
        self.malformed_response = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": "private prompt",
                "output_tokens": -1,
                "total_tokens": True,
                "raw_response_body": "private provider body",
            },
        )

    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        await context.observe_usage(self.first_response)
        return UsageTextOutput(
            "done",
            usage_responses=(
                self.first_response,
                self.second_response,
                self.malformed_response,
            ),
        )


class StaticEncryptionProvider:
    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        _ = purpose
        return EncryptedPrivacyValue(
            ciphertext=b"encrypted:" + value,
            key_id=key_id or "raw-value",
            algorithm="test-aead",
            metadata=context,
        )

    def decrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        algorithm: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        _ = purpose, key_id, algorithm, context
        prefix = b"encrypted:"
        assert value.startswith(prefix)
        return value[len(prefix) :]


class FakeQueue:
    def __init__(self, store: InMemoryTaskStore, now: datetime) -> None:
        self.store = store
        self.now = now
        self.item: TaskQueueItem | None = None
        self.completed: TaskQueueCompletion | None = None
        self.retried: TaskQueueRetry | None = None
        self.abandoned: TaskQueueAbandonment | None = None
        self.heartbeats: list[datetime] = []
        self.heartbeat_error: BaseException | None = None
        self.heartbeat_shutdown: TaskWorkerShutdown | None = None
        self.abandon_after_claim = False

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
        claim = TaskQueueClaim(queue_item=self.item, run=run, attempt=attempt)
        if self.abandon_after_claim:
            await self.abandon(
                self.item.queue_item_id,
                claim_token=claim_token,
                max_attempts=2,
                now=now or self.now,
                metadata=metadata,
            )
        return claim

    async def heartbeat(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        lease_expires_at: datetime,
        now: datetime | None = None,
    ) -> TaskQueueItem:
        if self.heartbeat_error is not None:
            raise self.heartbeat_error
        assert self.item is not None
        assert self.item.queue_item_id == queue_item_id
        assert self.item.claim_token == claim_token
        heartbeat_at = now or self.now
        self.heartbeats.append(heartbeat_at)
        if self.heartbeat_shutdown is not None:
            self.heartbeat_shutdown.request()
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=self.item.run_id,
            queue_name=self.item.queue_name,
            state=self.item.state,
            priority=self.item.priority,
            available_at=self.item.available_at,
            attempts=self.item.attempts,
            created_at=self.item.created_at,
            updated_at=heartbeat_at,
            run_state=self.item.run_state,
            claimed_at=self.item.claimed_at,
            lease_expires_at=lease_expires_at,
            worker_id=self.item.worker_id,
            claim_token=self.item.claim_token,
            heartbeat_at=heartbeat_at,
            metadata=self.item.metadata,
        )
        return self.item

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
        current_run = await self.store.get_run(self.item.run_id)
        attempt_id = current_run.last_attempt_id
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
            from_states={current_run.state},
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

    async def abandon(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        max_attempts: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueAbandonment:
        assert self.item is not None
        assert self.item.queue_item_id == queue_item_id
        current_run = await self.store.get_run(self.item.run_id)
        attempt_id = current_run.last_attempt_id
        attempt = await self.store.transition_attempt(
            attempt_id or "",
            from_states={
                TaskAttemptState.CREATED,
                TaskAttemptState.RUNNING,
            },
            to_state=TaskAttemptState.ABANDONED,
            reason="abandoned",
            claim_token=claim_token,
            metadata=metadata,
        )
        cancel_requested = current_run.state == TaskRunState.CANCEL_REQUESTED
        retryable = (
            attempt.attempt_number < max_attempts and not cancel_requested
        )
        run_state = (
            TaskRunState.CANCELLED
            if cancel_requested
            else TaskRunState.QUEUED if retryable else TaskRunState.FAILED
        )
        run = await self.store.transition_run(
            self.item.run_id,
            from_states={current_run.state},
            to_state=run_state,
            reason="abandoned",
            claim_token=claim_token,
            metadata=metadata,
        )
        if retryable:
            run = replace(run, claim=None)
            self.store._runs[run.run_id] = run
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=run.run_id,
            queue_name=self.item.queue_name,
            state=(
                TaskQueueItemState.AVAILABLE
                if retryable
                else TaskQueueItemState.DEAD
            ),
            priority=self.item.priority,
            available_at=now or self.now,
            attempts=self.item.attempts,
            created_at=self.item.created_at,
            updated_at=now or self.now,
            run_state=run.state,
        )
        self.abandoned = TaskQueueAbandonment(
            queue_item=self.item,
            run=run,
            attempt=attempt,
        )
        return self.abandoned

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

    async def _use_definition(self, definition: TaskDefinition) -> None:
        self.definition = definition
        self.store = InMemoryTaskStore(clock=lambda: self.now)
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
        )
        run = await self.store.create_run(
            TaskExecutionRequest(
                definition_id="hash-a",
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

    async def _use_request(self, request: TaskExecutionRequest) -> None:
        self.store = InMemoryTaskStore(clock=lambda: self.now)
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
        )
        run = await self.store.create_run(request)
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

    async def _claim(self) -> TaskQueueClaim:
        claim = await self.queue.claim(
            "default",
            worker_id="worker-1",
            lease_expires_at=self.now.replace(hour=1),
            now=self.now,
            metadata={"worker_id": "worker-1"},
        )
        assert claim is not None
        return claim

    async def _skills_failure_code(
        self,
        definition: TaskDefinition,
        *,
        skills_settings: TrustedSkillSettings | None = None,
        skills_registry: SkillRegistry | None = None,
    ) -> str:
        await self._use_definition(
            replace(definition, retry=TaskRetryPolicy(max_attempts=1))
        )
        target = FakeTarget("unused")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            skills_settings=skills_settings,
            skills_registry=skills_registry,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertEqual(target.contexts, [])
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        assert self.queue.completed.run.result is not None
        error = self.queue.completed.run.result.error
        assert isinstance(error, Mapping)
        details = error.get("details")
        if isinstance(details, Mapping):
            issues = details.get("issues")
            if isinstance(issues, list | tuple) and issues:
                issue = issues[0]
                if isinstance(issue, Mapping):
                    return cast(str, issue["code"])
        return cast(str, error["code"])

    def _target_context(self, claim: TaskQueueClaim) -> TaskTargetContext:
        async def check_cancelled() -> None:
            return None

        return TaskTargetContext(
            definition=self.definition,
            execution=claim.attempt.context,
            input_value={},
            cancellation_checker=check_cancelled,
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
        self.assertIsNone(target.contexts[0].input_value)

    async def test_process_once_revalidates_matching_skills_identity(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(root)
            await self._use_definition(await _definition_with_skills(settings))
            target = FakeTarget("safe output")
            worker = TaskWorker(
                self.store,
                cast(object, self.queue),
                target=target,
                worker_id="worker-1",
                skills_settings=settings,
                clock=lambda: self.now,
            )

            result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        assert target.contexts[0].definition.skills_identity is not None
        self.assertEqual(
            target.contexts[0].definition.skills_identity["status"],
            SkillStatus.OK.value,
        )

    async def test_process_once_fails_closed_on_missing_skills_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(root)
            definition = await _definition_with_skills(settings)

            code = await self._skills_failure_code(definition)

        self.assertEqual(code, "task.skills_registry_missing")

    async def test_process_once_fails_closed_on_stale_skills_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            skill_path = root / "pdf" / "SKILL.md"
            _write_skill(skill_path, body="# Body\nFIRST\n")
            settings = _trusted_skills(root)
            definition = await _definition_with_skills(settings)
            _write_skill(skill_path, body="# Body\nSECOND\n")

            code = await self._skills_failure_code(
                definition,
                skills_settings=settings,
            )

        self.assertEqual(code, "task.skills_registry_stale")

    async def test_process_once_fails_closed_on_unavailable_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(root)
            definition = await _definition_with_skills(settings)
            missing_settings = _trusted_skills(Path(directory) / "missing")

            code = await self._skills_failure_code(
                definition,
                skills_settings=missing_settings,
            )

        self.assertEqual(code, "task.skills_registry_unavailable")

    async def test_process_once_fails_closed_on_widened_registry(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            restricted = _trusted_skills(
                root,
                read_limits=SkillReadLimits(max_lines_per_read=20),
            )
            widened = _trusted_skills(root)
            definition = await _definition_with_skills(restricted)

            code = await self._skills_failure_code(
                definition,
                skills_settings=widened,
            )

        self.assertEqual(code, "task.skills_registry_widened")

    async def test_process_once_fails_closed_on_malformed_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(root)
            definition = await _definition_with_skills(settings)
            registry = await _registry_with_status(
                settings,
                SkillStatus.MALFORMED,
            )

            code = await self._skills_failure_code(
                definition,
                skills_registry=registry,
            )

        self.assertEqual(code, "task.skills_registry_malformed")

    async def test_process_once_fails_closed_on_policy_denied_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(root)
            definition = await _definition_with_skills(settings)
            registry = await _registry_with_status(
                settings,
                SkillStatus.POLICY_DENIED,
            )

            code = await self._skills_failure_code(
                definition,
                skills_registry=registry,
            )

        self.assertEqual(code, "task.skills_registry_policy_denied")

    async def test_process_once_uses_encrypted_execution_payload(self) -> None:
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(raw_retention_days=1),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
        )
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<redacted>"},
                input_payload=TaskExecutionPayload(
                    input_value=sanitizer.sanitize_with_action(
                        PrivacyAction.ENCRYPT,
                        "private prompt",
                    ),
                ),
                queue="default",
            )
        )
        target = FakeTarget("safe output")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            encryption_provider=StaticEncryptionProvider(),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        self.assertEqual(target.contexts[0].input_value, "private prompt")
        self.assertNotIn("private prompt", str(result.completion))

    async def test_process_once_uses_encrypted_file_payload(self) -> None:
        await self._use_definition(
            _definition(input_contract=TaskInputContract.file())
        )
        descriptor = TaskFileDescriptor.provider_reference_descriptor(
            "file-private",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
            mime_type="application/pdf",
            owner_scope="tenant-a",
            identity_hmac="hmac-value",
        )
        assert descriptor.provider_reference is not None
        entry = TaskExecutableInputFileEntry(
            file=TaskInputFile(
                logical_path="provider:openai:provider_file_id",
                provider_reference=descriptor.provider_reference,
                media_type="application/pdf",
            )
        )
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(raw_retention_days=1),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
        )
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_payload=TaskExecutionPayload(
                    file_values=(
                        sanitizer.sanitize_with_action(
                            PrivacyAction.ENCRYPT,
                            task_execution_file_entries_value((entry,))[0],
                        ),
                    ),
                    input_value=None,
                ),
                queue="default",
            )
        )
        target = FakeTarget("safe output")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            encryption_provider=StaticEncryptionProvider(),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        self.assertEqual(len(target.contexts[0].files), 1)
        file = target.contexts[0].files[0]
        self.assertIsNotNone(file.provider_reference)
        assert file.provider_reference is not None
        self.assertEqual(file.provider_reference.reference, "file-private")
        self.assertNotIn("file-private", str(result.completion))

    async def test_process_once_fails_without_file_payload_decryption(
        self,
    ) -> None:
        await self._use_definition(
            _definition(input_contract=TaskInputContract.file())
        )
        entry = TaskExecutableInputFileEntry(
            file=TaskInputFile(
                logical_path="artifact:input-1",
                artifact_ref=TaskArtifactRef(
                    artifact_id="input-1",
                    store="local",
                    storage_key="private/input-1",
                ),
            )
        )
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(raw_retention_days=1),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
        )
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_payload=TaskExecutionPayload(
                    file_values=(
                        sanitizer.sanitize_with_action(
                            PrivacyAction.ENCRYPT,
                            task_execution_file_entries_value((entry,))[0],
                        ),
                    ),
                    input_value=None,
                ),
                queue="default",
            )
        )
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
        self.assertEqual(target.contexts, [])
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertNotIn("private/input-1", str(self.queue.completed))

    async def test_process_once_fails_without_execution_payload(self) -> None:
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<redacted>"},
                queue="default",
            )
        )
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
        self.assertIsNone(result.retry)
        self.assertEqual(target.contexts, [])
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        rendered_result = str(self.queue.completed.run.result)
        self.assertIn("privacy.failure", rendered_result)
        self.assertNotIn("<redacted>", rendered_result)

    async def test_process_once_fails_when_payload_has_no_scalar_input(
        self,
    ) -> None:
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<redacted>"},
                input_payload=TaskExecutionPayload(
                    file_values=(),
                    input_value=None,
                ),
                queue="default",
            )
        )
        target = FakeTarget("safe output")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            encryption_provider=StaticEncryptionProvider(),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertEqual(target.contexts, [])
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertNotIn("<redacted>", str(self.queue.completed.run.result))

    async def test_process_once_fails_without_decryption_provider(
        self,
    ) -> None:
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(raw_retention_days=1),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
        )
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<redacted>"},
                input_payload=TaskExecutionPayload(
                    input_value=sanitizer.sanitize_with_action(
                        PrivacyAction.ENCRYPT,
                        "private prompt",
                    ),
                ),
                queue="default",
            )
        )
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
        self.assertEqual(target.contexts, [])
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertNotIn("private prompt", str(self.queue.completed))

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

    async def test_process_once_completes_terminal_failure(self) -> None:
        await self._use_definition(
            _definition(retry=TaskRetryPolicy(max_attempts=1))
        )
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
        self.assertIsNone(result.retry)
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertNotIn("private", str(self.queue.completed.run.result))

    async def test_process_once_completes_cancelled_failure(self) -> None:
        target = FakeTarget()
        target.run = _raise_cancelled
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(
            self.queue.completed.run.state,
            TaskRunState.CANCELLED,
        )

    async def test_process_once_records_output_artifacts(self) -> None:
        await self._use_definition(
            _definition(
                output_contract=TaskOutputContract.file(),
                artifact=TaskArtifactPolicy.references_only(retention_days=3),
            )
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=ArtifactOutputTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        records = await self.store.list_artifacts(
            self.run.run_id,
            purpose=TaskArtifactPurpose.OUTPUT,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].state, TaskArtifactState.READY)
        self.assertEqual(records[0].retention.delete_after_days, 3)
        self.assertEqual(records[0].ref.metadata, {"privacy": "<redacted>"})
        self.assertNotIn("private-output", str(records))

    async def test_process_once_records_usage_observations(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=UsageTarget("safe output"),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        records = await self.store.list_usage(self.run.run_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].totals.input_tokens, 3)
        self.assertEqual(records[0].totals.output_tokens, 5)

    async def test_process_once_records_returned_output_usage(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(
                UsageTextOutput(
                    "safe output",
                    usage={
                        "input_tokens": 7,
                        "cached_input_tokens": 2,
                        "output_tokens": 4,
                        "reasoning_tokens": 1,
                        "total_tokens": 11,
                        "provider_family": "openai",
                    },
                )
            ),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()
        records = await self.store.list_usage(self.run.run_id)
        totals = await self.store.usage_totals(self.run.run_id)

        self.assertIsNotNone(result.completion)
        self.assertEqual(result.output, "safe output")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, UsageSource.EXACT)
        self.assertEqual(records[0].totals.input_tokens, 7)
        self.assertEqual(records[0].totals.cached_input_tokens, 2)
        self.assertEqual(records[0].totals.output_tokens, 4)
        self.assertEqual(records[0].totals.reasoning_tokens, 1)
        self.assertEqual(records[0].totals.total_tokens, 11)
        self.assertEqual(records[0].metadata, {"provider_family": "openai"})
        self.assertEqual(totals.total_tokens, 11)

    async def test_process_once_turns_output_validation_into_failure(
        self,
    ) -> None:
        await self._use_definition(
            _definition(
                output_contract=TaskOutputContract.object(
                    schema={"type": "object"}
                ),
                retry=TaskRetryPolicy(max_attempts=1),
            )
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget("not an object"),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        assert self.queue.completed.run.result is not None
        self.assertEqual(
            self.queue.completed.run.result.error["code"],
            "output_contract.failed",
        )

    async def test_process_once_records_usage_before_output_failure(
        self,
    ) -> None:
        await self._use_definition(
            _definition(
                output_contract=TaskOutputContract.object(
                    schema={"type": "object"}
                ),
                retry=TaskRetryPolicy(max_attempts=1),
            )
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(
                UsageTextOutput(
                    "private invalid output",
                    usage={
                        "input_tokens": 2,
                        "cache_creation_input_tokens": 1,
                        "output_tokens": 3,
                        "total_tokens": 5,
                        "provider_family": "openai",
                        "raw_response_id": "private-response-id",
                    },
                )
            ),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()
        records = await self.store.list_usage(self.run.run_id)

        self.assertTrue(result.processed)
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, UsageSource.EXACT)
        self.assertEqual(records[0].totals.input_tokens, 2)
        self.assertEqual(records[0].totals.cache_creation_input_tokens, 1)
        self.assertEqual(records[0].totals.total_tokens, 5)
        self.assertNotIn("raw_response_id", records[0].metadata)
        self.assertNotIn(
            "private invalid output",
            str(self.queue.completed.run.result),
        )

    async def test_process_once_classifies_structured_output_failures(
        self,
    ) -> None:
        cases = (
            (
                TaskProviderStructuredOutputError(),
                "provider",
                "provider.structured_output_failed",
            ),
            (
                TaskOutputParseError(),
                "output_contract",
                "output.parse_failed",
            ),
        )

        for error, category, code in cases:
            with self.subTest(code=code):
                await self._use_definition(
                    _definition(retry=TaskRetryPolicy(max_attempts=1))
                )
                target = FakeTarget()

                async def fail(context: TaskTargetContext) -> object:
                    _ = context
                    raise error

                target.run = fail
                worker = TaskWorker(
                    self.store,
                    cast(object, self.queue),
                    target=target,
                    worker_id="worker-1",
                    clock=lambda: self.now,
                )

                result = await worker.process_once()

                self.assertTrue(result.processed)
                self.assertIsNotNone(self.queue.completed)
                assert self.queue.completed is not None
                self.assertEqual(
                    self.queue.completed.run.state,
                    TaskRunState.FAILED,
                )
                assert self.queue.completed.run.result is not None
                summary = cast(
                    Mapping[str, object],
                    self.queue.completed.run.result.error,
                )
                self.assertEqual(summary["category"], category)
                self.assertEqual(summary["code"], code)
                self.assertNotIn("private", str(summary))

    async def test_process_once_finalizes_invalid_target(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=InvalidTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNone(result.retry)
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertEqual(
            self.queue.completed.attempt.state,
            TaskAttemptState.FAILED,
        )
        assert self.queue.completed.run.result is not None
        self.assertEqual(
            self.queue.completed.run.result.error["code"],
            "runnable.failed",
        )

    async def test_process_once_skips_claim_after_shutdown_request(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()
        shutdown.request()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertFalse(result.processed)
        self.assertTrue(result.shutdown_requested)
        assert self.queue.item is not None
        self.assertEqual(self.queue.item.state, TaskQueueItemState.AVAILABLE)

    async def test_process_once_stops_after_claim_expires_before_start(
        self,
    ) -> None:
        self.queue.abandon_after_claim = True
        target = FakeTarget()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.completion)
        self.assertIsNone(result.retry)
        self.assertIsNone(result.abandonment)
        self.assertEqual(target.contexts, [])
        assert self.queue.item is not None
        self.assertEqual(self.queue.item.state, TaskQueueItemState.AVAILABLE)
        self.assertEqual(
            (await self.store.get_run(self.run.run_id)).state,
            TaskRunState.QUEUED,
        )
        attempts = await self.store.list_attempts(self.run.run_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].state, TaskAttemptState.ABANDONED)

    async def test_process_once_sanitizes_start_claim_conflict(
        self,
    ) -> None:
        target = FakeTarget()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        with patch.object(
            worker,
            "_start_claimed_attempt",
            side_effect=TaskStoreConflictError("private stale start"),
        ):
            result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertEqual(target.contexts, [])
        self.assertNotIn("private stale start", str(result))

    async def test_rejects_heartbeat_interval_not_shorter_than_lease(
        self,
    ) -> None:
        for heartbeat_seconds in (30, 30.0, 31):
            with (
                self.subTest(heartbeat_seconds=heartbeat_seconds),
                self.assertRaisesRegex(
                    AssertionError,
                    "heartbeat_seconds must be shorter than lease_seconds",
                ),
            ):
                TaskWorker(
                    self.store,
                    cast(object, self.queue),
                    target=FakeTarget(),
                    worker_id="worker-1",
                    lease_seconds=30,
                    heartbeat_seconds=heartbeat_seconds,
                    clock=lambda: self.now,
                )

    async def test_process_once_abandons_active_shutdown_for_reclaim(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()

        class ShutdownTarget(FakeTarget):
            async def run(self, context: TaskTargetContext) -> object:
                self.contexts.append(context)
                shutdown.request()
                await context.check_cancelled()
                return "unreachable"

        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=ShutdownTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.shutdown_requested)
        self.assertIsNotNone(result.abandonment)
        assert result.abandonment is not None
        self.assertIsNone(self.queue.completed)
        self.assertTrue(result.abandonment.retryable)
        self.assertEqual(
            result.abandonment.run.state,
            TaskRunState.QUEUED,
        )
        self.assertEqual(
            result.abandonment.attempt.state,
            TaskAttemptState.ABANDONED,
        )
        assert self.queue.item is not None
        self.assertEqual(self.queue.item.state, TaskQueueItemState.AVAILABLE)

    async def test_process_once_shutdown_abandon_can_exhaust_attempts(
        self,
    ) -> None:
        await self._use_definition(
            _definition(retry=TaskRetryPolicy(max_attempts=1))
        )
        shutdown = TaskWorkerShutdown()

        class ShutdownTarget(FakeTarget):
            async def run(self, context: TaskTargetContext) -> object:
                self.contexts.append(context)
                shutdown.request()
                await context.check_cancelled()
                return "unreachable"

        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=ShutdownTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.abandonment)
        assert result.abandonment is not None
        self.assertFalse(result.abandonment.retryable)
        self.assertEqual(result.abandonment.run.state, TaskRunState.FAILED)
        self.assertEqual(
            result.abandonment.queue_item.state,
            TaskQueueItemState.DEAD,
        )

    async def test_process_once_abandons_shutdown_after_target_return(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()
        target = ShutdownReturningTarget(shutdown)
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.shutdown_requested)
        self.assertIsNotNone(result.abandonment)
        assert result.abandonment is not None
        self.assertIsNone(result.completion)
        self.assertIsNone(self.queue.completed)
        self.assertEqual(
            result.abandonment.run.state,
            TaskRunState.QUEUED,
        )
        self.assertIsNone(result.abandonment.run.result)
        self.assertEqual(
            result.abandonment.attempt.state,
            TaskAttemptState.ABANDONED,
        )
        assert self.queue.item is not None
        self.assertEqual(self.queue.item.state, TaskQueueItemState.AVAILABLE)
        self.assertEqual(len(target.contexts), 1)

    async def test_process_once_stops_heartbeat_on_shutdown(self) -> None:
        shutdown = TaskWorkerShutdown()
        self.queue.heartbeat_shutdown = shutdown
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=WaitingTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            heartbeat_seconds=0.001,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertEqual(len(self.queue.heartbeats), 1)
        self.assertIsNotNone(result.abandonment)
        assert result.abandonment is not None
        self.assertEqual(
            result.abandonment.run.state,
            TaskRunState.QUEUED,
        )
        self.assertEqual(
            result.abandonment.attempt.state,
            TaskAttemptState.ABANDONED,
        )

    async def test_process_once_stops_after_heartbeat_failure(
        self,
    ) -> None:
        await self._use_definition(
            _definition(retry=TaskRetryPolicy(max_attempts=1))
        )
        self.queue.heartbeat_error = RuntimeError("private heartbeat")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=WaitingTarget(),
            worker_id="worker-1",
            heartbeat_seconds=0.001,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.completion)
        self.assertIsNone(result.retry)
        self.assertIsNone(result.abandonment)
        self.assertIsNone(self.queue.completed)
        self.assertIsNone(self.queue.retried)
        self.assertIsNone(self.queue.abandoned)
        self.assertEqual(
            (await self.store.get_run(self.run.run_id)).state,
            TaskRunState.RUNNING,
        )
        attempts = await self.store.list_attempts(self.run.run_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].state, TaskAttemptState.RUNNING)
        self.assertNotIn("private heartbeat", str(result))

    async def test_process_once_stops_after_heartbeat_claim_conflict(
        self,
    ) -> None:
        self.queue.heartbeat_error = TaskQueueConflictError(
            "private stale claim"
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=WaitingTarget(),
            worker_id="worker-1",
            heartbeat_seconds=0.001,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.completion)
        self.assertIsNone(result.retry)
        self.assertIsNone(result.abandonment)
        self.assertIsNone(self.queue.completed)
        self.assertIsNone(self.queue.retried)
        self.assertIsNone(self.queue.abandoned)
        self.assertEqual(
            (await self.store.get_run(self.run.run_id)).state,
            TaskRunState.RUNNING,
        )
        attempts = await self.store.list_attempts(self.run.run_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].state, TaskAttemptState.RUNNING)
        self.assertNotIn("private stale claim", str(result))

    async def test_process_once_stops_after_shutdown_finalize_conflict(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        with (
            patch.object(
                worker,
                "_execute",
                side_effect=_TaskWorkerShutdownRequested(),
            ),
            patch.object(
                worker,
                "_finalize_shutdown",
                side_effect=TaskQueueConflictError("private shutdown claim"),
            ),
        ):
            result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.shutdown_requested)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.abandonment)
        self.assertNotIn("private shutdown claim", str(result))

    async def test_process_once_stops_after_failure_finalize_conflict(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        with (
            patch.object(
                worker,
                "_execute",
                side_effect=RuntimeError("private target failure"),
            ),
            patch.object(
                worker,
                "_finalize_failure",
                side_effect=TaskQueueConflictError("private retry claim"),
            ),
        ):
            result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.retry)
        self.assertNotIn("private retry claim", str(result))

    async def test_process_once_stops_after_success_complete_conflict(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        with (
            patch.object(worker, "_execute", return_value="safe output"),
            patch.object(
                worker,
                "_complete_success",
                side_effect=TaskQueueConflictError("private complete claim"),
            ),
        ):
            result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.completion)
        self.assertNotIn("private complete claim", str(result))

    async def test_run_target_timeout_cancels_target_and_shutdown_watcher(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=WaitingTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )
        claim = await self._claim()

        with self.assertRaises(TimeoutError):
            await worker._run_target(
                self._target_context(claim),
                claim=claim,
                timeout=0.001,
            )

        self.assertFalse(shutdown.requested)

    async def test_run_target_shutdown_cancels_running_target(self) -> None:
        shutdown = TaskWorkerShutdown()
        target = PassiveWaitingTarget()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )
        claim = await self._claim()
        running = create_task(
            worker._run_target(
                self._target_context(claim),
                claim=claim,
                timeout=1,
            )
        )

        await target.started.wait()
        shutdown.request()

        with self.assertRaises(_TaskWorkerShutdownRequested):
            await running

    async def test_run_target_shutdown_wins_before_waiter_completes(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=ShutdownReturningTarget(shutdown),
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )
        claim = await self._claim()

        with (
            patch("avalan.task.worker.wait", new=_target_only_wait),
            self.assertRaises(_TaskWorkerShutdownRequested),
        ):
            await worker._run_target(
                self._target_context(claim),
                claim=claim,
                timeout=1,
            )

    async def test_run_target_heartbeat_shutdown_cancels_running_target(
        self,
    ) -> None:
        shutdown = DelayedWaitShutdown()
        shutdown.request()
        target = PassiveWaitingTarget()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            shutdown=shutdown,
            heartbeat_seconds=0.001,
            clock=lambda: self.now,
        )
        claim = await self._claim()

        with self.assertRaises(_TaskWorkerShutdownRequested):
            await worker._run_target(
                self._target_context(claim),
                claim=claim,
                timeout=1,
            )

    async def test_run_target_heartbeat_failure_is_sanitized(
        self,
    ) -> None:
        cases = (
            TaskQueueConflictError("private stale claim"),
            RuntimeError("private heartbeat outage"),
        )
        for heartbeat_error in cases:
            with self.subTest(error=type(heartbeat_error).__name__):
                await self._use_definition(_definition())
                self.queue.heartbeat_error = heartbeat_error
                worker = TaskWorker(
                    self.store,
                    cast(object, self.queue),
                    target=PassiveWaitingTarget(),
                    worker_id="worker-1",
                    heartbeat_seconds=0.001,
                    clock=lambda: self.now,
                )
                claim = await self._claim()

                with self.assertRaises(TaskQueueConflictError) as error:
                    await worker._run_target(
                        self._target_context(claim),
                        claim=claim,
                        timeout=1,
                    )

                self.assertEqual(
                    str(error.exception),
                    "task queue heartbeat failed",
                )

    async def test_run_target_returns_with_heartbeat_enabled(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget("safe output"),
            worker_id="worker-1",
            heartbeat_seconds=30,
            clock=lambda: self.now,
        )
        claim = await self._claim()

        output = await worker._run_target(
            self._target_context(claim),
            claim=claim,
            timeout=1,
        )

        self.assertEqual(output, "safe output")
        self.assertEqual(self.queue.heartbeats, [])

    async def test_heartbeat_claim_returns_after_shutdown_request(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            heartbeat_seconds=0.001,
            clock=lambda: self.now,
        )
        claim = await self._claim()
        shutdown.request()

        await worker._heartbeat_claim(claim)

        self.assertEqual(self.queue.heartbeats, [])

    async def test_check_cancelled_raises_cancelled_error(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        await self.store.transition_run(
            self.run.run_id,
            from_states={TaskRunState.QUEUED},
            to_state=TaskRunState.CANCEL_REQUESTED,
            reason="cancel requested",
        )

        with self.assertRaises(CancelledError):
            await worker._check_cancelled(self.run.run_id)

    async def test_helper_branches(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=_callable_target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        self.assertIsNone(
            worker._event_pipeline(
                _definition(observability=TaskObservabilityPolicy.noop()),
                run=self.run,
                attempt=await self.store.create_attempt(self.run.run_id),
                sanitizer=PrivacySanitizer(TaskPrivacyPolicy()),
            )
        )
        self.assertIsNone(
            worker._observability_sink_for(
                _definition(
                    observability=TaskObservabilityPolicy(
                        sinks=(ObservabilitySinkType.NOOP,),
                        metrics=False,
                        trace=False,
                        capture_events=False,
                    )
                )
            )
        )
        summary = worker._safe_task_error_summary(
            PrivacySanitizer(
                TaskPrivacyPolicy(errors=PrivacyAction.HASH),
            ),
            classify_task_error(RuntimeError("private")),
        )
        self.assertEqual(summary["privacy"], "<redacted>")
        self.assertIsNotNone(_target_runner(_callable_target))
        self.assertIsInstance(_utc_now(), datetime)
        self.assertTrue(_worker_id().startswith("worker-"))

    async def test_record_usage_handles_missing_and_failed_store_records(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        attempt = await self.store.create_attempt(self.run.run_id)

        await worker._record_usage(
            object(),
            definition=self.definition,
            run=self.run,
            attempt=attempt,
        )
        with patch.object(
            self.store,
            "append_usage",
            side_effect=RuntimeError("private telemetry failure"),
        ):
            await worker._record_usage(
                SimpleNamespace(input_token_count=1, output_token_count=2),
                definition=self.definition,
                run=self.run,
                attempt=attempt,
            )

        records = await self.store.list_usage(self.run.run_id)
        self.assertEqual(records, ())

    async def test_record_usage_persists_each_provider_call(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        attempt = await self.store.create_attempt(self.run.run_id)
        response = MultiCallUsageResponse(
            SimpleNamespace(
                usage={
                    "input_tokens": 2,
                    "cached_input_tokens": 1,
                    "output_tokens": 3,
                    "total_tokens": 5,
                    "provider_family": "openai",
                }
            ),
            SimpleNamespace(
                usage={
                    "input_tokens": 4,
                    "cache_creation_input_tokens": 2,
                    "output_tokens": 6,
                    "reasoning_tokens": 1,
                    "total_tokens": 10,
                    "provider_family": "openai",
                    "raw_response_id": "private-response-id",
                }
            ),
        )

        await worker._record_usage(
            response,
            definition=self.definition,
            run=self.run,
            attempt=attempt,
        )

        records = await self.store.list_usage(self.run.run_id)
        totals = await self.store.usage_totals(self.run.run_id)
        self.assertEqual(len(records), 2)
        self.assertEqual([record.sequence for record in records], [1, 2])
        self.assertEqual(
            [record.source for record in records],
            [UsageSource.EXACT, UsageSource.EXACT],
        )
        self.assertEqual(records[0].totals.cached_input_tokens, 1)
        self.assertEqual(records[1].totals.cache_creation_input_tokens, 2)
        self.assertEqual(records[1].totals.reasoning_tokens, 1)
        self.assertNotIn("raw_response_id", records[1].metadata)
        self.assertEqual(totals.input_tokens, 6)
        self.assertEqual(totals.output_tokens, 9)
        self.assertEqual(totals.total_tokens, 15)

    async def test_record_usage_deduplicates_reobserved_response(
        self,
    ) -> None:
        sink = RecordingUsageSink()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
            observability_sink=sink,
        )
        attempt = await self.store.create_attempt(self.run.run_id)
        response = SimpleNamespace(input_token_count=2, output_token_count=3)

        await worker._record_usage(
            response,
            definition=self.definition,
            run=self.run,
            attempt=attempt,
        )
        await worker._record_usage(
            response,
            definition=self.definition,
            run=self.run,
            attempt=attempt,
        )

        records = await self.store.list_usage(self.run.run_id)
        totals = await self.store.usage_totals(self.run.run_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].sequence, 1)
        self.assertEqual(records[0].totals.input_tokens, 2)
        self.assertEqual(records[0].totals.output_tokens, 3)
        self.assertEqual(totals.input_tokens, 2)
        self.assertEqual(totals.output_tokens, 3)
        self.assertEqual(len(sink.usage_totals), 1)
        self.assertEqual(sink.usage_event_count, 1)

    async def test_process_once_records_unobserved_returned_wrapper_calls(
        self,
    ) -> None:
        sink = RecordingUsageSink()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=PartiallyObservedUsageWrapperTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
            observability_sink=sink,
        )

        result = await worker.process_once()

        records = await self.store.list_usage(self.run.run_id)
        totals = await self.store.usage_totals(self.run.run_id)
        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(records), 2)
        self.assertEqual([record.sequence for record in records], [1, 2])
        self.assertEqual(records[0].totals.input_tokens, 4)
        self.assertEqual(records[0].totals.cached_input_tokens, 1)
        self.assertEqual(records[1].totals.input_tokens, 5)
        self.assertEqual(records[1].totals.cache_creation_input_tokens, 2)
        self.assertEqual(records[1].totals.reasoning_tokens, 3)
        self.assertEqual(totals.input_tokens, 9)
        self.assertEqual(totals.output_tokens, 13)
        self.assertEqual(totals.total_tokens, 22)
        self.assertEqual(len(sink.usage_totals), 2)
        self.assertEqual(sink.usage_event_count, 2)
        self.assertNotIn("private-first-response", str(records))
        self.assertNotIn("private-second-response", str(records))
        self.assertNotIn("private provider body", str(records))

    async def test_record_usage_conflict_does_not_emit_sink_usage(
        self,
    ) -> None:
        sink = RecordingUsageSink()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
            observability_sink=sink,
        )
        attempt = await self.store.create_attempt(self.run.run_id)

        with patch.object(
            self.store,
            "append_usage",
            side_effect=TaskStoreConflictError("private usage conflict"),
        ):
            await worker._record_usage(
                SimpleNamespace(input_token_count=2, output_token_count=3),
                definition=self.definition,
                run=self.run,
                attempt=attempt,
            )

        records = await self.store.list_usage(self.run.run_id)
        self.assertEqual(records, ())
        self.assertEqual(sink.usage_totals, [])
        self.assertEqual(sink.usage_event_count, 0)

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


def _definition(
    *,
    artifact: TaskArtifactPolicy | None = None,
    input_contract: TaskInputContract | None = None,
    observability: TaskObservabilityPolicy | None = None,
    output_contract: TaskOutputContract | None = None,
    retry: TaskRetryPolicy | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="worker_task", version="1"),
        input=input_contract or TaskInputContract.string(),
        output=output_contract or TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        artifact=artifact or TaskArtifactPolicy.references_only(),
        observability=observability or TaskObservabilityPolicy(),
        run=TaskRunPolicy.queued("default"),
        retry=retry or TaskRetryPolicy(max_attempts=2),
    )


async def _definition_with_skills(
    settings: TrustedSkillSettings,
) -> TaskDefinition:
    return await task_definition_with_skills_identity(
        replace(
            _definition(),
            execution=TaskExecutionTarget.tool("skills"),
            skills=settings,
        ),
    )


def _trusted_skills(
    root: Path,
    *,
    read_limits: SkillReadLimits | None = None,
) -> TrustedSkillSettings:
    return TrustedSkillSettings(
        sources=(
            SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
            ),
        ),
        read_limits=read_limits or SkillReadLimits(),
    )


async def _registry_with_status(
    settings: TrustedSkillSettings,
    status: SkillStatus,
) -> SkillRegistry:
    assert status in {SkillStatus.MALFORMED, SkillStatus.POLICY_DENIED}
    registry = await build_task_skill_registry(settings)
    diagnostic = SkillDiagnosticInfo(
        code=(
            SkillDiagnosticCode.MANIFEST_MALFORMED
            if status is SkillStatus.MALFORMED
            else SkillDiagnosticCode.POLICY_DENIED
        ),
        status=status,
        message="Registry is not usable.",
        path="skills",
        hint="Use an operator-approved registry.",
    )
    return replace(
        registry,
        diagnostics=(diagnostic,),
    )


def _write_skill(path: Path, *, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "name: pdf\n"
        "description: PDF rendering guidance.\n"
        'tags: ["pdf"]\n'
        "resources: []\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )


async def _raise_os_error(context: TaskTargetContext) -> object:
    raise OSError("private backend path")


async def _raise_cancelled(context: TaskTargetContext) -> object:
    raise CancelledError()


async def _target_only_wait(
    tasks: set[AsyncTask[object]],
    *,
    timeout: float | None,
    return_when: object,
) -> tuple[set[AsyncTask[object]], set[AsyncTask[object]]]:
    _ = timeout, return_when
    for _attempt in range(3):
        for task in tasks:
            if task.done() and task.result() == "safe output":
                return {task}, tasks - {task}
        await sleep(0)
    raise AssertionError("target task did not finish")


async def _callable_target(context: TaskTargetContext) -> object:
    return "safe output"


if __name__ == "__main__":
    main()
