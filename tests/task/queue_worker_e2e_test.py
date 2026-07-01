from asyncio import Event as AsyncEvent
from asyncio import Task as AsyncTask
from asyncio import sleep
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from inspect import isawaitable
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.entities import ToolCall, ToolCallContext, ToolCallResult
from avalan.event import Event, EventType
from avalan.flow import (
    FlowDefinition,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowEdgePlan,
    FlowEntryBehavior,
    FlowExecutionPlan,
    FlowExecutionTrace,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowMappingKind,
    FlowMappingPlan,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodePlan,
    FlowNodeState,
    FlowNodeTrace,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    InMemoryFlowStateStore,
    compile_flow_definition,
    parse_flow_selector,
)
from avalan.flow.flow import Flow
from avalan.flow.node import Node
from avalan.skill import (
    SkillSourceConfig,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
)
from avalan.task import (
    DROPPED_MARKER,
    ENCRYPTED_MARKER,
    HASHED_MARKER,
    REDACTED_MARKER,
    STORED_MARKER,
    EncryptedPrivacyValue,
    FileConverter,
    PrivacyAction,
    PrivacySanitizer,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactState,
    TaskAttemptState,
    TaskClient,
    TaskClientWaitTimeoutError,
    TaskDefinition,
    TaskExecutionPayload,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskFileConversionPageCollection,
    TaskFileConversionPageResult,
    TaskFileConversionRequest,
    TaskFileConversionResult,
    TaskFileConverterCapability,
    TaskFileDescriptor,
    TaskInputContract,
    TaskInputFile,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskQueue,
    TaskQueueAbandonment,
    TaskQueueArtifact,
    TaskQueueClaim,
    TaskQueueCompletion,
    TaskQueueConflictError,
    TaskQueueDepth,
    TaskQueueError,
    TaskQueueHealth,
    TaskQueueItem,
    TaskQueueItemState,
    TaskQueueRetry,
    TaskQueueSubmission,
    TaskRetryPolicy,
    TaskRunPolicy,
    TaskRunState,
    TaskStoreNotFoundError,
    TaskTargetContext,
    TaskTargetRunner,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
    TaskWorker,
    TaskWorkerShutdown,
    UsageSource,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.idempotency import TaskIdempotencyIdentity
from avalan.task.skills import (
    TASK_SKILLS_METADATA_KEY,
    build_task_skill_registry,
)
from avalan.task.stores import InMemoryTaskStore
from avalan.task.targets import (
    FLOW_TASK_INPUT_KEY,
    AgentTaskTargetRunner,
    FlowTaskTargetRunner,
)
from avalan.task.targets import flow as flow_target_module
from avalan.tool.context import ToolSettingsContext
from avalan.tool.manager import ToolManager
from avalan.tool.shell import ShellToolSettings
from avalan.tool.skills import SkillsToolSet


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


class QueueAgentOrchestrator:
    def __init__(self, loader: "QueueAgentLoader") -> None:
        self._loader = loader

    async def __aenter__(self) -> "QueueAgentOrchestrator":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        return None

    async def __call__(self, input: object) -> str:
        self._loader.inputs.append(input)
        return "should not execute"


class QueueAgentLoader:
    def __init__(self) -> None:
        self.inputs: list[object] = []

    async def from_file(
        self,
        path: str,
        *,
        agent_id: object | None,
        disable_memory: bool = False,
        uri: str | None = None,
        tool_settings: object | None = None,
    ) -> QueueAgentOrchestrator:
        _ = path, agent_id, disable_memory, uri, tool_settings
        return QueueAgentOrchestrator(self)


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


class SkillsToolTarget(TaskTargetRunner):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        assert context.definition.skills is not None
        registry = await build_task_skill_registry(context.definition.skills)
        manager = ToolManager.create_instance(
            available_toolsets=[SkillsToolSet(registry)],
            enable_tools=["skills"],
        )
        matched = await manager(
            ToolCall(
                id="match",
                name="skills.match",
                arguments={"query": "render a pdf"},
            ),
            ToolCallContext(),
        )
        self.calls.append("skills.match")
        match_result = _tool_result_dict(matched)
        matched_items = cast(list[Mapping[str, object]], match_result["items"])
        metadata = cast(Mapping[str, object], matched_items[0]["metadata"])
        skill_id = metadata["skill_id"]
        assert isinstance(skill_id, str)
        read = await manager(
            ToolCall(
                id="read",
                name="skills.read",
                arguments={"skill": skill_id},
            ),
            ToolCallContext(),
        )
        self.calls.append("skills.read")
        content = cast(dict[str, object], _tool_result_dict(read)["content"])
        text = content["text"]
        assert isinstance(text, str)
        return "answered after read" if "FOLLOW_THE_PDF_STEPS" in text else ""


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


class ReturnedUsageTextTarget(TextTarget):
    async def run(self, context: TaskTargetContext) -> object:
        self.inputs.append(context.input_value)
        await context.check_cancelled()
        return UsageTextOutput(
            "public answer",
            usage={
                "input_tokens": 6,
                "cached_input_tokens": 2,
                "output_tokens": 4,
                "reasoning_tokens": 1,
                "total_tokens": 10,
                "provider_family": "openai",
                "raw_response_id": "private-response-id",
            },
        )


class PrefixTextConverter:
    name = "prefix"
    version = "1"
    capability = TaskFileConverterCapability(
        source_mime_types=("text/plain",),
        output_mime_types=("text/plain",),
        supports_streaming=False,
        max_input_bytes=1024,
        max_output_bytes=1024,
    )

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        _ = source_media_type, options
        return TaskFileConversionResult(
            content=b"converted:" + content,
            media_type="text/plain",
            metadata={"status": "ready"},
        )


class PdfPageConverter:
    name = "pdf_image"
    version = "test"
    capability = TaskFileConverterCapability(
        source_mime_types=("application/pdf",),
        output_mime_types=("image/png",),
        supports_streaming=False,
        max_input_bytes=1024,
        max_output_bytes=1024,
        max_pages=8,
        max_pixels=10_000,
    )

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        _ = content, source_media_type, options
        raise AssertionError("page converter must use page output")

    async def convert_pages(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionPageCollection:
        _ = content, source_media_type, options
        return TaskFileConversionPageCollection(
            pages=(
                TaskFileConversionPageResult(
                    page_index=1,
                    page_count=2,
                    content=b"first page image",
                    media_type="image/png",
                    width_pixels=10,
                    height_pixels=20,
                    metadata={"filename": "private-page.png"},
                ),
                TaskFileConversionPageResult(
                    page_index=2,
                    page_count=2,
                    content=b"second page image",
                    media_type="image/png",
                    width_pixels=30,
                    height_pixels=40,
                ),
            ),
            metadata={"backend": "test"},
        )


class HeartbeatWaitingTarget(TaskTargetRunner):
    def __init__(self) -> None:
        self.started = AsyncEvent()
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
        self.started.set()
        while True:
            await sleep(0)


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


class ShutdownOnceTarget(TaskTargetRunner):
    def __init__(self, shutdown: TaskWorkerShutdown) -> None:
        self.shutdown = shutdown
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
        if len(self.inputs) == 1:
            self.shutdown.request()
            await context.check_cancelled()
            return "unused"
        await context.check_cancelled()
        return "public answer"


class ShutdownReturningOnceTarget(TaskTargetRunner):
    def __init__(self, shutdown: TaskWorkerShutdown) -> None:
        self.shutdown = shutdown
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
        if len(self.inputs) == 1:
            self.shutdown.request()
            return "unused"
        await context.check_cancelled()
        return "public answer"


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
        self.heartbeat_error: BaseException | None = None
        self.heartbeats: list[datetime] = []
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
            claim = TaskQueueClaim(
                queue_item=updated,
                run=claimed,
                attempt=attempt,
            )
            if self.abandon_after_claim:
                await self.abandon(
                    updated.queue_item_id,
                    claim_token=claim_token,
                    max_attempts=2,
                    now=current_time,
                    metadata=metadata,
                )
            return claim
        return None

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
        item = self.items[queue_item_id]
        assert item.claim_token == claim_token
        heartbeat_at = now or self.clock.now
        self.heartbeats.append(heartbeat_at)
        updated = replace(
            item,
            lease_expires_at=lease_expires_at,
            heartbeat_at=heartbeat_at,
            updated_at=heartbeat_at,
        )
        self.items[queue_item_id] = updated
        return updated

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

    async def abandon(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        max_attempts: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueAbandonment:
        item = self.items[queue_item_id]
        run = await self.store.get_run(item.run_id)
        attempt = await self.store.transition_attempt(
            run.last_attempt_id or "",
            from_states={
                TaskAttemptState.CREATED,
                TaskAttemptState.RUNNING,
            },
            to_state=TaskAttemptState.ABANDONED,
            reason="abandoned",
            claim_token=claim_token,
            metadata=metadata,
        )
        cancel_requested = run.state == TaskRunState.CANCEL_REQUESTED
        retryable = (
            attempt.attempt_number < max_attempts and not cancel_requested
        )
        next_run_state = (
            TaskRunState.CANCELLED
            if cancel_requested
            else TaskRunState.QUEUED if retryable else TaskRunState.FAILED
        )
        abandoned_run = await self.store.transition_run(
            run.run_id,
            from_states={run.state},
            to_state=next_run_state,
            reason="abandoned",
            claim_token=claim_token,
            metadata=metadata,
        )
        if retryable:
            abandoned_run = replace(abandoned_run, claim=None)
            self.store._runs[run.run_id] = abandoned_run
        updated = TaskQueueItem(
            queue_item_id=item.queue_item_id,
            run_id=item.run_id,
            queue_name=item.queue_name,
            state=(
                TaskQueueItemState.AVAILABLE
                if retryable
                else TaskQueueItemState.DEAD
            ),
            priority=item.priority,
            available_at=now or self.clock.now,
            attempts=item.attempts,
            created_at=item.created_at,
            updated_at=now or self.clock.now,
            run_state=abandoned_run.state,
        )
        self.items[queue_item_id] = updated
        return TaskQueueAbandonment(
            queue_item=updated,
            run=abandoned_run,
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
    async def test_pipeline_agent_worker_fails_closed_without_opt_in(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            agent_path = root / "agents" / "pipeline.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Pipeline"
task = "Inspect files."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"

[tool]
enable = ["shell.pipeline"]

[tool.shell]
allow_pipelines = true
""",
                encoding="utf-8",
            )
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            loader = QueueAgentLoader()
            enqueue_target = AgentTaskTargetRunner(
                loader,
                ref_base=root,
                tool_settings=ToolSettingsContext(
                    shell=ShellToolSettings(allow_pipelines=True),
                ),
                require_shell_pipeline_opt_in=True,
            )
            worker_target = AgentTaskTargetRunner(
                loader,
                ref_base=root,
                require_shell_pipeline_opt_in=True,
            )
            client = _client(
                store,
                queue,
                target=enqueue_target,
                execution_roots=(root,),
                clock=clock,
            )
            worker = _worker(
                store,
                queue,
                target=worker_target,
                clock=clock,
            )
            definition = _definition(
                execution=TaskExecutionTarget.agent("agents/pipeline.toml"),
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
        self.assertIsNone(processed.completion)
        self.assertIsNone(processed.retry)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(loader.inputs, [])
        rendered = str(inspection.as_dict())
        self.assertIn("runnable.failed", rendered)
        self.assertNotIn("private prompt", rendered)

    async def test_queued_task_uses_skills_tools_after_revalidation(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            _write_agent(root / "agent.toml", enable_skills=False)
            skills_root = root / "skills"
            _write_skill(
                skills_root / "pdf" / "SKILL.md",
                body="# PDF Body\nFOLLOW_THE_PDF_STEPS\n",
            )
            settings = _trusted_skills(skills_root)
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            target = SkillsToolTarget()
            client = _client(store, queue, target=target, clock=clock)
            worker = _worker(
                store,
                queue,
                target=target,
                skills_settings=settings,
                definition_base=root / "task.toml",
                clock=clock,
            )
            definition = replace(
                _definition(
                    observability=TaskObservabilityPolicy.noop(),
                    retry=TaskRetryPolicy(max_attempts=1),
                ),
                skills=settings,
            )

            submission = await client.enqueue(
                definition,
                input_value="private prompt",
            )
            queued_run = await store.get_run(submission.run.run_id)
            processed = await worker.process_once()
            output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(target.calls, ["skills.match", "skills.read"])
        self.assertEqual(processed.output, "answered after read")
        self.assertIn(TASK_SKILLS_METADATA_KEY, queued_run.request.metadata)
        rendered_metadata = str(queued_run.request.metadata)
        self.assertNotIn("FOLLOW_THE_PDF_STEPS", rendered_metadata)
        self.assertNotIn(str(root), rendered_metadata)
        self.assertNotIn("SKILL.md", rendered_metadata)

    async def test_queued_task_blocks_skills_tools_when_identity_is_stale(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            _write_agent(root / "agent.toml", enable_skills=False)
            skills_root = root / "skills"
            skill_path = skills_root / "pdf" / "SKILL.md"
            _write_skill(
                skill_path,
                body="# PDF Body\nFOLLOW_THE_PDF_STEPS\n",
            )
            settings = _trusted_skills(skills_root)
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            target = SkillsToolTarget()
            client = _client(store, queue, target=target, clock=clock)
            worker = _worker(
                store,
                queue,
                target=target,
                skills_settings=settings,
                definition_base=root / "task.toml",
                clock=clock,
            )
            definition = replace(
                _definition(
                    observability=TaskObservabilityPolicy.noop(),
                    retry=TaskRetryPolicy(max_attempts=1),
                ),
                skills=settings,
            )

            submission = await client.enqueue(
                definition,
                input_value="private prompt",
            )
            _write_skill(skill_path, body="# PDF Body\nCHANGED\n")
            processed = await worker.process_once()
            output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(target.calls, [])
        error = output.error
        assert isinstance(error, Mapping)
        details = error.get("details")
        assert isinstance(details, Mapping)
        issues = details.get("issues")
        assert isinstance(issues, list | tuple)
        issue = issues[0]
        assert isinstance(issue, Mapping)
        self.assertEqual(issue["code"], "task.skills_registry_stale")

    async def test_queued_agent_task_fails_when_ref_adds_skills_tools(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            agent_path = root / "agents" / "assistant.toml"
            _write_agent(agent_path, enable_skills=False)
            skills_root = root / "skills"
            _write_skill(skills_root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(skills_root)
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            target = TextTarget()
            client = _client(store, queue, target=target, clock=clock)
            worker = _worker(
                store,
                queue,
                target=target,
                skills_settings=settings,
                definition_base=root / "task.toml",
                clock=clock,
            )
            definition = replace(
                _definition(
                    execution=TaskExecutionTarget.agent(
                        "agents/assistant.toml"
                    ),
                    observability=TaskObservabilityPolicy.noop(),
                    retry=TaskRetryPolicy(max_attempts=1),
                ),
                definition_base=root / "task.toml",
                skills=settings,
            )

            submission = await client.enqueue(
                definition,
                input_value="private prompt",
            )
            _write_agent(agent_path, enable_skills=True)
            processed = await worker.process_once()
            output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(target.inputs, [])
        _assert_skills_issue(output.error, "task.skills_registry_widened")

    async def test_queued_flow_task_fails_when_ref_adds_skills_tools(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            flow_path = root / "flows" / "report.toml"
            _write_flow(flow_path, enable_skills=False)
            skills_root = root / "skills"
            _write_skill(skills_root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(skills_root)
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            target = TextTarget()
            client = _client(store, queue, target=target, clock=clock)
            worker = _worker(
                store,
                queue,
                target=target,
                skills_settings=settings,
                definition_base=root / "task.toml",
                clock=clock,
            )
            definition = replace(
                _definition(
                    execution=TaskExecutionTarget.flow("flows/report.toml"),
                    observability=TaskObservabilityPolicy.noop(),
                    retry=TaskRetryPolicy(max_attempts=1),
                ),
                definition_base=root / "task.toml",
                skills=settings,
            )

            submission = await client.enqueue(
                definition,
                input_value="private prompt",
            )
            _write_flow(flow_path, enable_skills=True)
            processed = await worker.process_once()
            output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(target.inputs, [])
        _assert_skills_issue(output.error, "task.skills_registry_widened")

    async def test_queued_flow_task_fails_when_agent_ref_adds_skills_tools(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            flow_path = root / "flows" / "report.toml"
            agent_path = root / "agents" / "assistant.toml"
            _write_agent(agent_path, enable_skills=False)
            _write_agent_flow(
                flow_path,
                agent_ref="agents/assistant.toml",
            )
            skills_root = root / "skills"
            _write_skill(skills_root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(skills_root)
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            target = TextTarget()
            client = _client(store, queue, target=target, clock=clock)
            worker = _worker(
                store,
                queue,
                target=target,
                skills_settings=settings,
                definition_base=root / "task.toml",
                clock=clock,
            )
            definition = replace(
                _definition(
                    execution=TaskExecutionTarget.flow("flows/report.toml"),
                    observability=TaskObservabilityPolicy.noop(),
                    retry=TaskRetryPolicy(max_attempts=1),
                ),
                definition_base=root / "task.toml",
                skills=settings,
            )

            submission = await client.enqueue(
                definition,
                input_value="private prompt",
            )
            _write_agent(agent_path, enable_skills=True)
            processed = await worker.process_once()
            output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(target.inputs, [])
        _assert_skills_issue(output.error, "task.skills_registry_widened")

    async def test_queued_agent_task_fails_without_worker_definition_base(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            _write_agent(
                root / "agents" / "assistant.toml",
                enable_skills=False,
            )
            skills_root = root / "skills"
            _write_skill(skills_root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(skills_root)
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            target = TextTarget()
            client = _client(store, queue, target=target, clock=clock)
            worker = _worker(
                store,
                queue,
                target=target,
                skills_settings=settings,
                clock=clock,
            )
            definition = replace(
                _definition(
                    execution=TaskExecutionTarget.agent(
                        "agents/assistant.toml"
                    ),
                    observability=TaskObservabilityPolicy.noop(),
                    retry=TaskRetryPolicy(max_attempts=1),
                ),
                definition_base=root / "task.toml",
                skills=settings,
            )

            submission = await client.enqueue(
                definition,
                input_value="private prompt",
            )
            processed = await worker.process_once()
            output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(target.inputs, [])
        _assert_skills_issue(output.error, "task.skills_registry_unavailable")

    async def test_queued_agent_task_fails_with_wrong_worker_definition_base(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            _write_agent(
                root / "agents" / "assistant.toml",
                enable_skills=False,
            )
            skills_root = root / "skills"
            _write_skill(skills_root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(skills_root)
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            target = TextTarget()
            client = _client(store, queue, target=target, clock=clock)
            worker = _worker(
                store,
                queue,
                target=target,
                skills_settings=settings,
                definition_base=root / "other" / "task.toml",
                clock=clock,
            )
            definition = replace(
                _definition(
                    execution=TaskExecutionTarget.agent(
                        "agents/assistant.toml"
                    ),
                    observability=TaskObservabilityPolicy.noop(),
                    retry=TaskRetryPolicy(max_attempts=1),
                ),
                definition_base=root / "task.toml",
                skills=settings,
            )

            submission = await client.enqueue(
                definition,
                input_value="private prompt",
            )
            processed = await worker.process_once()
            output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(target.inputs, [])
        _assert_skills_issue(output.error, "task.skills_registry_unavailable")

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

    async def test_returned_usage_output_reaches_queued_inspection(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = ReturnedUsageTextTarget()
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition()

        submission = await client.enqueue(
            definition,
            input_value="private prompt",
        )
        processed = await worker.process_once()
        inspection = await client.inspect(submission.run.run_id)
        output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertTrue(output.ready)
        self.assertEqual(output.output_summary, {"privacy": REDACTED_MARKER})
        self.assertEqual(target.inputs, ["private prompt"])
        self.assertEqual(len(inspection.usage), 1)
        self.assertEqual(inspection.usage[0].source, UsageSource.EXACT)
        self.assertEqual(inspection.usage[0].totals.input_tokens, 6)
        self.assertEqual(inspection.usage_totals.cached_input_tokens, 2)
        self.assertEqual(inspection.usage_totals.output_tokens, 4)
        self.assertEqual(inspection.usage_totals.reasoning_tokens, 1)
        self.assertEqual(inspection.usage_totals.total_tokens, 10)
        self.assertEqual(
            inspection.usage[0].metadata,
            {"provider_family": "openai"},
        )
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private prompt", inspection_value)
        self.assertNotIn("raw_response_id", inspection_value)
        self.assertNotIn("private-response-id", inspection_value)

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

    async def test_file_conversion_runs_before_worker_target(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            Path(root, "document.txt").write_bytes(b"private source")
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
                file_converters={"prefix": PrefixTextConverter()},
                execution_roots=(root,),
                clock=clock,
            )
            worker = _worker(
                store,
                queue,
                target=target,
                artifact_store=artifact_store,
                file_converters={"prefix": PrefixTextConverter()},
                clock=clock,
            )
            definition = _definition(
                input_contract=TaskInputContract.file(
                    conversions=("prefix",),
                    mime_types=("text/plain",),
                )
            )

            submission = await client.enqueue(
                definition,
                input_value=TaskFileDescriptor.local_path(
                    "document.txt",
                    mime_type="text/plain",
                    conversions=(TaskFileConversionRequest(name="prefix"),),
                    metadata={"filename": "document.txt"},
                ),
            )
            processed = await worker.process_once()
            artifacts = await store.list_artifacts(submission.run.run_id)
            inspection = await client.inspect(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertEqual(target.file_bodies, [b"converted:private source"])
        self.assertEqual(
            [artifact.purpose for artifact in artifacts],
            [TaskArtifactPurpose.INPUT, TaskArtifactPurpose.CONVERTED],
        )
        self.assertNotIn("private source", str(inspection.as_dict()))
        self.assertNotIn("document.txt", str(inspection.as_dict()))

    async def test_pdf_page_conversion_expands_before_worker_target(
        self,
    ) -> None:
        clock = Clock()
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            Path(root, "document.pdf").write_bytes(b"%PDF private source")
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
            )
            explicit_ref = await artifact_store.put(
                b"explicit image",
                media_type="image/png",
            )
            store = InMemoryTaskStore(clock=lambda: clock.now)
            queue = InMemoryTaskQueue(store, clock=clock)
            target = ReadingTarget()
            client = _client(
                store,
                queue,
                target=target,
                artifact_store=artifact_store,
                file_converters={"pdf_image": PdfPageConverter()},
                execution_roots=(root,),
                clock=clock,
            )
            worker = _worker(
                store,
                queue,
                target=target,
                artifact_store=artifact_store,
                file_converters={"pdf_image": PdfPageConverter()},
                clock=clock,
            )
            definition = _definition(
                input_contract=TaskInputContract.file(
                    conversions=("pdf_image",),
                    mime_types=("application/pdf",),
                )
            )

            submission = await client.enqueue(
                definition,
                input_value=TaskFileDescriptor.local_path(
                    "document.pdf",
                    mime_type="application/pdf",
                    conversions=(TaskFileConversionRequest(name="pdf_image"),),
                    metadata={"filename": "document.pdf"},
                ),
                files=(
                    TaskInputFile(
                        logical_path=f"artifact:{explicit_ref.artifact_id}",
                        artifact_ref=explicit_ref,
                        media_type="image/png",
                        size_bytes=explicit_ref.size_bytes,
                    ),
                ),
            )
            processed = await worker.process_once()
            artifacts = await store.list_artifacts(submission.run.run_id)
            inspection = await client.inspect(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertEqual(
            target.file_bodies,
            [b"explicit image", b"first page image", b"second page image"],
        )
        self.assertEqual(
            [artifact.purpose for artifact in artifacts],
            [
                TaskArtifactPurpose.INPUT,
                TaskArtifactPurpose.INPUT,
                TaskArtifactPurpose.CONVERTED,
                TaskArtifactPurpose.CONVERTED,
            ],
        )
        converted = artifacts[2:]
        self.assertEqual(
            [artifact.metadata["page_index"] for artifact in converted],
            [1, 2],
        )
        self.assertEqual(
            [artifact.ref.media_type for artifact in converted],
            ["image/png", "image/png"],
        )
        rendered = str(inspection.as_dict())
        self.assertNotIn("private source", rendered)
        self.assertNotIn("document.pdf", rendered)
        self.assertNotIn("private-page.png", rendered)
        for artifact in converted:
            assert artifact.ref.sha256 is not None
            self.assertNotIn(artifact.ref.sha256, rendered)

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
        self.assertEqual(target.inputs[0]["prompt"], "Review the document.")
        document = cast(Mapping[str, object], target.inputs[0]["document"])
        self.assertEqual(document["reference"], "document.txt")
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
        self.assertEqual(target.inputs, ["same prompt"])
        self.assertEqual(len(inspection.attempts), 1)
        inspection_run = cast(
            Mapping[str, object], inspection.as_dict()["run"]
        )
        self.assertEqual(
            inspection_run["input_payload"],
            {"privacy": ENCRYPTED_MARKER},
        )
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
        self.assertEqual(target.inputs, ["private priority prompt"])
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
        self.assertEqual(target.inputs, ["private prompt with attachment"])
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
        self.assertEqual(target.inputs, ["private running prompt"])
        self.assertIn("cancellation", str(output.error))
        self.assertNotIn("private running prompt", str(inspection.as_dict()))

    async def test_heartbeat_claim_conflict_does_not_finalize_run(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = HeartbeatWaitingTarget()
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(
            store,
            queue,
            target=target,
            heartbeat_seconds=0.001,
            clock=clock,
        )
        queue.heartbeat_error = TaskQueueConflictError("private stale token")

        submission = await client.enqueue(
            _definition(),
            input_value="private heartbeat prompt",
        )
        result = await worker.process_once()
        inspection = await client.inspect(submission.run.run_id)
        depth = await queue.depth("default")

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.completion)
        self.assertIsNone(result.retry)
        self.assertIsNone(result.abandonment)
        self.assertEqual(inspection.run.state, TaskRunState.RUNNING)
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(
            inspection.attempts[0].state,
            TaskAttemptState.RUNNING,
        )
        self.assertEqual(depth.claimed, 1)
        self.assertEqual(depth.dead, 0)
        self.assertEqual(len(target.inputs), 1)
        self.assertNotIn("private stale token", str(inspection.as_dict()))
        self.assertNotIn(
            "private heartbeat prompt",
            str(inspection.as_dict()),
        )

    async def test_stale_claim_before_start_waits_for_fresh_claim(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = TextTarget()
        client = _client(store, queue, target=target, clock=clock)
        first_worker = _worker(store, queue, target=target, clock=clock)
        second_worker = _worker(store, queue, target=target, clock=clock)
        queue.abandon_after_claim = True

        submission = await client.enqueue(
            _definition(),
            input_value="private stale claim prompt",
        )
        stale = await first_worker.process_once()
        pending = await client.inspect(submission.run.run_id)
        pending_depth = await queue.depth("default")
        target_inputs_after_stale = tuple(target.inputs)
        queue.abandon_after_claim = False
        completed = await second_worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)
        final_depth = await queue.depth("default")

        self.assertTrue(stale.processed)
        self.assertTrue(stale.lease_lost)
        self.assertIsNone(stale.completion)
        self.assertIsNone(stale.retry)
        self.assertIsNone(stale.abandonment)
        self.assertEqual(pending.run.state, TaskRunState.QUEUED)
        self.assertEqual(len(pending.attempts), 1)
        self.assertEqual(
            pending.attempts[0].state,
            TaskAttemptState.ABANDONED,
        )
        self.assertEqual(pending_depth.available, 1)
        self.assertEqual(target_inputs_after_stale, ())
        self.assertTrue(completed.processed)
        self.assertIsNotNone(completed.completion)
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(inspection.attempts), 2)
        self.assertEqual(
            inspection.attempts[1].state,
            TaskAttemptState.SUCCEEDED,
        )
        self.assertEqual(len(target.inputs), 1)
        self.assertEqual(final_depth.active, 0)
        self.assertEqual(final_depth.dead, 0)
        self.assertNotIn(
            "private stale claim prompt",
            str(inspection.as_dict()),
        )

    async def test_heartbeat_failure_does_not_finalize_run(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = HeartbeatWaitingTarget()
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(
            store,
            queue,
            target=target,
            heartbeat_seconds=0.001,
            clock=clock,
        )
        queue.heartbeat_error = TaskQueueError("private heartbeat outage")

        submission = await client.enqueue(
            _definition(),
            input_value="private outage prompt",
        )
        result = await worker.process_once()
        inspection = await client.inspect(submission.run.run_id)
        depth = await queue.depth("default")

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.completion)
        self.assertIsNone(result.retry)
        self.assertIsNone(result.abandonment)
        self.assertEqual(inspection.run.state, TaskRunState.RUNNING)
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(
            inspection.attempts[0].state,
            TaskAttemptState.RUNNING,
        )
        self.assertEqual(depth.claimed, 1)
        self.assertEqual(depth.dead, 0)
        self.assertEqual(len(target.inputs), 1)
        self.assertNotIn("private heartbeat outage", str(result))
        self.assertNotIn(
            "private outage prompt",
            str(inspection.as_dict()),
        )

    async def test_worker_shutdown_abandons_and_reclaims_queue_task(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        shutdown = TaskWorkerShutdown()
        target = ShutdownOnceTarget(shutdown)
        client = _client(store, queue, target=target, clock=clock)
        stopping_worker = _worker(
            store,
            queue,
            target=target,
            shutdown=shutdown,
            clock=clock,
        )
        replacement_worker = _worker(
            store,
            queue,
            target=target,
            clock=clock,
        )

        submission = await client.enqueue(
            _definition(),
            input_value="private shutdown prompt",
        )
        abandoned = await stopping_worker.process_once()
        pending = await client.inspect(submission.run.run_id)
        pending_depth = await queue.depth("default")
        completed = await replacement_worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)
        final_depth = await queue.depth("default")

        self.assertTrue(abandoned.processed)
        self.assertTrue(abandoned.shutdown_requested)
        self.assertIsNotNone(abandoned.abandonment)
        assert abandoned.abandonment is not None
        self.assertTrue(abandoned.abandonment.retryable)
        self.assertEqual(pending.run.state, TaskRunState.QUEUED)
        self.assertEqual(len(pending.attempts), 1)
        self.assertEqual(
            pending.attempts[0].state,
            TaskAttemptState.ABANDONED,
        )
        self.assertEqual(pending_depth.available, 1)
        self.assertTrue(completed.processed)
        self.assertIsNotNone(completed.completion)
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(inspection.attempts), 2)
        self.assertEqual(
            inspection.attempts[1].state,
            TaskAttemptState.SUCCEEDED,
        )
        self.assertEqual(final_depth.active, 0)
        self.assertEqual(final_depth.dead, 0)
        self.assertEqual(len(target.inputs), 2)
        self.assertNotIn("private shutdown prompt", str(inspection.as_dict()))

    async def test_worker_shutdown_after_target_return_reclaims_task(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        shutdown = TaskWorkerShutdown()
        target = ShutdownReturningOnceTarget(shutdown)
        client = _client(store, queue, target=target, clock=clock)
        stopping_worker = _worker(
            store,
            queue,
            target=target,
            shutdown=shutdown,
            clock=clock,
        )
        replacement_worker = _worker(
            store,
            queue,
            target=target,
            clock=clock,
        )

        submission = await client.enqueue(
            _definition(),
            input_value="private late shutdown prompt",
        )
        with patch("avalan.task.worker.wait", new=_target_done_wait):
            abandoned = await stopping_worker.process_once()
        pending = await client.inspect(submission.run.run_id)
        pending_depth = await queue.depth("default")
        completed = await replacement_worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)

        self.assertTrue(abandoned.processed)
        self.assertTrue(abandoned.shutdown_requested)
        self.assertIsNotNone(abandoned.abandonment)
        self.assertIsNone(abandoned.completion)
        self.assertEqual(pending.run.state, TaskRunState.QUEUED)
        self.assertEqual(len(pending.attempts), 1)
        self.assertEqual(
            pending.attempts[0].state,
            TaskAttemptState.ABANDONED,
        )
        self.assertEqual(pending_depth.available, 1)
        self.assertTrue(completed.processed)
        self.assertIsNotNone(completed.completion)
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(inspection.attempts), 2)
        self.assertEqual(
            inspection.attempts[1].state,
            TaskAttemptState.SUCCEEDED,
        )
        self.assertEqual(len(target.inputs), 2)
        self.assertNotIn(
            "private late shutdown prompt",
            str(inspection.as_dict()),
        )

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
        self.assertEqual(target.inputs, ["scheduled prompt"])
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
        self.assertEqual(
            target.inputs,
            [
                {"prompt": "private structured prompt", "limit": 2},
                {"prompt": "private structured prompt", "limit": 2},
            ],
        )
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
        self.assertEqual(
            target.inputs,
            [{"prompt": "private invalid prompt", "limit": 1}],
        )
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

    async def test_queued_strict_flow_uses_declared_output_state(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow_store = InMemoryFlowStateStore()
        flow_definition = _strict_declared_output_flow_definition()
        target = FlowTaskTargetRunner(
            strict_resolver=lambda _: flow_definition,
            flow_state_store=flow_store,
        )
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
            input_value="queued declared",
        )
        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        record = await flow_store.get_flow_execution(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertEqual(processed.output, "queued declared")
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(record.revision, 2)
        self.assertEqual(
            dict(record.selected_outputs),
            {"answer": "queued declared"},
        )
        self.assertEqual(
            dict(record.node_outputs),
            {
                "start": {"value": "queued declared"},
                "terminal": {"value": "terminal output"},
            },
        )
        self.assertNotIn("terminal output", str(output.output_summary))

    async def test_queued_strict_flow_rejects_invalid_resolver_result_safely(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow_store = InMemoryFlowStateStore()
        target = FlowTaskTargetRunner(
            strict_resolver=lambda _: cast(FlowDefinition, "private flow"),
            flow_state_store=flow_store,
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
            retry=TaskRetryPolicy(max_attempts=1),
        )

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value="private prompt",
        )
        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNone(processed.completion)
        self.assertIsNone(processed.retry)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(inspection.attempts[0].state, TaskAttemptState.FAILED)
        with self.assertRaises(TaskStoreNotFoundError):
            await flow_store.get_flow_execution(submission.run.run_id)
        self.assertNotIn("private flow", str(inspection.as_dict()))
        self.assertNotIn("private prompt", str(inspection.as_dict()))

    async def test_queued_strict_flow_rejects_review_without_state_store(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        target = FlowTaskTargetRunner(
            strict_resolver=lambda _: _strict_human_review_flow_plan(),
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/review.toml"),
            observability=TaskObservabilityPolicy.noop(),
            retry=TaskRetryPolicy(max_attempts=1),
        )

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value="private prompt",
        )
        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNone(processed.completion)
        self.assertIsNone(processed.retry)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(inspection.attempts[0].state, TaskAttemptState.FAILED)
        inspection_value = str(inspection.as_dict())
        self.assertIn("runnable.failed", inspection_value)
        self.assertNotIn("private prompt", inspection_value)

    async def test_queued_strict_flow_rejects_nested_review_without_store(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        calls = 0
        target = FlowTaskTargetRunner(
            strict_resolver=lambda _: _strict_nested_human_review_flow_plan(),
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/review.toml"),
            observability=TaskObservabilityPolicy.noop(),
            retry=TaskRetryPolicy(max_attempts=1),
        )

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value="private nested prompt",
        )

        async def fail_execute_flow_plan(
            *args: object,
            **kwargs: object,
        ) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError("flow runtime should not execute")

        with patch.object(
            flow_target_module,
            "execute_flow_plan",
            fail_execute_flow_plan,
        ):
            processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNone(processed.completion)
        self.assertIsNone(processed.retry)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(inspection.attempts[0].state, TaskAttemptState.FAILED)
        self.assertEqual(calls, 0)
        inspection_value = str(inspection.as_dict())
        self.assertIn("runnable.failed", inspection_value)
        self.assertNotIn("private nested prompt", inspection_value)

    async def test_queued_strict_flow_resumes_complete_state(self) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow_store = InMemoryFlowStateStore()
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
        )
        flow_definition = _strict_constant_flow_definition("fresh output")
        plan_result = await compile_flow_definition(flow_definition)
        assert plan_result.plan is not None
        target = FlowTaskTargetRunner(
            strict_resolver=lambda _: flow_definition,
            flow_state_store=flow_store,
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value="safe",
        )
        await flow_store.create_flow_execution(
            submission.run.run_id,
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="answer",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                ),
            ),
            selected_outputs={"answer": "resumed output"},
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                plan_result.plan
            ),
        )

        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        record = await flow_store.get_flow_execution(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertEqual(processed.output, "resumed output")
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(record.revision, 1)
        self.assertEqual(
            dict(record.selected_outputs),
            {"answer": "resumed output"},
        )

    async def test_queued_strict_flow_runs_subflow_plan(self) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow_store = InMemoryFlowStateStore()
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
        )

        async def resolve_strict_subflow(
            _: TaskTargetContext,
        ) -> FlowExecutionPlan:
            return await _strict_subflow_flow_plan()

        target = FlowTaskTargetRunner(
            strict_resolver=resolve_strict_subflow,
            flow_state_store=flow_store,
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value="queued seed",
        )
        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        record = await flow_store.get_flow_execution(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertEqual(processed.output, "queued seed")
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            dict(record.selected_outputs), {"answer": "queued seed"}
        )
        self.assertEqual(record.node_attempts[0].node, "child")
        self.assertEqual(
            record.node_attempts[0].state,
            FlowNodeState.SUCCEEDED,
        )

    async def test_queued_strict_flow_reports_subflow_failure_safely(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow_store = InMemoryFlowStateStore()
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
            retry=TaskRetryPolicy(max_attempts=1),
        )
        target = FlowTaskTargetRunner(
            strict_resolver=lambda _: _failing_strict_subflow_flow_plan(),
            flow_state_store=flow_store,
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value="private queued seed",
        )
        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)
        record = await flow_store.get_flow_execution(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNone(processed.completion)
        self.assertIsNone(processed.retry)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(inspection.attempts[0].state, TaskAttemptState.FAILED)
        self.assertEqual(
            record.diagnostics[0].code,
            "flow.execution.subflow_failed",
        )
        self.assertNotIn("private queued seed", str(inspection.as_dict()))
        self.assertNotIn("private queued seed", str(record.as_snapshot()))

    async def test_queued_strict_flow_resumes_partial_state(self) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow_store = InMemoryFlowStateStore()
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
        )
        flow_definition = _strict_two_step_flow_definition()
        plan_result = await compile_flow_definition(flow_definition)
        assert plan_result.plan is not None
        target = FlowTaskTargetRunner(
            strict_resolver=lambda _: flow_definition,
            flow_state_store=flow_store,
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        calls: list[str] = []
        real_runner_factory = flow_target_module.flow_node_registry_runner

        def counting_runner_factory(registry: object) -> object:
            runner = real_runner_factory(registry)

            async def run(
                node: FlowNodePlan, inputs: Mapping[str, object]
            ) -> object:
                calls.append(node.name)
                output = runner(node, inputs)
                if isawaitable(output):
                    return await output
                return output

            return run

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value="fresh input",
        )
        trace = FlowExecutionTrace.from_plan(plan_result.plan).with_node_state(
            "start", FlowNodeState.SUCCEEDED, attempts=1
        )
        await flow_store.create_flow_execution(
            submission.run.run_id,
            trace=trace,
            node_outputs={"start": {"value": "queued seed"}},
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                plan_result.plan
            ),
        )

        with patch.object(
            flow_target_module,
            "flow_node_registry_runner",
            counting_runner_factory,
        ):
            processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        record = await flow_store.get_flow_execution(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertEqual(calls, ["answer"])
        self.assertEqual(processed.output, "queued seed")
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            dict(record.node_outputs),
            {
                "start": {"value": "queued seed"},
                "answer": {"value": "queued seed"},
            },
        )
        self.assertEqual(
            dict(record.selected_outputs), {"answer": "queued seed"}
        )

    async def test_queued_strict_flow_resumes_human_review_matrix(
        self,
    ) -> None:
        decisions = {
            "approved": "approved_sink",
            "rejected": "rejected_sink",
            "needs-correction": "correction_sink",
            "expired": "expired_sink",
            "escalated": "escalated_sink",
        }

        for decision, target_node in decisions.items():
            with self.subTest(decision=decision):
                clock = Clock()
                store = InMemoryTaskStore(clock=lambda: clock.now)
                queue = InMemoryTaskQueue(store, clock=clock)
                flow_store = InMemoryFlowStateStore()
                plan = _strict_human_review_flow_plan()
                target = FlowTaskTargetRunner(
                    strict_resolver=lambda _: plan,
                    flow_state_store=flow_store,
                )
                client = _client(store, queue, target=target, clock=clock)
                worker = _worker(store, queue, target=target, clock=clock)
                definition = _definition(
                    execution=TaskExecutionTarget.flow("flows/review.toml"),
                    observability=TaskObservabilityPolicy.noop(),
                    retry=TaskRetryPolicy(max_attempts=2),
                )

                submission = await self._enqueue_raw_input(
                    store,
                    queue,
                    definition,
                    input_value="private prompt",
                )
                paused = await worker.process_once()
                paused_record = await flow_store.get_flow_execution(
                    submission.run.run_id,
                )
                await _requeue_run_with_metadata(
                    store,
                    queue,
                    submission.run.run_id,
                    {
                        (
                            flow_target_module.FLOW_RESUME_DECISIONS_METADATA_KEY
                        ): {
                            "review": {
                                "decision": decision,
                                "comment": "private-token",
                            },
                        },
                    },
                    clock=clock,
                )

                resumed = await worker.process_once()
                output = await client.output(submission.run.run_id)
                inspection = await client.inspect(submission.run.run_id)
                record = await flow_store.get_flow_execution(
                    submission.run.run_id,
                )

                self.assertTrue(paused.processed)
                self.assertIsNone(paused.retry)
                self.assertEqual(
                    {
                        node.node: node.state
                        for node in paused_record.trace.nodes
                    }["review"],
                    FlowNodeState.PAUSED,
                )
                self.assertTrue(resumed.processed)
                self.assertIsNotNone(resumed.completion)
                self.assertEqual(resumed.output, decision)
                self.assertTrue(output.ready)
                self.assertEqual(output.state, TaskRunState.SUCCEEDED)
                self.assertEqual(
                    output.output_summary,
                    {"privacy": REDACTED_MARKER},
                )
                self.assertEqual(dict(record.pause_tokens), {})
                self.assertEqual(
                    dict(record.selected_outputs), {"answer": decision}
                )
                self.assertEqual(
                    {node.node: node.state for node in record.trace.nodes}[
                        target_node
                    ],
                    FlowNodeState.SUCCEEDED,
                )
                audit = cast(
                    Mapping[str, object],
                    record.metadata["human_review_audit"],
                )
                review = cast(Mapping[str, object], audit["review"])
                self.assertEqual(review["state"], "resumed")
                self.assertEqual(review["decision"], decision)
                inspection_value = str(inspection.as_dict())
                self.assertNotIn("private prompt", str(audit))
                self.assertNotIn("private-token", str(audit))
                self.assertNotIn("private prompt", inspection_value)
                self.assertNotIn("private-token", inspection_value)

    async def test_queued_strict_flow_without_review_decision_stays_paused(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow_store = InMemoryFlowStateStore()
        plan = _strict_human_review_flow_plan()
        target = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=flow_store,
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/review.toml"),
            observability=TaskObservabilityPolicy.noop(),
            retry=TaskRetryPolicy(max_attempts=2),
        )
        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value="private prompt",
        )
        first = await worker.process_once()
        first_record = await flow_store.get_flow_execution(
            submission.run.run_id,
        )
        await _requeue_run_with_metadata(
            store,
            queue,
            submission.run.run_id,
            {},
            clock=clock,
        )

        second = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)
        record = await flow_store.get_flow_execution(submission.run.run_id)

        self.assertTrue(first.processed)
        self.assertIsNone(first.retry)
        self.assertTrue(second.processed)
        self.assertIsNone(second.retry)
        self.assertIsNone(second.completion)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(record.revision, first_record.revision + 1)
        self.assertEqual(set(record.pause_tokens), {"review"})
        self.assertEqual(
            {node.node: node.state for node in record.trace.nodes}["review"],
            FlowNodeState.PAUSED,
        )
        self.assertEqual(
            [
                attempt
                for attempt in record.node_attempts
                if attempt.node == "start"
            ],
            [
                attempt
                for attempt in first_record.node_attempts
                if attempt.node == "start"
            ],
        )
        inspection_value = str(inspection.as_dict())
        audit = record.metadata["human_review_audit"]
        self.assertIn("runnable.failed", inspection_value)
        self.assertNotIn("private prompt", inspection_value)
        self.assertNotIn("private prompt", str(audit))

    async def test_queued_strict_flow_rejects_invalid_review_resume_cases(
        self,
    ) -> None:
        cases = (
            (
                "unknown_decision",
                {
                    "review": {
                        "decision": "unknown",
                        "comment": "private-token",
                    },
                },
            ),
            (
                "schema",
                {
                    "review": {
                        "decision": "approved",
                        "comment": 7,
                    },
                },
            ),
            (
                "unknown_node",
                {
                    "missing": {
                        "decision": "approved",
                        "comment": "private-token",
                    },
                },
            ),
            (
                "non_review_node",
                {
                    "start": {
                        "decision": "approved",
                        "comment": "private-token",
                    },
                },
            ),
            ("metadata_shape", "private-token"),
        )

        for name, resume_value in cases:
            with self.subTest(name=name):
                clock = Clock()
                store = InMemoryTaskStore(clock=lambda: clock.now)
                queue = InMemoryTaskQueue(store, clock=clock)
                flow_store = InMemoryFlowStateStore()
                plan = _strict_human_review_flow_plan()
                target = FlowTaskTargetRunner(
                    strict_resolver=lambda _: plan,
                    flow_state_store=flow_store,
                )
                client = _client(store, queue, target=target, clock=clock)
                worker = _worker(store, queue, target=target, clock=clock)
                definition = _definition(
                    execution=TaskExecutionTarget.flow("flows/review.toml"),
                    observability=TaskObservabilityPolicy.noop(),
                    retry=TaskRetryPolicy(max_attempts=2),
                )
                submission = await self._enqueue_raw_input(
                    store,
                    queue,
                    definition,
                    input_value="private prompt",
                )
                paused = await worker.process_once()
                paused_record = await flow_store.get_flow_execution(
                    submission.run.run_id,
                )
                await _requeue_run_with_metadata(
                    store,
                    queue,
                    submission.run.run_id,
                    {
                        (
                            flow_target_module.FLOW_RESUME_DECISIONS_METADATA_KEY
                        ): resume_value,
                    },
                    clock=clock,
                )

                processed = await worker.process_once()
                output = await client.output(submission.run.run_id)
                inspection = await client.inspect(submission.run.run_id)
                record = await flow_store.get_flow_execution(
                    submission.run.run_id
                )

                self.assertTrue(paused.processed)
                self.assertIsNone(paused.retry)
                self.assertTrue(processed.processed)
                self.assertIsNone(processed.retry)
                self.assertIsNone(processed.completion)
                self.assertFalse(output.ready)
                self.assertEqual(output.state, TaskRunState.FAILED)
                self.assertEqual(record.revision, paused_record.revision)
                self.assertEqual(
                    dict(record.pause_tokens),
                    dict(paused_record.pause_tokens),
                )
                self.assertEqual(
                    {node.node: node.state for node in record.trace.nodes}[
                        "review"
                    ],
                    FlowNodeState.PAUSED,
                )
                inspection_value = str(inspection.as_dict())
                audit = record.metadata["human_review_audit"]
                self.assertIn("runnable.failed", inspection_value)
                self.assertNotIn("private prompt", inspection_value)
                self.assertNotIn("private-token", inspection_value)
                self.assertNotIn("private prompt", str(audit))
                self.assertNotIn("private-token", str(audit))

    async def test_queued_strict_flow_cancelled_review_resume_stays_paused(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow_store = InMemoryFlowStateStore()
        plan = _strict_human_review_flow_plan()
        target = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=flow_store,
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/review.toml"),
            observability=TaskObservabilityPolicy.noop(),
            retry=TaskRetryPolicy(max_attempts=2),
        )
        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value="private prompt",
        )
        paused = await worker.process_once()
        paused_record = await flow_store.get_flow_execution(
            submission.run.run_id,
        )
        metadata = {
            flow_target_module.FLOW_RESUME_DECISIONS_METADATA_KEY: {
                "review": {
                    "decision": "approved",
                    "comment": "private-token",
                },
            },
        }
        await _requeue_run_with_metadata(
            store,
            queue,
            submission.run.run_id,
            metadata,
            clock=clock,
        )
        cancelled = await client.cancel(submission.run.run_id)

        idle = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)
        record = await flow_store.get_flow_execution(submission.run.run_id)
        depth = await queue.depth("default")

        self.assertTrue(paused.processed)
        self.assertIsNone(paused.retry)
        self.assertEqual(cancelled.state, TaskRunState.CANCEL_REQUESTED)
        self.assertFalse(idle.processed)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.CANCEL_REQUESTED)
        self.assertEqual(inspection.run.state, TaskRunState.CANCEL_REQUESTED)
        self.assertEqual(record.revision, paused_record.revision)
        self.assertEqual(
            dict(record.pause_tokens), dict(paused_record.pause_tokens)
        )
        self.assertEqual(
            {node.node: node.state for node in record.trace.nodes}["review"],
            FlowNodeState.PAUSED,
        )
        self.assertEqual(depth.cancel_requested, 1)
        self.assertEqual(depth.available, 1)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private prompt", inspection_value)
        self.assertNotIn("private-token", inspection_value)
        self.assertNotIn("private-token", str(record.as_snapshot()))

    async def test_queued_strict_flow_rejects_mismatched_state_safely(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow_store = InMemoryFlowStateStore()
        definition = _definition(
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
            retry=TaskRetryPolicy(max_attempts=1),
        )
        previous_flow = _strict_constant_flow_definition("previous output")
        plan_result = await compile_flow_definition(previous_flow)
        assert plan_result.plan is not None
        current_flow = _strict_constant_flow_definition("fresh output")
        target = FlowTaskTargetRunner(
            strict_resolver=lambda _: current_flow,
            flow_state_store=flow_store,
        )
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value="private prompt",
        )
        await flow_store.create_flow_execution(
            submission.run.run_id,
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="answer",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                ),
            ),
            selected_outputs={"answer": "private stale output"},
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                plan_result.plan
            ),
        )

        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)
        record = await flow_store.get_flow_execution(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.claimed)
        self.assertIsNone(processed.completion)
        self.assertIsNone(processed.retry)
        self.assertIsNone(processed.abandonment)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(
            inspection.attempts[0].state,
            TaskAttemptState.FAILED,
        )
        self.assertEqual(record.revision, 1)
        self.assertEqual(
            dict(record.selected_outputs),
            {"answer": "private stale output"},
        )
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private prompt", inspection_value)
        self.assertNotIn("private stale output", inspection_value)
        self.assertNotIn("private stale output", str(output.error))

    async def test_queued_flow_array_input_returns_json_array_output(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow = Flow()
        flow.add_node(
            Node("A", func=lambda inputs: inputs[FLOW_TASK_INPUT_KEY])
        )
        target = FlowTaskTargetRunner(flow_resolver=lambda _: flow)
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(
            input_contract=TaskInputContract.array(
                schema={
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                }
            ),
            output_contract=TaskOutputContract.array(
                schema={
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                }
            ),
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
        )

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value=["safe", "done"],
        )
        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertEqual(processed.output, ["safe", "done"])
        self.assertIsInstance(processed.output, list)
        self.assertTrue(output.ready)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)

    async def test_queued_flow_array_output_contract_failure_is_safe(
        self,
    ) -> None:
        clock = Clock()
        store = InMemoryTaskStore(clock=lambda: clock.now)
        queue = InMemoryTaskQueue(store, clock=clock)
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: [
                    cast(list[object], inputs[FLOW_TASK_INPUT_KEY])[0],
                    "private invalid item",
                ],
            )
        )
        target = FlowTaskTargetRunner(flow_resolver=lambda _: flow)
        client = _client(store, queue, target=target, clock=clock)
        worker = _worker(store, queue, target=target, clock=clock)
        definition = _definition(
            input_contract=TaskInputContract.array(
                schema={
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                }
            ),
            output_contract=TaskOutputContract.array(
                schema={
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                }
            ),
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
            retry=TaskRetryPolicy(max_attempts=1),
        )

        submission = await self._enqueue_raw_input(
            store,
            queue,
            definition,
            input_value=[1],
        )
        processed = await worker.process_once()
        output = await client.output(submission.run.run_id)
        inspection = await client.inspect(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNone(processed.retry)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertIn("output_contract", str(output.error))
        self.assertNotIn("private invalid item", str(inspection.as_dict()))

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
            {"privacy": REDACTED_MARKER},
        )
        self.assertIsNotNone(run.request.input_payload)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(output.output_summary, {"status": "ready"})

    async def test_queued_flow_keeps_declared_privacy_marker_fields(
        self,
    ) -> None:
        for marker in (
            DROPPED_MARKER,
            ENCRYPTED_MARKER,
            HASHED_MARKER,
            REDACTED_MARKER,
        ):
            with self.subTest(marker=marker):
                clock = Clock()
                store = InMemoryTaskStore(clock=lambda: clock.now)
                queue = InMemoryTaskQueue(store, clock=clock)

                def use_full_input(
                    inputs: Mapping[str, object],
                ) -> dict[str, object]:
                    full_input = cast(
                        Mapping[str, object],
                        inputs[FLOW_TASK_INPUT_KEY],
                    )
                    return {
                        "status": "ready",
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
                            "required": ["privacy", "value"],
                            "additionalProperties": False,
                            "properties": {
                                "privacy": {
                                    "type": "string",
                                    "enum": [marker],
                                },
                                "value": {"type": "string"},
                            },
                        }
                    ),
                    output_contract=TaskOutputContract.object(
                        schema={
                            "type": "object",
                            "required": ["status", "privacy", "value"],
                            "additionalProperties": False,
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": ["ready"],
                                },
                                "privacy": {"type": "string"},
                                "value": {"type": "string"},
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
                        "privacy": marker,
                        "value": "safe marker value",
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
                        "privacy": marker,
                        "value": "safe marker value",
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
        worker = TaskWorker(
            store,
            cast(TaskQueue, queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: clock.now,
        )
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
        self.assertIn("privacy", str(output.error))
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
                input_summary={"privacy": REDACTED_MARKER},
                input_payload=TaskExecutionPayload(
                    input_value=PrivacySanitizer(
                        definition.privacy,
                        encryption_provider=StaticEncryptionProvider(),
                        raw_storage_allowed=True,
                    ).sanitize_with_action(
                        PrivacyAction.ENCRYPT,
                        input_value,
                    ),
                ),
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
    file_converters: Mapping[str, FileConverter] | None = None,
    execution_roots: tuple[Path, ...] = (),
    clock: Clock,
) -> TaskClient:
    return TaskClient(
        store,
        target=target,
        queue=cast(TaskQueue, queue),
        hmac_provider=StaticHmacProvider(),
        encryption_provider=StaticEncryptionProvider(),
        raw_storage_allowed=True,
        artifact_store=artifact_store,
        file_converters=file_converters,
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
    file_converters: Mapping[str, FileConverter] | None = None,
    shutdown: TaskWorkerShutdown | None = None,
    heartbeat_seconds: float | None = None,
    skills_settings: TrustedSkillSettings | None = None,
    definition_base: str | Path | None = None,
    clock: Clock,
) -> TaskWorker:
    return TaskWorker(
        store,
        cast(TaskQueue, queue),
        target=target,
        worker_id="worker-1",
        queue_name=queue_name,
        encryption_provider=StaticEncryptionProvider(),
        raw_storage_allowed=True,
        artifact_store=artifact_store,
        file_converters=file_converters,
        shutdown=shutdown,
        heartbeat_seconds=heartbeat_seconds,
        skills_settings=skills_settings,
        definition_base=definition_base,
        clock=lambda: clock.now,
    )


async def _requeue_run_with_metadata(
    store: InMemoryTaskStore,
    queue: InMemoryTaskQueue,
    run_id: str,
    metadata: Mapping[str, object],
    *,
    clock: Clock,
) -> None:
    run = await store.get_run(run_id)
    store._runs[run_id] = replace(
        run,
        state=TaskRunState.QUEUED,
        request=replace(run.request, metadata=metadata),
        claim=None,
        updated_at=clock.now,
    )
    queue_item_id = queue.items_by_run_id[run_id]
    item = queue.items[queue_item_id]
    queue.items[queue_item_id] = replace(
        item,
        state=TaskQueueItemState.AVAILABLE,
        run_state=TaskRunState.QUEUED,
        available_at=clock.now,
        updated_at=clock.now,
        claimed_at=None,
        lease_expires_at=None,
        worker_id=None,
        claim_token=None,
        heartbeat_at=None,
    )


async def _target_done_wait(
    tasks: set[AsyncTask[object]],
    *,
    timeout: float | None,
    return_when: object,
) -> tuple[set[AsyncTask[object]], set[AsyncTask[object]]]:
    _ = timeout, return_when
    for _attempt in range(3):
        for task in tasks:
            if task.done() and task.result() == "unused":
                return {task}, tasks - {task}
        await sleep(0)
    raise AssertionError("target task did not finish")


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


def _strict_constant_flow_definition(value: str) -> FlowDefinition:
    return FlowDefinition(
        name="queued-constant",
        version="1",
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.TEXT),
        ),
        entry_behavior=FlowEntryBehavior(node="answer"),
        output_behavior=FlowOutputBehavior(outputs={"answer": "answer.value"}),
        nodes=(
            FlowNodeDefinition(
                name="answer",
                type="constant",
                config={"value": value},
            ),
        ),
    )


def _strict_two_step_flow_definition() -> FlowDefinition:
    return FlowDefinition(
        name="queued-two-step",
        version="1",
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.TEXT),
        ),
        entry_behavior=FlowEntryBehavior(node="start"),
        output_behavior=FlowOutputBehavior(outputs={"answer": "answer.value"}),
        nodes=(
            FlowNodeDefinition(
                name="start",
                type="pass-through",
                mappings=(
                    FlowInputMapping(target="value", source="input.prompt"),
                ),
            ),
            FlowNodeDefinition(
                name="answer",
                type="pass-through",
                mappings=(
                    FlowInputMapping(target="value", source="start.value"),
                ),
            ),
        ),
        edges=(
            FlowEdgeDefinition(
                source="start",
                target="answer",
                kind=FlowEdgeKind.SUCCESS,
            ),
        ),
    )


def _strict_human_review_flow_plan() -> FlowExecutionPlan:
    decisions = {
        "approved": "approved_sink",
        "rejected": "rejected_sink",
        "needs-correction": "correction_sink",
        "expired": "expired_sink",
        "escalated": "escalated_sink",
    }
    return FlowExecutionPlan(
        name="queued-review",
        version="1",
        revision=None,
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.TEXT),
        ),
        entry_node="start",
        output_selectors={
            "answer": parse_flow_selector("review.result.decision")
        },
        nodes=(
            FlowNodePlan(
                name="start",
                type="pass-through",
                kind=FlowNodeKind.PASS_THROUGH,
                mappings=(
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("inputs.prompt"),
                    ),
                ),
                output_contracts=(
                    FlowNodeContract(name="value", type=FlowOutputType.JSON),
                ),
            ),
            FlowNodePlan(
                name="review",
                type="human_review",
                kind=FlowNodeKind.HUMAN_REVIEW,
                config={
                    "allowed_decisions": tuple(decisions),
                    "audit_metadata": {"risk": "medium", "queue": "ops"},
                    "decision_schema": {
                        "type": "object",
                        "required": ("decision",),
                        "properties": {
                            "decision": {"enum": tuple(decisions)},
                            "comment": {"type": "string"},
                        },
                    },
                    "timeout_seconds": 300,
                },
                mappings=(
                    FlowMappingPlan(
                        target="payload",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("start.value"),
                    ),
                ),
                output_contracts=(
                    FlowNodeContract(
                        name="result",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
            ),
        )
        + tuple(
            FlowNodePlan(
                name=node_name,
                type="pass-through",
                kind=FlowNodeKind.PASS_THROUGH,
                mappings=(
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("review.result.decision"),
                    ),
                ),
                output_contracts=(
                    FlowNodeContract(name="value", type=FlowOutputType.JSON),
                ),
            )
            for node_name in decisions.values()
        ),
        edges=(
            FlowEdgePlan(
                index=0,
                source="start",
                target="review",
                kind=FlowEdgeKind.SUCCESS,
            ),
        )
        + tuple(
            FlowEdgePlan(
                index=index,
                source="review",
                target=node_name,
                kind=FlowEdgeKind.RESUME,
                label=decision,
            )
            for index, (decision, node_name) in enumerate(
                decisions.items(),
                start=1,
            )
        ),
    )


def _strict_nested_human_review_flow_plan() -> FlowExecutionPlan:
    return FlowExecutionPlan(
        name="queued-nested-review",
        version="1",
        revision=None,
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.TEXT),
        ),
        entry_node="child",
        output_selectors={"answer": parse_flow_selector("child.result")},
        nodes=(
            FlowNodePlan(
                name="child",
                type="subflow",
                kind=FlowNodeKind.SUBFLOW,
                input_contracts=(
                    FlowNodeContract(
                        name="prompt",
                        type=FlowInputType.STRING,
                    ),
                ),
                output_contracts=(
                    FlowNodeContract(
                        name="result",
                        type=FlowOutputType.TEXT,
                    ),
                ),
                mappings=(
                    FlowMappingPlan(
                        target="prompt",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("inputs.prompt"),
                    ),
                ),
                metadata={
                    "subflow": {
                        "plan": _strict_human_review_flow_plan(),
                        "output_mapping": {"result": "answer"},
                    }
                },
            ),
        ),
    )


def _strict_declared_output_flow_definition() -> FlowDefinition:
    return FlowDefinition(
        name="queued-declared-output",
        version="1",
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.TEXT),
        ),
        entry_behavior=FlowEntryBehavior(node="start"),
        output_behavior=FlowOutputBehavior(outputs={"answer": "start.value"}),
        nodes=(
            FlowNodeDefinition(
                name="start",
                type="pass-through",
                mappings=(
                    FlowInputMapping(target="value", source="input.prompt"),
                ),
            ),
            FlowNodeDefinition(
                name="terminal",
                type="constant",
                config={"value": "terminal output"},
            ),
        ),
        edges=(
            FlowEdgeDefinition(
                source="start",
                target="terminal",
                kind=FlowEdgeKind.SUCCESS,
            ),
        ),
    )


async def _strict_subflow_flow_plan() -> FlowExecutionPlan:
    child_result = await compile_flow_definition(
        _strict_two_step_flow_definition()
    )
    assert child_result.plan is not None
    return FlowExecutionPlan(
        name="queued-subflow",
        version="1",
        revision=None,
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.TEXT),
        ),
        entry_node="child",
        output_selectors={"answer": parse_flow_selector("child.result")},
        nodes=(
            FlowNodePlan(
                name="child",
                type="subflow",
                kind=FlowNodeKind.SUBFLOW,
                input_contracts=(
                    FlowNodeContract(
                        name="prompt",
                        type=FlowInputType.STRING,
                    ),
                ),
                output_contracts=(
                    FlowNodeContract(
                        name="result",
                        type=FlowOutputType.TEXT,
                    ),
                ),
                mappings=(
                    FlowMappingPlan(
                        target="prompt",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("input.prompt"),
                    ),
                ),
                metadata={
                    "subflow": {
                        "plan": child_result.plan,
                        "output_mapping": {"result": "answer"},
                    }
                },
            ),
        ),
        edges=(),
    )


def _failing_strict_subflow_flow_plan() -> FlowExecutionPlan:
    child_plan = FlowExecutionPlan(
        name="queued-failing-child",
        version="1",
        revision=None,
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.TEXT),
        ),
        entry_node="missing",
        output_selectors={"answer": parse_flow_selector("missing.value")},
        nodes=(
            FlowNodePlan(
                name="missing",
                type="missing",
                kind=FlowNodeKind.PASS_THROUGH,
            ),
        ),
    )
    return FlowExecutionPlan(
        name="queued-failing-subflow",
        version="1",
        revision=None,
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.TEXT),
        ),
        entry_node="child",
        output_selectors={"answer": parse_flow_selector("child.result")},
        nodes=(
            FlowNodePlan(
                name="child",
                type="subflow",
                kind=FlowNodeKind.SUBFLOW,
                mappings=(
                    FlowMappingPlan(
                        target="prompt",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("input.prompt"),
                    ),
                ),
                metadata={
                    "subflow": {
                        "plan": child_plan,
                        "output_mapping": {"result": "answer"},
                    }
                },
            ),
        ),
        edges=(),
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
        privacy=TaskPrivacyPolicy(raw_retention_days=1),
        artifact=artifact or TaskArtifactPolicy.references_only(),
        observability=observability or TaskObservabilityPolicy(),
        retry=retry or TaskRetryPolicy(max_attempts=2),
    )


def _trusted_skills(root: Path) -> TrustedSkillSettings:
    return TrustedSkillSettings(
        sources=(
            SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
            ),
        ),
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


def _write_agent(path: Path, *, enable_skills: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tool_section = (
        '[tool]\nenable = ["skills.read"]\n' if enable_skills else ""
    )
    path.write_text(
        "[agent]\n"
        'name = "Assistant"\n'
        'task = "Answer."\n\n'
        "[engine]\n"
        'uri = "ai://env:KEY@openai/gpt-4o-mini"\n\n'
        f"{tool_section}",
        encoding="utf-8",
    )


def _write_flow(path: Path, *, enable_skills: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    node = (
        '[nodes.answer]\ntype = "tool"\nref = "skills.read"\n'
        if enable_skills
        else (
            '[nodes.answer]\ntype = "constant"\n\n[nodes.answer.config]\n'
            'value = "ok"\n'
        )
    )
    path.write_text(
        f'[flow]\nname = "Report"\nversion = "1"\n\n{node}',
        encoding="utf-8",
    )


def _write_agent_flow(path: Path, *, agent_ref: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[flow]\n"
        'name = "Agent Report"\n'
        'version = "1"\n\n'
        "[nodes.answer]\n"
        'type = "agent"\n'
        f'ref = "{agent_ref}"\n',
        encoding="utf-8",
    )


def _tool_result_dict(outcome: object) -> dict[str, object]:
    assert isinstance(outcome, ToolCallResult)
    assert isinstance(outcome.result, dict)
    return outcome.result


def _assert_skills_issue(error: object, code: str) -> None:
    assert isinstance(error, Mapping)
    details = error.get("details")
    assert isinstance(details, Mapping)
    issues = details.get("issues")
    assert isinstance(issues, list | tuple)
    issue = issues[0]
    assert isinstance(issue, Mapping)
    assert issue["code"] == code


if __name__ == "__main__":
    main()
