from base64 import b64decode
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, BinaryIO, cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.cli.commands import task as task_cmds
from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallOutcome,
    ToolCallResult,
    ToolDescriptor,
    ToolNameResolution,
    ToolNameResolutionStatus,
)
from avalan.event import Event, EventType
from avalan.flow import (
    FlowConditionOperator,
    FlowConditionPlan,
    FlowDefinition,
    FlowEdgeKind,
    FlowEdgePlan,
    FlowEntryBehavior,
    FlowExecutionPlan,
    FlowInputDefinition,
    FlowInputType,
    FlowJoinPlan,
    FlowJoinPolicyType,
    FlowMappingKind,
    FlowMappingPlan,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodePlan,
    FlowNodeState,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowRouteMatchPolicy,
    InMemoryFlowStateStore,
    compile_flow_definition,
    parse_flow_selector,
)
from avalan.flow.flow import Flow
from avalan.task import (
    ENCRYPTED_MARKER,
    HASHED_MARKER,
    REDACTED_MARKER,
    EncryptedPrivacyValue,
    FileConverter,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskAttemptState,
    TaskClient,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskFileConversionPageCollection,
    TaskFileConversionPageResult,
    TaskFileConversionRequest,
    TaskFileConversionResult,
    TaskFileConverterCapability,
    TaskInputContract,
    TaskInputFile,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskProviderReferenceKind,
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
from avalan.task.targets import (
    AgentTaskTargetRunner,
    FlowTaskTargetRunner,
    task_flow_node_registry,
)


class MatrixClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    async def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class MatrixHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        return TaskKeyMaterial(
            key_id=key_id or purpose.value,
            algorithm="hmac-sha256",
            secret=b"full-e2e-matrix-hmac",
        )


class MatrixEncryptionProvider:
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
            ciphertext=b"sealed:" + value,
            key_id=key_id or "matrix",
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
        prefix = b"sealed:"
        assert value.startswith(prefix)
        return value[len(prefix) :]


class PrefixConverter:
    name = "text"
    version = "matrix"
    capability = TaskFileConverterCapability(
        source_mime_types=("text/plain", "application/pdf"),
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
        _ = source_media_type
        prefix = str((options or {}).get("prefix", "converted:"))
        return TaskFileConversionResult(
            content=prefix.encode() + content,
            media_type="text/plain",
            metadata={"privacy": HASHED_MARKER},
        )


class MatrixPdfImageConverter:
    name = "pdf_image"
    version = "matrix"
    capability = TaskFileConverterCapability(
        source_mime_types=("application/pdf",),
        output_mime_types=("image/png",),
        supports_streaming=False,
        max_input_bytes=1024,
        max_output_bytes=1024,
        max_pages=4,
        max_pixels=100_000,
    )

    def __init__(
        self,
        pages: tuple[TaskFileConversionPageResult, ...],
    ) -> None:
        self.pages = pages
        self.calls: list[tuple[bytes, str | None, Mapping[str, object]]] = []

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        _ = content, source_media_type, options
        raise AssertionError("pdf_image must use page conversion")

    async def convert_pages(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionPageCollection:
        self.calls.append((content, source_media_type, dict(options or {})))
        return TaskFileConversionPageCollection(
            pages=self.pages,
            metadata={"renderer": "matrix-private-renderer"},
        )


class MatrixResponse:
    def __init__(
        self,
        text: str,
        *,
        input_token_count: int = 2,
        output_token_count: int = 1,
    ) -> None:
        self.text = text
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count
        self.total_token_count = input_token_count + output_token_count

    async def to_str(self) -> str:
        return self.text

    async def to_json(self) -> str:
        return self.text


class MatrixEventManager:
    def __init__(self) -> None:
        self.listeners: list[Callable[[Event], Awaitable[None] | None]] = []

    def add_listener(
        self,
        listener: Callable[[Event], Awaitable[None] | None],
    ) -> None:
        self.listeners.append(listener)

    def remove_listener(
        self,
        listener: Callable[[Event], Awaitable[None] | None],
    ) -> None:
        self.listeners.remove(listener)

    async def emit_private_token(self) -> None:
        for listener in tuple(self.listeners):
            result = listener(
                Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={
                        "token": "matrix-private-token",
                        "token_id": 77,
                        "file_id": "file-private",
                    },
                )
            )
            if result is not None:
                await result


class MatrixOrchestrator:
    def __init__(self, loader: "MatrixAgentLoader") -> None:
        self.loader = loader
        self.event_manager = loader.event_manager

    async def __aenter__(self) -> "MatrixOrchestrator":
        self.loader.entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        _ = exc_type, exc_value, traceback
        self.loader.exited += 1
        return None

    async def __call__(self, input: object) -> MatrixResponse:
        self.loader.inputs.append(input)
        await self.event_manager.emit_private_token()
        if self.loader.responses:
            response = self.loader.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        return MatrixResponse("public agent output")


class MatrixAgentLoader:
    def __init__(
        self,
        *,
        responses: tuple[MatrixResponse | BaseException, ...] = (),
    ) -> None:
        self.responses = list(responses)
        self.event_manager = MatrixEventManager()
        self.paths: list[str] = []
        self.inputs: list[object] = []
        self.entered = 0
        self.exited = 0

    async def from_file(
        self,
        path: str,
        *,
        agent_id: object | None,
        disable_memory: bool = False,
        uri: str | None = None,
        tool_settings: object | None = None,
    ) -> MatrixOrchestrator:
        _ = agent_id, disable_memory, uri, tool_settings
        self.paths.append(path)
        return MatrixOrchestrator(self)


class MatrixReadingTarget(TaskTargetRunner):
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
            reader: BinaryIO = await context.artifact_store.open(
                file.artifact_ref
            )
            try:
                self.file_bodies.append(reader.read())
            finally:
                reader.close()
        if context.event_listener is not None:
            event_result = context.event_listener(
                Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={
                        "token": "matrix-private-target-token",
                        "token_id": 9,
                        "body": "matrix private event body",
                    },
                )
            )
            if event_result is not None:
                await event_result
        await context.observe_usage(
            SimpleNamespace(
                input_token_count=5,
                output_token_count=3,
                total_token_count=8,
            )
        )
        return "public target output"


class MatrixVendorToolResolver:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.contexts: list[ToolCallContext] = []
        self._descriptors = [
            ToolDescriptor(
                name="vendor_sanctions_check",
                parameter_schema={
                    "type": "object",
                    "required": ["vendor_id", "risk_hint"],
                    "additionalProperties": False,
                    "properties": {
                        "vendor_id": {"type": "string"},
                        "risk_hint": {"type": "string"},
                    },
                },
                return_schema={
                    "type": "object",
                    "required": ["status", "risk_score", "vendor_id"],
                    "additionalProperties": False,
                    "properties": {
                        "status": {"type": "string"},
                        "risk_score": {"type": "integer"},
                        "vendor_id": {"type": "string"},
                    },
                },
            ),
            ToolDescriptor(
                name="vendor_bank_check",
                parameter_schema={
                    "type": "object",
                    "required": ["vendor_id"],
                    "additionalProperties": False,
                    "properties": {"vendor_id": {"type": "string"}},
                },
                return_schema={
                    "type": "object",
                    "required": ["status", "risk_score"],
                    "additionalProperties": False,
                    "properties": {
                        "status": {"type": "string"},
                        "risk_score": {"type": "integer"},
                    },
                },
            ),
        ]

    def list_tools(self) -> list[ToolDescriptor]:
        return list(self._descriptors)

    def resolve_tool_name(
        self,
        name: str,
        *,
        provider_originated: bool = False,
    ) -> ToolNameResolution:
        _ = provider_originated
        names = {descriptor.name for descriptor in self._descriptors}
        if name in names:
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.EXACT,
                canonical_name=name,
                candidates=[name],
            )
        return ToolNameResolution(
            requested_name=name,
            status=ToolNameResolutionStatus.UNKNOWN,
        )

    def validate_tool_call(
        self,
        call: ToolCall,
    ) -> ToolCallDiagnostic | None:
        _ = call
        return None

    async def execute_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
    ) -> ToolCallOutcome:
        self.calls.append(call)
        self.contexts.append(context)
        arguments = call.arguments or {}
        if call.name == "vendor_sanctions_check":
            result = {
                "status": "clear",
                "risk_score": 1,
                "vendor_id": arguments["vendor_id"],
            }
        else:
            result = {"status": "verified", "risk_score": 0}
        return ToolCallResult(
            id=f"{call.name}-result",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result=result,
        )


class MatrixQueue:
    def __init__(self, store: InMemoryTaskStore, clock: MatrixClock) -> None:
        self.store = store
        self.clock = clock
        self.items: dict[str, TaskQueueItem] = {}
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
        _ = idempotency_expires_at
        if idempotency is not None:
            existing = await self.store.lookup_idempotency_key(idempotency)
            if existing is not None:
                run = await self.store.get_run(existing.run_id)
                return TaskQueueSubmission(run=run, created=False)
        run = await self.store.create_run(request, metadata=run_metadata)
        artifact_records = tuple(
            [
                await self.store.append_artifact(
                    run.run_id,
                    ref=artifact.ref,
                    purpose=artifact.purpose,
                    state=artifact.state,
                    provenance=artifact.provenance,
                    retention=artifact.retention,
                    metadata=artifact.metadata,
                )
                for artifact in artifacts
            ]
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
        item = TaskQueueItem(
            queue_item_id=f"queue-item-{self.next_id}",
            run_id=run.run_id,
            queue_name=queue_name,
            state=TaskQueueItemState.AVAILABLE,
            priority=priority,
            available_at=available_at or self.clock.now,
            attempts=0,
            created_at=self.clock.now,
            updated_at=self.clock.now,
            run_state=run.state,
            metadata=queue_metadata or {},
        )
        self.next_id += 1
        self.items[item.queue_item_id] = item
        return TaskQueueSubmission(
            run=run,
            created=True,
            queue_item=item,
            idempotency=idempotency_result,
            artifacts=artifact_records,
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
            if item.queue_name != queue_name:
                continue
            if item.state != TaskQueueItemState.AVAILABLE:
                continue
            if item.available_at > current_time:
                continue
            run = await self.store.assign_claim(
                item.run_id,
                from_states={TaskRunState.QUEUED},
                worker_id=worker_id,
                lease_expires_at=lease_expires_at,
                reason="claimed",
                metadata=metadata,
            )
            assert run.claim is not None
            attempt = await self.store.create_attempt(
                run.run_id,
                claim_token=run.claim.claim_token,
                metadata=metadata,
            )
            claimed = replace(
                item,
                state=TaskQueueItemState.CLAIMED,
                attempts=item.attempts + 1,
                updated_at=current_time,
                run_state=run.state,
                claimed_at=run.claim.claimed_at,
                lease_expires_at=run.claim.lease_expires_at,
                worker_id=worker_id,
                claim_token=run.claim.claim_token,
                heartbeat_at=run.claim.heartbeat_at,
                metadata=metadata or {},
            )
            self.items[item.queue_item_id] = claimed
            return TaskQueueClaim(
                queue_item=claimed,
                run=run,
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
        item = self.items[queue_item_id]
        assert item.claim_token == claim_token
        updated = replace(
            item,
            heartbeat_at=now or self.clock.now,
            lease_expires_at=lease_expires_at,
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
        assert run.last_attempt_id is not None
        attempt = await self.store.transition_attempt(
            run.last_attempt_id,
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
        completed_item = replace(
            item,
            state=TaskQueueItemState.DONE,
            updated_at=now or self.clock.now,
            run_state=completed_run.state,
        )
        self.items[queue_item_id] = completed_item
        return TaskQueueCompletion(
            queue_item=completed_item,
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
        item = self.items[queue_item_id]
        run = await self.store.get_run(item.run_id)
        assert run.last_attempt_id is not None
        attempt = await self.store.transition_attempt(
            run.last_attempt_id,
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.FAILED,
            reason="retry",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        exhausted = attempt.attempt_number >= max_attempts
        next_run_state = (
            TaskRunState.FAILED if exhausted else TaskRunState.QUEUED
        )
        next_item_state = (
            TaskQueueItemState.DEAD
            if exhausted
            else TaskQueueItemState.AVAILABLE
        )
        next_run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=next_run_state,
            reason="retry",
            result=result if exhausted else None,
            claim_token=claim_token,
            metadata=metadata,
        )
        if not exhausted:
            next_run = replace(next_run, claim=None)
            self.store._runs[next_run.run_id] = next_run
        updated = replace(
            item,
            state=next_item_state,
            available_at=available_at,
            updated_at=now or self.clock.now,
            run_state=next_run.state,
        )
        self.items[queue_item_id] = updated
        return TaskQueueRetry(
            queue_item=updated, run=next_run, attempt=attempt
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
        _ = max_attempts
        item = self.items[queue_item_id]
        run = await self.store.get_run(item.run_id)
        assert run.last_attempt_id is not None
        attempt = await self.store.transition_attempt(
            run.last_attempt_id,
            from_states={TaskAttemptState.CREATED, TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.ABANDONED,
            reason="abandoned",
            claim_token=claim_token,
            metadata=metadata,
        )
        failed_run = await self.store.transition_run(
            run.run_id,
            from_states={run.state},
            to_state=TaskRunState.FAILED,
            reason="abandoned",
            claim_token=claim_token,
            metadata=metadata,
        )
        dead = replace(
            item,
            state=TaskQueueItemState.DEAD,
            updated_at=now or self.clock.now,
            run_state=failed_run.state,
        )
        self.items[queue_item_id] = dead
        return TaskQueueAbandonment(
            queue_item=dead,
            run=failed_run,
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
        _ = queue_name, max_attempts, limit, now, metadata
        return ()

    async def drain(
        self,
        queue_name: str,
        *,
        limit: int,
        now: datetime | None = None,
    ) -> tuple[TaskQueueItem, ...]:
        _ = now
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
        _ = now
        available = sum(
            1
            for item in self.items.values()
            if item.queue_name == queue_name
            and item.state == TaskQueueItemState.AVAILABLE
        )
        claimed = sum(
            1
            for item in self.items.values()
            if item.queue_name == queue_name
            and item.state == TaskQueueItemState.CLAIMED
        )
        dead = sum(
            1
            for item in self.items.values()
            if item.queue_name == queue_name
            and item.state == TaskQueueItemState.DEAD
        )
        return TaskQueueDepth(
            queue_name=queue_name,
            available=available,
            claimed=claimed,
            dead=dead,
        )

    async def health(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueHealth:
        current_time = now or self.clock.now
        return TaskQueueHealth(
            queue_name=queue_name,
            depth=await self.depth(queue_name),
            checked_at=current_time,
        )


class MatrixWorkspace:
    def __init__(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.clock = MatrixClock()
        self.store = InMemoryTaskStore(clock=lambda: self.clock.now)
        self.queue = MatrixQueue(self.store, self.clock)
        self.hmac_provider = MatrixHmacProvider()
        self.encryption_provider = MatrixEncryptionProvider()
        self._artifact_ids = iter(
            f"matrix-artifact-{index}" for index in range(1, 100)
        )
        self.artifact_store = LocalArtifactStore(
            self.root / "artifacts",
            raw_storage_allowed=True,
            id_factory=lambda: next(self._artifact_ids),
        )
        self._write_workspace()

    def close(self) -> None:
        self._tmp.cleanup()

    def __enter__(self) -> "MatrixWorkspace":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        _ = exc_type, exc, traceback
        self.close()

    def _write_workspace(self) -> None:
        (self.root / "agents").mkdir()
        (self.root / "uploads").mkdir()
        (self.root / "uploads" / "small.txt").write_bytes(
            b"matrix private small body"
        )
        (self.root / "uploads" / "document.pdf").write_bytes(
            b"matrix private pdf body"
        )
        (self.root / "agents" / "provider.toml").write_text(
            """
[agent]
name = "Provider"
task = "Review files."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
            encoding="utf-8",
        )
        (self.root / "agents" / "local_text.toml").write_text(
            """
[agent]
name = "Local text"
task = "Review text."

[engine]
uri = "ai://local/model"
""",
            encoding="utf-8",
        )
        (self.root / "agents" / "local_multi.toml").write_text(
            """
[agent]
name = "Local multimodal"
task = "Review media."

[engine]
uri = "ai://local/model"
file_delivery_profile = "multimodal"
""",
            encoding="utf-8",
        )

    async def put_artifact(
        self,
        content: bytes,
        *,
        media_type: str,
    ) -> TaskArtifactRef:
        return await self.artifact_store.put(
            content,
            media_type=media_type,
            metadata={"filename": "matrix-private-name.txt"},
        )

    def direct_client(
        self,
        target: TaskTargetRunner,
        *,
        file_converters: Mapping[str, FileConverter] | None = None,
    ) -> TaskClient:
        return TaskClient(
            self.store,
            target=target,
            hmac_provider=self.hmac_provider,
            encryption_provider=self.encryption_provider,
            raw_storage_allowed=True,
            artifact_store=self.artifact_store,
            file_converters=file_converters,
            execution_roots=(self.root,),
            definition_hash=lambda task: f"matrix-{task.task.name}",
            clock=lambda: self.clock.now,
            sleep=self.clock.sleep,
        )

    def queued_client(
        self,
        target: TaskTargetRunner,
        *,
        file_converters: Mapping[str, FileConverter] | None = None,
    ) -> TaskClient:
        return TaskClient(
            self.store,
            target=target,
            queue=cast(TaskQueue, self.queue),
            hmac_provider=self.hmac_provider,
            encryption_provider=self.encryption_provider,
            raw_storage_allowed=True,
            artifact_store=self.artifact_store,
            file_converters=file_converters,
            execution_roots=(self.root,),
            definition_hash=lambda task: f"matrix-{task.task.name}",
            clock=lambda: self.clock.now,
            sleep=self.clock.sleep,
        )

    def worker(
        self,
        target: TaskTargetRunner,
        *,
        file_converters: Mapping[str, FileConverter] | None = None,
    ) -> TaskWorker:
        return TaskWorker(
            self.store,
            cast(TaskQueue, self.queue),
            target=target,
            worker_id="matrix-worker",
            hmac_provider=self.hmac_provider,
            encryption_provider=self.encryption_provider,
            raw_storage_allowed=True,
            artifact_store=self.artifact_store,
            file_converters=file_converters,
            execution_roots=(self.root,),
            clock=lambda: self.clock.now,
        )

    def agent_target(
        self,
        loader: MatrixAgentLoader,
        *,
        uri: str | None = None,
    ) -> AgentTaskTargetRunner:
        return AgentTaskTargetRunner(loader, ref_base=self.root, uri=uri)


class FullE2EMatrixTest(IsolatedAsyncioTestCase):
    async def test_direct_provider_and_local_agent_matrix_is_sanitized(
        self,
    ) -> None:
        with MatrixWorkspace() as workspace:
            provider_loader = MatrixAgentLoader()
            provider_client = workspace.direct_client(
                workspace.agent_target(provider_loader)
            )
            provider_definition = _direct_agent_definition(
                name="provider_matrix",
                ref="agents/provider.toml",
                input_contract=TaskInputContract.file(
                    mime_types=("application/pdf",)
                ),
            )
            provider_result = await provider_client.run(
                provider_definition,
                input_value=TaskClient.provider_file_id(
                    "openai",
                    "file-private",
                    mime_type="application/pdf",
                    size_bytes=32,
                    owner_scope="matrix-tenant-secret",
                ),
            )
            provider_inspection = await provider_client.inspect(
                provider_result.run.run_id
            )
            object_loader = MatrixAgentLoader()
            object_client = workspace.direct_client(
                workspace.agent_target(
                    object_loader,
                    uri="ai://env:KEY@bedrock/us.anthropic.claude",
                )
            )
            object_result = await object_client.run(
                _direct_agent_definition(
                    name="provider_object_matrix",
                    input_contract=TaskInputContract.file(
                        mime_types=("text/plain",)
                    ),
                ),
                input_value=TaskClient.object_store_uri(
                    "bedrock",
                    "s3://matrix-private-bucket/file.txt",
                    mime_type="text/plain",
                    size_bytes=32,
                ),
            )
            object_inspection = await provider_client.inspect(
                object_result.run.run_id
            )
            inline_ref = await workspace.put_artifact(
                b"matrix private inline bytes",
                media_type="application/pdf",
            )
            inline_result = await provider_client.run(
                _direct_agent_definition(name="provider_inline_matrix"),
                input_value="review",
                files=(
                    TaskInputFile(
                        logical_path="artifact:inline",
                        artifact_ref=inline_ref,
                        media_type="application/pdf",
                        size_bytes=inline_ref.size_bytes,
                    ),
                ),
            )
            inline_inspection = await provider_client.inspect(
                inline_result.run.run_id
            )
            local_loader = MatrixAgentLoader(
                responses=(
                    MatrixResponse("public conversion output"),
                    MatrixResponse("public retrieval output"),
                    MatrixResponse("one"),
                    MatrixResponse("two"),
                    MatrixResponse("public map reduce output"),
                    MatrixResponse("public image output"),
                )
            )
            local_client = workspace.direct_client(
                workspace.agent_target(local_loader),
                file_converters={"text": PrefixConverter()},
            )
            conversion_result = await local_client.run(
                _direct_agent_definition(
                    name="local_conversion_matrix",
                    ref="agents/local_text.toml",
                    input_contract=TaskInputContract.file(
                        conversions=("text",),
                        mime_types=("text/plain",),
                    ),
                    limits=TaskLimitsPolicy(total_tokens=20),
                ),
                input_value=TaskClient.local_file(
                    "uploads/small.txt",
                    mime_type="text/plain",
                    conversions=(
                        TaskFileConversionRequest(
                            name="text",
                            options={"prefix": "converted:"},
                        ),
                    ),
                    metadata={"filename": "small.txt"},
                ),
            )
            conversion_inspection = await local_client.inspect(
                conversion_result.run.run_id
            )
            retrieval_ref = await workspace.put_artifact(
                b"zero one two needle five six seven eight",
                media_type="text/plain",
            )
            retrieval_result = await local_client.run(
                _direct_agent_definition(
                    name="local_retrieval_matrix",
                    ref="agents/local_text.toml",
                    limits=TaskLimitsPolicy(total_tokens=5),
                ),
                input_value="needle",
                files=(
                    TaskInputFile(
                        logical_path="artifact:retrieval",
                        artifact_ref=retrieval_ref,
                        media_type="text/plain",
                        size_bytes=retrieval_ref.size_bytes,
                    ),
                ),
            )
            retrieval_inspection = await local_client.inspect(
                retrieval_result.run.run_id
            )
            map_ref = await workspace.put_artifact(
                b"alpha beta gamma delta epsilon zeta",
                media_type="text/plain",
            )
            map_result = await local_client.run(
                _direct_agent_definition(
                    name="local_map_matrix",
                    ref="agents/local_text.toml",
                    limits=TaskLimitsPolicy(total_tokens=4),
                ),
                input_value="summarize",
                files=(
                    TaskInputFile(
                        logical_path="artifact:map",
                        artifact_ref=map_ref,
                        media_type="text/plain",
                        size_bytes=map_ref.size_bytes,
                    ),
                ),
            )
            map_inspection = await local_client.inspect(map_result.run.run_id)
            image_ref = await workspace.put_artifact(
                b"\x89PNG",
                media_type="image/png",
            )
            image_result = await local_client.run(
                _direct_agent_definition(
                    name="local_image_matrix",
                    ref="agents/local_multi.toml",
                ),
                input_value="describe",
                files=(
                    TaskInputFile(
                        logical_path="artifact:image",
                        artifact_ref=image_ref,
                        media_type="image/png",
                        size_bytes=image_ref.size_bytes,
                        metadata={
                            "dimensions": {
                                "width_pixels": 32,
                                "height_pixels": 16,
                            }
                        },
                    ),
                ),
            )
            image_inspection = await local_client.inspect(
                image_result.run.run_id
            )

        provider_block = _only_file_block(provider_loader.inputs[0])
        object_block = _only_file_block(object_loader.inputs[0])
        inline_block = _only_file_block(provider_loader.inputs[1])
        self.assertEqual(provider_block.file["file_id"], "file-private")
        self.assertEqual(
            object_block.file["file_url"],
            "s3://matrix-private-bucket/file.txt",
        )
        self.assertEqual(
            b64decode(cast(str, inline_block.file["file_data"])),
            b"matrix private inline bytes",
        )
        self.assertEqual(provider_result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(object_result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(inline_result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(conversion_result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(retrieval_result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(map_result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(image_result.run.state, TaskRunState.SUCCEEDED)
        conversion_content = _message_texts(local_loader.inputs[0])
        retrieval_content = _message_texts(local_loader.inputs[1])
        map_reduce_content = [
            _message_texts(value) for value in local_loader.inputs[2:5]
        ]
        image_block = _only_image_block(local_loader.inputs[5])
        self.assertIn(
            "converted:matrix private small body", conversion_content
        )
        self.assertEqual(retrieval_content, ["needle", "zero one two needle"])
        self.assertEqual(
            map_reduce_content,
            [
                ["summarize", "alpha beta gamma"],
                ["summarize", "delta epsilon zeta"],
                ["summarize", "one", "two"],
            ],
        )
        self.assertEqual(image_block.image_url["mime_type"], "image/png")
        self.assertEqual(b64decode(image_block.image_url["data"]), b"\x89PNG")
        _assert_no_sentinels(
            (
                provider_inspection.as_dict(),
                object_inspection.as_dict(),
                inline_inspection.as_dict(),
                conversion_inspection.as_dict(),
                retrieval_inspection.as_dict(),
                map_inspection.as_dict(),
                image_inspection.as_dict(),
                _cli_snapshot(provider_inspection.as_dict()),
                _cli_snapshot(image_inspection.as_dict()),
            ),
            (
                "matrix-private-bucket",
                "matrix-private-name",
                "matrix private inline bytes",
                "matrix private small body",
                "matrix private pdf body",
                "matrix-private-token",
                "file-private",
                "matrix-tenant-secret",
                "token_id",
                "small.txt",
            ),
        )

    async def test_queued_conversion_and_provider_reference_matrix(
        self,
    ) -> None:
        with MatrixWorkspace() as workspace:
            reading_target = MatrixReadingTarget()
            queued_client = workspace.queued_client(reading_target)
            reading_worker = workspace.worker(
                reading_target,
                file_converters={"text": PrefixConverter()},
            )
            conversion_definition = _queued_definition(
                name="queued_conversion_matrix",
                input_contract=TaskInputContract.file(
                    conversions=("text",),
                    mime_types=("text/plain",),
                ),
            )
            conversion_submission = await queued_client.enqueue(
                conversion_definition,
                input_value=TaskClient.local_file(
                    "uploads/small.txt",
                    mime_type="text/plain",
                    conversions=(TaskFileConversionRequest(name="text"),),
                    metadata={"filename": "small.txt"},
                ),
                idempotency_key="matrix-private-idempotency",
                owner_scope="matrix-private-owner",
            )
            conversion_processed = await reading_worker.process_once()
            conversion_output = await queued_client.output(
                conversion_submission.run.run_id
            )
            conversion_inspection = await queued_client.inspect(
                conversion_submission.run.run_id
            )
            provider_loader = MatrixAgentLoader()
            provider_target = workspace.agent_target(provider_loader)
            provider_client = workspace.queued_client(provider_target)
            provider_worker = workspace.worker(provider_target)
            provider_submission = await provider_client.enqueue(
                _queued_definition(
                    name="queued_provider_matrix",
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
                    execution=TaskExecutionTarget.agent(
                        "agents/provider.toml"
                    ),
                ),
                input_value={
                    "prompt": "matrix private queued prompt",
                    "document": {
                        "source_kind": "provider_reference",
                        "reference": "file-private",
                        "mime_type": "application/pdf",
                        "size_bytes": 128,
                        "provider_reference": {
                            "kind": TaskProviderReferenceKind.PROVIDER_FILE_ID,
                            "provider": "openai",
                            "reference": "file-private",
                            "mime_type": "application/pdf",
                            "owner_scope": "matrix-private-owner",
                        },
                    },
                },
            )
            provider_processed = await provider_worker.process_once()
            provider_output = await provider_client.output(
                provider_submission.run.run_id
            )
            provider_inspection = await provider_client.inspect(
                provider_submission.run.run_id
            )

        self.assertTrue(conversion_submission.created)
        self.assertTrue(conversion_processed.processed)
        self.assertIsNotNone(conversion_processed.completion)
        self.assertEqual(
            conversion_processed.completion.run.state,
            TaskRunState.SUCCEEDED,
        )
        self.assertEqual(
            reading_target.file_bodies,
            [b"converted:matrix private small body"],
        )
        self.assertEqual(
            conversion_output.output_summary,
            {"privacy": REDACTED_MARKER},
        )
        self.assertEqual(
            [
                cast(Mapping[str, object], artifact)["purpose"]
                for artifact in conversion_inspection.artifacts
            ],
            [
                TaskArtifactPurpose.INPUT.value,
                TaskArtifactPurpose.CONVERTED.value,
            ],
        )
        self.assertTrue(provider_processed.processed)
        self.assertIsNotNone(provider_processed.completion)
        self.assertEqual(
            provider_output.output_summary,
            {"privacy": REDACTED_MARKER},
        )
        queued_block = _only_file_block(provider_loader.inputs[0])
        self.assertEqual(queued_block.file["file_id"], "file-private")
        queue_payload = cast(
            Mapping[str, object],
            provider_inspection.as_dict()["run"],
        )["input_payload"]
        self.assertEqual(queue_payload, {"privacy": ENCRYPTED_MARKER})
        _assert_no_sentinels(
            (
                conversion_inspection.as_dict(),
                provider_inspection.as_dict(),
                _cli_snapshot(conversion_output.as_dict()),
                _cli_snapshot(provider_output.as_dict()),
            ),
            (
                "matrix private small body",
                "matrix-private-idempotency",
                "matrix-private-owner",
                "matrix-private-target-token",
                "matrix-private-token",
                "matrix private queued prompt",
                "file-private",
                "token_id",
                "small.txt",
            ),
        )

    async def test_queued_flow_image_conversion_matrix_is_sanitized(
        self,
    ) -> None:
        with MatrixWorkspace() as workspace:
            (workspace.root / "agents" / "image_provider.toml").write_text(
                """
[agent]
name = "Image provider"
task = "Review rendered pages."
instructions = "matrix private provider instructions"
goal_instructions = "matrix private goal instructions"
user = "matrix private image user prompt"

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            loader = MatrixAgentLoader(
                responses=(MatrixResponse('{"status":"ready","count":2}'),)
            )
            agent_target = workspace.agent_target(loader)
            converter = MatrixPdfImageConverter(
                (
                    TaskFileConversionPageResult(
                        page_index=1,
                        page_count=2,
                        content=b"matrix private first image",
                        media_type="image/png",
                        width_pixels=32,
                        height_pixels=16,
                        metadata={
                            "filename": "matrix-private-page-1.png",
                            "pdf_title": "matrix private pdf title",
                        },
                    ),
                    TaskFileConversionPageResult(
                        page_index=2,
                        page_count=2,
                        content=b"matrix private second image",
                        media_type="image/png",
                        width_pixels=32,
                        height_pixels=16,
                        metadata={"data_url": "data:image/png;base64,private"},
                    ),
                )
            )

            def resolve(context: TaskTargetContext) -> Flow:
                registry = task_flow_node_registry(
                    context,
                    agent_runner=agent_target,
                    execution_roots=(workspace.root,),
                )
                flow = Flow()
                flow.add_node(
                    registry.build(
                        FlowNodeDefinition(
                            name="render",
                            type="file_convert",
                            input="pdf",
                            output="files",
                            config={
                                "converter": "pdf_image",
                                "format": "png",
                                "max_pages": 2,
                                "pages": "1..2",
                            },
                        )
                    )
                )
                flow.add_node(
                    registry.build(
                        FlowNodeDefinition(
                            name="extract",
                            type="agent",
                            ref="agents/image_provider.toml",
                            config={
                                "files_input": "render.files",
                                "file_policy": "replace",
                            },
                        )
                    )
                )
                flow.add_connection("render", "extract")
                return flow

            flow_target = FlowTaskTargetRunner(flow_resolver=resolve)
            client = workspace.queued_client(
                flow_target,
                file_converters={"pdf_image": converter},
            )
            worker = workspace.worker(
                flow_target,
                file_converters={"pdf_image": converter},
            )
            definition_value = _queued_definition(
                name="queued_flow_image_matrix",
                input_contract=TaskInputContract.file(
                    mime_types=("application/pdf",),
                ),
                output_contract=TaskOutputContract.object(
                    {
                        "type": "object",
                        "required": ["status", "count"],
                        "additionalProperties": False,
                        "properties": {
                            "status": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                    }
                ),
                execution=TaskExecutionTarget.flow("flows/image.toml"),
            )

            submission = await client.enqueue(
                definition_value,
                input_value=TaskClient.local_file(
                    "uploads/document.pdf",
                    mime_type="application/pdf",
                    metadata={"filename": "matrix-private-document.pdf"},
                ),
                idempotency_key="matrix-private-flow-window",
                owner_scope="matrix-private-flow-owner",
            )
            processed = await worker.process_once()
            output = await client.output(submission.run.run_id)
            inspection = await client.inspect(submission.run.run_id)
            artifacts = await workspace.store.list_artifacts(
                submission.run.run_id
            )

        self.assertTrue(processed.processed)
        self.assertIsNotNone(processed.completion)
        self.assertEqual(
            processed.completion.run.state, TaskRunState.SUCCEEDED
        )
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            output.output_summary, {"status": "ready", "count": 2}
        )
        self.assertEqual(len(loader.inputs), 1)
        message = loader.inputs[0]
        assert isinstance(message, Message)
        content = cast(list[Any], message.content)
        text_blocks = [
            block for block in content if isinstance(block, MessageContentText)
        ]
        file_blocks = [
            block for block in content if isinstance(block, MessageContentFile)
        ]
        image_blocks = [
            block
            for block in content
            if isinstance(block, MessageContentImage)
        ]
        self.assertEqual(
            [block.text for block in text_blocks],
            ["matrix private image user prompt"],
        )
        self.assertEqual(file_blocks, [])
        self.assertEqual(len(image_blocks), 2)
        self.assertEqual(
            [
                b64decode(cast(str, block.image_url["data"]))
                for block in image_blocks
            ],
            [
                b"matrix private first image",
                b"matrix private second image",
            ],
        )
        self.assertEqual(
            [block.image_url["mime_type"] for block in image_blocks],
            ["image/png", "image/png"],
        )
        self.assertEqual(len(converter.calls), 1)
        self.assertEqual(converter.calls[0][0], b"matrix private pdf body")
        self.assertEqual(
            [artifact.purpose for artifact in artifacts],
            [
                TaskArtifactPurpose.INPUT,
                TaskArtifactPurpose.CONVERTED,
                TaskArtifactPurpose.CONVERTED,
            ],
        )
        queue_payload = cast(
            Mapping[str, object],
            inspection.as_dict()["run"],
        )["input_payload"]
        self.assertEqual(queue_payload, {"privacy": ENCRYPTED_MARKER})
        _assert_no_sentinels(
            (
                inspection.as_dict(),
                output.as_dict(),
                _cli_snapshot(inspection.as_dict()),
                tuple(artifact.summary() for artifact in artifacts),
            ),
            (
                "matrix private provider instructions",
                "matrix private goal instructions",
                "matrix private image user prompt",
                "matrix-private-document.pdf",
                "matrix private pdf body",
                "matrix private first image",
                "matrix private second image",
                "matrix-private-page-1.png",
                "matrix private pdf title",
                "data:image/png;base64,private",
                "matrix-private-renderer",
                "matrix-private-flow-window",
                "matrix-private-flow-owner",
                "matrix-private-token",
                "file-private",
                "token_id",
            ),
        )

    async def test_queued_flow_file_conversion_requires_artifact_store(
        self,
    ) -> None:
        with MatrixWorkspace() as workspace:
            flow_target = FlowTaskTargetRunner(flow_resolver=lambda _: Flow())
            client = TaskClient(
                workspace.store,
                target=flow_target,
                queue=cast(TaskQueue, workspace.queue),
                hmac_provider=workspace.hmac_provider,
                encryption_provider=workspace.encryption_provider,
                raw_storage_allowed=True,
                file_converters={
                    "pdf_image": MatrixPdfImageConverter(
                        (
                            TaskFileConversionPageResult(
                                page_index=1,
                                page_count=1,
                                content=b"matrix private fallback image",
                                media_type="image/png",
                                width_pixels=32,
                                height_pixels=16,
                            ),
                        )
                    )
                },
                execution_roots=(workspace.root,),
                definition_hash=lambda task: f"matrix-{task.task.name}",
                clock=lambda: workspace.clock.now,
                sleep=workspace.clock.sleep,
            )

            with self.assertRaises(TaskValidationError) as error:
                await client.enqueue(
                    _queued_definition(
                        name="queued_flow_image_no_store",
                        input_contract=TaskInputContract.file(
                            mime_types=("application/pdf",),
                        ),
                        execution=TaskExecutionTarget.flow("flows/image.toml"),
                    ),
                    input_value=TaskClient.local_file(
                        "uploads/document.pdf",
                        mime_type="application/pdf",
                        metadata={"filename": "matrix-private-document.pdf"},
                    ),
                    idempotency_key="matrix-private-no-store-window",
                    owner_scope="matrix-private-no-store-owner",
                )

        self.assertEqual(workspace.queue.items, {})
        rendered = str(error.exception)
        self.assertIn("artifact", rendered)
        self.assertNotIn("matrix-private-document.pdf", rendered)
        self.assertNotIn("matrix private pdf body", rendered)
        self.assertNotIn("matrix private fallback image", rendered)
        self.assertNotIn("matrix-private-no-store-window", rendered)
        self.assertNotIn("matrix-private-no-store-owner", rendered)

    async def test_direct_strict_vendor_onboarding_happy_path(
        self,
    ) -> None:
        with MatrixWorkspace() as workspace:
            _write_vendor_agent(workspace)
            converter = MatrixPdfImageConverter(
                (
                    TaskFileConversionPageResult(
                        page_index=1,
                        page_count=1,
                        content=b"matrix private vendor image",
                        media_type="image/png",
                        width_pixels=40,
                        height_pixels=20,
                        metadata={
                            "filename": "matrix-private-vendor-page.png"
                        },
                    ),
                )
            )
            loader = MatrixAgentLoader(
                responses=(
                    MatrixResponse(
                        '{"vendor_name":"Acme Vendor",'
                        '"vendor_id":"vendor-123",'
                        '"risk_hint":"low","document_pages":1}'
                    ),
                )
            )
            tool_resolver = MatrixVendorToolResolver()
            flow_store = InMemoryFlowStateStore()
            flow_target = FlowTaskTargetRunner(
                strict_resolver=lambda _: _strict_vendor_onboarding_plan(),
                flow_state_store=flow_store,
                agent_runner=workspace.agent_target(loader),
                execution_roots=(workspace.root,),
                tool_resolver=tool_resolver,
                concurrency_limit=2,
            )
            client = workspace.direct_client(
                flow_target,
                file_converters={"pdf_image": converter},
            )

            result = await client.run(
                _direct_vendor_definition("direct_vendor_onboarding_matrix"),
                input_value=TaskClient.local_file(
                    "uploads/document.pdf",
                    mime_type="application/pdf",
                    metadata={"filename": "matrix-private-vendor.pdf"},
                ),
            )
            output = await client.output(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)
            record = await flow_store.get_flow_execution(result.run.run_id)
            artifacts = await workspace.store.list_artifacts(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(output.output_summary, {"privacy": REDACTED_MARKER})
        summary = cast(
            Mapping[str, object],
            record.selected_outputs["summary"],
        )
        vendor = cast(Mapping[str, object], summary["vendor"])
        sanctions = cast(Mapping[str, object], summary["sanctions"])
        banking = cast(Mapping[str, object], summary["banking"])
        self.assertEqual(vendor["vendor_id"], "vendor-123")
        self.assertEqual(sanctions["status"], "clear")
        self.assertEqual(sanctions["risk_score"], 1)
        self.assertEqual(banking["status"], "verified")
        self.assertEqual(converter.calls[0][0], b"matrix private pdf body")
        self.assertEqual(len(loader.inputs), 1)
        image_blocks = _image_blocks(loader.inputs[0])
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(
            b64decode(cast(str, image_blocks[0].image_url["data"])),
            b"matrix private vendor image",
        )
        self.assertEqual(
            sorted(call.name for call in tool_resolver.calls),
            ["vendor_bank_check", "vendor_sanctions_check"],
        )
        self.assertTrue(
            all(context.flow_tool_node for context in tool_resolver.contexts)
        )
        node_states = {node.node: node.state for node in record.trace.nodes}
        self.assertEqual(node_states["approved"], FlowNodeState.SUCCEEDED)
        self.assertEqual(node_states["review"], FlowNodeState.SKIPPED)
        self.assertEqual(dict(record.pause_tokens), {})
        self.assertEqual(
            {artifact.purpose for artifact in artifacts},
            {TaskArtifactPurpose.INPUT, TaskArtifactPurpose.CONVERTED},
        )
        _assert_no_sentinels(
            (
                inspection.as_dict(),
                output.as_dict(),
                record.metadata,
                record.selected_outputs,
                tuple(artifact.summary() for artifact in artifacts),
                _cli_snapshot(inspection.as_dict()),
            ),
            (
                "matrix private pdf body",
                "matrix private vendor image",
                "matrix-private-vendor-page.png",
                "matrix-private-vendor.pdf",
            ),
        )

    async def test_direct_strict_vendor_onboarding_invalid_input_stops(
        self,
    ) -> None:
        with MatrixWorkspace() as workspace:
            _write_vendor_agent(workspace)
            converter = MatrixPdfImageConverter(
                (
                    TaskFileConversionPageResult(
                        page_index=1,
                        page_count=1,
                        content=b"matrix private invalid vendor image",
                        media_type="image/png",
                        width_pixels=40,
                        height_pixels=20,
                    ),
                )
            )
            loader = MatrixAgentLoader()
            tool_resolver = MatrixVendorToolResolver()
            flow_store = InMemoryFlowStateStore()
            flow_target = FlowTaskTargetRunner(
                strict_resolver=lambda _: _strict_vendor_onboarding_plan(
                    vendor_id=None
                ),
                flow_state_store=flow_store,
                agent_runner=workspace.agent_target(loader),
                execution_roots=(workspace.root,),
                tool_resolver=tool_resolver,
                concurrency_limit=2,
            )
            client = workspace.direct_client(
                flow_target,
                file_converters={"pdf_image": converter},
            )

            result = await client.run(
                _direct_vendor_definition(
                    "direct_vendor_onboarding_invalid_matrix"
                ),
                input_value=TaskClient.local_file(
                    "uploads/document.pdf",
                    mime_type="application/pdf",
                    metadata={"filename": "matrix-private-invalid-vendor.pdf"},
                ),
            )
            output = await client.output(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)
            record = await flow_store.get_flow_execution(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(converter.calls, [])
        self.assertEqual(loader.inputs, [])
        self.assertEqual(tool_resolver.calls, [])
        node_states = {node.node: node.state for node in record.trace.nodes}
        self.assertEqual(
            node_states["validate_metadata"],
            FlowNodeState.FAILED,
        )
        self.assertEqual(node_states["render"], FlowNodeState.SKIPPED)
        self.assertIn(
            "flow.execution.validation_failed",
            {diagnostic.code for diagnostic in record.diagnostics},
        )
        _assert_no_sentinels(
            (
                inspection.as_dict(),
                output.as_dict(),
                record.metadata,
                _cli_snapshot(inspection.as_dict()),
            ),
            (
                "matrix private pdf body",
                "matrix private invalid vendor image",
                "matrix-private-invalid-vendor.pdf",
            ),
        )

    async def test_strict_subflow_matrix_matches_direct_and_queue(
        self,
    ) -> None:
        with MatrixWorkspace() as workspace:
            flow_target = FlowTaskTargetRunner(
                strict_resolver=lambda _: _strict_matrix_subflow_plan()
            )
            direct_client = workspace.direct_client(flow_target)
            queued_client = workspace.queued_client(flow_target)
            worker = workspace.worker(flow_target)

            direct_result = await direct_client.run(
                _direct_flow_definition(name="direct_subflow_matrix"),
                input_value="matrix private subflow prompt",
            )
            submission = await queued_client.enqueue(
                _queued_definition(
                    name="queued_subflow_matrix",
                    execution=TaskExecutionTarget.flow("flows/subflow.toml"),
                ),
                input_value="matrix private subflow prompt",
                idempotency_key="matrix-private-subflow-idempotency",
                owner_scope="matrix-private-subflow-owner",
            )
            processed = await worker.process_once()
            queued_output = await queued_client.output(submission.run.run_id)
            direct_inspection = await direct_client.inspect(
                direct_result.run.run_id
            )
            queued_inspection = await queued_client.inspect(
                submission.run.run_id
            )

        self.assertIsNotNone(processed.completion)
        self.assertEqual(direct_result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(queued_output.state, TaskRunState.SUCCEEDED)
        self.assertEqual(direct_result.output, "subflow:public result")
        self.assertEqual(processed.output, direct_result.output)
        self.assertEqual(
            queued_output.output_summary,
            {"privacy": REDACTED_MARKER},
        )
        _assert_no_sentinels(
            (
                direct_inspection.as_dict(),
                queued_inspection.as_dict(),
                _cli_snapshot(queued_inspection.as_dict()),
            ),
            (
                "matrix private subflow prompt",
                "matrix-private-subflow-idempotency",
                "matrix-private-subflow-owner",
            ),
        )

    async def test_queued_strict_subflow_failure_is_sanitized(
        self,
    ) -> None:
        with MatrixWorkspace() as workspace:
            flow_target = FlowTaskTargetRunner(
                strict_resolver=lambda _: _strict_matrix_subflow_plan(
                    failing=True
                )
            )
            queued_client = workspace.queued_client(flow_target)
            worker = workspace.worker(flow_target)

            submission = await queued_client.enqueue(
                _queued_definition(
                    name="queued_subflow_failure_matrix",
                    execution=TaskExecutionTarget.flow("flows/subflow.toml"),
                ),
                input_value="matrix private failing subflow prompt",
                idempotency_key="matrix-private-subflow-failure",
                owner_scope="matrix-private-subflow-failure-owner",
            )
            processed = await worker.process_once()
            result = await queued_client.output(submission.run.run_id)
            inspection = await queued_client.inspect(submission.run.run_id)

        self.assertTrue(processed.processed)
        self.assertIsNone(processed.completion)
        self.assertIsNone(processed.retry)
        self.assertEqual(result.state, TaskRunState.FAILED)
        error = cast(Mapping[str, object], result.error)
        self.assertEqual(error["code"], "runnable.failed")
        _assert_no_sentinels(
            (result.as_dict(), inspection.as_dict()),
            (
                "matrix private failing subflow prompt",
                "matrix-private-subflow-failure",
                "matrix-private-subflow-failure-owner",
                "matrix private child failure",
            ),
        )

    async def test_negative_matrix_failures_keep_private_values_out(
        self,
    ) -> None:
        with MatrixWorkspace() as workspace:
            provider_loader = MatrixAgentLoader()
            provider_client = workspace.direct_client(
                workspace.agent_target(provider_loader)
            )
            large_validation = await provider_client.validate(
                _direct_agent_definition(
                    name="large_contract_matrix",
                    input_contract=TaskInputContract.file(
                        mime_types=("application/pdf",)
                    ),
                    limits=TaskLimitsPolicy(file_bytes=4),
                ),
                input_value=TaskClient.provider_file_id(
                    "openai",
                    "file-private",
                    mime_type="application/pdf",
                    size_bytes=10_000_000_000,
                    size_bucket="huge-private-bucket",
                ),
            )
            converter_definition = _direct_agent_definition(
                name="missing_converter_matrix",
                ref="agents/local_text.toml",
                input_contract=TaskInputContract.file(
                    conversions="missing",
                    mime_types=("text/plain",),
                ),
            )
            with self.assertRaises(TaskValidationError):
                await provider_client.run(
                    converter_definition,
                    input_value=TaskClient.local_file(
                        "uploads/small.txt",
                        mime_type="text/plain",
                        conversions=(
                            TaskFileConversionRequest(name="missing"),
                        ),
                        metadata={"filename": "small.txt"},
                    ),
                )
            converter_validation = await provider_client.validate(
                converter_definition,
                input_value=TaskClient.local_file(
                    "uploads/small.txt",
                    mime_type="text/plain",
                    conversions=(TaskFileConversionRequest(name="missing"),),
                    metadata={"filename": "small.txt"},
                ),
            )
            remote_definition = _direct_agent_definition(
                name="remote_url_matrix",
                input_contract=TaskInputContract.file(
                    mime_types=("text/plain",)
                ),
            )
            with self.assertRaises(TaskValidationError):
                await provider_client.run(
                    remote_definition,
                    input_value=TaskClient.remote_url_file(
                        "http://localhost/private",
                        mime_type="text/plain",
                        metadata={"filename": "remote-secret.txt"},
                    ),
                )
            remote_validation = await provider_client.validate(
                remote_definition,
                input_value=TaskClient.remote_url_file(
                    "http://localhost/private",
                    mime_type="text/plain",
                    metadata={"filename": "remote-secret.txt"},
                ),
            )
            escaping_result = await provider_client.run(
                _direct_agent_definition(
                    name="path_escape_matrix",
                    input_contract=TaskInputContract.file(
                        mime_types=("text/plain",)
                    ),
                ),
                input_value=TaskClient.local_file(
                    "../uploads/small.txt",
                    mime_type="text/plain",
                    metadata={"filename": "small.txt"},
                ),
            )
            escaping_inspection = await provider_client.inspect(
                escaping_result.run.run_id
            )
            failing_loader = MatrixAgentLoader(
                responses=(
                    RuntimeError(
                        "matrix private provider failure file-private"
                    ),
                )
            )
            failing_client = workspace.direct_client(
                workspace.agent_target(failing_loader)
            )
            provider_failure = await failing_client.run(
                _direct_agent_definition(name="provider_failure_matrix"),
                input_value="matrix private provider prompt",
            )
            provider_failure_inspection = await failing_client.inspect(
                provider_failure.run.run_id
            )

        self.assertFalse(large_validation.valid)
        self.assertFalse(converter_validation.valid)
        self.assertFalse(remote_validation.valid)
        self.assertEqual(escaping_result.run.state, TaskRunState.FAILED)
        self.assertEqual(provider_failure.run.state, TaskRunState.FAILED)
        self.assertEqual(provider_loader.inputs, [])
        self.assertEqual(len(failing_loader.inputs), 1)
        _assert_error_code(escaping_result, "input_contract.failed")
        _assert_error_code(provider_failure, "runnable.failed")
        _assert_no_sentinels(
            (
                large_validation.as_dict(),
                converter_validation.as_dict(),
                remote_validation.as_dict(),
                escaping_inspection.as_dict(),
                provider_failure_inspection.as_dict(),
                _cli_snapshot(provider_failure_inspection.as_dict()),
            ),
            (
                "../uploads/small.txt",
                "file-private",
                "huge-private-bucket",
                "http://localhost/private",
                "matrix private provider failure",
                "matrix private provider prompt",
                "matrix private small body",
                "remote-secret.txt",
                "small.txt",
            ),
        )


def _direct_agent_definition(
    *,
    name: str,
    ref: str = "agents/provider.toml",
    input_contract: TaskInputContract | None = None,
    output_contract: TaskOutputContract | None = None,
    limits: TaskLimitsPolicy | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name=name, version="1"),
        input=input_contract or TaskInputContract.string(),
        output=output_contract or TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent(ref),
        run=TaskRunPolicy.direct(timeout_seconds=60),
        privacy=TaskPrivacyPolicy(raw_retention_days=1),
        artifact=TaskArtifactPolicy.references_only(retention_days=3),
        limits=limits or TaskLimitsPolicy(),
        observability=TaskObservabilityPolicy(
            metrics=True,
            trace=False,
            capture_events=True,
        ),
        retry=TaskRetryPolicy(max_attempts=1),
    )


def _direct_flow_definition(*, name: str) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name=name, version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.flow("flows/subflow.toml"),
        run=TaskRunPolicy.direct(timeout_seconds=60),
        privacy=TaskPrivacyPolicy(raw_retention_days=1),
        artifact=TaskArtifactPolicy.references_only(retention_days=3),
        observability=TaskObservabilityPolicy(
            metrics=True,
            trace=False,
            capture_events=True,
        ),
        retry=TaskRetryPolicy(max_attempts=1),
    )


def _queued_definition(
    *,
    name: str,
    input_contract: TaskInputContract | None = None,
    output_contract: TaskOutputContract | None = None,
    execution: TaskExecutionTarget | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name=name, version="1"),
        input=input_contract or TaskInputContract.string(),
        output=output_contract or TaskOutputContract.text(),
        execution=execution
        or TaskExecutionTarget.agent("agents/provider.toml"),
        run=TaskRunPolicy.queued("default"),
        privacy=TaskPrivacyPolicy(raw_retention_days=1),
        artifact=TaskArtifactPolicy.references_only(retention_days=3),
        observability=TaskObservabilityPolicy(
            metrics=True,
            trace=False,
            capture_events=True,
        ),
        retry=TaskRetryPolicy(max_attempts=1),
    )


def _direct_vendor_definition(name: str) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name=name, version="1"),
        input=TaskInputContract.file(mime_types=("application/pdf",)),
        output=TaskOutputContract.object(_vendor_output_schema()),
        execution=TaskExecutionTarget.flow("flows/vendor_onboarding.toml"),
        run=TaskRunPolicy.direct(timeout_seconds=60),
        privacy=TaskPrivacyPolicy(raw_retention_days=1),
        artifact=TaskArtifactPolicy.references_only(retention_days=3),
        observability=TaskObservabilityPolicy(
            metrics=True,
            trace=False,
            capture_events=True,
        ),
        retry=TaskRetryPolicy(max_attempts=1),
    )


def _write_vendor_agent(workspace: MatrixWorkspace) -> None:
    (workspace.root / "agents" / "vendor_provider.toml").write_text(
        """
[agent]
name = "Vendor provider"
task = "Extract vendor metadata."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
        encoding="utf-8",
    )


def _vendor_output_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "vendor": {"type": "object"},
            "sanctions": {"type": "object"},
            "banking": {"type": "object"},
        },
    }


def _strict_vendor_onboarding_plan(
    *,
    vendor_id: str | None = "vendor-123",
) -> FlowExecutionPlan:
    dynamic_object = FlowNodeContract(
        name="value",
        type=FlowOutputType.OBJECT,
        metadata={"dynamic": True},
    )
    dynamic_result = FlowNodeContract(
        name="result",
        type=FlowOutputType.JSON,
        metadata={"dynamic": True},
    )
    return FlowExecutionPlan(
        name="vendor-onboarding",
        version="1",
        revision=None,
        inputs=(
            FlowInputDefinition(
                name="documents",
                type=FlowInputType.FILE_ARRAY,
                mime_types=("application/pdf",),
            ),
        ),
        outputs=(
            FlowOutputDefinition(name="summary", type=FlowOutputType.OBJECT),
        ),
        entry_node="metadata",
        output_selectors={
            "summary": parse_flow_selector("approved.value"),
        },
        nodes=(
            FlowNodePlan(
                name="metadata",
                type="constant",
                kind=FlowNodeKind.CONSTANT,
                output_contracts=(dynamic_object,),
                config={"value": _vendor_metadata(vendor_id=vendor_id)},
            ),
            FlowNodePlan(
                name="validate_metadata",
                type="validation",
                kind=FlowNodeKind.VALIDATION,
                input_contracts=(
                    FlowNodeContract(
                        name="value",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                output_contracts=(dynamic_object,),
                mappings=(
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("metadata.value"),
                    ),
                ),
                config={
                    "value_type": "object",
                    "required_fields": ("vendor_name", "vendor_id"),
                },
            ),
            FlowNodePlan(
                name="render",
                type="pdf_to_images",
                kind=FlowNodeKind.FILE_CONVERSION,
                input_contracts=(
                    FlowNodeContract(
                        name="files",
                        type=FlowInputType.FILE_ARRAY,
                    ),
                ),
                output_contracts=(
                    FlowNodeContract(
                        name="files",
                        type=FlowOutputType.FILE_ARRAY,
                    ),
                ),
                capabilities=(
                    FlowNodeCapability.ASYNC_ONLY,
                    FlowNodeCapability.TASK_BACKED,
                ),
                mappings=(
                    FlowMappingPlan(
                        target="files",
                        kind=FlowMappingKind.FILE_ARRAY,
                        source=parse_flow_selector("input.documents"),
                    ),
                ),
                config={"format": "png", "max_pages": 1, "pages": "1"},
            ),
            FlowNodePlan(
                name="extract",
                type="agent",
                kind=FlowNodeKind.AGENT,
                ref="agents/vendor_provider.toml",
                input_contracts=(
                    FlowNodeContract(name="input", type="any"),
                    FlowNodeContract(
                        name=None,
                        type="object",
                        metadata={"dynamic": True},
                    ),
                ),
                output_contracts=(dynamic_result,),
                capabilities=(
                    FlowNodeCapability.ASYNC_ONLY,
                    FlowNodeCapability.TASK_BACKED,
                ),
                mappings=(
                    FlowMappingPlan(
                        target="input",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("validate_metadata.value"),
                    ),
                    FlowMappingPlan(
                        target="render",
                        kind=FlowMappingKind.OBJECT,
                        fields={
                            "files": parse_flow_selector("render.files"),
                        },
                    ),
                ),
                join=FlowJoinPlan(type=FlowJoinPolicyType.ALL_SUCCESS),
                config={
                    "files_input": "render.files",
                    "file_policy": "replace",
                },
            ),
            _strict_vendor_tool_node(
                name="sanctions",
                ref="vendor_sanctions_check",
                mappings=(
                    FlowMappingPlan(
                        target="arguments",
                        kind=FlowMappingKind.OBJECT,
                        fields={
                            "vendor_id": parse_flow_selector(
                                "extract.result.vendor_id"
                            ),
                            "risk_hint": parse_flow_selector(
                                "extract.result.risk_hint"
                            ),
                        },
                    ),
                ),
                argument_bindings={
                    "vendor_id": "vendor_id",
                    "risk_hint": "risk_hint",
                },
            ),
            _strict_vendor_tool_node(
                name="banking",
                ref="vendor_bank_check",
                mappings=(
                    FlowMappingPlan(
                        target="arguments",
                        kind=FlowMappingKind.OBJECT,
                        fields={
                            "vendor_id": parse_flow_selector(
                                "extract.result.vendor_id"
                            ),
                        },
                    ),
                ),
                argument_bindings={"vendor_id": "vendor_id"},
            ),
            FlowNodePlan(
                name="checks",
                type="join",
                kind=FlowNodeKind.JOIN,
                output_contracts=(dynamic_object,),
                mappings=(
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.OBJECT,
                        fields={
                            "vendor": parse_flow_selector("extract.result"),
                            "sanctions": parse_flow_selector(
                                "sanctions.result"
                            ),
                            "banking": parse_flow_selector("banking.result"),
                        },
                    ),
                ),
                join=FlowJoinPlan(type=FlowJoinPolicyType.ALL_SUCCESS),
            ),
            FlowNodePlan(
                name="risk",
                type="decision",
                kind=FlowNodeKind.DECISION,
                input_contracts=(
                    FlowNodeContract(
                        name="value",
                        type=FlowInputType.OBJECT,
                        metadata={"dynamic": True},
                    ),
                ),
                output_contracts=(dynamic_object,),
                mappings=(
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("checks.value"),
                    ),
                ),
            ),
            FlowNodePlan(
                name="approved",
                type="pass-through",
                kind=FlowNodeKind.PASS_THROUGH,
                input_contracts=(
                    FlowNodeContract(
                        name="value",
                        type=FlowInputType.OBJECT,
                        metadata={"dynamic": True},
                    ),
                ),
                output_contracts=(dynamic_object,),
                mappings=(
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("risk.value"),
                    ),
                ),
            ),
            FlowNodePlan(
                name="review",
                type="human_review",
                kind=FlowNodeKind.HUMAN_REVIEW,
                input_contracts=(
                    FlowNodeContract(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                output_contracts=(dynamic_result,),
                capabilities=(
                    FlowNodeCapability.ASYNC_ONLY,
                    FlowNodeCapability.DURABLE_PAUSE,
                ),
                mappings=(
                    FlowMappingPlan(
                        target="payload",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("risk.value"),
                    ),
                ),
                config=_review_node_config(),
            ),
            _review_sink("review_approved"),
            _review_sink("review_rejected"),
            _review_sink("review_correction"),
            _review_sink("review_expired"),
        ),
        edges=_strict_vendor_edges(),
    )


def _vendor_metadata(*, vendor_id: str | None) -> Mapping[str, object]:
    metadata: dict[str, object] = {"vendor_name": "Acme Vendor"}
    if vendor_id is not None:
        metadata["vendor_id"] = vendor_id
    return metadata


def _strict_vendor_tool_node(
    *,
    name: str,
    ref: str,
    mappings: tuple[FlowMappingPlan, ...],
    argument_bindings: Mapping[str, str],
) -> FlowNodePlan:
    return FlowNodePlan(
        name=name,
        type="tool",
        kind=FlowNodeKind.TOOL,
        ref=ref,
        input_contracts=(
            FlowNodeContract(
                name="arguments",
                type=FlowInputType.OBJECT,
                metadata={"dynamic": True},
            ),
        ),
        output_contracts=(
            FlowNodeContract(
                name="result",
                type=FlowOutputType.JSON,
                metadata={"dynamic": True},
            ),
        ),
        capabilities=(FlowNodeCapability.ASYNC_ONLY,),
        mappings=mappings,
        config={"arguments": argument_bindings},
    )


def _review_sink(name: str) -> FlowNodePlan:
    return FlowNodePlan(
        name=name,
        type="pass-through",
        kind=FlowNodeKind.PASS_THROUGH,
        input_contracts=(
            FlowNodeContract(
                name="value",
                type=FlowInputType.OBJECT,
                metadata={"dynamic": True},
            ),
        ),
        output_contracts=(
            FlowNodeContract(
                name="value",
                type=FlowOutputType.OBJECT,
                metadata={"dynamic": True},
            ),
        ),
        mappings=(
            FlowMappingPlan(
                target="value",
                kind=FlowMappingKind.SELECT,
                source=parse_flow_selector("review.result"),
            ),
        ),
    )


def _review_node_config() -> Mapping[str, object]:
    decision_schema = {
        "type": "object",
        "required": ["decision"],
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": [
                    "approved",
                    "rejected",
                    "needs-correction",
                    "expired",
                ],
            },
        },
    }
    return {
        "allowed_decisions": (
            "approved",
            "rejected",
            "needs-correction",
            "expired",
        ),
        "decision_schema": decision_schema,
        "payload_schema": {"type": "object", "additionalProperties": True},
        "timeout_seconds": 3600,
        "audit_metadata": {"queue": "vendor"},
    }


def _strict_vendor_edges() -> tuple[FlowEdgePlan, ...]:
    return (
        FlowEdgePlan(
            index=0,
            source="metadata",
            target="validate_metadata",
            kind=FlowEdgeKind.SUCCESS,
        ),
        FlowEdgePlan(
            index=1,
            source="validate_metadata",
            target="render",
            kind=FlowEdgeKind.SUCCESS,
            routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
        ),
        FlowEdgePlan(
            index=2,
            source="validate_metadata",
            target="extract",
            kind=FlowEdgeKind.SUCCESS,
            routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
        ),
        FlowEdgePlan(
            index=3,
            source="render",
            target="extract",
            kind=FlowEdgeKind.SUCCESS,
        ),
        FlowEdgePlan(
            index=4,
            source="extract",
            target="sanctions",
            kind=FlowEdgeKind.SUCCESS,
            routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
        ),
        FlowEdgePlan(
            index=5,
            source="extract",
            target="banking",
            kind=FlowEdgeKind.SUCCESS,
            routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
        ),
        FlowEdgePlan(
            index=6,
            source="extract",
            target="checks",
            kind=FlowEdgeKind.SUCCESS,
            routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
        ),
        FlowEdgePlan(
            index=7,
            source="sanctions",
            target="checks",
            kind=FlowEdgeKind.SUCCESS,
        ),
        FlowEdgePlan(
            index=8,
            source="banking",
            target="checks",
            kind=FlowEdgeKind.SUCCESS,
        ),
        FlowEdgePlan(
            index=9,
            source="checks",
            target="risk",
            kind=FlowEdgeKind.SUCCESS,
        ),
        FlowEdgePlan(
            index=10,
            source="risk",
            target="approved",
            kind=FlowEdgeKind.SUCCESS,
            condition=FlowConditionPlan(
                operator=FlowConditionOperator.LT,
                selector=parse_flow_selector(
                    "risk.value.sanctions.risk_score"
                ),
                value=3,
            ),
            priority=0,
        ),
        FlowEdgePlan(
            index=11,
            source="risk",
            target="review",
            kind=FlowEdgeKind.SUCCESS,
            condition=FlowConditionPlan(
                operator=FlowConditionOperator.GTE,
                selector=parse_flow_selector(
                    "risk.value.sanctions.risk_score"
                ),
                value=3,
            ),
            priority=1,
        ),
        FlowEdgePlan(
            index=12,
            source="review",
            target="review_approved",
            kind=FlowEdgeKind.RESUME,
            label="approved",
        ),
        FlowEdgePlan(
            index=13,
            source="review",
            target="review_rejected",
            kind=FlowEdgeKind.RESUME,
            label="rejected",
        ),
        FlowEdgePlan(
            index=14,
            source="review",
            target="review_correction",
            kind=FlowEdgeKind.RESUME,
            label="needs-correction",
        ),
        FlowEdgePlan(
            index=15,
            source="review",
            target="review_expired",
            kind=FlowEdgeKind.RESUME,
            label="expired",
        ),
        FlowEdgePlan(
            index=16,
            source="review",
            target="review_expired",
            kind=FlowEdgeKind.TIMEOUT,
        ),
    )


async def _strict_matrix_subflow_plan(
    *,
    failing: bool = False,
) -> FlowExecutionPlan:
    child_plan = (
        _failing_matrix_child_plan()
        if failing
        else await _strict_matrix_child_plan()
    )
    return FlowExecutionPlan(
        name="matrix-subflow",
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
                        "plan": child_plan,
                        "output_mapping": {"result": "answer"},
                    }
                },
            ),
        ),
        edges=(),
    )


async def _strict_matrix_child_plan() -> FlowExecutionPlan:
    result = await compile_flow_definition(
        FlowDefinition(
            name="matrix-child",
            version="1",
            inputs=(
                FlowInputDefinition(
                    name="prompt",
                    type=FlowInputType.STRING,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.TEXT,
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="answer"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "answer.value"}
            ),
            nodes=(
                FlowNodeDefinition(
                    name="answer",
                    type="constant",
                    config={"value": "subflow:public result"},
                ),
            ),
        )
    )
    assert result.plan is not None
    return result.plan


def _failing_matrix_child_plan() -> FlowExecutionPlan:
    return FlowExecutionPlan(
        name="matrix-child-failure",
        version="1",
        revision=None,
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.TEXT),
        ),
        entry_node="answer",
        output_selectors={"answer": parse_flow_selector("answer.value")},
        nodes=(
            FlowNodePlan(
                name="answer",
                type="subflow",
                kind=FlowNodeKind.SUBFLOW,
                output_contracts=(
                    FlowNodeContract(
                        name="value",
                        type=FlowOutputType.TEXT,
                    ),
                ),
            ),
        ),
        edges=(),
    )


def _file_blocks(input_value: object) -> tuple[MessageContentFile, ...]:
    assert isinstance(input_value, Message)
    content = input_value.content
    assert isinstance(content, list)
    return tuple(
        block
        for block in cast(list[Any], content)
        if isinstance(block, MessageContentFile)
    )


def _only_file_block(input_value: object) -> MessageContentFile:
    blocks = _file_blocks(input_value)
    assert len(blocks) == 1
    return blocks[0]


def _image_blocks(input_value: object) -> tuple[MessageContentImage, ...]:
    assert isinstance(input_value, Message)
    content = input_value.content
    assert isinstance(content, list)
    return tuple(
        block
        for block in cast(list[Any], content)
        if isinstance(block, MessageContentImage)
    )


def _only_image_block(input_value: object) -> MessageContentImage:
    blocks = _image_blocks(input_value)
    assert len(blocks) == 1
    return blocks[0]


def _message_texts(input_value: object) -> list[str]:
    assert isinstance(input_value, Message)
    content = input_value.content
    assert isinstance(content, list)
    return [
        block.text
        for block in cast(list[Any], content)
        if isinstance(block, MessageContentText)
    ]


def _cli_snapshot(value: object) -> str:
    return task_cmds._format_task_cli_value(value)


def _assert_error_code(result: object, code: str) -> None:
    run_result = result.run.result
    assert run_result is not None
    error = cast(Mapping[str, object], run_result.error)
    assert error["code"] == code


def _assert_no_sentinels(
    snapshots: tuple[object, ...],
    sentinels: tuple[str, ...],
) -> None:
    rendered = "\n".join(str(snapshot) for snapshot in snapshots)
    for sentinel in sentinels:
        assert sentinel not in rendered


if __name__ == "__main__":
    main()
