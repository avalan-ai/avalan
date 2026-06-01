from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.event import Event, EventType
from avalan.task import (
    IdempotencyMode,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactStat,
    TaskClient,
    TaskClientUnsupportedOperationError,
    TaskClientValidationResult,
    TaskClientWaitTimeoutError,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskFileDescriptor,
    TaskIdempotencyIdentity,
    TaskInputContract,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskQueue,
    TaskQueueArtifact,
    TaskQueueItem,
    TaskQueueItemState,
    TaskQueueSubmission,
    TaskRun,
    TaskRunPolicy,
    TaskRunState,
    TaskTargetContext,
    TaskTargetRunner,
    TaskValidationCategory,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
    UsageSource,
)
from avalan.task.stores import InMemoryTaskStore
from avalan.task.targets import AgentTaskTargetRunner


class FakeResponse:
    input_token_count = 3
    output_token_count = 2

    def __init__(self, text: str) -> None:
        self.text = text

    async def to_str(self) -> str:
        return self.text


class FakeEventManager:
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

    async def trigger(self, event: Event) -> None:
        for listener in tuple(self.listeners):
            result = listener(event)
            if result is not None:
                await result


class FakeOrchestrator:
    def __init__(self, loader: "FakeLoader") -> None:
        self._loader = loader
        self.event_manager = loader.event_manager

    async def __aenter__(self) -> "FakeOrchestrator":
        self._loader.entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        self._loader.exited += 1
        return None

    async def __call__(self, input: object) -> FakeResponse:
        self._loader.inputs.append(input)
        await self.event_manager.trigger(
            Event(
                type=EventType.TOKEN_GENERATED,
                payload={
                    "token": "secret-token",
                    "token_id": 9,
                    "status": "ok",
                },
            )
        )
        return FakeResponse("short summary")


class FakeLoader:
    def __init__(self) -> None:
        self.event_manager = FakeEventManager()
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
    ) -> FakeOrchestrator:
        _ = path, agent_id, disable_memory, uri, tool_settings
        return FakeOrchestrator(self)


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
            secret=b"test-secret",
        )


class RejectingTarget(TaskTargetRunner):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return (
            TaskValidationIssue(
                code="execution.unknown_target",
                path="execution.ref",
                message="Task target could not be loaded.",
                hint="Use a supported execution target.",
                category=TaskValidationCategory.UNSUPPORTED,
            ),
        )

    async def run(self, context: TaskTargetContext) -> object:
        return "unused"


class RecordingArtifactStore:
    def __init__(self) -> None:
        self.puts: list[tuple[bytes, str | None, object]] = []

    async def put(
        self,
        content: bytes,
        *,
        artifact_id: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskArtifactRef:
        self.puts.append((content, media_type, metadata))
        artifact_number = len(self.puts)
        artifact_name = artifact_id or f"artifact-{artifact_number}"
        return TaskArtifactRef(
            artifact_id=artifact_name,
            store="memory",
            storage_key=f"artifacts/{artifact_name}",
            media_type=media_type,
            size_bytes=len(content),
            sha256=sha256(content).hexdigest(),
        )

    async def open(self, ref: TaskArtifactRef) -> BinaryIO:
        _ = ref
        raise NotImplementedError

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        return TaskArtifactStat(
            ref=ref,
            size_bytes=ref.size_bytes or 0,
            sha256=ref.sha256 or ("0" * 64),
        )

    async def delete(self, ref: TaskArtifactRef) -> None:
        _ = ref


class RecordingQueue:
    def __init__(self, store: InMemoryTaskStore) -> None:
        self.store = store
        self.requests: list[TaskExecutionRequest] = []
        self.artifacts: tuple[TaskQueueArtifact, ...] = ()
        self.idempotency: object = None
        self.queue_metadata: object = None
        self.priority = 0
        self.available_at: datetime | None = None
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

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
        self.requests.append(request)
        self.artifacts = artifacts
        self.idempotency = idempotency
        self.queue_metadata = queue_metadata
        self.priority = priority
        self.available_at = available_at
        run = await self.store.create_run(
            request,
            metadata=run_metadata,
        )
        records = []
        for artifact in artifacts:
            records.append(
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
        item = TaskQueueItem(
            queue_item_id="queue-item-1",
            run_id=queued.run_id,
            queue_name=queue_name,
            state=TaskQueueItemState.AVAILABLE,
            priority=priority,
            available_at=available_at or self.now,
            attempts=0,
            created_at=self.now,
            updated_at=self.now,
            run_state=queued.state,
            metadata=queue_metadata or {},
        )
        return TaskQueueSubmission(
            run=queued,
            created=True,
            queue_item=item,
            artifacts=tuple(records),
        )


class TaskClientTest(IsolatedAsyncioTestCase):
    async def test_agent_backed_direct_run_is_inspectable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
task = "Answer"
instructions = "Be brief."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            store = InMemoryTaskStore()
            loader = FakeLoader()
            client = TaskClient(
                store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                definition_hash=lambda task: "client-direct-hash",
                execution_roots=(root,),
            )

            result = await client.run(
                _definition(),
                input_value="private prompt",
                metadata={"request": 1},
            )
            output = await client.output(result.run.run_id)
            events = await client.events(result.run.run_id)
            usage = await client.usage(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "short summary")
        self.assertEqual(loader.inputs, ["private prompt"])
        self.assertEqual(loader.entered, 1)
        self.assertEqual(loader.exited, 1)
        self.assertTrue(output.ready)
        self.assertEqual(output.output_summary, {"privacy": "<redacted>"})
        self.assertIsNone(output.error)
        self.assertEqual(
            [event.event_type for event in events], ["token_generated"]
        )
        self.assertNotIn("secret-token", str(events[0].payload))
        self.assertNotIn("token_id", str(events[0].payload))
        self.assertEqual(usage[0].source, UsageSource.ESTIMATED)
        self.assertEqual(usage[0].totals.input_tokens, 3)
        self.assertEqual(usage[0].totals.output_tokens, 2)
        self.assertIsNone(usage[0].totals.total_tokens)
        self.assertEqual(inspection.run.run_id, result.run.run_id)
        self.assertEqual(inspection.output, output)
        self.assertEqual(inspection.events, events)
        self.assertEqual(inspection.usage, usage)
        self.assertIsNone(inspection.usage_totals.total_tokens)
        self.assertEqual(inspection.artifacts, ())
        self.assertEqual(loader.event_manager.listeners, [])

    async def test_direct_callable_target_uses_shared_validation(
        self,
    ) -> None:
        async def target(context: TaskTargetContext) -> object:
            _ = context
            return "callable summary"

        client = TaskClient(
            InMemoryTaskStore(),
            target=target,
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "client-callable-hash",
        )

        validation = await client.validate(
            _definition(),
            input_value="private prompt",
        )
        result = await client.run(_definition(), input_value="private prompt")

        self.assertTrue(validation.valid)
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "callable summary")

    async def test_validate_aggregates_definition_input_and_target_issues(
        self,
    ) -> None:
        client = TaskClient(
            InMemoryTaskStore(),
            target=RejectingTarget(),
        )

        result = await client.validate(
            _definition(privacy=TaskPrivacyPolicy()),
            input_value={"raw": "not text"},
        )

        self.assertFalse(result.valid)
        self.assertEqual(
            [issue.code for issue in result.issues],
            [
                "privacy.hmac_key_missing",
                "input.invalid_type",
                "execution.unknown_target",
            ],
        )
        self.assertNotIn("not text", str(result.issues))
        with self.assertRaises(TaskValidationError):
            result.raise_for_issues()

    async def test_validation_result_accepts_empty_issue_set(self) -> None:
        result = TaskClientValidationResult()

        self.assertTrue(result.valid)
        result.raise_for_issues()

    async def test_queued_run_and_enqueue_return_stable_diagnostic(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        client = TaskClient(
            store,
            target=RejectingTarget(),
            hmac_provider=StaticHmacProvider(),
        )
        definition = _definition(run=TaskRunPolicy.queued("private-queue"))

        with self.assertRaises(TaskClientUnsupportedOperationError) as run:
            await client.run(definition, input_value="private prompt")
        with self.assertRaises(TaskClientUnsupportedOperationError) as enqueue:
            await client.enqueue(definition, input_value="private prompt")
        direct_client = TaskClient(
            store,
            target=RejectingTarget(),
            queue=cast(TaskQueue, RecordingQueue(store)),
            hmac_provider=StaticHmacProvider(),
        )
        with self.assertRaises(TaskClientUnsupportedOperationError) as direct:
            await direct_client.enqueue(
                _definition(),
                input_value="private prompt",
            )

        self.assertEqual(run.exception.code, "task.queue_unsupported")
        self.assertEqual(run.exception.operation, "run")
        self.assertEqual(enqueue.exception.operation, "enqueue")
        self.assertEqual(direct.exception.operation, "enqueue")
        self.assertNotIn("private-queue", str(run.exception))

    async def test_enqueue_uses_default_definition_hash(self) -> None:
        store = InMemoryTaskStore()
        queue = RecordingQueue(store)
        client = TaskClient(
            store,
            target=_noop_target,
            queue=cast(TaskQueue, queue),
            hmac_provider=StaticHmacProvider(),
        )

        submission = await client.enqueue(
            _definition(
                run=TaskRunPolicy.queued(
                    "documents",
                    idempotency=IdempotencyMode.NONE,
                )
            ),
            input_value="private prompt",
        )

        self.assertEqual(len(submission.run.definition_id), 64)
        self.assertIsNone(queue.requests[0].idempotency_key)

    async def test_enqueue_materializes_files_and_waits_for_terminal_output(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "private.txt"
            input_path.write_text("private file body", encoding="utf-8")
            store = InMemoryTaskStore()
            queue = RecordingQueue(store)
            artifact_store = RecordingArtifactStore()
            client = TaskClient(
                store,
                target=_noop_target,
                queue=cast(TaskQueue, queue),
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                definition_hash=lambda task: "client-queue-hash",
                execution_roots=(root,),
                sleep=_no_sleep,
            )
            available_at = datetime(2026, 1, 1, 12, tzinfo=UTC)
            definition = _definition(
                input_contract=TaskInputContract.file(
                    mime_types=("text/plain",)
                ),
                run=TaskRunPolicy.queued("documents", priority=7),
            )

            submission = await client.enqueue(
                definition,
                input_value=TaskFileDescriptor.local_path(
                    "private.txt",
                    mime_type="text/plain",
                    metadata={"filename": "private.txt"},
                ),
                available_at=available_at,
                idempotency_key="private-idempotency-key",
                idempotency_expires_at=available_at + timedelta(days=1),
                owner_scope="customer-123",
                queue_metadata={"tenant": "safe"},
            )
            await store.transition_run(
                submission.run.run_id,
                from_states={TaskRunState.QUEUED},
                to_state=TaskRunState.FAILED,
                reason="worker_failed",
                result=TaskExecutionResult(error={"code": "runnable.failed"}),
            )
            output = await client.wait(
                submission.run.run_id,
                timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

        self.assertTrue(submission.created)
        self.assertEqual(submission.run.state, TaskRunState.QUEUED)
        queue_item = submission.queue_item
        self.assertIsNotNone(queue_item)
        assert queue_item is not None
        self.assertEqual(queue_item.priority, 7)
        self.assertEqual(queue_item.available_at, available_at)
        self.assertEqual(queue.priority, 7)
        self.assertEqual(queue.queue_metadata, {"tenant": "safe"})
        self.assertIsNotNone(queue.idempotency)
        self.assertEqual(len(queue.requests), 1)
        request = queue.requests[0]
        rendered_request = str(request)
        self.assertEqual(request.queue, "documents")
        self.assertNotIn("private file body", rendered_request)
        self.assertNotIn("private.txt", rendered_request)
        self.assertNotIn("customer-123", str(queue.idempotency))
        self.assertNotIn("private-idempotency-key", rendered_request)
        self.assertEqual(len(artifact_store.puts), 1)
        self.assertEqual(artifact_store.puts[0][0], b"private file body")
        self.assertEqual(len(queue.artifacts), 1)
        self.assertEqual(queue.artifacts[0].purpose, TaskArtifactPurpose.INPUT)
        self.assertEqual(
            queue.artifacts[0].retention,
            TaskArtifactRetention(
                delete_after_days=definition.artifact.retention_days
            ),
        )
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertFalse(output.ready)
        self.assertEqual(output.error, {"code": "runnable.failed"})

    async def test_cancel_requests_queued_run_cancellation(self) -> None:
        store = InMemoryTaskStore()
        queue = RecordingQueue(store)
        client = TaskClient(
            store,
            target=_noop_target,
            queue=cast(TaskQueue, queue),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "client-cancel-hash",
        )
        submission = await client.enqueue(
            _definition(run=TaskRunPolicy.queued("documents")),
            input_value="private prompt",
        )

        cancelled = await client.cancel(submission.run.run_id)
        repeated = await client.cancel(submission.run.run_id)

        self.assertEqual(cancelled.state, TaskRunState.CANCEL_REQUESTED)
        self.assertEqual(repeated.state, TaskRunState.CANCEL_REQUESTED)
        pending = await store.create_run(
            TaskExecutionRequest(definition_id="client-cancel-hash")
        )
        with self.assertRaises(TaskClientUnsupportedOperationError) as error:
            await client.cancel(pending.run_id)
        self.assertEqual(error.exception.operation, "cancel")

        await store.transition_run(
            pending.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.FAILED,
            reason="failed",
        )
        terminal = await client.cancel(pending.run_id)
        self.assertEqual(terminal.state, TaskRunState.FAILED)

    async def test_wait_timeout_uses_stable_diagnostic(self) -> None:
        store = InMemoryTaskStore()
        definition = _definition()
        await store.register_definition(
            definition,
            definition_hash="hash-wait",
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id="hash-wait")
        )
        client = TaskClient(
            store,
            target=_noop_target,
            hmac_provider=StaticHmacProvider(),
        )

        with self.assertRaises(TaskClientWaitTimeoutError) as error:
            await client.wait(
                run.run_id,
                timeout_seconds=0,
                poll_interval_seconds=0.01,
            )

        self.assertEqual(error.exception.code, "task.wait_timeout")
        self.assertEqual(error.exception.operation, "wait")
        self.assertEqual(error.exception.run_id, run.run_id)
        self.assertNotIn(run.run_id, str(error.exception))

    async def test_wait_polls_until_terminal_state(self) -> None:
        store = InMemoryTaskStore()
        definition = _definition()
        await store.register_definition(
            definition,
            definition_hash="hash-poll",
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id="hash-poll")
        )

        async def finish_without_deadline(delay: float) -> None:
            _ = delay
            await store.transition_run(
                run.run_id,
                from_states={TaskRunState.CREATED},
                to_state=TaskRunState.FAILED,
                reason="failed",
            )

        client = TaskClient(
            store,
            target=_noop_target,
            hmac_provider=StaticHmacProvider(),
            sleep=finish_without_deadline,
        )
        output = await client.wait(run.run_id, poll_interval_seconds=0.01)

        await store.register_definition(
            definition,
            definition_hash="hash-deadline",
        )
        deadline_run = await store.create_run(
            TaskExecutionRequest(definition_id="hash-deadline")
        )

        async def finish_with_deadline(delay: float) -> None:
            _ = delay
            await store.transition_run(
                deadline_run.run_id,
                from_states={TaskRunState.CREATED},
                to_state=TaskRunState.FAILED,
                reason="failed",
            )

        deadline_client = TaskClient(
            store,
            target=_noop_target,
            hmac_provider=StaticHmacProvider(),
            sleep=finish_with_deadline,
        )
        deadline_output = await deadline_client.wait(
            deadline_run.run_id,
            timeout_seconds=1,
            poll_interval_seconds=0.01,
        )

        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertEqual(deadline_output.state, TaskRunState.FAILED)

    async def test_output_and_artifacts_reflect_store_records(self) -> None:
        store = InMemoryTaskStore()
        definition = _definition()
        await store.register_definition(
            definition,
            definition_hash="hash-manual",
        )
        pending = await store.create_run(
            TaskExecutionRequest(definition_id="hash-manual")
        )
        persisted = await store.create_run(
            TaskExecutionRequest(definition_id="hash-manual")
        )
        tuple_refs = await store.create_run(
            TaskExecutionRequest(definition_id="hash-manual")
        )
        mapping_ref = await store.create_run(
            TaskExecutionRequest(definition_id="hash-manual")
        )
        client = TaskClient(
            store,
            target=RejectingTarget(),
            hmac_provider=StaticHmacProvider(),
        )

        pending_output = await client.output(pending.run_id)
        pending_artifacts = await client.artifacts(pending.run_id)
        await store.append_artifact(
            persisted.run_id,
            ref=TaskArtifactRef(
                artifact_id="artifact-persisted",
                store="local",
                storage_key="ar/artifact-persisted",
                media_type="text/plain",
                size_bytes=4,
                sha256="a" * 64,
            ),
            purpose=TaskArtifactPurpose.OUTPUT,
            metadata={"safe": "metadata"},
        )
        tuple_refs = await _fail_run(
            store,
            tuple_refs.run_id,
            metadata={"artifacts": ("artifact-1", "artifact-2")},
        )
        mapping_ref = await _fail_run(
            store,
            mapping_ref.run_id,
            metadata={"artifacts": {"artifact_id": "artifact-3"}},
        )

        self.assertFalse(pending_output.ready)
        self.assertIsNone(pending_output.output_summary)
        self.assertEqual(pending_artifacts, ())
        self.assertEqual(
            (await client.output(tuple_refs.run_id)).error,
            {"code": "runnable.failed"},
        )
        persisted_artifacts = await client.artifacts(persisted.run_id)
        self.assertEqual(len(persisted_artifacts), 1)
        self.assertNotIn("storage_key", str(persisted_artifacts))
        self.assertIn("artifact-persisted", str(persisted_artifacts))
        self.assertEqual(
            await client.artifacts(tuple_refs.run_id),
            ("artifact-1", "artifact-2"),
        )
        self.assertEqual(
            await client.artifacts(mapping_ref.run_id),
            ({"artifact_id": "artifact-3"},),
        )


def _definition(
    *,
    input_contract: TaskInputContract | None = None,
    privacy: TaskPrivacyPolicy | None = None,
    run: TaskRunPolicy | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="agent", version="1"),
        input=input_contract or TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/valid.toml"),
        privacy=privacy or TaskPrivacyPolicy(),
        run=run or TaskRunPolicy.direct(),
    )


async def _noop_target(context: TaskTargetContext) -> object:
    _ = context
    return "unused"


async def _no_sleep(delay: float) -> None:
    _ = delay


async def _fail_run(
    store: InMemoryTaskStore,
    run_id: str,
    *,
    metadata: dict[str, object],
) -> TaskRun:
    await store.transition_run(
        run_id,
        from_states={TaskRunState.CREATED},
        to_state=TaskRunState.VALIDATED,
        reason="validated",
    )
    await store.transition_run(
        run_id,
        from_states={TaskRunState.VALIDATED},
        to_state=TaskRunState.RUNNING,
        reason="started",
    )
    return await store.transition_run(
        run_id,
        from_states={TaskRunState.RUNNING},
        to_state=TaskRunState.FAILED,
        reason="failed",
        result=TaskExecutionResult(
            error={"code": "runnable.failed"},
            metadata=metadata,
        ),
    )


if __name__ == "__main__":
    main()
