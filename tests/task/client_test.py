from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.event import Event, EventType
from avalan.task import (
    REDACTED_MARKER,
    EncryptedPrivacyValue,
    IdempotencyMode,
    PrivacySanitizer,
    RunMode,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactStat,
    TaskAttemptState,
    TaskClient,
    TaskClientUnsupportedOperationError,
    TaskClientValidationResult,
    TaskClientWaitTimeoutError,
    TaskDefinition,
    TaskDefinitionLoader,
    TaskEventCategory,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskFileDescriptor,
    TaskIdempotencyIdentity,
    TaskInputContract,
    TaskInputFile,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskProviderReferenceKind,
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
    UsageTotals,
    read_artifact_stream_bytes,
    spec_hash,
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
        self.deleted: list[TaskArtifactRef] = []

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

    async def put_stream(
        self,
        stream: BinaryIO,
        *,
        artifact_id: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
        max_bytes: int | None = None,
        expected_size_bytes: int | None = None,
        expected_sha256: str | None = None,
    ) -> TaskArtifactRef:
        content = read_artifact_stream_bytes(
            stream,
            max_bytes=max_bytes,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        )
        return await self.put(
            content,
            artifact_id=artifact_id,
            media_type=media_type,
            metadata=metadata,
        )

    async def open(self, ref: TaskArtifactRef) -> BinaryIO:
        _ = ref
        return BytesIO()

    async def open_stream(
        self,
        ref: TaskArtifactRef,
        *,
        max_bytes: int | None = None,
    ) -> BinaryIO:
        _ = max_bytes
        return await self.open(ref)

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        return TaskArtifactStat(
            ref=ref,
            size_bytes=ref.size_bytes or 0,
            sha256=ref.sha256 or ("0" * 64),
        )

    async def delete(self, ref: TaskArtifactRef) -> None:
        self.deleted.append(ref)


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


class FailingQueue(RecordingQueue):
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
        _ = (
            request,
            queue_name,
            priority,
            available_at,
            idempotency,
            idempotency_expires_at,
            artifacts,
            run_metadata,
            queue_metadata,
        )
        raise RuntimeError("private queue persistence failure")


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
            inspection_snapshot = cast(
                Mapping[str, object],
                inspection.as_dict(),
            )

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
        self.assertNotIn("private prompt", str(inspection_snapshot))
        self.assertNotIn("secret-token", str(inspection_snapshot))
        self.assertNotIn("token_id", str(inspection_snapshot))
        self.assertNotIn("claim_token", str(inspection_snapshot))
        run_snapshot = cast(
            Mapping[str, object],
            inspection_snapshot["run"],
        )
        self.assertIn("hmac-sha256", str(run_snapshot["input_summary"]))
        event_snapshots = cast(
            tuple[Mapping[str, object], ...],
            inspection_snapshot["events"],
        )
        self.assertEqual(event_snapshots[0]["sequence"], 1)
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
        self.assertEqual(validation.as_dict(), {"valid": True, "issues": ()})
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
        self.assertFalse(cast(Mapping[str, object], result.as_dict())["valid"])
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

    async def test_sdk_file_descriptor_helpers_match_loaded_contract(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agents").mkdir()
            (root / "agents" / "reviewer.toml").write_text(
                """
[agent]
name = "Reviewer"
task = "Review the uploaded file."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            task_path = root / "review.task.toml"
            task_path.write_text(
                """
[task]
name = "document_review"
version = "1"

[input]
type = "file"
file_conversions = ["text"]
mime_types = ["application/pdf"]

[output]
type = "text"

[execution]
type = "agent"
ref = "agents/reviewer.toml"
""",
                encoding="utf-8",
            )
            loaded_definition = TaskDefinitionLoader().load(task_path)
            sdk_definition = TaskDefinition(
                task=TaskMetadata(name="document_review", version="1"),
                input=TaskInputContract.file(
                    conversions=("text",),
                    mime_types=("application/pdf",),
                ),
                output=TaskOutputContract.text(),
                execution=TaskExecutionTarget.agent("agents/reviewer.toml"),
            )
            client = TaskClient(
                InMemoryTaskStore(),
                target=_noop_target,
                hmac_provider=StaticHmacProvider(),
            )
            descriptor = TaskClient.local_file(
                "report.pdf",
                role="source",
                mime_type="application/pdf",
                size_bytes=2048,
                sha256="a" * 64,
                conversions=(
                    TaskClient.file_conversion(
                        "text",
                        options={"encoding": "utf-8"},
                    ),
                ),
                metadata={"source": "safe"},
            )

            validation = await client.validate(
                sdk_definition,
                input_value=descriptor,
            )

        self.assertEqual(
            spec_hash(loaded_definition), spec_hash(sdk_definition)
        )
        self.assertTrue(validation.valid)
        self.assertEqual(descriptor.reference, "report.pdf")
        self.assertEqual(descriptor.role, "source")
        self.assertEqual(descriptor.mime_type, "application/pdf")
        self.assertEqual(descriptor.size_bytes, 2048)
        self.assertEqual(descriptor.sha256, "a" * 64)
        self.assertEqual(descriptor.conversions[0].name, "text")
        self.assertEqual(
            descriptor.conversions[0].options,
            {"encoding": "utf-8"},
        )

    async def test_direct_provider_reference_runs_without_artifact_store(
        self,
    ) -> None:
        seen_files: list[tuple[TaskInputFile, ...]] = []

        async def target(context: TaskTargetContext) -> object:
            seen_files.append(context.files)
            return "accepted"

        client = TaskClient(
            InMemoryTaskStore(),
            target=target,
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "client-provider-direct-hash",
        )
        descriptor = TaskClient.provider_file_id(
            "openai",
            "file-private",
            role="source",
            mime_type="application/pdf",
            size_bytes=2048,
            sha256="b" * 64,
            size_bucket="1KB-1MB",
            identity_hmac="hmac-value",
            owner_scope="tenant-a",
            metadata={"classification": "safe"},
        )

        result = await client.run(
            _definition(
                input_contract=TaskInputContract.file(
                    mime_types=("application/pdf",)
                )
            ),
            input_value=descriptor,
        )
        inspection = await client.inspect(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "accepted")
        self.assertEqual(len(seen_files), 1)
        file = seen_files[0][0]
        self.assertIsNone(file.artifact_ref)
        self.assertEqual(file.media_type, "application/pdf")
        self.assertEqual(file.size_bytes, 2048)
        self.assertIsNotNone(file.provider_reference)
        assert file.provider_reference is not None
        self.assertEqual(file.provider_reference.provider, "openai")
        self.assertEqual(
            file.provider_reference.kind,
            TaskProviderReferenceKind.PROVIDER_FILE_ID,
        )
        self.assertEqual(file.provider_reference.reference, "file-private")
        self.assertEqual(inspection.artifacts, ())
        self.assertNotIn("file-private", str(inspection.as_dict()))
        self.assertNotIn("tenant-a", str(inspection.as_dict()))

    async def test_direct_provider_reference_conversion_is_rejected(
        self,
    ) -> None:
        target_calls = 0

        async def target(context: TaskTargetContext) -> object:
            nonlocal target_calls
            target_calls += 1
            return "unused"

        client = TaskClient(
            InMemoryTaskStore(),
            target=target,
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "client-provider-conversion-hash",
        )

        with self.assertRaises(TaskValidationError) as error:
            await client.run(
                _definition(
                    input_contract=TaskInputContract.file(
                        conversions=("text",)
                    )
                ),
                input_value={
                    "source_kind": "provider_reference",
                    "reference": "file-private",
                    "provider_reference": {
                        "kind": "provider_file_id",
                        "provider": "openai",
                        "reference": "file-private",
                    },
                    "conversions": [{"name": "text"}],
                },
            )

        self.assertEqual(target_calls, 0)
        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(
            error.exception.issues[0].path,
            "input.conversions",
        )
        self.assertNotIn("file-private", str(error.exception))

    async def test_sdk_file_helpers_cover_source_variants(self) -> None:
        expires_at = datetime(2026, 1, 1, tzinfo=UTC)

        remote = TaskClient.remote_url_file(
            "https://example.test/private.txt",
            mime_type="text/plain",
            conversions=(
                "native",
                {"name": "text", "options": {"mode": "safe"}},
            ),
        )
        artifact = TaskClient.artifact_file(
            "artifact-private",
            role="context",
            size_bytes=12,
        )
        inline = TaskClient.inline_file(
            "cHJpdmF0ZQ==",
            mime_type="text/plain",
            sha256="c" * 64,
        )
        hosted = TaskClient.hosted_url(
            "openai",
            "https://example.test/private.pdf",
            expires_at=expires_at,
            durable=False,
            mime_type="application/pdf",
        )
        object_uri = TaskClient.object_store_uri(
            "google",
            "gs://bucket/private.pdf",
            identity_hmac="hmac-value",
            size_bucket="1KB-1MB",
        )

        self.assertEqual(remote.source_kind.value, "remote_url")
        self.assertEqual(remote.conversions[0].name, "native")
        self.assertEqual(remote.conversions[1].options, {"mode": "safe"})
        self.assertEqual(artifact.source_kind.value, "artifact")
        self.assertEqual(artifact.role, "context")
        self.assertEqual(inline.source_kind.value, "inline_bytes")
        self.assertEqual(inline.sha256, "c" * 64)
        self.assertIsNotNone(hosted.provider_reference)
        assert hosted.provider_reference is not None
        self.assertEqual(
            hosted.provider_reference.kind,
            TaskProviderReferenceKind.HOSTED_URL,
        )
        self.assertEqual(hosted.provider_reference.expires_at, expires_at)
        self.assertFalse(hosted.provider_reference.durable)
        self.assertIsNotNone(object_uri.provider_reference)
        assert object_uri.provider_reference is not None
        self.assertEqual(
            object_uri.provider_reference.kind,
            TaskProviderReferenceKind.OBJECT_STORE_URI,
        )
        self.assertEqual(object_uri.provider_reference.provider, "google")

    async def test_sdk_file_helper_rejects_invalid_conversion_shape(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            TaskClient.local_file(
                "report.pdf",
                conversions=({"options": {}},),
            )

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
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
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
        self.assertIsNotNone(queue.requests[0].input_payload)
        self.assertNotIn("private prompt", str(queue.requests[0]))

    async def test_enqueue_rejects_scalar_input_without_payload_storage(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        queue = RecordingQueue(store)
        client = TaskClient(
            store,
            target=_noop_target,
            queue=cast(TaskQueue, queue),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "client-queue-missing-payload-hash",
        )

        with self.assertRaises(TaskValidationError) as error:
            await client.enqueue(
                _definition(
                    run=TaskRunPolicy.queued(
                        "documents",
                        idempotency=IdempotencyMode.NONE,
                    ),
                    privacy=TaskPrivacyPolicy(),
                ),
                input_value="private prompt",
            )

        self.assertEqual(queue.requests, [])
        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            [
                "queue.input_payload_unavailable",
                "queue.input_payload_unavailable",
                "privacy.encryption_key_missing",
            ],
        )
        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["privacy.raw_retention_days", "privacy", "privacy.input"],
        )
        self.assertNotIn("private prompt", str(error.exception))

    async def test_enqueue_rejects_non_json_payload_input_safely(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        queue = RecordingQueue(store)
        client = TaskClient(
            store,
            target=_noop_target,
            queue=cast(TaskQueue, queue),
            hmac_provider=StaticHmacProvider(),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
            definition_hash=lambda task: "client-queue-bad-payload-hash",
        )

        definition = _definition(
            input_contract=TaskInputContract.number(),
            run=TaskRunPolicy.queued(
                "documents",
                idempotency=IdempotencyMode.NONE,
            ),
        )
        sanitizer = PrivacySanitizer(
            definition.privacy,
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
        )

        with self.assertRaises(TaskValidationError) as error:
            client._queue_input_payload(
                definition,
                float("nan"),
                sanitizer,
            )

        self.assertEqual(queue.requests, [])
        self.assertEqual(len(error.exception.issues), 1)
        issue = error.exception.issues[0]
        self.assertEqual(issue.code, "queue.input_payload_unavailable")
        self.assertEqual(issue.path, "input")
        self.assertNotIn("nan", str(error.exception).lower())

    async def test_enqueue_uses_explicit_queue_name(self) -> None:
        store = InMemoryTaskStore()
        queue = RecordingQueue(store)
        client = TaskClient(
            store,
            target=_noop_target,
            queue=cast(TaskQueue, queue),
            hmac_provider=StaticHmacProvider(),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
            definition_hash=lambda task: "client-queue-override-hash",
        )
        definition = _definition(
            run=TaskRunPolicy.queued(
                "documents",
                idempotency=IdempotencyMode.NONE,
            )
        )

        submission = await client.enqueue(
            definition,
            input_value="private prompt",
            queue_name="priority-documents",
        )

        self.assertEqual(queue.requests[0].queue, "priority-documents")
        self.assertEqual(
            submission.run.request.queue,
            "priority-documents",
        )
        queue_item = submission.queue_item
        self.assertIsNotNone(queue_item)
        assert queue_item is not None
        self.assertEqual(queue_item.queue_name, "priority-documents")
        with self.assertRaises(AssertionError):
            await client.enqueue(
                definition,
                input_value="private prompt",
                queue_name=" ",
            )

    async def test_enqueue_sanitizes_sensitive_queue_metadata(self) -> None:
        store = InMemoryTaskStore()
        queue = RecordingQueue(store)
        client = TaskClient(
            store,
            target=_noop_target,
            queue=cast(TaskQueue, queue),
            hmac_provider=StaticHmacProvider(),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
            definition_hash=lambda task: "client-queue-safe-metadata-hash",
        )

        queue_metadata = cast(
            Mapping[str, object],
            {
                1: "private invalid key",
                "tenant": "safe",
                "copied": "prefix private prompt suffix",
                "path": "private.txt",
                "provider_file_id": "file-openai-private",
                "idempotency_key": "private-window",
                "owner_scope": "private-owner",
                "blob": b"private bytes",
                "enabled": True,
                "limit": 2,
                "ratio": 0.5,
                "invalid_ratio": float("nan"),
                "items": ["safe list", "private prompt", 3],
                "nested": {"label": "safe nested"},
                "optional": None,
            },
        )

        submission = await client.enqueue(
            _definition(
                input_contract=TaskInputContract.object(
                    schema={
                        "type": "object",
                        "required": ["prompt"],
                        "additionalProperties": False,
                        "properties": {
                            "prompt": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    }
                ),
                run=TaskRunPolicy.queued(
                    "documents",
                    idempotency=IdempotencyMode.INPUT_HASH,
                ),
            ),
            input_value={"prompt": ["private prompt"]},
            idempotency_key="private-window",
            owner_scope="private-owner",
            queue_metadata=queue_metadata,
        )

        redacted = {"privacy": REDACTED_MARKER}
        metadata = cast(Mapping[str, object], queue.queue_metadata)
        self.assertEqual(metadata["metadata"], redacted)
        self.assertEqual(metadata["tenant"], "safe")
        self.assertEqual(metadata["copied"], redacted)
        self.assertEqual(metadata["path"], redacted)
        self.assertEqual(metadata["provider_file_id"], redacted)
        self.assertEqual(metadata["idempotency_key"], redacted)
        self.assertEqual(metadata["owner_scope"], redacted)
        self.assertEqual(metadata["blob"], redacted)
        self.assertIs(metadata["enabled"], True)
        self.assertEqual(metadata["limit"], 2)
        self.assertEqual(metadata["ratio"], 0.5)
        self.assertEqual(metadata["invalid_ratio"], redacted)
        self.assertEqual(metadata["items"], ("safe list", redacted, 3))
        self.assertEqual(
            metadata["nested"],
            {"label": "safe nested"},
        )
        self.assertIsNone(metadata["optional"])
        self.assertIsNotNone(submission.queue_item)
        assert submission.queue_item is not None
        rendered_item = str(submission.queue_item)
        self.assertNotIn("private prompt", rendered_item)
        self.assertNotIn("private.txt", rendered_item)
        self.assertNotIn("file-openai-private", rendered_item)
        self.assertNotIn("private-window", rendered_item)
        self.assertNotIn("private-owner", rendered_item)

    async def test_enqueue_persists_explicit_durable_files(self) -> None:
        store = InMemoryTaskStore()
        queue = RecordingQueue(store)
        client = TaskClient(
            store,
            target=_noop_target,
            queue=cast(TaskQueue, queue),
            hmac_provider=StaticHmacProvider(),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
            definition_hash=lambda task: "client-explicit-file-hash",
        )
        ref = TaskArtifactRef(
            artifact_id="artifact-explicit",
            store="local",
            storage_key="artifacts/artifact-explicit",
            media_type="application/pdf",
            size_bytes=42,
            sha256="a" * 64,
            metadata={"filename": "private-ref.pdf"},
        )

        submission = await client.enqueue(
            _definition(
                run=TaskRunPolicy.queued(
                    "documents",
                    idempotency=IdempotencyMode.NONE,
                )
            ),
            input_value="private prompt",
            files=(
                TaskInputFile(
                    logical_path="private-report.pdf",
                    artifact_ref=ref,
                    media_type="application/pdf",
                    size_bytes=42,
                    metadata={"name": "private-report.pdf"},
                ),
            ),
        )
        artifacts = await store.list_artifacts(submission.run.run_id)

        self.assertEqual(len(queue.artifacts), 1)
        self.assertEqual(queue.artifacts[0].ref.artifact_id, ref.artifact_id)
        self.assertIn("privacy", queue.artifacts[0].ref.metadata)
        self.assertNotIn("private-ref", str(queue.artifacts[0].ref.metadata))
        self.assertEqual(
            queue.artifacts[0].provenance.operation,
            "client_attachment",
        )
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].ref.artifact_id, ref.artifact_id)
        self.assertIn("privacy", artifacts[0].ref.metadata)
        self.assertNotIn("private-ref", str(artifacts[0].ref.metadata))
        self.assertIn("privacy", queue.artifacts[0].metadata)
        self.assertNotIn("private-report", str(queue.artifacts[0].metadata))
        self.assertNotIn(
            "private-report",
            str(queue.artifacts[0].provenance.metadata),
        )
        self.assertNotIn("private-ref", str(artifacts))

    async def test_enqueue_rejects_non_durable_explicit_files(self) -> None:
        store = InMemoryTaskStore()
        queue = RecordingQueue(store)
        client = TaskClient(
            store,
            target=_noop_target,
            queue=cast(TaskQueue, queue),
            hmac_provider=StaticHmacProvider(),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
            definition_hash=lambda task: "client-nondurable-file-hash",
        )

        with self.assertRaises(TaskValidationError) as error:
            await client.enqueue(
                _definition(run=TaskRunPolicy.queued("documents")),
                input_value="private prompt",
                files=(TaskInputFile(logical_path="private-report.pdf"),),
            )

        self.assertEqual(queue.requests, [])
        self.assertEqual(len(error.exception.issues), 1)
        issue = error.exception.issues[0]
        self.assertEqual(issue.code, "input.invalid_file")
        self.assertEqual(issue.path, "files[0].artifact_ref")
        self.assertEqual(issue.category, TaskValidationCategory.UNSUPPORTED)
        self.assertNotIn("private-report", str(error.exception))

    async def test_enqueue_accepts_durable_provider_reference(self) -> None:
        store = InMemoryTaskStore()
        queue = RecordingQueue(store)
        client = TaskClient(
            store,
            target=_noop_target,
            queue=cast(TaskQueue, queue),
            hmac_provider=StaticHmacProvider(),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
            definition_hash=lambda task: "client-provider-file-hash",
        )
        provider_reference = TaskFileDescriptor.provider_reference_descriptor(
            "file-private",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
            owner_scope="tenant-a",
            mime_type="application/pdf",
            identity_hmac="hmac-value",
        ).provider_reference
        assert provider_reference is not None

        submission = await client.enqueue(
            _definition(run=TaskRunPolicy.queued("documents")),
            input_value="private prompt",
            files=(
                TaskInputFile(
                    logical_path="provider:openai:provider_file_id",
                    provider_reference=provider_reference,
                    media_type="application/pdf",
                ),
            ),
        )

        self.assertEqual(submission.run.state, TaskRunState.QUEUED)
        self.assertEqual(queue.artifacts, ())
        self.assertIn("<hmac-sha256>", str(submission.run.request))
        self.assertNotIn("file-private", str(submission.run.request))

    async def test_enqueue_rejects_expiring_provider_reference(self) -> None:
        store = InMemoryTaskStore()
        queue = RecordingQueue(store)
        client = TaskClient(
            store,
            target=_noop_target,
            queue=cast(TaskQueue, queue),
            hmac_provider=StaticHmacProvider(),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
            definition_hash=lambda task: "client-expiring-provider-file-hash",
        )
        provider_reference = TaskFileDescriptor.provider_reference_descriptor(
            "https://example.test/private",
            kind=TaskProviderReferenceKind.EXPIRING_PROVIDER_HANDLE,
            provider="openai",
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
            durable=False,
        ).provider_reference
        assert provider_reference is not None

        with self.assertRaises(TaskValidationError) as error:
            await client.enqueue(
                _definition(run=TaskRunPolicy.queued("documents")),
                input_value="private prompt",
                files=(
                    TaskInputFile(
                        logical_path="provider:openai:handle",
                        provider_reference=provider_reference,
                    ),
                ),
            )

        self.assertEqual(queue.requests, [])
        issue = error.exception.issues[0]
        self.assertEqual(issue.code, "input.invalid_file")
        self.assertEqual(issue.path, "files[0].provider_reference")
        self.assertNotIn("example.test", str(error.exception))

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
                encryption_provider=StaticEncryptionProvider(),
                raw_storage_allowed=True,
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

    async def test_enqueue_deletes_materialized_file_when_queue_fails(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            Path(root, "private.txt").write_text(
                "private file body",
                encoding="utf-8",
            )
            store = InMemoryTaskStore()
            queue = FailingQueue(store)
            artifact_store = RecordingArtifactStore()
            client = TaskClient(
                store,
                target=_noop_target,
                queue=cast(TaskQueue, queue),
                hmac_provider=StaticHmacProvider(),
                encryption_provider=StaticEncryptionProvider(),
                raw_storage_allowed=True,
                artifact_store=artifact_store,
                definition_hash=lambda task: "client-queue-failure-hash",
                execution_roots=(root,),
            )

            with self.assertRaises(RuntimeError) as error:
                await client.enqueue(
                    _definition(
                        input_contract=TaskInputContract.file(),
                        run=TaskRunPolicy.queued("documents"),
                    ),
                    input_value=TaskFileDescriptor.local_path(
                        "private.txt",
                        mime_type="text/plain",
                    ),
                )

        self.assertEqual(len(artifact_store.puts), 1)
        self.assertEqual(len(artifact_store.deleted), 1)
        self.assertEqual(
            artifact_store.deleted[0].artifact_id,
            "artifact-1",
        )
        self.assertNotIn("private.txt", str(error.exception))

    async def test_cancel_requests_queued_run_cancellation(self) -> None:
        store = InMemoryTaskStore()
        queue = RecordingQueue(store)
        client = TaskClient(
            store,
            target=_noop_target,
            queue=cast(TaskQueue, queue),
            hmac_provider=StaticHmacProvider(),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
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

    async def test_events_support_incremental_sequence_fetch(self) -> None:
        store = InMemoryTaskStore()
        definition = _definition()
        await store.register_definition(
            definition,
            definition_hash="hash-events",
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id="hash-events")
        )
        client = TaskClient(
            store,
            target=RejectingTarget(),
            hmac_provider=StaticHmacProvider(),
        )

        await store.append_event(
            run.run_id,
            event_type="start",
            category=TaskEventCategory.ENGINE,
            payload={"status": "first"},
        )
        await store.append_event(
            run.run_id,
            event_type="end",
            category=TaskEventCategory.ENGINE,
            payload={"status": "second"},
        )

        events = await client.events(run.run_id, after_sequence=1)
        inspection = await client.inspect(run.run_id, after_sequence=1)

        self.assertEqual([event.event_type for event in events], ["end"])
        self.assertEqual(inspection.events, events)
        with self.assertRaises(AssertionError):
            await client.events(run.run_id, after_sequence=-1)

    async def test_inspection_snapshot_includes_safe_optional_fields(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        definition = _definition()
        await store.register_definition(
            definition,
            definition_hash="hash-optional",
        )
        run = await store.create_run(
            TaskExecutionRequest(
                definition_id="hash-optional",
                input_summary={"privacy": "<redacted>"},
                file_summaries=({"artifact_id": "input-1"},),
                queue="documents",
            )
        )
        await store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        queued = await store.transition_run(
            run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        claimed = await store.assign_claim(
            queued.run_id,
            from_states={TaskRunState.QUEUED},
            worker_id="worker-1",
            lease_expires_at=queued.updated_at + timedelta(minutes=5),
            reason="claimed",
        )
        assert claimed.claim is not None
        attempt = await store.create_attempt(
            claimed.run_id,
            claim_token=claimed.claim.claim_token,
        )
        await store.transition_attempt(
            attempt.attempt_id,
            from_states={TaskAttemptState.CREATED},
            to_state=TaskAttemptState.RUNNING,
            reason="started",
            claim_token=claimed.claim.claim_token,
        )
        await store.append_usage(
            claimed.run_id,
            attempt_id=attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=1),
            metadata={"provider": "safe"},
        )
        await store.transition_attempt(
            attempt.attempt_id,
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.FAILED,
            reason="failed",
            result=TaskExecutionResult(error={"code": "attempt.failed"}),
            claim_token=claimed.claim.claim_token,
        )
        client = TaskClient(
            store,
            target=RejectingTarget(),
            hmac_provider=StaticHmacProvider(),
        )
        claimed_snapshot = str(
            (await client.inspect(claimed.run_id)).as_dict()
        )
        await store.transition_run(
            claimed.run_id,
            from_states={TaskRunState.CLAIMED},
            to_state=TaskRunState.FAILED,
            reason="failed",
            result=TaskExecutionResult(error={"code": "run.failed"}),
            claim_token=claimed.claim.claim_token,
        )

        output = await client.output(claimed.run_id)
        inspection = cast(
            Mapping[str, object],
            (await client.inspect(claimed.run_id)).as_dict(),
        )

        self.assertEqual(
            cast(Mapping[str, object], output.as_dict())["error"],
            {"code": "run.failed"},
        )
        rendered = str(inspection)
        self.assertIn("documents", rendered)
        self.assertIn("worker-1", claimed_snapshot)
        self.assertIn("input-1", rendered)
        self.assertIn("attempt.failed", rendered)
        self.assertIn("provider", rendered)
        self.assertNotIn(claimed.claim.claim_token, claimed_snapshot)
        self.assertNotIn(claimed.claim.claim_token, rendered)

    async def test_inspection_snapshot_omits_absent_optional_fields(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        definition = _definition()
        await store.register_definition(
            definition,
            definition_hash="hash-minimal",
        )
        minimal_run = await store.create_run(
            TaskExecutionRequest(definition_id="hash-minimal")
        )
        await store.append_event(
            minimal_run.run_id,
            event_type="start",
            category=TaskEventCategory.ENGINE,
            payload=None,
        )
        await store.append_usage(
            minimal_run.run_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=0),
        )
        attempt_run = await store.create_run(
            TaskExecutionRequest(definition_id="hash-minimal")
        )
        await store.create_attempt(attempt_run.run_id)
        client = TaskClient(
            store,
            target=RejectingTarget(),
            hmac_provider=StaticHmacProvider(),
        )

        minimal_snapshot = cast(
            Mapping[str, object],
            (await client.inspect(minimal_run.run_id)).as_dict(),
        )
        attempt_snapshot = cast(
            Mapping[str, object],
            (await client.inspect(attempt_run.run_id)).as_dict(),
        )

        run_snapshot = cast(
            Mapping[str, object],
            minimal_snapshot["run"],
        )
        self.assertNotIn("input_summary", run_snapshot)
        self.assertNotIn("file_summaries", run_snapshot)
        self.assertNotIn("queue", run_snapshot)
        self.assertNotIn("last_attempt_id", run_snapshot)
        self.assertNotIn("claim", run_snapshot)
        self.assertNotIn("result", run_snapshot)
        event_snapshots = cast(
            tuple[Mapping[str, object], ...],
            minimal_snapshot["events"],
        )
        self.assertNotIn("attempt_id", event_snapshots[0])
        self.assertNotIn("payload", event_snapshots[0])
        usage_snapshots = cast(
            tuple[Mapping[str, object], ...],
            minimal_snapshot["usage"],
        )
        self.assertNotIn("attempt_id", usage_snapshots[0])
        self.assertNotIn("metadata", usage_snapshots[0])
        attempt_snapshots = cast(
            tuple[Mapping[str, object], ...],
            attempt_snapshot["attempts"],
        )
        self.assertNotIn("result", attempt_snapshots[0])


def _definition(
    *,
    input_contract: TaskInputContract | None = None,
    privacy: TaskPrivacyPolicy | None = None,
    run: TaskRunPolicy | None = None,
) -> TaskDefinition:
    selected_run = run or TaskRunPolicy.direct()
    return TaskDefinition(
        task=TaskMetadata(name="agent", version="1"),
        input=input_contract or TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/valid.toml"),
        privacy=privacy
        or (
            TaskPrivacyPolicy(raw_retention_days=1)
            if selected_run.mode == RunMode.QUEUE
            else TaskPrivacyPolicy()
        ),
        run=selected_run,
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
