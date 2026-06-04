from base64 import b64decode
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, BinaryIO, cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import Message, MessageContentFile
from avalan.event import Event, EventType
from avalan.task import (
    HASHED_MARKER,
    RetryBackoff,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
    TaskArtifactState,
    TaskAttemptState,
    TaskClient,
    TaskDefinition,
    TaskDefinitionLoader,
    TaskExecutionTarget,
    TaskFileConversionRequest,
    TaskFileConversionResult,
    TaskFileConverterCapability,
    TaskFileDescriptor,
    TaskInputContract,
    TaskInputFile,
    TaskInputType,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskOutputType,
    TaskRetryPolicy,
    TaskRunPolicy,
    TaskRunState,
    TaskTargetContext,
    TaskTargetRunner,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
    UsageSource,
    spec_hash,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.stores import InMemoryTaskStore
from avalan.task.targets import AgentTaskTargetRunner


def _write_text_task_workspace(root: Path) -> Path:
    (root / "agents").mkdir()
    (root / "agents" / "summarizer.toml").write_text(
        """
[agent]
name = "Summarizer"
task = "Summarize the provided text."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
        encoding="utf-8",
    )
    task_path = root / "summarize.task.toml"
    task_path.write_text(
        """
[task]
name = "text_summary"
version = "1"

[input]
type = "string"

[output]
type = "text"

[execution]
type = "agent"
ref = "agents/summarizer.toml"

[run]
mode = "direct"
timeout_seconds = 60

[observability]
metrics = false
trace = false
capture_events = false
""",
        encoding="utf-8",
    )
    return task_path


def _write_task_workspace(root: Path) -> Path:
    (root / "agents").mkdir()
    (root / "agents" / "reviewer.toml").write_text(
        """
[agent]
name = "Reviewer"
task = "Review the uploaded document."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
        encoding="utf-8",
    )
    (root / "source.txt").write_text(
        "private source body",
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
mime_types = ["text/plain"]

[output]
type = "file"

[execution]
type = "agent"
ref = "agents/reviewer.toml"

[run]
mode = "direct"
timeout_seconds = 60

[artifact]
retention_days = 9
max_bytes = 4096

[observability]
metrics = true
trace = false
capture_events = true
""",
        encoding="utf-8",
    )
    return task_path


def _write_structured_task_workspace(root: Path) -> Path:
    (root / "agents").mkdir()
    (root / "agents" / "structured.toml").write_text(
        """
[agent]
name = "Structured reviewer"
task = "Return a JSON review result."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
        encoding="utf-8",
    )
    task_path = root / "structured.task.toml"
    task_path.write_text(
        """
[task]
name = "structured_review"
version = "1"

[input]
type = "object"

[input.schema]
type = "object"
required = ["question", "limit"]
additionalProperties = false

[input.schema.properties.question]
type = "string"
minLength = 1

[input.schema.properties.limit]
type = "integer"
minimum = 1
maximum = 5

[output]
type = "json"

[output.schema]
type = "object"
required = ["status", "count", "summary"]
additionalProperties = false

[output.schema.properties.status]
type = "string"
enum = ["ready"]

[output.schema.properties.count]
type = "integer"
minimum = 1

[output.schema.properties.summary]
type = "string"
minLength = 1

[execution]
type = "agent"
ref = "agents/structured.toml"

[run]
mode = "direct"
timeout_seconds = 60

[observability]
sinks = ["noop"]
metrics = false
trace = false
capture_events = false
""",
        encoding="utf-8",
    )
    return task_path


def _write_provider_task_workspace(
    root: Path,
    *,
    provider_uri: str,
    mime_type: str = "application/pdf",
    conversions: tuple[str, ...] = (),
    max_bytes: int = 4096,
) -> Path:
    (root / "agents").mkdir()
    (root / "agents" / "provider.toml").write_text(
        f"""
[agent]
name = "Provider reviewer"
task = "Review the supplied file."

[engine]
uri = "{provider_uri}"
""",
        encoding="utf-8",
    )
    conversion_lines = "".join(
        f'file_conversions = ["{conversion}"]\n' for conversion in conversions
    )
    task_path = root / "provider.task.toml"
    task_path.write_text(
        f"""
[task]
name = "provider_file_review"
version = "1"

[input]
type = "file"
{conversion_lines}mime_types = ["{mime_type}"]

[output]
type = "text"

[execution]
type = "agent"
ref = "agents/provider.toml"

[run]
mode = "direct"
timeout_seconds = 60

[limits]
file_bytes = {max_bytes}

[observability]
metrics = true
trace = false
capture_events = true
""",
        encoding="utf-8",
    )
    return task_path


def _provider_definition(
    *,
    mime_type: str = "application/pdf",
    conversions: tuple[str, ...] = (),
    file_bytes: int | None = 4096,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="provider_file_review", version="1"),
        input=TaskInputContract.file(
            conversions=conversions,
            mime_types=(mime_type,),
        ),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/provider.toml"),
        run=TaskRunPolicy.direct(timeout_seconds=60),
        limits=TaskLimitsPolicy(file_bytes=file_bytes),
        observability=TaskObservabilityPolicy(
            metrics=True,
            trace=False,
            capture_events=True,
        ),
    )


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
            secret=b"direct-client-e2e-secret",
        )


class PrefixingTextConverter:
    @property
    def name(self) -> str:
        return "text"

    @property
    def version(self) -> str:
        return "direct-e2e"

    @property
    def capability(self) -> TaskFileConverterCapability:
        return TaskFileConverterCapability(
            source_mime_types=("text/*",),
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
        prefix = str((options or {}).get("prefix", ""))
        return TaskFileConversionResult(
            content=f"{prefix}{content.decode()}".encode(),
            media_type="text/plain",
            metadata={"prefix": prefix},
        )


class PrefixingPdfTextConverter(PrefixingTextConverter):
    @property
    def capability(self) -> TaskFileConverterCapability:
        return TaskFileConverterCapability(
            source_mime_types=("application/pdf",),
            output_mime_types=("text/plain",),
            supports_streaming=False,
            max_input_bytes=1024,
            max_output_bytes=1024,
        )


class TextSummaryTarget(TaskTargetRunner):
    def __init__(self) -> None:
        self.definition_refs: list[str] = []
        self.input_values: list[object] = []
        self.metadata_values: list[Mapping[str, object]] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = context
        self.definition_refs.append(definition.execution.ref)
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        self.input_values.append(context.input_value)
        self.metadata_values.append(context.metadata)
        await context.check_cancelled()
        return "public summary"


class ReviewingTarget(TaskTargetRunner):
    def __init__(self) -> None:
        self.definition_refs: list[str] = []
        self.input_values: list[object] = []
        self.file_bodies: list[bytes] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = context
        self.definition_refs.append(definition.execution.ref)
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        self.input_values.append(context.input_value)
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
                        "status": "ready",
                        "token": "private-token-text",
                        "token_id": 41,
                    },
                )
            )
            if event_result is not None:
                await event_result
        await context.observe_usage(
            SimpleNamespace(
                input_token_count=13,
                output_token_count=8,
                total_token_count=21,
            )
        )
        return await context.artifact_store.put(
            b"private generated summary",
            media_type="text/plain",
            metadata={"filename": "summary.txt"},
        )


class StructuredTarget(TaskTargetRunner):
    def __init__(self, output: Mapping[str, object] | None = None) -> None:
        self.output = output or {
            "status": "ready",
            "count": 2,
            "summary": "private structured answer",
        }
        self.definition_refs: list[str] = []
        self.input_values: list[object] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = context
        self.definition_refs.append(definition.execution.ref)
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        self.input_values.append(context.input_value)
        await context.check_cancelled()
        return self.output


class FlakyTextSummaryTarget(TaskTargetRunner):
    def __init__(self) -> None:
        self.definition_refs: list[str] = []
        self.input_values: list[object] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = context
        self.definition_refs.append(definition.execution.ref)
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        self.input_values.append(context.input_value)
        attempt_number = len(self.input_values)
        await context.check_cancelled()
        if context.event_listener is not None:
            event_result = context.event_listener(
                Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={
                        "status": "retrying",
                        "token": f"private-token-{attempt_number}",
                        "token_id": attempt_number,
                    },
                )
            )
            if event_result is not None:
                await event_result
        await context.observe_usage(
            SimpleNamespace(
                input_token_count=attempt_number,
                output_token_count=attempt_number + 2,
                total_token_count=(attempt_number * 2) + 2,
            )
        )
        if attempt_number == 1:
            raise OSError("private transient path /tmp/customer-secret.txt")
        return "public retry summary"


class DirectCancellingTarget(TaskTargetRunner):
    def __init__(
        self,
        cancel: Callable[[str], Awaitable[object]],
    ) -> None:
        self.cancel = cancel
        self.definition_refs: list[str] = []
        self.input_values: list[object] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = context
        self.definition_refs.append(definition.execution.ref)
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        self.input_values.append(context.input_value)
        await self.cancel(context.execution.run_id)
        await context.check_cancelled()
        return {"status": "ready", "count": 1, "summary": "unused"}


class ProviderFakeResponse:
    input_token_count = 5
    output_token_count = 3
    total_token_count = 8

    def __init__(self, text: str = "provider accepted") -> None:
        self.text = text

    async def to_str(self) -> str:
        return self.text


class ProviderFakeEventManager:
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

    async def emit_token(self) -> None:
        for listener in tuple(self.listeners):
            result = listener(
                Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={
                        "token": "private-provider-token",
                        "token_id": 72,
                        "file_id": "file-private",
                    },
                )
            )
            if result is not None:
                await result


class ProviderFakeOrchestrator:
    def __init__(self, loader: "ProviderFakeLoader") -> None:
        self._loader = loader
        self.event_manager = loader.event_manager

    async def __aenter__(self) -> "ProviderFakeOrchestrator":
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

    async def __call__(self, input: object) -> ProviderFakeResponse:
        self._loader.inputs.append(input)
        await self.event_manager.emit_token()
        return ProviderFakeResponse()


class ProviderFakeLoader:
    def __init__(self) -> None:
        self.event_manager = ProviderFakeEventManager()
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
    ) -> ProviderFakeOrchestrator:
        _ = agent_id, disable_memory, uri, tool_settings
        self.paths.append(path)
        return ProviderFakeOrchestrator(self)


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


class DirectClientE2ETest(IsolatedAsyncioTestCase):
    async def test_loaded_provider_references_run_directly_and_inspect_safely(
        self,
    ) -> None:
        cases = (
            (
                "openai",
                "ai://env:KEY@openai/gpt-4o-mini",
                TaskClient.provider_file_id(
                    "openai",
                    "file-private",
                    mime_type="application/pdf",
                    size_bytes=2048,
                    owner_scope="tenant-secret",
                    identity_hmac="hmac-value",
                ),
                "file_id",
                "file-private",
            ),
            (
                "anthropic",
                "ai://env:KEY@anthropic/claude-3-5-sonnet",
                TaskClient.hosted_url(
                    "anthropic",
                    "https://files.example.test/private.pdf",
                    mime_type="application/pdf",
                    size_bytes=2048,
                    owner_scope="tenant-secret",
                ),
                "file_url",
                "https://files.example.test/private.pdf",
            ),
            (
                "google",
                "ai://env:KEY@google/gemini-2.0-flash",
                TaskClient.object_store_uri(
                    "google",
                    "gs://bucket/private.pdf",
                    mime_type="application/pdf",
                    size_bytes=2048,
                    owner_scope="tenant-secret",
                ),
                "file_url",
                "gs://bucket/private.pdf",
            ),
            (
                "bedrock",
                "ai://env:KEY@bedrock/us.anthropic.claude",
                TaskClient.object_store_uri(
                    "bedrock",
                    "s3://bucket/private.txt",
                    mime_type="text/plain",
                    size_bytes=2048,
                    owner_scope="tenant-secret",
                ),
                "file_url",
                "s3://bucket/private.txt",
            ),
        )

        for name, uri, descriptor, reference_key, reference in cases:
            with self.subTest(provider=name):
                with TemporaryDirectory() as root_name:
                    root = Path(root_name)
                    definition = TaskDefinitionLoader().load(
                        _write_provider_task_workspace(
                            root,
                            provider_uri=uri,
                            mime_type=descriptor.mime_type or "text/plain",
                        )
                    )
                    store = InMemoryTaskStore(
                        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
                    )
                    loader = ProviderFakeLoader()
                    client = TaskClient(
                        store,
                        target=AgentTaskTargetRunner(
                            loader,
                            ref_base=root,
                        ),
                        hmac_provider=StaticHmacProvider(),
                        execution_roots=(root,),
                        definition_hash=lambda task: (
                            f"provider-{name}-direct-e2e"
                        ),
                        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
                    )

                    validation = await client.validate(
                        definition,
                        input_value=descriptor,
                    )
                    result = await client.run(
                        definition,
                        input_value=descriptor,
                        metadata={"tenant": "safe"},
                    )
                    inspection = await client.inspect(result.run.run_id)

                self.assertTrue(validation.valid)
                self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
                self.assertEqual(result.output, "provider accepted")
                self.assertEqual(loader.entered, 1)
                self.assertEqual(loader.exited, 1)
                self.assertEqual(len(loader.inputs), 1)
                block = _only_file_block(loader.inputs[0])
                self.assertEqual(block.file[reference_key], reference)
                self.assertEqual(block.file["mime_type"], descriptor.mime_type)
                self.assertEqual(len(inspection.events), 1)
                self.assertEqual(
                    inspection.events[0].event_type, "token_generated"
                )
                self.assertEqual(len(inspection.usage), 1)
                self.assertEqual(inspection.usage_totals.input_tokens, 5)
                self.assertEqual(inspection.usage_totals.output_tokens, 3)
                self.assertEqual(inspection.usage_totals.total_tokens, 8)
                self.assertEqual(inspection.artifacts, ())
                input_summary = cast(
                    Mapping[str, object],
                    inspection.run.request.input_summary,
                )
                self.assertEqual(input_summary["privacy"], HASHED_MARKER)
                inspection_value = str(inspection.as_dict())
                self.assertNotIn(reference, inspection_value)
                self.assertNotIn("tenant-secret", inspection_value)
                self.assertNotIn("private-provider-token", inspection_value)
                self.assertNotIn("token_id", inspection_value)
                self.assertNotIn("file-private", inspection_value)

    async def test_sdk_provider_inline_file_runs_directly(self) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            _write_provider_task_workspace(
                root,
                provider_uri="ai://env:KEY@openai/gpt-4o-mini",
            )
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=lambda: "inline-input",
            )
            ref = await artifact_store.put(
                b"private inline bytes",
                media_type="application/pdf",
                metadata={"filename": "secret.pdf"},
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            loader = ProviderFakeLoader()
            client = TaskClient(
                store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                execution_roots=(root,),
                definition_hash=lambda task: "provider-inline-direct-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            result = await client.run(
                _provider_definition(),
                input_value=TaskClient.provider_file_id(
                    "openai",
                    "file-private",
                    mime_type="application/pdf",
                    size_bytes=ref.size_bytes,
                ),
                files=(
                    TaskInputFile(
                        logical_path="artifact:inline-input",
                        artifact_ref=ref,
                        media_type="application/pdf",
                        size_bytes=ref.size_bytes,
                        metadata={"filename": "secret.pdf"},
                    ),
                ),
            )
            inspection = await client.inspect(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        content = _file_blocks(loader.inputs[0])
        self.assertEqual(len(content), 2)
        self.assertEqual(
            b64decode(content[0].file["file_data"]),
            b"private inline bytes",
        )
        self.assertEqual(content[0].file["mime_type"], "application/pdf")
        self.assertEqual(content[1].file["file_id"], "file-private")
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private inline bytes", inspection_value)
        self.assertNotIn("secret.pdf", inspection_value)
        self.assertNotIn("file-private", inspection_value)

    async def test_sdk_provider_conversion_uses_converted_artifact(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            (root / "source.pdf").write_bytes(b"private pdf body")
            _write_provider_task_workspace(
                root,
                provider_uri="ai://env:KEY@bedrock/us.anthropic.claude",
                mime_type="application/pdf",
                conversions=("text",),
            )
            artifact_ids = iter(("input-artifact", "converted-artifact"))
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=lambda: next(artifact_ids),
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            loader = ProviderFakeLoader()
            client = TaskClient(
                store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                file_converters={"text": PrefixingPdfTextConverter()},
                execution_roots=(root,),
                definition_hash=lambda task: "provider-converted-direct-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            result = await client.run(
                _provider_definition(
                    mime_type="application/pdf",
                    conversions=("text",),
                ),
                input_value=TaskClient.local_file(
                    "source.pdf",
                    mime_type="application/pdf",
                    conversions=(
                        TaskClient.file_conversion(
                            "text",
                            options={"prefix": "converted: "},
                        ),
                    ),
                    metadata={"filename": "source.pdf"},
                ),
            )
            inspection = await client.inspect(result.run.run_id)
            artifacts = await store.list_artifacts(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        block = _only_file_block(loader.inputs[0])
        self.assertEqual(
            b64decode(block.file["file_data"]),
            b"converted: private pdf body",
        )
        self.assertEqual(block.file["mime_type"], "text/plain")
        self.assertEqual(
            [artifact.purpose for artifact in artifacts],
            [TaskArtifactPurpose.INPUT, TaskArtifactPurpose.CONVERTED],
        )
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private pdf body", inspection_value)
        self.assertNotIn("source.pdf", inspection_value)

    async def test_provider_inline_limit_failure_is_sanitized(self) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            _write_provider_task_workspace(
                root,
                provider_uri="ai://env:KEY@openai/gpt-4o-mini",
                max_bytes=4,
            )
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=lambda: "large-inline-input",
            )
            ref = await artifact_store.put(
                b"private large inline bytes",
                media_type="application/pdf",
                metadata={"filename": "large-secret.pdf"},
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            loader = ProviderFakeLoader()
            client = TaskClient(
                store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                execution_roots=(root,),
                definition_hash=lambda task: "provider-inline-limit-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            result = await client.run(
                _provider_definition(file_bytes=4),
                input_value=TaskClient.provider_file_id(
                    "openai",
                    "file-private",
                    mime_type="application/pdf",
                ),
                files=(
                    TaskInputFile(
                        logical_path="artifact:large-inline-input",
                        artifact_ref=ref,
                        media_type="application/pdf",
                        size_bytes=ref.size_bytes,
                        metadata={"filename": "large-secret.pdf"},
                    ),
                ),
            )
            inspection = await client.inspect(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(loader.inputs, [])
        error_summary = cast(Mapping[str, object], result.run.result.error)
        self.assertEqual(error_summary["category"], "input_contract")
        self.assertEqual(error_summary["code"], "input_contract.failed")
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private large inline bytes", inspection_value)
        self.assertNotIn("large-secret.pdf", inspection_value)
        self.assertNotIn("file-private", inspection_value)

    async def test_provider_mismatch_failure_is_sanitized(self) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = TaskDefinitionLoader().load(
                _write_provider_task_workspace(
                    root,
                    provider_uri="ai://env:KEY@anthropic/claude-3-5-sonnet",
                )
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            loader = ProviderFakeLoader()
            client = TaskClient(
                store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "provider-mismatch-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            result = await client.run(
                definition,
                input_value=TaskClient.provider_file_id(
                    "openai",
                    "file-private",
                    mime_type="application/pdf",
                    owner_scope="tenant-secret",
                ),
            )
            inspection = await client.inspect(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(loader.inputs, [])
        error_summary = cast(Mapping[str, object], result.run.result.error)
        self.assertEqual(error_summary["category"], "input_contract")
        self.assertEqual(error_summary["code"], "input_contract.failed")
        self.assertNotIn("file-private", str(error_summary))
        self.assertNotIn("tenant-secret", str(inspection.as_dict()))

    async def test_provider_reference_conversion_rejects_before_execution(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = TaskDefinitionLoader().load(
                _write_provider_task_workspace(
                    root,
                    provider_uri="ai://env:KEY@openai/gpt-4o-mini",
                    conversions=("text",),
                )
            )
            loader = ProviderFakeLoader()
            client = TaskClient(
                InMemoryTaskStore(),
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "provider-invalid-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            with self.assertRaises(TaskValidationError) as error:
                await client.run(
                    definition,
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

        self.assertEqual(loader.paths, [])
        self.assertEqual(loader.inputs, [])
        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.conversions")
        self.assertNotIn("file-private", str(error.exception))

    async def test_loaded_text_task_runs_directly_and_inspects_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = TaskDefinitionLoader().load(
                _write_text_task_workspace(root)
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            target = TextSummaryTarget()
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "direct-text-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            validation = await client.validate(
                definition,
                input_value="private text prompt",
            )
            result = await client.run(
                definition,
                input_value="private text prompt",
                metadata={"tenant": "safe"},
            )
            output = await client.output(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)

        self.assertTrue(validation.valid)
        self.assertEqual(
            target.definition_refs, ["agents/summarizer.toml"] * 2
        )
        self.assertEqual(target.input_values, ["private text prompt"])
        self.assertEqual(target.metadata_values, [{"tenant": "safe"}])
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.attempt.state, TaskAttemptState.SUCCEEDED)
        self.assertEqual(result.output, "public summary")
        self.assertTrue(output.ready)
        self.assertEqual(output.output_summary, {"privacy": "<redacted>"})
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(inspection.events, ())
        self.assertEqual(inspection.usage, ())
        self.assertEqual(inspection.artifacts, ())
        input_summary = cast(
            Mapping[str, object],
            inspection.run.request.input_summary,
        )
        self.assertEqual(input_summary["privacy"], HASHED_MARKER)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private text prompt", inspection_value)
        self.assertNotIn("public summary", inspection_value)

    async def test_sdk_equivalent_text_task_runs_with_same_identity(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            toml_definition = TaskDefinitionLoader().load(
                _write_text_task_workspace(root)
            )
            sdk_definition = TaskDefinition(
                task=TaskMetadata(name="text_summary", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.text(),
                execution=TaskExecutionTarget.agent("agents/summarizer.toml"),
                run=TaskRunPolicy.direct(timeout_seconds=60),
                observability=TaskObservabilityPolicy(
                    metrics=False,
                    trace=False,
                    capture_events=False,
                ),
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            target = TextSummaryTarget()
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            toml_result = await client.run(
                toml_definition,
                input_value="private text prompt",
            )
            sdk_result = await client.run(
                sdk_definition,
                input_value="private text prompt",
            )
            toml_inspection = await client.inspect(toml_result.run.run_id)
            sdk_inspection = await client.inspect(sdk_result.run.run_id)

        self.assertEqual(spec_hash(toml_definition), spec_hash(sdk_definition))
        self.assertEqual(
            toml_result.run.definition_id,
            sdk_result.run.definition_id,
        )
        self.assertEqual(
            target.definition_refs, ["agents/summarizer.toml"] * 2
        )
        self.assertEqual(
            target.input_values,
            ["private text prompt", "private text prompt"],
        )
        self.assertEqual(toml_result.output, "public summary")
        self.assertEqual(sdk_result.output, "public summary")
        self.assertEqual(toml_inspection.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(sdk_inspection.run.state, TaskRunState.SUCCEEDED)
        self.assertNotIn("private text prompt", str(toml_inspection.as_dict()))
        self.assertNotIn("private text prompt", str(sdk_inspection.as_dict()))

    async def test_loaded_text_task_retries_and_inspects_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = replace(
                TaskDefinitionLoader().load(_write_text_task_workspace(root)),
                retry=TaskRetryPolicy(
                    max_attempts=2,
                    backoff=RetryBackoff.LINEAR,
                ),
                observability=TaskObservabilityPolicy(
                    metrics=True,
                    trace=False,
                    capture_events=True,
                ),
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            target = FlakyTextSummaryTarget()
            delays: list[float] = []

            async def sleep(delay: float) -> None:
                delays.append(delay)

            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "direct-text-retry-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
                sleep=sleep,
            )

            validation = await client.validate(
                definition,
                input_value="private retry prompt",
            )
            result = await client.run(
                definition,
                input_value="private retry prompt",
                metadata={"tenant": "safe"},
            )
            output = await client.output(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)
            events_after_first = await client.events(
                result.run.run_id,
                after_sequence=1,
            )

        self.assertTrue(validation.valid)
        self.assertEqual(
            target.definition_refs, ["agents/summarizer.toml"] * 2
        )
        self.assertEqual(
            target.input_values,
            ["private retry prompt", "private retry prompt"],
        )
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.attempt.state, TaskAttemptState.SUCCEEDED)
        self.assertEqual(result.attempt.attempt_number, 2)
        self.assertEqual(result.output, "public retry summary")
        self.assertTrue(output.ready)
        self.assertEqual(output.output_summary, {"privacy": "<redacted>"})
        self.assertEqual(delays, [1])
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
        attempt_details = cast(Mapping[str, object], first_error["attempt"])
        self.assertEqual(attempt_details["failed_attempt_count"], 1)
        self.assertEqual(attempt_details["max_attempts"], 2)
        self.assertEqual(len(inspection.events), 2)
        self.assertEqual(events_after_first, (inspection.events[1],))
        self.assertEqual(inspection.usage_totals.input_tokens, 3)
        self.assertEqual(inspection.usage_totals.output_tokens, 7)
        self.assertEqual(inspection.usage_totals.total_tokens, 10)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private retry prompt", inspection_value)
        self.assertNotIn("private transient path", inspection_value)
        self.assertNotIn("customer-secret", inspection_value)
        self.assertNotIn("private-token", inspection_value)
        self.assertNotIn("token_id", inspection_value)

    async def test_direct_retry_expiry_finalizes_safely(self) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            current_time = datetime(2026, 1, 1, tzinfo=UTC)
            expires_at = current_time + timedelta(seconds=1)
            definition = replace(
                TaskDefinitionLoader().load(_write_text_task_workspace(root)),
                retry=TaskRetryPolicy(
                    max_attempts=2,
                    backoff=RetryBackoff.LINEAR,
                ),
                observability=TaskObservabilityPolicy(
                    metrics=True,
                    trace=False,
                    capture_events=True,
                ),
            )
            store = InMemoryTaskStore(clock=lambda: current_time)
            target = FlakyTextSummaryTarget()
            delays: list[float] = []

            async def sleep(delay: float) -> None:
                nonlocal current_time
                delays.append(delay)
                current_time += timedelta(seconds=delay)

            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "direct-retry-expiry-e2e",
                clock=lambda: current_time,
                sleep=sleep,
            )

            result = await client.run(
                definition,
                input_value="private expiring prompt",
                metadata={"tenant": "safe"},
                expires_at=expires_at,
            )
            output = await client.output(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)

        self.assertEqual(delays, [1])
        self.assertEqual(
            target.definition_refs,
            ["agents/summarizer.toml"],
        )
        self.assertEqual(target.input_values, ["private expiring prompt"])
        self.assertEqual(result.run.state, TaskRunState.EXPIRED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.EXPIRED)
        error_summary = cast(Mapping[str, object], output.error)
        self.assertEqual(error_summary["category"], "timeout")
        self.assertEqual(error_summary["code"], "timeout.exceeded")
        details = cast(Mapping[str, object], error_summary["details"])
        self.assertEqual(details["scope"], "run")
        self.assertEqual(
            [attempt.state for attempt in inspection.attempts],
            [TaskAttemptState.FAILED],
        )
        self.assertEqual(len(inspection.events), 1)
        self.assertEqual(inspection.usage_totals.input_tokens, 1)
        self.assertEqual(inspection.usage_totals.output_tokens, 3)
        self.assertEqual(inspection.usage_totals.total_tokens, 4)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private expiring prompt", inspection_value)
        self.assertNotIn("private transient path", inspection_value)
        self.assertNotIn("customer-secret", inspection_value)
        self.assertNotIn("private-token", inspection_value)
        self.assertNotIn("token_id", inspection_value)

    async def test_loaded_structured_task_runs_directly_and_redacts_output(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = TaskDefinitionLoader().load(
                _write_structured_task_workspace(root)
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            target = StructuredTarget()
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "direct-structured-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )
            input_value = {
                "question": "private customer escalation",
                "limit": 2,
            }

            validation = await client.validate(
                definition,
                input_value=input_value,
            )
            result = await client.run(
                definition,
                input_value=input_value,
                metadata={"tenant": "safe"},
            )
            output = await client.output(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)

        self.assertTrue(validation.valid)
        self.assertEqual(definition.input.type, TaskInputType.OBJECT)
        self.assertEqual(definition.output.type, TaskOutputType.JSON)
        self.assertEqual(
            target.definition_refs,
            ["agents/structured.toml"] * 2,
        )
        self.assertEqual(target.input_values, [input_value])
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, target.output)
        self.assertTrue(output.ready)
        self.assertEqual(
            output.output_summary,
            {"status": "ready", "count": 2},
        )
        input_summary = cast(
            Mapping[str, object],
            inspection.run.request.input_summary,
        )
        self.assertEqual(input_summary["privacy"], HASHED_MARKER)
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(
            inspection.attempts[0].state,
            TaskAttemptState.SUCCEEDED,
        )
        self.assertEqual(inspection.events, ())
        self.assertEqual(inspection.usage, ())
        self.assertIsNone(inspection.usage_totals.total_tokens)
        self.assertEqual(inspection.artifacts, ())
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private customer escalation", inspection_value)
        self.assertNotIn("private structured answer", inspection_value)

    async def test_direct_run_cancellation_finalizes_safely(self) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = TaskDefinitionLoader().load(
                _write_structured_task_workspace(root)
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            client_ref: list[TaskClient] = []

            async def cancel(run_id: str) -> object:
                return await client_ref[0].cancel(run_id)

            target = DirectCancellingTarget(cancel)
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "direct-cancel-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )
            client_ref.append(client)
            input_value = {
                "question": "private cancellation request",
                "limit": 1,
            }

            result = await client.run(definition, input_value=input_value)
            output = await client.output(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)

        self.assertEqual(target.definition_refs, ["agents/structured.toml"])
        self.assertEqual(target.input_values, [input_value])
        self.assertEqual(result.run.state, TaskRunState.CANCELLED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertFalse(output.ready)
        self.assertEqual(output.state, TaskRunState.CANCELLED)
        error_summary = cast(Mapping[str, object], output.error)
        self.assertEqual(error_summary["category"], "cancellation")
        self.assertEqual(error_summary["code"], "cancellation.requested")
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(
            inspection.attempts[0].state,
            TaskAttemptState.FAILED,
        )
        self.assertEqual(inspection.events, ())
        self.assertEqual(inspection.usage, ())
        self.assertEqual(inspection.artifacts, ())
        input_summary = cast(
            Mapping[str, object],
            inspection.run.request.input_summary,
        )
        self.assertEqual(input_summary["privacy"], HASHED_MARKER)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private cancellation request", inspection_value)
        self.assertNotIn("unused", inspection_value)

    async def test_loaded_structured_task_rejects_invalid_input_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = TaskDefinitionLoader().load(
                _write_structured_task_workspace(root)
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            target = StructuredTarget()
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "direct-structured-invalid",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )
            input_value = {
                "question": "private customer escalation",
                "limit": 9,
            }

            validation = await client.validate(
                definition,
                input_value=input_value,
            )
            with self.assertRaises(TaskValidationError) as error:
                await client.run(definition, input_value=input_value)

        self.assertFalse(validation.valid)
        self.assertEqual(len(validation.issues), 1)
        self.assertEqual(validation.issues[0].code, "input.invalid_type")
        self.assertEqual(validation.issues[0].path, "input")
        self.assertEqual(target.definition_refs, ["agents/structured.toml"])
        self.assertEqual(target.input_values, [])
        self.assertEqual(len(error.exception.issues), 1)
        self.assertNotIn("private customer escalation", str(error.exception))
        self.assertNotIn("9", str(error.exception))

    async def test_loaded_structured_task_rejects_invalid_output_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = TaskDefinitionLoader().load(
                _write_structured_task_workspace(root)
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            target = StructuredTarget(
                {
                    "status": "ready",
                    "summary": "private structured answer",
                }
            )
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "direct-structured-bad-output",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )
            input_value = {
                "question": "private customer escalation",
                "limit": 2,
            }

            result = await client.run(
                definition,
                input_value=input_value,
                metadata={"tenant": "safe"},
            )
            output = await client.output(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)

        self.assertEqual(target.input_values, [input_value])
        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertEqual(output.state, TaskRunState.FAILED)
        self.assertFalse(output.ready)
        error_summary = cast(Mapping[str, object], output.error)
        self.assertEqual(error_summary["category"], "output_contract")
        self.assertEqual(error_summary["code"], "output_contract.failed")
        details = cast(Mapping[str, object], error_summary["details"])
        issues = cast(tuple[Mapping[str, object], ...], details["issues"])
        self.assertEqual(issues[0]["code"], "output.invalid_type")
        self.assertEqual(issues[0]["path"], "output")
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(inspection.events, ())
        self.assertEqual(inspection.usage, ())
        self.assertEqual(inspection.artifacts, ())
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private customer escalation", inspection_value)
        self.assertNotIn("private structured answer", inspection_value)

    async def test_loaded_file_task_runs_directly_and_inspects_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            artifact_ids = iter(
                (
                    "input-artifact",
                    "converted-artifact",
                    "output-artifact",
                )
            )
            definition = TaskDefinitionLoader().load(
                _write_task_workspace(root)
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=lambda: next(artifact_ids),
            )
            target = ReviewingTarget()
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                execution_roots=(root,),
                definition_hash=lambda task: "direct-client-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )
            input_value = TaskFileDescriptor.local_path(
                "source.txt",
                mime_type="text/plain",
                conversions=(TaskFileConversionRequest(name="text"),),
                metadata={"filename": "source.txt"},
            )
            validation = await client.validate(
                definition,
                input_value=input_value,
            )

            result = await client.run(
                definition,
                input_value=input_value,
                metadata={"tenant": "safe"},
            )
            output = await client.output(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)
            artifacts = await store.list_artifacts(result.run.run_id)
            output_records = await store.list_artifacts(
                result.run.run_id,
                purpose=TaskArtifactPurpose.OUTPUT,
            )
            reader = await artifact_store.open(output_records[0].ref)
            try:
                output_body = reader.read()
            finally:
                reader.close()

        self.assertTrue(validation.valid)
        self.assertEqual(definition.input.type, TaskInputType.FILE)
        self.assertEqual(definition.output.type, TaskOutputType.FILE)
        self.assertEqual(target.definition_refs, ["agents/reviewer.toml"] * 2)
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(output.state, TaskRunState.SUCCEEDED)
        self.assertTrue(output.ready)
        self.assertIs(output.error, None)
        self.assertEqual(target.file_bodies, [b"private source body"])
        self.assertEqual(
            [artifact.purpose for artifact in artifacts],
            [
                TaskArtifactPurpose.INPUT,
                TaskArtifactPurpose.CONVERTED,
                TaskArtifactPurpose.OUTPUT,
            ],
        )
        self.assertEqual(
            [artifact.state for artifact in artifacts],
            [
                TaskArtifactState.READY,
                TaskArtifactState.READY,
                TaskArtifactState.READY,
            ],
        )
        self.assertEqual(output_records[0].retention.delete_after_days, 9)
        self.assertEqual(output_body, b"private generated summary")
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(len(inspection.events), 1)
        self.assertEqual(inspection.events[0].event_type, "token_generated")
        self.assertEqual(len(inspection.usage), 1)
        self.assertEqual(inspection.usage[0].source, UsageSource.ESTIMATED)
        self.assertEqual(inspection.usage_totals.input_tokens, 13)
        self.assertEqual(inspection.usage_totals.output_tokens, 8)
        self.assertEqual(inspection.usage_totals.total_tokens, 21)
        self.assertEqual(len(inspection.artifacts), 3)
        output_summary = cast(Mapping[str, object], output.output_summary)
        self.assertEqual(output_summary, {"state": "ready"})
        input_summary = cast(
            Mapping[str, object],
            inspection.run.request.input_summary,
        )
        self.assertEqual(input_summary["privacy"], HASHED_MARKER)
        inspection_value = str(inspection.as_dict())
        persisted_artifacts = str(artifacts)
        self.assertNotIn("private source body", inspection_value)
        self.assertNotIn("private generated summary", inspection_value)
        self.assertNotIn("source.txt", inspection_value)
        self.assertNotIn("summary.txt", persisted_artifacts)
        self.assertNotIn("private-token-text", inspection_value)
        self.assertNotIn("token_id", inspection_value)

    async def test_loaded_file_task_accessors_filter_safe_records(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            artifact_ids = iter(
                (
                    "input-artifact",
                    "converted-artifact",
                    "output-artifact",
                )
            )
            definition = TaskDefinitionLoader().load(
                _write_task_workspace(root)
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=lambda: next(artifact_ids),
            )
            target = ReviewingTarget()
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                execution_roots=(root,),
                definition_hash=lambda task: "direct-client-accessors-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            result = await client.run(
                definition,
                input_value=TaskFileDescriptor.local_path(
                    "source.txt",
                    mime_type="text/plain",
                    conversions=(TaskFileConversionRequest(name="text"),),
                    metadata={"filename": "source.txt"},
                ),
            )
            events = await client.events(
                result.run.run_id,
                attempt_id=result.attempt.attempt_id,
            )
            events_after_first = await client.events(
                result.run.run_id,
                after_sequence=events[0].sequence,
            )
            usage = await client.usage(
                result.run.run_id,
                attempt_id=result.attempt.attempt_id,
            )
            usage_totals = await client.usage_totals(result.run.run_id)
            artifacts = await client.artifacts(result.run.run_id)
            inspection = await client.inspect(
                result.run.run_id,
                after_sequence=events[0].sequence,
            )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attempt_id, result.attempt.attempt_id)
        self.assertEqual(events[0].event_type, "token_generated")
        self.assertEqual(events_after_first, ())
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0].attempt_id, result.attempt.attempt_id)
        self.assertEqual(usage[0].totals.input_tokens, 13)
        self.assertEqual(usage[0].totals.output_tokens, 8)
        self.assertEqual(usage[0].totals.total_tokens, 21)
        self.assertEqual(usage_totals.input_tokens, 13)
        self.assertEqual(usage_totals.output_tokens, 8)
        self.assertEqual(usage_totals.total_tokens, 21)
        self.assertEqual(len(artifacts), 3)
        output_artifact = cast(Mapping[str, object], artifacts[-1])
        self.assertEqual(
            output_artifact["purpose"],
            TaskArtifactPurpose.OUTPUT.value,
        )
        self.assertEqual(
            output_artifact["state"],
            TaskArtifactState.READY.value,
        )
        self.assertEqual(
            output_artifact["attempt_id"],
            result.attempt.attempt_id,
        )
        self.assertEqual(inspection.events, ())
        self.assertEqual(len(inspection.artifacts), 3)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private source body", inspection_value)
        self.assertNotIn("private generated summary", inspection_value)
        self.assertNotIn("private-token-text", inspection_value)
        self.assertNotIn("source.txt", inspection_value)
        self.assertNotIn("token_id", inspection_value)

    async def test_structured_file_input_materializes_for_direct_target(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            (root / "agents").mkdir()
            (root / "documents").mkdir()
            (root / "documents" / "brief.txt").write_bytes(
                b"private structured body"
            )
            artifact_ids = iter(("input-artifact", "output-artifact"))
            definition = TaskDefinition(
                task=TaskMetadata(name="structured_file_review", version="1"),
                input=TaskInputContract.object(
                    schema={
                        "type": "object",
                        "required": ["question", "document"],
                        "additionalProperties": False,
                        "properties": {
                            "question": {"type": "string"},
                            "document": {"type": "object"},
                        },
                    }
                ),
                output=TaskOutputContract.file(),
                execution=TaskExecutionTarget.agent("agents/reviewer.toml"),
                run=TaskRunPolicy.direct(timeout_seconds=60),
                artifact=TaskArtifactPolicy.references_only(retention_days=7),
                observability=TaskObservabilityPolicy(
                    metrics=True,
                    trace=False,
                    capture_events=True,
                ),
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=lambda: next(artifact_ids),
            )
            target = ReviewingTarget()
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                execution_roots=(root,),
                definition_hash=lambda task: "direct-structured-file-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )
            input_value = {
                "question": "Summarize the document.",
                "document": {
                    "source_kind": "local_path",
                    "reference": "documents/brief.txt",
                    "mime_type": "text/plain",
                },
            }

            result = await client.run(definition, input_value=input_value)
            inspection = await client.inspect(result.run.run_id)
            artifacts = await store.list_artifacts(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(target.file_bodies, [b"private structured body"])
        self.assertEqual(target.input_values, [input_value])
        self.assertEqual(
            [artifact.purpose for artifact in artifacts],
            [TaskArtifactPurpose.INPUT, TaskArtifactPurpose.OUTPUT],
        )
        self.assertEqual(artifacts[0].retention.delete_after_days, 7)
        self.assertEqual(len(inspection.artifacts), 2)
        input_summary = cast(
            Mapping[str, object],
            inspection.run.request.input_summary,
        )
        self.assertEqual(input_summary["privacy"], HASHED_MARKER)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private structured body", inspection_value)
        self.assertNotIn("documents/brief.txt", inspection_value)
        self.assertNotIn("Summarize the document.", inspection_value)

    async def test_direct_run_reads_explicit_and_converted_inputs(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            artifact_ids = iter(
                (
                    "explicit-artifact",
                    "input-artifact",
                    "converted-artifact",
                    "output-artifact",
                )
            )
            definition = TaskDefinitionLoader().load(
                _write_task_workspace(root)
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=lambda: next(artifact_ids),
            )
            explicit_ref = await artifact_store.put(
                b"private explicit body",
                media_type="text/plain",
                metadata={"filename": "explicit.txt"},
            )
            target = ReviewingTarget()
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                file_converters={"text": PrefixingTextConverter()},
                execution_roots=(root,),
                definition_hash=lambda task: "direct-client-e2e-converted",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )
            input_value = TaskFileDescriptor.local_path(
                "source.txt",
                mime_type="text/plain",
                conversions=(
                    TaskFileConversionRequest(
                        name="text",
                        options={"prefix": "converted: "},
                    ),
                ),
                metadata={"filename": "source.txt"},
            )

            result = await client.run(
                definition,
                input_value=input_value,
                files=(
                    TaskInputFile(
                        logical_path="provided/explicit.txt",
                        artifact_ref=explicit_ref,
                        media_type="text/plain",
                        size_bytes=explicit_ref.size_bytes,
                        metadata={"filename": "explicit.txt"},
                    ),
                ),
                metadata={"tenant": "safe"},
            )
            inspection = await client.inspect(result.run.run_id)
            artifacts = await store.list_artifacts(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            target.file_bodies,
            [
                b"private explicit body",
                b"converted: private source body",
            ],
        )
        self.assertEqual(
            [artifact.purpose for artifact in artifacts],
            [
                TaskArtifactPurpose.INPUT,
                TaskArtifactPurpose.CONVERTED,
                TaskArtifactPurpose.OUTPUT,
            ],
        )
        self.assertEqual(
            [artifact.artifact_id for artifact in artifacts],
            [
                "input-artifact",
                "converted-artifact",
                "output-artifact",
            ],
        )
        self.assertEqual(artifacts[1].provenance.converter, "text")
        self.assertEqual(artifacts[1].retention.delete_after_days, 9)
        self.assertEqual(len(inspection.artifacts), 3)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private explicit body", inspection_value)
        self.assertNotIn("private source body", inspection_value)
        self.assertNotIn("explicit.txt", inspection_value)
        self.assertNotIn("source.txt", inspection_value)

    async def test_loaded_file_task_rejects_escaped_input_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = TaskDefinitionLoader().load(
                _write_task_workspace(root)
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            target = ReviewingTarget()
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                artifact_store=LocalArtifactStore(
                    root / "artifacts",
                    raw_storage_allowed=True,
                    id_factory=lambda: "unexpected-artifact",
                ),
                execution_roots=(root,),
                definition_hash=lambda task: "direct-client-e2e-escape",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            result = await client.run(
                definition,
                input_value=TaskFileDescriptor.local_path(
                    "../source.txt",
                    mime_type="text/plain",
                    conversions=(TaskFileConversionRequest(name="text"),),
                ),
            )
            inspection = await client.inspect(result.run.run_id)
            artifacts = await store.list_artifacts(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertEqual(target.definition_refs, ["agents/reviewer.toml"])
        self.assertEqual(target.input_values, [])
        self.assertEqual(target.file_bodies, [])
        self.assertEqual(artifacts, ())
        self.assertEqual(len(inspection.attempts), 1)
        self.assertEqual(
            inspection.attempts[0].state,
            TaskAttemptState.FAILED,
        )
        error_summary = cast(
            Mapping[str, object],
            result.run.result.error if result.run.result else {},
        )
        self.assertEqual(error_summary["category"], "input_contract")
        self.assertEqual(error_summary["code"], "input_contract.failed")
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("../source", inspection_value)
        self.assertNotIn("private source body", inspection_value)


if __name__ == "__main__":
    main()
