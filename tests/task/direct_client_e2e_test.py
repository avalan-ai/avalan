from base64 import b64decode
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from json import dumps
from json import load as load_json
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from tomllib import load as load_toml
from types import SimpleNamespace
from typing import Any, BinaryIO, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch
from uuid import uuid4

from avalan.agent.loader import OrchestratorLoader
from avalan.entities import (
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallResult,
    ToolDescriptor,
    ToolNameResolution,
    ToolNameResolutionStatus,
)
from avalan.event import Event, EventType
from avalan.flow import (
    Flow,
    FlowDefinition,
    FlowDefinitionLoader,
    FlowEntryBehavior,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeDefinition,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.task import (
    HASHED_MARKER,
    FanoutObservabilitySink,
    ObservabilitySink,
    ObservabilitySinkHealth,
    OpenTelemetryObservabilitySink,
    PrivacyAction,
    PrometheusObservabilitySink,
    RetryBackoff,
    SanitizedTaskUsageEvent,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
    TaskArtifactState,
    TaskAttemptState,
    TaskClient,
    TaskDefinition,
    TaskDefinitionLoader,
    TaskEventCategory,
    TaskExecutionTarget,
    TaskFileConversionPageCollection,
    TaskFileConversionPageResult,
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
    TaskObservedEvent,
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
    UsageProviderFamily,
    UsageSource,
    UsageTotals,
    canonical_schema_json,
    pdf_image_converter_capability,
    spec_hash,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.stores import InMemoryTaskStore
from avalan.task.targets import (
    AgentTaskTargetRunner,
    FlowTaskTargetRunner,
    task_flow_node_registry,
)


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


def _extraction_output() -> dict[str, object]:
    return {
        "line_items": [
            {
                "line_number": 1,
                "vendor_name": "Northwind Office Supplies",
                "vendor_address": "42 Market St, Denver, CO 80202",
                "customer_name": "Contoso Research Lab",
                "customer_address": (
                    "100 Example Ave, Suite 1, Denver, CO 80202"
                ),
                "invoice_number": "INV-1001",
                "invoice_date": "01/15/2026",
                "due_date": "02/14/2026",
                "purchase_order": "PO-555100",
                "description": "Document processing services",
                "quantity": "5",
                "unit_price": "25.00",
                "line_amount": "125.00",
                "tax_amount": "0.00",
                "total_amount": "125.00",
                "currency": "USD",
                "notes": "Synthetic invoice fixture",
            }
        ]
    }


def _extraction_usage_fixture() -> Mapping[str, object]:
    fixture = (
        Path(__file__).parents[1]
        / "fixtures"
        / "poc_extraction"
        / "azure_responses_usage.json"
    )
    with fixture.open("r", encoding="utf-8") as file:
        value = load_json(file)
    assert isinstance(value, Mapping)
    return value


def _extraction_usage_payload() -> Mapping[str, object]:
    usage = _extraction_usage_fixture()["usage"]
    assert isinstance(usage, Mapping)
    return usage


def _extraction_provider_family() -> str:
    provider_family = _extraction_usage_fixture()["provider_family"]
    assert isinstance(provider_family, str)
    return provider_family


def _assert_extraction_usage(
    case: IsolatedAsyncioTestCase,
    usage: tuple[object, ...],
    totals: UsageTotals,
) -> None:
    case.assertEqual(len(usage), 1)
    record = usage[0]
    source = getattr(record, "source")
    record_totals = getattr(record, "totals")
    metadata = getattr(record, "metadata")
    case.assertEqual(source, UsageSource.EXACT)
    case.assertEqual(record_totals.input_tokens, 19)
    case.assertEqual(record_totals.cached_input_tokens, 7)
    case.assertIsNone(record_totals.cache_creation_input_tokens)
    case.assertEqual(record_totals.output_tokens, 23)
    case.assertEqual(record_totals.reasoning_tokens, 5)
    case.assertEqual(record_totals.total_tokens, 42)
    case.assertEqual(
        metadata,
        {"provider_family": UsageProviderFamily.AZURE_OPENAI.value},
    )
    case.assertEqual(totals.input_tokens, 19)
    case.assertEqual(totals.cached_input_tokens, 7)
    case.assertIsNone(totals.cache_creation_input_tokens)
    case.assertEqual(totals.output_tokens, 23)
    case.assertEqual(totals.reasoning_tokens, 5)
    case.assertEqual(totals.total_tokens, 42)


def _private_usage_sentinels() -> tuple[str, ...]:
    return (
        "private-deployment-name",
        "private-response-id",
        "private-cache-key",
        "private-api-key",
        "data:application/pdf;base64,private",
        "data:image/png;base64,private",
    )


def _fixture_agent_instructions(fixture: Path) -> str:
    with (fixture / "agent.toml").open("rb") as file:
        data = load_toml(file)
    agent = cast(Mapping[str, object], data["agent"])
    instructions = agent["instructions"]
    assert isinstance(instructions, str)
    return instructions


def _pipeline_flow_definition() -> FlowDefinition:
    return FlowDefinition(
        name="pipeline-flow",
        version="1",
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.JSON),
        ),
        entry_behavior=FlowEntryBehavior(node="pipeline"),
        output_behavior=FlowOutputBehavior(
            outputs={"answer": "pipeline.result"}
        ),
        nodes=(
            FlowNodeDefinition(
                name="pipeline",
                type="tool",
                ref="shell.pipeline",
                config={
                    "arguments": {
                        "steps": [
                            {
                                "id": "read",
                                "command": "cat",
                                "paths": ["README.md"],
                            },
                            {
                                "id": "count",
                                "command": "wc",
                                "options": {"lines": True},
                                "stdin_from": {
                                    "step_id": "read",
                                    "stream": "stdout",
                                },
                            },
                        ]
                    }
                },
            ),
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


class PipelineToolResolver:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.contexts: list[ToolCallContext] = []

    def list_tools(self) -> list[ToolDescriptor]:
        return [
            ToolDescriptor(
                name="shell.pipeline",
                parameter_schema={
                    "type": "object",
                    "properties": {
                        "steps": {"type": "array"},
                    },
                },
                return_schema={
                    "type": "string",
                    "description": "Formatted shell composition result.",
                },
            )
        ]

    def resolve_tool_name(
        self,
        name: str,
        *,
        provider_originated: bool = False,
    ) -> ToolNameResolution:
        _ = provider_originated
        if name == "shell.pipeline":
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.EXACT,
                canonical_name=name,
                candidates=[name],
            )
        return ToolNameResolution(
            requested_name=name,
            status=ToolNameResolutionStatus.UNKNOWN,
            candidates=[],
        )

    def validate_tool_call(self, call: ToolCall) -> ToolCallDiagnostic | None:
        _ = call
        return None

    async def execute_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
    ) -> ToolCallResult:
        self.calls.append(call)
        self.contexts.append(context)
        return ToolCallResult(
            id=call.id,
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="tool: shell.pipeline\nstatus: completed\nstdout:\n2\n",
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


class ReturnedUsageTextSummaryTarget(TextSummaryTarget):
    async def run(self, context: TaskTargetContext) -> object:
        self.input_values.append(context.input_value)
        self.metadata_values.append(context.metadata)
        await context.check_cancelled()
        return UsageTextOutput(
            "public summary",
            usage={
                "input_tokens": 9,
                "cached_input_tokens": 4,
                "output_tokens": 6,
                "reasoning_tokens": 2,
                "total_tokens": 15,
                "provider_family": "openai",
                "raw_response_id": "private-response-id",
            },
        )


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


class TerminalUsageStream:
    def __init__(self) -> None:
        self.usage: object | None = None
        self._sequence = 0

    def __call__(self, **_: object) -> "TerminalUsageStream":
        self.usage = None
        self._sequence = 0
        return self

    def __aiter__(self) -> "TerminalUsageStream":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        match self._sequence:
            case 0:
                item = CanonicalStreamItem(
                    stream_session_id="task-streaming-usage",
                    run_id="task-run",
                    turn_id="task-turn",
                    sequence=self._sequence,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                )
            case 1:
                item = CanonicalStreamItem(
                    stream_session_id="task-streaming-usage",
                    run_id="task-run",
                    turn_id="task-turn",
                    sequence=self._sequence,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="public stream summary",
                )
            case 2:
                item = CanonicalStreamItem(
                    stream_session_id="task-streaming-usage",
                    run_id="task-run",
                    turn_id="task-turn",
                    sequence=self._sequence,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                )
            case 3:
                self.usage = {
                    "input_tokens": 8,
                    "cached_input_tokens": 3,
                    "cache_creation_input_tokens": 2,
                    "output_tokens": 5,
                    "reasoning_tokens": 1,
                    "total_tokens": 13,
                }
                item = CanonicalStreamItem(
                    stream_session_id="task-streaming-usage",
                    run_id="task-run",
                    turn_id="task-turn",
                    sequence=self._sequence,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage=cast(Any, self.usage),
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                )
            case _:
                raise StopAsyncIteration
        self._sequence += 1
        return item


class StreamingUsageTarget(TaskTargetRunner):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        await context.check_cancelled()
        settings = GenerationSettings()
        response = TextGenerationResponse(
            TerminalUsageStream(),
            logger=getLogger("avalan.tests.task.streaming_usage"),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        output = await response.to_str()
        await context.observe_usage(response)
        return output


class AzureUsageTarget(TaskTargetRunner):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        await context.check_cancelled()
        await context.observe_usage(
            SimpleNamespace(
                provider_family=UsageProviderFamily.AZURE_OPENAI.value,
                usage={
                    "input_tokens": 12,
                    "input_tokens_details": {"cached_tokens": 5},
                    "output_tokens": 8,
                    "output_tokens_details": {"reasoning_tokens": 3},
                    "total_tokens": 20,
                    "model": "private-deployment-name",
                    "response_id": "private-response-id",
                },
            )
        )
        return "public azure summary"


class ProviderFakeClient:
    def __init__(
        self,
        *,
        reference_key: str,
        reference: str,
        mime_type: str,
    ) -> None:
        self.reference_key = reference_key
        self.reference = reference
        self.mime_type = mime_type
        self.inputs: list[object] = []

    async def __call__(self, input: object) -> ProviderFakeResponse:
        self.inputs.append(input)
        block = _only_file_block(input)
        self._assert_generic_payload(block.file)
        self._assert_provider_reference(block.file)
        return ProviderFakeResponse()

    def _assert_generic_payload(self, payload: Mapping[str, object]) -> None:
        self._assert_omitted(payload, "input_file")
        self._assert_omitted(payload, "document")
        self._assert_omitted(payload, "inline_data")
        self._assert_omitted(payload, "s3Location")

    def _assert_provider_reference(
        self,
        payload: Mapping[str, object],
    ) -> None:
        assert payload[self.reference_key] == self.reference
        assert payload["mime_type"] == self.mime_type

    def _assert_omitted(
        self,
        payload: Mapping[str, object],
        key: str,
    ) -> None:
        assert key not in payload


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
        if self._loader.provider_client is not None:
            return await self._loader.provider_client(input)
        return ProviderFakeResponse()


class ProviderFakeLoader:
    def __init__(
        self,
        provider_client: ProviderFakeClient | None = None,
    ) -> None:
        self.provider_client = provider_client
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


def _image_blocks(input_value: object) -> tuple[MessageContentImage, ...]:
    assert isinstance(input_value, Message)
    content = input_value.content
    assert isinstance(content, list)
    return tuple(
        block
        for block in cast(list[Any], content)
        if isinstance(block, MessageContentImage)
    )


def _only_file_block(input_value: object) -> MessageContentFile:
    blocks = _file_blocks(input_value)
    assert len(blocks) == 1
    return blocks[0]


def _extraction_message_summary(
    input_value: object,
    expected_file_bytes: bytes,
) -> Mapping[str, object]:
    assert isinstance(input_value, Message)
    content = input_value.content
    assert isinstance(content, list)
    blocks = cast(list[Any], content)
    text_blocks = [
        block for block in blocks if isinstance(block, MessageContentText)
    ]
    file_blocks = [
        block for block in blocks if isinstance(block, MessageContentFile)
    ]
    assert len(text_blocks) == 1
    assert len(file_blocks) == 1
    file = file_blocks[0].file
    file_data = file["file_data"]
    assert isinstance(file_data, str)
    assert b64decode(file_data) == expected_file_bytes
    return {
        "text": text_blocks[0].text,
        "mime_type": file["mime_type"],
        "file_bytes": len(expected_file_bytes),
    }


class ExtractionFakeResponse:
    input_token_count = 19
    output_token_count = 23
    total_token_count = 42

    def __init__(self, output: Mapping[str, object]) -> None:
        self.output = output

    @property
    def provider_family(self) -> str:
        return _extraction_provider_family()

    @property
    def usage(self) -> Mapping[str, object]:
        return _extraction_usage_payload()

    async def to_json(self) -> str:
        return dumps(self.output, sort_keys=True, separators=(",", ":"))

    async def to_str(self) -> str:
        return await self.to_json()


class ExtractionRawJsonResponse:
    input_token_count = 19
    output_token_count = 23
    total_token_count = 42

    def __init__(self, raw_json: str) -> None:
        self.raw_json = raw_json

    async def to_json(self) -> str:
        return self.raw_json

    async def to_str(self) -> str:
        return self.raw_json


class ExtractionProviderFailureResponse:
    input_token_count = 19
    output_token_count = 23
    total_token_count = 42

    async def to_json(self) -> str:
        raise RuntimeError(
            "private provider body api-key=sk-test-secret file_data=abc123"
        )

    async def to_str(self) -> str:
        return "private provider fallback body"


class ExtractionPdfImageConverter:
    name = "pdf_image"
    version = "fake"

    def __init__(
        self,
        pages: tuple[TaskFileConversionPageResult, ...],
    ) -> None:
        base = pdf_image_converter_capability()
        self.calls: list[tuple[bytes, str | None, Mapping[str, object]]] = []
        self._pages = pages
        self._capability = TaskFileConverterCapability(
            source_mime_types=base.source_mime_types,
            output_mime_types=base.output_mime_types,
            supports_streaming=base.supports_streaming,
            max_input_bytes=base.max_input_bytes,
            max_output_bytes=base.max_output_bytes,
            max_pages=base.max_pages,
            min_dpi=base.min_dpi,
            max_dpi=base.max_dpi,
            min_quality=base.min_quality,
            max_quality=base.max_quality,
            max_pixels=base.max_pixels,
            estimated_memory_bytes=base.estimated_memory_bytes,
            timeout_seconds=base.timeout_seconds,
            options_schema=base.options_schema,
        )

    @property
    def capability(self) -> TaskFileConverterCapability:
        return self._capability

    def validate_options(self, options: Mapping[str, object]) -> None:
        assert options.get("format") == "png"

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        _ = content, source_media_type, options
        raise AssertionError("page converter must use convert_pages")

    async def convert_pages(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionPageCollection:
        safe_options = dict(options or {})
        self.calls.append((content, source_media_type, safe_options))
        pages = self._selected_pages(safe_options.get("pages"))
        return TaskFileConversionPageCollection(
            pages=pages,
            metadata={"backend": "fake"},
        )

    def _selected_pages(
        self,
        value: object,
    ) -> tuple[TaskFileConversionPageResult, ...]:
        if not isinstance(value, Mapping):
            return self._pages
        start = value.get("start", 1)
        end = value.get("end", len(self._pages))
        assert isinstance(start, int)
        assert isinstance(end, int)
        return tuple(
            page for page in self._pages if start <= page.page_index <= end
        )


class ExtractionFakeOrchestrator:
    def __init__(self, output: Mapping[str, object]) -> None:
        self.output = output
        self.event_manager = ProviderFakeEventManager()
        self.inputs: list[object] = []
        self.text_formats: list[Mapping[str, object]] = []
        self.reasoning_options: list[Mapping[str, object]] = []
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> "ExtractionFakeOrchestrator":
        self.entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        self.exited += 1
        return None

    async def __call__(self, input: object) -> ExtractionFakeResponse:
        self.inputs.append(input)
        await self.event_manager.emit_token()
        return ExtractionFakeResponse(self.output)


class ExtractionFailureOrchestrator(ExtractionFakeOrchestrator):
    def __init__(self, response: object) -> None:
        super().__init__({})
        self.response = response

    async def __call__(self, input: object) -> object:
        self.inputs.append(input)
        await self.event_manager.emit_token()
        return self.response


class ExtractionFailureLoader:
    def __init__(self, orchestrator: ExtractionFailureOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.paths: list[str] = []

    async def from_file(
        self,
        path: str,
        *,
        agent_id: object | None,
        disable_memory: bool = False,
        uri: str | None = None,
        tool_settings: object | None = None,
    ) -> ExtractionFailureOrchestrator:
        _ = agent_id, disable_memory, uri, tool_settings
        self.paths.append(path)
        return self.orchestrator


@dataclass(slots=True)
class ExtractionRecordingSink(ObservabilitySink):
    events: list[TaskObservedEvent] = field(default_factory=list)
    usages: list[tuple[str, str | None, UsageSource, UsageTotals]] = field(
        default_factory=list
    )

    async def record_event(self, event: TaskObservedEvent) -> None:
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
        _ = metadata
        self.usages.append((run_id, attempt_id, source, totals))

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(
            name="extraction-recording",
            event_count=len(self.events),
            usage_count=len(self.usages),
        )


@dataclass(slots=True)
class ExtractionCounterChild:
    counter: "ExtractionCounter"
    label_values: tuple[tuple[str, str], ...]

    def inc(self, amount: float = 1.0) -> None:
        self.counter.samples[self.label_values] = (
            self.counter.samples.get(self.label_values, 0.0) + amount
        )


@dataclass(slots=True)
class ExtractionCounter:
    name: str
    labelnames: tuple[str, ...]
    samples: dict[tuple[tuple[str, str], ...], float] = field(
        default_factory=dict
    )

    def labels(self, **labels: str) -> ExtractionCounterChild:
        assert tuple(labels) == self.labelnames
        return ExtractionCounterChild(
            counter=self,
            label_values=tuple(sorted(labels.items())),
        )


@dataclass(slots=True)
class ExtractionCounterFactory:
    counters: dict[str, ExtractionCounter] = field(default_factory=dict)

    def __call__(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: tuple[str, ...] = (),
        registry: object | None = None,
    ) -> ExtractionCounter:
        _ = registry
        assert documentation
        counter = ExtractionCounter(name=name, labelnames=labelnames)
        self.counters[name] = counter
        return counter


@dataclass(slots=True)
class ExtractionSpanContext:
    tracer: "ExtractionTracer"
    name: str
    attributes: Mapping[str, object] | None

    def __enter__(self) -> object:
        self.tracer.spans.append((self.name, dict(self.attributes or {})))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        _ = exc_type, exc, traceback
        return None


@dataclass(slots=True)
class ExtractionTracer:
    spans: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, object] | None = None,
    ) -> ExtractionSpanContext:
        return ExtractionSpanContext(
            tracer=self,
            name=name,
            attributes=attributes,
        )


@dataclass(slots=True)
class ExtractionMetricCounter:
    samples: list[tuple[int | float, dict[str, object]]] = field(
        default_factory=list
    )

    def add(
        self,
        amount: int | float,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        self.samples.append((amount, dict(attributes or {})))


@dataclass(slots=True)
class ExtractionMeter:
    counters: dict[str, ExtractionMetricCounter] = field(default_factory=dict)

    def create_counter(
        self,
        name: str,
        *,
        description: str = "",
        unit: str = "1",
    ) -> ExtractionMetricCounter:
        assert description
        assert unit
        counter = ExtractionMetricCounter()
        self.counters[name] = counter
        return counter


def _prometheus_samples(
    factory: ExtractionCounterFactory,
    metric_name: str,
) -> dict[tuple[tuple[str, str], ...], float]:
    return factory.counters[metric_name].samples


class DirectClientE2ETest(IsolatedAsyncioTestCase):
    async def test_direct_agent_task_runs_pipeline_enabled_agent(self) -> None:
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
            loader = ProviderFakeLoader()
            definition = TaskDefinition(
                task=TaskMetadata(name="pipeline_agent", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.text(),
                execution=TaskExecutionTarget.agent("agents/pipeline.toml"),
                run=TaskRunPolicy.direct(timeout_seconds=60),
                observability=TaskObservabilityPolicy.noop(),
            )
            client = TaskClient(
                InMemoryTaskStore(
                    clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
                ),
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: f"direct-{task.task.name}",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            validation = await client.validate(
                definition,
                input_value="private prompt",
            )
            result = await client.run(
                definition,
                input_value="private prompt",
            )
            inspection = await client.inspect(result.run.run_id)

        self.assertTrue(validation.valid)
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "provider accepted")
        self.assertEqual(loader.paths, [str(agent_path)])
        self.assertEqual(loader.inputs, ["private prompt"])
        self.assertNotIn("private prompt", str(inspection.as_dict()))

    async def test_direct_flow_task_runs_shell_pipeline_with_tool_settings(
        self,
    ) -> None:
        resolver = PipelineToolResolver()
        definition = TaskDefinition(
            task=TaskMetadata(name="pipeline_flow", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.flow("flows/pipeline.toml"),
            run=TaskRunPolicy.direct(timeout_seconds=60),
            observability=TaskObservabilityPolicy.noop(),
        )
        client = TaskClient(
            InMemoryTaskStore(clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)),
            target=FlowTaskTargetRunner(
                strict_resolver=lambda _: _pipeline_flow_definition(),
                tool_resolver=resolver,
            ),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: f"direct-{task.task.name}",
            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        )

        validation = await client.validate(
            definition,
            input_value="count README lines",
        )
        result = await client.run(
            definition,
            input_value="count README lines",
        )
        inspection = await client.inspect(result.run.run_id)

        self.assertTrue(validation.valid)
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            result.output,
            "tool: shell.pipeline\nstatus: completed\nstdout:\n2\n",
        )
        self.assertEqual(len(resolver.calls), 1)
        self.assertTrue(resolver.contexts[0].flow_tool_node)
        self.assertEqual(resolver.calls[0].name, "shell.pipeline")
        self.assertEqual(
            cast(Mapping[str, object], resolver.calls[0].arguments)["steps"][
                0
            ]["id"],
            "read",
        )
        self.assertNotIn("count README lines", str(inspection.as_dict()))

    async def test_poc_extraction_fixture_sends_prompt_pdf_and_schema(
        self,
    ) -> None:
        fixture = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
        )
        instructions = _fixture_agent_instructions(fixture)
        pdf_bytes = (fixture / "sample.pdf").read_bytes()
        definition = await TaskDefinitionLoader().load(fixture / "task.toml")
        output = _extraction_output()
        orchestrator = ExtractionFakeOrchestrator(output)
        settings_values: list[Any] = []
        recording_sink = ExtractionRecordingSink()
        prometheus_factory = ExtractionCounterFactory()
        tracer = ExtractionTracer()
        meter = ExtractionMeter()
        sink = FanoutObservabilitySink(
            sinks=(
                recording_sink,
                PrometheusObservabilitySink(
                    counter_factory=prometheus_factory,
                ),
                OpenTelemetryObservabilitySink(
                    tracer=tracer,
                    meter=meter,
                ),
            )
        )

        async def from_settings(
            loader: OrchestratorLoader,
            settings: object,
            *,
            tool_settings: object | None = None,
            tool_format: object | None = None,
            **kwargs: object,
        ) -> ExtractionFakeOrchestrator:
            _ = loader, tool_settings, tool_format, kwargs
            call_options = cast(Any, settings).call_options
            assert isinstance(call_options, Mapping)
            response_format = cast(
                Mapping[str, object],
                call_options["response_format"],
            )
            text_format = {
                "type": response_format["type"],
                "name": response_format["name"],
                "schema": response_format["schema"],
                "strict": response_format["strict"],
            }
            orchestrator.text_formats.append(text_format)
            orchestrator.reasoning_options.append(
                cast(Mapping[str, object], call_options["reasoning"])
            )
            settings_values.append(settings)
            return orchestrator

        with TemporaryDirectory() as artifact_root:
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=cast(Any, object()),
                logger=getLogger("avalan.tests.task.extraction"),
                participant_id=uuid4(),
                stack=stack,
            )
            try:
                with patch.object(
                    OrchestratorLoader,
                    "from_settings",
                    new=from_settings,
                ):
                    client = TaskClient(
                        InMemoryTaskStore(
                            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
                        ),
                        target=AgentTaskTargetRunner(loader, ref_base=fixture),
                        artifact_store=LocalArtifactStore(
                            Path(artifact_root),
                            raw_storage_allowed=True,
                        ),
                        hmac_provider=StaticHmacProvider(),
                        execution_roots=(fixture,),
                        definition_hash=lambda task: "extraction-fixture-e2e",
                        observability_sink=sink,
                        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
                    )
                    descriptor = TaskClient.local_file(
                        "./sample.pdf",
                        mime_type="application/pdf",
                        size_bytes=len(pdf_bytes),
                    )

                    validation = await client.validate(
                        definition,
                        input_value=descriptor,
                    )
                    result = await client.run(
                        definition,
                        input_value=descriptor,
                    )
                    inspection = await client.inspect(result.run.run_id)
            finally:
                await stack.aclose()

        self.assertTrue(validation.valid)
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, output)
        self.assertEqual(orchestrator.entered, 1)
        self.assertEqual(orchestrator.exited, 1)
        self.assertEqual(len(orchestrator.inputs), 1)
        self.assertEqual(len(settings_values), 1)
        settings = cast(Any, settings_values[-1])
        agent_config = settings.agent_config
        self.assertIsInstance(agent_config, Mapping)
        self.assertEqual(agent_config["instructions"], instructions)
        self.assertNotIn("system", agent_config)
        self.assertNotIn("task", agent_config)
        self.assertEqual(settings.tools, [])
        self.assertEqual(
            settings.engine_config["base_url"],
            "https://tenant.openai.azure.com/openai/v1/",
        )
        call_options = settings.call_options
        self.assertIsInstance(call_options, Mapping)
        self.assertNotIn("tools", call_options)
        self.assertNotIn("tool_choice", call_options)
        self.assertEqual(
            orchestrator.reasoning_options[-1],
            {"effort": "high"},
        )
        self.assertEqual(orchestrator.text_formats[-1]["type"], "json_schema")
        self.assertEqual(
            orchestrator.text_formats[-1]["name"],
            "invoice_extraction",
        )
        self.assertTrue(orchestrator.text_formats[-1]["strict"])
        self.assertEqual(
            canonical_schema_json(orchestrator.text_formats[-1]["schema"]),
            canonical_schema_json(definition.output.schema),
        )

        agent_input = orchestrator.inputs[0]
        self.assertIsInstance(agent_input, Message)
        content = cast(Message, agent_input).content
        self.assertIsInstance(content, list)
        blocks = cast(list[Any], content)
        text_blocks = [
            block for block in blocks if isinstance(block, MessageContentText)
        ]
        file_blocks = [
            block for block in blocks if isinstance(block, MessageContentFile)
        ]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(len(file_blocks), 1)
        self.assertEqual(
            text_blocks[0].text,
            "Analyze the attached synthetic invoice PDF and "
            "extract all data according to the extraction instructions.",
        )
        self.assertNotIn(instructions, text_blocks[0].text)
        self.assertEqual(file_blocks[0].file["mime_type"], "application/pdf")
        self.assertEqual(
            b64decode(cast(str, file_blocks[0].file["file_data"])),
            pdf_bytes,
        )
        self.assertEqual(len(inspection.events), 1)
        _assert_extraction_usage(
            self,
            cast(tuple[object, ...], inspection.usage),
            inspection.usage_totals,
        )
        self.assertEqual(
            [event.category for event in recording_sink.events],
            [TaskEventCategory.TOKEN, TaskEventCategory.USAGE],
        )
        usage_events = [
            event
            for event in recording_sink.events
            if isinstance(event, SanitizedTaskUsageEvent)
        ]
        self.assertEqual(len(usage_events), 1)
        usage_payload = usage_events[0].payload
        assert isinstance(usage_payload, Mapping)
        self.assertEqual(usage_payload["source"], "exact")
        self.assertEqual(usage_payload["provider_family"], "azure_openai")
        self.assertEqual(usage_payload["cached_input_tokens"], 7)
        self.assertEqual(
            _prometheus_samples(
                prometheus_factory,
                "avalan_task_observability_usage_tokens",
            ),
            {
                (
                    ("counter", "input_tokens"),
                    ("source", "exact"),
                ): 19.0,
                (
                    ("counter", "cached_input_tokens"),
                    ("source", "exact"),
                ): 7.0,
                (
                    ("counter", "output_tokens"),
                    ("source", "exact"),
                ): 23.0,
                (
                    ("counter", "reasoning_tokens"),
                    ("source", "exact"),
                ): 5.0,
                (
                    ("counter", "total_tokens"),
                    ("source", "exact"),
                ): 42.0,
            },
        )
        self.assertIn(
            (
                7,
                {
                    "task.usage.source": "exact",
                    "task.usage.counter": "cached_input_tokens",
                },
            ),
            meter.counters["avalan.task.observability.usage_tokens"].samples,
        )
        inspection_value = str(inspection.as_dict())
        observability_value = (
            f"{recording_sink.events}"
            f"{recording_sink.usages}"
            f"{prometheus_factory.counters}"
            f"{tracer.spans}"
            f"{meter.counters}"
        )
        self.assertNotIn("sample.pdf", inspection_value)
        self.assertNotIn("private-provider-token", inspection_value)
        self.assertNotIn("file_data", inspection_value)
        for sentinel in _private_usage_sentinels():
            self.assertNotIn(sentinel, inspection_value)
            self.assertNotIn(sentinel, observability_value)

    async def test_poc_extraction_flow_matches_direct_provider_payload(
        self,
    ) -> None:
        fixture = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
        )
        instructions = _fixture_agent_instructions(fixture)
        pdf_bytes = (fixture / "sample.pdf").read_bytes()
        direct_definition = await TaskDefinitionLoader().load(
            fixture / "task.toml"
        )
        flow_definition = await TaskDefinitionLoader().load(
            fixture / "flow_task.toml"
        )
        output = _extraction_output()
        orchestrator = ExtractionFakeOrchestrator(output)
        settings_values: list[Any] = []

        async def from_settings(
            loader: OrchestratorLoader,
            settings: object,
            *,
            tool_settings: object | None = None,
            tool_format: object | None = None,
            **kwargs: object,
        ) -> ExtractionFakeOrchestrator:
            _ = loader, tool_settings, tool_format, kwargs
            call_options = cast(Any, settings).call_options
            assert isinstance(call_options, Mapping)
            response_format = cast(
                Mapping[str, object],
                call_options["response_format"],
            )
            orchestrator.text_formats.append(
                {
                    "type": response_format["type"],
                    "name": response_format["name"],
                    "schema": response_format["schema"],
                    "strict": response_format["strict"],
                }
            )
            orchestrator.reasoning_options.append(
                cast(Mapping[str, object], call_options["reasoning"])
            )
            settings_values.append(settings)
            return orchestrator

        with TemporaryDirectory() as artifact_root:
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=cast(Any, object()),
                logger=getLogger("avalan.tests.task.extraction.flow"),
                participant_id=uuid4(),
                stack=stack,
            )
            agent_runner = AgentTaskTargetRunner(loader, ref_base=fixture)

            async def resolve_flow(context: TaskTargetContext) -> Flow:
                result = await FlowDefinitionLoader(
                    registry=task_flow_node_registry(
                        context,
                        agent_runner=agent_runner,
                        execution_roots=(fixture,),
                    )
                ).load_result(fixture / "flow.toml")
                assert result.flow is not None, result.issues
                return result.flow

            try:
                with patch.object(
                    OrchestratorLoader,
                    "from_settings",
                    new=from_settings,
                ):
                    direct_client = TaskClient(
                        InMemoryTaskStore(
                            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
                        ),
                        target=agent_runner,
                        artifact_store=LocalArtifactStore(
                            Path(artifact_root) / "direct",
                            raw_storage_allowed=True,
                        ),
                        hmac_provider=StaticHmacProvider(),
                        execution_roots=(fixture,),
                        definition_hash=lambda task: (
                            f"direct-parity-{task.task.name}"
                        ),
                        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
                    )
                    flow_client = TaskClient(
                        InMemoryTaskStore(
                            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
                        ),
                        target=FlowTaskTargetRunner(
                            ref_base=fixture,
                            flow_resolver=resolve_flow,
                        ),
                        artifact_store=LocalArtifactStore(
                            Path(artifact_root) / "flow",
                            raw_storage_allowed=True,
                        ),
                        hmac_provider=StaticHmacProvider(),
                        execution_roots=(fixture,),
                        definition_hash=lambda task: (
                            f"flow-parity-{task.task.name}"
                        ),
                        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
                    )
                    direct_descriptor = TaskClient.local_file(
                        "./sample.pdf",
                        mime_type="application/pdf",
                        size_bytes=len(pdf_bytes),
                    )
                    flow_descriptor = TaskClient.local_file(
                        "./sample.pdf",
                        mime_type="application/pdf",
                        size_bytes=len(pdf_bytes),
                    )

                    direct_result = await direct_client.run(
                        direct_definition,
                        input_value=direct_descriptor,
                    )
                    flow_result = await flow_client.run(
                        flow_definition,
                        input_value=flow_descriptor,
                    )
                    direct_inspection = await direct_client.inspect(
                        direct_result.run.run_id
                    )
                    flow_inspection = await flow_client.inspect(
                        flow_result.run.run_id
                    )
            finally:
                await stack.aclose()

        self.assertEqual(direct_result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(flow_result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(direct_result.output, output)
        self.assertEqual(flow_result.output, output)
        self.assertEqual(len(settings_values), 2)
        for settings in settings_values:
            agent_config = settings.agent_config
            self.assertIsInstance(agent_config, Mapping)
            self.assertEqual(agent_config["instructions"], instructions)
            self.assertNotIn("system", agent_config)
            self.assertNotIn("task", agent_config)
            self.assertEqual(settings.tools, [])
            call_options = settings.call_options
            self.assertIsInstance(call_options, Mapping)
            self.assertNotIn("tools", call_options)
            self.assertNotIn("tool_choice", call_options)
        self.assertEqual(
            [
                settings.engine_config["base_url"]
                for settings in settings_values
            ],
            [
                "https://tenant.openai.azure.com/openai/v1/",
                "https://tenant.openai.azure.com/openai/v1/",
            ],
        )
        self.assertEqual(
            orchestrator.reasoning_options,
            [{"effort": "high"}, {"effort": "high"}],
        )
        self.assertEqual(
            [
                canonical_schema_json(text_format["schema"])
                for text_format in orchestrator.text_formats
            ],
            [
                canonical_schema_json(direct_definition.output.schema),
                canonical_schema_json(flow_definition.output.schema),
            ],
        )
        self.assertEqual(len(orchestrator.inputs), 2)
        direct_message = _extraction_message_summary(
            orchestrator.inputs[0],
            pdf_bytes,
        )
        flow_message = _extraction_message_summary(
            orchestrator.inputs[1],
            pdf_bytes,
        )
        self.assertEqual(flow_message, direct_message)
        self.assertNotIn(instructions, str(direct_message))
        self.assertNotIn(instructions, str(flow_message))
        for inspection in (direct_inspection, flow_inspection):
            inspection_value = str(inspection.as_dict())
            _assert_extraction_usage(
                self,
                cast(tuple[object, ...], inspection.usage),
                inspection.usage_totals,
            )
            self.assertNotIn("Analyze the attached", inspection_value)
            self.assertNotIn("sample.pdf", inspection_value)
            self.assertNotIn("file_data", inspection_value)
            self.assertNotIn("private-provider-token", inspection_value)
            for sentinel in _private_usage_sentinels():
                self.assertNotIn(sentinel, inspection_value)

    async def test_poc_extraction_image_flow_matches_native_contract(
        self,
    ) -> None:
        fixture = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
        )
        instructions = _fixture_agent_instructions(fixture)
        pdf_bytes = (fixture / "sample.pdf").read_bytes()
        direct_definition = await TaskDefinitionLoader().load(
            fixture / "task.toml"
        )
        image_definition = await TaskDefinitionLoader().load(
            fixture / "image_flow_task.toml"
        )
        output = _extraction_output()
        orchestrator = ExtractionFakeOrchestrator(output)
        converter = ExtractionPdfImageConverter(
            (
                TaskFileConversionPageResult(
                    page_index=1,
                    page_count=2,
                    content=b"page one",
                    media_type="image/png",
                    width_pixels=20,
                    height_pixels=30,
                    metadata={"filename": "private-page-1.png"},
                ),
                TaskFileConversionPageResult(
                    page_index=2,
                    page_count=2,
                    content=b"page two",
                    media_type="image/png",
                    width_pixels=40,
                    height_pixels=50,
                    metadata={"filename": "private-page-2.png"},
                ),
            )
        )
        settings_values: list[Any] = []

        async def from_settings(
            loader: OrchestratorLoader,
            settings: object,
            *,
            tool_settings: object | None = None,
            tool_format: object | None = None,
            **kwargs: object,
        ) -> ExtractionFakeOrchestrator:
            _ = loader, tool_settings, tool_format, kwargs
            call_options = cast(Any, settings).call_options
            assert isinstance(call_options, Mapping)
            response_format = cast(
                Mapping[str, object],
                call_options["response_format"],
            )
            orchestrator.text_formats.append(
                {
                    "type": response_format["type"],
                    "name": response_format["name"],
                    "schema": response_format["schema"],
                    "strict": response_format["strict"],
                }
            )
            orchestrator.reasoning_options.append(
                cast(Mapping[str, object], call_options["reasoning"])
            )
            settings_values.append(settings)
            return orchestrator

        with TemporaryDirectory() as artifact_root:
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=cast(Any, object()),
                logger=getLogger("avalan.tests.task.extraction.image_flow"),
                participant_id=uuid4(),
                stack=stack,
            )
            agent_runner = AgentTaskTargetRunner(loader, ref_base=fixture)

            async def resolve_image_flow(context: TaskTargetContext) -> Flow:
                result = await FlowDefinitionLoader(
                    registry=task_flow_node_registry(
                        context,
                        agent_runner=agent_runner,
                        execution_roots=(fixture,),
                    )
                ).load_result(fixture / "image_flow.toml")
                assert result.flow is not None, result.issues
                return result.flow

            direct_store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            image_store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            direct_ids = iter(("direct-source",))
            image_ids = iter(
                (
                    "image-source",
                    "task-file-page-0001",
                    "task-file-page-0002",
                )
            )
            try:
                with patch.object(
                    OrchestratorLoader,
                    "from_settings",
                    new=from_settings,
                ):
                    direct_client = TaskClient(
                        direct_store,
                        target=agent_runner,
                        artifact_store=LocalArtifactStore(
                            Path(artifact_root) / "direct",
                            raw_storage_allowed=True,
                            id_factory=lambda: next(direct_ids),
                        ),
                        hmac_provider=StaticHmacProvider(),
                        execution_roots=(fixture,),
                        definition_hash=lambda task: (
                            f"native-image-parity-{task.task.name}"
                        ),
                        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
                    )
                    image_client = TaskClient(
                        image_store,
                        target=FlowTaskTargetRunner(
                            ref_base=fixture,
                            flow_resolver=resolve_image_flow,
                        ),
                        artifact_store=LocalArtifactStore(
                            Path(artifact_root) / "image",
                            raw_storage_allowed=True,
                            id_factory=lambda: next(image_ids),
                        ),
                        file_converters={"pdf_image": converter},
                        hmac_provider=StaticHmacProvider(),
                        execution_roots=(fixture,),
                        definition_hash=lambda task: (
                            f"image-flow-parity-{task.task.name}"
                        ),
                        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
                    )
                    direct_descriptor = TaskClient.local_file(
                        "./sample.pdf",
                        mime_type="application/pdf",
                        size_bytes=len(pdf_bytes),
                    )
                    image_descriptor = TaskClient.local_file(
                        "./sample.pdf",
                        mime_type="application/pdf",
                        size_bytes=len(pdf_bytes),
                    )

                    direct_result = await direct_client.run(
                        direct_definition,
                        input_value=direct_descriptor,
                    )
                    image_result = await image_client.run(
                        image_definition,
                        input_value=image_descriptor,
                    )
                    direct_inspection = await direct_client.inspect(
                        direct_result.run.run_id
                    )
                    image_artifacts = await image_store.list_artifacts(
                        image_result.run.run_id
                    )
                    image_inspection = await image_client.inspect(
                        image_result.run.run_id
                    )
            finally:
                await stack.aclose()

        self.assertEqual(direct_result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(image_result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(direct_result.output, output)
        self.assertEqual(image_result.output, output)
        self.assertEqual(len(settings_values), 2)
        for settings in settings_values:
            agent_config = settings.agent_config
            self.assertIsInstance(agent_config, Mapping)
            self.assertEqual(agent_config["instructions"], instructions)
            self.assertNotIn("system", agent_config)
            self.assertNotIn("task", agent_config)
            self.assertEqual(settings.tools, [])
            call_options = settings.call_options
            self.assertIsInstance(call_options, Mapping)
            self.assertNotIn("tools", call_options)
            self.assertNotIn("tool_choice", call_options)
        self.assertEqual(
            orchestrator.reasoning_options,
            [{"effort": "high"}, {"effort": "high"}],
        )
        self.assertEqual(
            [text_format["name"] for text_format in orchestrator.text_formats],
            ["invoice_extraction", "invoice_extraction"],
        )
        self.assertEqual(
            [
                canonical_schema_json(text_format["schema"])
                for text_format in orchestrator.text_formats
            ],
            [
                canonical_schema_json(direct_definition.output.schema),
                canonical_schema_json(image_definition.output.schema),
            ],
        )
        self.assertEqual(len(orchestrator.inputs), 2)
        direct_message = _extraction_message_summary(
            orchestrator.inputs[0],
            pdf_bytes,
        )
        self.assertEqual(len(_file_blocks(orchestrator.inputs[0])), 1)
        self.assertEqual(_image_blocks(orchestrator.inputs[0]), ())
        image_message = orchestrator.inputs[1]
        self.assertEqual(
            direct_message["text"],
            "Analyze the attached synthetic invoice PDF and "
            "extract all data according to the extraction instructions.",
        )
        self.assertEqual(_file_blocks(image_message), ())
        image_blocks = _image_blocks(image_message)
        self.assertEqual(len(image_blocks), 2)
        self.assertEqual(
            [block.image_url["mime_type"] for block in image_blocks],
            ["image/png", "image/png"],
        )
        self.assertEqual(
            [
                b64decode(cast(str, block.image_url["data"]))
                for block in image_blocks
            ],
            [b"page one", b"page two"],
        )
        self.assertEqual(len(converter.calls), 1)
        self.assertEqual(converter.calls[0][0], pdf_bytes)
        self.assertEqual(converter.calls[0][1], "application/pdf")
        self.assertEqual(
            converter.calls[0][2],
            {"dpi": 144, "format": "png", "pages": {"start": 1}},
        )
        self.assertEqual(
            [artifact.purpose for artifact in image_artifacts],
            [
                TaskArtifactPurpose.INPUT,
                TaskArtifactPurpose.CONVERTED,
                TaskArtifactPurpose.CONVERTED,
            ],
        )
        self.assertEqual(
            [artifact.artifact_id for artifact in image_artifacts],
            ["image-source", "task-file-page-0001", "task-file-page-0002"],
        )
        self.assertEqual(
            [
                artifact.metadata["page_index"]
                for artifact in image_artifacts[1:]
            ],
            [1, 2],
        )
        for inspection in (direct_inspection, image_inspection):
            _assert_extraction_usage(
                self,
                cast(tuple[object, ...], inspection.usage),
                inspection.usage_totals,
            )
        inspection_value = str(image_inspection.as_dict())
        self.assertNotIn(instructions, str(image_message))
        self.assertNotIn(instructions, inspection_value)
        self.assertNotIn("sample.pdf", str(image_message))
        self.assertNotIn("sample.pdf", inspection_value)
        self.assertNotIn("private-page", inspection_value)
        self.assertNotIn("file_data", inspection_value)
        self.assertNotIn("page one", inspection_value)
        self.assertNotIn("private-provider-token", inspection_value)
        for sentinel in _private_usage_sentinels():
            self.assertNotIn(sentinel, inspection_value)

    async def test_poc_extraction_failures_are_classified_and_sanitized(
        self,
    ) -> None:
        fixture = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
        )
        pdf_bytes = (fixture / "sample.pdf").read_bytes()
        definition = await TaskDefinitionLoader().load(fixture / "task.toml")
        definition = replace(
            definition,
            privacy=replace(
                definition.privacy,
                errors=PrivacyAction.REDACT,
            ),
        )
        cases = (
            (
                "provider",
                ExtractionProviderFailureResponse(),
                "provider",
                "provider.structured_output_failed",
                (),
            ),
            (
                "parse",
                ExtractionRawJsonResponse(
                    '{"line_items":["private provider body sk-test-secret"'
                ),
                "output_contract",
                "output.parse_failed",
                (),
            ),
            (
                "schema",
                ExtractionRawJsonResponse('{"line_items":[] }'),
                "output_contract",
                "output_contract.failed",
                ("output.invalid_type",),
            ),
        )

        for name, response, category, code, issue_codes in cases:
            with self.subTest(name=name):
                with TemporaryDirectory() as artifact_root:
                    orchestrator = ExtractionFailureOrchestrator(response)
                    loader = ExtractionFailureLoader(orchestrator)
                    client = TaskClient(
                        InMemoryTaskStore(
                            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
                        ),
                        target=AgentTaskTargetRunner(loader, ref_base=fixture),
                        artifact_store=LocalArtifactStore(
                            Path(artifact_root),
                            raw_storage_allowed=True,
                        ),
                        hmac_provider=StaticHmacProvider(),
                        execution_roots=(fixture,),
                        definition_hash=lambda task: (
                            f"extraction-failure-{name}"
                        ),
                        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
                    )
                    descriptor = TaskClient.local_file(
                        "./sample.pdf",
                        mime_type="application/pdf",
                        size_bytes=len(pdf_bytes),
                    )

                    result = await client.run(
                        definition,
                        input_value=descriptor,
                    )
                    output = await client.output(result.run.run_id)
                    inspection = await client.inspect(result.run.run_id)

                self.assertEqual(result.run.state, TaskRunState.FAILED)
                self.assertIsNone(result.output)
                self.assertFalse(output.ready)
                error_summary = cast(Mapping[str, object], output.error)
                self.assertEqual(error_summary["category"], category)
                self.assertEqual(error_summary["code"], code)
                if issue_codes:
                    details = cast(
                        Mapping[str, object],
                        error_summary["details"],
                    )
                    issues = cast(
                        tuple[Mapping[str, object], ...],
                        details["issues"],
                    )
                    self.assertEqual(
                        tuple(issue["code"] for issue in issues),
                        issue_codes,
                    )
                self.assertEqual(orchestrator.entered, 1)
                self.assertEqual(orchestrator.exited, 1)
                self.assertEqual(len(inspection.events), 1)
                inspection_value = str(inspection.as_dict())
                self.assertNotIn("Analyze the attached", inspection_value)
                self.assertNotIn("sample.pdf", inspection_value)
                self.assertNotIn("file_data", inspection_value)
                self.assertNotIn("sk-test-secret", inspection_value)
                self.assertNotIn("private provider", inspection_value)
                self.assertNotIn("private-provider-token", inspection_value)
                self.assertNotIn("token_id", inspection_value)

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
                    definition = await TaskDefinitionLoader().load(
                        _write_provider_task_workspace(
                            root,
                            provider_uri=uri,
                            mime_type=descriptor.mime_type or "text/plain",
                        )
                    )
                    store = InMemoryTaskStore(
                        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
                    )
                    provider_client = ProviderFakeClient(
                        reference_key=reference_key,
                        reference=reference,
                        mime_type=descriptor.mime_type or "text/plain",
                    )
                    loader = ProviderFakeLoader(provider_client)
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
                self.assertEqual(provider_client.inputs, loader.inputs)
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

    async def test_sdk_provider_reference_modes_run_directly(
        self,
    ) -> None:
        cases = (
            (
                "openai-url",
                "ai://env:KEY@openai/gpt-4o-mini",
                TaskClient.hosted_url(
                    "openai",
                    "https://files.example.test/openai-private.pdf",
                    mime_type="application/pdf",
                    size_bytes=2048,
                    owner_scope="tenant-secret",
                ),
                "file_url",
                "https://files.example.test/openai-private.pdf",
            ),
            (
                "anthropic-file-id",
                "ai://env:KEY@anthropic/claude-3-5-sonnet",
                TaskClient.provider_file_id(
                    "anthropic",
                    "file-anthropic-private",
                    mime_type="application/pdf",
                    size_bytes=2048,
                    owner_scope="tenant-secret",
                ),
                "file_id",
                "file-anthropic-private",
            ),
            (
                "google-object-uri",
                "ai://env:KEY@google/gemini-2.0-flash",
                TaskClient.object_store_uri(
                    "google",
                    "gs://bucket/google-private.pdf",
                    mime_type="application/pdf",
                    size_bytes=2048,
                    owner_scope="tenant-secret",
                ),
                "file_url",
                "gs://bucket/google-private.pdf",
            ),
            (
                "bedrock-object-uri",
                "ai://env:KEY@bedrock/us.anthropic.claude",
                TaskClient.object_store_uri(
                    "bedrock",
                    "s3://bucket/bedrock-private.txt",
                    mime_type="text/plain",
                    size_bytes=2048,
                    owner_scope="tenant-secret",
                ),
                "file_url",
                "s3://bucket/bedrock-private.txt",
            ),
        )

        for name, uri, descriptor, reference_key, reference in cases:
            with self.subTest(provider=name):
                with TemporaryDirectory() as root_name:
                    root = Path(root_name)
                    _write_provider_task_workspace(
                        root,
                        provider_uri=uri,
                        mime_type=descriptor.mime_type or "text/plain",
                    )
                    provider_client = ProviderFakeClient(
                        reference_key=reference_key,
                        reference=reference,
                        mime_type=descriptor.mime_type or "text/plain",
                    )
                    loader = ProviderFakeLoader(provider_client)
                    client = TaskClient(
                        InMemoryTaskStore(
                            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
                        ),
                        target=AgentTaskTargetRunner(loader, ref_base=root),
                        hmac_provider=StaticHmacProvider(),
                        execution_roots=(root,),
                        definition_hash=lambda task: (
                            f"provider-sdk-{name}-e2e"
                        ),
                        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
                    )

                    result = await client.run(
                        _provider_definition(
                            mime_type=descriptor.mime_type or "text/plain"
                        ),
                        input_value=descriptor,
                    )
                    inspection = await client.inspect(result.run.run_id)

                self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
                self.assertEqual(result.output, "provider accepted")
                self.assertEqual(loader.entered, 1)
                self.assertEqual(loader.exited, 1)
                self.assertEqual(len(loader.inputs), 1)
                self.assertEqual(provider_client.inputs, loader.inputs)
                block = _only_file_block(loader.inputs[0])
                self.assertEqual(block.file[reference_key], reference)
                self.assertEqual(block.file["mime_type"], descriptor.mime_type)
                self.assertEqual(len(inspection.events), 1)
                self.assertEqual(len(inspection.usage), 1)
                self.assertEqual(inspection.usage_totals.total_tokens, 8)
                self.assertEqual(inspection.artifacts, ())
                inspection_value = str(inspection.as_dict())
                self.assertNotIn(reference, inspection_value)
                self.assertNotIn("tenant-secret", inspection_value)
                self.assertNotIn("private-provider-token", inspection_value)
                self.assertNotIn("token_id", inspection_value)

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
        self.assertEqual(content[0].file["filename"], "task-file.pdf")
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
            definition = await TaskDefinitionLoader().load(
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

    async def test_unsupported_provider_rejects_before_provider_client(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = await TaskDefinitionLoader().load(
                _write_provider_task_workspace(
                    root,
                    provider_uri="ai://env:KEY@unknown/model",
                )
            )
            provider_client = ProviderFakeClient(
                reference_key="file_id",
                reference="file-private",
                mime_type="application/pdf",
            )
            loader = ProviderFakeLoader(provider_client)
            client = TaskClient(
                InMemoryTaskStore(),
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "provider-unsupported-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )
            descriptor = TaskClient.provider_file_id(
                "unknown",
                "file-private",
                mime_type="application/pdf",
                owner_scope="tenant-secret",
            )

            validation = await client.validate(
                definition,
                input_value=descriptor,
            )
            with self.assertRaises(TaskValidationError) as error:
                await client.run(definition, input_value=descriptor)

        self.assertFalse(validation.valid)
        self.assertEqual(validation.issues[0].code, "input.invalid_file")
        self.assertEqual(validation.issues[0].path, "input.type")
        self.assertEqual(loader.paths, [])
        self.assertEqual(loader.inputs, [])
        self.assertEqual(provider_client.inputs, [])
        self.assertNotIn("file-private", str(error.exception))
        self.assertNotIn("tenant-secret", str(error.exception))

    async def test_provider_reference_conversion_rejects_before_execution(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = await TaskDefinitionLoader().load(
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

    async def test_provider_object_store_scheme_rejects_before_execution(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            _write_provider_task_workspace(
                root,
                provider_uri="ai://env:KEY@google/gemini-2.0-flash",
            )
            provider_client = ProviderFakeClient(
                reference_key="file_url",
                reference="s3://bucket/google-private.pdf",
                mime_type="application/pdf",
            )
            loader = ProviderFakeLoader(provider_client)
            client = TaskClient(
                InMemoryTaskStore(
                    clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
                ),
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "provider-object-uri-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            result = await client.run(
                _provider_definition(),
                input_value=TaskClient.object_store_uri(
                    "google",
                    "s3://bucket/google-private.pdf",
                    mime_type="application/pdf",
                    owner_scope="tenant-secret",
                ),
            )
            inspection = await client.inspect(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(loader.paths, [])
        self.assertEqual(loader.inputs, [])
        self.assertEqual(provider_client.inputs, [])
        error_summary = cast(Mapping[str, object], result.run.result.error)
        self.assertEqual(error_summary["category"], "input_contract")
        self.assertEqual(error_summary["code"], "input_contract.failed")
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("s3://bucket/google-private.pdf", inspection_value)
        self.assertNotIn("tenant-secret", inspection_value)

    async def test_loaded_text_task_runs_directly_and_inspects_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = await TaskDefinitionLoader().load(
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
            toml_definition = await TaskDefinitionLoader().load(
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

        self.assertEqual(
            await spec_hash(toml_definition),
            await spec_hash(sdk_definition),
        )
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

    async def test_returned_usage_output_reaches_direct_inspection(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = replace(
                await TaskDefinitionLoader().load(
                    _write_text_task_workspace(root)
                ),
                observability=TaskObservabilityPolicy(
                    metrics=True,
                    trace=False,
                    capture_events=False,
                ),
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            target = ReturnedUsageTextSummaryTarget()
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda task: "direct-returned-usage-e2e",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            result = await client.run(
                definition,
                input_value="private text prompt",
                metadata={"tenant": "safe"},
            )
            output = await client.output(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "public summary")
        self.assertTrue(output.ready)
        self.assertEqual(output.output_summary, {"privacy": "<redacted>"})
        self.assertEqual(target.input_values, ["private text prompt"])
        self.assertEqual(target.metadata_values, [{"tenant": "safe"}])
        self.assertEqual(len(inspection.usage), 1)
        self.assertEqual(inspection.usage[0].source, UsageSource.EXACT)
        self.assertEqual(inspection.usage[0].totals.input_tokens, 9)
        self.assertEqual(inspection.usage_totals.cached_input_tokens, 4)
        self.assertEqual(inspection.usage_totals.output_tokens, 6)
        self.assertEqual(inspection.usage_totals.reasoning_tokens, 2)
        self.assertEqual(inspection.usage_totals.total_tokens, 15)
        self.assertEqual(
            inspection.usage[0].metadata,
            {"provider_family": "openai"},
        )
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private text prompt", inspection_value)
        self.assertNotIn("raw_response_id", inspection_value)
        self.assertNotIn("private-response-id", inspection_value)
        self.assertNotIn("public summary", inspection_value)

    async def test_sdk_schema_ref_task_runs_with_same_identity_as_toml(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            (root / "agents").mkdir()
            (root / "schemas").mkdir()
            (root / "agents" / "structured.toml").write_text(
                """
[agent]
name = "Structured"
task = "Return structured answers."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            (root / "schemas" / "answer.json").write_text(
                """
                {
                  "type": "object",
                  "required": ["answer"],
                  "properties": {"answer": {"type": "string"}}
                }
                """,
                encoding="utf-8",
            )
            task_path = root / "structured.task.toml"
            task_path.write_text(
                """
[task]
name = "structured_schema_ref"
version = "1"

[input]
type = "string"

[output]
type = "object"
schema_ref = "schemas/answer.json"

[execution]
type = "agent"
ref = "agents/structured.toml"

[observability]
metrics = false
trace = false
capture_events = false
""",
                encoding="utf-8",
            )
            toml_definition = await TaskDefinitionLoader().load(task_path)
            sdk_definition = TaskDefinition(
                task=TaskMetadata(name="structured_schema_ref", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.object(
                    schema_ref="schemas/answer.json"
                ),
                execution=TaskExecutionTarget.agent("agents/structured.toml"),
                definition_base=root / "sdk_structured.task.toml",
                observability=TaskObservabilityPolicy(
                    metrics=False,
                    trace=False,
                    capture_events=False,
                ),
            )
            store = InMemoryTaskStore(
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            target = StructuredTarget(output={"answer": "ok"})
            client = TaskClient(
                store,
                target=target,
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            toml_result = await client.run(
                toml_definition,
                input_value="private question",
            )
            sdk_result = await client.run(
                sdk_definition,
                input_value="private question",
            )
            record = await store.get_definition(sdk_result.run.definition_id)
            toml_hash = await spec_hash(toml_definition)
            sdk_hash = await spec_hash(sdk_definition)
            root_text = str(root)

        self.assertEqual(toml_hash, sdk_hash)
        self.assertEqual(
            toml_result.run.definition_id,
            sdk_result.run.definition_id,
        )
        self.assertEqual(toml_result.output, {"answer": "ok"})
        self.assertEqual(sdk_result.output, {"answer": "ok"})
        self.assertIsNone(record.definition.output.schema_ref)
        self.assertIsNone(record.definition.definition_base)
        self.assertEqual(
            record.definition.output.schema,
            toml_definition.output.schema,
        )
        self.assertEqual(
            target.definition_refs, ["agents/structured.toml"] * 2
        )
        self.assertNotIn(root_text, str(record.definition))
        self.assertNotIn("private question", str(record.definition))

    async def test_loaded_text_task_retries_and_inspects_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as root_name:
            root = Path(root_name)
            definition = replace(
                await TaskDefinitionLoader().load(
                    _write_text_task_workspace(root)
                ),
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
                await TaskDefinitionLoader().load(
                    _write_text_task_workspace(root)
                ),
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
            definition = await TaskDefinitionLoader().load(
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
            definition = await TaskDefinitionLoader().load(
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
            definition = await TaskDefinitionLoader().load(
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
            definition = await TaskDefinitionLoader().load(
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
            definition = await TaskDefinitionLoader().load(
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

    async def test_completed_streaming_usage_records_exact_totals(
        self,
    ) -> None:
        definition = TaskDefinition(
            task=TaskMetadata(name="streaming_summary", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/summarizer.toml"),
            run=TaskRunPolicy.direct(timeout_seconds=60),
            observability=TaskObservabilityPolicy(
                metrics=True,
                trace=False,
                capture_events=False,
            ),
        )
        store = InMemoryTaskStore(
            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
        )
        client = TaskClient(
            store,
            target=StreamingUsageTarget(),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "streaming-usage-e2e",
            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        )

        result = await client.run(
            definition,
            input_value="private prompt",
        )
        inspection = await client.inspect(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "public stream summary")
        self.assertEqual(len(inspection.usage), 1)
        self.assertEqual(inspection.usage[0].source, UsageSource.EXACT)
        self.assertEqual(inspection.usage_totals.input_tokens, 8)
        self.assertEqual(inspection.usage_totals.cached_input_tokens, 3)
        self.assertEqual(
            inspection.usage_totals.cache_creation_input_tokens,
            2,
        )
        self.assertEqual(inspection.usage_totals.output_tokens, 5)
        self.assertEqual(inspection.usage_totals.reasoning_tokens, 1)
        self.assertEqual(inspection.usage_totals.total_tokens, 13)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private prompt", inspection_value)

    async def test_azure_response_usage_records_exact_safe_totals(
        self,
    ) -> None:
        definition = TaskDefinition(
            task=TaskMetadata(name="azure_summary", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/summarizer.toml"),
            run=TaskRunPolicy.direct(timeout_seconds=60),
            observability=TaskObservabilityPolicy(
                metrics=True,
                trace=False,
                capture_events=True,
            ),
        )
        store = InMemoryTaskStore(
            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
        )
        client = TaskClient(
            store,
            target=AzureUsageTarget(),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "azure-usage-e2e",
            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        )

        result = await client.run(
            definition,
            input_value="private prompt",
        )
        inspection = await client.inspect(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "public azure summary")
        self.assertEqual(len(inspection.usage), 1)
        self.assertEqual(inspection.usage[0].source, UsageSource.EXACT)
        self.assertEqual(
            inspection.usage[0].metadata,
            {"provider_family": UsageProviderFamily.AZURE_OPENAI.value},
        )
        self.assertEqual(inspection.usage_totals.input_tokens, 12)
        self.assertEqual(inspection.usage_totals.cached_input_tokens, 5)
        self.assertIsNone(inspection.usage_totals.cache_creation_input_tokens)
        self.assertEqual(inspection.usage_totals.output_tokens, 8)
        self.assertEqual(inspection.usage_totals.reasoning_tokens, 3)
        self.assertEqual(inspection.usage_totals.total_tokens, 20)
        inspection_value = str(inspection.as_dict())
        self.assertNotIn("private prompt", inspection_value)
        self.assertNotIn("private-deployment-name", inspection_value)
        self.assertNotIn("private-response-id", inspection_value)

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
            definition = await TaskDefinitionLoader().load(
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
            definition = await TaskDefinitionLoader().load(
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
            definition = await TaskDefinitionLoader().load(
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
