from ast import Attribute, Call, ImportFrom, Name, parse, walk
from asyncio import CancelledError, sleep, wait_for
from asyncio import run as asyncio_run
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from inspect import getsource, isawaitable
from pathlib import Path
from sys import path as sys_path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

sys_path.append(str(Path(__file__).parents[1] / "stores"))

from pgsql_contract_test import (  # type: ignore[import-not-found]
    FakePgsqlTaskDatabase,
)
from store_contract_test import (  # type: ignore[import-not-found]
    SequenceClock,
    SequenceIds,
)

from avalan.cli.commands import task as task_cmds
from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentText,
    MessageRole,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallOutcome,
    ToolDescriptor,
    ToolNameResolution,
    ToolNameResolutionStatus,
)
from avalan.event import Event, EventObservabilityPayload, EventType
from avalan.flow import (
    FlowConditionOperator,
    FlowConditionPlan,
    FlowConditionValueType,
    FlowDefinition,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowEdgePlan,
    FlowEdgeState,
    FlowEdgeTrace,
    FlowEntryBehavior,
    FlowExecutionPlan,
    FlowExecutionTrace,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowJoinPlan,
    FlowJoinPolicyType,
    FlowLoopPlan,
    FlowMappingKind,
    FlowMappingPlan,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodePlan,
    FlowNodeState,
    FlowNodeTrace,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowRetryBackoffStrategy,
    FlowRetryPlan,
    FlowSourceSpan,
    FlowTimeoutPlan,
    InMemoryFlowStateStore,
    PgsqlFlowStateStore,
    compile_flow_definition,
    parse_flow_selector,
    validate_flow_definition,
)
from avalan.flow.flow import Flow
from avalan.flow.loader import FlowDefinitionLoader
from avalan.flow.node import Node
from avalan.flow.registry import FlowNodeConfigurationError
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
)
from avalan.server.routers import flow as flow_router_module
from avalan.task import (
    DROPPED_MARKER,
    ENCRYPTED_MARKER,
    HASHED_MARKER,
    REDACTED_MARKER,
    STORED_ENVELOPE_MARKER,
    STORED_MARKER,
    DirectTaskRunner,
    PrivacyAction,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactState,
    TaskDefinition,
    TaskEventCategory,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskFeature,
    TaskFileConversionError,
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
    TaskProviderReference,
    TaskProviderReferenceKind,
    TaskRunPolicy,
    TaskRunState,
    TaskStoreNotFoundError,
    TaskTargetContext,
    TaskValidationCategory,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
    pdf_image_converter_capability,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.store import TaskExecutionContext
from avalan.task.stores import InMemoryTaskStore, PgsqlTaskStore
from avalan.task.targets import (
    FLOW_TASK_FILES_KEY,
    FLOW_TASK_INPUT_KEY,
    AgentTaskTargetRunner,
    FlowTaskTargetRunner,
    flow_task_input_binding,
    task_flow_node_registry,
    validate_flow_task_compatibility,
)
from avalan.task.targets import flow as flow_target_module
from avalan.task.usage import usage_flow_node


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
            secret=b"flow-target-secret",
        )


class FlowAgentResponse:
    input_token_count = 5
    output_token_count = 3
    total_token_count = 8

    def __init__(self, text: str) -> None:
        self.text = text

    async def to_str(self) -> str:
        return self.text


class FlowAgentEventManager:
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

    async def emit(self) -> None:
        for listener in tuple(self.listeners):
            result = listener(
                Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={"token": "private-token", "file_id": "file-123"},
                )
            )
            if result is not None:
                await result


class FlowAgentOrchestrator:
    def __init__(self, loader: "FlowAgentLoader") -> None:
        self._loader = loader
        self.event_manager = loader.event_manager

    async def __aenter__(self) -> "FlowAgentOrchestrator":
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

    async def __call__(self, input: object) -> FlowAgentResponse:
        self._loader.inputs.append(input)
        await self.event_manager.emit()
        if self._loader.error is not None:
            raise self._loader.error
        return FlowAgentResponse(self._loader.text)


class FlowAgentLoader:
    def __init__(
        self,
        *,
        text: str = "flow agent summary",
        error: BaseException | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.event_manager = FlowAgentEventManager()
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
    ) -> FlowAgentOrchestrator:
        _ = agent_id, disable_memory, uri, tool_settings
        self.paths.append(path)
        return FlowAgentOrchestrator(self)


class CapturingTaskTargetRunner:
    def __init__(
        self,
        *,
        issues: tuple[TaskValidationIssue, ...] = (),
        output: object = "agent output",
    ) -> None:
        self.issues = issues
        self.output = output
        self.contexts: list[TaskTargetContext] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return self.issues

    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        return self.output


class FailingArtifactStore:
    async def stat(self, ref: TaskArtifactRef) -> object:
        _ = ref
        raise AssertionError("private artifact metadata was read")

    async def open_stream(
        self,
        ref: TaskArtifactRef,
        *,
        max_bytes: int | None = None,
    ) -> object:
        _ = ref, max_bytes
        raise AssertionError("private bytes were read")


class RecordingPdfPageConverter:
    name = "pdf_image"
    version = "fake"

    def __init__(
        self,
        pages: tuple[TaskFileConversionPageResult, ...],
        *,
        dependency_gates: tuple[TaskFeature, ...] = (),
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
            dependency_gates=dependency_gates,
        )

    @property
    def capability(self) -> TaskFileConverterCapability:
        return self._capability

    def validate_options(self, options: Mapping[str, object]) -> None:
        if options.get("format") == "gif":
            raise TaskFileConversionError("private invalid format")

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
        self.calls.append((content, source_media_type, dict(options or {})))
        return TaskFileConversionPageCollection(
            pages=self._pages,
            metadata={"backend": "fake"},
        )


class SingleOutputConverter:
    name = "single"
    version = "fake"
    capability = TaskFileConverterCapability(
        source_mime_types=("application/pdf",),
        output_mime_types=("image/png",),
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
        _ = content, source_media_type, options
        return TaskFileConversionResult(
            content=b"page",
            media_type="image/png",
            metadata={},
        )


class EmptyToolResolver:
    def list_tools(self) -> list[ToolDescriptor]:
        return []

    def resolve_tool_name(
        self,
        name: str,
        *,
        provider_originated: bool = False,
    ) -> ToolNameResolution:
        _ = provider_originated
        return ToolNameResolution(
            requested_name=name,
            status=ToolNameResolutionStatus.UNKNOWN,
            canonical_name=None,
            candidates=[],
        )

    def validate_tool_call(self, call: ToolCall) -> ToolCallDiagnostic | None:
        _ = call
        return None

    async def execute_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
    ) -> ToolCallOutcome:
        _ = call, context
        raise AssertionError("tool execution is not expected")


class StaticToolResolver(EmptyToolResolver):
    def __init__(self, descriptors: tuple[ToolDescriptor, ...]) -> None:
        self._descriptors = descriptors

    def list_tools(self) -> list[ToolDescriptor]:
        return list(self._descriptors)

    def resolve_tool_name(
        self,
        name: str,
        *,
        provider_originated: bool = False,
    ) -> ToolNameResolution:
        _ = provider_originated
        for descriptor in self._descriptors:
            if descriptor.name == name:
                return ToolNameResolution(
                    requested_name=name,
                    status=ToolNameResolutionStatus.EXACT,
                    canonical_name=descriptor.name,
                    candidates=[descriptor.name],
                )
        return super().resolve_tool_name(
            name,
            provider_originated=provider_originated,
        )


def _page_result(
    page_index: int,
    content: bytes,
    *,
    width_pixels: int = 10,
    height_pixels: int = 10,
) -> TaskFileConversionPageResult:
    return TaskFileConversionPageResult(
        page_index=page_index,
        page_count=2,
        content=content,
        media_type="image/png",
        width_pixels=width_pixels,
        height_pixels=height_pixels,
        metadata={"page": page_index},
    )


def _id_factory(values: tuple[str, ...]) -> Callable[[], str]:
    ids = iter(values)

    def next_id() -> str:
        return next(ids)

    return next_id


def _write_agent_flow_workspace(
    root: Path,
    *,
    uri: str = "ai://env:KEY@openai/gpt-4o-mini",
) -> Path:
    agents = root / "agents"
    agents.mkdir()
    (agents / "review.toml").write_text(
        f"""
[agent]
name = "Flow reviewer"
task = "Review the supplied file."
user = "Review."

[engine]
uri = "{uri}"
""",
        encoding="utf-8",
    )
    flow_path = root / "flow.toml"
    flow_path.write_text(
        """
[flow]
name = "file_review"
entrypoint = "review"
output_node = "review"

[flow.input]
name = "input"
type = "file"
mime_types = ["application/pdf"]

[flow.output]
name = "summary"
type = "text"

[nodes.review]
type = "agent"
ref = "agents/review.toml"
input = "__task_input__"
""",
        encoding="utf-8",
    )
    return flow_path


def _flow_loader_resolver(
    path: Path,
    *,
    agent_runner: AgentTaskTargetRunner,
    root: Path,
) -> Callable[[TaskTargetContext], Awaitable[Flow]]:
    async def resolve(context: TaskTargetContext) -> Flow:
        result = await FlowDefinitionLoader(
            registry=task_flow_node_registry(
                context,
                agent_runner=agent_runner,
                execution_roots=(root,),
            )
        ).load_result(path)
        assert result.flow is not None, result.issues
        return result.flow

    return resolve


def _strict_flow_loader_resolver(
    root: Path,
) -> Callable[[TaskTargetContext], Awaitable[FlowDefinition]]:
    async def resolve(context: TaskTargetContext) -> FlowDefinition:
        flow_ref = context.definition.execution.ref
        path = Path(flow_ref)
        if not path.is_absolute():
            path = root / path
        result = await FlowDefinitionLoader(
            registry=task_flow_node_registry(context)
        ).load_validation_result(path)
        if result.definition is None:
            raise TaskValidationError(
                tuple(
                    TaskValidationIssue(
                        code=issue.code,
                        path=issue.path,
                        message=issue.message,
                        hint=issue.hint,
                        category=TaskValidationCategory.UNSUPPORTED,
                    )
                    for issue in result.issues
                )
            )
        return result.definition

    return resolve


def _task_flow_loading_async_violations(
    source: str,
) -> tuple[str, ...]:
    tree = parse(source)
    violations: list[str] = []
    for node in walk(tree):
        if isinstance(node, ImportFrom) and node.module == "asyncio":
            for alias in node.names:
                if alias.name in {"run", "to_thread"}:
                    violations.append(
                        f"line {node.lineno} imports asyncio.{alias.name}"
                    )
        if isinstance(node, Call):
            name = _call_name(node)
            if name in {
                "asyncio.run",
                "asyncio.to_thread",
                "to_thread",
                "run_until_complete",
                "read_text",
                "read_bytes",
                "write_text",
                "write_bytes",
            }:
                violations.append(f"line {node.lineno} calls {name}")
    return tuple(violations)


def _call_name(node: Call) -> str:
    func = node.func
    if isinstance(func, Name):
        return func.id
    if isinstance(func, Attribute):
        if isinstance(func.value, Name) and func.value.id == "asyncio":
            return f"asyncio.{func.attr}"
        return func.attr
    return ""


def _write_strict_graph_flow(
    path: Path,
    *,
    diagram: str,
    graph_path: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = "file" if graph_path is not None else "inline"
    graph_source = (
        f'path = "{graph_path}"'
        if graph_path is not None
        else f"diagram = '''\n{diagram}'''"
    )
    path.write_text(
        f"""
[flow]
name = "graph-task-flow"
version = "1"

[[inputs]]
name = "prompt"
type = "string"

[[outputs]]
name = "answer"
type = "text"

[entry]
type = "node"
node = "start"

[output_behavior]
type = "map"

[output_behavior.outputs]
answer = "finish.value"

[graph]
format = "mermaid"
source = "{source}"
mode = "executable"
{graph_source}

[nodes.start]
type = "pass-through"

[nodes.start.mapping.value]
type = "select"
source = "input.prompt"

[nodes.finish]
type = "pass-through"

[nodes.finish.mapping.value]
type = "select"
source = "start.value"
""",
        encoding="utf-8",
    )


def _agent_node_flow(
    context: TaskTargetContext,
    *,
    agent_runner: CapturingTaskTargetRunner,
) -> Flow:
    registry = task_flow_node_registry(context, agent_runner=agent_runner)
    flow = Flow()
    flow.add_node(
        registry.build(
            FlowNodeDefinition(
                name="review",
                type="agent",
                ref="agents/review.toml",
            )
        )
    )
    return flow


def _single_incoming_agent_flow(
    context: TaskTargetContext,
    *,
    agent_runner: CapturingTaskTargetRunner,
) -> Flow:
    registry = task_flow_node_registry(context, agent_runner=agent_runner)
    flow = Flow()
    flow.add_node(Node("start", func=lambda _: "ready"))
    flow.add_node(
        registry.build(
            FlowNodeDefinition(
                name="review",
                type="agent",
                ref="agents/review.toml",
            )
        )
    )
    flow.add_connection("start", "review")
    return flow


def _file_selecting_agent_flow(
    context: TaskTargetContext,
    *,
    agent_runner: CapturingTaskTargetRunner,
    generated_files: tuple[TaskInputFile, ...],
    file_policy: str,
) -> Flow:
    registry = task_flow_node_registry(context, agent_runner=agent_runner)
    flow = Flow()
    flow.add_node(Node("start", func=lambda _: "ready"))
    flow.add_node(
        Node("render", func=lambda _: {"files": list(generated_files)})
    )
    flow.add_node(
        registry.build(
            FlowNodeDefinition(
                name="review",
                type="agent",
                ref="agents/review.toml",
                config={
                    "files_input": "render.files",
                    "file_policy": file_policy,
                },
            )
        )
    )
    flow.add_connection("start", "render")
    flow.add_connection("render", "review")
    return flow


def _multi_incoming_agent_flow(
    context: TaskTargetContext,
    *,
    agent_runner: CapturingTaskTargetRunner,
) -> Flow:
    registry = task_flow_node_registry(context, agent_runner=agent_runner)
    flow = Flow()
    flow.add_node(Node("start", func=lambda _: "seed"))
    flow.add_node(Node("left", func=lambda _: "left"))
    flow.add_node(Node("right", func=lambda _: "right"))
    flow.add_node(
        registry.build(
            FlowNodeDefinition(
                name="review",
                type="agent",
                ref="agents/review.toml",
            )
        )
    )
    flow.add_connection("start", "left")
    flow.add_connection("start", "right")
    flow.add_connection("left", "review")
    flow.add_connection("right", "review")
    return flow


class FlowTaskTargetRunnerValidationTest(TestCase):
    def test_flow_node_event_listener_adds_flow_node(self) -> None:
        captured: list[Event] = []
        listener = flow_target_module._flow_node_event_listener(
            "analyze_pov_1",
            captured.append,
        )
        observability = EventObservabilityPayload.canonical_stream(
            {
                "kind": "reasoning.delta",
                "channel": "reasoning",
                "sequence": 1,
            }
        )

        assert listener is not None
        result = listener(
            Event(
                type=EventType.TOKEN_GENERATED,
                payload={"token_type": "ReasoningToken"},
                observability_payload=observability,
                started=1.0,
                finished=2.0,
                elapsed=1.0,
            )
        )

        self.assertIsNone(result)
        self.assertEqual(len(captured), 1)
        payload = cast(dict[str, object], captured[0].payload)
        self.assertEqual(payload["flow_node"], "analyze_pov_1")
        self.assertEqual(payload["token_type"], "ReasoningToken")
        self.assertIs(captured[0].observability_payload, observability)
        self.assertEqual(captured[0].started, 1.0)
        self.assertEqual(captured[0].finished, 2.0)
        self.assertEqual(captured[0].elapsed, 1.0)

    def test_flow_node_event_listener_handles_non_mapping_payload(
        self,
    ) -> None:
        captured: list[Event] = []
        listener = flow_target_module._flow_node_event_listener(
            "analyze_pov_1",
            captured.append,
        )

        assert listener is not None
        listener(Event(type=EventType.TOOL_PROCESS, payload=None))

        self.assertEqual(
            captured[0].payload,
            {"flow_node": "analyze_pov_1"},
        )

    def test_flow_node_event_listener_returns_none_without_listener(
        self,
    ) -> None:
        self.assertIsNone(
            flow_target_module._flow_node_event_listener(
                "analyze_pov_1",
                None,
            )
        )

    def test_flow_node_usage_observer_adds_flow_node(self) -> None:
        captured: list[object] = []
        observer = flow_target_module._flow_node_usage_observer(
            "analyze_pov_1",
            captured.append,
        )
        response = SimpleNamespace()

        assert observer is not None
        result = observer(response)

        self.assertIsNone(result)
        self.assertEqual(captured, [response])
        self.assertEqual(usage_flow_node(response), "analyze_pov_1")

    def test_flow_node_usage_observer_returns_none_without_observer(
        self,
    ) -> None:
        self.assertIsNone(
            flow_target_module._flow_node_usage_observer(
                "analyze_pov_1",
                None,
            )
        )

    def test_flow_target_accepts_observability_without_path_leaks(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_path = root / "flows" / "private.toml"
            flow_path.parent.mkdir()
            flow_path.write_text("secret = 'private flow'\n", encoding="utf-8")
            runner = FlowTaskTargetRunner(ref_base=root)

            issues = self._run_validate(
                runner,
                self._definition(),
                TaskValidationContext(execution_roots=(root,)),
            )

        self.assertEqual(issues, ())
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("private.toml", rendered)
        self.assertNotIn("private flow", rendered)

    def test_flow_target_accepts_file_input_and_artifact_output(
        self,
    ) -> None:
        runner = FlowTaskTargetRunner()

        issues = self._run_validate(
            runner,
            self._definition(
                input_contract=TaskInputContract.file(),
                output_contract=TaskOutputContract.artifact_array(),
            ),
            TaskValidationContext(),
        )

        self.assertEqual(issues, ())

    def test_flow_target_rejects_unsafe_references_without_raw_ref(
        self,
    ) -> None:
        runner = FlowTaskTargetRunner()
        cases = (
            "../secret/private.toml",
            "/secret/private.toml",
            "flows\\private.toml",
            "https://example.test/private.toml",
        )

        for ref in cases:
            with self.subTest(ref=ref):
                issues = self._run_validate(
                    runner,
                    self._definition(
                        execution=TaskExecutionTarget.flow(ref),
                    ),
                    TaskValidationContext(execution_roots=(Path("/tmp"),)),
                )
                self.assertEqual(issues[0].code, "execution.path_escape")
                rendered = " ".join(
                    value
                    for issue in issues
                    for value in issue.as_dict().values()
                )
                self.assertNotIn("private.toml", rendered)
                self.assertNotIn(ref, rendered)

    def test_flow_target_rejects_symlink_escape(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            (outside / "private.toml").write_text(
                "secret = 'outside'\n",
                encoding="utf-8",
            )
            symlink = root / "flow.toml"
            symlink.symlink_to(outside / "private.toml")
            runner = FlowTaskTargetRunner(ref_base=root)

            issues = self._run_validate(
                runner,
                self._definition(
                    execution=TaskExecutionTarget.flow("flow.toml"),
                ),
                TaskValidationContext(execution_roots=(root,)),
            )

        self.assertEqual(issues[0].code, "execution.path_escape")
        self.assertEqual(issues[0].path, "execution.ref")
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("private.toml", rendered)
        self.assertNotIn("outside", rendered)

    def test_flow_target_fails_closed_when_reference_cannot_resolve(
        self,
    ) -> None:
        runner = FlowTaskTargetRunner()

        with patch(
            "avalan.task.targets.flow.Path.resolve",
            side_effect=(Path("/tmp/allowed"), OSError("resolver secret")),
        ):
            issues = self._run_validate(
                runner,
                self._definition(),
                TaskValidationContext(execution_roots=(Path("/tmp/allowed"),)),
            )

        self.assertEqual(issues[0].code, "execution.path_escape")
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("resolver secret", rendered)
        self.assertNotIn("report.toml", rendered)

    def test_non_flow_target_returns_unknown_target_issue(self) -> None:
        runner = FlowTaskTargetRunner()

        issues = self._run_validate(
            runner,
            self._definition(
                execution=TaskExecutionTarget.agent("agents/valid.toml"),
            ),
            TaskValidationContext(),
        )

        self.assertEqual(
            [issue.code for issue in issues],
            ["execution.unknown_target"],
        )
        self.assertEqual(issues[0].path, "execution.type")

    def test_strict_validation_compiles_graph_reference_safely(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_path = root / "flows" / "valid.toml"
            _write_strict_graph_flow(
                flow_path,
                diagram="flowchart LR\nstart route_1@--> finish\n",
            )
            runner = FlowTaskTargetRunner(
                ref_base=root,
                strict_resolver=_strict_flow_loader_resolver(root),
            )

            issues = self._run_validate(
                runner,
                self._definition(
                    execution=TaskExecutionTarget.flow("flows/valid.toml"),
                ),
                TaskValidationContext(execution_roots=(root,)),
            )

        self.assertEqual(issues, ())

    def test_strict_validation_reports_graph_reference_issues_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_path = root / "flows" / "invalid.toml"
            _write_strict_graph_flow(
                flow_path,
                diagram=(
                    "flowchart LR\nstart -->|Private customer route| finish\n"
                ),
            )
            runner = FlowTaskTargetRunner(
                ref_base=root,
                strict_resolver=_strict_flow_loader_resolver(root),
            )

            issues = self._run_validate(
                runner,
                self._definition(
                    execution=TaskExecutionTarget.flow("flows/invalid.toml"),
                ),
                TaskValidationContext(execution_roots=(root,)),
            )

        self.assertEqual(
            [issue.code for issue in issues],
            ["flow.graph.unsupported_executable_edge"],
        )
        self.assertEqual(issues[0].path, "graph.edges")
        self.assertNotIn("Private customer route", str(issues))
        self.assertNotIn("invalid.toml", str(issues))

    def test_strict_validation_reports_invalid_resolved_definition(
        self,
    ) -> None:
        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: FlowDefinition(
                name="invalid",
                version="1",
                entry_behavior=FlowEntryBehavior(node="missing"),
                output_behavior=FlowOutputBehavior(outputs={"answer": "x.y"}),
                nodes=(FlowNodeDefinition(name="missing", type="private"),),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.TEXT,
                    ),
                ),
            )
        )

        issues = self._run_validate(
            runner,
            self._definition(),
            TaskValidationContext(),
        )

        self.assertEqual(issues[0].code, "flow.unknown_node_type")
        self.assertEqual(issues[0].path, "nodes.missing.type")

    def test_strict_validation_reports_missing_pipeline_tool_resolver(
        self,
    ) -> None:
        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: _strict_pipeline_definition()
        )

        issues = self._run_validate(
            runner,
            self._definition(),
            TaskValidationContext(),
        )

        self.assertEqual(
            [(issue.code, issue.path) for issue in issues],
            [("flow.unsupported_node_type", "nodes.pipeline.type")],
        )

    def test_strict_validation_accepts_pipeline_tool_resolver(self) -> None:
        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: _strict_pipeline_definition(),
            tool_resolver=StaticToolResolver(
                (
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
                            "description": (
                                "Formatted shell composition result."
                            ),
                        },
                    ),
                )
            ),
        )

        issues = self._run_validate(
            runner,
            self._definition(),
            TaskValidationContext(),
        )

        self.assertEqual(issues, ())

    def test_strict_validation_rejects_invalid_resolver_result_safely(
        self,
    ) -> None:
        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: cast(Any, "private flow")
        )

        issues = self._run_validate(
            runner,
            self._definition(),
            TaskValidationContext(),
        )

        self.assertEqual(
            [issue.code for issue in issues],
            ["execution.unsupported_flow"],
        )
        self.assertEqual(issues[0].path, "execution.ref")
        self.assertNotIn("private flow", str(issues))

    def test_flow_target_loading_keeps_async_boundaries(self) -> None:
        module_source = Path(flow_target_module.__file__).read_text(
            encoding="utf-8"
        )

        self.assertEqual(
            _task_flow_loading_async_violations(module_source),
            (),
        )

    def test_task_flow_loading_integrations_keep_async_boundaries(
        self,
    ) -> None:
        sources = {
            "task cli validation helper": getsource(
                task_cmds._validate_task_flow_reference
            ),
            "task cli strict resolver": getsource(
                task_cmds._task_strict_flow_resolver
            ),
            "server validate route": getsource(
                flow_router_module.validate_flow
            ),
            "server compile route": getsource(flow_router_module.compile_flow),
            "server graph inspect route": getsource(
                flow_router_module.inspect_graph
            ),
            "server run route": getsource(flow_router_module.run_flow),
            "server resume route": getsource(flow_router_module.resume_run),
        }

        for label, source in sources.items():
            with self.subTest(label=label):
                self.assertEqual(
                    _task_flow_loading_async_violations(source),
                    (),
                )

    def test_flow_target_loading_audit_rejects_sync_bridges(self) -> None:
        violations = _task_flow_loading_async_violations("""
from asyncio import run, to_thread

def load(path):
    path.read_text()
    loop.run_until_complete(load_async())
    asyncio.run(load_async())
    return to_thread(path.write_text, "private")
""")

        self.assertEqual(
            violations,
            (
                "line 2 imports asyncio.run",
                "line 2 imports asyncio.to_thread",
                "line 5 calls read_text",
                "line 6 calls run_until_complete",
                "line 7 calls asyncio.run",
                "line 8 calls to_thread",
            ),
        )

    def test_compatibility_report_marks_scalar_flow_compatible(self) -> None:
        report = validate_flow_task_compatibility(
            self._definition(),
            TaskValidationContext(),
        )

        self.assertTrue(report.compatible)
        self.assertEqual(report.issues, ())

    def test_compatibility_report_marks_noop_scalar_flow_compatible(
        self,
    ) -> None:
        report = validate_flow_task_compatibility(
            self._definition(observability=TaskObservabilityPolicy.noop()),
            TaskValidationContext(),
        )

        self.assertTrue(report.compatible)
        self.assertEqual(report.issues, ())

    def _run_validate(
        self,
        runner: FlowTaskTargetRunner,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        return asyncio_run(runner.validate_definition(definition, context))

    def _definition(
        self,
        *,
        input_contract: TaskInputContract | None = None,
        output_contract: TaskOutputContract | None = None,
        execution: TaskExecutionTarget | None = None,
        observability: TaskObservabilityPolicy | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="flow-task", version="1"),
            input=input_contract or TaskInputContract.string(),
            output=output_contract or TaskOutputContract.text(),
            execution=execution
            or TaskExecutionTarget.flow("flows/report.toml"),
            observability=observability or TaskObservabilityPolicy(),
        )


class FlowTaskTargetRunnerExecutionTest(IsolatedAsyncioTestCase):
    async def test_task_flow_stream_listener_preserves_custom_event_type(
        self,
    ) -> None:
        captured: list[Event] = []
        listener = flow_target_module._task_flow_stream_listener(
            self._context(event_listener=captured.append)
        )

        assert listener is not None
        result = listener(
            CanonicalStreamItem(
                stream_session_id="flow-session",
                run_id="run-1",
                turn_id="turn-1",
                sequence=3,
                kind=StreamItemKind.FLOW_EVENT,
                channel=StreamChannel.FLOW,
                correlation=StreamItemCorrelation(
                    flow_run_id="flow-run-1",
                    node_id="custom-node",
                ),
                data={"detail": "kept"},
                metadata={
                    "event_type": "flow.custom_event",
                    "started": 1,
                    "finished": 2.5,
                    "elapsed": "not-a-float",
                },
            )
        )
        if result is not None:
            await result

        self.assertEqual(len(captured), 1)
        event = captured[0]
        self.assertEqual(event.type, "flow.custom_event")
        self.assertEqual(event.payload, {"detail": "kept"})
        self.assertEqual(event.started, 1.0)
        self.assertEqual(event.finished, 2.5)
        self.assertIsNone(event.elapsed)
        observability = event.observability_payload
        assert observability is not None
        self.assertEqual(observability.data["kind"], "flow.event")
        self.assertEqual(observability.data["channel"], "flow")
        self.assertEqual(
            observability.data["correlation"],
            {"flow_run_id": "flow-run-1", "node_id": "custom-node"},
        )

    def test_task_scoped_registry_accepts_tool_resolver(self) -> None:
        registry = task_flow_node_registry(
            self._context(),
            tool_resolver=EmptyToolResolver(),
        )

        self.assertTrue(registry.supports("tool"))
        self.assertTrue(registry.supports_tool_resolution("tool"))

    def test_task_scoped_registry_loads_file_convert_toml_shape(self) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))
        context = self._context(file_converters={"pdf_image": converter})

        result = asyncio_run(
            FlowDefinitionLoader(
                registry=task_flow_node_registry(context)
            ).loads_result(
                """
            [flow]
            name = "render"
            entrypoint = "render_pages"
            output_node = "render_pages"

            [nodes.render_pages]
            type = "file_convert"
            input = "pdf"
            output = "files"

            [nodes.render_pages.config]
            converter = "pdf_image"
            format = "png"
            dpi = 144
            pages = "1..2"
            max_pages = 2
            max_pixels_per_page = 12000000
            max_total_pixels = 24000000
            """
            )
        )

        self.assertTrue(result.ok)
        assert result.definition is not None
        node = result.definition.node_map["render_pages"]
        self.assertEqual(node.config["converter"], "pdf_image")
        self.assertIsNotNone(result.flow)

    def test_strict_file_convert_node_preflights_task_contract(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))
        context = self._context(
            artifact_store=FailingArtifactStore(),
            task_store=InMemoryTaskStore(),
            file_converters={"pdf_image": converter},
        )
        definition = FlowDefinition(
            name="render",
            version="1",
            inputs=(
                FlowInputDefinition(
                    name="documents",
                    type=FlowInputType.FILE_ARRAY,
                    mime_types=("application/pdf",),
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="pages",
                    type=FlowOutputType.FILE_ARRAY,
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="render"),
            output_behavior=FlowOutputBehavior(
                outputs={"pages": "render.files"},
            ),
            nodes=(
                FlowNodeDefinition(
                    name="render",
                    type="pdf_to_images",
                    mappings=(
                        FlowInputMapping(
                            target="files",
                            kind=FlowMappingKind.FILE_ARRAY,
                            source="input.documents",
                        ),
                    ),
                ),
            ),
        )

        result = validate_flow_definition(
            definition,
            task_flow_node_registry(context),
        )

        self.assertTrue(result.ok, result.diagnostics)
        self.assertEqual(converter.calls, [])

    def test_strict_file_convert_node_rejects_missing_backing_safely(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))
        definition = FlowDefinition(
            name="render",
            version="1",
            inputs=(
                FlowInputDefinition(
                    name="documents",
                    type=FlowInputType.FILE_ARRAY,
                    mime_types=("application/pdf",),
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="pages",
                    type=FlowOutputType.FILE_ARRAY,
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="render"),
            output_behavior=FlowOutputBehavior(
                outputs={"pages": "render.files"},
            ),
            nodes=(
                FlowNodeDefinition(
                    name="render",
                    type="file_convert",
                    config={"converter": "pdf_image"},
                    mappings=(
                        FlowInputMapping(
                            target="files",
                            kind=FlowMappingKind.FILE_ARRAY,
                            source="input.documents",
                        ),
                    ),
                ),
            ),
        )

        result = validate_flow_definition(
            definition,
            task_flow_node_registry(
                self._context(file_converters={"pdf_image": converter})
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.missing_artifact_store", "flow.missing_task_store"],
        )
        self.assertNotIn("documents", str(result.public_diagnostics))
        self.assertEqual(converter.calls, [])

    def test_strict_file_convert_node_rejects_mime_mismatch_safely(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))
        context = self._context(
            artifact_store=FailingArtifactStore(),
            task_store=InMemoryTaskStore(),
            file_converters={"pdf_image": converter},
        )
        definition = FlowDefinition(
            name="render",
            version="1",
            inputs=(
                FlowInputDefinition(
                    name="private_upload",
                    type=FlowInputType.FILE_ARRAY,
                    mime_types=("image/jpeg",),
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="pages",
                    type=FlowOutputType.FILE_ARRAY,
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="render"),
            output_behavior=FlowOutputBehavior(
                outputs={"pages": "render.files"},
            ),
            nodes=(
                FlowNodeDefinition(
                    name="render",
                    type="pdf_to_images",
                    mappings=(
                        FlowInputMapping(
                            target="files",
                            kind=FlowMappingKind.FILE_ARRAY,
                            source="input.private_upload",
                        ),
                    ),
                ),
            ),
        )

        result = validate_flow_definition(
            definition,
            task_flow_node_registry(context),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.incompatible_file_mime",
        )
        self.assertNotIn("private_upload", str(result.public_diagnostics))
        self.assertEqual(converter.calls, [])

    def test_file_conversion_mime_private_helper_covers_safe_skips(
        self,
    ) -> None:
        class EmptySourceCapability:
            source_mime_types: tuple[str, ...] = ()

        class EmptySourceConverter:
            capability = EmptySourceCapability()

        definition = FlowDefinition(
            name="render",
            version="1",
            inputs=(
                FlowInputDefinition(
                    name="documents",
                    type=FlowInputType.FILE_ARRAY,
                    mime_types=("application/pdf",),
                ),
                FlowInputDefinition(
                    name="unknown_mime",
                    type=FlowInputType.FILE_ARRAY,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="pages",
                    type=FlowOutputType.FILE_ARRAY,
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="render"),
            output_behavior=FlowOutputBehavior(
                outputs={"pages": "render.files"},
            ),
            nodes=(FlowNodeDefinition(name="render", type="pdf_to_images"),),
        )
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))

        self.assertEqual(
            flow_target_module._validate_file_conversion_mime(
                definition,
                FlowNodeDefinition(
                    name="render",
                    type="pdf_to_images",
                    mappings=(
                        FlowInputMapping(
                            target="files",
                            kind=FlowMappingKind.FILE_ARRAY,
                            source="input.documents",
                        ),
                    ),
                ),
                cast(Any, EmptySourceConverter()),
            ),
            (),
        )
        for node in (
            FlowNodeDefinition(
                name="render",
                type="pdf_to_images",
                mappings=(
                    FlowInputMapping(
                        target="input",
                        source="input.documents",
                    ),
                ),
            ),
            FlowNodeDefinition(
                name="render",
                type="pdf_to_images",
                mappings=(
                    FlowInputMapping(
                        target="files",
                        kind=FlowMappingKind.FILE_ARRAY,
                        source="input.unknown_mime",
                    ),
                ),
            ),
            FlowNodeDefinition(
                name="render",
                type="pdf_to_images",
                mappings=(
                    FlowInputMapping(
                        target="files",
                        kind=FlowMappingKind.ARRAY,
                        items=("input.documents", "__task_files__.files"),
                    ),
                ),
            ),
            FlowNodeDefinition(
                name="render",
                type="pdf_to_images",
                mappings=(
                    FlowInputMapping(
                        target="files",
                        kind=FlowMappingKind.OBJECT,
                        fields={"value": "input.documents"},
                    ),
                ),
            ),
        ):
            with self.subTest(node=node):
                validate_mime = (
                    flow_target_module._validate_file_conversion_mime
                )
                diagnostics = validate_mime(
                    definition,
                    node,
                    cast(
                        Any,
                        flow_target_module._FlowFileConverter(
                            converter,
                            limits=flow_target_module._FlowConversionLimits(),
                        ),
                    ),
                )

                self.assertEqual(diagnostics, ())

    def test_strict_agent_node_preflights_task_contract(self) -> None:
        context = self._context(task_store=InMemoryTaskStore())
        definition = FlowDefinition(
            name="review",
            version="1",
            inputs=(
                FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            ),
            outputs=(
                FlowOutputDefinition(name="answer", type=FlowOutputType.JSON),
            ),
            entry_behavior=FlowEntryBehavior(node="review"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "review.result"},
            ),
            nodes=(
                FlowNodeDefinition(
                    name="review",
                    type="agent",
                    ref="agents/review.toml",
                    mappings=(
                        FlowInputMapping(
                            target="input",
                            source="input.payload",
                        ),
                    ),
                ),
            ),
        )

        result = validate_flow_definition(
            definition,
            task_flow_node_registry(
                context,
                agent_runner=CapturingTaskTargetRunner(),
            ),
        )

        self.assertTrue(result.ok, result.diagnostics)

    def test_strict_agent_node_accepts_file_selector_mapping(self) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))
        context = self._context(
            artifact_store=FailingArtifactStore(),
            task_store=InMemoryTaskStore(),
            file_converters={"pdf_image": converter},
        )
        definition = FlowDefinition(
            name="review-pages",
            version="1",
            inputs=(
                FlowInputDefinition(
                    name="documents",
                    type=FlowInputType.FILE_ARRAY,
                    mime_types=("application/pdf",),
                ),
            ),
            outputs=(
                FlowOutputDefinition(name="answer", type=FlowOutputType.JSON),
            ),
            entry_behavior=FlowEntryBehavior(node="render"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "review.result"},
            ),
            nodes=(
                FlowNodeDefinition(
                    name="render",
                    type="file_convert",
                    config={"converter": "pdf_image"},
                    mappings=(
                        FlowInputMapping(
                            target="files",
                            kind=FlowMappingKind.FILE_ARRAY,
                            source="input.documents",
                        ),
                    ),
                ),
                FlowNodeDefinition(
                    name="review",
                    type="agent",
                    ref="agents/review.toml",
                    config={
                        "files_input": "render.files",
                        "file_policy": "replace",
                    },
                    mappings=(
                        FlowInputMapping(
                            target="input",
                            source="input.documents",
                        ),
                        FlowInputMapping(
                            target="render",
                            kind=FlowMappingKind.OBJECT,
                            fields={"files": "render.files"},
                        ),
                    ),
                ),
            ),
            edges=(FlowEdgeDefinition(source="render", target="review"),),
        )

        result = validate_flow_definition(
            definition,
            task_flow_node_registry(
                context,
                agent_runner=CapturingTaskTargetRunner(),
            ),
        )

        self.assertTrue(result.ok, result.diagnostics)

    def test_strict_agent_node_rejects_ref_escape_safely(self) -> None:
        context = self._context(task_store=InMemoryTaskStore())
        definition = FlowDefinition(
            name="review",
            version="1",
            inputs=(
                FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            ),
            outputs=(
                FlowOutputDefinition(name="answer", type=FlowOutputType.JSON),
            ),
            entry_behavior=FlowEntryBehavior(node="review"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "review.result"},
            ),
            nodes=(
                FlowNodeDefinition(
                    name="review",
                    type="agent",
                    ref="../private-review.toml",
                    mappings=(
                        FlowInputMapping(
                            target="input",
                            source="input.payload",
                        ),
                    ),
                ),
            ),
        )

        result = validate_flow_definition(
            definition,
            task_flow_node_registry(
                context,
                agent_runner=CapturingTaskTargetRunner(),
            ),
        )

        self.assertFalse(result.ok)
        self.assertIn(
            "flow.path_escape",
            [diagnostic.code for diagnostic in result.diagnostics],
        )
        self.assertNotIn("private-review", str(result.public_diagnostics))

    def test_strict_agent_node_rejects_missing_task_store_safely(self) -> None:
        definition = FlowDefinition(
            name="review",
            version="1",
            inputs=(
                FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            ),
            outputs=(
                FlowOutputDefinition(name="answer", type=FlowOutputType.JSON),
            ),
            entry_behavior=FlowEntryBehavior(node="review"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "review.result"},
            ),
            nodes=(
                FlowNodeDefinition(
                    name="review",
                    type="agent",
                    ref="agents/private-review.toml",
                    mappings=(
                        FlowInputMapping(
                            target="input",
                            source="input.payload",
                        ),
                    ),
                ),
            ),
        )

        result = validate_flow_definition(
            definition,
            task_flow_node_registry(
                self._context(),
                agent_runner=CapturingTaskTargetRunner(),
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.missing_task_store")
        self.assertNotIn("private-review", str(result.public_diagnostics))

    def test_file_convert_direct_build_rejects_preflight_failures(
        self,
    ) -> None:
        cases = (
            (
                "missing",
                self._context(),
                FlowNodeDefinition(
                    name="render",
                    type="file_convert",
                    config={"converter": "missing"},
                ),
                "flow.converter_unsupported",
            ),
            (
                "dependency",
                self._context(
                    file_converters={
                        "pdf_image": RecordingPdfPageConverter(
                            (_page_result(1, b"page"),),
                            dependency_gates=(
                                TaskFeature.PDF_IMAGE_CONVERSION,
                            ),
                        )
                    },
                ),
                FlowNodeDefinition(name="render", type="pdf_to_images"),
                "dependency.task_pdf_images_missing",
            ),
            (
                "options",
                self._context(
                    file_converters={
                        "pdf_image": RecordingPdfPageConverter(
                            (_page_result(1, b"page"),),
                        )
                    },
                ),
                FlowNodeDefinition(
                    name="render",
                    type="file_convert",
                    config={"converter": "pdf_image", "format": "gif"},
                ),
                "flow.invalid_node",
            ),
        )

        for name, context, definition, expected_code in cases:
            with self.subTest(name=name):
                if name == "dependency":
                    patcher = patch(
                        "avalan.task.targets.flow.feature_available",
                        return_value=False,
                    )
                else:
                    patcher = patch(
                        "avalan.task.targets.flow.feature_available",
                        return_value=True,
                    )
                with patcher:
                    with self.assertRaises(
                        FlowNodeConfigurationError
                    ) as error:
                        task_flow_node_registry(context).build(definition)

                self.assertEqual(error.exception.code, expected_code)

    def test_task_scoped_registry_rejects_bad_file_convert_config(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))
        context = self._context(file_converters={"pdf_image": converter})
        cases = (
            (
                "pdf_to_images",
                "raw = 'private.pdf'",
                "unsupported option",
                "flow.invalid_node",
            ),
            (
                "pdf_to_images",
                "pages = '../private.pdf'",
                "bad pages",
                "flow.invalid_node",
            ),
            (
                "pdf_to_images",
                "max_pages = 0",
                "bad max pages",
                "flow.invalid_node",
            ),
            (
                "pdf_to_images",
                "converter = 'text'",
                "wrong pdf alias",
                "flow.invalid_node",
            ),
            (
                "file_convert",
                "",
                "missing converter",
                "flow.invalid_node",
            ),
            (
                "file_convert",
                "converter = 'missing'",
                "unknown converter",
                "flow.converter_unsupported",
            ),
            (
                "file_convert",
                "converter = 'pdf_image'\nformat = 'gif'",
                "bad converter option",
                "flow.invalid_node",
            ),
        )

        for node_type, config, name, expected_code in cases:
            with self.subTest(name=name):
                result = asyncio_run(
                    FlowDefinitionLoader(
                        registry=task_flow_node_registry(context)
                    ).loads_result(
                        f"""
                    [flow]
                    name = "render"
                    entrypoint = "render_pages"
                    output_node = "render_pages"

                    [nodes.render_pages]
                    type = "{node_type}"

                    [nodes.render_pages.config]
                    {config}
                    """
                    )
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.issues[0].code, expected_code)
                self.assertNotIn("private.pdf", str(result.issues))

    def test_file_convert_private_helpers_cover_option_shapes(self) -> None:
        self.assertEqual(
            flow_target_module._page_range_option(2),  # type: ignore[attr-defined]
            {"start": 2, "end": 2},
        )
        self.assertEqual(
            flow_target_module._page_range_option(  # type: ignore[attr-defined]
                {"start": 1, "end": 2}
            ),
            {"start": 1, "end": 2},
        )
        self.assertEqual(
            flow_target_module._page_range_option("3"),  # type: ignore[attr-defined]
            {"start": 3, "end": 3},
        )
        self.assertEqual(
            flow_target_module._page_range_option("2.."),  # type: ignore[attr-defined]
            {"start": 2},
        )
        self.assertIsNone(
            flow_target_module._positive_config_int(  # type: ignore[attr-defined]
                {},
                "max_pages",
            )
        )
        self.assertEqual(
            flow_target_module._lower_limit(None, 3),  # type: ignore[attr-defined]
            3,
        )
        self.assertEqual(
            flow_target_module._lower_limit(4, None),  # type: ignore[attr-defined]
            4,
        )

        for value in ([], "", "1-2", "0", "0..", "3..2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    flow_target_module._page_range_option(value)  # type: ignore[attr-defined]

    async def test_file_converter_wrapper_rejects_non_page_converter(
        self,
    ) -> None:
        converter = flow_target_module._FlowFileConverter(  # type: ignore[attr-defined]
            SingleOutputConverter(),
            limits=flow_target_module._FlowConversionLimits(),  # type: ignore[attr-defined]
        )

        converter.validate_options({"format": "png"})
        result = await converter.convert(
            b"%PDF-private source",
            source_media_type="application/pdf",
        )
        with self.assertRaises(TaskFileConversionError) as error:
            await converter.convert_pages(
                b"%PDF-private source",
                source_media_type="application/pdf",
            )

        self.assertEqual(result.media_type, "image/png")
        self.assertNotIn("%PDF-private", str(error.exception))
        with self.assertRaises(TaskFileConversionError):
            flow_target_module._validate_file_conversion_preflight(  # type: ignore[attr-defined]
                converter,
                TaskFileConversionRequest(name="single"),
            )

    async def test_run_fails_closed_without_resolver(self) -> None:
        runner = FlowTaskTargetRunner()

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(self._context(input_value="private prompt"))

        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["execution.unsupported_flow"],
        )
        self.assertNotIn("private prompt", str(error.exception))

    async def test_run_executes_resolved_flow(self) -> None:
        flow = Flow()
        flow.add_node(
            Node("A", func=lambda inputs: inputs[FLOW_TASK_INPUT_KEY] + "!")
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(self._context(input_value="ready"))

        self.assertEqual(result, "ready!")

    async def test_run_executes_strict_definition_and_records_state(
        self,
    ) -> None:
        flow_store = InMemoryFlowStateStore()
        events: list[Event] = []
        execution_count = 0
        real_execute_flow_plan = flow_target_module.execute_flow_plan

        async def counting_execute_flow_plan(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            nonlocal execution_count
            execution_count += 1
            return await real_execute_flow_plan(*args, **kwargs)

        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: _strict_echo_definition(),
            flow_state_store=flow_store,
        )

        with patch.object(
            flow_target_module,
            "execute_flow_plan",
            counting_execute_flow_plan,
        ):
            result = await runner.run(
                self._context(
                    input_value="ready",
                    event_listener=events.append,
                )
            )
            second = await runner.run(
                self._context(
                    input_value="ready",
                    event_listener=events.append,
                )
            )
        record = await flow_store.get_flow_execution("run-1")

        self.assertEqual(result, "ready")
        self.assertEqual(second, "ready")
        self.assertEqual(execution_count, 1)
        self.assertEqual(record.revision, 2)
        self.assertEqual(dict(record.selected_outputs), {"answer": "ready"})
        self.assertIn("strict_flow", record.metadata)
        self.assertEqual(record.trace.nodes[0].state, FlowNodeState.SUCCEEDED)
        self.assertEqual(
            record.node_attempts[0].state, FlowNodeState.SUCCEEDED
        )
        self.assertEqual(record.node_attempts[0].attempt, 1)
        self.assertEqual(record.artifact_refs, ())
        self.assertEqual(
            [
                cast(Mapping[str, object], event.payload)["status"]
                for event in events
                if event.type
                in (
                    EventType.FLOW_MANAGER_CALL_BEFORE,
                    EventType.FLOW_MANAGER_CALL_AFTER,
                )
            ],
            ["started", "succeeded", "started", "succeeded"],
        )
        self.assertEqual(
            [
                event.type
                for event in events
                if event.type == EventType.FLOW_NODE_COMPLETED
            ],
            [EventType.FLOW_NODE_COMPLETED],
        )

    async def test_run_executes_incomplete_strict_state(
        self,
    ) -> None:
        compile_result = await compile_flow_definition(
            _strict_echo_definition()
        )
        assert compile_result.plan is not None
        flow_store = InMemoryFlowStateStore()
        await flow_store.create_flow_execution(
            "run-1",
            trace=FlowExecutionTrace.from_plan(compile_result.plan),
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                compile_result.plan
            ),
        )
        execution_count = 0
        real_execute_flow_plan = flow_target_module.execute_flow_plan

        async def counting_execute_flow_plan(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            nonlocal execution_count
            execution_count += 1
            return await real_execute_flow_plan(*args, **kwargs)

        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: compile_result.plan,
            flow_state_store=flow_store,
        )

        with patch.object(
            flow_target_module,
            "execute_flow_plan",
            counting_execute_flow_plan,
        ):
            result = await runner.run(self._context(input_value="ready"))
        record = await flow_store.get_flow_execution("run-1")

        self.assertEqual(result, "ready")
        self.assertEqual(execution_count, 1)
        self.assertEqual(record.revision, 2)
        self.assertEqual(dict(record.selected_outputs), {"answer": "ready"})

    async def test_run_resumes_partial_strict_state_without_rerunning_node(
        self,
    ) -> None:
        compile_result = await compile_flow_definition(
            _strict_two_step_definition()
        )
        assert compile_result.plan is not None
        flow_store = InMemoryFlowStateStore()
        trace = FlowExecutionTrace.from_plan(
            compile_result.plan
        ).with_node_state("start", FlowNodeState.SUCCEEDED, attempts=1)
        await flow_store.create_flow_execution(
            "run-1",
            trace=trace,
            node_outputs={"start": {"value": "stored seed"}},
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                compile_result.plan
            ),
        )
        calls: list[str] = []
        real_runner_factory = flow_target_module.flow_node_registry_runner

        def counting_runner_factory(registry: Any) -> Any:
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

        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: compile_result.plan,
            flow_state_store=flow_store,
        )

        with patch.object(
            flow_target_module,
            "flow_node_registry_runner",
            counting_runner_factory,
        ):
            result = await runner.run(self._context(input_value="fresh"))
        record = await flow_store.get_flow_execution("run-1")

        self.assertEqual(result, "stored seed")
        self.assertEqual(calls, ["answer"])
        self.assertEqual(record.revision, 2)
        self.assertEqual(
            dict(record.node_outputs),
            {
                "start": {"value": "stored seed"},
                "answer": {"value": "stored seed"},
            },
        )
        self.assertEqual(
            dict(record.selected_outputs), {"answer": "stored seed"}
        )

    async def test_run_records_strict_human_review_pause(
        self,
    ) -> None:
        plan = _strict_human_review_plan()
        flow_store = InMemoryFlowStateStore()
        calls: list[str] = []
        real_runner_factory = flow_target_module.flow_node_registry_runner

        def counting_runner_factory(registry: Any) -> Any:
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

        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=flow_store,
        )

        with patch.object(
            flow_target_module,
            "flow_node_registry_runner",
            counting_runner_factory,
        ):
            with self.assertRaises(TaskValidationError) as error:
                await runner.run(self._context(input_value="safe"))
        record = await flow_store.get_flow_execution("run-1")

        self.assertEqual(
            error.exception.issues[0].code, "flow.execution.paused"
        )
        self.assertEqual(calls, ["start"])
        self.assertEqual(record.revision, 2)
        self.assertEqual(set(record.pause_tokens), {"review"})
        self.assertTrue(record.pause_tokens["review"])
        self.assertEqual(
            {node.node: node.state for node in record.trace.nodes},
            {
                "start": FlowNodeState.SUCCEEDED,
                "review": FlowNodeState.PAUSED,
                "finish": FlowNodeState.PENDING,
                "rejected": FlowNodeState.PENDING,
            },
        )
        self.assertEqual(
            dict(record.node_outputs),
            {"start": {"value": "safe"}},
        )
        self.assertEqual(dict(record.selected_outputs), {})
        audit = cast(
            Mapping[str, object],
            record.metadata["human_review_audit"],
        )
        review = cast(Mapping[str, object], audit["review"])
        request = cast(Mapping[str, object], review["request"])
        self.assertEqual(review["state"], "paused")
        self.assertEqual(request["node"], "review")
        self.assertEqual(
            request["allowed_decisions"],
            ("approved", "rejected"),
        )
        self.assertEqual(request["timeout_seconds"], 300)
        self.assertEqual(
            request["audit_metadata"],
            {"risk": "medium", "queue": "ops"},
        )

    async def test_run_rejects_strict_human_review_without_state_store_safely(
        self,
    ) -> None:
        plan = _strict_human_review_plan()
        events: list[Event] = []
        calls = 0

        async def fail_execute_flow_plan(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            raise AssertionError("flow runtime should not execute")

        runner = FlowTaskTargetRunner(strict_resolver=lambda _: plan)

        with patch.object(
            flow_target_module,
            "execute_flow_plan",
            fail_execute_flow_plan,
        ):
            with self.assertRaises(TaskValidationError) as error:
                await runner.run(
                    self._context(
                        input_value="private prompt",
                        event_listener=events.append,
                    )
                )

        self.assertEqual(
            error.exception.issues[0].code,
            "flow.unsupported_human_review_direct_mode",
        )
        self.assertEqual(
            error.exception.issues[0].category,
            TaskValidationCategory.DEPENDENCY,
        )
        self.assertEqual(calls, 0)
        self.assertEqual(events, [])
        self.assertNotIn("private prompt", str(error.exception))

    async def test_run_rejects_nested_human_review_without_state_store_safely(
        self,
    ) -> None:
        plan = _strict_nested_human_review_plan()
        events: list[Event] = []
        calls = 0

        async def fail_execute_flow_plan(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            raise AssertionError("flow runtime should not execute")

        runner = FlowTaskTargetRunner(strict_resolver=lambda _: plan)

        with patch.object(
            flow_target_module,
            "execute_flow_plan",
            fail_execute_flow_plan,
        ):
            with self.assertRaises(TaskValidationError) as error:
                await runner.run(
                    self._context(
                        input_value="private nested prompt",
                        event_listener=events.append,
                    )
                )

        self.assertEqual(
            error.exception.issues[0].code,
            "flow.unsupported_human_review_direct_mode",
        )
        self.assertEqual(calls, 0)
        self.assertEqual(events, [])
        self.assertNotIn("private nested prompt", str(error.exception))

    async def test_run_resumes_strict_human_review_from_metadata(
        self,
    ) -> None:
        plan = _strict_human_review_plan()
        flow_store = InMemoryFlowStateStore()
        first_runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=flow_store,
        )
        with self.assertRaises(TaskValidationError):
            await first_runner.run(self._context(input_value="safe"))
        calls: list[str] = []
        real_runner_factory = flow_target_module.flow_node_registry_runner

        def counting_runner_factory(registry: Any) -> Any:
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

        resume_runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=flow_store,
        )

        with patch.object(
            flow_target_module,
            "flow_node_registry_runner",
            counting_runner_factory,
        ):
            metadata_key = (
                flow_target_module.FLOW_RESUME_DECISIONS_METADATA_KEY
            )
            result = await resume_runner.run(
                self._context(
                    input_value="new",
                    metadata={
                        metadata_key: {
                            "review": {
                                "decision": "approved",
                                "comment": "safe",
                            },
                        },
                    },
                )
            )
        record = await flow_store.get_flow_execution("run-1")

        self.assertEqual(result, "approved")
        self.assertEqual(calls, ["finish"])
        self.assertEqual(record.revision, 3)
        self.assertEqual(dict(record.pause_tokens), {})
        self.assertEqual(dict(record.selected_outputs), {"answer": "approved"})
        review_output = cast(
            Mapping[str, object],
            cast(Mapping[str, object], record.node_outputs["review"])[
                "result"
            ],
        )
        self.assertEqual(review_output["decision"], "approved")
        audit = cast(
            Mapping[str, object],
            record.metadata["human_review_audit"],
        )
        review = cast(Mapping[str, object], audit["review"])
        self.assertEqual(review["state"], "resumed")
        self.assertEqual(review["decision"], "approved")
        self.assertNotIn("comment", str(review))

    async def test_run_resumes_pgsql_strict_human_review_after_restart(
        self,
    ) -> None:
        database = FakePgsqlTaskDatabase()
        clock = SequenceClock()
        task_store = PgsqlTaskStore(
            database,
            clock=clock,
            id_factory=SequenceIds(),
        )
        flow_store = PgsqlFlowStateStore(database, clock=clock)
        definition = self._context_definition()
        await task_store.register_definition(
            definition,
            definition_hash="flow-pgsql-review",
        )
        run = await task_store.create_run(
            TaskExecutionRequest(definition_id="flow-pgsql-review")
        )
        attempt = await task_store.create_attempt(run.run_id)
        plan = _strict_human_review_plan()
        first_runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=flow_store,
        )
        with self.assertRaises(TaskValidationError):
            await first_runner.run(
                self._context(
                    definition=definition,
                    execution=attempt.context,
                    input_value="safe",
                )
            )
        paused_record = await flow_store.get_flow_execution(run.run_id)
        calls: list[str] = []
        real_runner_factory = flow_target_module.flow_node_registry_runner

        def counting_runner_factory(registry: Any) -> Any:
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

        resumed_store = PgsqlFlowStateStore(database, clock=clock)
        resume_runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=resumed_store,
        )
        metadata_key = flow_target_module.FLOW_RESUME_DECISIONS_METADATA_KEY

        with patch.object(
            flow_target_module,
            "flow_node_registry_runner",
            counting_runner_factory,
        ):
            result = await resume_runner.run(
                self._context(
                    definition=definition,
                    execution=attempt.context,
                    input_value="fresh",
                    metadata={
                        metadata_key: {
                            "review": {
                                "decision": "approved",
                                "comment": "safe",
                            },
                        },
                    },
                )
            )
        record = await resumed_store.get_flow_execution(run.run_id)

        self.assertEqual(paused_record.revision, 2)
        self.assertEqual(result, "approved")
        self.assertEqual(calls, ["finish"])
        self.assertEqual(record.revision, 3)
        self.assertEqual(dict(record.pause_tokens), {})
        self.assertEqual(dict(record.selected_outputs), {"answer": "approved"})
        audit = cast(
            Mapping[str, object],
            record.metadata["human_review_audit"],
        )
        review = cast(Mapping[str, object], audit["review"])
        self.assertEqual(review["state"], "resumed")
        self.assertEqual(review["decision"], "approved")
        self.assertNotIn("fresh", str(record.as_snapshot()))

    async def test_run_rejects_pgsql_human_review_resume_after_restart_safely(
        self,
    ) -> None:
        database = FakePgsqlTaskDatabase()
        clock = SequenceClock()
        task_store = PgsqlTaskStore(
            database,
            clock=clock,
            id_factory=SequenceIds(),
        )
        flow_store = PgsqlFlowStateStore(database, clock=clock)
        definition = self._context_definition()
        await task_store.register_definition(
            definition,
            definition_hash="flow-pgsql-review-invalid",
        )
        run = await task_store.create_run(
            TaskExecutionRequest(definition_id="flow-pgsql-review-invalid")
        )
        attempt = await task_store.create_attempt(run.run_id)
        plan = _strict_human_review_plan()
        first_runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=flow_store,
        )
        with self.assertRaises(TaskValidationError):
            await first_runner.run(
                self._context(
                    definition=definition,
                    execution=attempt.context,
                    input_value="safe",
                )
            )
        paused_record = await flow_store.get_flow_execution(run.run_id)
        calls: list[str] = []
        real_runner_factory = flow_target_module.flow_node_registry_runner

        def counting_runner_factory(registry: Any) -> Any:
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

        resumed_store = PgsqlFlowStateStore(database, clock=clock)
        resume_runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=resumed_store,
        )
        metadata_key = flow_target_module.FLOW_RESUME_DECISIONS_METADATA_KEY

        with patch.object(
            flow_target_module,
            "flow_node_registry_runner",
            counting_runner_factory,
        ):
            with self.assertRaises(TaskValidationError) as error:
                await resume_runner.run(
                    self._context(
                        definition=definition,
                        execution=attempt.context,
                        input_value="fresh private prompt",
                        metadata={
                            metadata_key: {
                                "review": {
                                    "decision": "delayed",
                                    "comment": "private-token",
                                },
                            },
                        },
                    )
                )
        record = await resumed_store.get_flow_execution(run.run_id)

        self.assertEqual(
            error.exception.issues[0].code,
            "flow.execution.unknown_resume_decision",
        )
        self.assertEqual(calls, [])
        self.assertEqual(record.revision, paused_record.revision)
        self.assertEqual(
            dict(record.pause_tokens), dict(paused_record.pause_tokens)
        )
        self.assertEqual(
            {node.node: node.state for node in record.trace.nodes}["review"],
            FlowNodeState.PAUSED,
        )
        self.assertNotIn("private-token", str(error.exception))
        self.assertNotIn("fresh private prompt", str(error.exception))
        self.assertNotIn("private-token", str(record.as_snapshot()))
        self.assertNotIn("fresh private prompt", str(record.as_snapshot()))

    async def test_run_rejects_invalid_strict_human_review_resume_cases_safely(
        self,
    ) -> None:
        cases = (
            (
                "unknown_decision",
                "review",
                {
                    "decision": "delayed",
                    "comment": "private-token",
                },
                "flow.execution.unknown_resume_decision",
            ),
            (
                "schema",
                "review",
                {
                    "decision": "approved",
                    "comment": 7,
                },
                "flow.execution.invalid_resume_payload",
            ),
            (
                "unknown_node",
                "missing",
                {
                    "decision": "approved",
                    "comment": "private-token",
                },
                "flow.execution.unknown_resume_node",
            ),
            (
                "non_review_node",
                "start",
                {
                    "decision": "approved",
                    "comment": "private-token",
                },
                "flow.execution.invalid_resume_node",
            ),
        )

        for name, node_name, payload, code in cases:
            with self.subTest(name=name):
                plan = _strict_human_review_plan()
                flow_store = InMemoryFlowStateStore()
                first_runner = FlowTaskTargetRunner(
                    strict_resolver=lambda _: plan,
                    flow_state_store=flow_store,
                )
                with self.assertRaises(TaskValidationError):
                    await first_runner.run(self._context(input_value="safe"))

                resume_runner = FlowTaskTargetRunner(
                    strict_resolver=lambda _: plan,
                    flow_state_store=flow_store,
                )
                with self.assertRaises(TaskValidationError) as error:
                    await resume_runner.run(
                        self._context(
                            input_value="fresh",
                            metadata={
                                (
                                    flow_target_module.FLOW_RESUME_DECISIONS_METADATA_KEY
                                ): {
                                    node_name: payload,
                                },
                            },
                        )
                    )
                record = await flow_store.get_flow_execution("run-1")

                self.assertEqual(error.exception.issues[0].code, code)
                self.assertEqual(record.revision, 2)
                self.assertEqual(set(record.pause_tokens), {"review"})
                self.assertEqual(
                    {node.node: node.state for node in record.trace.nodes}[
                        "review"
                    ],
                    FlowNodeState.PAUSED,
                )
                self.assertNotIn("private-token", str(error.exception))
                self.assertNotIn("fresh", str(record.as_snapshot()))
                self.assertNotIn("private-token", str(record.as_snapshot()))

    async def test_run_rejects_invalid_strict_human_review_resume_metadata(
        self,
    ) -> None:
        plan = _strict_human_review_plan()
        flow_store = InMemoryFlowStateStore()
        await flow_store.create_flow_execution(
            "run-1",
            trace=FlowExecutionTrace.from_plan(plan),
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                plan
            ),
        )
        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=flow_store,
        )
        cases = (
            "private-token",
            {"review": "private-token"},
        )

        for metadata_value in cases:
            with self.subTest(metadata_value=metadata_value):
                with self.assertRaises(TaskValidationError) as error:
                    await runner.run(
                        self._context(
                            input_value="private prompt",
                            metadata={
                                (
                                    flow_target_module.FLOW_RESUME_DECISIONS_METADATA_KEY
                                ): metadata_value,
                            },
                        )
                    )
                record = await flow_store.get_flow_execution("run-1")

                self.assertEqual(
                    error.exception.issues[0].code,
                    "flow.execution.invalid_resume_payload",
                )
                self.assertEqual(record.revision, 1)
                self.assertNotIn("private-token", str(error.exception))
                self.assertNotIn("private prompt", str(error.exception))

    async def test_run_routes_strict_human_review_resume_decisions(
        self,
    ) -> None:
        decisions = {
            "rejected": "rejected_sink",
            "needs-correction": "correction_sink",
            "expired": "expired_sink",
            "escalated": "escalated_sink",
        }

        for decision, target in decisions.items():
            with self.subTest(decision=decision):
                plan = _strict_human_review_matrix_plan()
                flow_store = InMemoryFlowStateStore()
                first_runner = FlowTaskTargetRunner(
                    strict_resolver=lambda _: plan,
                    flow_state_store=flow_store,
                )
                with self.assertRaises(TaskValidationError):
                    await first_runner.run(
                        self._context(input_value="private prompt")
                    )

                calls: list[str] = []
                real_runner_factory = (
                    flow_target_module.flow_node_registry_runner
                )

                def counting_runner_factory(registry: Any) -> Any:
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

                metadata_key = (
                    flow_target_module.FLOW_RESUME_DECISIONS_METADATA_KEY
                )
                resume_runner = FlowTaskTargetRunner(
                    strict_resolver=lambda _: plan,
                    flow_state_store=flow_store,
                )
                with patch.object(
                    flow_target_module,
                    "flow_node_registry_runner",
                    counting_runner_factory,
                ):
                    result = await resume_runner.run(
                        self._context(
                            input_value="fresh private prompt",
                            metadata={
                                metadata_key: {
                                    "review": {
                                        "decision": decision,
                                        "comment": "private-token",
                                    },
                                },
                            },
                        )
                    )
                record = await flow_store.get_flow_execution("run-1")

                self.assertEqual(result, decision)
                self.assertEqual(calls, [target])
                self.assertEqual(dict(record.pause_tokens), {})
                self.assertEqual(
                    dict(record.selected_outputs), {"answer": decision}
                )
                self.assertEqual(
                    {node.node: node.state for node in record.trace.nodes}[
                        target
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
                self.assertNotIn("private-token", str(review))
                self.assertNotIn(
                    "fresh private prompt",
                    str(record.as_snapshot()),
                )

    async def test_run_keeps_strict_human_review_paused_when_resume_cancels(
        self,
    ) -> None:
        plan = _strict_human_review_matrix_plan()
        flow_store = InMemoryFlowStateStore()
        first_runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=flow_store,
        )
        with self.assertRaises(TaskValidationError):
            await first_runner.run(self._context(input_value="safe"))
        paused_record = await flow_store.get_flow_execution("run-1")

        async def cancel() -> None:
            raise CancelledError()

        resume_runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: plan,
            flow_state_store=flow_store,
        )
        with self.assertRaises(CancelledError):
            await resume_runner.run(
                self._context(
                    input_value="fresh private prompt",
                    cancellation_checker=cancel,
                    metadata={
                        (
                            flow_target_module.FLOW_RESUME_DECISIONS_METADATA_KEY
                        ): {
                            "review": {
                                "decision": "approved",
                                "comment": "private-token",
                            },
                        },
                    },
                )
            )
        record = await flow_store.get_flow_execution("run-1")

        self.assertEqual(record.revision, paused_record.revision)
        self.assertEqual(
            dict(record.pause_tokens), dict(paused_record.pause_tokens)
        )
        self.assertEqual(
            {node.node: node.state for node in record.trace.nodes}["review"],
            FlowNodeState.PAUSED,
        )
        self.assertNotIn("fresh private prompt", str(record.as_snapshot()))
        self.assertNotIn("private-token", str(record.as_snapshot()))

    async def test_run_reruns_partial_strict_state_without_node_outputs(
        self,
    ) -> None:
        compile_result = await compile_flow_definition(
            _strict_two_step_definition()
        )
        assert compile_result.plan is not None
        flow_store = InMemoryFlowStateStore()
        trace = FlowExecutionTrace.from_plan(
            compile_result.plan
        ).with_node_state("start", FlowNodeState.SUCCEEDED, attempts=1)
        await flow_store.create_flow_execution(
            "run-1",
            trace=trace,
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                compile_result.plan
            ),
        )
        calls: list[str] = []
        real_runner_factory = flow_target_module.flow_node_registry_runner

        def counting_runner_factory(registry: Any) -> Any:
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

        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: compile_result.plan,
            flow_state_store=flow_store,
        )

        with patch.object(
            flow_target_module,
            "flow_node_registry_runner",
            counting_runner_factory,
        ):
            result = await runner.run(self._context(input_value="fresh"))
        record = await flow_store.get_flow_execution("run-1")

        self.assertEqual(result, "fresh")
        self.assertEqual(calls, ["start", "answer"])
        self.assertEqual(
            dict(record.node_outputs),
            {
                "start": {"value": "fresh"},
                "answer": {"value": "fresh"},
            },
        )

    async def test_run_rejects_stale_completed_state_for_changed_node(
        self,
    ) -> None:
        old_compile_result = await compile_flow_definition(
            _strict_constant_definition("old result")
        )
        new_compile_result = await compile_flow_definition(
            _strict_constant_definition("new result")
        )
        assert old_compile_result.plan is not None
        assert new_compile_result.plan is not None
        flow_store = InMemoryFlowStateStore()
        await flow_store.create_flow_execution(
            "run-1",
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="answer",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                ),
            ),
            selected_outputs={"answer": "old result"},
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                old_compile_result.plan
            ),
        )

        async def fail_on_execute(
            *args: Any,
            **kwargs: Any,
        ) -> object:
            _ = args, kwargs
            raise AssertionError("execution should not start")

        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: new_compile_result.plan,
            flow_state_store=flow_store,
        )

        with patch.object(
            flow_target_module,
            "execute_flow_plan",
            fail_on_execute,
        ):
            with self.assertRaises(TaskValidationError) as error:
                await runner.run(self._context(input_value="private prompt"))
        record = await flow_store.get_flow_execution("run-1")

        self.assertEqual(
            error.exception.issues[0].code,
            "flow.execution_state_mismatch",
        )
        self.assertEqual(record.revision, 1)
        self.assertEqual(
            dict(record.selected_outputs), {"answer": "old result"}
        )
        self.assertNotIn("private prompt", str(error.exception))

    async def test_run_rejects_stale_completed_state_for_changed_route(
        self,
    ) -> None:
        old_compile_result = await compile_flow_definition(
            _strict_routed_definition("answer")
        )
        new_compile_result = await compile_flow_definition(
            _strict_routed_definition("alternate")
        )
        assert old_compile_result.plan is not None
        assert new_compile_result.plan is not None
        flow_store = InMemoryFlowStateStore()
        await flow_store.create_flow_execution(
            "run-1",
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="start",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                    FlowNodeTrace(
                        node="answer",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                    FlowNodeTrace(
                        node="alternate",
                        state=FlowNodeState.SKIPPED,
                    ),
                ),
            ),
            selected_outputs={"answer": "stored result"},
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                old_compile_result.plan
            ),
        )

        async def fail_on_execute(
            *args: Any,
            **kwargs: Any,
        ) -> object:
            _ = args, kwargs
            raise AssertionError("execution should not start")

        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: new_compile_result.plan,
            flow_state_store=flow_store,
        )

        with patch.object(
            flow_target_module,
            "execute_flow_plan",
            fail_on_execute,
        ):
            with self.assertRaises(TaskValidationError) as error:
                await runner.run(self._context(input_value="private prompt"))
        record = await flow_store.get_flow_execution("run-1")

        self.assertEqual(
            error.exception.issues[0].code,
            "flow.execution_state_mismatch",
        )
        self.assertEqual(record.revision, 1)
        self.assertEqual(
            dict(record.selected_outputs),
            {"answer": "stored result"},
        )
        self.assertNotIn("private prompt", str(error.exception))

    async def test_run_resumes_pgsql_strict_state_after_json_round_trip(
        self,
    ) -> None:
        database = FakePgsqlTaskDatabase()
        clock = SequenceClock()
        task_store = PgsqlTaskStore(
            database,
            clock=clock,
            id_factory=SequenceIds(),
        )
        flow_store = PgsqlFlowStateStore(database, clock=clock)
        definition = self._context_definition()
        await task_store.register_definition(
            definition,
            definition_hash="flow-pgsql-resume",
        )
        run = await task_store.create_run(
            TaskExecutionRequest(definition_id="flow-pgsql-resume")
        )
        attempt = await task_store.create_attempt(run.run_id)
        compile_result = await compile_flow_definition(
            _strict_echo_definition()
        )
        assert compile_result.plan is not None
        await flow_store.create_flow_execution(
            run.run_id,
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="echo",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                ),
            ),
            selected_outputs={"answer": "stored answer"},
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                compile_result.plan
            ),
        )

        async def fail_on_execute(
            *args: Any,
            **kwargs: Any,
        ) -> object:
            _ = args, kwargs
            raise AssertionError("execution should not start")

        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: compile_result.plan,
            flow_state_store=flow_store,
        )

        with patch.object(
            flow_target_module,
            "execute_flow_plan",
            fail_on_execute,
        ):
            result = await runner.run(
                self._context(
                    definition=definition,
                    execution=attempt.context,
                    input_value="private prompt",
                )
            )
        record = await flow_store.get_flow_execution(run.run_id)

        self.assertEqual(result, "stored answer")
        self.assertEqual(record.revision, 1)
        self.assertEqual(
            dict(record.selected_outputs),
            {"answer": "stored answer"},
        )
        self.assertNotIn("private prompt", str(record.as_snapshot()))

    async def test_run_rejects_mismatched_strict_state_safely(
        self,
    ) -> None:
        compile_result = await compile_flow_definition(
            _strict_echo_definition()
        )
        assert compile_result.plan is not None
        flow_store = InMemoryFlowStateStore()
        await flow_store.create_flow_execution(
            "run-1",
            trace=FlowExecutionTrace.from_plan(compile_result.plan),
            metadata={"strict_flow": {"name": "other"}},
        )
        events: list[Event] = []

        async def fail_on_execute(
            *args: Any,
            **kwargs: Any,
        ) -> object:
            _ = args, kwargs
            raise AssertionError("execution should not start")

        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: compile_result.plan,
            flow_state_store=flow_store,
        )

        with patch.object(
            flow_target_module,
            "execute_flow_plan",
            fail_on_execute,
        ):
            with self.assertRaises(TaskValidationError) as error:
                await runner.run(
                    self._context(
                        input_value="private prompt",
                        event_listener=events.append,
                    )
                )

        record = await flow_store.get_flow_execution("run-1")
        self.assertEqual(
            error.exception.issues[0].code,
            "flow.execution_state_mismatch",
        )
        self.assertEqual(record.revision, 1)
        self.assertEqual(
            [
                cast(Mapping[str, object], event.payload)["status"]
                for event in events
                if event.type
                in (
                    EventType.FLOW_MANAGER_CALL_BEFORE,
                    EventType.FLOW_MANAGER_CALL_AFTER,
                )
            ],
            ["started", "failed"],
        )
        self.assertNotIn("private prompt", str(error.exception))
        self.assertNotIn("private prompt", str(events))

    async def test_strict_state_helpers_cover_unfinished_records(
        self,
    ) -> None:
        compile_result = await compile_flow_definition(
            _strict_echo_definition()
        )
        assert compile_result.plan is not None
        flow_store = InMemoryFlowStateStore()
        runner = FlowTaskTargetRunner(flow_state_store=flow_store)
        diagnostic = FlowDiagnostic(
            code="flow.safe",
            category=FlowDiagnosticCategory.TASK_DURABILITY,
            severity=FlowDiagnosticSeverity.ERROR,
            message="Flow failed.",
            path="execution",
        )

        await runner._record_strict_flow_state(  # type: ignore[attr-defined]
            self._context(input_value="ready"),
            plan=compile_result.plan,
            trace=FlowExecutionTrace.from_plan(compile_result.plan),
            outputs={},
            node_outputs={},
            diagnostics=(diagnostic,),
        )
        diagnostic_record = await flow_store.get_flow_execution("run-1")
        diagnostic_resume_record = await flow_store.create_flow_execution(
            "run-diagnostic-resume",
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="echo",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                ),
            ),
            node_outputs={"echo": {"value": "ready"}},
            diagnostics=(diagnostic,),
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                compile_result.plan
            ),
        )
        missing_node_record = await flow_store.create_flow_execution(
            "run-missing-node",
            trace=FlowExecutionTrace(nodes=()),
            selected_outputs={"answer": "ready"},
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                compile_result.plan
            ),
        )
        missing_output_record = await flow_store.create_flow_execution(
            "run-missing-output",
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="echo",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                ),
            ),
            node_outputs={"other": {"value": "ready"}},
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                compile_result.plan
            ),
        )
        old_record = await flow_store.create_flow_execution(
            "run-old",
            trace=FlowExecutionTrace.from_plan(compile_result.plan),
            selected_outputs={"answer": "ready"},
        )
        routed_compile_result = await compile_flow_definition(
            _strict_routed_definition("answer")
        )
        assert routed_compile_result.plan is not None
        missing_edge_record = await flow_store.create_flow_execution(
            "run-missing-edge",
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="start",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                    FlowNodeTrace(
                        node="answer",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                    FlowNodeTrace(
                        node="alternate",
                        state=FlowNodeState.SKIPPED,
                    ),
                ),
            ),
            selected_outputs={"answer": "ready"},
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                routed_compile_result.plan
            ),
        )
        failed_edge_record = await flow_store.create_flow_execution(
            "run-failed-edge",
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="start",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                    FlowNodeTrace(
                        node="answer",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                    FlowNodeTrace(
                        node="alternate",
                        state=FlowNodeState.SKIPPED,
                    ),
                ),
                edges=(
                    FlowEdgeTrace(
                        index=0,
                        source="start",
                        target="answer",
                        state=FlowEdgeState.FAILED,
                    ),
                ),
            ),
            selected_outputs={"answer": "ready"},
            metadata=flow_target_module._strict_flow_record_metadata(  # type: ignore[attr-defined]
                routed_compile_result.plan
            ),
        )

        self.assertEqual(diagnostic_record.revision, 1)
        self.assertEqual(diagnostic_record.diagnostics, (diagnostic,))
        self.assertIs(
            flow_target_module._strict_resumed_output(  # type: ignore[attr-defined]
                compile_result.plan,
                diagnostic_record,
            ),
            flow_target_module._NO_STRICT_RESUME,  # type: ignore[attr-defined]
        )
        self.assertIs(
            flow_target_module._strict_resumed_output(  # type: ignore[attr-defined]
                compile_result.plan,
                missing_node_record,
            ),
            flow_target_module._NO_STRICT_RESUME,  # type: ignore[attr-defined]
        )
        self.assertIs(
            flow_target_module._strict_resumed_output(  # type: ignore[attr-defined]
                compile_result.plan,
                old_record,
            ),
            flow_target_module._NO_STRICT_RESUME,  # type: ignore[attr-defined]
        )
        self.assertIs(
            flow_target_module._strict_resumed_output(  # type: ignore[attr-defined]
                routed_compile_result.plan,
                missing_edge_record,
            ),
            flow_target_module._NO_STRICT_RESUME,  # type: ignore[attr-defined]
        )
        self.assertIs(
            flow_target_module._strict_resumed_output(  # type: ignore[attr-defined]
                routed_compile_result.plan,
                failed_edge_record,
            ),
            flow_target_module._NO_STRICT_RESUME,  # type: ignore[attr-defined]
        )
        self.assertIsNone(
            flow_target_module._strict_resume_node_outputs(  # type: ignore[attr-defined]
                compile_result.plan,
                None,
            )
        )
        self.assertIsNone(
            flow_target_module._strict_resume_node_outputs(  # type: ignore[attr-defined]
                compile_result.plan,
                diagnostic_resume_record,
            )
        )
        self.assertIsNone(
            flow_target_module._strict_resume_node_outputs(  # type: ignore[attr-defined]
                compile_result.plan,
                old_record,
            )
        )
        self.assertIsNone(
            flow_target_module._strict_resume_node_outputs(  # type: ignore[attr-defined]
                compile_result.plan,
                missing_output_record,
            )
        )
        self.assertFalse(
            flow_target_module._strict_flow_record_mismatches_plan(  # type: ignore[attr-defined]
                compile_result.plan,
                old_record,
            )
        )
        review_plan = _strict_human_review_plan()
        record_metadata = getattr(
            flow_target_module,
            "_strict_flow_record_metadata",
        )
        failed_review_metadata = record_metadata(
            review_plan,
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="review",
                        state=FlowNodeState.FAILED,
                        attempts=1,
                    ),
                ),
            ),
            resume_decisions={"review": {"decision": "approved"}},
        )
        blank_decision_metadata = record_metadata(
            review_plan,
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="review",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                ),
            ),
            resume_decisions={"review": {"decision": ""}},
        )
        missing_node_metadata = record_metadata(
            review_plan,
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="missing",
                        state=FlowNodeState.SUCCEEDED,
                        attempts=1,
                    ),
                ),
            ),
            resume_decisions={"missing": {"decision": "approved"}},
        )
        non_review_pause_metadata = record_metadata(
            review_plan,
            trace=FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(
                        node="start",
                        state=FlowNodeState.PAUSED,
                        attempts=1,
                    ),
                ),
            ),
            pause_tokens={"start": "private-token"},
        )
        invalid_decision_node = FlowNodePlan(
            name="review",
            type="human_review",
            kind=FlowNodeKind.HUMAN_REVIEW,
            config={"allowed_decisions": "approved"},
        )
        request_metadata = getattr(
            flow_target_module,
            "_strict_human_review_request_metadata",
        )
        audit_metadata = getattr(
            flow_target_module,
            "_strict_human_review_audit_from_record",
        )
        invalid_request = request_metadata(invalid_decision_node)
        object.__setattr__(
            old_record,
            "metadata",
            {
                "human_review_audit": {
                    "bad": "private audit",
                    "good": {"state": "paused"},
                }
            },
        )
        audit_from_record = audit_metadata(old_record)
        self.assertNotIn("human_review_audit", failed_review_metadata)
        self.assertNotIn("human_review_audit", blank_decision_metadata)
        self.assertNotIn("human_review_audit", missing_node_metadata)
        self.assertNotIn("human_review_audit", non_review_pause_metadata)
        self.assertNotIn("timeout_seconds", invalid_request)
        self.assertNotIn("audit_metadata", invalid_request)
        self.assertEqual(audit_from_record, {"good": {"state": "paused"}})
        self.assertEqual(
            flow_target_module._strict_human_review_decisions(  # type: ignore[attr-defined]
                invalid_decision_node
            ),
            (),
        )
        object.__setattr__(old_record, "node_outputs", {"": {"value": "x"}})
        self.assertEqual(
            flow_target_module._strict_record_node_outputs(  # type: ignore[attr-defined]
                old_record,
            ),
            {},
        )
        object.__setattr__(
            old_record,
            "node_outputs",
            {"start": "bad", "ok": {"value": "x"}},
        )
        self.assertEqual(
            flow_target_module._strict_record_node_outputs(  # type: ignore[attr-defined]
                old_record,
            ),
            {"ok": {"value": "x"}},
        )

    def test_strict_signature_includes_optional_plan_metadata(self) -> None:
        child_condition = FlowConditionPlan(
            operator=FlowConditionOperator.EXISTS,
            selector=parse_flow_selector("input.prompt"),
        )
        condition = FlowConditionPlan(
            operator=FlowConditionOperator.ALL,
            selector=parse_flow_selector("input.prompt"),
            value={"allowed": [FlowOutputType.TEXT]},
            value_selector=parse_flow_selector("input.prompt"),
            values=([FlowOutputType.TEXT],),
            value_type=FlowConditionValueType.STRING,
            conditions=(child_condition,),
            condition=FlowConditionPlan(
                operator=FlowConditionOperator.NOT,
                condition=child_condition,
            ),
        )
        node = FlowNodePlan(
            name="complex",
            type="pass-through",
            kind=FlowNodeKind.PASS_THROUGH,
            input_contracts=(
                FlowNodeContract(
                    name="prompt",
                    type=FlowInputType.STRING,
                    metadata={
                        "capabilities": [FlowNodeCapability.DIRECT_ASYNC]
                    },
                ),
            ),
            mappings=(
                FlowMappingPlan(
                    target="value",
                    kind=FlowMappingKind.SELECT,
                    source=parse_flow_selector("input.prompt"),
                ),
            ),
            join=FlowJoinPlan(
                type=FlowJoinPolicyType.QUORUM,
                quorum=1,
                optional_inputs=("fallback",),
            ),
            retry=FlowRetryPlan(
                max_attempts=2,
                backoff=FlowRetryBackoffStrategy.CONSTANT,
                initial_delay_seconds=1,
                max_delay_seconds=2,
                retryable_categories=("transient",),
                non_retryable_categories=("validation",),
                exhausted_route="fallback",
            ),
            timeout=FlowTimeoutPlan(per_attempt_seconds=3),
            loop=FlowLoopPlan(
                max_iterations=2,
                max_elapsed_seconds=5,
                continue_condition=condition,
                exit_condition=child_condition,
                output_selector=parse_flow_selector("complex.value"),
                limit_route="fallback",
            ),
            config={"enum_values": [FlowOutputType.TEXT]},
        )

        signature = flow_target_module._flow_node_signature(node)  # type: ignore[attr-defined]
        join_signature = cast(Mapping[str, object], signature["join"])
        retry_signature = cast(Mapping[str, object], signature["retry"])
        loop_signature = cast(Mapping[str, object], signature["loop"])

        self.assertEqual(join_signature["type"], "quorum")
        self.assertEqual(retry_signature["max_attempts"], 2)
        self.assertEqual(
            signature["timeout"],
            {"per_attempt_seconds": 3},
        )
        self.assertEqual(loop_signature["limit_route"], "fallback")
        self.assertEqual(
            signature["config"],
            {"enum_values": ("text",)},
        )

    async def test_run_executes_compiled_strict_plan_for_direct_and_queue(
        self,
    ) -> None:
        compile_result = await compile_flow_definition(
            _strict_echo_definition()
        )
        assert compile_result.plan is not None

        async def resolve(_: TaskTargetContext) -> FlowExecutionPlan:
            await sleep(0)
            return compile_result.plan

        runner = FlowTaskTargetRunner(strict_resolver=resolve)

        direct = await runner.run(self._context(input_value="ready"))
        queued = await runner.run(
            self._context(
                definition=self._context_definition(
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value={
                    "format": STORED_ENVELOPE_MARKER,
                    "privacy": STORED_MARKER,
                    "value": "ready",
                },
            )
        )

        self.assertEqual(direct, queued)
        self.assertEqual(queued, "ready")

    async def test_run_rejects_invalid_strict_input_before_events(
        self,
    ) -> None:
        events: list[Event] = []
        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: _strict_echo_definition()
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    definition=self._context_definition(
                        input_contract=TaskInputContract.integer(),
                    ),
                    input_value="private prompt",
                    event_listener=events.append,
                )
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_type")
        self.assertEqual(events, [])
        self.assertNotIn("private prompt", str(error.exception))

    async def test_run_maps_strict_runtime_diagnostics_safely(self) -> None:
        compile_result = await compile_flow_definition(
            _strict_echo_definition()
        )
        assert compile_result.plan is not None
        plan = replace(
            compile_result.plan,
            output_selectors={
                "answer": parse_flow_selector("echo.missing"),
            },
        )
        events: list[Event] = []
        runner = FlowTaskTargetRunner(strict_resolver=lambda _: plan)

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    input_value="private prompt",
                    event_listener=events.append,
                )
            )

        self.assertEqual(
            error.exception.issues[0].code,
            "flow.execution.missing_output",
        )
        self.assertEqual(
            [
                cast(Mapping[str, object], event.payload)["status"]
                for event in events
                if event.type
                in (
                    EventType.FLOW_MANAGER_CALL_BEFORE,
                    EventType.FLOW_MANAGER_CALL_AFTER,
                )
            ],
            ["started", "failed"],
        )
        self.assertNotIn("private prompt", str(error.exception))
        self.assertNotIn("private prompt", str(events))

    async def test_run_rejects_invalid_strict_output_safely(self) -> None:
        events: list[Event] = []
        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: _strict_echo_definition()
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    definition=self._context_definition(
                        output_contract=TaskOutputContract.object(
                            {"type": "object"}
                        ),
                    ),
                    input_value="private prompt",
                    event_listener=events.append,
                )
            )

        self.assertEqual(error.exception.issues[0].code, "output.invalid_type")
        self.assertEqual(
            [
                cast(Mapping[str, object], event.payload)["status"]
                for event in events
                if event.type
                in (
                    EventType.FLOW_MANAGER_CALL_BEFORE,
                    EventType.FLOW_MANAGER_CALL_AFTER,
                )
            ],
            ["started", "failed"],
        )
        self.assertNotIn("private prompt", str(error.exception))
        self.assertNotIn("private prompt", str(events))

    async def test_run_rejects_non_strict_definition_safely(self) -> None:
        flow_store = InMemoryFlowStateStore()
        events: list[Event] = []
        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: FlowDefinition(
                name="loose",
                entrypoint="echo",
                output_node="echo",
                nodes=(FlowNodeDefinition(name="echo", type="pass-through"),),
            ),
            flow_state_store=flow_store,
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    input_value="private prompt",
                    event_listener=events.append,
                )
            )

        self.assertEqual(
            error.exception.issues[0].code,
            "flow.execution.plan_requires_strict_definition",
        )
        self.assertEqual(events, [])
        with self.assertRaises(TaskStoreNotFoundError):
            await flow_store.get_flow_execution("run-1")
        self.assertNotIn("private prompt", str(error.exception))

    async def test_run_rejects_invalid_strict_resolver_result_safely(
        self,
    ) -> None:
        runner = FlowTaskTargetRunner(
            strict_resolver=lambda _: cast(Any, "private flow")
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(self._context(input_value="private prompt"))

        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["execution.ref"],
        )
        self.assertNotIn("private prompt", str(error.exception))
        self.assertNotIn("private flow", str(error.exception))

    def test_strict_helpers_cover_safe_runtime_snapshots(self) -> None:
        compile_result = asyncio_run(
            compile_flow_definition(_strict_file_binding_definition())
        )
        assert compile_result.plan is not None
        file = TaskInputFile(
            logical_path="artifact:source-1",
            artifact_ref=TaskArtifactRef(
                artifact_id="source-1",
                store="memory",
                storage_key="source-1",
                media_type="application/pdf",
                size_bytes=10,
                sha256="0" * 64,
            ),
            media_type="application/pdf",
            size_bytes=10,
        )
        fallback = {
            "document": "fallback",
            "documents": ["fallback"],
            "extra": "ready",
        }

        with_files = flow_target_module._strict_flow_input_binding(  # type: ignore[attr-defined]
            compile_result.plan,
            fallback,
            files=(file,),
        )
        without_files = flow_target_module._strict_flow_input_binding(  # type: ignore[attr-defined]
            compile_result.plan,
            {"document": "fallback", "documents": ["fallback"]},
            files=(),
        )

        self.assertIs(with_files["document"], file)
        self.assertEqual(with_files["documents"], [file])
        self.assertEqual(with_files["extra"], "ready")
        self.assertEqual(without_files["document"], "fallback")
        self.assertEqual(without_files["documents"], ["fallback"])
        self.assertIsNone(without_files["extra"])
        self.assertEqual(
            flow_target_module._strict_task_output(  # type: ignore[attr-defined]
                compile_result.plan,
                {"other": "value"},
            ),
            {"other": "value"},
        )
        two_output_plan = replace(
            compile_result.plan,
            outputs=(
                compile_result.plan.outputs[0],
                FlowOutputDefinition(
                    name="summary",
                    type=FlowOutputType.TEXT,
                ),
            ),
        )
        self.assertEqual(
            flow_target_module._strict_task_output(  # type: ignore[attr-defined]
                two_output_plan,
                {"answer": "ready", "summary": "done"},
            ),
            {"answer": "ready", "summary": "done"},
        )
        self.assertEqual(
            flow_target_module._flow_files_from_value(  # type: ignore[attr-defined]
                {"items": [file]}
            ),
            ({"items": [file]},),
        )

        issues = flow_target_module._flow_diagnostics_to_issues(())  # type: ignore[attr-defined]
        self.assertEqual(issues[0].code, "execution.unsupported_flow")
        category_cases = (
            (
                FlowDiagnosticCategory.PRIVACY,
                TaskValidationCategory.PRIVACY,
            ),
            (
                FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
                TaskValidationCategory.STRUCTURE,
            ),
            (
                FlowDiagnosticCategory.TASK_DURABILITY,
                TaskValidationCategory.DEPENDENCY,
            ),
        )
        for category, expected in category_cases:
            with self.subTest(category=category):
                issue = flow_target_module._flow_diagnostics_to_issues(  # type: ignore[attr-defined]
                    (
                        FlowDiagnostic(
                            code="flow.safe",
                            category=category,
                            severity=FlowDiagnosticSeverity.ERROR,
                            message="Flow failed.",
                            source_span=FlowSourceSpan(
                                start_line=1,
                                start_column=1,
                            ),
                        ),
                    )
                )[
                    0
                ]
                self.assertEqual(issue.category, expected)
                self.assertEqual(issue.path, "execution")
                self.assertEqual(issue.hint, "Inspect the flow diagnostics.")

        trace = FlowExecutionTrace(
            nodes=(
                FlowNodeTrace(
                    node="retry",
                    state=FlowNodeState.SUCCEEDED,
                    attempts=2,
                    duration_ms=4,
                ),
            )
        )
        attempts = flow_target_module._node_attempt_records(trace)  # type: ignore[attr-defined]
        self.assertEqual(
            [attempt.state for attempt in attempts],
            [FlowNodeState.FAILED, FlowNodeState.SUCCEEDED],
        )

        loop_plan = _loop_counter_plan()
        counters = flow_target_module._loop_counters(loop_plan, trace)  # type: ignore[attr-defined]
        self.assertEqual(dict(counters), {"retry": 2})

        artifact = TaskArtifactRef(
            artifact_id="output-1",
            store="memory",
            storage_key="output-1",
            media_type="text/plain",
            size_bytes=4,
            sha256="1" * 64,
            metadata={"private": "metadata"},
        )
        snapshot = flow_target_module._flow_snapshot_value(  # type: ignore[attr-defined]
            {"file": file, "artifact": artifact, "items": [object()]}
        )
        refs = flow_target_module._artifact_refs(  # type: ignore[attr-defined]
            {"file": file, "artifacts": [artifact, artifact]}
        )
        no_ref_file = TaskInputFile(
            logical_path="private.pdf",
            artifact_ref=None,
            media_type="application/pdf",
            size_bytes=10,
        )

        self.assertIn("artifact", snapshot)
        self.assertEqual(
            [ref["artifact_id"] for ref in refs],
            ["source-1", "output-1"],
        )
        self.assertNotIn("private", str(refs))
        self.assertEqual(
            flow_target_module._artifact_refs(no_ref_file),  # type: ignore[attr-defined]
            (),
        )

    async def test_run_rejects_flow_agent_node_validation_issues(
        self,
    ) -> None:
        agent_runner = CapturingTaskTargetRunner(
            issues=(
                TaskValidationIssue(
                    code="execution.unsupported_flow",
                    path="nodes.review.ref",
                    message="Flow agent node is invalid.",
                    hint="Use a valid agent node reference.",
                    category=TaskValidationCategory.UNSUPPORTED,
                ),
            )
        )
        runner = FlowTaskTargetRunner(
            flow_resolver=lambda context: _agent_node_flow(
                context,
                agent_runner=agent_runner,
            )
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(self._context(input_value="private prompt"))

        self.assertEqual(
            error.exception.issues[0].path,
            "nodes.review.ref",
        )
        self.assertEqual(agent_runner.contexts, [])
        self.assertNotIn("private prompt", str(error.exception))

    async def test_run_agent_node_uses_flow_input_fallbacks(self) -> None:
        cases = (
            ("task_input", _agent_node_flow, "ready"),
            ("single_incoming", _single_incoming_agent_flow, "ready"),
            (
                "multi_incoming",
                _multi_incoming_agent_flow,
                {"left": "left", "right": "right"},
            ),
        )

        for name, factory, expected_input in cases:
            with self.subTest(name=name):
                agent_runner = CapturingTaskTargetRunner()
                runner = FlowTaskTargetRunner(
                    flow_resolver=lambda context: factory(
                        context,
                        agent_runner=agent_runner,
                    )
                )

                result = await runner.run(self._context(input_value="ready"))

                self.assertEqual(result, "agent output")
                self.assertEqual(
                    agent_runner.contexts[0].input_value,
                    expected_input,
                )

    async def test_run_agent_node_replaces_files_from_selector(self) -> None:
        parent_file = TaskInputFile(
            logical_path="artifact:source-private",
            media_type="application/pdf",
        )
        generated_files = (
            TaskInputFile(
                logical_path="artifact:page-1",
                media_type="image/png",
                size_bytes=11,
            ),
            TaskInputFile(
                logical_path="artifact:page-2",
                media_type="image/png",
                size_bytes=12,
            ),
        )
        agent_runner = CapturingTaskTargetRunner()
        runner = FlowTaskTargetRunner(
            flow_resolver=lambda context: _file_selecting_agent_flow(
                context,
                agent_runner=agent_runner,
                generated_files=generated_files,
                file_policy="replace",
            )
        )

        result = await runner.run(
            self._context(input_value="ready", files=(parent_file,))
        )

        self.assertEqual(result, "agent output")
        self.assertEqual(agent_runner.contexts[0].files, generated_files)
        self.assertNotIn("source-private", str(agent_runner.contexts[0].files))

    async def test_run_agent_node_appends_files_from_selector(self) -> None:
        parent_file = TaskInputFile(
            logical_path="artifact:source",
            media_type="application/pdf",
        )
        generated_file = TaskInputFile(
            logical_path="artifact:page-1",
            media_type="image/png",
        )
        agent_runner = CapturingTaskTargetRunner()
        runner = FlowTaskTargetRunner(
            flow_resolver=lambda context: _file_selecting_agent_flow(
                context,
                agent_runner=agent_runner,
                generated_files=(generated_file,),
                file_policy="append",
            )
        )

        result = await runner.run(
            self._context(input_value="ready", files=(parent_file,))
        )

        self.assertEqual(result, "agent output")
        self.assertEqual(
            agent_runner.contexts[0].files[0],
            parent_file,
        )
        appended_file = agent_runner.contexts[0].files[1]
        self.assertEqual(
            appended_file.logical_path, generated_file.logical_path
        )
        self.assertEqual(appended_file.media_type, generated_file.media_type)
        self.assertEqual(appended_file.metadata["file_policy"], "append")
        self.assertEqual(generated_file.metadata, {})

    async def test_run_agent_node_rejects_bad_file_selector_outputs(
        self,
    ) -> None:
        cases: tuple[tuple[str, object, str], ...] = (
            ("missing_key", {"other": []}, "has no output"),
            ("scalar", {"files": "private.pdf"}, "not a file array"),
            ("empty", {"files": []}, "empty"),
            ("bad_item", {"files": ["private.pdf"]}, "item is invalid"),
        )

        for name, render_output, expected in cases:
            with self.subTest(name=name):
                agent_runner = CapturingTaskTargetRunner()

                def resolve(context: TaskTargetContext) -> Flow:
                    registry = task_flow_node_registry(
                        context,
                        agent_runner=agent_runner,
                    )
                    flow = Flow()
                    flow.add_node(Node("start", func=lambda _: "ready"))
                    flow.add_node(Node("render", func=lambda _: render_output))
                    flow.add_node(
                        registry.build(
                            FlowNodeDefinition(
                                name="review",
                                type="agent",
                                ref="agents/review.toml",
                                config={
                                    "files_input": "render.files",
                                    "file_policy": "replace",
                                },
                            )
                        )
                    )
                    flow.add_connection("start", "render")
                    flow.add_connection("render", "review")
                    return flow

                runner = FlowTaskTargetRunner(flow_resolver=resolve)

                with self.assertRaises(TaskValidationError) as error:
                    await runner.run(self._context(input_value="ready"))

                self.assertIn(expected, error.exception.issues[0].message)
                self.assertEqual(agent_runner.contexts, [])
                self.assertNotIn("private.pdf", str(error.exception))

    def test_task_scoped_registry_rejects_bad_agent_file_config(
        self,
    ) -> None:
        context = self._context()
        cases = (
            (
                "unsupported",
                "raw = 'private.pdf'",
                "flow.invalid_node",
                "nodes.review.config.raw",
            ),
            (
                "missing_policy",
                'files_input = "render.files"',
                "flow.invalid_node",
                "nodes.review.config.file_policy",
            ),
            (
                "policy_without_selector",
                'file_policy = "replace"',
                "flow.invalid_node",
                "nodes.review.config.files_input",
            ),
            (
                "bad_policy",
                'files_input = "render.files"\nfile_policy = "merge"',
                "flow.invalid_node",
                "nodes.review.config.file_policy",
            ),
            (
                "non_string_selector",
                'files_input = 42\nfile_policy = "replace"',
                "flow.invalid_type",
                "nodes.review.config.files_input",
            ),
            (
                "blank_selector",
                'files_input = ""\nfile_policy = "replace"',
                "flow.invalid_type",
                "nodes.review.config.files_input",
            ),
            (
                "malformed_selector",
                'files_input = "render"\nfile_policy = "replace"',
                "flow.invalid_output_selector",
                "nodes.review.config.files_input",
            ),
            (
                "unknown_source",
                'files_input = "missing.files"\nfile_policy = "replace"',
                "flow.bad_reference",
                "nodes.review.config.files_input",
            ),
            (
                "reserved_selector",
                (
                    'files_input = "__task_files__.files"\n'
                    'file_policy = "replace"'
                ),
                "flow.reserved_selector",
                "nodes.review.config.files_input",
            ),
            (
                "missing_edge",
                'files_input = "render.files"\nfile_policy = "replace"',
                "flow.bad_reference",
                "nodes.review.config.files_input",
            ),
        )

        for name, config, expected_code, expected_path in cases:
            with self.subTest(name=name):
                edge = (
                    ""
                    if name == "missing_edge"
                    else '[[edges]]\nsource = "render"\ntarget = "review"'
                )
                result = asyncio_run(
                    FlowDefinitionLoader(
                        registry=task_flow_node_registry(
                            context,
                            agent_runner=CapturingTaskTargetRunner(),
                        )
                    ).loads_result(
                        f"""
                    [flow]
                    name = "select"
                    entrypoint = "render"
                    output_node = "review"

                    [nodes.render]
                    type = "constant"
                    value = {{files = []}}

                    [nodes.review]
                    type = "agent"
                    ref = "agents/review.toml"

                    [nodes.review.config]
                    {config}

                    {edge}
                    """
                    )
                )

                self.assertFalse(result.ok)
                diagnostics = [
                    (issue.code, issue.path) for issue in result.issues
                ]
                self.assertIn(
                    (expected_code, expected_path),
                    diagnostics,
                )
                self.assertNotIn("private.pdf", str(result.issues))

    async def test_agent_file_private_helpers_reject_impossible_shapes(
        self,
    ) -> None:
        with self.assertRaises(FlowNodeConfigurationError):
            flow_target_module._validate_agent_files_input_selector(  # type: ignore[attr-defined]
                "review",
                "render",
            )
        with self.assertRaises(FlowNodeConfigurationError):
            flow_target_module._validate_agent_files_input_selector(  # type: ignore[attr-defined]
                "review",
                "files.items",
            )
        definition = FlowNodeDefinition(
            name="review",
            type="agent",
            ref="agents/review.toml",
        )
        with self.assertRaises(TaskValidationError) as missing:
            flow_target_module._agent_node_files(  # type: ignore[attr-defined]
                definition,
                {},
                context=self._context(),
                file_plan=flow_target_module._AgentFilePlan(  # type: ignore[attr-defined]
                    files_input="render.files",
                    file_policy="replace",
                ),
            )
        self.assertIn("unavailable", missing.exception.issues[0].message)
        with self.assertRaises(AssertionError):
            flow_target_module._agent_node_files(  # type: ignore[attr-defined]
                definition,
                {
                    "render": {
                        "files": [
                            TaskInputFile(
                                logical_path="artifact:page-1",
                                media_type="image/png",
                            )
                        ]
                    }
                },
                context=self._context(),
                file_plan=flow_target_module._AgentFilePlan(  # type: ignore[attr-defined]
                    files_input="render.files",
                    file_policy="private",
                ),
            )

    async def test_run_awaits_async_resolver(self) -> None:
        flow = Flow()
        flow.add_node(
            Node("A", func=lambda inputs: inputs[FLOW_TASK_INPUT_KEY] + 1)
        )

        async def resolver(_: TaskTargetContext) -> Flow:
            await sleep(0)
            return flow

        runner = FlowTaskTargetRunner(flow_resolver=resolver)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.integer(),
                    output_contract=TaskOutputContract.json(
                        {"type": "integer"}
                    ),
                ),
                input_value=1,
            )
        )

        self.assertEqual(result, 2)

    async def test_run_binds_object_fields_and_full_input(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "name": inputs["name"],
                    "limit": inputs[FLOW_TASK_INPUT_KEY]["limit"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.object(
                        {
                            "type": "object",
                            "required": ["name", "limit"],
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                        }
                    ),
                    output_contract=self._object_output_contract(),
                ),
                input_value={"name": "report", "limit": 3},
            )
        )

        self.assertEqual(result, {"name": "report", "limit": 3})

    async def test_run_does_not_trust_reserved_object_input_key(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "reserved": inputs[FLOW_TASK_INPUT_KEY][
                        FLOW_TASK_INPUT_KEY
                    ],
                    "limit": inputs[FLOW_TASK_INPUT_KEY]["limit"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.object(
                        {
                            "type": "object",
                            "required": [FLOW_TASK_INPUT_KEY, "limit"],
                            "additionalProperties": False,
                            "properties": {
                                FLOW_TASK_INPUT_KEY: {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                        }
                    ),
                    output_contract=self._object_output_contract(),
                ),
                input_value={
                    FLOW_TASK_INPUT_KEY: "private spoofed input",
                    "limit": 3,
                },
            )
        )

        self.assertEqual(
            result,
            {"reserved": "private spoofed input", "limit": 3},
        )

    async def test_run_isolates_object_input_from_node_mutation(self) -> None:
        def mutate(inputs: dict[str, object]) -> dict[str, object]:
            nested = cast(dict[str, object], inputs["nested"])
            items = cast(list[str], nested["items"])
            items.append("mutated")
            full_input = cast(dict[str, object], inputs[FLOW_TASK_INPUT_KEY])
            full_nested = cast(dict[str, object], full_input["nested"])
            return {
                "field": tuple(items),
                "full": tuple(cast(list[str], full_nested["items"])),
            }

        flow = Flow()
        flow.add_node(Node("A", func=mutate))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)
        input_value = {"nested": {"items": ["original"]}}

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.object(
                        {
                            "type": "object",
                            "required": ["nested"],
                            "additionalProperties": False,
                            "properties": {
                                "nested": {
                                    "type": "object",
                                    "required": ["items"],
                                    "additionalProperties": False,
                                    "properties": {
                                        "items": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        }
                    ),
                    output_contract=self._object_output_contract(),
                ),
                input_value=input_value,
            )
        )

        self.assertEqual(
            result,
            {
                "field": ("original", "mutated"),
                "full": ("original",),
            },
        )
        self.assertEqual(input_value, {"nested": {"items": ["original"]}})

    async def test_run_binds_scalar_input_value(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "full": inputs[FLOW_TASK_INPUT_KEY],
                    "value": inputs["value"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    output_contract=self._object_output_contract(),
                ),
                input_value="ready",
            )
        )

        self.assertEqual(result, {"full": "ready", "value": "ready"})

    async def test_run_binds_array_input_as_json_lists(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "full": inputs[FLOW_TASK_INPUT_KEY],
                    "items": inputs["items"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.array(
                        {
                            "type": "array",
                            "items": {"type": "array"},
                        }
                    ),
                    output_contract=self._object_output_contract(),
                ),
                input_value=[("first", "second")],
            )
        )

        self.assertEqual(
            result,
            {
                "full": [["first", "second"]],
                "items": [["first", "second"]],
            },
        )
        assert isinstance(result, Mapping)
        self.assertIsInstance(result["full"], list)
        self.assertIsInstance(result["items"], list)

    async def test_run_rejects_invalid_input_contract_safely(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: "unused private output"))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    definition=self._context_definition(
                        input_contract=TaskInputContract.integer(),
                    ),
                    input_value="private prompt",
                )
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_type")
        self.assertNotIn("private prompt", str(error.exception))
        self.assertNotIn("unused private output", str(error.exception))

    async def test_run_rejects_invalid_output_before_success_event(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda _: {
                    "status": "ready",
                    "count": "private invalid count",
                },
            )
        )
        events: list[Event] = []
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    definition=self._context_definition(
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
                        )
                    ),
                    input_value="private prompt",
                    event_listener=events.append,
                )
            )

        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["output.invalid_type"],
        )
        self.assertEqual(
            [event.type for event in events],
            [
                EventType.FLOW_MANAGER_CALL_BEFORE,
                EventType.FLOW_MANAGER_CALL_AFTER,
            ],
        )
        failed_payload = cast(Mapping[str, object], events[1].payload)
        self.assertEqual(failed_payload["status"], "failed")
        self.assertNotIn("private prompt", str(events))
        self.assertNotIn("private invalid count", str(events))
        self.assertNotIn("private invalid count", str(error.exception))

    async def test_run_unwraps_stored_queued_input(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: inputs[FLOW_TASK_INPUT_KEY] + "!",
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value={
                    "format": STORED_ENVELOPE_MARKER,
                    "privacy": STORED_MARKER,
                    "value": "ready",
                },
            )
        )

        self.assertEqual(result, "ready!")

    async def test_run_unwraps_legacy_stored_queued_input(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: inputs[FLOW_TASK_INPUT_KEY] + "!",
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value={
                    "privacy": STORED_MARKER,
                    "value": "ready",
                },
            )
        )

        self.assertEqual(result, "ready!")

    async def test_run_keeps_legacy_object_input_envelope_collision(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "privacy": inputs["privacy"],
                    "value": inputs["value"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.object(
                        {
                            "type": "object",
                            "required": ["privacy", "value"],
                            "additionalProperties": False,
                            "properties": {
                                "privacy": {
                                    "type": "string",
                                    "enum": [STORED_MARKER],
                                },
                                "value": {"type": "string"},
                            },
                        }
                    ),
                    output_contract=self._object_output_contract(),
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value={
                    "privacy": STORED_MARKER,
                    "value": "ready",
                },
            )
        )

        self.assertEqual(
            result,
            {"privacy": STORED_MARKER, "value": "ready"},
        )

    async def test_run_keeps_declared_object_input_with_privacy_marker(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "privacy": inputs["privacy"],
                    "value": inputs["value"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        for marker in (
            DROPPED_MARKER,
            ENCRYPTED_MARKER,
            HASHED_MARKER,
            REDACTED_MARKER,
        ):
            with self.subTest(marker=marker):
                result = await runner.run(
                    self._context(
                        definition=self._context_definition(
                            input_contract=TaskInputContract.object(
                                {
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
                            output_contract=self._object_output_contract(),
                            run=TaskRunPolicy.queued("default"),
                        ),
                        input_value={
                            "privacy": marker,
                            "value": "ready",
                        },
                    )
                )

                self.assertEqual(
                    result,
                    {"privacy": marker, "value": "ready"},
                )

    async def test_run_accepts_plain_queued_input(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: inputs[FLOW_TASK_INPUT_KEY] + "!",
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value="ready",
            )
        )

        self.assertEqual(result, "ready!")

    async def test_run_accepts_plain_queued_object_input(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "name": inputs["name"],
                    "limit": inputs[FLOW_TASK_INPUT_KEY]["limit"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.object(
                        {
                            "type": "object",
                            "required": ["name", "limit"],
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                        }
                    ),
                    output_contract=self._object_output_contract(),
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value={"name": "ready", "limit": 2},
            )
        )

        self.assertEqual(result, {"name": "ready", "limit": 2})

    async def test_run_accepts_queued_file_array_from_durable_refs(
        self,
    ) -> None:
        def inspect(inputs: Mapping[str, object]) -> dict[str, object]:
            descriptors = cast(
                list[TaskFileDescriptor],
                inputs[FLOW_TASK_INPUT_KEY],
            )
            return {
                "references": [
                    descriptor.reference for descriptor in descriptors
                ],
                "source_kinds": [
                    descriptor.source_kind.value for descriptor in descriptors
                ],
                "mime_types": [
                    descriptor.mime_type for descriptor in descriptors
                ],
            }

        flow = Flow()
        flow.add_node(Node("A", func=inspect))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)
        artifact_file = TaskInputFile(
            logical_path="artifact:source-1",
            artifact_ref=TaskArtifactRef(
                artifact_id="source-1",
                store="local",
                storage_key="so/source-1",
                media_type="application/pdf",
                size_bytes=10,
                sha256="0" * 64,
            ),
            media_type="application/pdf",
            size_bytes=10,
        )
        provider_file = TaskInputFile(
            logical_path="provider:file-123",
            provider_reference=TaskProviderReference(
                kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                provider="openai",
                reference="file-123",
                owner_scope="private-owner",
                mime_type="application/pdf",
                size_bucket="private-bucket",
                identity_hmac="safe-hmac",
            ),
            media_type="application/pdf",
            size_bytes=20,
        )

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.file_array(
                        mime_types=("application/pdf",),
                    ),
                    output_contract=self._object_output_contract(),
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value={
                    "privacy": ENCRYPTED_MARKER,
                    "raw": "private prompt",
                },
                files=(artifact_file, provider_file),
            )
        )

        self.assertEqual(
            result,
            {
                "references": ["source-1", "file-123"],
                "source_kinds": ["artifact", "provider_reference"],
                "mime_types": ["application/pdf", "application/pdf"],
            },
        )

    async def test_run_accepts_queued_file_from_durable_ref(self) -> None:
        def inspect(inputs: Mapping[str, object]) -> dict[str, object]:
            descriptor = cast(
                TaskFileDescriptor,
                inputs[FLOW_TASK_INPUT_KEY],
            )
            return {
                "reference": descriptor.reference,
                "source_kind": descriptor.source_kind.value,
                "mime_type": descriptor.mime_type,
            }

        flow = Flow()
        flow.add_node(Node("A", func=inspect))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)
        file = TaskInputFile(
            logical_path="artifact:source-1",
            artifact_ref=TaskArtifactRef(
                artifact_id="source-1",
                store="local",
                storage_key="so/source-1",
                media_type="application/pdf",
                size_bytes=10,
                sha256="0" * 64,
            ),
            media_type="application/pdf",
            size_bytes=10,
        )

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("application/pdf",),
                    ),
                    output_contract=self._object_output_contract(),
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value={
                    "privacy": ENCRYPTED_MARKER,
                    "raw": "private prompt",
                },
                files=(file,),
            )
        )

        self.assertEqual(
            result,
            {
                "reference": "source-1",
                "source_kind": "artifact",
                "mime_type": "application/pdf",
            },
        )

    async def test_run_rejects_queued_file_inputs_without_refs_safely(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: "unused output"))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        cases = (
            (
                "missing_single_file",
                TaskInputContract.file(mime_types=("application/pdf",)),
                (),
            ),
            (
                "volatile_file_array",
                TaskInputContract.file_array(mime_types=("application/pdf",)),
                (
                    TaskInputFile(
                        logical_path="private/report.pdf",
                        media_type="application/pdf",
                        size_bytes=10,
                    ),
                ),
            ),
        )
        for name, input_contract, files in cases:
            with self.subTest(name=name):
                with self.assertRaises(TaskValidationError) as error:
                    await runner.run(
                        self._context(
                            definition=self._context_definition(
                                input_contract=input_contract,
                                run=TaskRunPolicy.queued("default"),
                            ),
                            input_value={
                                "privacy": ENCRYPTED_MARKER,
                                "raw": "private prompt",
                            },
                            files=files,
                        )
                    )

                self.assertEqual(
                    error.exception.issues[0].code,
                    "execution.unsupported_flow",
                )
                self.assertNotIn("private prompt", str(error.exception))
                self.assertNotIn("private/report.pdf", str(error.exception))

    async def test_run_rejects_unavailable_queued_input_safely(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: "unused output"))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        for marker in (
            DROPPED_MARKER,
            ENCRYPTED_MARKER,
            HASHED_MARKER,
            REDACTED_MARKER,
        ):
            with self.subTest(marker=marker):
                with self.assertRaises(TaskValidationError) as error:
                    await runner.run(
                        self._context(
                            definition=self._context_definition(
                                run=TaskRunPolicy.queued("default"),
                            ),
                            input_value={
                                "privacy": marker,
                                "raw": "private prompt",
                            },
                        )
                    )

                self.assertEqual(
                    error.exception.issues[0].code,
                    "execution.unsupported_flow",
                )
                self.assertNotIn("private prompt", str(error.exception))
                self.assertNotIn("unused output", str(error.exception))

    async def test_run_rejects_multiple_start_nodes_safely(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: "private A"))
        flow.add_node(Node("B", func=lambda _: "private B"))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(self._context(input_value="private prompt"))

        self.assertEqual(
            error.exception.issues[0].code, "execution.unsupported_flow"
        )
        self.assertNotIn("private prompt", str(error.exception))
        self.assertNotIn("private A", str(error.exception))

    async def test_run_rejects_invalid_resolver_result_safely(self) -> None:
        def resolver(_: TaskTargetContext) -> Flow:
            return cast(Flow, "private flow")

        runner = FlowTaskTargetRunner(flow_resolver=resolver)

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(self._context(input_value="private prompt"))

        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["execution.ref"],
        )
        self.assertNotIn("private prompt", str(error.exception))
        self.assertNotIn("private flow", str(error.exception))

    async def test_run_checks_cancellation_before_success(self) -> None:
        executed: list[str] = []

        def start(_: dict[str, object]) -> str:
            executed.append("A")
            return "done"

        async def cancel_after_node() -> None:
            if executed == ["A"]:
                raise CancelledError()

        flow = Flow()
        flow.add_node(Node("A", func=start))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        with self.assertRaises(CancelledError):
            await runner.run(
                self._context(
                    input_value="private prompt",
                    cancellation_checker=cancel_after_node,
                )
            )

        self.assertEqual(executed, ["A"])

    async def test_run_timeout_covers_flow_work(self) -> None:
        async def slow(_: dict[str, object]) -> str:
            await sleep(0.05)
            return "done"

        flow = Flow()
        flow.add_node(Node("A", func=slow))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        with self.assertRaises(TimeoutError):
            await wait_for(
                runner.run(self._context(input_value="private prompt")),
                timeout=0.001,
            )

    async def test_file_convert_node_rejects_missing_converter_safely(
        self,
    ) -> None:
        context = self._context()

        result = await FlowDefinitionLoader(
            registry=task_flow_node_registry(context)
        ).loads_result(
            """
            [flow]
            name = "render"
            entrypoint = "render"
            output_node = "render"

            [nodes.render]
            type = "file_convert"

            [nodes.render.config]
            converter = "missing"
            """
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.converter_unsupported")
        self.assertEqual(
            result.issues[0].path,
            "nodes.render.config.converter",
        )
        self.assertNotIn("missing", str(result.issues))

    async def test_file_convert_node_reports_missing_dependency_safely(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter(
            (_page_result(1, b"page"),),
            dependency_gates=(TaskFeature.PDF_IMAGE_CONVERSION,),
        )
        context = self._context(file_converters={"pdf_image": converter})

        with patch(
            "avalan.task.targets.flow.feature_available",
            return_value=False,
        ):
            result = await FlowDefinitionLoader(
                registry=task_flow_node_registry(context)
            ).loads_result(
                """
                [flow]
                name = "render"
                entrypoint = "render"
                output_node = "render"

                [nodes.render]
                type = "pdf_to_images"
                output = "files"
                """
            )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "dependency.task_pdf_images_missing",
        )
        self.assertEqual(
            result.issues[0].path,
            "nodes.render.config.converter",
        )
        self.assertEqual(converter.calls, [])

    async def test_file_convert_node_reports_runtime_dependency_safely(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter(
            (_page_result(1, b"page"),),
            dependency_gates=(TaskFeature.PDF_IMAGE_CONVERSION,),
        )
        with TemporaryDirectory() as artifacts:
            context, file, store = await self._conversion_context(
                artifacts,
                converter,
            )
            with patch(
                "avalan.task.targets.flow.feature_available",
                return_value=True,
            ):
                node = task_flow_node_registry(context).build(
                    FlowNodeDefinition(
                        name="render",
                        type="pdf_to_images",
                        output="files",
                    )
                )

            with patch(
                "avalan.task.converters.feature_available",
                return_value=False,
            ):
                with self.assertRaises(TaskValidationError) as error:
                    await node.execute_async({"file": file})

            records = await store.list_artifacts(context.execution.run_id)

        self.assertEqual(
            error.exception.issues[0].code,
            "dependency.task_pdf_images_missing",
        )
        self.assertEqual(records, ())
        self.assertEqual(converter.calls, [])
        self.assertNotIn("%PDF-private", str(error.exception))

    async def test_file_convert_node_rejects_non_file_input_safely(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))
        context = self._context(
            file_converters={"pdf_image": converter},
        )
        node = task_flow_node_registry(context).build(
            FlowNodeDefinition(
                name="render",
                type="pdf_to_images",
            )
        )

        with self.assertRaises(TaskValidationError) as error:
            await node.execute_async({"file": "private.pdf"})

        self.assertEqual(
            error.exception.issues[0].path,
            "nodes.render.input[0]",
        )
        self.assertEqual(converter.calls, [])
        self.assertNotIn("private.pdf", str(error.exception))

    async def test_file_convert_node_rejects_missing_stores_and_refs_safely(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))
        file = TaskInputFile(
            logical_path="artifact:source-1",
            artifact_ref=TaskArtifactRef(
                artifact_id="source-1",
                store="local",
                storage_key="so/source-1",
                media_type="application/pdf",
                size_bytes=10,
            ),
            media_type="application/pdf",
            size_bytes=10,
        )
        cases = (
            (
                "artifact_store",
                self._context(
                    files=(file,),
                    file_converters={"pdf_image": converter},
                ),
                file,
                "artifact store",
            ),
            (
                "task_store",
                self._context(
                    files=(file,),
                    artifact_store=FailingArtifactStore(),
                    file_converters={"pdf_image": converter},
                ),
                file,
                "task store",
            ),
            (
                "artifact_ref",
                self._context(
                    files=(file,),
                    artifact_store=FailingArtifactStore(),
                    task_store=InMemoryTaskStore(),
                    file_converters={"pdf_image": converter},
                ),
                TaskInputFile(
                    logical_path="provider:openai:file",
                    media_type="application/pdf",
                ),
                "artifact backed",
            ),
        )

        for name, context, input_file, expected in cases:
            with self.subTest(name=name):
                node = task_flow_node_registry(context).build(
                    FlowNodeDefinition(name="render", type="pdf_to_images")
                )

                with self.assertRaises(TaskValidationError) as error:
                    await node.execute_async({"file": input_file})

                self.assertIn(expected, error.exception.issues[0].message)
                self.assertNotIn("source-1", str(error.exception))

    async def test_file_convert_node_enforces_total_pixel_limit_safely(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter(
            (
                _page_result(1, b"page one", width_pixels=10),
                _page_result(2, b"page two", width_pixels=10),
            )
        )
        with TemporaryDirectory() as artifacts:
            context, file, store = await self._conversion_context(
                artifacts,
                converter,
            )
            node = task_flow_node_registry(context).build(
                FlowNodeDefinition(
                    name="render",
                    type="file_convert",
                    config={
                        "converter": "pdf_image",
                        "max_total_pixels": 50,
                    },
                )
            )

            with self.assertRaises(TaskValidationError) as error:
                await node.execute_async({"file": file})

            records = await store.list_artifacts(context.execution.run_id)

        self.assertEqual(records, ())
        self.assertEqual(len(converter.calls), 1)
        self.assertNotIn("%PDF-private", str(error.exception))
        self.assertNotIn("page one", str(error.exception))

    async def test_file_converter_wrapper_accepts_total_pixel_limit(
        self,
    ) -> None:
        converter = flow_target_module._FlowFileConverter(  # type: ignore[attr-defined]
            RecordingPdfPageConverter(
                (
                    _page_result(1, b"page one", width_pixels=10),
                    _page_result(2, b"page two", width_pixels=10),
                )
            ),
            limits=flow_target_module._FlowConversionLimits(  # type: ignore[attr-defined]
                max_total_pixels=500,
            ),
        )

        result = await converter.convert_pages(
            b"%PDF-private source",
            source_media_type="application/pdf",
        )

        self.assertEqual(len(result.pages), 2)

    async def test_file_convert_node_enforces_page_limit_safely(self) -> None:
        converter = RecordingPdfPageConverter(
            (
                _page_result(1, b"page one"),
                _page_result(2, b"page two"),
            )
        )
        with TemporaryDirectory() as artifacts:
            context, file, store = await self._conversion_context(
                artifacts,
                converter,
            )
            node = task_flow_node_registry(context).build(
                FlowNodeDefinition(
                    name="render",
                    type="file_convert",
                    config={
                        "converter": "pdf_image",
                        "max_pages": 1,
                    },
                )
            )

            with self.assertRaises(TaskValidationError) as error:
                await node.execute_async({"file": file})

            records = await store.list_artifacts(context.execution.run_id)

        self.assertEqual(records, ())
        self.assertEqual(len(converter.calls), 1)
        self.assertNotIn("%PDF-private", str(error.exception))
        self.assertNotIn("page one", str(error.exception))

    async def test_file_convert_node_returns_unwrapped_files(self) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))
        with TemporaryDirectory() as artifacts:
            context, file, _ = await self._conversion_context(
                artifacts,
                converter,
            )
            node = task_flow_node_registry(context).build(
                FlowNodeDefinition(
                    name="render",
                    type="pdf_to_images",
                    input="bundle",
                )
            )

            result = await node.execute_async({"bundle": {"files": [file]}})

        files = cast(list[TaskInputFile], result)
        self.assertEqual(
            [file.logical_path for file in files],
            ["artifact:page-1"],
        )
        self.assertEqual(files[0].media_type, "image/png")
        with TemporaryDirectory() as artifacts:
            context, file, _ = await self._conversion_context(
                artifacts,
                converter,
            )
            node = task_flow_node_registry(context).build(
                FlowNodeDefinition(
                    name="render",
                    type="pdf_to_images",
                    input="bundle",
                )
            )

            result = await node.execute_async({"bundle": {"file": file}})

        files = cast(list[TaskInputFile], result)
        self.assertEqual(
            [file.logical_path for file in files],
            ["artifact:page-1"],
        )
        with TemporaryDirectory() as artifacts:
            context, file, _ = await self._conversion_context(
                artifacts,
                converter,
            )
            node = task_flow_node_registry(context).build(
                FlowNodeDefinition(name="render", type="pdf_to_images")
            )

            result = await node.execute_async({FLOW_TASK_FILES_KEY: [file]})

        files = cast(list[TaskInputFile], result)
        self.assertEqual(
            [file.logical_path for file in files],
            ["artifact:page-1"],
        )

    async def test_file_convert_node_rejects_empty_and_missing_selectors(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))
        for inputs, definition in (
            (
                {"files": []},
                FlowNodeDefinition(name="render", type="pdf_to_images"),
            ),
            (
                {"value": "private.pdf"},
                FlowNodeDefinition(
                    name="render",
                    type="pdf_to_images",
                    input="missing",
                ),
            ),
            (
                {"value": "private.pdf"},
                FlowNodeDefinition(name="render", type="pdf_to_images"),
            ),
        ):
            with self.subTest(inputs=tuple(inputs)):
                context = self._context(
                    file_converters={"pdf_image": converter},
                )
                node = task_flow_node_registry(context).build(definition)

                with self.assertRaises(TaskValidationError) as error:
                    await node.execute_async(inputs)

                self.assertNotIn("private.pdf", str(error.exception))

    async def test_file_convert_node_checks_cancellation_before_work(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))

        async def cancel() -> None:
            raise CancelledError()

        context = self._context(
            cancellation_checker=cancel,
            file_converters={"pdf_image": converter},
        )
        node = task_flow_node_registry(context).build(
            FlowNodeDefinition(name="render", type="pdf_to_images")
        )

        with self.assertRaises(CancelledError):
            await node.execute_async({"file": "private.pdf"})

        self.assertEqual(converter.calls, [])

    async def test_run_rejects_non_flow_definition(self) -> None:
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: Flow())

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    definition=TaskDefinition(
                        task=TaskMetadata(name="flow-task", version="1"),
                        input=TaskInputContract.string(),
                        output=TaskOutputContract.text(),
                        execution=TaskExecutionTarget.agent(
                            "agents/valid.toml"
                        ),
                    )
                )
            )

        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["execution.unknown_target"],
        )

    async def _conversion_context(
        self,
        artifacts: str,
        converter: RecordingPdfPageConverter,
    ) -> tuple[TaskTargetContext, TaskInputFile, InMemoryTaskStore]:
        artifact_store = LocalArtifactStore(
            artifacts,
            raw_storage_allowed=True,
            id_factory=_id_factory(("page-1", "page-2")),
        )
        source_ref = await artifact_store.put(
            b"%PDF-private source",
            artifact_id="source-1",
            media_type="application/pdf",
        )
        task_store = InMemoryTaskStore()
        definition = self._context_definition(
            input_contract=TaskInputContract.file(
                mime_types=("application/pdf",)
            ),
            output_contract=self._object_output_contract(),
        )
        await task_store.register_definition(
            definition,
            definition_hash="flow-conversion-node",
        )
        run = await task_store.create_run(
            TaskExecutionRequest(definition_id="flow-conversion-node")
        )
        attempt = await task_store.create_attempt(run.run_id)
        file = TaskInputFile(
            logical_path="artifact:source-1",
            artifact_ref=source_ref,
            media_type="application/pdf",
            size_bytes=source_ref.size_bytes,
        )
        context = TaskTargetContext(
            definition=definition,
            execution=attempt.context,
            files=(file,),
            artifact_store=artifact_store,
            task_store=task_store,
            file_converters={"pdf_image": converter},
        )
        return context, file, task_store

    def _context(
        self,
        *,
        definition: TaskDefinition | None = None,
        input_value: object = None,
        files: tuple[TaskInputFile, ...] = (),
        cancellation_checker: Callable[[], Awaitable[None]] | None = None,
        event_listener: (
            Callable[[Event], Awaitable[None] | None] | None
        ) = None,
        execution: TaskExecutionContext | None = None,
        artifact_store: object | None = None,
        task_store: object | None = None,
        file_converters: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskTargetContext:
        return TaskTargetContext(
            definition=definition or self._context_definition(),
            execution=execution
            or TaskExecutionContext(
                run_id="run-1",
                attempt_id="attempt-1",
                attempt_number=1,
            ),
            input_value=input_value,
            files=files,
            cancellation_checker=cancellation_checker,
            event_listener=event_listener,
            artifact_store=cast(Any, artifact_store),
            task_store=cast(Any, task_store),
            file_converters=cast(Mapping[str, Any], file_converters or {}),
            metadata=metadata or {},
        )

    def _context_definition(
        self,
        *,
        input_contract: TaskInputContract | None = None,
        output_contract: TaskOutputContract | None = None,
        privacy: TaskPrivacyPolicy | None = None,
        run: TaskRunPolicy | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="flow-task", version="1"),
            input=input_contract or TaskInputContract.string(),
            output=output_contract or TaskOutputContract.text(),
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
            privacy=privacy or TaskPrivacyPolicy(),
            run=run or TaskRunPolicy.direct(),
        )

    def _object_output_contract(self) -> TaskOutputContract:
        return TaskOutputContract.object({"type": "object"})


class FlowTaskTargetRunnerE2ETest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryTaskStore()

    async def test_direct_runner_commits_object_output_after_validation(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "status": "ready",
                    "count": inputs["limit"],
                },
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-success",
        )

        result = await runner.run(
            self._definition(
                input_contract=self._object_input_contract(),
                output_contract=self._object_output_contract(),
            ),
            input_value={"prompt": "safe summary", "limit": 2},
        )

        self.assertEqual(
            result.run.state,
            TaskRunState.SUCCEEDED,
        )
        self.assertEqual(result.output, {"status": "ready", "count": 2})

    async def test_direct_runner_records_strict_flow_state(self) -> None:
        flow_store = InMemoryFlowStateStore()
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(
                strict_resolver=lambda _: _strict_echo_definition(),
                flow_state_store=flow_store,
            ),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-strict-state",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.text(),
                observability=TaskObservabilityPolicy(),
            ),
            input_value="ready",
        )
        record = await flow_store.get_flow_execution(result.run.run_id)
        events = await self.store.list_events(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "ready")
        self.assertEqual(record.revision, 2)
        self.assertEqual(dict(record.selected_outputs), {"answer": "ready"})
        self.assertEqual(
            record.node_attempts[0].state, FlowNodeState.SUCCEEDED
        )
        self.assertEqual(
            [event.event_type for event in events],
            [
                "flow_manager_call_before",
                "flow_validation",
                "flow_started",
                "flow_node_started",
                "flow_node_completed",
                "flow_output_selected",
                "flow_completed",
                "flow_manager_call_after",
            ],
        )

    async def test_direct_runner_executes_graph_authored_strict_flow(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_path = root / "flows" / "inline.toml"
            _write_strict_graph_flow(
                flow_path,
                diagram=(
                    "flowchart LR\n"
                    "start route_1@-->|Private customer route| finish\n"
                    "private_note decorative_1@-->|Private visual note| "
                    "private_sink\n"
                ),
            )
            flow_store = InMemoryFlowStateStore()
            runner = DirectTaskRunner(
                self.store,
                target=FlowTaskTargetRunner(
                    ref_base=root,
                    strict_resolver=_strict_flow_loader_resolver(root),
                    flow_state_store=flow_store,
                ),
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda _: "flow-direct-graph-strict",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.string(),
                    output_contract=TaskOutputContract.text(),
                    execution=TaskExecutionTarget.flow("flows/inline.toml"),
                    observability=TaskObservabilityPolicy(),
                ),
                input_value="ready",
            )
            record = await flow_store.get_flow_execution(result.run.run_id)
            events = await self.store.list_events(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "ready")
        self.assertEqual(dict(record.selected_outputs), {"answer": "ready"})
        self.assertEqual(
            [(edge.source, edge.target) for edge in record.trace.edges],
            [("start", "finish")],
        )
        rendered = f"{result.run.result} {record.as_snapshot()} {events}"
        self.assertNotIn("flowchart", rendered)
        self.assertNotIn("Private customer route", rendered)
        self.assertNotIn("Private visual note", rendered)
        self.assertNotIn("private_note", rendered)
        self.assertNotIn("private_sink", rendered)

    async def test_direct_runner_executes_strict_subflow_plan(self) -> None:
        flow_store = InMemoryFlowStateStore()

        async def resolve_strict_subflow(
            _: TaskTargetContext,
        ) -> FlowExecutionPlan:
            return await _strict_subflow_plan()

        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(
                strict_resolver=resolve_strict_subflow,
                flow_state_store=flow_store,
            ),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-strict-subflow",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.text(),
                observability=TaskObservabilityPolicy(),
            ),
            input_value="ready",
        )
        record = await flow_store.get_flow_execution(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "ready")
        self.assertEqual(dict(record.selected_outputs), {"answer": "ready"})
        self.assertEqual(record.node_attempts[0].node, "child")
        self.assertEqual(
            record.node_attempts[0].state,
            FlowNodeState.SUCCEEDED,
        )

    async def test_direct_runner_replaces_unsigned_strict_state(self) -> None:
        flow_store = InMemoryFlowStateStore()
        seeded = False

        async def resolve(context: TaskTargetContext) -> FlowDefinition:
            nonlocal seeded
            if not seeded:
                seeded = True
                await flow_store.create_flow_execution(
                    context.execution.run_id,
                    trace=FlowExecutionTrace(nodes=()),
                    selected_outputs={"answer": "private stale output"},
                )
            return _strict_constant_definition()

        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(
                strict_resolver=resolve,
                flow_state_store=flow_store,
            ),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-replace-unsigned-state",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.text(),
                observability=TaskObservabilityPolicy(),
            ),
            input_value="private prompt",
        )
        record = await flow_store.get_flow_execution(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "public result")
        self.assertEqual(record.revision, 2)
        self.assertEqual(
            dict(record.selected_outputs),
            {"answer": "public result"},
        )
        self.assertIn("strict_flow", record.metadata)
        self.assertNotIn("private prompt", str(record.as_snapshot()))
        self.assertNotIn("private stale output", str(result.run.result))

    async def test_direct_runner_records_strict_state_without_input_leak(
        self,
    ) -> None:
        flow_store = InMemoryFlowStateStore()
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(
                strict_resolver=lambda _: _strict_constant_definition(),
                flow_state_store=flow_store,
            ),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-strict-private-state",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.text(),
                observability=TaskObservabilityPolicy(),
            ),
            input_value="private prompt",
        )
        record = await flow_store.get_flow_execution(result.run.run_id)
        events = await self.store.list_events(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "public result")
        self.assertEqual(
            dict(record.selected_outputs),
            {"answer": "public result"},
        )
        self.assertEqual(record.artifact_refs, ())
        self.assertNotIn("private prompt", str(record.as_snapshot()))
        self.assertNotIn("private prompt", str(events))

    async def test_direct_runner_keeps_flow_input_mutation_local(
        self,
    ) -> None:
        def mutate(inputs: dict[str, object]) -> dict[str, object]:
            nested = cast(dict[str, object], inputs["nested"])
            items = cast(list[str], nested["items"])
            items.append("mutated")
            full_input = cast(dict[str, object], inputs[FLOW_TASK_INPUT_KEY])
            full_nested = cast(dict[str, object], full_input["nested"])
            return {
                "field": items,
                "full": cast(list[str], full_nested["items"]),
            }

        flow = Flow()
        flow.add_node(Node("A", func=mutate))
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-input-isolation",
        )
        input_value = {"nested": {"items": ["original"]}}

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.object(
                    {
                        "type": "object",
                        "required": ["nested"],
                        "additionalProperties": False,
                        "properties": {
                            "nested": {
                                "type": "object",
                                "required": ["items"],
                                "additionalProperties": False,
                                "properties": {
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                    }
                ),
                output_contract=TaskOutputContract.object(
                    {
                        "type": "object",
                        "required": ["field", "full"],
                        "additionalProperties": False,
                        "properties": {
                            "field": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "full": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    }
                ),
            ),
            input_value=input_value,
        )

        self.assertEqual(
            result.run.state,
            TaskRunState.SUCCEEDED,
        )
        self.assertEqual(
            result.output,
            {
                "field": ["original", "mutated"],
                "full": ["original"],
            },
        )
        self.assertEqual(input_value, {"nested": {"items": ["original"]}})

    async def test_direct_runner_persists_sanitized_flow_events(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: (
                    inputs[FLOW_TASK_INPUT_KEY] + " private output"
                ),
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-events",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.text(),
                observability=TaskObservabilityPolicy(),
            ),
            input_value="private prompt",
        )

        events = await self.store.list_events(result.run.run_id)
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            [event.event_type for event in events],
            [
                "flow_manager_call_before",
                "flow_manager_call_after",
            ],
        )
        self.assertEqual(
            [event.category for event in events],
            [TaskEventCategory.ENGINE, TaskEventCategory.ENGINE],
        )
        start_payload = cast(Mapping[str, object], events[0].payload)
        end_payload = cast(Mapping[str, object], events[1].payload)
        self.assertEqual(start_payload["status"], "started")
        self.assertEqual(end_payload["status"], "succeeded")
        self.assertIn("duration_ms", end_payload)
        self.assertNotIn("private prompt", str(events))
        self.assertNotIn("private output", str(events))

    async def test_direct_runner_persists_failed_flow_event_safely(
        self,
    ) -> None:
        def fail(_: dict[str, object]) -> str:
            raise RuntimeError("private node failure")

        flow = Flow()
        flow.add_node(Node("A", func=fail))
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-failed-event",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.text(),
                observability=TaskObservabilityPolicy(),
            ),
            input_value="private prompt",
        )

        events = await self.store.list_events(result.run.run_id)
        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(events[1].event_type, "flow_manager_call_after")
        end_payload = cast(Mapping[str, object], events[1].payload)
        self.assertEqual(end_payload["status"], "failed")
        self.assertNotIn("private node failure", str(events))
        self.assertNotIn("private prompt", str(events))

    async def test_direct_runner_classifies_flow_validation_failure(
        self,
    ) -> None:
        def resolver(_: TaskTargetContext) -> Flow:
            return cast(Flow, "private invalid flow")

        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=resolver),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-runtime-validation",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.text(),
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertIsNone(result.output)
        assert result.run.result is not None
        error = cast(Mapping[str, object], result.run.result.error)
        self.assertEqual(error["category"], "runnable")
        self.assertEqual(error["code"], "runnable.failed")
        self.assertNotIn("private invalid flow", str(error))
        self.assertNotIn("private prompt", str(error))

    async def test_direct_runner_records_flow_file_output_artifact(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=lambda: "flow-output-1",
            )

            async def produce_artifact(
                _: dict[str, object],
            ) -> TaskArtifactRef:
                return await artifact_store.put(
                    b"private flow bytes",
                    media_type="text/plain",
                    metadata={"filename": "private-flow.txt"},
                )

            flow = Flow()
            flow.add_node(Node("A", func=produce_artifact))
            runner = DirectTaskRunner(
                self.store,
                target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                definition_hash=lambda _: "flow-direct-artifact-output",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.string(),
                    output_contract=TaskOutputContract.file(),
                    artifact=TaskArtifactPolicy.references_only(
                        retention_days=6,
                    ),
                    privacy=TaskPrivacyPolicy(output=PrivacyAction.REDACT),
                ),
                input_value="private prompt",
            )

        records = await self.store.list_artifacts(
            result.run.run_id,
            purpose=TaskArtifactPurpose.OUTPUT,
        )
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].state, TaskArtifactState.READY)
        self.assertEqual(records[0].retention.delete_after_days, 6)
        self.assertEqual(records[0].ref.metadata, {"privacy": "<redacted>"})
        self.assertNotIn("private-flow.txt", str(records))
        self.assertNotIn("private flow bytes", str(records))
        self.assertNotIn("private prompt", str(records))

    async def test_direct_runner_rejects_invalid_flow_artifact_output(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: object()))
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-invalid-artifact-output",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.file(),
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(
            await self.store.list_artifacts(
                result.run.run_id,
                purpose=TaskArtifactPurpose.OUTPUT,
            ),
            (),
        )
        self.assertNotIn("private prompt", str(result.run.result))

    async def test_direct_runner_rejects_invalid_flow_output(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "status": "ready",
                    "count": "private invalid count",
                },
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-invalid-output",
        )

        result = await runner.run(
            self._definition(
                input_contract=self._object_input_contract(),
                output_contract=self._object_output_contract(),
                observability=TaskObservabilityPolicy(),
            ),
            input_value={"prompt": "private prompt", "limit": 1},
        )

        events = await self.store.list_events(result.run.run_id)
        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertIsNone(result.output)
        self.assertEqual(events[1].event_type, "flow_manager_call_after")
        end_payload = cast(Mapping[str, object], events[1].payload)
        self.assertEqual(end_payload["status"], "failed")
        self.assertNotIn("private invalid count", str(events))
        self.assertNotIn("private prompt", str(events))
        self.assertNotIn("private invalid count", str(result.run.result))
        self.assertNotIn("private prompt", str(result.run.result))

    async def test_direct_runner_commits_array_output_from_flow_input(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(
            Node("A", func=lambda inputs: inputs[FLOW_TASK_INPUT_KEY])
        )
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-array-output",
        )

        result = await runner.run(
            self._definition(
                input_contract=self._array_input_contract(),
                output_contract=self._array_output_contract(),
            ),
            input_value=["safe", "done"],
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, ["safe", "done"])
        self.assertIsInstance(result.output, list)

    async def test_direct_runner_sends_file_input_to_flow_agent_node_output(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_path = _write_agent_flow_workspace(root)
            loader = FlowAgentLoader(text='{"status": "ready", "count": 2}')
            agent_runner = AgentTaskTargetRunner(loader, ref_base=root)
            runner = DirectTaskRunner(
                self.store,
                target=FlowTaskTargetRunner(
                    ref_base=root,
                    flow_resolver=_flow_loader_resolver(
                        flow_path,
                        agent_runner=agent_runner,
                        root=root,
                    ),
                ),
                hmac_provider=StaticHmacProvider(),
                artifact_store=cast(Any, FailingArtifactStore()),
                execution_roots=(root,),
                definition_hash=lambda _: "flow-agent-file-success",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("application/pdf",)
                    ),
                    output_contract=self._object_output_contract(),
                    execution=TaskExecutionTarget.flow("flow.toml"),
                    observability=TaskObservabilityPolicy(),
                ),
                input_value=TaskFileDescriptor.provider_reference_descriptor(
                    "file-123",
                    kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                    provider="openai",
                    mime_type="application/pdf",
                    owner_scope="private-tenant",
                ),
            )

            expected_path = str(root / "agents" / "review.toml")

        self.assertEqual(
            result.run.state,
            TaskRunState.SUCCEEDED,
        )
        self.assertEqual(result.output, {"status": "ready", "count": 2})
        self.assertEqual(loader.paths, [expected_path])
        self.assertEqual(loader.entered, 1)
        self.assertEqual(loader.exited, 1)
        self.assertEqual(len(loader.inputs), 1)
        message = loader.inputs[0]
        assert isinstance(message, Message)
        self.assertEqual(message.role, MessageRole.USER)
        content = cast(list[object], message.content)
        text_blocks = [
            block for block in content if isinstance(block, MessageContentText)
        ]
        file_blocks = [
            block for block in content if isinstance(block, MessageContentFile)
        ]
        self.assertEqual([block.text for block in text_blocks], ["Review."])
        self.assertEqual(len(file_blocks), 1)
        self.assertEqual(file_blocks[0].file["file_id"], "file-123")
        self.assertEqual(
            file_blocks[0].file["mime_type"],
            "application/pdf",
        )
        self.assertNotIn("private-tenant", str(result.run.result))

    async def test_direct_runner_converts_pdf_flow_file_to_ordered_images(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter(
            (
                _page_result(1, b"page one"),
                _page_result(2, b"page two"),
            )
        )

        def resolve(context: TaskTargetContext) -> Flow:
            registry = task_flow_node_registry(context)
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
                            "dpi": 144,
                            "format": "png",
                            "max_pages": 2,
                            "pages": "1..2",
                        },
                    )
                )
            )

            def inspect(inputs: dict[str, object]) -> Mapping[str, object]:
                rendered = cast(Mapping[str, object], inputs["render"])
                files = cast(list[TaskInputFile], rendered["files"])
                refs = [file.artifact_ref for file in files]
                return {
                    "artifact_ids": [
                        ref.artifact_id for ref in refs if ref is not None
                    ],
                    "count": len(files),
                    "media_types": [file.media_type for file in files],
                    "paths": [file.logical_path for file in files],
                }

            flow.add_node(Node("inspect", func=inspect))
            flow.add_connection("render", "inspect")
            return flow

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "private.pdf").write_bytes(b"%PDF-private source")
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=_id_factory(("source-1", "page-1", "page-2")),
            )
            runner = DirectTaskRunner(
                self.store,
                target=FlowTaskTargetRunner(flow_resolver=resolve),
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                file_converters={"pdf_image": converter},
                execution_roots=(root,),
                definition_hash=lambda _: "flow-direct-file-convert",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("application/pdf",)
                    ),
                    output_contract=TaskOutputContract.object(
                        {
                            "type": "object",
                            "required": [
                                "artifact_ids",
                                "count",
                                "media_types",
                                "paths",
                            ],
                            "additionalProperties": False,
                            "properties": {
                                "artifact_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "count": {"type": "integer"},
                                "media_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "paths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        }
                    ),
                    artifact=TaskArtifactPolicy.references_only(
                        retention_days=4,
                    ),
                ),
                input_value=TaskFileDescriptor.local_path(
                    "private.pdf",
                    mime_type="application/pdf",
                ),
            )

        records = await self.store.list_artifacts(
            result.run.run_id,
            purpose=TaskArtifactPurpose.CONVERTED,
        )
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            result.output,
            {
                "artifact_ids": ["page-1", "page-2"],
                "count": 2,
                "media_types": ["image/png", "image/png"],
                "paths": ["artifact:page-1", "artifact:page-2"],
            },
        )
        self.assertEqual(
            [record.artifact_id for record in records],
            ["page-1", "page-2"],
        )
        self.assertEqual(
            [record.retention.delete_after_days for record in records],
            [4, 4],
        )
        self.assertEqual(len(converter.calls), 1)
        self.assertEqual(converter.calls[0][0], b"%PDF-private source")
        self.assertEqual(converter.calls[0][1], "application/pdf")
        self.assertEqual(
            converter.calls[0][2],
            {
                "dpi": 144,
                "format": "png",
                "pages": {"start": 1, "end": 2},
            },
        )
        self.assertNotIn("private.pdf", str(result.run.result))
        self.assertNotIn("%PDF-private", str(result.run.result))

    async def test_direct_runner_replaces_pdf_with_images_for_agent_node(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter(
            (
                _page_result(1, b"page one"),
                _page_result(2, b"page two"),
            )
        )
        agent_runner = CapturingTaskTargetRunner(
            output={"status": "ready", "count": 2}
        )

        def resolve(context: TaskTargetContext) -> Flow:
            registry = task_flow_node_registry(
                context,
                agent_runner=agent_runner,
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
                        },
                    )
                )
            )
            flow.add_node(
                registry.build(
                    FlowNodeDefinition(
                        name="extract",
                        type="agent",
                        ref="agents/extract.toml",
                        config={
                            "files_input": "render.files",
                            "file_policy": "replace",
                        },
                    )
                )
            )
            flow.add_connection("render", "extract")
            return flow

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "private.pdf").write_bytes(b"%PDF-private source")
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=_id_factory(("source-1", "page-1", "page-2")),
            )
            runner = DirectTaskRunner(
                self.store,
                target=FlowTaskTargetRunner(flow_resolver=resolve),
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                file_converters={"pdf_image": converter},
                execution_roots=(root,),
                definition_hash=lambda _: "flow-direct-agent-images",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("application/pdf",)
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
                    artifact=TaskArtifactPolicy.references_only(
                        retention_days=4,
                    ),
                ),
                input_value=TaskFileDescriptor.local_path(
                    "private.pdf",
                    mime_type="application/pdf",
                ),
            )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, {"status": "ready", "count": 2})
        self.assertEqual(len(agent_runner.contexts), 1)
        child_files = agent_runner.contexts[0].files
        self.assertEqual(
            [file.logical_path for file in child_files],
            ["artifact:page-1", "artifact:page-2"],
        )
        self.assertEqual(
            [file.media_type for file in child_files],
            ["image/png", "image/png"],
        )
        self.assertEqual(
            [
                file.artifact_ref.artifact_id
                for file in child_files
                if file.artifact_ref is not None
            ],
            ["page-1", "page-2"],
        )
        self.assertNotIn(
            "artifact:source-1",
            " ".join(file.logical_path for file in child_files),
        )
        self.assertNotIn("private.pdf", str(result.run.result))
        self.assertNotIn("%PDF-private", str(result.run.result))

    async def test_direct_runner_rejects_appended_converted_images(
        self,
    ) -> None:
        converter = RecordingPdfPageConverter(
            (_page_result(1, b"private image bytes"),)
        )
        loader = FlowAgentLoader(text='{"status": "ready", "count": 1}')

        def resolve(context: TaskTargetContext) -> Flow:
            agent_runner = AgentTaskTargetRunner(loader, ref_base=root)
            registry = task_flow_node_registry(
                context,
                agent_runner=agent_runner,
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
                            "max_pages": 1,
                        },
                    )
                )
            )
            flow.add_node(
                registry.build(
                    FlowNodeDefinition(
                        name="extract",
                        type="agent",
                        ref="agents/extract.toml",
                        config={
                            "files_input": "render.files",
                            "file_policy": "append",
                        },
                    )
                )
            )
            flow.add_connection("render", "extract")
            return flow

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agents").mkdir()
            (root / "agents" / "extract.toml").write_text(
                """
[agent]
name = "Extract"
user = "Review."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            (root / "private.pdf").write_bytes(b"%PDF-private source")
            artifact_store = LocalArtifactStore(
                root / "artifacts",
                raw_storage_allowed=True,
                id_factory=_id_factory(("source-1", "page-1")),
            )
            runner = DirectTaskRunner(
                self.store,
                target=FlowTaskTargetRunner(flow_resolver=resolve),
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                file_converters={"pdf_image": converter},
                execution_roots=(root,),
                definition_hash=lambda _: "flow-direct-agent-append-reject",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("application/pdf",)
                    ),
                    output_contract=self._object_output_contract(),
                    artifact=TaskArtifactPolicy.references_only(
                        retention_days=4,
                    ),
                ),
                input_value=TaskFileDescriptor.local_path(
                    "private.pdf",
                    mime_type="application/pdf",
                ),
            )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertIsNone(result.output)
        self.assertEqual(loader.inputs, [])
        self.assertEqual(len(converter.calls), 1)
        self.assertNotIn("private.pdf", str(result.run.result))
        self.assertNotIn("%PDF-private", str(result.run.result))
        self.assertNotIn("private image bytes", str(result.run.result))

    async def test_direct_runner_rejects_flow_agent_provider_mismatch(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_path = _write_agent_flow_workspace(
                root,
                uri="ai://env:KEY@anthropic/claude-3-5-sonnet",
            )
            loader = FlowAgentLoader()
            agent_runner = AgentTaskTargetRunner(loader, ref_base=root)
            runner = DirectTaskRunner(
                self.store,
                target=FlowTaskTargetRunner(
                    ref_base=root,
                    flow_resolver=_flow_loader_resolver(
                        flow_path,
                        agent_runner=agent_runner,
                        root=root,
                    ),
                ),
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda _: "flow-agent-provider-mismatch",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.file_array(
                        mime_types=("application/pdf",)
                    ),
                    output_contract=TaskOutputContract.text(),
                    execution=TaskExecutionTarget.flow("flow.toml"),
                ),
                input_value=[
                    TaskFileDescriptor.provider_reference_descriptor(
                        "file-private",
                        kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                        provider="openai",
                        mime_type="application/pdf",
                        owner_scope="private-tenant",
                    )
                ],
            )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(loader.inputs, [])
        self.assertEqual(loader.paths, [])
        self.assertNotIn("file-private", str(result.run.result))
        self.assertNotIn("private-tenant", str(result.run.result))

    async def test_direct_runner_rejects_invalid_flow_array_output(
        self,
    ) -> None:
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
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-invalid-array-output",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.array(
                    {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                    }
                ),
                output_contract=TaskOutputContract.array(
                    {
                        "type": "array",
                        "items": {"type": "integer"},
                    }
                ),
            ),
            input_value=[1],
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertIsNone(result.output)
        self.assertNotIn("private invalid item", str(result.run.result))

    def test_input_binding_exposes_scalar_array_and_object_shapes(
        self,
    ) -> None:
        self.assertEqual(
            flow_task_input_binding("ready"),
            {FLOW_TASK_INPUT_KEY: "ready", "value": "ready"},
        )
        self.assertEqual(
            flow_task_input_binding([1, 2]),
            {FLOW_TASK_INPUT_KEY: [1, 2], "items": [1, 2]},
        )
        self.assertEqual(
            flow_task_input_binding({"limit": 2}),
            {FLOW_TASK_INPUT_KEY: {"limit": 2}, "limit": 2},
        )
        self.assertEqual(
            flow_task_input_binding(
                {
                    FLOW_TASK_INPUT_KEY: "spoofed",
                    "limit": 2,
                }
            ),
            {
                FLOW_TASK_INPUT_KEY: {
                    FLOW_TASK_INPUT_KEY: "spoofed",
                    "limit": 2,
                },
                "limit": 2,
            },
        )

    def test_input_binding_exposes_task_file_descriptors(self) -> None:
        file = TaskInputFile(
            logical_path="provider:file-123",
            media_type="application/pdf",
            provider_reference=TaskProviderReference(
                kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                provider="openai",
                reference="file-123",
            ),
        )

        binding = flow_task_input_binding(
            {"source_kind": "provider_reference"},
            files=(file,),
        )

        self.assertIs(binding["file"], file)
        self.assertEqual(binding["files"], [file])
        self.assertEqual(binding[FLOW_TASK_FILES_KEY], [file])

    def test_input_binding_isolates_nested_object_mutation(self) -> None:
        value = {"nested": {"items": ["original"]}}
        binding = flow_task_input_binding(value)
        nested = cast(dict[str, object], binding["nested"])
        cast(list[str], nested["items"]).append("mutated")
        full_input = cast(dict[str, object], binding[FLOW_TASK_INPUT_KEY])
        full_nested = cast(dict[str, object], full_input["nested"])

        self.assertEqual(
            cast(list[str], nested["items"]),
            ["original", "mutated"],
        )
        self.assertEqual(cast(list[str], full_nested["items"]), ["original"])
        self.assertEqual(value, {"nested": {"items": ["original"]}})

    def test_input_binding_isolates_nested_array_mutation(self) -> None:
        value = [{"items": ["original"]}]
        binding = flow_task_input_binding(value)
        field_items = cast(list[object], binding["items"])
        field_item = cast(dict[str, object], field_items[0])
        cast(list[str], field_item["items"]).append("mutated")
        full_items = cast(list[object], binding[FLOW_TASK_INPUT_KEY])
        full_item = cast(dict[str, object], full_items[0])

        self.assertEqual(
            cast(list[str], field_item["items"]),
            ["original", "mutated"],
        )
        self.assertEqual(cast(list[str], full_item["items"]), ["original"])
        self.assertEqual(value, [{"items": ["original"]}])

    def test_input_binding_normalizes_nested_tuple_values(self) -> None:
        self.assertEqual(
            flow_task_input_binding([("first", "second")]),
            {
                FLOW_TASK_INPUT_KEY: [["first", "second"]],
                "items": [["first", "second"]],
            },
        )

    def test_flow_validator_requires_structured_output_schema(self) -> None:
        report = validate_flow_task_compatibility(
            self._definition(
                input_contract=self._object_input_contract(),
                output_contract=TaskOutputContract.object(),
            ),
            TaskValidationContext(),
        )

        self.assertFalse(report.compatible)
        self.assertEqual(
            [issue.path for issue in report.issues],
            ["output.schema"],
        )
        self.assertNotIn("private", str(report.issues))

    def _definition(
        self,
        *,
        input_contract: TaskInputContract,
        output_contract: TaskOutputContract,
        artifact: TaskArtifactPolicy | None = None,
        execution: TaskExecutionTarget | None = None,
        observability: TaskObservabilityPolicy | None = None,
        privacy: TaskPrivacyPolicy | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="flow-task", version="1"),
            input=input_contract,
            output=output_contract,
            execution=execution
            or TaskExecutionTarget.flow("flows/report.toml"),
            artifact=artifact or TaskArtifactPolicy.references_only(),
            observability=observability or TaskObservabilityPolicy.noop(),
            privacy=privacy or TaskPrivacyPolicy(),
        )

    def _object_input_contract(self) -> TaskInputContract:
        return TaskInputContract.object(
            {
                "type": "object",
                "required": ["prompt", "limit"],
                "additionalProperties": False,
                "properties": {
                    "prompt": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1},
                },
            }
        )

    def _object_output_contract(self) -> TaskOutputContract:
        return TaskOutputContract.object(
            {
                "type": "object",
                "required": ["status", "count"],
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["ready"]},
                    "count": {"type": "integer", "minimum": 1},
                },
            }
        )

    def _array_input_contract(self) -> TaskInputContract:
        return TaskInputContract.array(
            {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        )

    def _array_output_contract(self) -> TaskOutputContract:
        return TaskOutputContract.array(
            {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        )


def _strict_echo_definition() -> FlowDefinition:
    return FlowDefinition(
        name="task-echo",
        version="1",
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.JSON),
        ),
        entry_behavior=FlowEntryBehavior(node="echo"),
        output_behavior=FlowOutputBehavior(outputs={"answer": "echo.value"}),
        nodes=(
            FlowNodeDefinition(
                name="echo",
                type="pass-through",
                mappings=(
                    FlowInputMapping(target="value", source="input.prompt"),
                ),
            ),
        ),
    )


def _strict_pipeline_definition() -> FlowDefinition:
    return FlowDefinition(
        name="task-pipeline",
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
                        "steps": (
                            {
                                "id": "read",
                                "command": "cat",
                                "paths": ("README.md",),
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
                        ),
                    }
                },
            ),
        ),
    )


def _strict_constant_definition(
    value: str = "public result",
) -> FlowDefinition:
    return FlowDefinition(
        name="task-constant",
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


def _strict_routed_definition(target: str) -> FlowDefinition:
    return FlowDefinition(
        name="task-routed",
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
                name="answer",
                type="constant",
                config={"value": "answer result"},
            ),
            FlowNodeDefinition(
                name="alternate",
                type="constant",
                config={"value": "alternate result"},
            ),
        ),
        edges=(
            FlowEdgeDefinition(
                source="start",
                target=target,
                kind=FlowEdgeKind.SUCCESS,
            ),
        ),
    )


def _strict_two_step_definition() -> FlowDefinition:
    return FlowDefinition(
        name="task-two-step",
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


def _strict_human_review_plan() -> FlowExecutionPlan:
    return FlowExecutionPlan(
        name="task-review",
        version="1",
        revision=None,
        inputs=(
            FlowInputDefinition(name="prompt", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.TEXT),
        ),
        entry_node="start",
        output_selectors={"answer": parse_flow_selector("finish.value")},
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
                    "allowed_decisions": ("approved", "rejected"),
                    "audit_metadata": {"risk": "medium", "queue": "ops"},
                    "decision_schema": {
                        "type": "object",
                        "required": ("decision",),
                        "properties": {
                            "decision": {"enum": ("approved", "rejected")},
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
            FlowNodePlan(
                name="finish",
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
            ),
            FlowNodePlan(
                name="rejected",
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
            ),
        ),
        edges=(
            FlowEdgePlan(
                index=0,
                source="start",
                target="review",
                kind=FlowEdgeKind.SUCCESS,
            ),
            FlowEdgePlan(
                index=1,
                source="review",
                target="finish",
                kind=FlowEdgeKind.RESUME,
                label="approved",
            ),
            FlowEdgePlan(
                index=2,
                source="review",
                target="rejected",
                kind=FlowEdgeKind.RESUME,
                label="rejected",
            ),
        ),
    )


def _strict_nested_human_review_plan() -> FlowExecutionPlan:
    return FlowExecutionPlan(
        name="task-nested-review",
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
                        "plan": _strict_human_review_plan(),
                        "output_mapping": {"result": "answer"},
                    }
                },
            ),
        ),
    )


def _strict_human_review_matrix_plan() -> FlowExecutionPlan:
    decisions = {
        "approved": "approved_sink",
        "rejected": "rejected_sink",
        "needs-correction": "correction_sink",
        "expired": "expired_sink",
        "escalated": "escalated_sink",
    }
    return FlowExecutionPlan(
        name="task-review-matrix",
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
                name=target,
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
            for target in decisions.values()
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
                target=target,
                kind=FlowEdgeKind.RESUME,
                label=decision,
            )
            for index, (decision, target) in enumerate(
                decisions.items(),
                start=1,
            )
        )
        + (
            FlowEdgePlan(
                index=len(decisions) + 1,
                source="review",
                target="expired_sink",
                kind=FlowEdgeKind.TIMEOUT,
                label="expired",
            ),
        ),
    )


def _strict_file_binding_definition() -> FlowDefinition:
    return FlowDefinition(
        name="task-files",
        version="1",
        inputs=(
            FlowInputDefinition(name="document", type=FlowInputType.FILE),
            FlowInputDefinition(
                name="documents",
                type=FlowInputType.FILE_ARRAY,
            ),
            FlowInputDefinition(name="extra", type=FlowInputType.STRING),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.TEXT),
        ),
        entry_behavior=FlowEntryBehavior(node="echo"),
        output_behavior=FlowOutputBehavior(outputs={"answer": "echo.value"}),
        nodes=(
            FlowNodeDefinition(
                name="echo",
                type="pass-through",
                mappings=(
                    FlowInputMapping(target="value", source="input.extra"),
                ),
            ),
        ),
    )


async def _strict_subflow_plan() -> FlowExecutionPlan:
    child_result = await compile_flow_definition(_strict_echo_definition())
    assert child_result.plan is not None
    return FlowExecutionPlan(
        name="task-subflow",
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


def _loop_counter_plan() -> FlowExecutionPlan:
    compile_result = asyncio_run(
        compile_flow_definition(_strict_echo_definition())
    )
    assert compile_result.plan is not None
    node = compile_result.plan.nodes[0]
    loop = FlowLoopPlan(
        max_iterations=2,
        continue_condition=FlowConditionPlan(
            operator=FlowConditionOperator.EXISTS,
            selector=parse_flow_selector("input.prompt"),
        ),
        exit_condition=FlowConditionPlan(
            operator=FlowConditionOperator.EXISTS,
            selector=parse_flow_selector("retry.value"),
        ),
        output_selector=parse_flow_selector("retry.value"),
        limit_route="retry",
    )
    return replace(
        compile_result.plan,
        nodes=(
            FlowNodePlan(
                name="retry",
                type=node.type,
                kind=node.kind,
                input_contracts=node.input_contracts,
                output_contracts=node.output_contracts,
                capabilities=node.capabilities,
                mappings=node.mappings,
                loop=loop,
            ),
        ),
    )


if __name__ == "__main__":
    main()
