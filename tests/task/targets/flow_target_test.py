from asyncio import CancelledError, sleep, wait_for
from asyncio import run as asyncio_run
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentText,
    MessageRole,
)
from avalan.event import Event, EventType
from avalan.flow import FlowNodeDefinition
from avalan.flow.flow import Flow
from avalan.flow.loader import FlowDefinitionLoader
from avalan.flow.node import Node
from avalan.flow.registry import FlowNodeConfigurationError
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
    TaskTargetContext,
    TaskValidationCategory,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
    pdf_image_converter_capability,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.store import TaskExecutionContext
from avalan.task.stores import InMemoryTaskStore
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
) -> Callable[[TaskTargetContext], Flow]:
    def resolve(context: TaskTargetContext) -> Flow:
        result = FlowDefinitionLoader(
            registry=task_flow_node_registry(
                context,
                agent_runner=agent_runner,
                execution_roots=(root,),
            )
        ).load_result(path)
        assert result.flow is not None, result.issues
        return result.flow

    return resolve


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
    def test_task_scoped_registry_loads_file_convert_toml_shape(self) -> None:
        converter = RecordingPdfPageConverter((_page_result(1, b"page"),))
        context = self._context(file_converters={"pdf_image": converter})

        result = FlowDefinitionLoader(
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

        self.assertTrue(result.ok)
        assert result.definition is not None
        node = result.definition.node_map["render_pages"]
        self.assertEqual(node.config["converter"], "pdf_image")
        self.assertIsNotNone(result.flow)

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
                result = FlowDefinitionLoader(
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
                "flow.invalid_node",
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
                "flow.invalid_node",
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
                result = FlowDefinitionLoader(
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

        result = FlowDefinitionLoader(
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
            result = FlowDefinitionLoader(
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
        artifact_store: object | None = None,
        task_store: object | None = None,
        file_converters: Mapping[str, object] | None = None,
    ) -> TaskTargetContext:
        return TaskTargetContext(
            definition=definition or self._context_definition(),
            execution=TaskExecutionContext(
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


if __name__ == "__main__":
    main()
