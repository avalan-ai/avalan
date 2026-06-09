from argparse import Namespace
from base64 import b64decode
from collections.abc import Callable, Iterator, Mapping
from contextlib import AsyncExitStack, contextmanager
from datetime import UTC, datetime
from io import StringIO
from json import dumps
from json import load as load_json
from os import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rich.console import Console

from avalan.cli.commands import task as task_cmds
from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
)
from avalan.task import (
    ArtifactStoreError,
    CallableTaskTargetRunner,
    SanitizedTaskEvent,
    TaskClient,
    TaskClientUnsupportedOperationError,
    TaskClientWaitTimeoutError,
    TaskDefinition,
    TaskEventCategory,
    TaskExecutionContext,
    TaskExecutionTarget,
    TaskInputContract,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskRetentionAction,
    TaskRetentionStoreNotFoundError,
    TaskRunState,
    TaskStoreNotFoundError,
    TaskTargetContext,
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    UsageSource,
    canonical_schema_json,
)
from avalan.task import client as task_client_module
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.converters import (
    TaskFileConversionError,
    TaskFileConversionPageCollection,
    TaskFileConversionPageResult,
    TaskFileConversionResult,
    TaskFileConverterCapability,
)
from avalan.task.converters.pdf_image import pdf_image_converter_capability
from avalan.task.stores import InMemoryTaskStore
from avalan.task.targets import AgentTaskTargetRunner

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "task" / "fixtures"
TASK_HMAC_ENV = {
    "AVALAN_TASK_HMAC_KEY_ID": "cli-test-v1",
    "AVALAN_TASK_HMAC_KEY_B64": "dGFzay1obWFjLXRlc3Qta2V5",
}


def _extraction_cli_usage_fixture() -> Mapping[str, object]:
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


def _extraction_cli_usage_payload() -> Mapping[str, object]:
    usage = _extraction_cli_usage_fixture()["usage"]
    assert isinstance(usage, Mapping)
    return usage


def _extraction_cli_provider_family() -> str:
    provider_family = _extraction_cli_usage_fixture()["provider_family"]
    assert isinstance(provider_family, str)
    return provider_family


class _FakeTaskClient:
    def __init__(
        self,
        *,
        run_result: object | None = None,
        enqueue_result: object | None = None,
        wait_result: object | None = None,
        inspect_result: object | None = None,
        usage_result: object | None = None,
        output_result: object | None = None,
        events_result: tuple[object, ...] = (),
        artifacts_result: tuple[object, ...] = (),
        run_error: BaseException | None = None,
        enqueue_error: BaseException | None = None,
        wait_error: BaseException | None = None,
        inspect_error: BaseException | None = None,
        usage_error: BaseException | None = None,
        output_error: BaseException | None = None,
        events_error: BaseException | None = None,
        artifacts_error: BaseException | None = None,
    ) -> None:
        self.run_result = run_result
        self.enqueue_result = enqueue_result
        self.wait_result = wait_result
        self.inspect_result = inspect_result
        self.usage_result = usage_result
        self.output_result = output_result
        self.events_result = events_result
        self.artifacts_result = artifacts_result
        self.run_error = run_error
        self.enqueue_error = enqueue_error
        self.wait_error = wait_error
        self.inspect_error = inspect_error
        self.usage_error = usage_error
        self.output_error = output_error
        self.events_error = events_error
        self.artifacts_error = artifacts_error
        self.input_value: object = None
        self.queue_name: str | None = None
        self.queue_metadata: object = None
        self.wait_timeout: float | None = None
        self.poll_interval: float | None = None
        self.after_sequence: int | None = None
        self.attempt_id: str | None = None
        self.source: object | None = None

    async def run(
        self,
        definition: object,
        *,
        input_value: object = None,
        metadata: object | None = None,
    ) -> object:
        if self.run_error is not None:
            raise self.run_error
        self.input_value = input_value
        return self.run_result

    async def enqueue(
        self,
        definition: object,
        *,
        input_value: object = None,
        queue_name: str | None = None,
        queue_metadata: object | None = None,
    ) -> object:
        if self.enqueue_error is not None:
            raise self.enqueue_error
        self.input_value = input_value
        self.queue_name = queue_name
        self.queue_metadata = queue_metadata
        return self.enqueue_result

    async def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> object:
        if self.wait_error is not None:
            raise self.wait_error
        self.wait_timeout = timeout_seconds
        self.poll_interval = poll_interval_seconds
        return self.wait_result

    async def inspect(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> object:
        _ = run_id
        if self.inspect_error is not None:
            raise self.inspect_error
        self.after_sequence = after_sequence
        return self.inspect_result

    async def output(self, run_id: str) -> object:
        _ = run_id
        if self.output_error is not None:
            raise self.output_error
        return self.output_result

    async def usage_inspection(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        source: object | None = None,
    ) -> object:
        _ = run_id
        if self.usage_error is not None:
            raise self.usage_error
        self.attempt_id = attempt_id
        self.source = source
        return self.usage_result

    async def events(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        after_sequence: int | None = None,
    ) -> tuple[object, ...]:
        _ = run_id
        if self.events_error is not None:
            raise self.events_error
        self.attempt_id = attempt_id
        self.after_sequence = after_sequence
        return self.events_result

    async def artifacts(self, run_id: str) -> tuple[object, ...]:
        _ = run_id
        if self.artifacts_error is not None:
            raise self.artifacts_error
        return self.artifacts_result


class _FakeTaskClientContext:
    def __init__(self, client: _FakeTaskClient) -> None:
        self.client = client

    async def __aenter__(self) -> _FakeTaskClient:
        return self.client

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        return None


class _Snapshot:
    def __init__(self, value: object) -> None:
        self.value = value

    def as_dict(self) -> object:
        return self.value


class _FakeResource:
    def __init__(
        self,
        *,
        open_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.open_error = open_error
        self.close_error = close_error
        self.entered = False
        self.exited = False
        self.opened = False
        self.closed = False

    async def __aenter__(self) -> "_FakeResource":
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        self.exited = True
        return None

    async def open(self) -> None:
        self.opened = True
        if self.open_error is not None:
            raise self.open_error

    async def aclose(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _FakeTaskWorker:
    instances: list["_FakeTaskWorker"] = []
    results: list[object] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls = 0
        self.instances.append(self)

    async def process_once(self) -> object:
        self.calls += 1
        if self.results:
            result = self.results.pop(0)
            if callable(result):
                return result(self)
            return result
        return SimpleNamespace(processed=False)


class _FakeRetentionService:
    instances: list["_FakeRetentionService"] = []
    results: tuple[object, ...] = ()
    error: BaseException | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.purposes: object = None
        self.limit: int | None = None
        self.instances.append(self)

    async def sweep_expired(
        self,
        *,
        purposes: object = None,
        limit: int = 100,
    ) -> object:
        if self.error is not None:
            raise self.error
        self.purposes = purposes
        self.limit = limit
        return SimpleNamespace(limit=limit, results=self.results)


class _ExtractionCliResponse:
    input_token_count = 19
    output_token_count = 23
    total_token_count = 42

    def __init__(self, output: Mapping[str, object]) -> None:
        self.output = output

    @property
    def provider_family(self) -> str:
        return _extraction_cli_provider_family()

    @property
    def usage(self) -> Mapping[str, object]:
        return _extraction_cli_usage_payload()

    async def to_json(self) -> str:
        return dumps(self.output, sort_keys=True, separators=(",", ":"))

    async def to_str(self) -> str:
        return await self.to_json()


class _ExtractionCliOrchestrator:
    def __init__(self, output: Mapping[str, object]) -> None:
        self.output = output
        self.inputs: list[object] = []
        self.text_formats: list[Mapping[str, object]] = []
        self.reasoning_options: list[Mapping[str, object]] = []

    async def __aenter__(self) -> "_ExtractionCliOrchestrator":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        return None

    async def __call__(self, input: object) -> _ExtractionCliResponse:
        self.inputs.append(input)
        return _ExtractionCliResponse(self.output)


class _ExtractionCliLoader:
    def __init__(self, orchestrator: _ExtractionCliOrchestrator) -> None:
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
    ) -> _ExtractionCliOrchestrator:
        _ = agent_id, disable_memory, uri, tool_settings
        self.paths.append(path)
        return self.orchestrator


class _CliPdfPageConverter:
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


def _extraction_cli_output() -> dict[str, object]:
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


def _cli_page_result(
    page_index: int,
    page_count: int,
    content: bytes,
) -> TaskFileConversionPageResult:
    return TaskFileConversionPageResult(
        page_index=page_index,
        page_count=page_count,
        content=content,
        media_type="image/png",
        width_pixels=10,
        height_pixels=10,
        metadata={"page": page_index},
    )


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    chdir(path)
    try:
        yield
    finally:
        chdir(previous)


class CliTaskValidateTestCase(TestCase):
    def setUp(self) -> None:
        self.theme = MagicMock()

    def test_validate_prints_success_for_valid_definition(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
            result = task_cmds.task_validate(
                Namespace(definition=str(FIXTURE_ROOT / "minimal.task.toml")),
                console,
                self.theme,
            )

        self.assertTrue(result)
        self.assertIn(
            "Task definition is valid: person_explainer 1",
            console.export_text(),
        )

    def test_validate_reports_missing_hmac_key(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_validate(
                Namespace(definition=str(FIXTURE_ROOT / "minimal.task.toml")),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("privacy.hmac_key_missing", output)
        self.assertNotIn("Ada Lovelace", output)

    def test_validate_prints_load_issues(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_validate(
            Namespace(
                definition=str(FIXTURE_ROOT / "missing_sections.task.toml")
            ),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition could not be loaded.", output)
        self.assertIn("task.missing_section", output)
        self.assertIn("input", output)

    def test_validate_prints_validation_issues(self) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "bad.task.toml"
            definition.write_text(
                """
                [task]
                name = "bad"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "object"
                schema = {type = "array"}

                [execution]
                type = "flow"
                ref = "flows/private.toml"
                """,
                encoding="utf-8",
            )

            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                result = task_cmds.task_validate(
                    Namespace(definition=str(definition)),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition is invalid.", output)
        self.assertIn("output.invalid_schema", output)
        self.assertNotIn("feature.flow_backed_tasks_disabled", output)
        self.assertNotIn("flows/private.toml", output)

    def test_validate_reports_flow_load_issues(self) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            definition = root / "task.toml"
            definition.write_text(
                """
                [task]
                name = "flow-task"
                version = "1"

                [input]
                type = "file"
                mime_types = ["application/pdf"]

                [output]
                type = "object"
                schema = {type = "object", additionalProperties = true}

                [execution]
                type = "flow"
                ref = "flow.toml"
                """,
                encoding="utf-8",
            )
            (root / "flow.toml").write_text(
                """
                [flow]
                name = "render"
                entrypoint = "render"
                output_node = "render"

                [nodes.render]
                type = "file_convert"

                [nodes.render.config]
                converter = "private_converter"
                """,
                encoding="utf-8",
            )

            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                result = task_cmds.task_validate(
                    Namespace(definition=str(definition)),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition is invalid.", output)
        self.assertIn("flow.converter_unsupported", output)
        self.assertNotIn("private_converter", output)

    def test_validate_reports_graph_load_issues_without_runtime_build(
        self,
    ) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            definition = root / "task.toml"
            definition.write_text(
                """
                [task]
                name = "flow-task"
                version = "1"

                [input]
                type = "object"
                schema = {type = "object", additionalProperties = true}

                [output]
                type = "object"
                schema = {type = "object", additionalProperties = true}

                [execution]
                type = "flow"
                ref = "flow.toml"
                """,
                encoding="utf-8",
            )
            (root / "flow.toml").write_text(
                """
                [flow]
                name = "graph-task-flow"
                version = "1"

                [[inputs]]
                name = "payload"
                type = "object"

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
                source = "inline"
                mode = "executable"
                diagram = '''
                flowchart LR
                start -->|Private customer route| finish
                '''

                [nodes.start]
                type = "constant"
                value = "ok"

                [nodes.finish]
                type = "echo"
                """,
                encoding="utf-8",
            )

            with (
                patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
                patch(
                    "avalan.flow.loader.build_flow",
                    side_effect=AssertionError("runtime build not expected"),
                ),
            ):
                result = task_cmds.task_validate(
                    Namespace(definition=str(definition)),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition is invalid.", output)
        self.assertIn("flow.graph.unsupported_executable_edge", output)
        self.assertNotIn("Private customer route", output)

    def test_validate_reports_graph_flow_node_issues_without_runtime_build(
        self,
    ) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            definition = root / "task.toml"
            definition.write_text(
                """
                [task]
                name = "flow-task"
                version = "1"

                [input]
                type = "file"
                mime_types = ["application/pdf"]

                [output]
                type = "object"
                schema = {type = "object", additionalProperties = true}

                [execution]
                type = "flow"
                ref = "flow.toml"
                """,
                encoding="utf-8",
            )
            (root / "flow.toml").write_text(
                """
                [flow]
                name = "graph-render"
                entrypoint = "render"
                output_node = "render"

                [graph]
                format = "mermaid"
                source = "inline"
                mode = "executable"
                diagram = '''
                flowchart LR
                render
                '''

                [nodes.render]
                type = "file_convert"

                [nodes.render.config]
                converter = "private_converter"
                """,
                encoding="utf-8",
            )

            with (
                patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
                patch(
                    "avalan.flow.loader.build_flow",
                    side_effect=AssertionError("runtime build not expected"),
                ) as build_flow,
            ):
                result = task_cmds.task_validate(
                    Namespace(definition=str(definition)),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition is invalid.", output)
        self.assertIn("flow.converter_unsupported", output)
        self.assertNotIn("private_converter", output)
        build_flow.assert_not_called()

    def test_validate_checks_readable_flow_reference(self) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            definition = root / "task.toml"
            definition.write_text(
                """
                [task]
                name = "flow-task"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "object"
                schema = {type = "object", additionalProperties = true}

                [execution]
                type = "flow"
                ref = "flow.toml"
                """,
                encoding="utf-8",
            )
            (root / "flow.toml").write_text(
                """
                [flow]
                name = "constant"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "constant"
                value = {ok = true}
                """,
                encoding="utf-8",
            )

            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                result = task_cmds.task_validate(
                    Namespace(definition=str(definition)),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("Task definition is valid: flow-task 1", output)

    def test_validate_uses_task_registry_for_agent_flow_node(self) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            definition = root / "task.toml"
            definition.write_text(
                """
                [task]
                name = "flow-agent-task"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "object"
                schema = {type = "object", additionalProperties = true}

                [execution]
                type = "flow"
                ref = "flow.toml"
                """,
                encoding="utf-8",
            )
            (root / "flow.toml").write_text(
                """
                [flow]
                name = "extract"
                entrypoint = "extract"
                output_node = "extract"

                [nodes.extract]
                type = "agent"
                ref = "agent.toml"
                input = "value"
                output = "extraction"
                """,
                encoding="utf-8",
            )

            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                result = task_cmds.task_validate(
                    Namespace(definition=str(definition)),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("Task definition is valid: flow-agent-task 1", output)
        self.assertNotIn("flow.unsupported_node_type", output)

    def test_validate_reports_missing_flow_reference_safely(self) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            definition = root / "task.toml"
            definition.write_text(
                """
                [task]
                name = "flow-task"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "object"
                schema = {type = "object", additionalProperties = true}

                [execution]
                type = "flow"
                ref = "private-flow.toml"
                """,
                encoding="utf-8",
            )

            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                result = task_cmds.task_validate(
                    Namespace(definition=str(definition)),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("flow.read_failed", output)
        self.assertNotIn("private-flow.toml", output)

    def test_validate_missing_file_prints_safe_diagnostic(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_validate(
            Namespace(definition="/tmp/private/missing.task.toml"),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition could not be read.", output)
        self.assertIn("file.read", output)
        self.assertNotIn("/tmp/private/missing.task.toml", output)


class CliTaskCommandShellTestCase(TestCase):
    def setUp(self) -> None:
        self.theme = MagicMock()

    def test_inspection_commands_require_durable_store(self) -> None:
        commands = [
            task_cmds.task_artifacts,
            task_cmds.task_events,
            task_cmds.task_inspect,
            task_cmds.task_output,
            task_cmds.task_usage,
        ]

        for command in commands:
            console = Console(record=True, width=160)
            with (
                self.subTest(command=command.__name__),
                patch.dict(task_cmds.environ, {}, clear=True),
            ):
                result = command(
                    Namespace(
                        run_id="run-private",
                        store_dsn=None,
                        store_schema=None,
                        attempt_id=None,
                        after_sequence=None,
                    ),
                    console,
                    self.theme,
                )

            output = console.export_text()
            self.assertFalse(result)
            self.assertIn("store.missing", output)
            self.assertNotIn("run-private", output)

    def test_inspection_commands_print_stable_snapshots(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        client = _FakeTaskClient(
            inspect_result=_Snapshot(
                {
                    "run": {
                        "run_id": "run-1",
                        "state": "succeeded",
                        "input_summary": {"privacy": "<redacted>"},
                    },
                    "events": (),
                }
            ),
            usage_result=_Snapshot(
                {
                    "usage": (
                        {
                            "usage_id": "usage-1",
                            "run_id": "run-1",
                            "attempt_id": "attempt-1",
                            "sequence": 1,
                            "source": "exact",
                            "totals": {
                                "input_tokens": 1,
                                "cached_input_tokens": 0,
                                "cache_creation_input_tokens": None,
                                "output_tokens": 2,
                                "reasoning_tokens": None,
                                "total_tokens": 3,
                            },
                            "metadata": {"provider_family": "openai"},
                        },
                    ),
                    "usage_totals": {
                        "input_tokens": 1,
                        "cached_input_tokens": 0,
                        "cache_creation_input_tokens": None,
                        "output_tokens": 2,
                        "reasoning_tokens": None,
                        "total_tokens": 3,
                    },
                }
            ),
            output_result=_Snapshot(
                {
                    "run_id": "run-1",
                    "state": "failed",
                    "ready": False,
                    "error": {"code": "runnable.failed"},
                }
            ),
            events_result=(
                SanitizedTaskEvent(
                    event_id="event-1",
                    run_id="run-1",
                    sequence=2,
                    event_type="token_generated",
                    category=TaskEventCategory.TOKEN,
                    created_at=now,
                    attempt_id="attempt-1",
                    payload={"privacy": "<redacted>"},
                ),
            ),
            artifacts_result=(
                {
                    "artifact_id": "artifact-1",
                    "state": "ready",
                    "ref": {
                        "artifact_id": "artifact-1",
                        "store": "local",
                    },
                },
            ),
        )

        with patch.object(
            task_cmds,
            "_task_cli_inspection_client_context",
            return_value=_FakeTaskClientContext(client),
        ):
            inspect_console = Console(record=True, width=200)
            inspect_result = task_cmds.task_inspect(
                Namespace(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    after_sequence=1,
                ),
                inspect_console,
                self.theme,
            )
            output_console = Console(record=True, width=200)
            output_result = task_cmds.task_output(
                Namespace(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                ),
                output_console,
                self.theme,
            )
            usage_console = Console(record=True, width=200)
            usage_result = task_cmds.task_usage(
                Namespace(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    attempt_id="attempt-1",
                    source="exact",
                ),
                usage_console,
                self.theme,
            )
            events_console = Console(record=True, width=200)
            events_result = task_cmds.task_events(
                Namespace(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    attempt_id="attempt-1",
                    after_sequence=1,
                ),
                events_console,
                self.theme,
            )
            artifacts_console = Console(record=True, width=200)
            artifacts_result = task_cmds.task_artifacts(
                Namespace(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                ),
                artifacts_console,
                self.theme,
            )

        self.assertTrue(inspect_result)
        self.assertTrue(output_result)
        self.assertTrue(usage_result)
        self.assertTrue(events_result)
        self.assertTrue(artifacts_result)
        self.assertEqual(client.after_sequence, 1)
        self.assertEqual(client.attempt_id, "attempt-1")
        self.assertEqual(client.source, UsageSource.EXACT)
        rendered = (
            inspect_console.export_text()
            + output_console.export_text()
            + usage_console.export_text()
            + events_console.export_text()
            + artifacts_console.export_text()
        )
        self.assertIn("inspect", rendered)
        self.assertIn("usage", rendered)
        self.assertIn('"input_summary":{"privacy":"<redacted>"}', rendered)
        self.assertIn('"cached_input_tokens":0', rendered)
        self.assertIn('"provider_family":"openai"', rendered)
        self.assertIn('"error":{"code":"runnable.failed"}', rendered)
        self.assertIn('"sequence":2', rendered)
        self.assertIn('"artifact_id":"artifact-1"', rendered)
        self.assertNotIn("secret", rendered)
        self.assertNotIn("token_id", rendered)

    def test_inspection_commands_report_not_found_safely(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            output_error=TaskStoreNotFoundError("private run secret")
        )

        with patch.object(
            task_cmds,
            "_task_cli_inspection_client_context",
            return_value=_FakeTaskClientContext(client),
        ):
            result = task_cmds.task_output(
                Namespace(
                    run_id="run-private",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("task.not_found", output)
        self.assertNotIn("private run secret", output)
        self.assertNotIn("run-private", output)

    def test_usage_rejects_invalid_source_safely(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(usage_result=_Snapshot({}))

        with patch.object(
            task_cmds,
            "_task_cli_inspection_client_context",
            return_value=_FakeTaskClientContext(client),
        ):
            result = task_cmds.task_usage(
                Namespace(
                    run_id="run-private",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    attempt_id=None,
                    source="raw-provider",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("task.inspection", output)
        self.assertNotIn("raw-provider", output)
        self.assertIsNone(client.source)

    def test_inspection_commands_report_safe_errors(self) -> None:
        cases = (
            (
                task_cmds.task_inspect,
                _FakeTaskClient(inspect_error=ImportError("private")),
                "dependency.missing",
                Namespace(
                    run_id="run-private",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    after_sequence=None,
                ),
            ),
            (
                task_cmds.task_events,
                _FakeTaskClient(events_error=OSError("private")),
                "io.failure",
                Namespace(
                    run_id="run-private",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    attempt_id=None,
                    after_sequence=None,
                ),
            ),
            (
                task_cmds.task_usage,
                _FakeTaskClient(usage_error=AssertionError("private")),
                "task.inspection",
                Namespace(
                    run_id="run-private",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    attempt_id=None,
                    source=None,
                ),
            ),
            (
                task_cmds.task_artifacts,
                _FakeTaskClient(artifacts_error=AssertionError("private")),
                "task.inspection",
                Namespace(
                    run_id="run-private",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                ),
            ),
        )
        for command, client, expected, args in cases:
            console = Console(record=True, width=160)
            with (
                self.subTest(command=command.__name__),
                patch.object(
                    task_cmds,
                    "_task_cli_inspection_client_context",
                    return_value=_FakeTaskClientContext(client),
                ),
            ):
                result = command(args, console, self.theme)

            output = console.export_text()
            self.assertFalse(result)
            self.assertIn(expected, output)
            self.assertNotIn("private", output)

    def test_run_requires_store_without_ephemeral(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_run(
                Namespace(
                    definition=str(FIXTURE_ROOT / "minimal.task.toml"),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task store is not configured.", output)
        self.assertIn("store.missing", output)

    def test_worker_requires_durable_store(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_worker(
                Namespace(
                    queue="default",
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("store.missing", console.export_text())

    def test_worker_reports_missing_dependency_gate(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "require_feature",
            return_value=(
                SimpleNamespace(
                    code="dependency.task_worker_pgsql_missing",
                    message="Task queue workers require the task-pgsql extra.",
                    hint="Install avalan[task-pgsql] before starting workers.",
                ),
            ),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="default",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("dependency.task_worker_pgsql_missing", output)
        self.assertIn("avalan[task-pgsql]", output)
        self.assertNotIn("postgresql://db/tasks", output)

    def test_run_rejects_queued_definition(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "queued.task.toml"
            _write_queued_definition(definition)

            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn(
            "Task run requires a direct-mode definition.",
            console.export_text(),
        )

    def test_run_reports_client_error(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(run_error=ImportError("missing private dep"))

        with patch.object(
            task_cmds,
            "_task_cli_client_context",
            return_value=_FakeTaskClientContext(client),
        ):
            result = task_cmds.task_run(
                Namespace(
                    definition=str(FIXTURE_ROOT / "minimal.task.toml"),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("dependency.missing", output)

    def test_run_uses_client_and_prints_sanitized_output(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.SUCCEEDED,
                    result=SimpleNamespace(
                        output_summary={"privacy": "<redacted>"}
                    ),
                )
            )
        )

        with patch.object(
            task_cmds,
            "_task_cli_client_context",
            return_value=_FakeTaskClientContext(client),
        ):
            result = task_cmds.task_run(
                Namespace(
                    definition=str(FIXTURE_ROOT / "minimal.task.toml"),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertEqual(client.input_value, "Ada Lovelace")
        self.assertIn("Task run completed (non-durable): run-1", output)
        self.assertIn('"privacy":"<redacted>"', output)

    def test_run_json_prints_only_structured_output(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.SUCCEEDED,
                    result=SimpleNamespace(output_summary={"ignored": True}),
                ),
                output={"b": 2, "a": [1]},
            )
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
        ):
            definition = _write_direct_object_definition(Path(tmpdir))
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=True,
                    task_output_path=None,
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), '{"a":[1],"b":2}\n')

    def test_run_json_and_output_write_same_structured_value(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.SUCCEEDED,
                    result=SimpleNamespace(output_summary={"ignored": True}),
                ),
                output={"answer": "ok"},
            )
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
        ):
            root = Path(tmpdir)
            definition = _write_direct_object_definition(root)
            output_path = root / "result.json"
            output_path.write_text("old\n", encoding="utf-8")
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=True,
                    task_output_path=str(output_path),
                    task_pdf=None,
                    quiet=True,
                ),
                console,
                self.theme,
            )
            written = output_path.read_text(encoding="utf-8")

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), '{"answer":"ok"}\n')
        self.assertEqual(written, '{"answer":"ok"}\n')

    def test_run_output_parent_failure_skips_client(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient()

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
        ):
            root = Path(tmpdir)
            definition = _write_direct_object_definition(root)
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=False,
                    task_output_path=str(root / "missing" / "result.json"),
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIsNone(client.input_value)
        self.assertIn("output.write", output)
        self.assertNotIn("Ada Lovelace", output)

    def test_run_returns_false_when_output_write_fails(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.SUCCEEDED,
                    result=SimpleNamespace(output_summary={"ignored": True}),
                ),
                output={"answer": "ok"},
            )
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
            patch.object(
                task_cmds,
                "_write_task_run_structured_output",
                return_value=False,
            ),
        ):
            definition = _write_direct_object_definition(Path(tmpdir))
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=False,
                    task_output_path="result.json",
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertEqual(client.input_value, "Ada Lovelace")

    def test_run_json_failure_does_not_write_stdout(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        diagnostic_console = Console(record=True, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.FAILED,
                    result=SimpleNamespace(
                        output_summary=None,
                        error={"code": "output_contract.failed"},
                    ),
                ),
                output={"private": "partial"},
            )
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
            patch.object(
                task_cmds,
                "_task_diagnostic_console",
                return_value=diagnostic_console,
            ),
        ):
            definition = _write_direct_object_definition(Path(tmpdir))
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=True,
                    task_output_path=None,
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )

        diagnostics = diagnostic_console.export_text()
        self.assertFalse(result)
        self.assertEqual(stream.getvalue(), "")
        self.assertIn("task.run_failed", diagnostics)
        self.assertIn("output_contract.failed", diagnostics)
        self.assertNotIn("partial", diagnostics)

    def test_run_json_failure_codes_are_sanitized(self) -> None:
        cases = (
            ("provider.structured_output_failed", "provider"),
            ("output.parse_failed", "output_contract"),
            ("output_contract.failed", "output_contract"),
        )

        for code, category in cases:
            with self.subTest(code=code):
                stream = StringIO()
                console = Console(file=stream, width=160)
                diagnostic_console = Console(record=True, width=160)
                client = _FakeTaskClient(
                    run_result=SimpleNamespace(
                        run=SimpleNamespace(
                            run_id="run-1",
                            state=TaskRunState.FAILED,
                            result=SimpleNamespace(
                                output_summary=None,
                                error={
                                    "category": category,
                                    "code": code,
                                    "message": "safe failure summary",
                                },
                            ),
                        ),
                        output={"private": "partial provider body"},
                    )
                )

                with (
                    TemporaryDirectory() as tmpdir,
                    patch.object(
                        task_cmds,
                        "_task_cli_client_context",
                        return_value=_FakeTaskClientContext(client),
                    ),
                    patch.object(
                        task_cmds,
                        "_task_diagnostic_console",
                        return_value=diagnostic_console,
                    ),
                ):
                    definition = _write_direct_object_definition(Path(tmpdir))
                    result = task_cmds.task_run(
                        Namespace(
                            definition=str(definition),
                            task_input="Ada Lovelace",
                            task_input_json=None,
                            task_input_fields=(),
                            task_files=(),
                            store_dsn=None,
                            store_schema=None,
                            ephemeral=True,
                            task_run_json=True,
                            task_output_path=None,
                            task_pdf=None,
                            quiet=False,
                        ),
                        console,
                        self.theme,
                    )

                diagnostics = diagnostic_console.export_text()
                self.assertFalse(result)
                self.assertEqual(stream.getvalue(), "")
                self.assertIn(code, diagnostics)
                self.assertNotIn("partial provider body", diagnostics)
                self.assertNotIn("Ada Lovelace", diagnostics)

    def test_run_quiet_failure_suppresses_summary(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.FAILED,
                    result=SimpleNamespace(
                        output_summary=None,
                        error={
                            "category": "output_contract",
                            "code": "output.parse_failed",
                            "message": "safe failure summary",
                        },
                    ),
                ),
                output={"private": "partial provider body"},
            )
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ),
        ):
            definition = _write_direct_object_definition(Path(tmpdir))
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=False,
                    task_output_path=None,
                    task_pdf=None,
                    quiet=True,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("output.parse_failed", output)
        self.assertNotIn("Task run completed", output)
        self.assertNotIn("partial provider body", output)
        self.assertNotIn("Ada Lovelace", output)

    def test_run_pdf_missing_file_reports_safe_diagnostic(self) -> None:
        console = Console(record=True, width=160)

        async def target(context: TaskTargetContext) -> object:
            _ = context
            raise AssertionError("target should not run")

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_agent_task_target",
                return_value=CallableTaskTargetRunner(target),
            ),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            root = Path(tmpdir)
            definition = root / "missing_pdf.task.toml"
            definition.write_text(
                """
                [task]
                name = "missing_pdf"
                version = "1"

                [input]
                type = "file"
                mime_types = ["application/pdf"]

                [output]
                type = "text"

                [execution]
                type = "agent"
                ref = "agent.toml"
                """,
                encoding="utf-8",
            )
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=False,
                    task_output_path=None,
                    task_pdf="/tmp/private/customer-secret.pdf",
                    quiet=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task run did not succeed.", output)
        self.assertIn("input_contract.failed", output)
        self.assertNotIn("customer-secret.pdf", output)
        self.assertNotIn("/tmp/private", output)

    def test_run_json_rejects_text_output_contract(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        diagnostic_console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "_task_diagnostic_console",
            return_value=diagnostic_console,
        ):
            result = task_cmds.task_run(
                Namespace(
                    definition=str(FIXTURE_ROOT / "minimal.task.toml"),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=True,
                    task_output_path=None,
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertEqual(stream.getvalue(), "")
        self.assertIn("output.unsupported", diagnostic_console.export_text())

    def test_run_json_uses_real_ephemeral_client(self) -> None:
        async def target(context: TaskTargetContext) -> object:
            return {"answer": "ok", "input": context.input_value}

        stream = StringIO()
        console = Console(file=stream, width=160)

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds,
                "_agent_task_target",
                return_value=CallableTaskTargetRunner(target),
            ),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            root = Path(tmpdir)
            definition = _write_direct_object_definition(root)
            output_path = root / "result.json"
            result = task_cmds.task_run(
                Namespace(
                    definition=str(definition),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=True,
                    task_output_path=str(output_path),
                    task_pdf=None,
                    quiet=False,
                ),
                console,
                self.theme,
            )
            written = output_path.read_text(encoding="utf-8")

        expected = '{"answer":"ok","input":"Ada Lovelace"}\n'
        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), expected)
        self.assertEqual(written, expected)

    def test_run_poc_extraction_fixture_reaches_fake_provider(self) -> None:
        repo_root = Path(__file__).parents[2]
        fixture = repo_root / "docs" / "examples" / "tasks" / "poc_extraction"
        pdf_bytes = (fixture / "sample.pdf").read_bytes()
        output = _extraction_cli_output()
        expected = dumps(output, sort_keys=True, separators=(",", ":")) + "\n"
        cases = (
            (
                "repo_root_pdf",
                repo_root,
                "docs/examples/tasks/poc_extraction/task.toml",
                {
                    "task_pdf": (
                        "docs/examples/tasks/poc_extraction/sample.pdf"
                    ),
                    "task_files": (),
                    "task_file_mime_types": (),
                },
            ),
            (
                "repo_root_file",
                repo_root,
                "docs/examples/tasks/poc_extraction/task.toml",
                {
                    "task_pdf": None,
                    "task_files": (
                        "input=docs/examples/tasks/poc_extraction/sample.pdf",
                    ),
                    "task_file_mime_types": ("input=application/pdf",),
                },
            ),
            (
                "example_directory_pdf",
                fixture,
                "task.toml",
                {
                    "task_pdf": "sample.pdf",
                    "task_files": (),
                    "task_file_mime_types": (),
                },
            ),
        )

        for name, cwd, definition, input_args in cases:
            with self.subTest(command=name):
                stream = StringIO()
                console = Console(file=stream, width=160)
                orchestrator = _ExtractionCliOrchestrator(output)
                settings_values: list[Any] = []

                async def from_settings(
                    loader: object,
                    settings: object,
                    *,
                    tool_settings: object | None = None,
                    tool_format: object | None = None,
                ) -> _ExtractionCliOrchestrator:
                    _ = loader, tool_settings, tool_format
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

                with (
                    TemporaryDirectory() as tmpdir,
                    patch.object(
                        task_cmds.OrchestratorLoader,
                        "from_settings",
                        new=from_settings,
                    ),
                    patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
                ):
                    output_path = Path(tmpdir) / "extraction.json"
                    with _working_directory(cwd):
                        result = task_cmds.task_run(
                            Namespace(
                                definition=definition,
                                task_input=None,
                                task_input_json=None,
                                task_input_fields=(),
                                task_file_descriptors=(),
                                task_provider_file_ids=(),
                                task_hosted_urls=(),
                                task_object_store_uris=(),
                                task_file_roles=(),
                                task_file_sizes=(),
                                task_file_sha256=(),
                                task_file_conversions=(),
                                store_dsn=None,
                                store_schema=None,
                                ephemeral=True,
                                task_run_json=True,
                                task_output_path=str(output_path),
                                quiet=True,
                                **input_args,
                            ),
                            console,
                            self.theme,
                        )
                    written = output_path.read_text(encoding="utf-8")

                self.assertTrue(result)
                self.assertEqual(stream.getvalue(), expected)
                self.assertEqual(written, expected)
                self.assertEqual(len(settings_values), 1)
                settings = settings_values[0]
                agent_config = settings.agent_config
                self.assertIsInstance(agent_config, Mapping)
                self.assertIn("instructions", agent_config)
                self.assertNotIn("system", agent_config)
                self.assertNotIn("task", agent_config)
                self.assertEqual(settings.tools, [])
                call_options = settings.call_options
                self.assertIsInstance(call_options, Mapping)
                self.assertNotIn("tools", call_options)
                self.assertNotIn("tool_choice", call_options)
                self.assertEqual(
                    orchestrator.reasoning_options,
                    [{"effort": "high"}],
                )
                self.assertEqual(
                    canonical_schema_json(
                        orchestrator.text_formats[0]["schema"]
                    ),
                    canonical_schema_json(
                        task_cmds.TaskDefinitionLoader()
                        .load(fixture / "task.toml")
                        .output.schema
                    ),
                )
                self.assertEqual(len(orchestrator.inputs), 1)
                agent_input = orchestrator.inputs[0]
                self.assertIsInstance(agent_input, Message)
                content = cast(Message, agent_input).content
                self.assertIsInstance(content, list)
                blocks = cast(list[Any], content)
                text_blocks = [
                    block
                    for block in blocks
                    if isinstance(block, MessageContentText)
                ]
                file_blocks = [
                    block
                    for block in blocks
                    if isinstance(block, MessageContentFile)
                ]
                self.assertEqual(len(text_blocks), 1)
                self.assertIn(
                    "Analyze the attached synthetic invoice PDF",
                    text_blocks[0].text,
                )
                self.assertEqual(len(file_blocks), 1)
                self.assertEqual(
                    file_blocks[0].file["mime_type"],
                    "application/pdf",
                )
                self.assertEqual(
                    b64decode(cast(str, file_blocks[0].file["file_data"])),
                    pdf_bytes,
                )

    def test_run_file_and_output_paths_use_caller_cwd(self) -> None:
        fixture = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
        )
        pdf_bytes = b"%PDF-cwd-private\n"
        output = _extraction_cli_output()
        expected = dumps(output, sort_keys=True, separators=(",", ":")) + "\n"
        stream = StringIO()
        console = Console(file=stream, width=160)
        orchestrator = _ExtractionCliOrchestrator(output)

        async def from_settings(
            loader: object,
            settings: object,
            *,
            tool_settings: object | None = None,
            tool_format: object | None = None,
        ) -> _ExtractionCliOrchestrator:
            _ = loader, settings, tool_settings, tool_format
            return orchestrator

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds.OrchestratorLoader,
                "from_settings",
                new=from_settings,
            ),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            cwd = Path(tmpdir)
            (cwd / "sample.pdf").write_bytes(pdf_bytes)
            (cwd / "agent.toml").write_text("[agent\n", encoding="utf-8")
            with _working_directory(cwd):
                result = task_cmds.task_run(
                    Namespace(
                        definition=str(fixture / "task.toml"),
                        task_input=None,
                        task_input_json=None,
                        task_input_fields=(),
                        task_file_descriptors=(),
                        task_provider_file_ids=(),
                        task_hosted_urls=(),
                        task_object_store_uris=(),
                        task_file_roles=(),
                        task_file_sizes=(),
                        task_file_sha256=(),
                        task_file_conversions=(),
                        store_dsn=None,
                        store_schema=None,
                        ephemeral=True,
                        task_run_json=True,
                        task_output_path="extraction.json",
                        task_pdf="sample.pdf",
                        task_files=(),
                        task_file_mime_types=(),
                        quiet=True,
                    ),
                    console,
                    self.theme,
                )
                written = (cwd / "extraction.json").read_text(encoding="utf-8")

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), expected)
        self.assertEqual(written, expected)
        self.assertEqual(len(orchestrator.inputs), 1)
        message = orchestrator.inputs[0]
        self.assertIsInstance(message, Message)
        content = cast(list[Any], cast(Message, message).content)
        file_blocks = [
            block for block in content if isinstance(block, MessageContentFile)
        ]
        self.assertEqual(len(file_blocks), 1)
        self.assertEqual(
            b64decode(cast(str, file_blocks[0].file["file_data"])),
            pdf_bytes,
        )

    def test_run_image_flow_fixture_replaces_pdf_with_images(self) -> None:
        fixture = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
        )
        output = _extraction_cli_output()
        expected = dumps(output, sort_keys=True, separators=(",", ":")) + "\n"
        stream = StringIO()
        console = Console(file=stream, width=160)
        orchestrator = _ExtractionCliOrchestrator(output)
        converter = _CliPdfPageConverter(
            (
                _cli_page_result(1, 2, b"page one"),
                _cli_page_result(2, 2, b"page two"),
            )
        )

        async def from_settings(
            loader: object,
            settings: object,
            *,
            tool_settings: object | None = None,
            tool_format: object | None = None,
        ) -> _ExtractionCliOrchestrator:
            _ = loader, settings, tool_settings, tool_format
            return orchestrator

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                task_cmds.OrchestratorLoader,
                "from_settings",
                new=from_settings,
            ),
            patch.object(
                task_client_module,
                "_file_converters",
                side_effect=lambda converters: {"pdf_image": converter},
            ),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
            _working_directory(fixture),
        ):
            output_path = Path(tmpdir) / "image.json"
            result = task_cmds.task_run(
                Namespace(
                    definition="image_flow_task.toml",
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_file_descriptors=(),
                    task_provider_file_ids=(),
                    task_hosted_urls=(),
                    task_object_store_uris=(),
                    task_file_roles=(),
                    task_file_sizes=(),
                    task_file_sha256=(),
                    task_file_conversions=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                    task_run_json=True,
                    task_output_path=str(output_path),
                    task_pdf="sample.pdf",
                    task_files=(),
                    task_file_mime_types=(),
                    quiet=True,
                ),
                console,
                self.theme,
            )
            written = output_path.read_text(encoding="utf-8")

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), expected)
        self.assertEqual(written, expected)
        self.assertEqual(len(converter.calls), 1)
        self.assertEqual(converter.calls[0][1], "application/pdf")
        self.assertEqual(
            converter.calls[0][2],
            {"dpi": 144, "format": "png", "pages": {"start": 1}},
        )
        self.assertEqual(len(orchestrator.inputs), 1)
        message = orchestrator.inputs[0]
        self.assertIsInstance(message, Message)
        content = cast(list[Any], cast(Message, message).content)
        text_blocks = [
            block for block in content if isinstance(block, MessageContentText)
        ]
        image_blocks = [
            block
            for block in content
            if isinstance(block, MessageContentImage)
        ]
        file_blocks = [
            block for block in content if isinstance(block, MessageContentFile)
        ]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(len(image_blocks), 2)
        self.assertEqual(file_blocks, [])
        self.assertEqual(
            [
                b64decode(cast(str, block.image_url["data"]))
                for block in image_blocks
            ],
            [b"page one", b"page two"],
        )
        self.assertEqual(
            [block.image_url["mime_type"] for block in image_blocks],
            ["image/png", "image/png"],
        )

    def test_run_poc_extraction_fixture_can_be_inspected_durably(
        self,
    ) -> None:
        fixture = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
        )
        output = _extraction_cli_output()
        store = InMemoryTaskStore(
            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
        )
        orchestrator = _ExtractionCliOrchestrator(output)
        loader = _ExtractionCliLoader(orchestrator)

        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
            _working_directory(fixture),
        ):
            artifact_store = LocalArtifactStore(
                Path(tmpdir) / "artifacts",
                raw_storage_allowed=True,
            )
            client = TaskClient(
                store,
                target=AgentTaskTargetRunner(loader, ref_base=fixture),
                artifact_store=artifact_store,
                hmac_provider=task_cmds._task_hmac_provider(),
                execution_roots=(fixture,),
                input_roots=(fixture,),
                definition_hash=lambda task: f"cli-{task.task.name}",
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )

            def client_context(
                definition_path: Path,
                *,
                dsn: str | None,
                schema: str | None,
                queue: bool,
                ephemeral: bool,
                hub: object | None,
                logger: object | None,
                input_value: object = None,
            ) -> task_cmds._TaskCliClientContext:
                _ = (
                    definition_path,
                    dsn,
                    schema,
                    queue,
                    ephemeral,
                    hub,
                    logger,
                    input_value,
                )
                return task_cmds._TaskCliClientContext(client)

            def inspection_context(
                args: Namespace,
                console: Console,
            ) -> task_cmds._TaskCliClientContext:
                _ = args, console
                return task_cmds._TaskCliClientContext(client)

            run_console = Console(record=True, width=200)
            with (
                patch.object(
                    task_cmds,
                    "_task_cli_client_context",
                    side_effect=client_context,
                ),
                patch.object(
                    task_cmds,
                    "_task_cli_inspection_client_context",
                    side_effect=inspection_context,
                ),
            ):
                run_result = task_cmds.task_run(
                    Namespace(
                        definition="task.toml",
                        task_input=None,
                        task_input_json=None,
                        task_input_fields=(),
                        task_file_descriptors=(),
                        task_provider_file_ids=(),
                        task_hosted_urls=(),
                        task_object_store_uris=(),
                        task_file_roles=(),
                        task_file_sizes=(),
                        task_file_sha256=(),
                        task_file_conversions=(),
                        store_dsn="memory://task-test",
                        store_schema=None,
                        ephemeral=False,
                        task_run_json=False,
                        task_output_path=None,
                        task_pdf="sample.pdf",
                        task_files=(),
                        task_file_mime_types=(),
                        quiet=False,
                    ),
                    run_console,
                    self.theme,
                )
                run_id = next(iter(store._runs))
                inspect_console = Console(record=True, width=240)
                inspect_result = task_cmds.task_inspect(
                    Namespace(
                        run_id=run_id,
                        store_dsn="memory://task-test",
                        store_schema=None,
                        after_sequence=None,
                    ),
                    inspect_console,
                    self.theme,
                )
                usage_console = Console(record=True, width=240)
                usage_result = task_cmds.task_usage(
                    Namespace(
                        run_id=run_id,
                        store_dsn="memory://task-test",
                        store_schema=None,
                        attempt_id=None,
                        source="exact",
                    ),
                    usage_console,
                    self.theme,
                )
                estimated_console = Console(record=True, width=240)
                estimated_result = task_cmds.task_usage(
                    Namespace(
                        run_id=run_id,
                        store_dsn="memory://task-test",
                        store_schema=None,
                        attempt_id=None,
                        source="estimated",
                    ),
                    estimated_console,
                    self.theme,
                )

        estimated_output = estimated_console.export_text()
        rendered = (
            run_console.export_text()
            + inspect_console.export_text()
            + usage_console.export_text()
            + estimated_output
        )
        self.assertTrue(run_result)
        self.assertTrue(inspect_result)
        self.assertTrue(usage_result)
        self.assertTrue(estimated_result)
        self.assertEqual(len(orchestrator.inputs), 1)
        self.assertIn("Task run completed:", rendered)
        self.assertIn('"source":"exact"', rendered)
        self.assertIn('"cached_input_tokens":7', rendered)
        self.assertIn('"reasoning_tokens":5', rendered)
        self.assertIn('"provider_family":"azure_openai"', rendered)
        self.assertIn('"usage":[]', estimated_output)
        self.assertIn('"total_tokens":null', estimated_output)
        self.assertNotIn("sample.pdf", rendered)
        self.assertNotIn("file_data", rendered)
        self.assertNotIn("private-deployment-name", rendered)
        self.assertNotIn("private-response-id", rendered)
        self.assertNotIn("private-cache-key", rendered)
        self.assertNotIn("private-api-key", rendered)

    def test_structured_output_writer_reports_safe_failures(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        diagnostic_console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "_write_task_run_output_file",
            return_value=False,
        ):
            self.assertFalse(
                task_cmds._write_task_run_structured_output(
                    Namespace(task_run_json=True, task_output_path="out.json"),
                    console,
                    diagnostic_console,
                    {"answer": "ok"},
                )
            )

        self.assertEqual(stream.getvalue(), "")

    def test_output_file_writer_reports_safe_failures(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing_parent = root / "missing" / "result.json"
            self.assertFalse(
                task_cmds._write_task_run_output_file(
                    str(missing_parent),
                    "{}\n",
                    console,
                )
            )

            with (
                patch.object(
                    task_cmds.Path,
                    "replace",
                    side_effect=OSError("private replace failure"),
                ),
                patch.object(
                    task_cmds.Path,
                    "unlink",
                    side_effect=OSError("private cleanup failure"),
                ),
            ):
                self.assertFalse(
                    task_cmds._write_task_run_output_file(
                        str(root / "result.json"),
                        "{}\n",
                        console,
                    )
                )
            with patch.object(
                task_cmds.Path,
                "replace",
                side_effect=OSError("private replace failure"),
            ):
                self.assertFalse(
                    task_cmds._write_task_run_output_file(
                        str(root / "cleanup.json"),
                        "{}\n",
                        console,
                    )
                )
            with patch.object(
                task_cmds,
                "NamedTemporaryFile",
                side_effect=OSError("private create failure"),
            ):
                self.assertFalse(
                    task_cmds._write_task_run_output_file(
                        str(root / "create.json"),
                        "{}\n",
                        console,
                    )
                )

        output = console.export_text()
        self.assertIn("output.write", output)
        self.assertNotIn("private", output)

    def test_run_without_result_skips_output_line(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            run_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-1",
                    state=TaskRunState.FAILED,
                    result=None,
                )
            )
        )

        with patch.object(
            task_cmds,
            "_task_cli_client_context",
            return_value=_FakeTaskClientContext(client),
        ):
            result = task_cmds.task_run(
                Namespace(
                    definition=str(FIXTURE_ROOT / "minimal.task.toml"),
                    task_input="Ada Lovelace",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    ephemeral=True,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task run completed (non-durable): run-1", output)
        self.assertIn("state failed", output)
        self.assertNotIn("output ", output)

    def test_enqueue_rejects_ephemeral_storage(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "queued.task.toml"
            _write_queued_definition(definition)

            result = task_cmds.task_enqueue(
                Namespace(
                    definition=str(definition),
                    task_input="Ada",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    wait=False,
                    wait_timeout=None,
                    poll_interval=1.0,
                    ephemeral=True,
                    queue="default",
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("store.ephemeral_unsupported", console.export_text())

    def test_enqueue_rejects_direct_definition(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_enqueue(
            Namespace(
                definition=str(FIXTURE_ROOT / "minimal.task.toml"),
                task_input="Ada",
                task_input_json=None,
                task_input_fields=(),
                task_files=(),
                store_dsn="postgresql://db/tasks",
                store_schema=None,
                wait=False,
                wait_timeout=None,
                poll_interval=1.0,
                ephemeral=False,
                queue="default",
            ),
            console,
            self.theme,
        )

        self.assertFalse(result)
        self.assertIn(
            "Task enqueue requires a queued-mode definition.",
            console.export_text(),
        )

    def test_enqueue_requires_store(self) -> None:
        console = Console(record=True, width=160)

        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(task_cmds.environ, {}, clear=True),
        ):
            definition = Path(tmpdir) / "queued.task.toml"
            _write_queued_definition(definition)

            result = task_cmds.task_enqueue(
                Namespace(
                    definition=str(definition),
                    task_input="Ada",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                    store_dsn=None,
                    store_schema=None,
                    wait=False,
                    wait_timeout=None,
                    poll_interval=1.0,
                    ephemeral=False,
                    queue="default",
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("store.missing", console.export_text())

    def test_enqueue_without_wait_returns_after_submission(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            enqueue_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-queued",
                    state=TaskRunState.QUEUED,
                )
            )
        )

        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "queued.task.toml"
            _write_queued_definition(definition)
            with patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ):
                result = task_cmds.task_enqueue(
                    Namespace(
                        definition=str(definition),
                        task_input="Ada",
                        task_input_json=None,
                        task_input_fields=(),
                        task_files=(),
                        store_dsn="postgresql://db/tasks",
                        store_schema=None,
                        wait=False,
                        wait_timeout=None,
                        poll_interval=1.0,
                        ephemeral=False,
                        queue="priority-documents",
                    ),
                    console,
                    self.theme,
                )

        self.assertTrue(result)
        self.assertEqual(client.queue_name, "priority-documents")
        self.assertEqual(
            client.queue_metadata,
            {"cli_queue": "priority-documents"},
        )
        self.assertIn("Task enqueued: run-queued", console.export_text())

    def test_enqueue_reports_wait_timeout(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            enqueue_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-queued",
                    state=TaskRunState.QUEUED,
                )
            ),
            wait_error=TaskClientWaitTimeoutError(run_id="run-queued"),
        )

        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "queued.task.toml"
            _write_queued_definition(definition)
            with patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ):
                result = task_cmds.task_enqueue(
                    Namespace(
                        definition=str(definition),
                        task_input="Ada",
                        task_input_json=None,
                        task_input_fields=(),
                        task_files=(),
                        store_dsn="postgresql://db/tasks",
                        store_schema=None,
                        wait=True,
                        wait_timeout=0.01,
                        poll_interval=0.01,
                        ephemeral=False,
                        queue="default",
                    ),
                    console,
                    self.theme,
                )

        self.assertFalse(result)
        self.assertIn("task.wait_timeout", console.export_text())

    def test_enqueue_waits_for_terminal_output(self) -> None:
        console = Console(record=True, width=160)
        client = _FakeTaskClient(
            enqueue_result=SimpleNamespace(
                run=SimpleNamespace(
                    run_id="run-queued",
                    state=TaskRunState.QUEUED,
                )
            ),
            wait_result=SimpleNamespace(
                run_id="run-queued",
                state=TaskRunState.SUCCEEDED,
                output_summary={"privacy": "<redacted>"},
                ready=True,
            ),
        )

        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "queued.task.toml"
            definition.write_text(
                """
                [task]
                name = "queued"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "text"

                [execution]
                type = "agent"
                ref = "agent.toml"

                [run]
                mode = "queue"
                queue = "documents"
                """,
                encoding="utf-8",
            )
            with patch.object(
                task_cmds,
                "_task_cli_client_context",
                return_value=_FakeTaskClientContext(client),
            ):
                result = task_cmds.task_enqueue(
                    Namespace(
                        definition=str(definition),
                        task_input="Ada",
                        task_input_json=None,
                        task_input_fields=(),
                        task_files=(),
                        store_dsn="postgresql://user:secret@db/tasks",
                        store_schema=None,
                        wait=True,
                        wait_timeout=5.0,
                        poll_interval=0.1,
                        ephemeral=False,
                        queue="documents",
                    ),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertTrue(result)
        self.assertEqual(client.wait_timeout, 5.0)
        self.assertEqual(client.poll_interval, 0.1)
        self.assertIn("Task enqueued: run-queued", output)
        self.assertIn("Task finished: run-queued", output)
        self.assertNotIn("secret", output)

    def test_worker_rejects_ephemeral_storage(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_worker(
            Namespace(
                queue="default",
                store_dsn="postgresql://db/tasks",
                store_schema=None,
                ephemeral=True,
            ),
            console,
            self.theme,
        )

        self.assertFalse(result)
        self.assertIn("store.ephemeral_unsupported", console.export_text())

    def test_worker_rejects_heartbeat_interval_not_shorter_than_lease(
        self,
    ) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_worker(
            Namespace(
                queue="default",
                store_dsn="postgresql://user:secret@db/tasks",
                store_schema=None,
                worker_id="worker-a",
                once=True,
                limit=10,
                lease_seconds=30,
                heartbeat_seconds=30,
                ephemeral=False,
            ),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("worker.heartbeat_interval", output)
        self.assertNotIn("secret", output)

    def test_worker_processes_until_no_work(self) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeTaskWorker.instances = []
        _FakeTaskWorker.results = [
            SimpleNamespace(
                processed=True,
                completion=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-1",
                        state=TaskRunState.SUCCEEDED,
                    )
                ),
                retry=None,
            ),
            SimpleNamespace(processed=False, completion=None, retry=None),
        ]

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(task_cmds, "PgsqlTaskQueue", return_value=object()),
            patch.object(
                task_cmds, "_agent_task_target", return_value=object()
            ),
            patch.object(task_cmds, "TaskWorker", _FakeTaskWorker),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="documents",
                    store_dsn="postgresql://db/tasks",
                    store_schema="tasks",
                    worker_id="worker-a",
                    once=False,
                    limit=2,
                    lease_seconds=30,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertTrue(database.entered)
        self.assertTrue(database.exited)
        self.assertIsNotNone(
            _FakeTaskWorker.instances[0].kwargs["hmac_provider"]
        )
        self.assertIn("Task processed: run-1 succeeded", output)
        self.assertIn("Task worker processed 1 run.", output)

    def test_worker_stops_after_shutdown_request(self) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeTaskWorker.instances = []

        def stop_after_first(worker: _FakeTaskWorker) -> object:
            worker.kwargs["shutdown"].request()
            return SimpleNamespace(
                processed=True,
                completion=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-1",
                        state=TaskRunState.SUCCEEDED,
                    )
                ),
                retry=None,
            )

        _FakeTaskWorker.results = [
            stop_after_first,
            SimpleNamespace(
                processed=True,
                completion=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-2",
                        state=TaskRunState.SUCCEEDED,
                    )
                ),
                retry=None,
            ),
        ]

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(task_cmds, "PgsqlTaskQueue", return_value=object()),
            patch.object(
                task_cmds, "_agent_task_target", return_value=object()
            ),
            patch.object(task_cmds, "TaskWorker", _FakeTaskWorker),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="documents",
                    store_dsn="postgresql://db/tasks",
                    store_schema="tasks",
                    worker_id="worker-a",
                    once=False,
                    limit=2,
                    lease_seconds=30,
                    heartbeat_seconds=0.25,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertEqual(_FakeTaskWorker.instances[0].calls, 1)
        self.assertEqual(
            _FakeTaskWorker.instances[0].kwargs["heartbeat_seconds"],
            0.25,
        )
        self.assertIn("Task processed: run-1 succeeded", output)
        self.assertNotIn("run-2", output)
        self.assertIn("Task worker processed 1 run.", output)

    def test_worker_reports_shutdown_abandonment(self) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeTaskWorker.instances = []
        _FakeTaskWorker.results = [
            SimpleNamespace(
                processed=True,
                completion=None,
                retry=None,
                abandonment=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-abandoned",
                        state=TaskRunState.QUEUED,
                    )
                ),
                shutdown_requested=True,
                lease_lost=False,
            ),
            SimpleNamespace(
                processed=True,
                completion=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-2",
                        state=TaskRunState.SUCCEEDED,
                    )
                ),
                retry=None,
            ),
        ]

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(task_cmds, "PgsqlTaskQueue", return_value=object()),
            patch.object(
                task_cmds, "_agent_task_target", return_value=object()
            ),
            patch.object(task_cmds, "TaskWorker", _FakeTaskWorker),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="documents",
                    store_dsn="postgresql://db/tasks",
                    store_schema="tasks",
                    worker_id="worker-a",
                    once=False,
                    limit=2,
                    lease_seconds=30,
                    heartbeat_seconds=0.25,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertEqual(_FakeTaskWorker.instances[0].calls, 1)
        self.assertIn("Task processed: run-abandoned queued", output)
        self.assertIn("Task worker shutdown requested.", output)
        self.assertNotIn("run-2", output)
        self.assertIn("Task worker processed 1 run.", output)

    def test_worker_reports_claim_loss_as_failure(self) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeTaskWorker.instances = []
        _FakeTaskWorker.results = [
            SimpleNamespace(
                processed=True,
                completion=None,
                retry=None,
                abandonment=None,
                shutdown_requested=False,
                lease_lost=True,
                claimed=SimpleNamespace(
                    run=SimpleNamespace(run_id="run-lost")
                ),
                private_detail="private heartbeat outage",
            ),
            SimpleNamespace(
                processed=True,
                completion=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-2",
                        state=TaskRunState.SUCCEEDED,
                    )
                ),
                retry=None,
            ),
        ]

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(task_cmds, "PgsqlTaskQueue", return_value=object()),
            patch.object(
                task_cmds, "_agent_task_target", return_value=object()
            ),
            patch.object(task_cmds, "TaskWorker", _FakeTaskWorker),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="documents",
                    store_dsn="postgresql://db/tasks",
                    store_schema="tasks",
                    worker_id="worker-a",
                    once=False,
                    limit=2,
                    lease_seconds=30,
                    heartbeat_seconds=0.25,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertEqual(_FakeTaskWorker.instances[0].calls, 1)
        self.assertIn("Task claim lost: run-lost", output)
        self.assertNotIn("run-2", output)
        self.assertNotIn("private heartbeat outage", output)
        self.assertIn("Task worker processed 1 run.", output)

    def test_worker_reports_claim_loss_without_claim(self) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeTaskWorker.instances = []
        _FakeTaskWorker.results = [
            SimpleNamespace(
                processed=True,
                completion=None,
                retry=None,
                abandonment=None,
                shutdown_requested=False,
                lease_lost=True,
                claimed=None,
                private_detail="private missing claim",
            )
        ]

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(task_cmds, "PgsqlTaskQueue", return_value=object()),
            patch.object(
                task_cmds, "_agent_task_target", return_value=object()
            ),
            patch.object(task_cmds, "TaskWorker", _FakeTaskWorker),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="documents",
                    store_dsn="postgresql://db/tasks",
                    store_schema="tasks",
                    worker_id="worker-a",
                    once=True,
                    limit=2,
                    lease_seconds=30,
                    heartbeat_seconds=0.25,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task claim lost.", output)
        self.assertNotIn("private missing claim", output)

    def test_worker_reports_retry_and_counts_missing_run_results(
        self,
    ) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeTaskWorker.instances = []
        _FakeTaskWorker.results = [
            SimpleNamespace(
                processed=True,
                completion=None,
                retry=SimpleNamespace(
                    run=SimpleNamespace(
                        run_id="run-retry",
                        state=TaskRunState.QUEUED,
                    )
                ),
            ),
            SimpleNamespace(processed=True, completion=None, retry=None),
        ]

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(task_cmds, "PgsqlTaskQueue", return_value=object()),
            patch.object(
                task_cmds, "_agent_task_target", return_value=object()
            ),
            patch.object(task_cmds, "TaskWorker", _FakeTaskWorker),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="documents",
                    store_dsn="postgresql://db/tasks",
                    store_schema="tasks",
                    worker_id="worker-a",
                    once=False,
                    limit=2,
                    lease_seconds=30,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("Task processed: run-retry queued", output)
        self.assertIn("Task worker processed 2 runs.", output)
        self.assertNotIn("None", output)

    def test_worker_reports_startup_error(self) -> None:
        console = Console(record=True, width=160)

        with (
            patch.object(
                task_cmds,
                "_task_pgsql_database",
                side_effect=OSError("private dsn"),
            ),
            patch.object(task_cmds, "require_feature", return_value=()),
        ):
            result = task_cmds.task_worker(
                Namespace(
                    queue="default",
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    worker_id=None,
                    once=True,
                    limit=10,
                    lease_seconds=30,
                    ephemeral=False,
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("io.failure", console.export_text())

    def test_retention_sweep_processes_expired_artifacts(self) -> None:
        console = Console(record=True, width=160)
        database = _FakeResource()
        _FakeRetentionService.instances = []
        _FakeRetentionService.error = None
        _FakeRetentionService.results = (
            SimpleNamespace(action=TaskRetentionAction.DELETED),
            SimpleNamespace(action=TaskRetentionAction.LOST),
        )

        with (
            patch.object(
                task_cmds, "_task_pgsql_database", return_value=database
            ),
            patch.object(task_cmds, "PgsqlTaskStore", return_value=object()),
            patch.object(
                task_cmds, "_task_artifact_store", return_value=object()
            ),
            patch.object(
                task_cmds,
                "TaskRetentionService",
                _FakeRetentionService,
            ),
        ):
            result = task_cmds.task_retention_sweep(
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema="tasks",
                    purpose=("input", "output"),
                    limit=2,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertTrue(database.entered)
        self.assertEqual(
            _FakeRetentionService.instances[0].purposes,
            (
                task_cmds.TaskArtifactPurpose.INPUT,
                task_cmds.TaskArtifactPurpose.OUTPUT,
            ),
        )
        self.assertEqual(_FakeRetentionService.instances[0].limit, 2)
        self.assertIn("Task retention sweep processed 2 artifacts.", output)
        self.assertIn('"deleted":1', output)
        self.assertIn('"lost":1', output)
        self.assertNotIn("secret", output)

    def test_retention_sweep_requires_store_and_artifact_root(self) -> None:
        missing_store_console = Console(record=True, width=160)
        with patch.dict(task_cmds.environ, {}, clear=True):
            missing_store = task_cmds.task_retention_sweep(
                Namespace(
                    store_dsn=None, store_schema=None, purpose=(), limit=100
                ),
                missing_store_console,
                self.theme,
            )

        missing_root_console = Console(record=True, width=160)
        with (
            patch.object(task_cmds, "_task_artifact_store", return_value=None),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            missing_root = task_cmds.task_retention_sweep(
                Namespace(
                    store_dsn="postgresql://db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=100,
                ),
                missing_root_console,
                self.theme,
            )

        self.assertFalse(missing_store)
        self.assertIn("store.missing", missing_store_console.export_text())
        self.assertFalse(missing_root)
        self.assertIn(
            "artifact_store.missing",
            missing_root_console.export_text(),
        )

    def test_retention_sweep_reports_safe_errors(self) -> None:
        cases = (
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=("bad",),
                    limit=100,
                ),
                None,
                "retention.sweep",
            ),
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=0,
                ),
                None,
                "retention.sweep",
            ),
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=100,
                ),
                TaskRetentionStoreNotFoundError("private local path"),
                "artifact_store.missing",
            ),
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=100,
                ),
                ArtifactStoreError("private artifact path"),
                "artifact_store.failure",
            ),
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=100,
                ),
                ImportError("private dependency"),
                "dependency.missing",
            ),
            (
                Namespace(
                    store_dsn="postgresql://user:secret@db/tasks",
                    store_schema=None,
                    purpose=(),
                    limit=100,
                ),
                OSError("private dsn"),
                "io.failure",
            ),
        )
        for args, error, expected in cases:
            console = Console(record=True, width=160)
            database = _FakeResource()
            _FakeRetentionService.instances = []
            _FakeRetentionService.error = error
            _FakeRetentionService.results = ()
            with (
                self.subTest(expected=expected),
                patch.object(
                    task_cmds, "_task_pgsql_database", return_value=database
                ),
                patch.object(
                    task_cmds, "PgsqlTaskStore", return_value=object()
                ),
                patch.object(
                    task_cmds, "_task_artifact_store", return_value=object()
                ),
                patch.object(
                    task_cmds,
                    "TaskRetentionService",
                    _FakeRetentionService,
                ),
            ):
                result = task_cmds.task_retention_sweep(
                    args,
                    console,
                    self.theme,
                )

            output = console.export_text()
            self.assertFalse(result)
            self.assertIn(expected, output)
            self.assertNotIn("secret", output)
            self.assertNotIn("private", output)

    def test_run_awaitable_requests_interrupt_callback(self) -> None:
        interrupted: list[bool] = []

        class _Future:
            def __init__(self, target: Callable[[], None]) -> None:
                self.target = target
                self.calls = 0

            def result(self) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise KeyboardInterrupt()
                self.target()

        class _Executor:
            def __init__(self, max_workers: int) -> None:
                self.max_workers = max_workers

            def __enter__(self) -> "_Executor":
                return self

            def __exit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                return None

            def submit(self, target: Callable[[], None]) -> _Future:
                return _Future(target)

        async def complete() -> bool:
            return True

        with (
            patch.object(task_cmds, "ThreadPoolExecutor", _Executor),
            self.assertRaises(KeyboardInterrupt),
        ):
            task_cmds._run_awaitable(
                complete(),
                on_interrupt=lambda: interrupted.append(True),
            )

        self.assertEqual(interrupted, [True])

    def test_run_awaitable_reraises_interrupt_without_callback(self) -> None:
        class _Future:
            def __init__(self, target: Callable[[], None]) -> None:
                self.target = target

            def result(self) -> None:
                raise KeyboardInterrupt()

        class _Executor:
            def __init__(self, max_workers: int) -> None:
                self.max_workers = max_workers

            def __enter__(self) -> "_Executor":
                return self

            def __exit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                return None

            def submit(self, target: Callable[[], None]) -> _Future:
                return _Future(target)

        async def complete() -> bool:
            return True

        coroutine = complete()
        try:
            with (
                patch.object(task_cmds, "ThreadPoolExecutor", _Executor),
                self.assertRaises(KeyboardInterrupt),
            ):
                task_cmds._run_awaitable(coroutine)
        finally:
            coroutine.close()

    def test_client_context_enters_database_and_stack(self) -> None:
        database = _FakeResource()
        stack_resource = _FakeResource()
        stack = AsyncExitStack()

        async def exercise() -> bool:
            await stack.enter_async_context(stack_resource)
            context = task_cmds._TaskCliClientContext(
                client=object(),
                database=database,
                stack=stack,
            )
            async with context as client:
                self.assertIsNotNone(client)
            return True

        task_cmds._run_awaitable(exercise())

        self.assertTrue(database.opened)
        self.assertTrue(database.closed)
        self.assertTrue(stack_resource.entered)
        self.assertTrue(stack_resource.exited)

    def test_client_context_closes_stack_when_database_open_fails(
        self,
    ) -> None:
        database = _FakeResource(open_error=OSError("private open"))
        stack_resource = _FakeResource()
        stack = AsyncExitStack()

        async def exercise() -> bool:
            await stack.enter_async_context(stack_resource)
            context = task_cmds._TaskCliClientContext(
                client=object(),
                database=database,
                stack=stack,
            )
            with self.assertRaises(OSError):
                async with context:
                    pass
            return True

        task_cmds._run_awaitable(exercise())

        self.assertTrue(database.opened)
        self.assertFalse(database.closed)
        self.assertTrue(stack_resource.entered)
        self.assertTrue(stack_resource.exited)

    def test_client_context_closes_stack_when_database_close_fails(
        self,
    ) -> None:
        database = _FakeResource(close_error=OSError("private close"))
        stack_resource = _FakeResource()
        stack = AsyncExitStack()

        async def exercise() -> bool:
            await stack.enter_async_context(stack_resource)
            context = task_cmds._TaskCliClientContext(
                client=object(),
                database=database,
                stack=stack,
            )
            with self.assertRaises(OSError):
                async with context:
                    pass
            return True

        task_cmds._run_awaitable(exercise())

        self.assertTrue(database.opened)
        self.assertTrue(database.closed)
        self.assertTrue(stack_resource.entered)
        self.assertTrue(stack_resource.exited)

    def test_client_context_handles_optional_resources(self) -> None:
        client = object()
        database = _FakeResource()
        stack_resource = _FakeResource()
        stack = AsyncExitStack()
        failing_database = _FakeResource(open_error=OSError("private open"))

        async def exercise() -> bool:
            async with task_cmds._TaskCliClientContext(
                client=client,
            ) as returned:
                self.assertIs(returned, client)
            async with task_cmds._TaskCliClientContext(
                client=client,
                database=database,
            ):
                pass
            await stack.enter_async_context(stack_resource)
            async with task_cmds._TaskCliClientContext(
                client=client,
                stack=stack,
            ):
                pass
            with self.assertRaises(OSError):
                async with task_cmds._TaskCliClientContext(
                    client=client,
                    database=failing_database,
                ):
                    pass
            return True

        task_cmds._run_awaitable(exercise())

        self.assertTrue(database.opened)
        self.assertTrue(database.closed)
        self.assertTrue(stack_resource.entered)
        self.assertTrue(stack_resource.exited)
        self.assertTrue(failing_database.opened)
        self.assertFalse(failing_database.closed)

    def test_client_context_factories_and_helpers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            definition_path = Path(tmpdir) / "task.toml"
            definition_path.write_text("", encoding="utf-8")
            with (
                patch.object(
                    task_cmds, "_agent_task_target", return_value=object()
                ),
                patch.object(
                    task_cmds, "_task_artifact_store", return_value=None
                ),
                patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
            ):
                ephemeral_context = task_cmds._task_cli_client_context(
                    definition_path,
                    dsn=None,
                    schema=None,
                    queue=False,
                    ephemeral=True,
                    hub=None,
                    logger=None,
                )
            database = _FakeResource()
            durable_flow_store = object()
            with (
                patch.object(
                    task_cmds, "_agent_task_target", return_value=object()
                ),
                patch.object(
                    task_cmds, "_task_pgsql_database", return_value=database
                ),
                patch.object(
                    task_cmds, "PgsqlTaskStore", return_value=object()
                ),
                patch.object(
                    task_cmds, "PgsqlTaskQueue", return_value=object()
                ),
                patch.object(
                    task_cmds,
                    "PgsqlFlowStateStore",
                    return_value=durable_flow_store,
                ),
                patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
            ):
                durable_context = task_cmds._task_cli_client_context(
                    definition_path,
                    dsn="postgresql://db/tasks",
                    schema="tasks",
                    queue=True,
                    ephemeral=False,
                    hub=None,
                    logger=None,
                )
            inspection_database = _FakeResource()
            with (
                patch.object(
                    task_cmds,
                    "_task_pgsql_database",
                    return_value=inspection_database,
                ),
                patch.object(
                    task_cmds, "PgsqlTaskStore", return_value=object()
                ),
            ):
                inspection_context = (
                    task_cmds._task_cli_inspection_client_context(
                        Namespace(
                            store_dsn="postgresql://db/tasks",
                            store_schema="tasks",
                        ),
                        Console(record=True, width=160),
                    )
                )

        self.assertIsNone(ephemeral_context.database)
        self.assertIsNotNone(ephemeral_context.client._hmac_provider)
        ephemeral_target = cast(Any, ephemeral_context.client)._target
        ephemeral_flow = ephemeral_target._runners[
            task_cmds.TaskTargetType.FLOW
        ]
        self.assertIsNone(ephemeral_flow._flow_resolver)
        self.assertIsNotNone(ephemeral_flow._strict_resolver)
        self.assertIsInstance(
            ephemeral_flow._flow_state_store,
            task_cmds.InMemoryFlowStateStore,
        )
        self.assertIs(database, durable_context.database)
        self.assertIsNotNone(durable_context.client._hmac_provider)
        durable_target = cast(Any, durable_context.client)._target
        durable_flow = durable_target._runners[task_cmds.TaskTargetType.FLOW]
        self.assertIsNone(durable_flow._flow_resolver)
        self.assertIsNotNone(durable_flow._strict_resolver)
        self.assertIs(durable_flow._flow_state_store, durable_flow_store)
        self.assertIsNotNone(inspection_context)
        assert inspection_context is not None
        self.assertIs(inspection_database, inspection_context.database)

    def test_ephemeral_client_context_cleans_temporary_artifacts(self) -> None:
        async def exercise() -> bool:
            with TemporaryDirectory() as tmpdir:
                definition_path = Path(tmpdir) / "task.toml"
                definition_path.write_text("", encoding="utf-8")
                with (
                    patch.object(
                        task_cmds,
                        "_agent_task_target",
                        return_value=object(),
                    ),
                    patch.object(
                        task_cmds,
                        "_task_artifact_store",
                        return_value=None,
                    ),
                    patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
                ):
                    context = task_cmds._task_cli_client_context(
                        definition_path,
                        dsn=None,
                        schema=None,
                        queue=False,
                        ephemeral=True,
                        hub=None,
                        logger=None,
                        input_value={
                            "source_kind": "local_path",
                            "reference": "sample.pdf",
                        },
                    )
                async with context as client:
                    artifact_store = client._artifact_store
                    self.assertIsNotNone(artifact_store)
                    assert artifact_store is not None
                    artifact_root = artifact_store._root
                    self.assertTrue(artifact_root.exists())
                self.assertFalse(artifact_root.exists())
            return True

        self.assertTrue(task_cmds._run_awaitable(exercise()))

    def test_hmac_provider_uses_environment_key(self) -> None:
        with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
            provider = task_cmds._task_hmac_provider()

        self.assertIsNotNone(provider)
        assert provider is not None
        default_key = provider.hmac_key(purpose=TaskKeyPurpose.PRIVACY_HASH)
        override_key = provider.hmac_key(
            purpose=TaskKeyPurpose.IDEMPOTENCY,
            key_id="cli-test-override",
        )
        self.assertEqual(default_key.key_id, "cli-test-v1")
        self.assertEqual(default_key.algorithm, "hmac-sha256")
        self.assertEqual(default_key.secret, b"task-hmac-test-key")
        self.assertEqual(override_key.key_id, "cli-test-override")

        with self.assertRaises(AssertionError):
            provider.hmac_key(
                purpose=TaskKeyPurpose.PRIVACY_HASH,
                key_id=" ",
            )

    def test_hmac_provider_rejects_incomplete_environment(self) -> None:
        cases = (
            {},
            {"AVALAN_TASK_HMAC_KEY_ID": "cli-test-v1"},
            {
                "AVALAN_TASK_HMAC_KEY_B64": TASK_HMAC_ENV[
                    "AVALAN_TASK_HMAC_KEY_B64"
                ]
            },
            {
                "AVALAN_TASK_HMAC_KEY_ID": "cli-test-v1",
                "AVALAN_TASK_HMAC_KEY_B64": "not base64",
            },
            {
                "AVALAN_TASK_HMAC_KEY_ID": "cli-test-v1",
                "AVALAN_TASK_HMAC_KEY_B64": "",
            },
        )

        for env in cases:
            with (
                self.subTest(env=env),
                patch.dict(
                    task_cmds.environ,
                    env,
                    clear=True,
                ),
            ):
                self.assertIsNone(task_cmds._task_hmac_provider())

        with self.assertRaises(AssertionError):
            task_cmds._TaskCliHmacProvider(key_id="", secret=b"secret")
        with self.assertRaises(AssertionError):
            task_cmds._TaskCliHmacProvider(key_id="cli-test-v1", secret=b"")

    def test_agent_target_and_database_helpers_construct(self) -> None:
        stack = AsyncExitStack()

        target = task_cmds._agent_task_target(
            Path("."),
            hub=None,
            logger=None,
            stack=stack,
        )
        database = task_cmds._task_pgsql_database(
            "postgresql://user:secret@db/tasks",
            "tasks",
        )

        self.assertIsNotNone(target)
        self.assertIsNotNone(database)

    def test_flow_resolver_loads_flow_and_reports_load_issues(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "constant"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "constant"
                value = "ok"
                """,
                encoding="utf-8",
            )
            broken_path = root / "broken.toml"
            broken_path.write_text("[flow]\nname = 'broken'", encoding="utf-8")
            resolver = task_cmds._task_flow_resolver(root)

            flow = resolver(
                _flow_task_context(TaskExecutionTarget.flow("flow.toml"))
            )
            absolute_flow = resolver(
                _flow_task_context(TaskExecutionTarget.flow(str(flow_path)))
            )
            absolute_issues = task_cmds._validate_task_flow_reference(
                root / "task.toml",
                _flow_task_context(
                    TaskExecutionTarget.flow(str(flow_path))
                ).definition,
            )
            with self.assertRaises(TaskValidationError) as context:
                resolver(
                    _flow_task_context(TaskExecutionTarget.flow("broken.toml"))
                )

        self.assertIsNotNone(flow)
        self.assertIsNotNone(absolute_flow)
        self.assertEqual(absolute_issues, ())
        self.assertEqual(
            context.exception.issues[0].category,
            TaskValidationCategory.UNSUPPORTED,
        )
        self.assertEqual(
            context.exception.issues[0].code,
            "flow.missing_section",
        )

    def test_strict_flow_resolver_loads_definition_and_reports_issues(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "strict.toml"
            flow_path.write_text(
                """
                [flow]
                name = "constant"
                version = "1"

                [[inputs]]
                name = "payload"
                type = "object"

                [[outputs]]
                name = "answer"
                type = "text"

                [entry]
                type = "node"
                node = "start"

                [output_behavior]
                type = "map"

                [output_behavior.outputs]
                answer = "start.value"

                [nodes.start]
                type = "constant"
                value = "ok"
                """,
                encoding="utf-8",
            )
            broken_path = root / "broken.toml"
            broken_path.write_text("[flow]\nname = 'broken'", encoding="utf-8")
            resolver = task_cmds._task_strict_flow_resolver(root)

            definition = resolver(
                _flow_task_context(TaskExecutionTarget.flow("strict.toml"))
            )
            absolute_definition = resolver(
                _flow_task_context(TaskExecutionTarget.flow(str(flow_path)))
            )
            with self.assertRaises(TaskValidationError) as context:
                resolver(
                    _flow_task_context(TaskExecutionTarget.flow("broken.toml"))
                )

        self.assertEqual(definition.name, "constant")
        self.assertEqual(absolute_definition.name, "constant")
        self.assertIsNone(definition.entrypoint)
        self.assertIsNone(definition.output_node)
        self.assertEqual(definition.outputs[0].name, "answer")
        self.assertEqual(
            context.exception.issues[0].category,
            TaskValidationCategory.UNSUPPORTED,
        )
        self.assertEqual(
            context.exception.issues[0].code,
            "flow.missing_section",
        )

    def test_flow_reference_validation_skips_graph_runtime_build(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "graph-task-flow"
                version = "1"

                [[inputs]]
                name = "payload"
                type = "object"

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
                source = "inline"
                mode = "executable"
                diagram = '''
                flowchart LR
                start route_1@--> finish
                '''

                [nodes.start]
                type = "constant"
                value = "ok"

                [nodes.finish]
                type = "echo"
                """,
                encoding="utf-8",
            )

            with patch(
                "avalan.flow.loader.build_flow",
                side_effect=AssertionError("runtime build not expected"),
            ) as build_flow:
                issues = task_cmds._validate_task_flow_reference(
                    root / "task.toml",
                    _flow_task_context(
                        TaskExecutionTarget.flow(str(flow_path))
                    ).definition,
                )

        self.assertEqual(issues, ())
        build_flow.assert_not_called()

    def test_flow_reference_validation_reports_native_full_load_issues(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "flow.toml"
            flow_path.write_text("[flow]\nname = 'native'\n", encoding="utf-8")
            issue = TaskValidationIssue(
                code="flow.invalid_node",
                path="nodes.secret",
                message="Flow node configuration is invalid.",
                hint="Use a supported node configuration.",
                category=TaskValidationCategory.UNSUPPORTED,
            )
            loader = MagicMock()
            loader.load_validation_result.return_value = SimpleNamespace(
                definition=object(),
                authoring_graph=False,
                issues=(),
            )
            loader.load_result.return_value = SimpleNamespace(
                definition=None,
                authoring_graph=False,
                issues=(issue,),
            )

            with patch(
                "avalan.cli.commands.task.FlowDefinitionLoader",
                return_value=loader,
            ):
                issues = task_cmds._validate_task_flow_reference(
                    root / "task.toml",
                    _flow_task_context(
                        TaskExecutionTarget.flow(str(flow_path))
                    ).definition,
                )

        self.assertEqual(issues, (issue,))
        loader.load_validation_result.assert_called_once_with(flow_path)
        loader.load_result.assert_called_once_with(flow_path)

    def test_strict_flow_resolver_compiles_graph_without_runtime_build(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "strict.toml"
            flow_path.write_text(
                """
                [flow]
                name = "graph-strict"
                version = "1"

                [[inputs]]
                name = "payload"
                type = "object"

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
                source = "inline"
                mode = "executable"
                diagram = '''
                flowchart LR
                start route_1@--> finish
                '''

                [nodes.start]
                type = "constant"
                value = "ok"

                [nodes.finish]
                type = "echo"
                """,
                encoding="utf-8",
            )
            resolver = task_cmds._task_strict_flow_resolver(root)

            with patch(
                "avalan.flow.loader.build_flow",
                side_effect=AssertionError("runtime build not expected"),
            ):
                definition = resolver(
                    _flow_task_context(TaskExecutionTarget.flow("strict.toml"))
                )

        self.assertEqual(definition.name, "graph-strict")
        self.assertEqual(
            [(edge.source, edge.target) for edge in definition.edges],
            [("start", "finish")],
        )

    def test_strict_flow_resolver_reports_graph_issues_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "strict.toml"
            flow_path.write_text(
                """
                [flow]
                name = "graph-strict"
                version = "1"

                [[inputs]]
                name = "payload"
                type = "object"

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
                source = "inline"
                mode = "executable"
                diagram = '''
                flowchart LR
                start -->|Private customer route| finish
                '''

                [nodes.start]
                type = "constant"
                value = "ok"

                [nodes.finish]
                type = "echo"
                """,
                encoding="utf-8",
            )
            resolver = task_cmds._task_strict_flow_resolver(root)

            with (
                patch(
                    "avalan.flow.loader.build_flow",
                    side_effect=AssertionError("runtime build not expected"),
                ),
                self.assertRaises(TaskValidationError) as context,
            ):
                resolver(
                    _flow_task_context(TaskExecutionTarget.flow("strict.toml"))
                )

        self.assertEqual(
            context.exception.issues[0].code,
            "flow.graph.unsupported_executable_edge",
        )
        self.assertEqual(context.exception.issues[0].path, "graph.edges")
        rendered = " ".join(
            issue.message for issue in context.exception.issues
        )
        self.assertNotIn("Private customer route", rendered)

    def test_low_level_helpers_cover_safe_branches(self) -> None:
        console = Console(record=True, width=160)
        task_cmds._print_task_command_error(
            console,
            "message",
            "code.value",
            "hint",
        )
        task_cmds._print_task_result(console, None)
        task_cmds._print_task_execution_error(
            console,
            TaskClientUnsupportedOperationError(
                code="task.unsupported",
                operation="run",
                message="unsupported",
            ),
        )
        task_cmds._print_task_execution_error(console, OSError("private"))
        task_cmds._print_task_execution_error(console, RuntimeError("private"))
        task_cmds._print_task_inspection_error(
            console,
            ImportError("private"),
        )
        task_cmds._print_task_inspection_error(console, OSError("private"))
        task_cmds._print_task_inspection_error(
            console,
            AssertionError("private"),
        )
        task_cmds._print_task_execution_error(
            console,
            TaskValidationError(
                (
                    TaskValidationIssue(
                        code="bad",
                        path="input",
                        message="bad input",
                        hint="fix it",
                        category=TaskValidationCategory.VALUE,
                    ),
                )
            ),
        )
        with self.assertRaises(RuntimeError):
            task_cmds._run_awaitable(_raise_runtime_error())
        with self.assertRaises(TaskClientUnsupportedOperationError):
            task_cmds._run_awaitable(
                task_cmds._task_cli_inspection_target(
                    cast(TaskTargetContext, object())
                )
            )
        self.assertIsNone(
            task_cmds._task_cli_after_sequence(Namespace(after_sequence=None))
        )
        with self.assertRaises(AssertionError):
            task_cmds._task_cli_after_sequence(Namespace(after_sequence=-1))
        with patch.dict(
            task_cmds.environ,
            {
                "AVALAN_TASK_STORE_SCHEMA": "tasks",
                "AVALAN_TASK_ARTIFACT_ROOT": "/tmp/task-artifacts",
            },
            clear=True,
        ):
            self.assertEqual(
                task_cmds._task_store_schema(Namespace()), "tasks"
            )
            self.assertIsNotNone(task_cmds._task_artifact_store())
        with patch.dict(task_cmds.environ, {}, clear=True):
            self.assertIsNone(task_cmds._task_artifact_store())
        self.assertEqual(
            task_cmds._safe_queue_metadata(Namespace(queue="")), {}
        )
        self.assertEqual(
            task_cmds._safe_queue_metadata(Namespace(queue="q")),
            {"cli_queue": "q"},
        )
        self.assertIsNone(task_cmds._task_cli_queue_name(Namespace(queue="")))
        self.assertEqual(
            task_cmds._task_command_metadata(ephemeral=True)["store_mode"],
            "ephemeral-memory",
        )
        self.assertTrue(
            task_cmds._task_cli_contains_local_file(
                {
                    "nested": {
                        "source_kind": "local_path",
                        "reference": "sample.pdf",
                    }
                }
            )
        )
        self.assertTrue(
            task_cmds._task_cli_contains_local_file(
                [{"source_kind": "local_path", "reference": "sample.pdf"}]
            )
        )
        self.assertFalse(
            task_cmds._task_cli_contains_local_file(
                {"nested": {"source_kind": "remote_url", "reference": "url"}}
            )
        )
        event_value = task_cmds._task_event_cli_value(
            SimpleNamespace(
                event_id="event-1",
                run_id="run-1",
                sequence=1,
                event_type="start",
                category=TaskEventCategory.ENGINE,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                attempt_id=None,
                payload=None,
            )
        )
        self.assertNotIn("attempt_id", event_value)
        self.assertNotIn("payload", event_value)

        output = console.export_text()
        self.assertIn("task.unsupported", output)
        self.assertIn("dependency.missing", output)
        self.assertIn("task.inspection", output)
        self.assertIn("task.execution", output)

    def test_validate_task_cli_input_for_command_success_path(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds._validate_task_cli_input_for_command(
            Namespace(
                definition=str(FIXTURE_ROOT / "minimal.task.toml"),
                task_input="Ada Lovelace",
                task_input_json=None,
                task_input_fields=(),
                task_files=(),
            ),
            console,
        )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("Task input is valid.", output)
        self.assertIn("<redacted>", output)

    def test_validate_task_cli_input_for_command_failure_paths(self) -> None:
        cases = (
            (
                str(FIXTURE_ROOT / "missing_sections.task.toml"),
                "Ada",
                None,
                "Task definition could not be loaded.",
            ),
            (
                str(FIXTURE_ROOT / "minimal.task.toml"),
                None,
                "{not json",
                "Task input could not be parsed.",
            ),
            (
                str(FIXTURE_ROOT / "minimal.task.toml"),
                None,
                '{"name":"Ada"}',
                "Task input is invalid.",
            ),
        )
        for definition, task_input, task_input_json, expected in cases:
            console = Console(record=True, width=160)
            with self.subTest(expected=expected):
                result = task_cmds._validate_task_cli_input_for_command(
                    Namespace(
                        definition=definition,
                        task_input=task_input,
                        task_input_json=task_input_json,
                        task_input_fields=(),
                        task_files=(),
                    ),
                    console,
                )

            self.assertFalse(result)
            self.assertIn(expected, console.export_text())

    def test_validate_task_cli_input_for_command_non_input_paths(self) -> None:
        console = Console(record=True, width=160)
        self.assertTrue(
            task_cmds._validate_task_cli_input_for_command(
                Namespace(
                    definition=None,
                    task_input=None,
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                ),
                console,
            )
        )
        self.assertTrue(
            task_cmds._validate_task_cli_input_for_command(
                Namespace(
                    definition=123,
                    task_input="Ada",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                ),
                console,
            )
        )
        self.assertFalse(
            task_cmds._validate_task_cli_input_for_command(
                Namespace(
                    definition="/tmp/private/missing.task.toml",
                    task_input="Ada",
                    task_input_json=None,
                    task_input_fields=(),
                    task_files=(),
                ),
                console,
            )
        )
        self.assertIn("file.read", console.export_text())


class CliTaskPgsqlTestCase(TestCase):
    def setUp(self) -> None:
        self.theme = MagicMock()

    def test_pgsql_status_dispatches_current_with_safe_success(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(task_cmds, "run_task_pgsql_current") as current:
            result = task_cmds.task_pgsql_status(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema="tenant_tasks",
                    verbose=True,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        current.assert_called_once()
        settings = current.call_args.args[0]
        self.assertEqual(
            settings.url, "postgresql://user:secret@db.example.com/tasks"
        )
        self.assertEqual(settings.schema, "tenant_tasks")
        self.assertEqual(current.call_args.kwargs["verbose"], True)
        self.assertIn("migration status checked", output)
        self.assertNotIn("secret", output)
        self.assertNotIn("db.example.com", output)

    def test_pgsql_migrate_dispatches_upgrade_revision(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(task_cmds, "run_task_pgsql_upgrade") as upgrade:
            result = task_cmds.task_pgsql_migrate(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                    migration_revision="head",
                ),
                console,
                self.theme,
            )

        self.assertTrue(result)
        upgrade.assert_called_once()
        self.assertEqual(upgrade.call_args.kwargs["revision"], "head")
        self.assertNotIn("secret", console.export_text())

    def test_pgsql_check_dispatches_check(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(task_cmds, "run_task_pgsql_check") as check:
            result = task_cmds.task_pgsql_check(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                ),
                console,
                self.theme,
            )

        self.assertTrue(result)
        check.assert_called_once()
        self.assertIn("migrations are current", console.export_text())

    def test_pgsql_stamp_dispatches_stamp_revision(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(task_cmds, "run_task_pgsql_stamp") as stamp:
            result = task_cmds.task_pgsql_stamp(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                    migration_revision="20260530_0001",
                ),
                console,
                self.theme,
            )

        self.assertTrue(result)
        stamp.assert_called_once()
        self.assertEqual(stamp.call_args.kwargs["revision"], "20260530_0001")
        self.assertIn("20260530_0001", console.export_text())
        self.assertNotIn("secret", console.export_text())

    def test_pgsql_diagnose_uses_env_without_printing_dsn(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(
            task_cmds.environ,
            {
                "AVALAN_TASK_PGSQL_DSN": (
                    "postgresql://user:secret@db.example.com/tasks"
                ),
                "AVALAN_TASK_PGSQL_SCHEMA": "tenant_tasks",
            },
        ):
            result = task_cmds.task_pgsql_diagnose(
                Namespace(dsn=None, schema=None),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("configured", output)
        self.assertIn("tenant_tasks", output)
        self.assertIn("20260530_0001", output)
        self.assertNotIn("secret", output)
        self.assertNotIn("db.example.com", output)

    def test_pgsql_commands_require_configured_dsn(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_pgsql_check(
                Namespace(dsn=None, schema=None),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("DSN is not configured", output)
        self.assertIn("AVALAN_TASK_PGSQL_DSN", output)

    def test_pgsql_status_requires_configured_dsn(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_pgsql_status(
                Namespace(dsn=None, schema=None, verbose=False),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("DSN is not configured", console.export_text())

    def test_pgsql_migrate_requires_configured_dsn(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_pgsql_migrate(
                Namespace(
                    dsn=None,
                    schema=None,
                    migration_revision="head",
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("DSN is not configured", console.export_text())

    def test_pgsql_stamp_requires_configured_dsn(self) -> None:
        console = Console(record=True, width=160)

        with patch.dict(task_cmds.environ, {}, clear=True):
            result = task_cmds.task_pgsql_stamp(
                Namespace(
                    dsn=None,
                    schema=None,
                    migration_revision="head",
                ),
                console,
                self.theme,
            )

        self.assertFalse(result)
        self.assertIn("DSN is not configured", console.export_text())

    def test_pgsql_status_errors_are_sanitized(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "run_task_pgsql_current",
            side_effect=task_cmds.PgsqlTaskMigrationError(
                "dependency.task_pgsql_migrations_missing: install extras"
            ),
        ):
            result = task_cmds.task_pgsql_status(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                    verbose=False,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("dependency.task_pgsql_migrations_missing", output)
        self.assertNotIn("secret", output)

    def test_pgsql_check_errors_are_sanitized(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "run_task_pgsql_check",
            side_effect=task_cmds.PgsqlTaskMigrationError(
                "dependency.task_pgsql_migrations_missing: install extras"
            ),
        ):
            result = task_cmds.task_pgsql_check(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("dependency.task_pgsql_migrations_missing", output)
        self.assertNotIn("secret", output)

    def test_pgsql_errors_are_sanitized(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "run_task_pgsql_upgrade",
            side_effect=task_cmds.PgsqlTaskMigrationError(
                "dependency.task_pgsql_migrations_missing: install extras"
            ),
        ):
            result = task_cmds.task_pgsql_migrate(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                    migration_revision="head",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("dependency.task_pgsql_migrations_missing", output)
        self.assertNotIn("secret", output)
        self.assertNotIn("db.example.com", output)

    def test_pgsql_stamp_errors_are_sanitized(self) -> None:
        console = Console(record=True, width=160)

        with patch.object(
            task_cmds,
            "run_task_pgsql_stamp",
            side_effect=task_cmds.PgsqlTaskMigrationError(
                "dependency.task_pgsql_migrations_missing: install extras"
            ),
        ):
            result = task_cmds.task_pgsql_stamp(
                Namespace(
                    dsn="postgresql://user:secret@db.example.com/tasks",
                    schema=None,
                    migration_revision="head",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("dependency.task_pgsql_migrations_missing", output)
        self.assertNotIn("secret", output)

    def test_invalid_revision_prints_safe_diagnostic(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_pgsql_migrate(
            Namespace(
                dsn="postgresql://user:secret@db.example.com/tasks",
                schema=None,
                migration_revision="head;drop",
            ),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Invalid PostgreSQL migration argument", output)
        self.assertNotIn("head;drop", output)
        self.assertNotIn("secret", output)


def _write_queued_definition(path: Path) -> None:
    path.write_text(
        """
        [task]
        name = "queued"
        version = "1"

        [input]
        type = "string"

        [output]
        type = "text"

        [execution]
        type = "agent"
        ref = "agent.toml"

        [run]
        mode = "queue"
        queue = "documents"
        """,
        encoding="utf-8",
    )


def _write_direct_object_definition(path: Path) -> Path:
    definition = path / "direct_object.task.toml"
    definition.write_text(
        """
        [task]
        name = "direct_object"
        version = "1"

        [input]
        type = "string"

        [output]
        type = "object"

        [output.schema]
        type = "object"
        required = ["answer", "input"]

        [output.schema.properties.answer]
        type = "string"

        [output.schema.properties.input]
        type = "string"

        [execution]
        type = "agent"
        ref = "agents/direct_object.toml"
        """,
        encoding="utf-8",
    )
    return definition


def _flow_task_context(execution: TaskExecutionTarget) -> TaskTargetContext:
    return TaskTargetContext(
        definition=TaskDefinition(
            task=TaskMetadata(name="flow", version="1"),
            input=TaskInputContract.object(),
            output=TaskOutputContract.json({}),
            execution=execution,
        ),
        execution=TaskExecutionContext(
            run_id="run-1",
            attempt_id="attempt-1",
            attempt_number=1,
        ),
        input_value={},
        files=(),
        metadata={},
    )


async def _raise_runtime_error() -> bool:
    raise RuntimeError("private")


if __name__ == "__main__":
    main()
