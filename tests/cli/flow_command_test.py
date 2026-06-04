from argparse import Namespace
from asyncio import run as asyncio_run
from base64 import b64decode
from io import StringIO
from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase, main
from unittest.mock import patch

from rich.console import Console

from avalan.cli.commands import flow as flow_cmds
from avalan.cli.commands import task as task_cmds
from avalan.entities import Message, MessageContentFile, MessageContentText
from avalan.flow import (
    FlowDefinition,
    FlowInputDefinition,
    FlowInputType,
    FlowLoadIssue,
    FlowLoadIssueCategory,
    FlowNodeDefinition,
    FlowOutputDefinition,
    FlowOutputType,
)
from avalan.task import (
    TaskInputType,
    TaskOutputType,
    TaskValidationCategory,
)

TASK_HMAC_ENV = {
    "AVALAN_TASK_HMAC_KEY_ID": "flow-cli-test-v1",
    "AVALAN_TASK_HMAC_KEY_B64": "Zmxvdy1jbGktaG1hYy10ZXN0LWtleQ==",
}
TASK_ARGS = {
    "task_input": None,
    "task_input_json": None,
    "task_input_fields": (),
    "task_files": (),
    "task_file_descriptors": (),
    "task_provider_file_ids": (),
    "task_hosted_urls": (),
    "task_object_store_uris": (),
    "task_file_mime_types": (),
    "task_file_roles": (),
    "task_file_sizes": (),
    "task_file_sha256": (),
    "task_file_conversions": (),
    "task_pdf": None,
    "task_run_json": False,
    "task_output_path": None,
    "quiet": False,
}


class FlowRunCommandTestCase(TestCase):
    def setUp(self) -> None:
        self.theme = SimpleNamespace()

    def test_flow_run_json_prints_only_output(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = _write_object_echo_flow(root)
            result = flow_cmds.flow_run(
                _args(
                    flow=flow_path,
                    task_input_json='{"answer":"ok"}',
                    task_run_json=True,
                ),
                console,
                self.theme,
            )

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), '{"answer":"ok"}\n')

    def test_flow_run_writes_output_file_and_quiet_suppresses_summary(
        self,
    ) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = _write_object_echo_flow(root)
            output_path = root / "result.json"
            result = flow_cmds.flow_run(
                _args(
                    flow=flow_path,
                    task_input_json='{"answer":"ok"}',
                    task_output_path=str(output_path),
                    quiet=True,
                ),
                console,
                self.theme,
            )
            written = output_path.read_text(encoding="utf-8")

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), "")
        self.assertEqual(written, '{"answer":"ok"}\n')

    def test_flow_run_text_output_prints_human_summary(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "text.flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "text"
                entrypoint = "start"
                output_node = "start"

                [flow.input]
                name = "value"
                type = "string"

                [flow.output]
                name = "result"
                type = "text"

                [nodes.start]
                type = "echo"
                input = "value"
                """,
                encoding="utf-8",
            )
            result = flow_cmds.flow_run(
                _args(flow=flow_path, task_input="ready"),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("Flow run completed.", output)
        self.assertIn('"ready"', output)

    def test_flow_run_reports_load_failure_without_private_toml(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "broken.flow.toml"
            flow_path.write_text(
                "[flow\nsecret = 'private customer prompt'",
                encoding="utf-8",
            )
            result = flow_cmds.flow_run(
                _args(flow=flow_path),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("flow.malformed_toml", output)
        self.assertNotIn("private customer prompt", output)
        self.assertNotIn("broken.flow.toml", output)

    def test_flow_run_reports_read_failure_without_private_path(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            flow_path = Path(temporary_directory) / "missing.flow.toml"
            result = flow_cmds.flow_run(
                _args(flow=flow_path),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("file.read", output)
        self.assertNotIn("missing.flow.toml", output)

    def test_flow_run_reports_input_and_output_failures(self) -> None:
        cases = (
            ("input", _args(task_input="not-json"), "input.json"),
            (
                "output",
                _args(task_input_json='{"answer":3}'),
                "output.invalid_type",
            ),
        )

        for _name, args, expected in cases:
            with self.subTest(expected=expected):
                console = Console(record=True, width=160)
                with TemporaryDirectory() as temporary_directory:
                    flow_path = _write_object_echo_flow(
                        Path(temporary_directory)
                    )
                    args.flow = str(flow_path)
                    result = flow_cmds.flow_run(args, console, self.theme)

                output = console.export_text()
                self.assertFalse(result)
                self.assertIn(expected, output)

    def test_flow_run_reports_input_validation_failure(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "string.flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "string"
                entrypoint = "start"
                output_node = "start"

                [flow.input]
                name = "value"
                type = "string"

                [flow.output]
                name = "result"
                type = "text"

                [nodes.start]
                type = "echo"
                input = "value"
                """,
                encoding="utf-8",
            )
            result = flow_cmds.flow_run(
                _args(flow=flow_path, task_input_json="3"),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Flow input is invalid.", output)

    def test_flow_run_reports_execution_failure_safely(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "select.flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "select"
                entrypoint = "start"
                output_node = "start"

                [flow.input]
                name = "payload"
                type = "object"

                [flow.output]
                name = "result"
                type = "json"

                [nodes.start]
                type = "select"
                input = "payload"
                path = "private.missing"
                """,
                encoding="utf-8",
            )
            result = flow_cmds.flow_run(
                _args(flow=flow_path, task_input_json='{"answer":"ok"}'),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("flow.execution", output)
        self.assertNotIn("private.missing", output)

    def test_flow_run_pdf_and_missing_file_paths(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf = root / "sample.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            flow_path = _write_file_echo_flow(root)
            stream = StringIO()
            console = Console(file=stream, width=160)

            success = flow_cmds.flow_run(
                _args(
                    flow=flow_path,
                    task_pdf=str(pdf),
                    task_run_json=True,
                ),
                console,
                self.theme,
            )
            failure_console = Console(record=True, width=160)
            failure = flow_cmds.flow_run(
                _args(flow=flow_path, task_pdf=str(root / "missing.pdf")),
                failure_console,
                self.theme,
            )

        self.assertTrue(success)
        self.assertIn('"mime_type":"application/pdf"', stream.getvalue())
        self.assertFalse(failure)
        self.assertIn("input.file_missing", failure_console.export_text())

    def test_flow_run_agent_node_uses_task_context(self) -> None:
        fixture = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
        )
        pdf_bytes = (fixture / "sample.pdf").read_bytes()
        output = _flow_cli_extraction_output()
        expected = dumps(output, sort_keys=True, separators=(",", ":")) + "\n"
        stream = StringIO()
        console = Console(file=stream, width=160)
        orchestrator = _FlowCliAgentOrchestrator(output)

        async def from_settings(
            loader: object,
            settings: object,
            *,
            tool_settings: object | None = None,
            tool_format: object | None = None,
        ) -> _FlowCliAgentOrchestrator:
            _ = loader, tool_settings, tool_format
            call_options = cast(Any, settings).call_options
            orchestrator.reasoning_options.append(call_options["reasoning"])
            return orchestrator

        with (
            patch.object(
                task_cmds.OrchestratorLoader,
                "from_settings",
                new=from_settings,
            ),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            result = flow_cmds.flow_run(
                _args(
                    flow=fixture / "flow.toml",
                    task_pdf=str(fixture / "sample.pdf"),
                    task_run_json=True,
                ),
                console,
                self.theme,
            )

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), expected)
        self.assertEqual(orchestrator.reasoning_options, [{"effort": "high"}])
        self.assertEqual(len(orchestrator.inputs), 1)
        message = orchestrator.inputs[0]
        self.assertIsInstance(message, Message)
        content = cast(Message, message).content
        self.assertIsInstance(content, list)
        blocks = cast(list[object], content)
        text_blocks = [
            block for block in blocks if isinstance(block, MessageContentText)
        ]
        file_blocks = [
            block for block in blocks if isinstance(block, MessageContentFile)
        ]
        self.assertEqual(len(text_blocks), 1)
        self.assertIn(
            "Analyze the attached synthetic invoice PDF",
            text_blocks[0].text,
        )
        self.assertEqual(len(file_blocks), 1)
        self.assertEqual(file_blocks[0].file["mime_type"], "application/pdf")
        self.assertEqual(
            b64decode(cast(str, file_blocks[0].file["file_data"])),
            pdf_bytes,
        )

    def test_flow_run_reports_bad_output_schema_ref_safely(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "invalid-schema.flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "invalid_schema"
                entrypoint = "start"
                output_node = "start"

                [flow.input]
                name = "payload"
                type = "object"

                [flow.output]
                name = "result"
                type = "object"
                schema_ref = "../private/schema.json"

                [nodes.start]
                type = "echo"
                input = "payload"
                """,
                encoding="utf-8",
            )
            result = flow_cmds.flow_run(
                _args(flow=flow_path, task_input_json='{"answer":"ok"}'),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("output.invalid_schema", output)
        self.assertNotIn("private/schema.json", output)

    def test_flow_run_agent_context_reports_failures_safely(self) -> None:
        cases = (
            (
                "metadata",
                {"node_ref": "../private/agent.toml"},
                _args(task_input_json='{"answer":"ok"}'),
                "flow.path_escape",
            ),
            (
                "schema",
                {"schema_ref": "../private/schema.json"},
                _args(task_input_json='{"answer":"ok"}'),
                "output.invalid_schema",
            ),
            (
                "parse",
                {},
                _args(task_input_json="{bad"),
                "input.json",
            ),
            (
                "input",
                {"input_type": "string"},
                _args(task_input_json="3"),
                "input.invalid_type",
            ),
            (
                "missing_file",
                {"input_type": "file"},
                _args(task_pdf="missing.pdf"),
                "input.file_missing",
            ),
            (
                "output_unsupported",
                {"output_type": "text"},
                _args(
                    task_input_json='{"answer":"ok"}',
                    task_output_path="result.json",
                ),
                "output.unsupported",
            ),
            (
                "output_path",
                {},
                _args(
                    task_input_json='{"answer":"ok"}',
                    task_output_path="missing/result.json",
                ),
                "output.write",
            ),
        )

        for name, flow_options, args, expected in cases:
            with self.subTest(name=name):
                console = Console(record=True, width=160)
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    flow_path = _write_agent_context_flow(
                        root,
                        **flow_options,
                    )
                    args.flow = str(flow_path)
                    result = flow_cmds.flow_run(args, console, self.theme)

                output = console.export_text()
                self.assertFalse(result)
                self.assertIn(expected, output)
                self.assertNotIn("private/agent.toml", output)
                self.assertNotIn("private/schema.json", output)

    def test_flow_run_agent_context_handles_client_failures(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            flow_path = _write_agent_context_flow(Path(temporary_directory))
            with patch.object(
                flow_cmds,
                "_task_cli_client_context",
                return_value=_FailingFlowClientContext(),
            ):
                result = flow_cmds.flow_run(
                    _args(flow=flow_path, task_input_json='{"answer":"ok"}'),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("io.failure", output)
        self.assertNotIn("private client failure", output)

    def test_flow_run_agent_context_reports_failed_run(self) -> None:
        console = Console(record=True, width=160)
        orchestrator = _FlowCliAgentOrchestrator({"answer": 3})

        async def from_settings(
            loader: object,
            settings: object,
            *,
            tool_settings: object | None = None,
            tool_format: object | None = None,
        ) -> _FlowCliAgentOrchestrator:
            _ = loader, settings, tool_settings, tool_format
            return orchestrator

        with (
            TemporaryDirectory() as temporary_directory,
            patch.object(
                task_cmds.OrchestratorLoader,
                "from_settings",
                new=from_settings,
            ),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            flow_path = _write_agent_context_flow(Path(temporary_directory))
            result = flow_cmds.flow_run(
                _args(flow=flow_path, task_input_json='{"answer":"ok"}'),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("task.run_failed", output)
        self.assertIn("output.invalid_type", output)

    def test_flow_run_agent_context_writer_and_human_output_paths(
        self,
    ) -> None:
        success_output = {"answer": "ok"}

        async def from_settings(
            loader: object,
            settings: object,
            *,
            tool_settings: object | None = None,
            tool_format: object | None = None,
        ) -> _FlowCliAgentOrchestrator:
            _ = loader, settings, tool_settings, tool_format
            return _FlowCliAgentOrchestrator(success_output)

        with (
            TemporaryDirectory() as temporary_directory,
            patch.object(
                task_cmds.OrchestratorLoader,
                "from_settings",
                new=from_settings,
            ),
            patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
        ):
            root = Path(temporary_directory)
            flow_path = _write_agent_context_flow(root)
            writer_console = Console(record=True, width=160)
            with patch.object(
                flow_cmds,
                "_write_task_run_structured_output",
                return_value=False,
            ):
                writer_result = flow_cmds.flow_run(
                    _args(
                        flow=flow_path,
                        task_input_json='{"answer":"ok"}',
                        task_run_json=True,
                    ),
                    writer_console,
                    self.theme,
                )
            human_console = Console(record=True, width=160)
            human_result = flow_cmds.flow_run(
                _args(flow=flow_path, task_input_json='{"answer":"ok"}'),
                human_console,
                self.theme,
            )

        self.assertFalse(writer_result)
        self.assertTrue(human_result)
        human_output = human_console.export_text()
        self.assertIn("Flow run completed.", human_output)
        self.assertIn('"answer":"ok"', human_output)

    def test_flow_run_stops_when_structured_writer_fails(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = _write_object_echo_flow(root)
            with patch.object(
                flow_cmds,
                "_write_task_run_structured_output",
                return_value=False,
            ):
                result = flow_cmds.flow_run(
                    _args(
                        flow=flow_path,
                        task_input_json='{"answer":"ok"}',
                        task_run_json=True,
                    ),
                    console,
                    self.theme,
                )

        self.assertFalse(result)

    def test_flow_run_output_parent_failure_skips_execution(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = _write_object_echo_flow(root)
            result = flow_cmds.flow_run(
                _args(
                    flow=flow_path,
                    task_input_json='{"answer":"ok"}',
                    task_output_path=str(root / "missing" / "result.json"),
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("output.write", output)
        self.assertNotIn("answer", output)

    def test_flow_task_contract_helpers_cover_all_types(self) -> None:
        input_types = {
            None: TaskInputType.OBJECT,
            FlowInputType.STRING: TaskInputType.STRING,
            FlowInputType.INTEGER: TaskInputType.INTEGER,
            FlowInputType.NUMBER: TaskInputType.NUMBER,
            FlowInputType.BOOLEAN: TaskInputType.BOOLEAN,
            FlowInputType.OBJECT: TaskInputType.OBJECT,
            FlowInputType.ARRAY: TaskInputType.ARRAY,
            FlowInputType.FILE: TaskInputType.FILE,
            FlowInputType.FILE_ARRAY: TaskInputType.FILE_ARRAY,
        }
        output_types = {
            None: TaskOutputType.JSON,
            FlowOutputType.TEXT: TaskOutputType.TEXT,
            FlowOutputType.JSON: TaskOutputType.JSON,
            FlowOutputType.OBJECT: TaskOutputType.OBJECT,
            FlowOutputType.ARRAY: TaskOutputType.ARRAY,
            FlowOutputType.FILE: TaskOutputType.FILE,
            FlowOutputType.FILE_ARRAY: TaskOutputType.FILE_ARRAY,
        }

        for flow_type, expected in input_types.items():
            with self.subTest(flow_input_type=flow_type):
                definition = _flow_definition(
                    input_definition=(
                        FlowInputDefinition(
                            name="value",
                            type=flow_type,
                            mime_types=("application/pdf",),
                        )
                        if flow_type is not None
                        else None
                    )
                )

                self.assertEqual(
                    flow_cmds._flow_task_input(definition).type,
                    expected,
                )

        for flow_type, expected in output_types.items():
            with self.subTest(flow_output_type=flow_type):
                definition = _flow_definition(
                    output_definition=(
                        FlowOutputDefinition(name="result", type=flow_type)
                        if flow_type is not None
                        else None
                    )
                )

                self.assertEqual(
                    flow_cmds._flow_task_output(definition).type,
                    expected,
                )

    def test_flow_metadata_helpers_cover_guard_paths(self) -> None:
        definition = _flow_definition(output_definition=None)
        node = flow_cmds._flow_agent_metadata_node(
            FlowNodeDefinition(name="agent", type="agent", ref="agent.toml")
        )

        self.assertIsNone(flow_cmds._flow_output_schema(definition))
        with self.assertRaises(RuntimeError):
            asyncio_run(node.execute_async({}))

    def test_flow_load_issue_helpers_cover_categories_and_files(self) -> None:
        issues = tuple(
            FlowLoadIssue(
                code=f"flow.{category.value}",
                path="flow",
                message="message",
                hint="hint",
                category=category,
            )
            for category in FlowLoadIssueCategory
        )
        categories = [
            issue.category
            for issue in flow_cmds._flow_load_task_issues(issues)
        ]
        descriptors = flow_cmds._flow_local_file_descriptors(
            [
                {"source_kind": "local_path", "reference": "one.pdf"},
                {
                    "nested": {
                        "source_kind": "local_path",
                        "reference": "two.pdf",
                    }
                },
            ]
        )

        self.assertEqual(
            categories,
            [
                TaskValidationCategory.STRUCTURE,
                TaskValidationCategory.STRUCTURE,
                TaskValidationCategory.VALUE,
                TaskValidationCategory.UNSUPPORTED,
                TaskValidationCategory.PRIVACY,
            ],
        )
        self.assertEqual(len(descriptors), 2)


def _args(**overrides: object) -> Namespace:
    values = dict(TASK_ARGS)
    values["flow"] = "flow.toml"
    values.update(overrides)
    flow = values["flow"]
    if isinstance(flow, Path):
        values["flow"] = str(flow)
    return Namespace(**values)


def _write_object_echo_flow(root: Path) -> Path:
    flow_path = root / "object.flow.toml"
    flow_path.write_text(
        """
        [flow]
        name = "object"
        entrypoint = "start"
        output_node = "start"

        [flow.input]
        name = "payload"
        type = "object"

        [flow.output]
        name = "result"
        type = "object"

        [flow.output.schema]
        type = "object"
        required = ["answer"]

        [flow.output.schema.properties.answer]
        type = "string"

        [nodes.start]
        type = "echo"
        input = "payload"
        """,
        encoding="utf-8",
    )
    return flow_path


def _write_file_echo_flow(root: Path) -> Path:
    flow_path = root / "file.flow.toml"
    flow_path.write_text(
        """
        [flow]
        name = "file"
        entrypoint = "start"
        output_node = "start"

        [flow.input]
        name = "document"
        type = "file"
        mime_types = ["application/pdf"]

        [flow.output]
        name = "result"
        type = "json"

        [nodes.start]
        type = "echo"
        input = "document"
        """,
        encoding="utf-8",
    )
    return flow_path


class _FlowCliAgentResponse:
    input_token_count = 5
    output_token_count = 7
    total_token_count = 12

    def __init__(self, output: object) -> None:
        self.output = output

    async def to_json(self) -> str:
        return dumps(self.output, sort_keys=True, separators=(",", ":"))

    async def to_str(self) -> str:
        return await self.to_json()


class _FlowCliAgentOrchestrator:
    def __init__(self, output: object) -> None:
        self.output = output
        self.inputs: list[object] = []
        self.reasoning_options: list[object] = []

    async def __aenter__(self) -> "_FlowCliAgentOrchestrator":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        _ = exc_type, exc_value, traceback
        return None

    async def __call__(self, input: object) -> _FlowCliAgentResponse:
        self.inputs.append(input)
        return _FlowCliAgentResponse(self.output)


class _FailingFlowClientContext:
    async def __aenter__(self) -> object:
        raise OSError("private client failure")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        _ = exc_type, exc, traceback
        return None


def _flow_cli_extraction_output() -> dict[str, object]:
    return {
        "line_items": [
            {
                "line_number": 1,
                "vendor_name": "Northwind Office Supplies",
                "vendor_address": "42 Market St, Denver, CO 80202",
                "customer_name": "Contoso Research Lab",
                "customer_address": "100 Example Ave, Suite 1, Denver, CO 80202",
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


def _write_agent_context_flow(
    root: Path,
    *,
    input_type: str = "object",
    output_type: str = "object",
    schema_ref: str | None = "schema.json",
    node_ref: str = "agent.toml",
) -> Path:
    if schema_ref == "schema.json":
        (root / "schema.json").write_text(
            """
            {
              "type": "object",
              "additionalProperties": false,
              "required": ["answer"],
              "properties": {
                "answer": {"type": "string"}
              }
            }
            """,
            encoding="utf-8",
        )
    (root / "agent.toml").write_text(
        """
        [agent]
        name = "Flow Agent"
        task = "Return a JSON object."
        user = "Return the answer."

        [engine]
        uri = "ai://env:KEY@openai/gpt-4o-mini"
        """,
        encoding="utf-8",
    )
    schema_line = (
        f'schema_ref = "{schema_ref}"'
        if schema_ref is not None
        else 'schema = {type = "object"}'
    )
    flow_path = root / "agent.flow.toml"
    flow_path.write_text(
        f"""
        [flow]
        name = "agent_context"
        entrypoint = "extract"
        output_node = "extract"

        [flow.input]
        name = "payload"
        type = "{input_type}"
        mime_types = ["application/pdf"]

        [flow.output]
        name = "result"
        type = "{output_type}"
        {schema_line}

        [nodes.extract]
        type = "agent"
        ref = "{node_ref}"
        input = "__task_input__"
        """,
        encoding="utf-8",
    )
    return flow_path


def _flow_definition(
    *,
    input_definition: FlowInputDefinition | None = None,
    output_definition: FlowOutputDefinition | None = None,
) -> FlowDefinition:
    return FlowDefinition(
        name="contract",
        entrypoint="start",
        output_node="start",
        input=input_definition,
        output=output_definition,
        nodes=(FlowNodeDefinition(name="start", type="echo"),),
    )


if __name__ == "__main__":
    main()
