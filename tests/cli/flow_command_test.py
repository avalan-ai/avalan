from argparse import Namespace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from rich.console import Console

from avalan.cli.commands import flow as flow_cmds
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
