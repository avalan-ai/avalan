from argparse import Namespace
from asyncio import run as asyncio_run
from base64 import b64decode
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from enum import Enum
from io import StringIO
from json import dumps, loads
from os import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase, main
from unittest.mock import patch

from rich.console import Console

from avalan.cli.commands import flow as flow_cmds
from avalan.cli.commands import task as task_cmds
from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    ToolManagerSettings,
)
from avalan.flow import (
    FlowDefinition,
    FlowDefinitionLoader,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowEntryBehavior,
    FlowExecutionTrace,
    FlowExecutor,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowJoinPolicy,
    FlowJoinPolicyType,
    FlowLoadIssue,
    FlowLoadIssueCategory,
    FlowLoopPolicy,
    FlowMappingKind,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodeRegistry,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowRetryBackoffStrategy,
    FlowRetryPolicy,
    FlowTimeoutPolicy,
    FlowViewImportMode,
    MermaidRenderResult,
    Node,
    compare_flow_topology,
    default_flow_node_registry,
    parse_mermaid_view,
    render_flow_view,
    skeleton_from_mermaid_view,
)
from avalan.task import (
    TaskClientUnsupportedOperationError,
    TaskInputType,
    TaskOutputType,
    TaskRunState,
    TaskValidationCategory,
)
from avalan.task import client as task_client_module
from avalan.task.converters import (
    TaskFileConversionError,
    TaskFileConversionPageCollection,
    TaskFileConversionPageResult,
    TaskFileConversionResult,
    TaskFileConverterCapability,
)
from avalan.task.converters.pdf_image import pdf_image_converter_capability
from avalan.task.store import TaskStoreConflictError, TaskStoreNotFoundError
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager

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
    "tool": None,
    "tools": None,
}


async def flow_cli_adder(a: int, b: int) -> int:
    return a + b


class FlowRunCommandTestCase(TestCase):
    def setUp(self) -> None:
        self.theme = SimpleNamespace()

    def test_flow_validate_json_success(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)

        with TemporaryDirectory() as temporary_directory:
            flow_path = _write_strict_constant_flow(Path(temporary_directory))
            result = flow_cmds.flow_validate(
                _args(flow=flow_path, flow_json=True),
                console,
                self.theme,
            )

        payload = loads(stream.getvalue())
        self.assertTrue(result)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["diagnostics"], [])

    def test_flow_validate_reports_load_failure_safely(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "private.flow.toml"
            flow_path.write_text(
                "[flow\nsecret = 'private customer prompt'",
                encoding="utf-8",
            )
            result = flow_cmds.flow_validate(
                _args(flow=flow_path),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Flow definition is invalid.", output)
        self.assertIn("flow.malformed_toml", output)
        self.assertNotIn("private customer prompt", output)
        self.assertNotIn("private.flow.toml", output)

    def test_flow_validate_reports_read_failure_as_json_safely(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)

        with TemporaryDirectory() as temporary_directory:
            flow_path = Path(temporary_directory) / "private.flow.toml"
            result = flow_cmds.flow_validate(
                _args(flow=flow_path, flow_json=True),
                console,
                self.theme,
            )

        payload = loads(stream.getvalue())
        self.assertFalse(result)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["diagnostics"][0]["code"], "file.read")
        self.assertNotIn("private.flow.toml", stream.getvalue())

    def test_flow_cli_sdk_validate_parity_positive_and_negative(self) -> None:
        success_stream = StringIO()
        success_console = Console(file=success_stream, width=160)
        failure_stream = StringIO()
        failure_console = Console(file=failure_stream, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            valid_path = _write_strict_constant_flow(root)
            invalid_path = root / "private.flow.toml"
            invalid_path.write_text(
                "[flow\nsecret = 'private customer prompt'",
                encoding="utf-8",
            )

            sdk_valid = FlowDefinitionLoader().load_validation_result(
                valid_path
            )
            cli_valid = flow_cmds.flow_validate(
                _args(flow=valid_path, flow_json=True),
                success_console,
                self.theme,
            )
            sdk_invalid = FlowDefinitionLoader().load_validation_result(
                invalid_path
            )
            cli_invalid = flow_cmds.flow_validate(
                _args(flow=invalid_path, flow_json=True),
                failure_console,
                self.theme,
            )

        valid_payload = loads(success_stream.getvalue())
        invalid_payload = loads(failure_stream.getvalue())
        self.assertTrue(cli_valid)
        self.assertEqual(valid_payload["ok"], sdk_valid.ok)
        self.assertEqual(valid_payload["diagnostics"], [])
        self.assertFalse(cli_invalid)
        self.assertEqual(invalid_payload["ok"], sdk_invalid.ok)
        self.assertEqual(
            [item["code"] for item in invalid_payload["diagnostics"]],
            [
                diagnostic.as_public_dict()["code"]
                for diagnostic in sdk_invalid.diagnostics
            ],
        )
        self.assertNotIn("private customer prompt", failure_stream.getvalue())
        self.assertNotIn("private.flow.toml", failure_stream.getvalue())

    def test_flow_mermaid_parse_json_success(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)

        with TemporaryDirectory() as temporary_directory:
            diagram = Path(temporary_directory) / "topology.mmd"
            diagram.write_text("graph TD\nA[Start] --> B[Done]", "utf-8")
            result = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="parse",
                    flow_json=True,
                ),
                console,
                self.theme,
            )

        payload = loads(stream.getvalue())
        self.assertTrue(result)
        self.assertTrue(payload["ok"])
        self.assertEqual(
            [(node["id"], node["label"]) for node in payload["view"]["nodes"]],
            [("A", "Start"), ("B", "Done")],
        )

    def test_flow_mermaid_parse_json_metadata_and_read_failure(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        failure_stream = StringIO()
        failure_console = Console(file=failure_stream, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagram = root / "metadata.mmd"
            diagram.write_text(
                "\n".join(
                    (
                        "graph LR",
                        "subgraph lane[Lane]",
                        "A[Start] --> B[Done]",
                        "end",
                        "classDef active fill:#fff,stroke:#333",
                        "class A active",
                        "style B fill:#eee,stroke:#111",
                        "linkStyle 0 stroke:#f00",
                        "%% note",
                    )
                ),
                "utf-8",
            )
            success = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="parse",
                    flow_json=True,
                ),
                console,
                self.theme,
            )
            failure = flow_cmds.flow_mermaid(
                _args(
                    diagram=root / "missing.mmd",
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="parse",
                    flow_json=True,
                ),
                failure_console,
                self.theme,
            )

        payload = loads(stream.getvalue())
        failure_payload = loads(failure_stream.getvalue())
        self.assertTrue(success)
        self.assertEqual(payload["view"]["groups"][0]["id"], "lane")
        self.assertEqual(
            payload["view"]["class_definitions"][0]["name"],
            "active",
        )
        self.assertEqual(payload["view"]["styles"][0]["target"], "B")
        self.assertEqual(payload["view"]["link_styles"][0]["edge_index"], 0)
        self.assertEqual(payload["view"]["comments"][0]["text"], "note")
        self.assertFalse(failure)
        self.assertEqual(
            failure_payload["diagnostics"][0]["code"], "file.read"
        )

    def test_flow_mermaid_parse_executable_negative_is_safe(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            diagram = Path(temporary_directory) / "private-topology.mmd"
            diagram.write_text(
                "graph TD\nA & B --> C\n%% private customer prompt",
                "utf-8",
            )
            result = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="executable",
                    flow_command="mermaid",
                    flow_mermaid_command="parse",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("flow.mermaid.security.ambiguous_shorthand", output)
        self.assertNotIn("private customer prompt", output)
        self.assertNotIn("private-topology.mmd", output)

    def test_flow_mermaid_parse_human_success_without_diagnostics(
        self,
    ) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            diagram = Path(temporary_directory) / "topology.mmd"
            diagram.write_text("graph TD\nA --> B", "utf-8")
            result = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="parse",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("Mermaid diagram parsed: 2 nodes, 1 edges.", output)
        self.assertNotIn("Mermaid diagnostics.", output)

    def test_flow_cli_sdk_mermaid_authoring_parity(self) -> None:
        streams = {
            "parse": StringIO(),
            "render": StringIO(),
            "compare": StringIO(),
            "skeleton": StringIO(),
            "negative": StringIO(),
        }
        consoles = {
            name: Console(file=stream, width=160)
            for name, stream in streams.items()
        }

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagram = root / "topology.mmd"
            diagram.write_text("graph TD\nA[Start] --> C[Done]", "utf-8")
            negative = root / "private-topology.mmd"
            negative.write_text(
                "graph TD\nA & B --> C\n%% private customer prompt",
                "utf-8",
            )
            flow_path = _write_strict_topology_flow(root)
            load_result = FlowDefinitionLoader().load_validation_result(
                flow_path
            )
            assert load_result.definition is not None
            source = diagram.read_text(encoding="utf-8")
            parsed = parse_mermaid_view(source)
            rendered = render_flow_view(parsed.view)
            comparison = compare_flow_topology(
                parsed.view,
                load_result.definition,
            )
            skeleton = skeleton_from_mermaid_view(
                parsed.view,
                name="topology",
                version="1",
            )
            negative_sdk = parse_mermaid_view(
                negative.read_text(encoding="utf-8"),
                import_mode=FlowViewImportMode.EXECUTABLE,
                source="/private/customer/topology.mmd",
            )

            parse_ok = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="parse",
                    flow_json=True,
                ),
                consoles["parse"],
                self.theme,
            )
            render_ok = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="render",
                    flow_json=True,
                ),
                consoles["render"],
                self.theme,
            )
            compare_ok = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    flow=flow_path,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="compare",
                    flow_json=True,
                ),
                consoles["compare"],
                self.theme,
            )
            skeleton_ok = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="presentation",
                    name="topology",
                    version="1",
                    revision=None,
                    flow_command="mermaid",
                    flow_mermaid_command="skeleton",
                    flow_json=True,
                ),
                consoles["skeleton"],
                self.theme,
            )
            negative_ok = flow_cmds.flow_mermaid(
                _args(
                    diagram=negative,
                    mode="executable",
                    flow_command="mermaid",
                    flow_mermaid_command="parse",
                    flow_json=True,
                ),
                consoles["negative"],
                self.theme,
            )

        parse_payload = loads(streams["parse"].getvalue())
        render_payload = loads(streams["render"].getvalue())
        compare_payload = loads(streams["compare"].getvalue())
        skeleton_payload = loads(streams["skeleton"].getvalue())
        negative_payload = loads(streams["negative"].getvalue())
        self.assertTrue(parse_ok)
        self.assertEqual(parse_payload["ok"], parsed.ok)
        self.assertEqual(
            parse_payload["view"],
            flow_cmds._flow_public_value(
                flow_cmds._flow_view_public_dict(parsed.view)
            ),
        )
        self.assertTrue(render_ok)
        self.assertEqual(render_payload["source"], rendered.source)
        self.assertEqual(render_payload["ok"], rendered.ok)
        self.assertTrue(compare_ok)
        self.assertEqual(compare_payload["ok"], comparison.ok)
        self.assertEqual(compare_payload["diagnostics"], [])
        self.assertTrue(skeleton_ok)
        self.assertEqual(skeleton_payload["ok"], skeleton.ok)
        self.assertEqual(
            skeleton_payload["definition"],
            flow_cmds._flow_public_value(
                flow_cmds._flow_definition_public_dict(skeleton.definition)
            ),
        )
        self.assertFalse(negative_ok)
        self.assertEqual(negative_payload["ok"], negative_sdk.ok)
        self.assertNotIn("view", negative_payload)
        self.assertEqual(
            [item["code"] for item in negative_payload["diagnostics"]],
            [
                diagnostic.as_public_dict()["code"]
                for diagnostic in negative_sdk.diagnostics
            ],
        )
        self.assertNotIn(
            "private customer prompt",
            streams["negative"].getvalue(),
        )
        self.assertNotIn(
            "private-topology.mmd",
            streams["negative"].getvalue(),
        )

    def test_flow_mermaid_render_outputs_safe_source(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            diagram = Path(temporary_directory) / "topology.mmd"
            diagram.write_text("graph TD\nA[Start] --> B[Done]", "utf-8")
            result = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="render",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("flowchart TD", output)
        self.assertIn('A["Start"]', output)

    def test_flow_mermaid_compare_reports_mismatch(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagram = root / "topology.mmd"
            diagram.write_text("graph TD\nA --> B", "utf-8")
            flow_path = _write_strict_topology_flow(root)
            result = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    flow=flow_path,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="compare",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Flow topology does not match.", output)
        self.assertIn("flow.view.binding.extra_node", output)
        self.assertIn("flow.view.binding.missing_node", output)

    def test_flow_mermaid_compare_human_success_without_diagnostics(
        self,
    ) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagram = root / "topology.mmd"
            diagram.write_text("graph TD\nA --> C", "utf-8")
            flow_path = _write_strict_topology_flow(root)
            result = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    flow=flow_path,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="compare",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn("Flow topology matches.", output)
        self.assertNotIn("Flow topology diagnostics.", output)

    def test_flow_mermaid_skeleton_prints_toml(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            diagram = Path(temporary_directory) / "topology.mmd"
            diagram.write_text("graph TD\nA --> B", "utf-8")
            result = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="presentation",
                    name="topology",
                    version="1",
                    revision=None,
                    flow_command="mermaid",
                    flow_mermaid_command="skeleton",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertTrue(result)
        self.assertIn('[nodes."A"]', output)
        self.assertIn('type = "flow_view_skeleton"', output)
        self.assertIn('"executable" = false', output)

    def test_flow_mermaid_skeleton_json_negative(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)

        with TemporaryDirectory() as temporary_directory:
            diagram = Path(temporary_directory) / "topology.mmd"
            diagram.write_text("graph TD\nA & B --> C", "utf-8")
            result = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="executable",
                    name="topology",
                    version=None,
                    revision=None,
                    flow_command="mermaid",
                    flow_mermaid_command="skeleton",
                    flow_json=True,
                ),
                console,
                self.theme,
            )

        payload = loads(stream.getvalue())
        self.assertFalse(result)
        self.assertFalse(payload["ok"])
        self.assertEqual(
            payload["diagnostics"][0]["code"],
            "flow.mermaid.security.ambiguous_shorthand",
        )

    def test_flow_validate_human_success_and_missing_file(self) -> None:
        success_console = Console(record=True, width=160)
        failure_console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = _write_strict_constant_flow(root)
            success = flow_cmds.flow_validate(
                _args(flow=flow_path),
                success_console,
                self.theme,
            )
            failure = flow_cmds.flow_validate(
                _args(flow=root / "missing.flow.toml"),
                failure_console,
                self.theme,
            )

        self.assertTrue(success)
        self.assertIn(
            "Flow definition is valid: strict 1",
            success_console.export_text(),
        )
        self.assertFalse(failure)
        self.assertIn(
            "Flow definition could not be read.",
            failure_console.export_text(),
        )

    def test_flow_mermaid_dispatch_rejects_unknown_command(self) -> None:
        console = Console(record=True, width=160)

        with self.assertRaises(AssertionError):
            flow_cmds.flow_mermaid(
                _args(flow_command="mermaid", flow_mermaid_command="bogus"),
                console,
                self.theme,
            )

    def test_flow_mermaid_parse_human_warning_and_read_failure(self) -> None:
        warning_console = Console(record=True, width=160)
        failure_console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagram = root / "topology.mmd"
            diagram.write_text("graph TD\nA & B --> C", "utf-8")
            warning = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="parse",
                ),
                warning_console,
                self.theme,
            )
            failure = flow_cmds.flow_mermaid(
                _args(
                    diagram=root / "missing.mmd",
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="parse",
                ),
                failure_console,
                self.theme,
            )

        self.assertTrue(warning)
        warning_output = warning_console.export_text()
        self.assertIn("Mermaid diagnostics.", warning_output)
        self.assertIn(
            "flow.mermaid.security.ambiguous_shorthand", warning_output
        )
        self.assertFalse(failure)
        self.assertIn(
            "Mermaid diagram could not be read.",
            failure_console.export_text(),
        )

    def test_flow_mermaid_render_negative_json_and_forced_failure(
        self,
    ) -> None:
        json_stream = StringIO()
        json_console = Console(file=json_stream, width=160)
        human_console = Console(record=True, width=160)
        forced_console = Console(record=True, width=160)
        diagnostic = _flow_cli_diagnostic("flow.execution.render_failed")

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            invalid = root / "invalid.mmd"
            invalid.write_text(
                "sequenceDiagram\nA->>B: private prompt", "utf-8"
            )
            valid = root / "valid.mmd"
            valid.write_text("graph TD\nA --> B", "utf-8")
            json_result = flow_cmds.flow_mermaid(
                _args(
                    diagram=invalid,
                    mode="executable",
                    flow_command="mermaid",
                    flow_mermaid_command="render",
                    flow_json=True,
                ),
                json_console,
                self.theme,
            )
            human_result = flow_cmds.flow_mermaid(
                _args(
                    diagram=invalid,
                    mode="executable",
                    flow_command="mermaid",
                    flow_mermaid_command="render",
                ),
                human_console,
                self.theme,
            )
            with patch.object(
                flow_cmds,
                "render_flow_view",
                return_value=MermaidRenderResult(
                    source="",
                    diagnostics=(diagnostic,),
                ),
            ):
                forced_result = flow_cmds.flow_mermaid(
                    _args(
                        diagram=valid,
                        mode="presentation",
                        flow_command="mermaid",
                        flow_mermaid_command="render",
                    ),
                    forced_console,
                    self.theme,
                )

        payload = loads(json_stream.getvalue())
        self.assertFalse(json_result)
        self.assertFalse(payload["ok"])
        self.assertFalse(human_result)
        self.assertIn(
            "Mermaid diagram is invalid.", human_console.export_text()
        )
        self.assertNotIn("private prompt", human_console.export_text())
        self.assertFalse(forced_result)
        self.assertIn(
            "Mermaid diagram could not be rendered.",
            forced_console.export_text(),
        )

    def test_flow_mermaid_render_json_success_and_read_failure(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        failure_console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagram = root / "topology.mmd"
            diagram.write_text("graph TD\nA --> B", "utf-8")
            success = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="render",
                    flow_json=True,
                ),
                console,
                self.theme,
            )
            failure = flow_cmds.flow_mermaid(
                _args(
                    diagram=root / "missing.mmd",
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="render",
                ),
                failure_console,
                self.theme,
            )

        payload = loads(stream.getvalue())
        self.assertTrue(success)
        self.assertIn("flowchart TD", payload["source"])
        self.assertFalse(failure)
        self.assertIn(
            "Mermaid diagram could not be read.",
            failure_console.export_text(),
        )

    def test_flow_mermaid_compare_json_warning_and_read_failures(self) -> None:
        json_stream = StringIO()
        json_console = Console(file=json_stream, width=160)
        warning_console = Console(record=True, width=160)
        source_failure_console = Console(record=True, width=160)
        flow_failure_console = Console(record=True, width=160)
        invalid_console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagram = root / "topology.mmd"
            diagram.write_text("graph TD\nA --> B", "utf-8")
            ambiguous = root / "ambiguous.mmd"
            ambiguous.write_text("graph TD\nA & B --> C", "utf-8")
            flow_path = _write_strict_topology_flow(root)
            ambiguous_flow = _write_ambiguous_topology_flow(root)
            json_result = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    flow=flow_path,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="compare",
                    flow_json=True,
                ),
                json_console,
                self.theme,
            )
            warning_result = flow_cmds.flow_mermaid(
                _args(
                    diagram=ambiguous,
                    flow=ambiguous_flow,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="compare",
                ),
                warning_console,
                self.theme,
            )
            source_failure = flow_cmds.flow_mermaid(
                _args(
                    diagram=root / "missing.mmd",
                    flow=flow_path,
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="compare",
                ),
                source_failure_console,
                self.theme,
            )
            flow_failure = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    flow=root / "missing.flow.toml",
                    mode="presentation",
                    flow_command="mermaid",
                    flow_mermaid_command="compare",
                ),
                flow_failure_console,
                self.theme,
            )
            invalid_result = flow_cmds.flow_mermaid(
                _args(
                    diagram=ambiguous,
                    flow=flow_path,
                    mode="executable",
                    flow_command="mermaid",
                    flow_mermaid_command="compare",
                ),
                invalid_console,
                self.theme,
            )

        payload = loads(json_stream.getvalue())
        self.assertFalse(json_result)
        self.assertFalse(payload["ok"])
        self.assertTrue(warning_result)
        warning_output = warning_console.export_text()
        self.assertIn("Flow topology matches.", warning_output)
        self.assertIn("Flow topology diagnostics.", warning_output)
        self.assertFalse(source_failure)
        self.assertIn(
            "Mermaid diagram could not be read.",
            source_failure_console.export_text(),
        )
        self.assertFalse(flow_failure)
        self.assertIn(
            "Flow definition could not be read.",
            flow_failure_console.export_text(),
        )
        self.assertFalse(invalid_result)
        self.assertIn(
            "Flow topology does not match.",
            invalid_console.export_text(),
        )

    def test_flow_mermaid_skeleton_json_success_and_negative_human(
        self,
    ) -> None:
        json_stream = StringIO()
        json_console = Console(file=json_stream, width=160)
        negative_console = Console(record=True, width=160)
        read_console = Console(record=True, width=160)
        forced_console = Console(record=True, width=160)
        diagnostic = _flow_cli_diagnostic("flow.execution.skeleton_failed")

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagram = root / "topology.mmd"
            diagram.write_text("graph TD\nA -->|yes| B", "utf-8")
            invalid = root / "invalid.mmd"
            invalid.write_text("graph TD\nA & B --> C", "utf-8")
            json_result = flow_cmds.flow_mermaid(
                _args(
                    diagram=diagram,
                    mode="presentation",
                    name="topology",
                    version=None,
                    revision="r1",
                    flow_command="mermaid",
                    flow_mermaid_command="skeleton",
                    flow_json=True,
                ),
                json_console,
                self.theme,
            )
            negative_result = flow_cmds.flow_mermaid(
                _args(
                    diagram=invalid,
                    mode="executable",
                    name="topology",
                    version=None,
                    revision=None,
                    flow_command="mermaid",
                    flow_mermaid_command="skeleton",
                ),
                negative_console,
                self.theme,
            )
            read_result = flow_cmds.flow_mermaid(
                _args(
                    diagram=root / "missing.mmd",
                    mode="presentation",
                    name="topology",
                    version=None,
                    revision=None,
                    flow_command="mermaid",
                    flow_mermaid_command="skeleton",
                ),
                read_console,
                self.theme,
            )
            with patch.object(
                flow_cmds,
                "skeleton_from_mermaid_view",
                return_value=SimpleNamespace(
                    ok=False,
                    diagnostics=(diagnostic,),
                    definition=FlowDefinition(name="failed", nodes=()),
                ),
            ):
                forced_result = flow_cmds.flow_mermaid(
                    _args(
                        diagram=diagram,
                        mode="presentation",
                        name="topology",
                        version=None,
                        revision=None,
                        flow_command="mermaid",
                        flow_mermaid_command="skeleton",
                    ),
                    forced_console,
                    self.theme,
                )

        payload = loads(json_stream.getvalue())
        self.assertTrue(json_result)
        self.assertEqual(payload["definition"]["revision"], "r1")
        self.assertEqual(payload["definition"]["edges"][0]["label"], "yes")
        self.assertFalse(negative_result)
        self.assertIn(
            "Mermaid diagram is invalid.", negative_console.export_text()
        )
        self.assertFalse(read_result)
        self.assertIn(
            "Mermaid diagram could not be read.",
            read_console.export_text(),
        )
        self.assertFalse(forced_result)
        self.assertIn(
            "Flow skeleton could not be created.",
            forced_console.export_text(),
        )

    def test_flow_cli_private_serializers_cover_branches(self) -> None:
        class LocalEnum(Enum):
            VALUE = "value"

        definition = FlowDefinition(
            name="full",
            revision="r2",
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                    schema={"type": "object"},
                    schema_ref="schema/input.json",
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="result",
                    type=FlowOutputType.OBJECT,
                    schema={"type": "object"},
                    schema_ref="schema/output.json",
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="start"),
            output_behavior=FlowOutputBehavior(
                outputs={"result": "finish.value"}
            ),
            nodes=(
                FlowNodeDefinition(name="start", type="input"),
                FlowNodeDefinition(
                    name="finish",
                    type="pass-through",
                    ref="safe.toml",
                    input="start.value",
                    output="value",
                    join_policy=FlowJoinPolicy(
                        type=FlowJoinPolicyType.ALL_DONE,
                        optional_inputs=("start",),
                    ),
                    retry_policy=FlowRetryPolicy(
                        max_attempts=2,
                        backoff=FlowRetryBackoffStrategy.CONSTANT,
                        initial_delay_seconds=1,
                        max_delay_seconds=2,
                        retryable_categories=("transient",),
                        non_retryable_categories=("validation",),
                        exhausted_route="fallback",
                    ),
                    timeout_policy=FlowTimeoutPolicy(per_attempt_seconds=3),
                    loop_policy=FlowLoopPolicy(
                        max_iterations=1,
                        max_elapsed_seconds=5,
                        output_selector="finish.value",
                        limit_route="fallback",
                    ),
                    mappings=(
                        FlowInputMapping(
                            target="payload",
                            kind=FlowMappingKind.OBJECT,
                            fields={"answer": "inputs.payload.answer"},
                        ),
                    ),
                    config={
                        "count": 3,
                        "nested": {"enabled": True},
                    },
                ),
            ),
            edges=(
                FlowEdgeDefinition(
                    source="start", target="finish", label="ok"
                ),
            ),
            tags=("cli",),
            variables={"rank": 1},
        )

        public = flow_cmds._flow_definition_public_dict(definition)
        toml = flow_cmds._flow_definition_toml(definition)
        minimal_toml = flow_cmds._flow_definition_toml(
            FlowDefinition(name="minimal", nodes=())
        )

        self.assertEqual(
            public["inputs"][0]["schema_ref"], "schema/input.json"
        )
        self.assertEqual(public["nodes"][1]["join_policy"]["type"], "all_done")
        self.assertEqual(public["nodes"][1]["retry_policy"]["max_attempts"], 2)
        self.assertEqual(
            public["nodes"][1]["timeout_policy"]["per_attempt_seconds"],
            3,
        )
        self.assertEqual(
            public["nodes"][1]["loop_policy"]["limit_route"], "fallback"
        )
        self.assertEqual(public["nodes"][1]["mappings"][0]["type"], "object")
        self.assertIn('revision = "r2"', toml)
        self.assertIn('label = "ok"', toml)
        self.assertNotIn("tags", minimal_toml)
        self.assertNotIn("[variables]", minimal_toml)
        self.assertEqual(
            flow_cmds._flow_definition_identity(
                FlowDefinition(name="revisioned", revision="r3", nodes=())
            ),
            "r3",
        )
        self.assertEqual(
            flow_cmds._flow_definition_identity(
                FlowDefinition(name="unversioned", nodes=())
            ),
            "unversioned",
        )
        self.assertEqual(flow_cmds._flow_diagnostic_location({}), "")
        self.assertEqual(
            flow_cmds._flow_diagnostic_location(
                {"source_span": {"start_line": 1}}
            ),
            "",
        )
        self.assertEqual(
            flow_cmds._flow_source_span_public_dict(None),
            None,
        )
        self.assertEqual(
            flow_cmds._flow_public_value(FlowRetryBackoffStrategy.CONSTANT),
            "constant",
        )
        self.assertEqual(flow_cmds._toml_value(5), "5")
        self.assertEqual(flow_cmds._toml_value(LocalEnum.VALUE), '"value"')
        self.assertIn(
            '"nested"', flow_cmds._toml_value({"nested": {"ok": True}})
        )
        with self.assertRaises(AssertionError):
            flow_cmds._toml_value(object())

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

    def test_flow_cli_sdk_runtime_parity(self) -> None:
        run_stream = StringIO()
        inspect_stream = StringIO()
        trace_stream = StringIO()
        resume_stream = StringIO()
        store = _FakeFlowStateStore()

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = _write_strict_constant_flow(root)
            load_result = FlowDefinitionLoader().load_validation_result(
                flow_path
            )
            assert load_result.definition is not None
            sdk_executor = FlowExecutor()
            sdk_run = asyncio_run(
                sdk_executor.run(
                    load_result.definition,
                    inputs={"payload": {"private": "customer"}},
                )
            )
            sdk_inspection = sdk_executor.inspect(sdk_run).as_public_dict()
            sdk_trace = sdk_executor.export_trace(sdk_run)
            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                cli_run = flow_cmds.flow_run(
                    _args(
                        flow=flow_path,
                        task_input_json='{"private":"customer"}',
                        task_run_json=True,
                    ),
                    Console(file=run_stream, width=160),
                    self.theme,
                )

            review_definition = _sdk_review_definition()
            review_executor = FlowExecutor(
                registry=_sdk_review_registry(),
            )
            sdk_paused = asyncio_run(
                review_executor.run(
                    review_definition,
                    inputs={"payload": {"risk": "medium"}},
                )
            )
            sdk_resumed = asyncio_run(
                review_executor.resume(
                    review_definition,
                    sdk_paused,
                    decisions={"review": {"decision": "approved"}},
                )
            )
            with (
                patch.object(
                    flow_cmds,
                    "_task_cli_inspection_client_context",
                    return_value=_FakeFlowClientContext(object()),
                ),
                patch.object(
                    flow_cmds,
                    "PgsqlFlowStateStore",
                    return_value=object(),
                ),
                patch.object(
                    flow_cmds,
                    "FlowTaskExecutor",
                    return_value=_SdkParityFlowTaskExecutor(
                        sdk_inspection,
                        sdk_trace,
                    ),
                ),
            ):
                cli_inspect = flow_cmds.flow_inspect(
                    _args(
                        run_id="run-1",
                        store_dsn="postgresql://db/tasks",
                        flow_json=True,
                    ),
                    Console(file=inspect_stream, width=160),
                    self.theme,
                )
                cli_trace = flow_cmds.flow_trace(
                    _args(
                        run_id="run-1",
                        store_dsn="postgresql://db/tasks",
                        flow_json=True,
                    ),
                    Console(file=trace_stream, width=160),
                    self.theme,
                )
            with (
                patch.object(
                    flow_cmds,
                    "_flow_state_store_context",
                    return_value=_FakeFlowStateStoreContext(store),
                ),
                patch.object(
                    flow_cmds,
                    "_flow_load_validation_result",
                    return_value=SimpleNamespace(
                        ok=True,
                        definition=review_definition,
                    ),
                ),
                patch.object(
                    flow_cmds,
                    "FlowExecutor",
                    return_value=_SdkParityResumeFlowExecutor(sdk_resumed),
                ),
            ):
                cli_resume = flow_cmds.flow_resume(
                    _args(
                        flow=flow_path,
                        run_id="run-1",
                        decision_json='{"review":{"decision":"approved"}}',
                        store_dsn="postgresql://db/tasks",
                        flow_json=True,
                    ),
                    Console(file=resume_stream, width=160),
                    self.theme,
                )

        self.assertTrue(cli_run)
        self.assertEqual(
            loads(run_stream.getvalue()), sdk_run.outputs["result"]
        )
        self.assertTrue(cli_inspect)
        self.assertEqual(
            loads(inspect_stream.getvalue())["flow"],
            flow_cmds._flow_public_value(sdk_inspection),
        )
        self.assertTrue(cli_trace)
        self.assertEqual(
            loads(trace_stream.getvalue()),
            flow_cmds._flow_public_value(sdk_trace),
        )
        self.assertTrue(cli_resume)
        self.assertEqual(loads(resume_stream.getvalue()), sdk_resumed.outputs)
        self.assertEqual(store.updated_revision, 7)
        self.assertNotIn("customer", inspect_stream.getvalue())
        self.assertNotIn("customer", trace_stream.getvalue())
        self.assertNotIn("customer", resume_stream.getvalue())

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

    def test_flow_run_strict_builtin_flow_uses_task_context(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "strict.flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "strict_constant"
                version = "1"

                [[inputs]]
                name = "payload"
                type = "object"
                schema = {type = "object"}

                [[outputs]]
                name = "result"
                type = "object"

                [outputs.schema]
                type = "object"
                required = ["answer"]

                [outputs.schema.properties.answer]
                type = "string"

                [entry]
                type = "node"
                node = "start"

                [output_behavior]
                type = "map"

                [output_behavior.outputs]
                result = "start.value"

                [nodes.start]
                type = "constant"
                value = {answer = "ok"}
                """,
                encoding="utf-8",
            )
            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                result = flow_cmds.flow_run(
                    _args(
                        flow=flow_path,
                        task_input_json="{}",
                        task_run_json=True,
                    ),
                    console,
                    self.theme,
                )

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), '{"answer":"ok"}\n')

    def test_flow_run_strict_file_flow_uses_flow_privacy(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf_path = root / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            flow_path = _write_strict_file_privacy_flow(root)
            with (
                patch.dict(task_cmds.environ, {}, clear=True),
                _working_directory(root),
            ):
                result = flow_cmds.flow_run(
                    _args(
                        flow=flow_path,
                        task_pdf="sample.pdf",
                        task_run_json=True,
                    ),
                    console,
                    self.theme,
                )

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), '{"answer":"ok"}\n')

    def test_flow_run_strict_tool_node_uses_enabled_resolver(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        manager = ToolManager.create_instance(
            enable_tools=["flow_cli_adder"],
            available_toolsets=[ToolSet(tools=[flow_cli_adder])],
            settings=ToolManagerSettings(),
        )

        with TemporaryDirectory() as temporary_directory:
            flow_path = _write_strict_tool_flow(Path(temporary_directory))
            with (
                patch.object(
                    flow_cmds,
                    "_flow_tool_manager",
                    return_value=manager,
                ),
                patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True),
            ):
                result = flow_cmds.flow_run(
                    _args(
                        flow=flow_path,
                        task_input_json='{"left":2,"right":5}',
                        task_run_json=True,
                        tool=["flow_cli_adder"],
                    ),
                    console,
                    self.theme,
                )

        self.assertTrue(result)
        self.assertEqual(stream.getvalue(), "7\n")

    def test_flow_run_strict_tool_node_requires_enabled_resolver(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            flow_path = _write_strict_tool_flow(Path(temporary_directory))
            with patch.dict(task_cmds.environ, TASK_HMAC_ENV, clear=True):
                result = flow_cmds.flow_run(
                    _args(
                        flow=flow_path,
                        task_input_json='{"left":2,"right":"private"}',
                    ),
                    console,
                    self.theme,
                )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("flow.unsupported_node_type", output)
        self.assertNotIn("private", output)

    def test_flow_tool_manager_builds_enabled_resolver_from_args(self) -> None:
        def toolset_factory(**kwargs: object) -> ToolSet:
            return ToolSet(
                namespace=cast(str | None, kwargs.get("namespace")),
                tools=[],
            )

        with (
            patch.object(flow_cmds, "HAS_GRAPH_DEPENDENCIES", True),
            patch.object(flow_cmds, "HAS_CODE_DEPENDENCIES", True),
            patch.object(flow_cmds, "HAS_BROWSER_DEPENDENCIES", True),
            patch.object(
                flow_cmds, "GraphToolSet", side_effect=toolset_factory
            ),
            patch.object(
                flow_cmds, "CodeToolSet", side_effect=toolset_factory
            ),
            patch.object(
                flow_cmds,
                "BrowserToolSet",
                side_effect=toolset_factory,
            ),
            patch.object(
                flow_cmds,
                "DatabaseToolSet",
                side_effect=toolset_factory,
            ),
        ):
            manager = flow_cmds._flow_tool_manager(
                _args(
                    tool=["math.calculator"],
                    tools=["math"],
                    tool_database_dsn="postgresql://example.invalid/db",
                )
            )

        self.assertIsNotNone(manager)
        assert manager is not None
        self.assertIsNotNone(manager.describe_tool("math.calculator"))

    def test_flow_tool_manager_skips_unavailable_optional_toolsets(
        self,
    ) -> None:
        with (
            patch.object(flow_cmds, "HAS_GRAPH_DEPENDENCIES", False),
            patch.object(flow_cmds, "HAS_CODE_DEPENDENCIES", False),
            patch.object(flow_cmds, "HAS_BROWSER_DEPENDENCIES", False),
        ):
            manager = flow_cmds._flow_tool_manager(
                _args(tool=["math.calculator"])
            )

        self.assertIsNotNone(manager)
        assert manager is not None
        self.assertIsNotNone(manager.describe_tool("math.calculator"))
        self.assertIsNone(manager.describe_tool("code.python"))

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
        self.assertIn("Legacy native flow run completed.", output)
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
        settings_values: list[Any] = []

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
            settings_values.append(settings)
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

    def test_flow_run_image_flow_uses_task_context(self) -> None:
        fixture = (
            Path(__file__).parents[2]
            / "docs"
            / "examples"
            / "tasks"
            / "poc_extraction"
        )
        output = _flow_cli_extraction_output()
        expected = dumps(output, sort_keys=True, separators=(",", ":")) + "\n"
        stream = StringIO()
        console = Console(file=stream, width=160)
        orchestrator = _FlowCliAgentOrchestrator(output)
        converter = _FlowCliPdfPageConverter(
            (
                _flow_cli_page_result(1, 2, b"page one"),
                _flow_cli_page_result(2, 2, b"page two"),
            )
        )

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
            result = flow_cmds.flow_run(
                _args(
                    flow="image_flow.toml",
                    task_pdf="sample.pdf",
                    task_run_json=True,
                    task_output_path=str(output_path),
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
        self.assertEqual(len(orchestrator.inputs), 1)
        message = orchestrator.inputs[0]
        self.assertIsInstance(message, Message)
        content = cast(list[Any], cast(Message, message).content)
        image_blocks = [
            block
            for block in content
            if isinstance(block, MessageContentImage)
        ]
        file_blocks = [
            block for block in content if isinstance(block, MessageContentFile)
        ]
        self.assertEqual(file_blocks, [])
        self.assertEqual(
            [
                b64decode(cast(str, block.image_url["data"]))
                for block in image_blocks
            ],
            [b"page one", b"page two"],
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
                schema_ref = "missing.json"

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
        self.assertIn("flow.output.schema_ref", output)
        self.assertNotIn("missing.json", output)

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
                "flow.path_escape",
            ),
            (
                "missing_schema",
                {"schema_ref": "missing.json"},
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
                "output_contract_mismatch",
                {"output_type": "text"},
                _args(
                    task_input_json='{"answer":"ok"}',
                    task_output_path="result.json",
                ),
                "flow.incompatible_output_selection",
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

    def test_flow_run_rejects_structured_output_for_text_flow(self) -> None:
        console = Console(record=True, width=160)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_path = root / "text.flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "text"
                version = "1"

                [[inputs]]
                name = "payload"
                type = "string"

                [[outputs]]
                name = "result"
                type = "text"

                [entry]
                type = "node"
                node = "start"

                [output_behavior]
                type = "map"

                [output_behavior.outputs]
                result = "start.value"

                [nodes.start]
                type = "echo"
                """,
                encoding="utf-8",
            )
            result = flow_cmds.flow_run(
                _args(
                    flow=flow_path,
                    task_input="ready",
                    task_output_path="result.json",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("output.unsupported", output)

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

    def test_flow_inspect_and_trace_use_sanitized_task_facade(self) -> None:
        inspect_stream = StringIO()
        trace_console = Console(record=True, width=160)
        inspect_console = Console(file=inspect_stream, width=160)
        client = object()

        with (
            patch.object(
                flow_cmds,
                "_task_cli_inspection_client_context",
                return_value=_FakeFlowClientContext(client),
            ),
            patch.object(
                flow_cmds, "PgsqlFlowStateStore", return_value=object()
            ),
            patch.object(
                flow_cmds,
                "FlowTaskExecutor",
                return_value=_FakeFlowTaskExecutor(),
            ),
        ):
            inspect_result = flow_cmds.flow_inspect(
                _args(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    flow_json=True,
                    after_sequence=4,
                ),
                inspect_console,
                self.theme,
            )
            trace_result = flow_cmds.flow_trace(
                _args(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    flow_json=False,
                ),
                trace_console,
                self.theme,
            )

        inspect_payload = loads(inspect_stream.getvalue())
        trace_output = trace_console.export_text()
        self.assertTrue(inspect_result)
        self.assertTrue(trace_result)
        self.assertEqual(inspect_payload["task"]["run_id"], "run-1")
        self.assertEqual(inspect_payload["flow"]["state"], "paused")
        self.assertIn('"flow_name":"safe-flow"', trace_output)
        self.assertNotIn("private", inspect_stream.getvalue())
        self.assertNotIn("private", trace_output)

    def test_flow_inspect_reports_missing_store(self) -> None:
        console = Console(record=True, width=160)

        result = flow_cmds.flow_inspect(
            _args(run_id="run-1"),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("store.missing", output)

    def test_flow_state_store_context_handles_optional_and_sync_database(
        self,
    ) -> None:
        sync_database = _SyncFlowDatabase()

        async def exercise() -> bool:
            async with flow_cmds._FlowStateStoreContext(
                store="memory-store",  # type: ignore[arg-type]
            ) as store:
                self.assertEqual(store, "memory-store")
            async with flow_cmds._FlowStateStoreContext(
                store="pgsql-store",  # type: ignore[arg-type]
                database=sync_database,
            ) as store:
                self.assertEqual(store, "pgsql-store")
            return True

        self.assertTrue(task_cmds._run_awaitable(exercise()))
        self.assertTrue(sync_database.opened)
        self.assertTrue(sync_database.closed)

    def test_flow_cancel_delegates_to_task_client(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        client = _FakeFlowCancelClient()

        with patch.object(
            flow_cmds,
            "_task_cli_inspection_client_context",
            return_value=_FakeFlowClientContext(client),
        ):
            result = flow_cmds.flow_cancel(
                _args(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                    flow_json=True,
                ),
                console,
                self.theme,
            )

        payload = loads(stream.getvalue())
        self.assertTrue(result)
        self.assertEqual(client.cancelled, "run-1")
        self.assertEqual(payload["state"], "cancel_requested")

    def test_flow_resume_updates_state_store(self) -> None:
        stream = StringIO()
        console = Console(file=stream, width=160)
        store = _FakeFlowStateStore()
        executor = _FakeFlowExecutor()

        with (
            patch.object(
                flow_cmds,
                "_flow_state_store_context",
                return_value=_FakeFlowStateStoreContext(store),
            ),
            patch.object(flow_cmds, "FlowExecutor", return_value=executor),
            patch.object(
                flow_cmds,
                "_flow_load_validation_result",
                return_value=SimpleNamespace(
                    ok=True,
                    definition=FlowDefinition(name="strict", nodes=()),
                ),
            ),
        ):
            result = flow_cmds.flow_resume(
                _args(
                    flow="private.flow.toml",
                    run_id="run-1",
                    decision_json=(
                        '{"review":{"decision":"approved",'
                        '"comment":"private"}}'
                    ),
                    store_dsn="postgresql://db/tasks",
                    flow_json=True,
                ),
                console,
                self.theme,
            )

        payload = loads(stream.getvalue())
        self.assertTrue(result)
        self.assertEqual(payload["answer"], "approved")
        self.assertEqual(
            executor.decisions,
            {"review": {"decision": "approved", "comment": "private"}},
        )
        self.assertEqual(store.updated_revision, 7)
        update = store.updated_update
        self.assertIsNotNone(update)
        assert update is not None
        self.assertEqual(update.selected_outputs, {"answer": "approved"})

    def test_flow_resume_rejects_invalid_decision_json_safely(self) -> None:
        console = Console(record=True, width=160)

        result = flow_cmds.flow_resume(
            _args(
                flow="flow.toml",
                run_id="run-1",
                decision_json='{"review":"private-token"}',
            ),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("flow.resume_decision_shape", output)
        self.assertNotIn("private-token", output)

    def test_flow_runtime_command_error_paths_are_sanitized(self) -> None:
        cases = (
            (
                "inspect",
                flow_cmds.flow_inspect,
                _FailingFlowTaskExecutor(TaskStoreNotFoundError("private")),
                "flow.not_found",
            ),
            (
                "trace",
                flow_cmds.flow_trace,
                _FailingFlowTaskExecutor(ImportError("private")),
                "dependency.missing",
            ),
        )
        for name, command, executor, expected in cases:
            with self.subTest(command=name):
                console = Console(record=True, width=160)
                with (
                    patch.object(
                        flow_cmds,
                        "_task_cli_inspection_client_context",
                        return_value=_FakeFlowClientContext(object()),
                    ),
                    patch.object(
                        flow_cmds,
                        "PgsqlFlowStateStore",
                        return_value=object(),
                    ),
                    patch.object(
                        flow_cmds,
                        "FlowTaskExecutor",
                        return_value=executor,
                    ),
                ):
                    result = command(
                        _args(
                            run_id="run-1",
                            store_dsn="postgresql://db/tasks",
                        ),
                        console,
                        self.theme,
                    )

                output = console.export_text()
                self.assertFalse(result)
                self.assertIn(expected, output)
                self.assertNotIn("private", output)

        missing_store_console = Console(record=True, width=160)
        self.assertFalse(
            flow_cmds.flow_trace(
                _args(run_id="run-1"),
                missing_store_console,
                self.theme,
            )
        )
        self.assertIn("store.missing", missing_store_console.export_text())

        unsupported_client = _FailingFlowCancelClient(
            TaskClientUnsupportedOperationError(
                code="task.cancel_unsupported",
                operation="cancel",
                message="private",
            )
        )
        console = Console(record=True, width=160)
        with patch.object(
            flow_cmds,
            "_task_cli_inspection_client_context",
            return_value=_FakeFlowClientContext(unsupported_client),
        ):
            result = flow_cmds.flow_cancel(
                _args(
                    run_id="run-1",
                    store_dsn="postgresql://db/tasks",
                ),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("task.cancel_unsupported", output)
        self.assertNotIn("private", output)

        missing_cancel_console = Console(record=True, width=160)
        self.assertFalse(
            flow_cmds.flow_cancel(
                _args(run_id="run-1"),
                missing_cancel_console,
                self.theme,
            )
        )
        self.assertIn("store.missing", missing_cancel_console.export_text())

    def test_flow_resume_handles_failure_and_diagnostic_paths(self) -> None:
        load_result = SimpleNamespace(
            ok=True,
            definition=FlowDefinition(name="strict", nodes=()),
        )
        cases = (
            (
                "missing_loader",
                {"_flow_load_validation_result": None},
                _args(
                    flow="flow.toml",
                    run_id="run-1",
                    decision_json="{}",
                ),
                None,
            ),
            (
                "invalid_loader",
                {
                    "_flow_load_validation_result": SimpleNamespace(
                        ok=False,
                        definition=None,
                    )
                },
                _args(
                    flow="flow.toml",
                    run_id="run-1",
                    decision_json="{}",
                ),
                None,
            ),
            (
                "missing_store",
                {
                    "_flow_load_validation_result": load_result,
                    "_flow_state_store_context": None,
                },
                _args(
                    flow="flow.toml",
                    run_id="run-1",
                    decision_json="{}",
                ),
                None,
            ),
            (
                "store_failure",
                {
                    "_flow_load_validation_result": load_result,
                    "_flow_state_store_context": _FakeFlowStateStoreContext(
                        _FailingFlowStateStore(
                            TaskStoreConflictError("private")
                        )
                    ),
                },
                _args(
                    flow="flow.toml",
                    run_id="run-1",
                    decision_json="{}",
                ),
                "flow.conflict",
            ),
        )

        for name, patches, args, expected in cases:
            with self.subTest(case=name):
                console = Console(record=True, width=160)
                with ExitStack() as stack:
                    for patch_name, value in patches.items():
                        stack.enter_context(
                            patch.object(
                                flow_cmds,
                                patch_name,
                                return_value=value,
                            )
                        )
                    result = flow_cmds.flow_resume(
                        args,
                        console,
                        self.theme,
                    )

                output = console.export_text()
                self.assertFalse(result)
                if expected is not None:
                    self.assertIn(expected, output)
                self.assertNotIn("private", output)

        diagnostic = _flow_cli_diagnostic("flow.resume.blocked")
        diagnostic_cases = (
            (
                True,
                _FakeDiagnosticFlowExecutor(
                    ok=False,
                    diagnostics=(diagnostic,),
                ),
                '"flow.resume.blocked"',
            ),
            (
                False,
                _FakeDiagnosticFlowExecutor(
                    ok=False,
                    diagnostics=(diagnostic,),
                ),
                "Flow resume produced diagnostics.",
            ),
        )
        for flow_json, executor, expected in diagnostic_cases:
            with self.subTest(flow_json=flow_json):
                stream = StringIO()
                console = Console(file=stream, record=not flow_json, width=160)
                with (
                    patch.object(
                        flow_cmds,
                        "_flow_load_validation_result",
                        return_value=load_result,
                    ),
                    patch.object(
                        flow_cmds,
                        "_flow_state_store_context",
                        return_value=_FakeFlowStateStoreContext(
                            _FakeFlowStateStore()
                        ),
                    ),
                    patch.object(
                        flow_cmds,
                        "FlowExecutor",
                        return_value=executor,
                    ),
                ):
                    result = flow_cmds.flow_resume(
                        _args(
                            flow="flow.toml",
                            run_id="run-1",
                            decision_json="{}",
                            flow_json=flow_json,
                        ),
                        console,
                        self.theme,
                    )

                output = stream.getvalue()
                self.assertFalse(result)
                self.assertIn(expected, output)

    def test_flow_state_store_context_helpers_cover_store_setup(self) -> None:
        console = Console(record=True, width=160)

        missing = flow_cmds._flow_state_store_context(_args(), console)

        self.assertIsNone(missing)
        self.assertIn("store.missing", console.export_text())

        database = object()
        store = object()
        with (
            patch.object(
                flow_cmds,
                "_task_pgsql_database",
                return_value=database,
            ) as database_factory,
            patch.object(
                flow_cmds,
                "PgsqlFlowStateStore",
                return_value=store,
            ) as store_factory,
        ):
            context = flow_cmds._flow_state_store_context(
                _args(
                    store_dsn="postgresql://db/tasks",
                    store_schema="workflow",
                ),
                Console(record=True, width=160),
            )

        self.assertIsNotNone(context)
        assert context is not None
        self.assertIs(context.store, store)
        self.assertIs(context.database, database)
        database_factory.assert_called_once_with(
            "postgresql://db/tasks",
            "workflow",
        )
        store_factory.assert_called_once_with(database)

        async_database = _FakeFlowDatabase()
        async_context = flow_cmds._FlowStateStoreContext(
            store=store,
            database=async_database,
        )

        async def exercise_context() -> object:
            async with async_context as opened:
                return opened

        opened = asyncio_run(exercise_context())

        self.assertIs(opened, store)
        self.assertTrue(async_database.opened)
        self.assertTrue(async_database.closed)

    def test_flow_resume_decision_json_sources_and_errors(self) -> None:
        invalid_cases = (
            (
                "missing",
                _args(decision_json=" "),
                "flow.resume_decision_missing",
            ),
            (
                "missing_file",
                _args(decision_json="@missing-private.json"),
                "file.read",
            ),
            (
                "invalid_json",
                _args(decision_json='{"review":'),
                "flow.resume_decision_json",
            ),
            (
                "non_mapping",
                _args(decision_json='["private-token"]'),
                "flow.resume_decision_shape",
            ),
        )
        for name, args, expected in invalid_cases:
            with self.subTest(case=name):
                console = Console(record=True, width=160)
                decisions = flow_cmds._flow_resume_decisions(args, console)

                output = console.export_text()
                self.assertIsNone(decisions)
                self.assertIn(expected, output)
                self.assertNotIn("private-token", output)

        with TemporaryDirectory() as temporary:
            decision_path = Path(temporary) / "decision.json"
            decision_path.write_text(
                '{"review":{"decision":"approved","score":1}}',
                encoding="utf-8",
            )

            decisions = flow_cmds._flow_resume_decisions(
                _args(decision_json=f"@{decision_path}"),
                Console(record=True, width=160),
            )

        self.assertEqual(
            decisions,
            {"review": {"decision": "approved", "score": 1}},
        )

    def test_flow_error_printers_cover_public_error_codes(self) -> None:
        inspection_cases = (
            (TaskStoreNotFoundError("private"), "flow.not_found"),
            (ImportError("private"), "dependency.missing"),
            (OSError("private"), "io.failure"),
            (AssertionError("private"), "flow.inspection"),
        )
        for error, expected in inspection_cases:
            with self.subTest(kind="inspection", expected=expected):
                console = Console(record=True, width=160)

                flow_cmds._print_flow_inspection_error(console, error)

                output = console.export_text()
                self.assertIn(expected, output)
                self.assertNotIn("private", output)

        command_cases = (
            (TaskStoreNotFoundError("private"), "flow.not_found"),
            (TaskStoreConflictError("private"), "flow.conflict"),
            (
                TaskClientUnsupportedOperationError(
                    code="task.cancel_unsupported",
                    operation="cancel",
                    message="private",
                ),
                "task.cancel_unsupported",
            ),
            (ImportError("private"), "dependency.missing"),
            (OSError("private"), "io.failure"),
            (AssertionError("private"), "flow.command"),
        )
        for error, expected in command_cases:
            with self.subTest(kind="command", expected=expected):
                console = Console(record=True, width=160)

                flow_cmds._print_flow_command_error(console, error)

                output = console.export_text()
                self.assertIn(expected, output)
                self.assertNotIn("private", output)

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

    def test_flow_task_input_schema_preserves_multiple_inputs(self) -> None:
        definition = FlowDefinition(
            name="contract",
            inputs=(
                FlowInputDefinition(
                    name="pdf",
                    type=FlowInputType.FILE_ARRAY,
                    mime_types=("application/pdf",),
                ),
                FlowInputDefinition(
                    name="image",
                    type=FlowInputType.BOOLEAN,
                ),
            ),
            nodes=(FlowNodeDefinition(name="start", type="echo"),),
        )

        contract = flow_cmds._flow_task_input(definition)
        schema = cast(Mapping[str, object], contract.schema)
        properties = cast(Mapping[str, object], schema["properties"])
        pdf = cast(Mapping[str, object], properties["pdf"])
        image = cast(Mapping[str, object], properties["image"])

        self.assertEqual(contract.type, TaskInputType.OBJECT)
        self.assertEqual(pdf["x-avalan-input-type"], "file[]")
        self.assertEqual(pdf["x-avalan-mime-types"], ("application/pdf",))
        self.assertEqual(pdf["type"], "array")
        self.assertEqual(image["type"], "boolean")

    def test_flow_task_input_schema_covers_property_types(self) -> None:
        definition = FlowDefinition(
            name="contract",
            inputs=(
                FlowInputDefinition(
                    name="text",
                    type=FlowInputType.STRING,
                ),
                FlowInputDefinition(
                    name="count",
                    type=FlowInputType.INTEGER,
                ),
                FlowInputDefinition(
                    name="ratio",
                    type=FlowInputType.NUMBER,
                ),
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                    schema={"type": "object", "required": ["answer"]},
                ),
                FlowInputDefinition(
                    name="items",
                    type=FlowInputType.ARRAY,
                    schema={"type": "array", "minItems": 1},
                ),
                FlowInputDefinition(
                    name="document",
                    type=FlowInputType.FILE,
                    mime_types=("application/pdf",),
                ),
            ),
            nodes=(FlowNodeDefinition(name="start", type="echo"),),
        )

        contract = flow_cmds._flow_task_input(definition)
        schema = cast(Mapping[str, object], contract.schema)
        properties = cast(Mapping[str, object], schema["properties"])

        self.assertEqual(properties["text"], {"type": "string"})
        self.assertEqual(properties["count"], {"type": "integer"})
        self.assertEqual(properties["ratio"], {"type": "number"})
        self.assertEqual(
            properties["payload"],
            {"type": "object", "required": ("answer",)},
        )
        self.assertEqual(
            properties["items"],
            {"type": "array", "minItems": 1},
        )
        document = cast(Mapping[str, object], properties["document"])
        self.assertEqual(document["x-avalan-input-type"], "file")
        self.assertEqual(document["x-avalan-mime-types"], ("application/pdf",))

    def test_flow_metadata_helpers_cover_guard_paths(self) -> None:
        definition = _flow_definition(output_definition=None)
        node = flow_cmds._flow_task_context_metadata_node(
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


class _FakeFlowClientContext:
    database = object()

    def __init__(self, client: object) -> None:
        self.client = client

    async def __aenter__(self) -> object:
        return self.client

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        _ = exc_type, exc, traceback


class _FakeFlowTaskInspection:
    def as_public_dict(self) -> Mapping[str, object]:
        return {
            "task": {"run_id": "run-1"},
            "flow": {"flow_name": "safe-flow", "state": "paused"},
        }


class _FakeFlowTaskExecutor:
    async def inspect(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> _FakeFlowTaskInspection:
        self.run_id = run_id
        self.after_sequence = after_sequence
        return _FakeFlowTaskInspection()

    async def export_trace(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> Mapping[str, object]:
        self.run_id = run_id
        self.after_sequence = after_sequence
        return {"flow_name": "safe-flow", "state": "paused"}


class _SdkParityFlowTaskInspection:
    def __init__(self, flow: Mapping[str, object]) -> None:
        self.flow = flow

    def as_public_dict(self) -> Mapping[str, object]:
        return {"task": {"run_id": "run-1"}, "flow": self.flow}


class _SdkParityFlowTaskExecutor:
    def __init__(
        self,
        inspection: Mapping[str, object],
        trace: Mapping[str, object],
    ) -> None:
        self.inspection = inspection
        self.trace = trace

    async def inspect(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> _SdkParityFlowTaskInspection:
        self.run_id = run_id
        self.after_sequence = after_sequence
        return _SdkParityFlowTaskInspection(self.inspection)

    async def export_trace(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> Mapping[str, object]:
        self.run_id = run_id
        self.after_sequence = after_sequence
        return self.trace


class _FailingFlowTaskExecutor:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def inspect(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> object:
        _ = run_id, after_sequence
        raise self.error

    async def export_trace(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> object:
        _ = run_id, after_sequence
        raise self.error


class _FakeFlowCancelClient:
    cancelled: str | None = None

    async def cancel(self, run_id: str) -> object:
        self.cancelled = run_id
        return SimpleNamespace(
            run_id=run_id,
            state=TaskRunState.CANCEL_REQUESTED,
        )


class _FailingFlowCancelClient:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def cancel(self, run_id: str) -> object:
        _ = run_id
        raise self.error


class _FakeFlowStateStoreContext:
    def __init__(self, store: object) -> None:
        self.store = store

    async def __aenter__(self) -> object:
        return self.store

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        _ = exc_type, exc, traceback


class _SyncFlowDatabase:
    opened = False
    closed = False

    def open(self) -> None:
        self.opened = True

    def aclose(self) -> None:
        self.closed = True


class _FakeFlowStateStore:
    updated_revision: int | None = None
    updated_update: Any = None

    async def get_flow_execution(self, run_id: str) -> object:
        self.run_id = run_id
        return SimpleNamespace(revision=7)

    async def update_flow_execution(
        self,
        run_id: str,
        update: object,
        *,
        expected_revision: int,
    ) -> object:
        self.updated_run_id = run_id
        self.updated_update = update
        self.updated_revision = expected_revision
        return SimpleNamespace(revision=expected_revision + 1)


class _FailingFlowStateStore:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def get_flow_execution(self, run_id: str) -> object:
        _ = run_id
        raise self.error


class _FakeFlowExecutor:
    decisions: object = None

    async def resume(
        self,
        flow: object,
        previous: object,
        *,
        decisions: Mapping[str, Mapping[str, object]],
    ) -> object:
        _ = flow, previous
        self.decisions = decisions
        result = SimpleNamespace(
            trace=FlowExecutionTrace(nodes=()),
            node_outputs={"review": {"result": "approved"}},
            outputs={"answer": "approved"},
            diagnostics=(),
            pause_tokens={},
        )
        return SimpleNamespace(
            ok=True,
            outputs={"answer": "approved"},
            diagnostics=(),
            result=result,
        )


class _FakeDiagnosticFlowExecutor:
    def __init__(
        self,
        *,
        ok: bool,
        diagnostics: tuple[FlowDiagnostic, ...],
    ) -> None:
        self.ok = ok
        self.diagnostics = diagnostics

    async def resume(
        self,
        flow: object,
        previous: object,
        *,
        decisions: Mapping[str, Mapping[str, object]],
    ) -> object:
        _ = flow, previous, decisions
        return SimpleNamespace(
            ok=self.ok,
            outputs={},
            diagnostics=self.diagnostics,
            result=None,
        )


class _SdkParityResumeFlowExecutor:
    def __init__(self, result: object) -> None:
        self.result = result

    async def resume(
        self,
        flow: object,
        previous: object,
        *,
        decisions: Mapping[str, Mapping[str, object]],
    ) -> object:
        self.flow = flow
        self.previous = previous
        self.decisions = decisions
        return self.result


class _FakeFlowDatabase:
    opened = False
    closed = False

    async def open(self) -> None:
        self.opened = True

    async def aclose(self) -> None:
        self.closed = True


def _args(**overrides: object) -> Namespace:
    values = dict(TASK_ARGS)
    values["flow"] = "flow.toml"
    values.update(overrides)
    flow = values["flow"]
    if isinstance(flow, Path):
        values["flow"] = str(flow)
    return Namespace(**values)


def _flow_cli_diagnostic(code: str) -> FlowDiagnostic:
    return FlowDiagnostic(
        code=code,
        path="flow",
        category=FlowDiagnosticCategory.EXECUTION,
        message="Flow command test diagnostic.",
    )


def _sdk_review_registry() -> FlowNodeRegistry:
    registry = default_flow_node_registry()
    registry.register(
        "human_review",
        lambda definition: Node(definition.name),
        metadata=FlowNodeMetadata(
            kind=FlowNodeKind.HUMAN_REVIEW,
            async_only=True,
            capabilities=(FlowNodeCapability.DURABLE_PAUSE,),
            input_contract=FlowNodeContract(
                name="payload",
                type=FlowInputType.OBJECT,
            ),
            output_contract=FlowNodeContract(
                name="result",
                type=FlowOutputType.OBJECT,
            ),
        ),
    )
    return registry


def _sdk_review_definition() -> FlowDefinition:
    decisions = ("approved", "rejected")
    return FlowDefinition(
        name="review",
        version="1",
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="decision", type=FlowOutputType.JSON),
        ),
        entry_behavior=FlowEntryBehavior(node="review"),
        output_behavior=FlowOutputBehavior(
            outputs={"decision": "review.result.decision"},
        ),
        nodes=(
            FlowNodeDefinition(
                name="review",
                type="human_review",
                mappings=(
                    FlowInputMapping(
                        target="payload",
                        kind=FlowMappingKind.SELECT,
                        source="inputs.payload",
                    ),
                ),
                config={
                    "allowed_decisions": decisions,
                    "payload_schema": {"type": "object"},
                    "decision_schema": {
                        "type": "object",
                        "properties": {
                            "decision": {"enum": decisions},
                        },
                        "required": ("decision",),
                    },
                    "timeout_seconds": 60,
                },
            ),
            FlowNodeDefinition(
                name="approved_sink",
                type="constant",
                config={"value": "approved"},
            ),
            FlowNodeDefinition(
                name="rejected_sink",
                type="constant",
                config={"value": "rejected"},
            ),
            FlowNodeDefinition(
                name="timeout_sink",
                type="constant",
                config={"value": "expired"},
            ),
        ),
        edges=(
            FlowEdgeDefinition(
                source="review",
                target="approved_sink",
                label="approved",
                kind=FlowEdgeKind.RESUME,
                priority=0,
            ),
            FlowEdgeDefinition(
                source="review",
                target="rejected_sink",
                label="rejected",
                kind=FlowEdgeKind.RESUME,
                priority=1,
            ),
            FlowEdgeDefinition(
                source="review",
                target="timeout_sink",
                label="expired",
                kind=FlowEdgeKind.TIMEOUT,
            ),
        ),
    )


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


def _write_strict_file_privacy_flow(root: Path) -> Path:
    flow_path = root / "strict-file.flow.toml"
    flow_path.write_text(
        """
        [flow]
        name = "strict_file"
        version = "1"

        [[inputs]]
        name = "input"
        type = "file"
        mime_types = ["application/pdf"]

        [[outputs]]
        name = "result"
        type = "object"

        [outputs.schema]
        type = "object"
        required = ["answer"]

        [outputs.schema.properties.answer]
        type = "string"

        [entry]
        type = "node"
        node = "start"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        result = "start.value"

        [privacy]
        input = "drop"
        prompt = "drop"
        output = "drop"
        files = "drop"
        file_bytes = "drop"
        token_text = "drop"
        tool_arguments = "drop"
        tool_results = "drop"
        events = "drop"
        errors = "drop"
        raw_retention_days = 0

        [nodes.start]
        type = "constant"
        value = {answer = "ok"}
        """,
        encoding="utf-8",
    )
    return flow_path


def _write_strict_constant_flow(root: Path) -> Path:
    flow_path = root / "strict.flow.toml"
    flow_path.write_text(
        """
        [flow]
        name = "strict"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "result"
        type = "object"

        [entry]
        type = "node"
        node = "start"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        result = "start.value"

        [nodes.start]
        type = "constant"
        value = {answer = "ok"}
        """,
        encoding="utf-8",
    )
    return flow_path


def _write_strict_tool_flow(root: Path) -> Path:
    flow_path = root / "strict_tool.flow.toml"
    flow_path.write_text(
        """
        [flow]
        name = "strict_tool"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "json"

        [entry]
        type = "node"
        node = "calculate"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "calculate.result"

        [nodes.calculate]
        type = "tool"
        ref = "flow_cli_adder"

        [nodes.calculate.mapping.arguments]
        type = "object"

        [nodes.calculate.mapping.arguments.fields]
        left = "input.payload.left"
        right = "input.payload.right"

        [nodes.calculate.config.arguments]
        a = "left"
        b = "right"
        """,
        encoding="utf-8",
    )
    return flow_path


def _write_strict_topology_flow(root: Path) -> Path:
    flow_path = root / "topology.flow.toml"
    flow_path.write_text(
        """
        [flow]
        name = "topology"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "result"
        type = "object"

        [entry]
        type = "node"
        node = "A"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        result = "C.value"

        [nodes.A]
        type = "input"

        [nodes.C]
        type = "pass-through"
        input = "A.value"

        [[edges]]
        source = "A"
        target = "C"
        """,
        encoding="utf-8",
    )
    return flow_path


def _write_ambiguous_topology_flow(root: Path) -> Path:
    flow_path = root / "ambiguous.flow.toml"
    flow_path.write_text(
        """
        [flow]
        name = "ambiguous"
        entrypoint = "A"
        output_node = "C"

        [nodes.A]
        type = "echo"

        [nodes.B]
        type = "echo"

        [nodes.C]
        type = "echo"

        [[edges]]
        source = "A"
        target = "C"

        [[edges]]
        source = "B"
        target = "C"
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


class _FlowCliPdfPageConverter:
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


def _flow_cli_page_result(
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
        version = "1"

        [[inputs]]
        name = "payload"
        type = "{input_type}"
        mime_types = ["application/pdf"]

        [[outputs]]
        name = "result"
        type = "{output_type}"
        {schema_line}

        [entry]
        type = "node"
        node = "extract"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        result = "extract.result"

        [nodes.extract]
        type = "agent"
        ref = "{node_ref}"
        input = "input"

        [nodes.extract.mapping]
        input = "input.payload"
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
