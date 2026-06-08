from argparse import Namespace
from asyncio import run as asyncio_run
from base64 import b64decode
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
)
from avalan.flow import (
    FlowDefinition,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeDefinition,
    FlowEntryBehavior,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowJoinPolicy,
    FlowJoinPolicyType,
    FlowLoadIssue,
    FlowLoadIssueCategory,
    FlowLoopPolicy,
    FlowMappingKind,
    FlowNodeDefinition,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowRetryBackoffStrategy,
    FlowRetryPolicy,
    FlowTimeoutPolicy,
    MermaidRenderResult,
)
from avalan.task import (
    TaskInputType,
    TaskOutputType,
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
