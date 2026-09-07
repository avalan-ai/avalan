from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from avalan.task.deployment import DeploymentFileRoot, ExecutionDeploymentError
from avalan.task.deployment_closure import ExecutionClosure


class ExecutionClosureTest(TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.application = self.root / "application"
        self.runtime = self.root / "runtime"
        self.application.mkdir()
        self.runtime.mkdir()
        self.write("agent/templates/agent.md", "base", runtime=True)
        self.write(
            "agent/templates/agent_json.md",
            '{% extends "agent.md" %}',
            runtime=True,
        )

    def write(self, path: str, text: str, *, runtime: bool = False) -> None:
        destination = (self.runtime if runtime else self.application) / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)

    def closure(self) -> ExecutionClosure:
        return ExecutionClosure(self.application, self.runtime)

    def test_nested_flow_agent_schema_and_template_closure(self) -> None:
        self.write(
            "task.toml",
            '[execution]\ntype="flow"\nref="flow.toml"\n[input]\nschema_ref="schema.json"',
        )
        self.write("schema.json", '{"$ref":"nested.json#/$defs/value"}')
        self.write("nested.json", '{"$defs":{"value":{"$ref":"schema.json"}}}')
        self.write(
            "flow.toml",
            '[graph]\npath="graph.mmd"\n[nodes.nested]\ntype="subflow"\nref="nested.toml"\n[nodes.tool]\ntype="tool"\nref="search"',
        )
        self.write("graph.mmd", "graph TD; a-->b")
        self.write(
            "nested.toml", '[nodes.agent]\ntype="agent"\nref="agent.toml"'
        )
        self.write(
            "agent.toml",
            '[agent]\ntype="json"\nuser_template="user.md"\n'
            "[run.response_format.json_schema]\n"
            'schema_ref="schema.json"',
        )
        self.write("user.md", '{% include "parts/first.md" %}')
        self.write("parts/first.md", '{% include "second.md" %}')
        self.write("second.md", "application input")
        self.write("agent/templates/user.md", "runtime input", runtime=True)
        closure = self.closure()
        files = closure.collect("task.toml", file_delivery=True)
        self.assertEqual(closure.tools, {"search"})
        self.assertEqual(
            {file.path for file in files},
            {
                "task.toml",
                "flow.toml",
                "nested.toml",
                "agent.toml",
                "graph.mmd",
                "schema.json",
                "nested.json",
                "user.md",
                "parts/first.md",
                "second.md",
                "agent/templates/agent.md",
                "agent/templates/agent_json.md",
                "agent/templates/user.md",
            },
        )
        self.assertEqual(
            files, closure.collect("task.toml", file_delivery=True)
        )
        self.assertTrue(all(len(file.sha256) == 64 for file in files))
        self.write("second.md", "changed")
        with self.assertRaisesRegex(ExecutionDeploymentError, "file.changed"):
            closure.read(DeploymentFileRoot.APPLICATION, "second.md")

    def test_rejects_unresolved_references_and_unavailable_bytes(self) -> None:
        for ref in (
            '"missing.toml"',
            '"../outside.toml"',
            '"https://example.org/agent.toml"',
            '"{{ dynamic }}"',
            "1",
        ):
            self.write("task.toml", f'[execution]\ntype="agent"\nref={ref}')
            with (
                self.subTest(ref=ref),
                self.assertRaises(ExecutionDeploymentError),
            ):
                self.closure().collect("task.toml")
        self.write("task.toml", '[execution]\ntype="agent"\nref="agent.toml"')
        self.write("agent.toml", '[agent]\nuser_template="user.md"')
        self.write("user.md", "{% include selected %}")
        with self.assertRaisesRegex(ExecutionDeploymentError, "reference"):
            self.closure().collect("task.toml", file_delivery=True)
        self.write("user.md", "{% broken %}")
        with self.assertRaisesRegex(ExecutionDeploymentError, "template"):
            self.closure().collect("task.toml", file_delivery=True)
        self.write("agent.toml", "[agent]\nuser='{% include \"hidden\" %}'")
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "inline.include"
        ):
            self.closure().collect("task.toml")
        self.write("agent.toml", '[agent]\nuser="{% broken %}"')
        with self.assertRaisesRegex(ExecutionDeploymentError, "template"):
            self.closure().collect("task.toml")

    def test_canonical_root_and_invalid_configuration(self) -> None:
        self.write("outside", "unsealed")
        (self.application / "link").symlink_to(self.application / "outside")
        with self.assertRaisesRegex(ExecutionDeploymentError, "file.root"):
            self.closure().read(DeploymentFileRoot.APPLICATION, "link")
        (self.application / "directory").mkdir()
        with self.assertRaisesRegex(ExecutionDeploymentError, "file.type"):
            self.closure().read(DeploymentFileRoot.APPLICATION, "directory")

        for source, error in (
            ("[", "configuration"),
            ('execution="not a section"', "configuration.section"),
            ('[execution]\ntype="callable"\nref="agent.toml"', "target"),
        ):
            self.write("task.toml", source)
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(ExecutionDeploymentError, error),
            ):
                self.closure().collect("task.toml")
        self.write(
            "task.toml",
            '[execution]\ntype="flow"\nref="flow.toml"\n[input]\nschema_ref="schema.json"',
        )
        self.write("schema.json", "{")
        self.write("flow.toml", "[nodes]\ninvalid=1")
        with self.assertRaisesRegex(ExecutionDeploymentError, "flow.nodes"):
            self.closure().collect("task.toml")
        self.write("flow.toml", '[nodes.tool]\ntype="tool"\nref=1')
        with self.assertRaisesRegex(ExecutionDeploymentError, "flow.tool"):
            self.closure().collect("task.toml")
        self.write("flow.toml", "")
        with self.assertRaisesRegex(ExecutionDeploymentError, "schema"):
            self.closure().collect("task.toml")
        self.write("schema.json", '{"allOf":[{"$ref":"#/$defs/value"}]}')
        self.assertEqual(len(self.closure().collect("task.toml")), 3)

    def test_literal_tools_inline_schema_and_repeated_template(self) -> None:
        self.write(
            "task.toml",
            '[execution]\ntype="agent"\nref="agent.toml"\n[input.schema]\ntype="object"',
        )
        self.write(
            "agent.toml",
            '[agent]\nuser_template="user.md"\n[tool]\nenable="search.*"',
        )
        self.write("user.md", '{% include "part.md" %}{% include "part.md" %}')
        self.write("part.md", "retained")
        self.write("agent/templates/user.md", "runtime", runtime=True)
        closure = self.closure()
        closure.collect("task.toml", file_delivery=True)
        self.assertEqual(closure.tool_filters, [("search.*",)])
        self.assertEqual(
            closure._reference("schema.json", "#/$defs/value"), "schema.json"
        )
        self.write("agent.toml", "[agent]\n[tool]\nenable=[1]")
        with self.assertRaisesRegex(ExecutionDeploymentError, "agent.tools"):
            self.closure().collect("task.toml")
