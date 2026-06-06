from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import ToolManagerSettings
from avalan.flow import (
    FLOW_INPUT_KEY,
    FLOW_TOOL_NODE_TYPE,
    FlowDefinition,
    FlowDefinitionLoader,
    FlowInputType,
    FlowLoadError,
    FlowLoadIssueCategory,
    FlowNodeDefinition,
    FlowNodeMetadata,
    FlowNodeRegistry,
    build_flow,
    flow_input_binding,
    load_flow_definition,
    load_flow_definition_result,
    loads_flow_definition,
    loads_flow_definition_result,
    tool_flow_node_registry,
)
from avalan.flow import loader as flow_loader
from avalan.flow.node import Node
from avalan.flow.registry import FlowNodeConfigurationError
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.mcp import McpToolSet

VALID_FLOW = """
[flow]
name = "simple"
version = "1"
entrypoint = "start"
output_node = "finish"

[flow.input]
name = "payload"
type = "object"

[flow.output]
name = "result"
type = "json"

[nodes.start]
type = "input"
input = "payload"

[nodes.finish]
type = "echo"
input = "start"

[[edges]]
source = "start"
target = "finish"
"""


async def loader_adder(a: int, b: int) -> int:
    return a + b


loader_adder.aliases = ["loader_sum"]  # type: ignore[attr-defined]


async def loader_adder_alt(a: int, b: int) -> int:
    return a + b


loader_adder_alt.aliases = ["loader_sum"]  # type: ignore[attr-defined]


async def loader_disabled(a: int) -> int:
    return a


def _tool_loader(
    *,
    enable_tools: list[str] | None = None,
) -> FlowDefinitionLoader:
    manager = ToolManager.create_instance(
        enable_tools=enable_tools or ["loader_adder", "mcp.call"],
        available_toolsets=[
            ToolSet(tools=[loader_adder, loader_adder_alt]),
            ToolSet(namespace="disabled", tools=[loader_disabled]),
            McpToolSet(),
        ],
        settings=ToolManagerSettings(),
    )
    return FlowDefinitionLoader(tool_flow_node_registry(manager))


class FlowDefinitionLoaderTestCase(IsolatedAsyncioTestCase):
    async def test_loads_valid_flow_and_builds_executable_graph(self) -> None:
        result = loads_flow_definition_result(VALID_FLOW)

        self.assertTrue(result.ok)
        assert result.definition is not None
        assert result.flow is not None
        self.assertEqual(result.definition.name, "simple")
        self.assertEqual(result.definition.input.type, FlowInputType.OBJECT)

        output = await result.flow.execute_async(
            initial_node=result.definition.entrypoint,
            initial_inputs=flow_input_binding(
                result.definition.input,
                {"answer": "ok"},
            ),
        )

        self.assertEqual(output, {"answer": "ok"})

    async def test_load_accepts_top_level_input_and_custom_registry(
        self,
    ) -> None:
        def factory(definition: FlowNodeDefinition) -> Node:
            return Node(
                definition.name,
                func=lambda inputs: inputs[FLOW_INPUT_KEY] + "!",
            )

        loader = FlowDefinitionLoader(FlowNodeRegistry({"excited": factory}))
        result = loader.loads_result("""
            [flow]
            name = "custom"
            entrypoint = "start"
            output_node = "start"

            [input]
            name = "value"
            type = "string"

            [output]
            name = "result"
            type = "text"

            [variables]
            owner = "tests"

            [nodes.start]
            type = "excited"
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        assert result.flow is not None
        self.assertEqual(result.definition.variables["owner"], "tests")
        self.assertEqual(
            await result.flow.execute_async(
                initial_node="start",
                initial_inputs=flow_input_binding(
                    result.definition.input,
                    "ready",
                ),
            ),
            "ready!",
        )

    def test_tool_nodes_require_tool_aware_registry(self) -> None:
        result = loads_flow_definition_result(f"""
            [flow]
            name = "tool_node"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "{FLOW_TOOL_NODE_TYPE}"
            ref = "loader_adder"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.unknown_node_type")
        self.assertEqual(result.issues[0].path, "nodes.start.type")

    def test_tool_node_accepts_enabled_ref_and_mcp_remote_name(self) -> None:
        loader = _tool_loader()

        result = loader.loads_result(f"""
            [flow]
            name = "mcp_tool"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "{FLOW_TOOL_NODE_TYPE}"
            ref = "mcp.call"

            [nodes.start.config]
            uri = "http://localhost:4000"
            name = "remote.calculator"
            arguments = {{expression = "1 + 1"}}
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        assert result.flow is not None
        self.assertEqual(result.definition.nodes[0].ref, "mcp.call")
        self.assertEqual(
            result.definition.nodes[0].config["name"],
            "remote.calculator",
        )

    def test_tool_node_rejects_invalid_refs(self) -> None:
        loader = _tool_loader(
            enable_tools=["loader_adder", "loader_adder_alt"]
        )
        cases = (
            ("missing", "", "flow.missing_ref"),
            ("unknown", 'ref = "missing"', "flow.tool_unknown"),
            (
                "disabled",
                'ref = "disabled.loader_disabled"',
                "flow.tool_disabled",
            ),
            ("ambiguous", 'ref = "loader_sum"', "flow.tool_ambiguous"),
            ("path", 'ref = "tools/adder.py"', "flow.invalid_ref"),
            ("uri", 'ref = "urn:mcp:tool"', "flow.invalid_ref"),
        )

        for name, ref_toml, code in cases:
            with self.subTest(name=name):
                result = loader.loads_result(f"""
                    [flow]
                    name = "tool_node"
                    entrypoint = "start"
                    output_node = "start"

                    [nodes.start]
                    type = "{FLOW_TOOL_NODE_TYPE}"
                    {ref_toml}
                    """)

                self.assertFalse(result.ok)
                self.assertEqual(result.issues[0].code, code)
                self.assertEqual(result.issues[0].path, "nodes.start.ref")

    async def test_load_accepts_ref_when_registry_metadata_allows_it(
        self,
    ) -> None:
        def factory(definition: FlowNodeDefinition) -> Node:
            return Node(definition.name, func=lambda _: definition.ref)

        registry = FlowNodeRegistry(
            {"external": factory},
            {"external": FlowNodeMetadata(supports_ref=True)},
        )
        result = FlowDefinitionLoader(registry).loads_result("""
            [flow]
            name = "custom_ref"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "external"
            ref = "safe.toml"
            """)

        self.assertTrue(result.ok)
        assert result.flow is not None
        self.assertEqual(
            await result.flow.execute_async(initial_node="start"),
            "safe.toml",
        )

    def test_load_rejects_ref_when_registry_metadata_does_not_allow_it(
        self,
    ) -> None:
        def factory(definition: FlowNodeDefinition) -> Node:
            return Node(definition.name)

        result = FlowDefinitionLoader(
            FlowNodeRegistry({"external": factory})
        ).loads_result(
            """
            [flow]
            name = "custom_ref"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "external"
            ref = "safe.toml"
            """
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [issue.code for issue in result.issues],
            ["flow.untrusted_callable"],
        )

    async def test_load_accepts_nested_node_config(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "constant"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "constant"

            [nodes.start.config]
            value = "ok"
            """)

        self.assertTrue(result.ok)
        assert result.flow is not None
        self.assertEqual(await result.flow.execute_async(), "ok")

    async def test_load_accepts_shorthand_node_config_and_schema_metadata(
        self,
    ) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "select"
            entrypoint = "start"
            output_node = "pick"

            [flow.input]
            name = "payload"
            type = "object"

            [flow.input.schema]
            type = "object"
            required = ["nested"]

            [flow.input.schema.properties.nested]
            type = "object"

            [flow.output]
            name = "result"
            type = "json"

            [flow.output.schema]
            anyOf = [{type = "string"}, {type = "integer"}]

            [nodes.start]
            type = "constant"
            value = {nested = {items = ["first", "second"]}}

            [nodes.pick]
            type = "select"
            path = "nested.items.1"

            [[edges]]
            source = "start"
            target = "pick"
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        assert result.flow is not None
        assert result.definition.input is not None
        assert result.definition.output is not None
        self.assertEqual(
            result.definition.input.schema["properties"],
            {"nested": {"type": "object"}},
        )
        self.assertEqual(
            result.definition.output.schema["anyOf"],
            ({"type": "string"}, {"type": "integer"}),
        )
        self.assertEqual(
            await result.flow.execute_async(initial_node="start"),
            "second",
        )

    def test_load_and_wrapper_helpers_read_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "flow.toml"
            path.write_text(VALID_FLOW, encoding="utf-8")

            definition = load_flow_definition(path)
            result = load_flow_definition_result(path)
            loaded = loads_flow_definition(VALID_FLOW)

        self.assertIsInstance(definition, FlowDefinition)
        self.assertIsInstance(loaded, FlowDefinition)
        self.assertTrue(result.ok)
        self.assertIsNotNone(result.flow)

    def test_load_wrapper_raises_structured_error_for_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "flow.toml"
            path.write_text("[flow]\nname = 'missing'", encoding="utf-8")

            with self.assertRaises(FlowLoadError) as context:
                FlowDefinitionLoader().load(path)

        self.assertEqual(
            context.exception.issues[0].code, "flow.missing_section"
        )

    def test_loads_wrapper_raises_structured_error(self) -> None:
        with self.assertRaises(FlowLoadError) as context:
            loads_flow_definition("[flow]\nname = 'missing'")

        self.assertEqual(
            context.exception.issues[0].code, "flow.missing_section"
        )
        self.assertIn("flow.missing_section", str(context.exception))

    def test_malformed_toml_uses_safe_generic_path(self) -> None:
        result = loads_flow_definition_result(
            "[flow\nsecret = 'private'",
            source_path="/private/customer.toml",
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.malformed_toml")
        self.assertEqual(result.issues[0].path, "toml")
        self.assertEqual(
            result.issues[0].category, FlowLoadIssueCategory.PARSE
        )
        self.assertNotIn("private", str(result.issues[0].as_dict()))
        self.assertNotIn("customer", str(result.issues[0].as_dict()))

    def test_missing_sections_and_invalid_shapes_are_aggregated(self) -> None:
        result = loads_flow_definition_result("""
            flow = "invalid"
            nodes = "invalid"
            edges = "invalid"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(
            [issue.code for issue in result.issues],
            [
                "flow.invalid_section",
                "flow.invalid_section",
                "flow.invalid_section",
            ],
        )

    def test_rejects_invalid_child_sections_and_missing_fields(self) -> None:
        cases = (
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"
                input = "invalid"

                [nodes.start]
                type = "echo"
                """,
                "flow.invalid_section",
            ),
            (
                """
                [flow]
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                """,
                "flow.missing_field",
            ),
            (
                """
                [flow]
                name = 3
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [flow]
                name = "invalid"
                version = 3
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                """,
                "flow.invalid_type",
            ),
        )

        for source, code in cases:
            with self.subTest(code=code):
                result = loads_flow_definition_result(source)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])

    def test_rejects_duplicate_and_unsupported_sections(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "unsupported"
            entrypoint = "start"
            output_node = "start"

            [flow.input]
            name = "value"
            type = "string"

            [input]
            name = "value"
            type = "string"

            [nodes.start]
            type = "echo"

            [cli]
            runner = "python private.py"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(
            [issue.code for issue in result.issues],
            ["flow.unsupported_section", "flow.duplicate_section"],
        )
        self.assertNotIn("private.py", str(result.issues))

    def test_rejects_unsupported_fields_and_values(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "invalid"
            entrypoint = "start"
            output_node = "start"

            [flow.input]
            name = "document"
            type = "file"
            delivery = "direct"
            memory = false

            [flow.output]
            name = "result"
            type = "unknown"

            [nodes.start]
            type = "agent"
            ref = "../private/agent.toml"
            user_prompt_ref = "private.txt"
            response_format_ref = "format.json"
            """)

        self.assertFalse(result.ok)
        codes = [issue.code for issue in result.issues]
        self.assertIn("flow.unsupported_field", codes)
        self.assertIn("flow.invalid_enum", codes)
        self.assertIn("flow.path_escape", codes)
        self.assertIn("flow.unsupported_node_type", codes)
        self.assertNotIn("private", str(result.issues))

    def test_rejects_invalid_input_and_output_shapes(self) -> None:
        cases = (
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [flow.input]
                name = "document"
                mime_types = "application/pdf"

                [nodes.start]
                type = "echo"
                """,
                ("flow.missing_field", "flow.invalid_type"),
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [flow.input]
                name = "document"
                type = "file"
                mime_types = [3]

                [nodes.start]
                type = "echo"
                """,
                ("flow.invalid_type",),
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [flow.input]
                name = "payload"
                type = "object"
                schema = "invalid"

                [nodes.start]
                type = "echo"
                """,
                ("flow.invalid_type",),
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [flow.output]
                name = "result"

                [nodes.start]
                type = "echo"
                """,
                ("flow.missing_field",),
            ),
        )

        for source, expected_codes in cases:
            with self.subTest(expected_codes=expected_codes):
                result = loads_flow_definition_result(source)
                codes = [issue.code for issue in result.issues]

                self.assertFalse(result.ok)
                for code in expected_codes:
                    self.assertIn(code, codes)

    def test_rejects_unknown_and_untrusted_node_types(self) -> None:
        for node_type, code in (
            ("unknown", "flow.unknown_node_type"),
            ("callable", "flow.untrusted_callable"),
        ):
            with self.subTest(node_type=node_type):
                result = loads_flow_definition_result(f"""
                    [flow]
                    name = "invalid"
                    entrypoint = "start"
                    output_node = "start"

                    [nodes.start]
                    type = "{node_type}"
                    ref = "package.module:function"
                    """)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])
                self.assertNotIn("package.module", str(result.issues))

    def test_rejects_task_scoped_file_conversion_without_task_context(
        self,
    ) -> None:
        for node_type in ("file_convert", "pdf_to_images"):
            with self.subTest(node_type=node_type):
                result = loads_flow_definition_result(f"""
                    [flow]
                    name = "convert"
                    entrypoint = "render"
                    output_node = "render"

                    [nodes.render]
                    type = "{node_type}"

                    [nodes.render.config]
                    converter = "pdf_image"
                    """)

                self.assertFalse(result.ok)
                self.assertEqual(
                    [issue.code for issue in result.issues],
                    ["flow.unsupported_node_type"],
                )

    def test_rejects_invalid_node_and_edge_shapes(self) -> None:
        cases = (
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.""]
                type = "echo"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes]
                start = "invalid"
                """,
                "flow.invalid_section",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                input = "payload"
                """,
                "flow.missing_field",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                config = "invalid"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                ref = "safe.toml"
                """,
                "flow.untrusted_callable",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"

                [edges]
                start = "finish"
                """,
                "flow.invalid_section",
            ),
            (
                """
                edges = ["invalid"]

                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "agent"
                ref = "https://example.invalid/private.toml"
                """,
                "flow.path_escape",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "agent"
                ref = "/private/agent.toml"
                """,
                "flow.path_escape",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"

                [[edges]]
                source = "start"
                """,
                "flow.missing_field",
            ),
        )

        for source, code in cases:
            with self.subTest(code=code):
                result = loads_flow_definition_result(source)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])

    def test_rejects_bad_edge_references_and_cycles(self) -> None:
        cases = (
            (
                """
                [flow]
                name = "bad_source"
                entrypoint = "start"
                output_node = "end"

                [nodes.start]
                type = "echo"

                [nodes.end]
                type = "echo"

                [[edges]]
                source = "missing"
                target = "end"
                """,
                "flow.bad_reference",
            ),
            (
                """
                [flow]
                name = "bad_edge"
                entrypoint = "start"
                output_node = "end"

                [nodes.start]
                type = "echo"

                [[edges]]
                source = "start"
                target = "missing"
                """,
                "flow.bad_reference",
            ),
            (
                """
                [flow]
                name = "cycle"
                entrypoint = "a"
                output_node = "b"

                [nodes.a]
                type = "echo"

                [nodes.b]
                type = "echo"

                [[edges]]
                source = "a"
                target = "b"

                [[edges]]
                source = "b"
                target = "a"
                """,
                "flow.invalid_entrypoint",
            ),
        )

        for source, code in cases:
            with self.subTest(code=code):
                result = loads_flow_definition_result(source)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])

    def test_rejects_unknown_entrypoint_and_output_node(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "unknown_refs"
            entrypoint = "missing_start"
            output_node = "missing_end"

            [nodes.start]
            type = "echo"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(
            [issue.code for issue in result.issues],
            ["flow.unknown_entrypoint", "flow.unknown_output_node"],
        )

    def test_rejects_multiple_and_non_terminal_outputs(self) -> None:
        cases = (
            (
                """
                [flow]
                name = "multiple"
                entrypoint = "start"
                output_node = "left"

                [nodes.start]
                type = "echo"

                [nodes.left]
                type = "echo"

                [nodes.right]
                type = "echo"

                [[edges]]
                source = "start"
                target = "left"

                [[edges]]
                source = "start"
                target = "right"
                """,
                "flow.multiple_outputs",
            ),
            (
                """
                [flow]
                name = "non_terminal"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"

                [nodes.end]
                type = "echo"

                [[edges]]
                source = "start"
                target = "end"
                """,
                "flow.invalid_output_node",
            ),
        )

        for source, code in cases:
            with self.subTest(code=code):
                result = loads_flow_definition_result(source)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])

    def test_invalid_node_factory_returns_stable_issue(self) -> None:
        def factory(_: FlowNodeDefinition) -> Node:
            raise AssertionError("private factory failure")

        loader = FlowDefinitionLoader(FlowNodeRegistry({"broken": factory}))

        result = loader.loads_result("""
            [flow]
            name = "broken"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "broken"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.invalid_node")
        self.assertNotIn("private factory failure", str(result.issues[0]))

    def test_node_configuration_error_returns_declared_issue(self) -> None:
        def factory(_: FlowNodeDefinition) -> Node:
            raise FlowNodeConfigurationError(
                code="flow.invalid_node",
                path="nodes.start.config",
                message="Flow node configuration is invalid.",
                hint="Fix the node configuration.",
            )

        loader = FlowDefinitionLoader(FlowNodeRegistry({"broken": factory}))

        result = loader.loads_result("""
            [flow]
            name = "broken"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "broken"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.invalid_node")
        self.assertEqual(result.issues[0].path, "nodes.start.config")
        self.assertEqual(
            result.issues[0].hint,
            "Fix the node configuration.",
        )

    def test_rejects_invalid_agent_file_selectors(self) -> None:
        cases: tuple[tuple[str, object, tuple[str, ...]], ...] = (
            ("non_string", 3, ("start",)),
            ("invalid", "start", ("start",)),
            ("reserved", "file.content", ("start",)),
            ("unknown", "missing.content", ("start",)),
            ("disconnected", "start.content", ("middle",)),
        )

        for name, selector, edge_sources in cases:
            with self.subTest(name=name):
                result = self._load_agent_selector_case(
                    selector,
                    edge_sources=edge_sources,
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    result.issues[0].code,
                    {
                        "flow.bad_reference",
                        "flow.invalid_node",
                        "flow.invalid_type",
                    },
                )
                self.assertEqual(
                    result.issues[0].path,
                    "nodes.agent.config.files_input",
                )

    def test_private_helpers_cover_toml_impossible_shapes(self) -> None:
        issues: list[flow_loader.FlowLoadIssue] = []
        raw = {
            1: "ignored",
            "flow": {
                "name": "raw",
                "entrypoint": "start",
                "output_node": "start",
            },
            "nodes": {"start": {"type": "echo"}},
        }

        result = flow_loader._build_result(  # type: ignore[attr-defined]
            raw,  # type: ignore[arg-type]
            registry=FlowNodeRegistry(),
            source_path=None,
        )
        tuple_value = flow_loader._string_tuple(  # type: ignore[attr-defined]
            {"mime_types": None},
            "flow.input.mime_types",
            "mime_types",
            issues,
        )
        list_value = flow_loader._string_tuple(  # type: ignore[attr-defined]
            {"mime_types": ["text/plain"]},
            "flow.input.mime_types",
            "mime_types",
            issues,
        )
        metadata = flow_loader._metadata(  # type: ignore[attr-defined]
            {1: "bad"},  # type: ignore[dict-item]
            "metadata",
            issues,
        )

        self.assertFalse(result.ok)
        self.assertEqual(tuple_value, ())
        self.assertEqual(list_value, ("text/plain",))
        self.assertIsNone(metadata)
        self.assertIn("flow.invalid_type", [issue.code for issue in issues])

    def _load_agent_selector_case(
        self,
        selector: object,
        *,
        edge_sources: tuple[str, ...],
    ) -> flow_loader.FlowLoadResult:
        def agent_factory(definition: FlowNodeDefinition) -> Node:
            return Node(definition.name)

        def echo_factory(definition: FlowNodeDefinition) -> Node:
            return Node(definition.name)

        selector_toml = (
            f'"{selector}"' if isinstance(selector, str) else str(selector)
        )
        edge_toml = "\n".join(f"""
            [[edges]]
            source = "{source}"
            target = "agent"
            """ for source in edge_sources)
        middle_node_toml = ""
        if "middle" in edge_sources:
            middle_node_toml = """
            [nodes.middle]
            type = "echo"

            [[edges]]
            source = "start"
            target = "middle"
            """
        return FlowDefinitionLoader(
            FlowNodeRegistry(
                {
                    "agent": agent_factory,
                    "echo": echo_factory,
                }
            )
        ).loads_result(
            f"""
            [flow]
            name = "agent_selector"
            entrypoint = "start"
            output_node = "agent"

            [nodes.start]
            type = "echo"

            {middle_node_toml}

            [nodes.agent]
            type = "agent"

            [nodes.agent.config]
            files_input = {selector_toml}
            {edge_toml}
            """
        )


class FlowBuildTestCase(TestCase):
    def test_build_flow_from_definition(self) -> None:
        definition = FlowDefinition(
            name="manual",
            entrypoint="start",
            output_node="start",
            nodes=(
                FlowNodeDefinition(
                    name="start",
                    type="constant",
                    config={"value": "ok"},
                ),
            ),
        )

        flow = build_flow(definition)

        self.assertEqual(flow.execute(), "ok")


if __name__ == "__main__":
    main()
