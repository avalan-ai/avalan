from asyncio import CancelledError
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallDiagnosticCode,
    ToolFilter,
    ToolFilterResult,
    ToolManagerSettings,
)
from avalan.flow import (
    FLOW_INPUT_KEY,
    FLOW_TOOL_NODE_TYPE,
    FlowConditionOperator,
    FlowDefinition,
    FlowDefinitionLoader,
    FlowDiagnosticCategory,
    FlowEdgeKind,
    FlowInputType,
    FlowJoinPolicyType,
    FlowLoadError,
    FlowLoadIssueCategory,
    FlowLoopPolicy,
    FlowNodeDefinition,
    FlowNodeMetadata,
    FlowNodeRegistry,
    FlowRetryBackoffStrategy,
    FlowRouteMatchPolicy,
    FlowTimeoutPolicy,
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


async def loader_multiplier(a: int, b: int) -> int:
    return a * b


async def loader_status() -> str:
    return "ready"


async def loader_disabled(a: int) -> int:
    return a


ToolFilterCallable = Callable[
    [ToolCall, ToolCallContext],
    tuple[ToolCall, ToolCallContext] | ToolFilterResult | None,
]
ToolFilterConfig = list[ToolFilterCallable | ToolFilter]


def _tool_loader(
    *,
    enable_tools: list[str] | None = None,
    filters: ToolFilterConfig | None = None,
) -> FlowDefinitionLoader:
    manager = ToolManager.create_instance(
        enable_tools=enable_tools or ["loader_adder", "mcp.call"],
        available_toolsets=[
            ToolSet(
                tools=[
                    loader_adder,
                    loader_adder_alt,
                    loader_multiplier,
                    loader_status,
                ]
            ),
            ToolSet(namespace="disabled", tools=[loader_disabled]),
            McpToolSet(),
        ],
        settings=ToolManagerSettings(filters=filters),
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

    def test_loads_validation_result_does_not_build_nodes(self) -> None:
        build_calls = 0

        def factory(definition: FlowNodeDefinition) -> Node:
            nonlocal build_calls
            _ = definition
            build_calls += 1
            raise AssertionError("factory should not build")

        loader = FlowDefinitionLoader(
            FlowNodeRegistry(
                {"external": factory},
                {"external": FlowNodeMetadata(supports_ref=True)},
            )
        )
        source = """
            [flow]
            name = "validation_only"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "external"
            ref = "safe.toml"
            """

        result = loader.loads_validation_result(source)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.toml"
            path.write_text(source, encoding="utf-8")
            file_result = loader.load_validation_result(path)

        self.assertTrue(result.ok)
        self.assertIsNone(result.flow)
        self.assertTrue(file_result.ok)
        self.assertIsNone(file_result.flow)
        self.assertEqual(build_calls, 0)

    async def test_loads_edge_conditions_and_applies_them(self) -> None:
        cases = (("pass", "go", "go"), ("block", "stop", None))

        for name, expected, output in cases:
            with self.subTest(name=name):
                result = loads_flow_definition_result(f"""
                    [flow]
                    name = "conditioned"
                    entrypoint = "start"
                    output_node = "finish"

                    [nodes.start]
                    type = "constant"
                    value = "go"

                    [nodes.finish]
                    type = "echo"

                    [[edges]]
                    source = "start"
                    target = "finish"

                    [edges.condition]
                    op = "eq"
                    selector = "start.value"
                    value = "{expected}"
                    """)

                self.assertTrue(result.ok)
                assert result.definition is not None
                assert result.flow is not None
                assert result.definition.edges[0].condition is not None
                self.assertEqual(
                    result.definition.edges[0].condition.operator,
                    FlowConditionOperator.EQ,
                )
                self.assertEqual(
                    await result.flow.execute_async(
                        initial_node=result.definition.entrypoint,
                    ),
                    output,
                )

    def test_loads_nested_edge_conditions(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "conditioned"
            entrypoint = "start"
            output_node = "finish"

            [nodes.start]
            type = "constant"
            value = "go"

            [nodes.finish]
            type = "echo"

            [[edges]]
            source = "start"
            target = "finish"

            [edges.condition]
            op = "all"

            [[edges.condition.conditions]]
            op = "is_type"
            selector = "start.value"
            value_type = "string"

            [[edges.condition.conditions]]
            op = "in"
            selector = "start.value"
            values = ["go", "stop"]
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        condition = result.definition.edges[0].condition
        assert condition is not None
        self.assertEqual(condition.operator.value, "all")
        self.assertEqual(len(condition.conditions), 2)

    def test_loader_rejects_invalid_edge_conditions(self) -> None:
        cases = (
            (
                """
                condition = "invalid"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [edges.condition]
                op = "eval"
                selector = "start.value"
                value = "private-token"
                """,
                "flow.invalid_enum",
            ),
            (
                """
                [edges.condition]
                op = "eq"
                selector = "start.value"
                eval = "import os"
                """,
                "flow.unsupported_field",
            ),
            (
                """
                [edges.condition]
                op = "all"
                conditions = "invalid"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [edges.condition]
                op = "in"
                selector = "start.value"
                values = "invalid"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [edges.condition]
                op = "all"

                [[edges.condition.conditions]]
                selector = "start.value"
                """,
                "flow.missing_field",
            ),
        )

        for condition_toml, code in cases:
            with self.subTest(code=code):
                result = loads_flow_definition_result(f"""
                    [flow]
                    name = "conditioned"
                    entrypoint = "start"
                    output_node = "finish"

                    [nodes.start]
                    type = "constant"
                    value = "go"

                    [nodes.finish]
                    type = "echo"

                    [[edges]]
                    source = "start"
                    target = "finish"
                    {condition_toml}
                    """)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])
                self.assertNotIn(
                    "private-token", str(result.public_diagnostics)
                )

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
        self.assertEqual(result.issues[0].code, "flow.unsupported_node_type")
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

            [nodes.start.config.arguments]
            uri = "uri"
            name = "remote_name"
            arguments = "remote_arguments"
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        assert result.flow is not None
        self.assertEqual(result.definition.nodes[0].ref, "mcp.call")
        self.assertEqual(
            result.definition.nodes[0].config["arguments"],
            {
                "uri": "uri",
                "name": "remote_name",
                "arguments": "remote_arguments",
            },
        )

    async def test_tool_node_executes_from_loaded_flow(self) -> None:
        loader = _tool_loader()

        result = loader.loads_result(f"""
            [flow]
            name = "tool_node"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "{FLOW_TOOL_NODE_TYPE}"
            ref = "loader_adder"

            [nodes.start.config.arguments]
            a = "left"
            b = "right"
            """)

        self.assertTrue(result.ok)
        assert result.flow is not None
        self.assertEqual(
            await result.flow.execute_async(
                initial_node="start",
                initial_inputs={"left": 2, "right": 5},
            ),
            7,
        )

    async def test_loaded_tool_node_validates_arguments_before_filters(
        self,
    ) -> None:
        events: list[str] = []

        async def checked_value(value: int) -> int:
            events.append("tool")
            return value + 1

        def repair_argument(
            call: ToolCall,
            context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            events.append("filter")
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={"value": 3},
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["checked_value"],
            available_toolsets=[ToolSet(tools=[checked_value])],
            settings=ToolManagerSettings(filters=[repair_argument]),
        )
        loader = FlowDefinitionLoader(tool_flow_node_registry(manager))

        result = loader.loads_result(f"""
            [flow]
            name = "tool_node"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "{FLOW_TOOL_NODE_TYPE}"
            ref = "checked_value"

            [nodes.start.config]
            output_mode = "envelope"

            [nodes.start.config.arguments]
            value = "value"
            """)

        self.assertTrue(result.ok)
        assert result.flow is not None
        output = await result.flow.execute_async(
            initial_node="start",
            initial_inputs={"payload": {"value": "bad"}},
        )

        assert isinstance(output, dict)
        diagnostic = output["diagnostic"]
        assert isinstance(diagnostic, dict)
        self.assertEqual(output["status"], "diagnostic")
        self.assertEqual(
            diagnostic["code"],
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED.value,
        )
        self.assertEqual(events, [])

    async def test_tool_node_executes_no_argument_tool_from_loaded_flow(
        self,
    ) -> None:
        loader = _tool_loader(enable_tools=["loader_status"])

        result = loader.loads_result(f"""
            [flow]
            name = "tool_node"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "{FLOW_TOOL_NODE_TYPE}"
            ref = "loader_status"
            """)

        self.assertTrue(result.ok)
        assert result.flow is not None
        self.assertEqual(
            await result.flow.execute_async(initial_node="start"),
            "ready",
        )

    async def test_tool_node_blocks_provider_name_rewrite_from_loaded_flow(
        self,
    ) -> None:
        def set_provider_name(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    provider_name="loader_multiplier",
                ),
                ToolCallContext(),
            )

        loader = _tool_loader(
            enable_tools=["loader_adder", "loader_multiplier"],
            filters=[set_provider_name],
        )

        result = loader.loads_result(f"""
            [flow]
            name = "tool_node"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "{FLOW_TOOL_NODE_TYPE}"
            ref = "loader_adder"

            [nodes.start.config]
            output_mode = "envelope"

            [nodes.start.config.arguments]
            a = "left"
            b = "right"
            """)

        self.assertTrue(result.ok)
        assert result.flow is not None
        output = await result.flow.execute_async(
            initial_node="start",
            initial_inputs={"left": 2, "right": 5},
        )

        assert isinstance(output, dict)
        diagnostic = output["diagnostic"]
        assert isinstance(diagnostic, dict)
        self.assertEqual(output["status"], "diagnostic")
        self.assertEqual(output["result"], None)
        self.assertEqual(output["canonical_name"], "loader_adder")
        self.assertEqual(
            diagnostic["code"],
            ToolCallDiagnosticCode.FILTER_SUPPRESSED.value,
        )
        self.assertEqual(
            diagnostic["details"], {"filtered_name": "loader_multiplier"}
        )

    async def test_loaded_tool_node_matches_filter_after_alias_rewrite(
        self,
    ) -> None:
        called: list[str] = []

        def set_alias(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            called.append("alias")
            return (
                ToolCall(
                    id=call.id,
                    name="loader_sum",
                    arguments=call.arguments,
                ),
                ToolCallContext(),
            )

        def adjust_argument(
            call: ToolCall,
            context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            assert call.arguments is not None
            called.append("adder")
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={
                        "a": call.arguments["a"],
                        "b": 8,
                    },
                ),
                context,
            )

        def unrelated(
            call: ToolCall,
            context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            called.append("multiplier")
            return call, context

        loader = _tool_loader(
            enable_tools=["loader_adder"],
            filters=[
                set_alias,
                ToolFilter(func=unrelated, namespace="loader_multiplier"),
                ToolFilter(func=adjust_argument, namespace="loader_adder"),
            ],
        )

        result = loader.loads_result(f"""
            [flow]
            name = "tool_node"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "{FLOW_TOOL_NODE_TYPE}"
            ref = "loader_adder"

            [nodes.start.config.arguments]
            a = "left"
            b = "right"
            """)

        self.assertTrue(result.ok)
        assert result.flow is not None
        output = await result.flow.execute_async(
            initial_node="start",
            initial_inputs={"left": 2, "right": 5},
        )

        self.assertEqual(output, 10)
        self.assertEqual(called, ["alias", "adder"])

    async def test_loaded_tool_node_propagates_cancellation_after_filter(
        self,
    ) -> None:
        called: list[str] = []

        def replace_context(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            called.append("filter")
            return call, ToolCallContext()

        async def cancel() -> None:
            called.append("cancel")
            if called.count("cancel") == 6:
                raise CancelledError()

        loader = _tool_loader(
            enable_tools=["loader_adder"],
            filters=[replace_context],
        )
        result = loader.loads_result(f"""
            [flow]
            name = "tool_node"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "{FLOW_TOOL_NODE_TYPE}"
            ref = "loader_adder"

            [nodes.start.config]
            output_mode = "envelope"

            [nodes.start.config.arguments]
            a = "left"
            b = "right"
            """)

        self.assertTrue(result.ok)
        assert result.flow is not None
        with self.assertRaises(CancelledError):
            await result.flow.execute_async(
                initial_node="start",
                initial_inputs={"left": 2, "right": 5},
                cancellation_checker=cancel,
            )

        self.assertEqual(
            called,
            [
                "cancel",
                "cancel",
                "cancel",
                "cancel",
                "cancel",
                "filter",
                "cancel",
            ],
        )

    def test_tool_node_rejects_invalid_argument_bindings_on_load(
        self,
    ) -> None:
        loader = _tool_loader()
        cases = (
            (
                "unknown",
                """
                [nodes.start.config.arguments]
                expression = "left"
                a = "left"
                b = "right"
                """,
                "flow.unknown_argument_binding",
                "nodes.start.config.arguments.expression",
            ),
            (
                "missing",
                """
                [nodes.start.config.arguments]
                a = "left"
                """,
                "flow.missing_argument_binding",
                "nodes.start.config.arguments.b",
            ),
        )

        for name, config, code, path in cases:
            with self.subTest(name=name):
                result = loader.loads_result(f"""
                    [flow]
                    name = "tool_node"
                    entrypoint = "start"
                    output_node = "start"

                    [nodes.start]
                    type = "{FLOW_TOOL_NODE_TYPE}"
                    ref = "loader_adder"

                    {config}
                    """)

                self.assertFalse(result.ok)
                self.assertEqual(result.issues[0].code, code)
                self.assertEqual(result.issues[0].path, path)

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
        self.assertEqual(
            result.diagnostics[0].category,
            FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
        )
        self.assertEqual(
            result.public_diagnostics,
            (result.issues[0].as_public_diagnostic_dict(),),
        )
        self.assertNotIn("private", str(result.issues[0].as_dict()))
        self.assertNotIn("customer", str(result.issues[0].as_dict()))
        self.assertNotIn("private", str(result.public_diagnostics))
        self.assertNotIn("customer", str(result.public_diagnostics))

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
            (
                """
                [flow]
                name = "missing"

                [nodes.start]
                type = "echo"
                """,
                "flow.missing_field",
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

    def test_loads_strict_definition_contract(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "strict"
            version = "2026-06-07"
            tags = ["ops"]

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "finish.result"

            [runtime_limits]
            timeout_seconds = 30

            [privacy]
            store_raw = false

            [observability]
            events = ["node"]

            [ownership]
            team = "platform"

            [nodes.start]
            type = "echo"

            [nodes.finish]
            type = "echo"

            [[edges]]
            source = "start"
            target = "finish"
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        self.assertIsNotNone(result.flow)
        self.assertIsNone(result.definition.entrypoint)
        self.assertIsNone(result.definition.output_node)
        self.assertEqual(result.definition.inputs[0].name, "payload")
        self.assertEqual(result.definition.outputs[0].name, "answer")
        self.assertEqual(result.definition.tags, ("ops",))
        self.assertEqual(
            result.definition.runtime_limits["timeout_seconds"],
            30,
        )
        self.assertEqual(result.definition.privacy_policy["store_raw"], False)
        self.assertEqual(
            result.definition.observability_policy["events"],
            ("node",),
        )
        self.assertEqual(result.definition.ownership["team"], "platform")

    def test_loads_strict_definition_with_revision_identity(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "strict"
            revision = "rev-1"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "start.result"

            [nodes.start]
            type = "echo"
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        self.assertEqual(result.definition.revision, "rev-1")

    def test_loads_strict_edge_routing_policy(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "strict"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "finish.result"

            [nodes.start]
            type = "echo"

            [nodes.finish]
            type = "echo"

            [[edges]]
            source = "start"
            target = "finish"
            kind = "error"
            priority = 3
            routing_policy = "all_matching"
            default = true
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        edge = result.definition.edges[0]
        self.assertEqual(edge.kind, FlowEdgeKind.ERROR)
        self.assertEqual(edge.priority, 3)
        self.assertEqual(
            edge.routing_policy, FlowRouteMatchPolicy.ALL_MATCHING
        )
        self.assertTrue(edge.default)

    def test_loads_strict_node_policies(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "strict"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "finish.result"

            [nodes.start]
            type = "echo"

            [nodes.start.retry_policy]
            max_attempts = 3
            backoff = "exponential"
            initial_delay_seconds = 1
            max_delay_seconds = 8
            retryable_categories = ["transient"]
            non_retryable_categories = ["validation"]
            exhausted_route = "failed"

            [nodes.start.timeout_policy]
            per_attempt_seconds = 30

            [nodes.start.loop_policy]
            max_iterations = 4
            max_elapsed_seconds = 60
            output_selector = "start.result"
            limit_route = "limited"

            [nodes.start.loop_policy.continue_condition]
            op = "exists"
            selector = "start.result.more"

            [nodes.start.loop_policy.exit_condition]
            op = "exists"
            selector = "start.result.done"

            [nodes.finish]
            type = "echo"

            [nodes.failed]
            type = "echo"

            [nodes.limited]
            type = "echo"

            [[edges]]
            source = "start"
            target = "finish"

            [[edges]]
            source = "start"
            target = "failed"
            kind = "error"

            [[edges]]
            source = "start"
            target = "limited"
            kind = "timeout"
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        node = result.definition.nodes[0]
        assert node.retry_policy is not None
        self.assertEqual(node.retry_policy.max_attempts, 3)
        self.assertEqual(
            node.retry_policy.backoff,
            FlowRetryBackoffStrategy.EXPONENTIAL,
        )
        self.assertEqual(node.retry_policy.exhausted_route, "failed")
        self.assertEqual(
            node.timeout_policy,
            FlowTimeoutPolicy(per_attempt_seconds=30),
        )
        self.assertIsInstance(node.loop_policy, FlowLoopPolicy)
        assert node.loop_policy is not None
        self.assertEqual(node.loop_policy.limit_route, "limited")
        assert node.loop_policy.continue_condition is not None
        self.assertEqual(
            node.loop_policy.continue_condition.operator,
            FlowConditionOperator.EXISTS,
        )

    def test_loads_strict_join_policy(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "strict"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "left"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "finish.value"

            [nodes.left]
            type = "echo"

            [nodes.right]
            type = "echo"

            [nodes.finish]
            type = "echo"

            [nodes.finish.join_policy]
            type = "quorum"
            quorum = 2
            optional_inputs = ["value"]

            [[edges]]
            source = "left"
            target = "finish"

            [[edges]]
            source = "right"
            target = "finish"
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        join_policy = result.definition.nodes[2].join_policy
        assert join_policy is not None
        self.assertEqual(join_policy.type, FlowJoinPolicyType.QUORUM)
        self.assertEqual(join_policy.quorum, 2)
        self.assertEqual(join_policy.optional_inputs, ("value",))

    def test_loads_declarative_node_mappings(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "strict"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "finish.value"

            [nodes.start]
            type = "input"

            [nodes.finish]
            type = "select"

            [nodes.finish.mapping.value]
            type = "object"

            [nodes.finish.mapping.value.fields]
            name = "input.payload.name"
            first = "start.value.items[0]"

            [[edges]]
            source = "start"
            target = "finish"
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        mapping = result.definition.nodes[1].mappings[0]
        self.assertEqual(mapping.target, "value")
        self.assertEqual(mapping.fields["name"], "input.payload.name")
        self.assertEqual(mapping.fields["first"], "start.value.items[0]")

    def test_loads_shorthand_node_mapping(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "strict"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "finish.value"

            [nodes.start]
            type = "input"

            [nodes.finish]
            type = "select"

            [nodes.finish.mapping]
            value = "input.payload"

            [[edges]]
            source = "start"
            target = "finish"
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        self.assertEqual(
            result.definition.nodes[1].mappings[0].source,
            "input.payload",
        )

    def test_loader_rejects_invalid_declarative_node_mappings(self) -> None:
        cases = (
            (
                """
                [nodes.finish]
                type = "select"
                mapping = "invalid"
                """,
                ["flow.invalid_type"],
            ),
            (
                """
                [nodes.finish]
                type = "select"

                [nodes.finish.mapping.value]
                type = "object"
                secret = "private-token"
                """,
                ["flow.unsupported_field", "flow.empty_mapping"],
            ),
            (
                """
                [nodes.finish]
                type = "select"

                [nodes.finish.mapping.value]
                type = "select"
                source = "env.SECRET"
                """,
                ["flow.reserved_selector"],
            ),
            (
                """
                [nodes.finish]
                type = "select"

                [nodes.finish.mapping.value]
                type = "array"
                items = ["input.payload"]
                """,
                ["flow.incompatible_mapping"],
            ),
            (
                """
                [nodes.finish]
                type = "select"

                [nodes.finish.mapping]
                value = 3
                """,
                ["flow.invalid_type"],
            ),
            (
                """
                [nodes.finish]
                type = "select"

                [nodes.finish.mapping.value]
                source = "input.payload"
                """,
                ["flow.missing_field"],
            ),
            (
                """
                [nodes.finish]
                type = "select"

                [nodes.finish.mapping.value]
                type = "object"
                fields = "invalid"
                """,
                ["flow.invalid_type", "flow.empty_mapping"],
            ),
            (
                """
                [nodes.finish]
                type = "select"

                [nodes.finish.mapping.value]
                type = "object"

                [nodes.finish.mapping.value.fields]
                name = 3
                """,
                ["flow.invalid_type", "flow.empty_mapping"],
            ),
        )
        for node_source, expected_codes in cases:
            with self.subTest(expected_codes=expected_codes):
                result = loads_flow_definition_result(f"""
                    [flow]
                    name = "strict"
                    version = "2026-06-07"

                    [[inputs]]
                    name = "payload"
                    type = "object"

                    [[outputs]]
                    name = "answer"
                    type = "object"

                    [entry]
                    type = "node"
                    node = "start"

                    [output_behavior]
                    type = "map"

                    [output_behavior.outputs]
                    answer = "finish.value"

                    [nodes.start]
                    type = "input"

                    {node_source}

                    [[edges]]
                    source = "start"
                    target = "finish"
                    """)

                self.assertFalse(result.ok)
                codes = [issue.code for issue in result.issues]
                for code in expected_codes:
                    self.assertIn(code, codes)
                self.assertNotIn(
                    "private-token", str(result.public_diagnostics)
                )
                self.assertNotIn("SECRET", str(result.public_diagnostics))

    def test_strict_loader_rejects_scalar_input_and_output_aliases(
        self,
    ) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "strict"
            version = "2026-06-07"

            [flow.input]
            name = "payload"
            type = "object"

            [flow.output]
            name = "answer"
            type = "object"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "start.result"

            [nodes.start]
            type = "echo"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(
            [issue.code for issue in result.issues],
            ["flow.scalar_input_alias", "flow.scalar_output_alias"],
        )

    def test_strict_loader_rejects_invalid_behavior_shapes(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "strict"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "terminal"
            node = "start"
            strategy = "implicit"

            [output_behavior]
            type = "terminal"
            result = "start.result"

            [output_behavior.outputs]
            answer = 3

            [nodes.start]
            type = "echo"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(
            [issue.code for issue in result.issues],
            [
                "flow.unsupported_field",
                "flow.invalid_enum",
                "flow.unsupported_field",
                "flow.invalid_enum",
                "flow.invalid_type",
                "flow.missing_entry_behavior",
                "flow.missing_output_behavior",
            ],
        )
        self.assertNotIn("implicit", str(result.public_diagnostics))

    def test_strict_loader_rejects_invalid_input_output_arrays(self) -> None:
        cases = (
            (
                """
                inputs = "invalid"

                [flow]
                name = "strict"
                version = "2026-06-07"

                [[outputs]]
                name = "answer"
                type = "object"

                [entry]
                type = "node"
                node = "start"

                [output_behavior]
                type = "map"

                [output_behavior.outputs]
                answer = "start.result"

                [nodes.start]
                type = "echo"
                """,
                ["flow.invalid_section", "flow.missing_inputs"],
            ),
            (
                """
                inputs = ["invalid"]

                [flow]
                name = "strict"
                version = "2026-06-07"

                [[outputs]]
                name = "answer"
                type = "object"

                [entry]
                type = "node"
                node = "start"

                [output_behavior]
                type = "map"

                [output_behavior.outputs]
                answer = "start.result"

                [nodes.start]
                type = "echo"
                """,
                ["flow.invalid_section", "flow.missing_inputs"],
            ),
        )
        for source, expected_codes in cases:
            with self.subTest(expected_codes=expected_codes):
                result = loads_flow_definition_result(source)

                self.assertFalse(result.ok)
                self.assertEqual(
                    [issue.code for issue in result.issues],
                    expected_codes,
                )

    def test_strict_loader_rejects_invalid_output_maps(self) -> None:
        cases = (
            (
                """
                [flow]
                name = "strict"
                version = "2026-06-07"

                [[inputs]]
                name = "payload"
                type = "object"

                [[outputs]]
                name = "answer"
                type = "object"

                [entry]
                type = "node"
                node = "start"

                [output_behavior]
                type = "map"

                [nodes.start]
                type = "echo"
                """,
                ["flow.missing_field", "flow.missing_output_behavior"],
            ),
            (
                """
                [flow]
                name = "strict"
                version = "2026-06-07"

                [[inputs]]
                name = "payload"
                type = "object"

                [[outputs]]
                name = "answer"
                type = "object"

                [entry]
                type = "node"
                node = "start"

                [output_behavior]
                type = "map"
                outputs = "invalid"

                [nodes.start]
                type = "echo"
                """,
                ["flow.invalid_type", "flow.missing_output_behavior"],
            ),
        )
        for source, expected_codes in cases:
            with self.subTest(expected_codes=expected_codes):
                result = loads_flow_definition_result(source)

                self.assertFalse(result.ok)
                self.assertEqual(
                    [issue.code for issue in result.issues],
                    expected_codes,
                )

    def test_strict_loader_rejects_missing_output_selection(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "strict"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [nodes.start]
            type = "echo"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(
            [issue.code for issue in result.issues],
            ["flow.missing_output_behavior"],
        )

    def test_rejects_unknown_and_untrusted_node_types(self) -> None:
        cases = (
            ("unknown", "flow.unknown_node_type"),
            ("callable", "flow.untrusted_callable"),
            ("function", "flow.untrusted_callable"),
            ("module", "flow.untrusted_callable"),
            ("python", "flow.untrusted_callable"),
            ("python_callable", "flow.untrusted_callable"),
        )
        for node_type, code in cases:
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

    def test_privacy_load_issues_project_to_privacy_diagnostics(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "invalid"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "echo"
            ref = "../private-token"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.path_escape")
        self.assertEqual(
            result.diagnostics[0].category, FlowDiagnosticCategory.PRIVACY
        )
        self.assertNotIn("private-token", str(result.public_diagnostics))

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

    def test_loader_rejects_invalid_edge_routing_policy(self) -> None:
        cases = (
            ('kind = "unknown"', "flow.invalid_enum", "strict"),
            ("priority = -1", "flow.invalid_route_priority", "strict"),
            ("priority = true", "flow.invalid_type", "strict"),
            ('default = "yes"', "flow.invalid_type", "strict"),
            ('routing_policy = "first"', "flow.invalid_enum", "strict"),
            ('kind = "error"', "flow.unsupported_edge_policy", "legacy"),
        )

        for edge_toml, code, mode in cases:
            with self.subTest(code=code, mode=mode):
                strict_toml = """
                    version = "2026-06-07"
                    """
                entry_toml = """
                    [[inputs]]
                    name = "payload"
                    type = "object"

                    [[outputs]]
                    name = "answer"
                    type = "object"

                    [entry]
                    type = "node"
                    node = "start"

                    [output_behavior]
                    type = "map"

                    [output_behavior.outputs]
                    answer = "finish.result"
                    """
                if mode == "legacy":
                    strict_toml = """
                    entrypoint = "start"
                    output_node = "finish"
                    """
                    entry_toml = ""
                result = loads_flow_definition_result(f"""
                    [flow]
                    name = "invalid"
                    {strict_toml}

                    {entry_toml}

                    [nodes.start]
                    type = "echo"

                    [nodes.finish]
                    type = "echo"

                    [[edges]]
                    source = "start"
                    target = "finish"
                    {edge_toml}
                    """)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])

    def test_loader_rejects_invalid_join_policy(self) -> None:
        cases = (
            (
                """
                join_policy = "all_success"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [nodes.finish.join_policy]
                type = "unknown"
                """,
                "flow.invalid_enum",
            ),
            (
                """
                [nodes.finish.join_policy]
                type = "quorum"
                quorum = true
                """,
                "flow.invalid_type",
            ),
            (
                """
                [nodes.finish.join_policy]
                type = "all_success"
                secret = "private-token"
                """,
                "flow.unsupported_field",
            ),
            (
                """
                [nodes.finish.join_policy]
                type = "all_success"
                optional_inputs = "value"
                """,
                "flow.invalid_type",
            ),
        )

        for join_policy_toml, code in cases:
            with self.subTest(code=code):
                result = loads_flow_definition_result(f"""
                    [flow]
                    name = "strict"
                    version = "2026-06-07"

                    [[inputs]]
                    name = "payload"
                    type = "object"

                    [[outputs]]
                    name = "answer"
                    type = "object"

                    [entry]
                    type = "node"
                    node = "left"

                    [output_behavior]
                    type = "map"

                    [output_behavior.outputs]
                    answer = "finish.value"

                    [nodes.left]
                    type = "echo"

                    [nodes.right]
                    type = "echo"

                    [nodes.finish]
                    type = "echo"
                    {join_policy_toml}

                    [[edges]]
                    source = "left"
                    target = "finish"

                    [[edges]]
                    source = "right"
                    target = "finish"
                    """)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])
                self.assertNotIn(
                    "private-token",
                    str(result.public_diagnostics),
                )

    def test_loader_rejects_invalid_node_policies(self) -> None:
        cases = (
            (
                """
                retry_policy = "retry"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [nodes.start.retry_policy]
                max_attempts = 2
                backoff = "unknown"
                """,
                "flow.invalid_enum",
            ),
            (
                """
                [nodes.start.retry_policy]
                max_attempts = true
                """,
                "flow.invalid_type",
            ),
            (
                """
                [nodes.start.retry_policy]
                secret = "private-token"
                """,
                "flow.unsupported_field",
            ),
            (
                """
                [nodes.start.retry_policy]
                max_attempts = 0
                """,
                "flow.invalid_retry_attempts",
            ),
            (
                """
                [nodes.start.timeout_policy]
                per_attempt_seconds = "slow"
                """,
                "flow.invalid_type",
            ),
            (
                """
                timeout_policy = "timeout"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [nodes.start.timeout_policy]
                per_attempt_seconds = 0
                """,
                "flow.invalid_timeout",
            ),
            (
                """
                [nodes.start.loop_policy]
                max_iterations = true
                """,
                "flow.invalid_type",
            ),
            (
                """
                loop_policy = "loop"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [nodes.start.loop_policy]
                max_iterations = 2
                output_selector = "start.result"
                limit_route = "limited"

                [nodes.start.loop_policy.continue_condition]
                op = "exists"
                selector = "start.result.more"
                """,
                "flow.missing_loop_exit_condition",
            ),
            (
                """
                [nodes.start.loop_policy]
                max_iterations = 2
                output_selector = "start.result"
                limit_route = "limited"

                [nodes.start.loop_policy.continue_condition]
                op = "exists"
                selector = "start.result.more"

                [nodes.start.loop_policy.exit_condition]
                op = "exists"
                selector = "start.result.done"
                """,
                "flow.missing_loop_limit_route",
            ),
        )

        for policy_toml, code in cases:
            with self.subTest(code=code):
                result = loads_flow_definition_result(f"""
                    [flow]
                    name = "strict"
                    version = "2026-06-07"

                    [[inputs]]
                    name = "payload"
                    type = "object"

                    [[outputs]]
                    name = "answer"
                    type = "object"

                    [entry]
                    type = "node"
                    node = "start"

                    [output_behavior]
                    type = "map"

                    [output_behavior.outputs]
                    answer = "finish.result"

                    [nodes.start]
                    type = "echo"
                    {policy_toml}

                    [nodes.finish]
                    type = "echo"

                    [nodes.limited]
                    type = "echo"

                    [[edges]]
                    source = "start"
                    target = "finish"
                    """)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])
                self.assertNotIn(
                    "private-token",
                    str(result.public_diagnostics),
                )

    def test_loader_rejects_duplicate_default_routes(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "strict"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "left.result"

            [nodes.start]
            type = "echo"

            [nodes.left]
            type = "echo"

            [nodes.right]
            type = "echo"

            [[edges]]
            source = "start"
            target = "left"
            default = true

            [[edges]]
            source = "start"
            target = "right"
            default = true
            """)

        self.assertFalse(result.ok)
        self.assertIn(
            "flow.duplicate_default_route",
            [issue.code for issue in result.issues],
        )

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
                        "flow.invalid_output_selector",
                        "flow.invalid_node",
                        "flow.invalid_type",
                        "flow.reserved_selector",
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
            build_runtime=True,
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
        string_mapping = flow_loader._string_mapping(  # type: ignore[attr-defined]
            {"outputs": {1: "bad"}},  # type: ignore[dict-item]
            "flow.output_behavior.outputs",
            "outputs",
            issues,
        )
        node_mappings = flow_loader._node_mappings(  # type: ignore[attr-defined]
            {1: "input.payload"},  # type: ignore[dict-item]
            issues,
            path="nodes.start.mapping",
        )
        optional_mapping = flow_loader._optional_string_mapping(  # type: ignore[attr-defined]
            {"fields": {1: "bad"}},  # type: ignore[dict-item]
            "nodes.start.mapping.value.fields",
            "fields",
            issues,
        )

        self.assertFalse(result.ok)
        self.assertEqual(tuple_value, ())
        self.assertEqual(list_value, ("text/plain",))
        self.assertIsNone(metadata)
        self.assertIsNone(string_mapping)
        self.assertEqual(node_mappings, ())
        self.assertIsNone(optional_mapping)
        self.assertEqual(
            flow_loader._condition_output_names(  # type: ignore[attr-defined]
                FlowNodeRegistry(
                    {"custom": lambda definition: Node("custom")}
                ),
                "custom",
            ),
            ("value", "result"),
        )
        self.assertFalse(
            flow_loader._nodes_use_strict_definition(  # type: ignore[attr-defined]
                None,
            )
        )
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


class FlowBuildTestCase(IsolatedAsyncioTestCase):
    async def test_build_flow_from_definition(self) -> None:
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

        self.assertEqual(await flow.execute_async(), "ok")


if __name__ == "__main__":
    main()
