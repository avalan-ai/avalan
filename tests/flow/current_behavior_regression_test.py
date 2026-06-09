from unittest import IsolatedAsyncioTestCase, main

from async_helpers import run_async

from avalan.entities import ToolCallDiagnosticCode, ToolCallDiagnosticStage
from avalan.flow import (
    FLOW_TOOL_NODE_TYPE,
    FlowDefinitionLoader,
    FlowNodeDefinition,
    FlowNodeMetadata,
    FlowNodeRegistry,
    tool_flow_node_registry,
)
from avalan.flow.flow import Flow
from avalan.flow.node import Node
from avalan.flow.registry import FlowNodeConfigurationError
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager

_AsyncFlowDefinitionLoader = FlowDefinitionLoader


class FlowDefinitionLoader(_AsyncFlowDefinitionLoader):  # type: ignore[no-redef]
    def loads_result(self, *args: object, **kwargs: object) -> object:
        return run_async(super().loads_result(*args, **kwargs))

    def loads_validation_result(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        return run_async(super().loads_validation_result(*args, **kwargs))


async def current_adder(a: int, b: int) -> int:
    return a + b


current_adder.aliases = ["current_sum"]  # type: ignore[attr-defined]


class FlowCurrentBehaviorRegressionTest(IsolatedAsyncioTestCase):
    async def test_native_graph_execution_surface_is_async_only(self) -> None:
        flow = Flow()
        flow.add_node(Node("start", func=lambda _: "ready"))

        result = await flow.execute_async()

        self.assertEqual(result, "ready")
        self.assertFalse(hasattr(flow, "execute"))
        self.assertFalse(hasattr(Node("detached"), "execute"))

    def test_loader_rejects_unknown_fields_before_building_nodes(self) -> None:
        build_calls = 0

        def factory(definition: FlowNodeDefinition) -> Node:
            nonlocal build_calls
            build_calls += 1
            raise AssertionError(definition.name)

        loader = FlowDefinitionLoader(
            FlowNodeRegistry(
                {"external": factory},
                {"external": FlowNodeMetadata()},
            )
        )

        result = loader.loads_result("""
            [flow]
            name = "strictness"
            entrypoint = "start"
            output_node = "start"
            unsupported = "private prompt"

            [nodes.start]
            type = "external"
            unsupported = "private node payload"
            """)

        self.assertFalse(result.ok)
        self.assertIsNone(result.definition)
        self.assertIsNone(result.flow)
        self.assertEqual(build_calls, 0)
        self.assertEqual(
            [(issue.code, issue.path) for issue in result.issues],
            [
                ("flow.unsupported_field", "flow.unsupported"),
                ("flow.unsupported_field", "nodes.start.unsupported"),
            ],
        )
        public = str(result.public_diagnostics)
        self.assertNotIn("private prompt", public)
        self.assertNotIn("private node payload", public)

    def test_validation_loader_reports_malformed_toml_safely(self) -> None:
        result = FlowDefinitionLoader().loads_validation_result(
            "[flow\nprivate = 'source'"
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.definition)
        self.assertIsNone(result.flow)
        self.assertEqual(result.issues[0].code, "flow.malformed_toml")
        self.assertNotIn("private", str(result.public_diagnostics))

    async def test_tool_node_uses_enabled_descriptors_and_safe_diagnostics(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["current_adder"],
            available_toolsets=[ToolSet(tools=[current_adder])],
        )
        registry = tool_flow_node_registry(manager)
        definition = FlowNodeDefinition(
            name="calculate",
            type=FLOW_TOOL_NODE_TYPE,
            ref="current_sum",
            config={"arguments": {"a": "left", "b": "right"}},
        )

        descriptor = registry.validate_tool_definition(
            definition,
            require_explicit_arguments=True,
        )
        node = registry.build(definition)

        self.assertEqual(descriptor.name, "current_adder")
        self.assertEqual(node.label, "current_adder")
        self.assertTrue(node.async_only)
        self.assertEqual(
            await node.execute_async(
                {"payload": {"left": 2, "right": 3}},
            ),
            5,
        )

        diagnostic_node = registry.build(
            FlowNodeDefinition(
                name="typed",
                type=FLOW_TOOL_NODE_TYPE,
                ref="current_adder",
                config={
                    "arguments": {"a": "left", "b": "right"},
                    "output_mode": "envelope",
                },
            )
        )

        envelope = await diagnostic_node.execute_async(
            {"payload": {"left": "private token", "right": 3}}
        )

        assert isinstance(envelope, dict)
        diagnostic = envelope["diagnostic"]
        assert isinstance(diagnostic, dict)
        self.assertEqual(envelope["status"], "diagnostic")
        self.assertEqual(envelope["canonical_name"], "current_adder")
        self.assertEqual(
            diagnostic["code"],
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED.value,
        )
        self.assertEqual(
            diagnostic["stage"],
            ToolCallDiagnosticStage.VALIDATE.value,
        )
        self.assertNotIn("private token", str(envelope))
        self.assertNotIn("arguments", diagnostic)

    def test_tool_node_rejects_disabled_enabled_descriptor(self) -> None:
        manager = ToolManager.create_instance(
            enable_tools=[],
            available_toolsets=[ToolSet(tools=[current_adder])],
        )
        registry = tool_flow_node_registry(manager)

        with self.assertRaises(FlowNodeConfigurationError) as context:
            registry.build(
                FlowNodeDefinition(
                    name="calculate",
                    type=FLOW_TOOL_NODE_TYPE,
                    ref="current_adder",
                )
            )

        self.assertEqual(context.exception.code, "flow.tool_disabled")


if __name__ == "__main__":
    main()
