from collections.abc import Mapping
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import ToolManagerSettings
from avalan.flow import (
    FLOW_INPUT_KEY,
    FLOW_TOOL_NODE_TYPE,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeMetadata,
    FlowNodeRegistry,
    FlowOutputType,
    default_flow_node_registry,
    flow_input_binding,
    tool_flow_node_registry,
)
from avalan.flow.node import Node
from avalan.flow.registry import FlowNodeConfigurationError
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.mcp import McpToolSet


async def flow_adder(a: int, b: int) -> int:
    return a + b


flow_adder.aliases = ["sum"]  # type: ignore[attr-defined]


async def flow_adder_alt(a: int, b: int) -> int:
    return a + b


flow_adder_alt.aliases = ["sum"]  # type: ignore[attr-defined]


async def flow_disabled(a: int) -> int:
    return a


def _tool_manager(
    *,
    enable_tools: list[str] | None = None,
) -> ToolManager:
    return ToolManager.create_instance(
        enable_tools=enable_tools
        or [
            "flow_adder",
            "mcp.call",
        ],
        available_toolsets=[
            ToolSet(tools=[flow_adder, flow_adder_alt]),
            ToolSet(namespace="disabled", tools=[flow_disabled]),
            McpToolSet(),
        ],
        settings=ToolManagerSettings(),
    )


class FlowNodeRegistryTestCase(TestCase):
    def test_default_nodes_execute_without_mutating_inputs(self) -> None:
        registry = default_flow_node_registry()
        payload = {"items": ["first"]}
        binding = flow_input_binding(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            payload,
        )
        payload["items"].append("changed")

        input_node = registry.build(
            FlowNodeDefinition(name="input", type="input", input="payload")
        )
        echo_node = registry.build(
            FlowNodeDefinition(name="echo", type="echo", input="input")
        )
        select_node = registry.build(
            FlowNodeDefinition(
                name="select",
                type="select",
                input="input",
                config={"path": "items.0"},
            )
        )
        constant_node = registry.build(
            FlowNodeDefinition(
                name="constant",
                type="constant",
                config={"value": {"answer": "ok"}},
            )
        )

        input_value = input_node.execute(binding)
        self.assertEqual(input_value, {"items": ["first"]})
        self.assertEqual(
            echo_node.execute({"input": input_value}), input_value
        )
        self.assertEqual(select_node.execute({"input": input_value}), "first")
        self.assertEqual(
            constant_node.execute({}),
            {"answer": "ok"},
        )
        self.assertEqual(payload, {"items": ["first", "changed"]})

    def test_binding_handles_scalar_mapping_and_missing_definition(
        self,
    ) -> None:
        scalar = flow_input_binding(
            FlowInputDefinition(name="text", type=FlowInputType.STRING),
            "ready",
        )
        no_definition = flow_input_binding(None, "ready")
        mapping = flow_input_binding(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            {"name": "Ada"},
        )

        self.assertEqual(scalar[FLOW_INPUT_KEY], "ready")
        self.assertEqual(scalar["text"], "ready")
        self.assertEqual(scalar["value"], "ready")
        self.assertEqual(no_definition["value"], "ready")
        self.assertEqual(mapping["payload"], {"name": "Ada"})
        self.assertNotIn("value", mapping)

    def test_echo_node_falls_back_to_available_inputs(self) -> None:
        node = default_flow_node_registry().build(
            FlowNodeDefinition(name="echo", type="echo")
        )

        self.assertEqual(node.execute({FLOW_INPUT_KEY: "flow"}), "flow")
        self.assertEqual(node.execute({"only": "value"}), "value")
        self.assertEqual(
            node.execute({"left": "L", "right": "R"}),
            {"left": "L", "right": "R"},
        )

    def test_custom_registry_and_select_errors(self) -> None:
        registry = FlowNodeRegistry()

        def factory(definition: FlowNodeDefinition) -> Node:
            return Node(
                definition.name,
                func=lambda inputs: str(inputs["value"]).upper(),
            )

        registry.register("upper", factory)

        self.assertTrue(registry.supports("upper"))
        self.assertEqual(
            registry.build(
                FlowNodeDefinition(name="upper", type="upper")
            ).execute({"value": "ok"}),
            "OK",
        )

        select_node = default_flow_node_registry().build(
            FlowNodeDefinition(
                name="select",
                type="select",
                input="input",
                config={"field": "missing"},
            )
        )
        with self.assertRaises(KeyError):
            select_node.execute({"input": {"answer": "ok"}})

    def test_registry_exposes_node_metadata(self) -> None:
        registry = default_flow_node_registry()

        self.assertFalse(registry.supports_ref("echo"))
        self.assertFalse(registry.is_async_only("echo"))
        self.assertIsNone(registry.metadata("missing"))
        self.assertIsNone(registry.input_contract("missing"))
        self.assertIsNone(registry.output_contract("missing"))
        echo_input = registry.input_contract("echo")
        echo_output = registry.output_contract("echo")

        assert echo_input is not None
        assert echo_output is not None
        self.assertEqual(echo_input.name, "value")
        self.assertTrue(echo_input.metadata["dynamic"])
        self.assertEqual(echo_output.name, "value")
        self.assertTrue(echo_output.metadata["dynamic"])

    def test_custom_registry_metadata_supports_refs_and_contracts(
        self,
    ) -> None:
        def factory(definition: FlowNodeDefinition) -> Node:
            return Node(definition.name, func=lambda _: definition.ref)

        metadata = FlowNodeMetadata(
            supports_ref=True,
            async_only=True,
            input_contract=FlowNodeContract(
                name="payload",
                type=FlowInputType.OBJECT,
                schema={"type": "object"},
            ),
            output_contract=FlowNodeContract(
                name="result",
                type=FlowOutputType.JSON,
                schema={"type": "string"},
            ),
        )
        registry = FlowNodeRegistry({"remote": factory}, {"remote": metadata})

        self.assertTrue(registry.supports("remote"))
        self.assertTrue(registry.supports_ref("remote"))
        self.assertTrue(registry.is_async_only("remote"))
        self.assertEqual(
            registry.input_contract("remote"), metadata.input_contract
        )
        self.assertEqual(
            registry.output_contract("remote"),
            metadata.output_contract,
        )
        self.assertEqual(
            registry.build(
                FlowNodeDefinition(
                    name="remote",
                    type="remote",
                    ref="safe.toml",
                )
            ).execute({}),
            "safe.toml",
        )

    def test_registry_rejects_invalid_metadata(self) -> None:
        registry = FlowNodeRegistry()

        with self.assertRaises(AssertionError):
            registry.register(
                "bad",
                lambda definition: Node(definition.name),
                metadata=object(),  # type: ignore[arg-type]
            )

    def test_configuration_error_carries_public_fields(self) -> None:
        error = FlowNodeConfigurationError(
            code="flow.invalid_node",
            path="nodes.start",
            message="Flow node configuration is invalid.",
            hint="Fix the node configuration.",
        )

        self.assertEqual(str(error), "flow.invalid_node")
        self.assertEqual(error.code, "flow.invalid_node")
        self.assertEqual(error.path, "nodes.start")
        self.assertEqual(
            error.message,
            "Flow node configuration is invalid.",
        )
        self.assertEqual(error.hint, "Fix the node configuration.")
        with self.assertRaises(AssertionError):
            FlowNodeConfigurationError(
                code="",
                path="nodes.start",
                message="Flow node configuration is invalid.",
                hint="Fix the node configuration.",
            )


class FlowToolNodeRegistryTestCase(IsolatedAsyncioTestCase):
    async def test_tool_registry_builds_async_tool_node_with_metadata(
        self,
    ) -> None:
        registry = tool_flow_node_registry(_tool_manager())

        self.assertTrue(registry.supports(FLOW_TOOL_NODE_TYPE))
        self.assertTrue(registry.supports_ref(FLOW_TOOL_NODE_TYPE))
        self.assertTrue(registry.is_async_only(FLOW_TOOL_NODE_TYPE))
        metadata = registry.metadata(FLOW_TOOL_NODE_TYPE)
        input_contract = registry.input_contract(FLOW_TOOL_NODE_TYPE)
        output_contract = registry.output_contract(FLOW_TOOL_NODE_TYPE)
        assert metadata is not None
        assert input_contract is not None
        assert output_contract is not None
        tools = metadata.metadata["tools"]
        assert isinstance(tools, Mapping)
        self.assertIn("flow_adder", tools)
        self.assertEqual(input_contract.name, "arguments")
        self.assertEqual(input_contract.type, FlowInputType.OBJECT)
        self.assertEqual(output_contract.name, "result")
        self.assertEqual(output_contract.type, FlowOutputType.JSON)

        node = registry.build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
            )
        )

        self.assertEqual(node.label, "flow_adder")
        with self.assertRaises(NotImplementedError):
            await node.execute_async({})

    def test_tool_registry_can_extend_custom_base_registry(self) -> None:
        def factory(definition: FlowNodeDefinition) -> Node:
            return Node(definition.name, func=lambda _: "custom")

        base_registry = FlowNodeRegistry({"custom": factory})

        registry = tool_flow_node_registry(
            _tool_manager(),
            base_registry=base_registry,
        )

        self.assertIs(registry, base_registry)
        self.assertTrue(registry.supports("custom"))
        self.assertTrue(registry.supports(FLOW_TOOL_NODE_TYPE))

    def test_tool_registry_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(AssertionError):
            tool_flow_node_registry(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            tool_flow_node_registry(
                _tool_manager(),
                base_registry=object(),  # type: ignore[arg-type]
            )

    def test_tool_node_factory_rejects_bad_refs(self) -> None:
        manager = _tool_manager(enable_tools=["flow_adder", "flow_adder_alt"])
        registry = tool_flow_node_registry(manager)
        cases = (
            (None, "flow.missing_ref"),
            ("tools/adder.py", "flow.invalid_ref"),
            ("mcp://server/tool", "flow.invalid_ref"),
            ("missing", "flow.tool_unknown"),
            ("disabled.flow_disabled", "flow.tool_disabled"),
            ("sum", "flow.tool_ambiguous"),
        )

        for ref, code in cases:
            with self.subTest(ref=ref):
                definition = FlowNodeDefinition(
                    name="calculate",
                    type=FLOW_TOOL_NODE_TYPE,
                    ref=ref,
                )
                with self.assertRaises(FlowNodeConfigurationError) as context:
                    registry.build(definition)

                self.assertEqual(context.exception.code, code)
                self.assertEqual(context.exception.path, "nodes.calculate.ref")


if __name__ == "__main__":
    main()
