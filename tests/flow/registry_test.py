from unittest import TestCase, main

from avalan.flow import (
    FLOW_INPUT_KEY,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeMetadata,
    FlowNodeRegistry,
    FlowOutputType,
    default_flow_node_registry,
    flow_input_binding,
)
from avalan.flow.node import Node
from avalan.flow.registry import FlowNodeConfigurationError


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


if __name__ == "__main__":
    main()
