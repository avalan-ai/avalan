from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from avalan.flow import (
    FlowDefinition,
    FlowEdgeDefinition,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeMetadata,
    FlowOutputDefinition,
    FlowOutputType,
)


class FlowDefinitionTestCase(TestCase):
    def test_flow_entities_are_frozen_and_copy_nested_mappings(self) -> None:
        variables = {"settings": {"locale": "en"}, "items": ["draft"]}
        node_config = {"value": {"answer": "ok"}}
        definition = FlowDefinition(
            name="flow",
            version="1",
            description="A test flow",
            entrypoint="start",
            output_node="start",
            input=FlowInputDefinition(
                name="payload",
                type=FlowInputType.OBJECT,
                schema={"type": "object"},
                schema_ref="schemas/input.json",
            ),
            output=FlowOutputDefinition(
                name="result",
                type=FlowOutputType.OBJECT,
                schema={"type": "object"},
            ),
            variables=variables,
            nodes=(
                FlowNodeDefinition(
                    name="start",
                    type="constant",
                    output="result",
                    config=node_config,
                ),
            ),
            edges=(
                FlowEdgeDefinition(
                    source="start",
                    target="start",
                    label="done",
                ),
            ),
        )

        variables["settings"]["locale"] = "es"
        variables["items"].append("publish")
        node_config["value"]["answer"] = "changed"

        self.assertEqual(
            cast(dict[str, object], definition.variables["settings"])[
                "locale"
            ],
            "en",
        )
        self.assertEqual(definition.variables["items"], ("draft",))
        self.assertEqual(
            cast(dict[str, object], definition.nodes[0].config["value"])[
                "answer"
            ],
            "ok",
        )
        with self.assertRaises(FrozenInstanceError):
            definition.name = "changed"  # type: ignore[misc]
        self.assertEqual(definition.node_map["start"].type, "constant")

    def test_invalid_entities_raise_assertion_errors(self) -> None:
        with self.assertRaises(AssertionError):
            FlowInputDefinition(name="", type=FlowInputType.STRING)
        with self.assertRaises(AssertionError):
            FlowInputDefinition(
                name="input",
                type=FlowInputType.STRING,
                mime_types=("",),
            )
        with self.assertRaises(AssertionError):
            FlowOutputDefinition(
                name="output",
                type=FlowOutputType.JSON,
                schema_ref="",
            )
        with self.assertRaises(AssertionError):
            FlowNodeDefinition(name="node", type="")
        with self.assertRaises(AssertionError):
            FlowDefinition(
                name="flow",
                entrypoint="start",
                output_node="start",
                nodes=(object(),),  # type: ignore[arg-type]
            )

    def test_node_metadata_is_frozen_and_copy_nested_mappings(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "str"}}}
        metadata = {"source": {"name": "tool"}, "tags": ["runtime"]}
        node_metadata = FlowNodeMetadata(
            supports_ref=True,
            async_only=True,
            input_contract=FlowNodeContract(
                name="payload",
                type=FlowInputType.OBJECT,
                schema=schema,
                metadata=metadata,
            ),
            output_contract=FlowNodeContract(
                name="result",
                type=FlowOutputType.JSON,
                schema_ref="schemas/result.json",
            ),
            metadata={"canonical_schema": schema},
        )

        schema["properties"]["name"]["type"] = "number"
        metadata["source"]["name"] = "changed"
        metadata["tags"].append("changed")

        input_contract = node_metadata.input_contract
        assert input_contract is not None
        assert input_contract.schema is not None
        self.assertEqual(
            cast(dict[str, object], input_contract.schema["properties"])[
                "name"
            ],
            {"type": "str"},
        )
        self.assertEqual(
            cast(dict[str, object], input_contract.metadata["source"])["name"],
            "tool",
        )
        self.assertEqual(input_contract.metadata["tags"], ("runtime",))
        self.assertTrue(node_metadata.supports_ref)
        self.assertTrue(node_metadata.async_only)
        with self.assertRaises(FrozenInstanceError):
            node_metadata.supports_ref = False  # type: ignore[misc]

    def test_invalid_node_metadata_raise_assertion_errors(self) -> None:
        with self.assertRaises(AssertionError):
            FlowNodeContract(name="")
        with self.assertRaises(AssertionError):
            FlowNodeContract(type="")
        with self.assertRaises(AssertionError):
            FlowNodeContract(schema_ref="")
        with self.assertRaises(AssertionError):
            FlowNodeMetadata(
                input_contract=object(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    main()
