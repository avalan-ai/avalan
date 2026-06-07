from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from avalan.flow import (
    FlowCondition,
    FlowConditionOperator,
    FlowDefinition,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowEntryBehavior,
    FlowEntryBehaviorType,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowMappingKind,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowOutputBehavior,
    FlowOutputBehaviorType,
    FlowOutputDefinition,
    FlowOutputType,
    FlowRouteMatchPolicy,
)


class FlowDefinitionTestCase(TestCase):
    def test_flow_entities_are_frozen_and_copy_nested_mappings(self) -> None:
        variables = {"settings": {"locale": "en"}, "items": ["draft"]}
        node_config = {"value": {"answer": "ok"}}
        definition = FlowDefinition(
            name="flow",
            version="1",
            revision="r1",
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
            inputs=(
                FlowInputDefinition(
                    name="strict_payload",
                    type=FlowInputType.OBJECT,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="strict_result",
                    type=FlowOutputType.OBJECT,
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="start"),
            output_behavior=FlowOutputBehavior(
                outputs={"strict_result": "start.result"},
            ),
            runtime_limits={"timeout_seconds": 30},
            privacy_policy={"store_raw": False},
            observability_policy={"events": ["node"]},
            tags=("ops",),
            ownership={"team": "platform"},
            variables=variables,
            nodes=(
                FlowNodeDefinition(
                    name="start",
                    type="constant",
                    mappings=(
                        FlowInputMapping(
                            target="payload",
                            kind=FlowMappingKind.OBJECT,
                            fields={"name": "input.strict_payload.name"},
                        ),
                    ),
                    output="result",
                    config=node_config,
                ),
            ),
            edges=(
                FlowEdgeDefinition(
                    source="start",
                    target="start",
                    label="done",
                    kind=FlowEdgeKind.FINALLY,
                    condition=FlowCondition(
                        operator=FlowConditionOperator.EXISTS,
                        selector="start.result",
                    ),
                    priority=5,
                    default=False,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
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
        self.assertEqual(definition.revision, "r1")
        self.assertEqual(definition.inputs[0].name, "strict_payload")
        self.assertEqual(definition.outputs[0].name, "strict_result")
        self.assertEqual(
            definition.entry_behavior,
            FlowEntryBehavior(node="start"),
        )
        self.assertEqual(
            definition.output_behavior,
            FlowOutputBehavior(outputs={"strict_result": "start.result"}),
        )
        self.assertEqual(definition.runtime_limits["timeout_seconds"], 30)
        self.assertEqual(definition.privacy_policy["store_raw"], False)
        self.assertEqual(definition.observability_policy["events"], ("node",))
        self.assertEqual(definition.tags, ("ops",))
        self.assertEqual(definition.ownership["team"], "platform")
        self.assertTrue(definition.is_strict)
        self.assertEqual(
            definition.edges[0].condition,
            FlowCondition(
                operator=FlowConditionOperator.EXISTS,
                selector="start.result",
            ),
        )
        self.assertEqual(definition.edges[0].kind, FlowEdgeKind.FINALLY)
        self.assertEqual(definition.edges[0].priority, 5)
        self.assertFalse(definition.edges[0].default)
        self.assertEqual(
            definition.edges[0].routing_policy,
            FlowRouteMatchPolicy.ALL_MATCHING,
        )
        self.assertEqual(
            cast(dict[str, object], definition.nodes[0].config["value"])[
                "answer"
            ],
            "ok",
        )
        self.assertEqual(
            definition.nodes[0].mappings[0].fields["name"],
            "input.strict_payload.name",
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
                nodes=(object(),),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowDefinition(
                name="flow",
                nodes=(),
                inputs=(object(),),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowDefinition(
                name="flow",
                nodes=(),
                outputs=(object(),),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowEntryBehavior(
                type="node",  # type: ignore[arg-type]
                node="start",
            )
        with self.assertRaises(AssertionError):
            FlowEntryBehavior(node="")
        with self.assertRaises(AssertionError):
            FlowOutputBehavior(
                type="map",  # type: ignore[arg-type]
                outputs={"result": "start.value"},
            )
        with self.assertRaises(AssertionError):
            FlowOutputBehavior(outputs={"": "start.value"})
        with self.assertRaises(AssertionError):
            FlowDefinition(
                name="flow",
                nodes=(),
                tags=["ops"],  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowInputMapping(target="", source="input.payload")
        with self.assertRaises(AssertionError):
            FlowInputMapping(
                target="value",
                kind="select",  # type: ignore[arg-type]
                source="input.payload",
            )
        with self.assertRaises(AssertionError):
            FlowInputMapping(
                target="value",
                sources=("input.payload", ""),
            )
        with self.assertRaises(AssertionError):
            FlowNodeDefinition(
                name="node",
                type="echo",
                mappings=(object(),),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowEdgeDefinition(
                source="start",
                target="finish",
                condition=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowEdgeDefinition(
                source="start",
                target="finish",
                kind="success",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowEdgeDefinition(
                source="start",
                target="finish",
                priority=True,  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowEdgeDefinition(
                source="start",
                target="finish",
                default="yes",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowEdgeDefinition(
                source="start",
                target="finish",
                routing_policy="exclusive",  # type: ignore[arg-type]
            )

    def test_behavior_type_enums_are_stable(self) -> None:
        self.assertEqual(FlowEntryBehaviorType.NODE.value, "node")
        self.assertEqual(FlowOutputBehaviorType.MAP.value, "map")
        self.assertEqual(FlowConditionOperator.CONTAINS.value, "contains")
        self.assertEqual(FlowMappingKind.FILE_ARRAY.value, "file[]")
        self.assertEqual(FlowEdgeKind.CANCELLATION.value, "cancellation")
        self.assertEqual(FlowEdgeKind.ERROR.value, "error")
        self.assertEqual(FlowEdgeKind.FINALLY.value, "finally")
        self.assertEqual(FlowEdgeKind.PAUSE.value, "pause")
        self.assertEqual(FlowEdgeKind.RESUME.value, "resume")
        self.assertEqual(FlowEdgeKind.SUCCESS.value, "success")
        self.assertEqual(FlowEdgeKind.TIMEOUT.value, "timeout")
        self.assertEqual(
            FlowRouteMatchPolicy.ALL_MATCHING.value,
            "all_matching",
        )
        self.assertEqual(
            FlowRouteMatchPolicy.EXCLUSIVE.value,
            "exclusive",
        )
        self.assertEqual(FlowNodeKind.HUMAN_REVIEW.value, "human_review")
        self.assertEqual(FlowNodeKind.SUBFLOW.value, "subflow")
        self.assertEqual(
            FlowNodeCapability.DURABLE_PAUSE.value,
            "durable_pause",
        )

    def test_node_metadata_is_frozen_and_copy_nested_mappings(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "str"}}}
        metadata = {"source": {"name": "tool"}, "tags": ["runtime"]}
        node_metadata = FlowNodeMetadata(
            kind=FlowNodeKind.TOOL,
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
            input_contracts=(
                FlowNodeContract(name="extra", type=FlowInputType.STRING),
            ),
            output_contracts=(
                FlowNodeContract(name="audit", type=FlowOutputType.JSON),
            ),
            capabilities=(FlowNodeCapability.TASK_BACKED,),
            requires_ref=True,
            required_config_keys=("tool_mode",),
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
        self.assertEqual(node_metadata.kind, FlowNodeKind.TOOL)
        self.assertTrue(node_metadata.supports_ref)
        self.assertTrue(node_metadata.async_only)
        self.assertEqual(
            node_metadata.input_contracts,
            (FlowNodeContract(name="extra", type=FlowInputType.STRING),),
        )
        self.assertEqual(
            node_metadata.output_contracts,
            (FlowNodeContract(name="audit", type=FlowOutputType.JSON),),
        )
        self.assertEqual(
            node_metadata.capabilities,
            (
                FlowNodeCapability.TASK_BACKED,
                FlowNodeCapability.ASYNC_ONLY,
            ),
        )
        self.assertTrue(node_metadata.requires_ref)
        self.assertEqual(node_metadata.required_config_keys, ("tool_mode",))
        with self.assertRaises(FrozenInstanceError):
            node_metadata.supports_ref = False  # type: ignore[misc]

    def test_node_metadata_uses_single_contract_aliases(self) -> None:
        input_contract = FlowNodeContract(name="payload")
        output_contract = FlowNodeContract(name="result")

        metadata = FlowNodeMetadata(
            input_contract=input_contract,
            output_contract=output_contract,
        )

        self.assertEqual(metadata.input_contracts, (input_contract,))
        self.assertEqual(metadata.output_contracts, (output_contract,))
        self.assertEqual(
            metadata.capabilities,
            (FlowNodeCapability.DIRECT_ASYNC,),
        )

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
        with self.assertRaises(AssertionError):
            FlowNodeMetadata(
                kind="tool",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowNodeMetadata(
                input_contracts=[FlowNodeContract()],  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowNodeMetadata(
                output_contracts=(object(),),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowNodeMetadata(
                capabilities=("async_only",),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowNodeMetadata(required_config_keys=("",))


if __name__ == "__main__":
    main()
