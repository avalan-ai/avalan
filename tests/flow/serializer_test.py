from collections.abc import Mapping
from enum import Enum
from unittest import TestCase, main

from avalan.flow import (
    FlowCondition,
    FlowConditionOperator,
    FlowConditionValueType,
    FlowDefinition,
    FlowDefinitionLoader,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowEntryBehavior,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowJoinPolicy,
    FlowJoinPolicyType,
    FlowLoopPolicy,
    FlowMappingKind,
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
    FlowRouteMatchPolicy,
    FlowTimeoutPolicy,
    compile_flow_definition,
    default_flow_node_registry,
    serialize_flow_definition,
)
from avalan.flow.node import Node


class SerializerMetadataEnum(Enum):
    VALUE = "custom"


class FlowDefinitionSerializerTestCase(TestCase):
    def test_serializes_strict_definition_deterministically(self) -> None:
        definition = self._strict_definition()

        source = serialize_flow_definition(definition)
        repeated = serialize_flow_definition(definition)

        self.assertEqual(source, repeated)
        self.assertIn('[flow]\nname = "runtime"', source)
        self.assertIn("[[inputs]]", source)
        self.assertIn("[[outputs]]", source)
        self.assertIn("[nodes.mapper.mapping.constructed]", source)
        self.assertIn("[[edges]]", source)
        self.assertNotIn("[graph]", source)
        self.assertNotIn("diagram", source)
        self.assertLess(
            source.index("[nodes.source]"),
            source.index("[nodes.mapper]"),
        )
        self.assertLess(
            source.index('source = "source"'),
            source.index('source = "mapper"'),
        )
        self.assertIn('"1bad" = "quoted"', source)
        self.assertIn('"with.dot" = "quoted"', source)

    def test_serialized_strict_definition_round_trips_to_same_plan(
        self,
    ) -> None:
        registry = self._registry()
        definition = self._strict_definition()
        source = serialize_flow_definition(definition)

        result = FlowDefinitionLoader(registry).loads_validation_result(source)

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.definition is not None
        self.assertEqual(serialize_flow_definition(result.definition), source)
        original_plan = compile_flow_definition(definition, registry)
        reloaded_plan = compile_flow_definition(result.definition, registry)
        self.assertTrue(original_plan.ok, original_plan.public_diagnostics)
        self.assertTrue(reloaded_plan.ok, reloaded_plan.public_diagnostics)
        self.assertEqual(original_plan.plan, reloaded_plan.plan)

    def test_serializes_legacy_alias_definition(self) -> None:
        definition = FlowDefinition(
            name="legacy",
            version="1",
            entrypoint="start",
            output_node="finish",
            input=FlowInputDefinition(
                name="payload",
                type=FlowInputType.OBJECT,
                mime_types=("application/json",),
                schema={"type": "object"},
                schema_ref="schemas/input.json",
            ),
            output=FlowOutputDefinition(
                name="answer",
                type=FlowOutputType.JSON,
                schema={"type": "object"},
                schema_ref="schemas/output.json",
            ),
            nodes=(
                FlowNodeDefinition(
                    name="start",
                    type="constant",
                    config={"value": {"answer": 42}},
                ),
                FlowNodeDefinition(
                    name="finish",
                    type="select",
                    input="start",
                    config={"path": "answer"},
                ),
            ),
            edges=(FlowEdgeDefinition(source="start", target="finish"),),
        )

        source = serialize_flow_definition(definition)
        result = FlowDefinitionLoader().loads_validation_result(source)

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.definition is not None
        self.assertEqual(result.definition.input, definition.input)
        self.assertEqual(result.definition.output, definition.output)
        self.assertEqual(result.definition.entrypoint, "start")
        self.assertEqual(result.definition.output_node, "finish")
        self.assertIn("[flow.input]", source)
        self.assertIn("[flow.output]", source)

    def test_rejects_unsupported_toml_values(self) -> None:
        definition = FlowDefinition(
            name="invalid",
            version="1",
            entry_behavior=FlowEntryBehavior(node="start"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "start.value"},
            ),
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.OBJECT,
                ),
            ),
            variables={"unsupported": None},
            nodes=(FlowNodeDefinition(name="start", type="echo"),),
        )

        with self.assertRaises(AssertionError):
            serialize_flow_definition(definition)

        with self.assertRaises(AssertionError):
            serialize_flow_definition("invalid")  # type: ignore[arg-type]

    def test_rejects_non_finite_toml_numbers(self) -> None:
        definition = FlowDefinition(
            name="invalid",
            version="1",
            entry_behavior=FlowEntryBehavior(node="start"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "start.value"},
            ),
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.OBJECT,
                ),
            ),
            variables={"limit": float("inf")},
            nodes=(FlowNodeDefinition(name="start", type="echo"),),
        )

        with self.assertRaises(AssertionError):
            serialize_flow_definition(definition)

    def _strict_definition(self) -> FlowDefinition:
        return FlowDefinition(
            name="runtime",
            version="2026-06-09",
            revision="rev-1",
            description="A strict runtime flow",
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                    mime_types=("application/json",),
                    schema={"type": "object"},
                    schema_ref="schemas/payload.json",
                ),
                FlowInputDefinition(
                    name="document",
                    type=FlowInputType.FILE,
                ),
                FlowInputDefinition(
                    name="documents",
                    type=FlowInputType.FILE_ARRAY,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.OBJECT,
                    schema={"type": "object"},
                    schema_ref="schemas/answer.json",
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="source"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "mapper.result"},
            ),
            runtime_limits={"timeout_seconds": 30, "budget": 1.5},
            privacy_policy={"store_raw": False},
            observability_policy={"events": ("node", "edge")},
            tags=("ops", "runtime"),
            ownership={"team": "platform"},
            variables={
                "1bad": "quoted",
                "custom": SerializerMetadataEnum.VALUE,
                "mode": FlowEdgeKind.ERROR,
                "settings": {"locale": "en"},
                "with.dot": "quoted",
            },
            nodes=(
                FlowNodeDefinition(
                    name="source",
                    type="constant",
                    config={
                        "value": {
                            "blocked": False,
                            "items": ("first", "second"),
                            "status": "ready",
                        }
                    },
                ),
                FlowNodeDefinition(
                    name="mapper",
                    type="mapper",
                    ref="trusted",
                    output="result",
                    join_policy=FlowJoinPolicy(
                        type=FlowJoinPolicyType.QUORUM,
                        quorum=1,
                        optional_inputs=("optional",),
                    ),
                    retry_policy=FlowRetryPolicy(
                        max_attempts=3,
                        backoff=FlowRetryBackoffStrategy.EXPONENTIAL,
                        initial_delay_seconds=1,
                        max_delay_seconds=8,
                        retryable_categories=("transient",),
                        non_retryable_categories=("validation",),
                        exhausted_route="failed",
                    ),
                    timeout_policy=FlowTimeoutPolicy(
                        per_attempt_seconds=5,
                    ),
                    loop_policy=FlowLoopPolicy(
                        max_iterations=2,
                        max_elapsed_seconds=30,
                        continue_condition=FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="mapper.result.more",
                        ),
                        exit_condition=FlowCondition(
                            operator=FlowConditionOperator.IS_TYPE,
                            selector="mapper.result",
                            value_type=FlowConditionValueType.OBJECT,
                        ),
                        output_selector="mapper.result",
                        limit_route="limited",
                    ),
                    mappings=(
                        FlowInputMapping(
                            target="selected",
                            source="input.payload",
                        ),
                        FlowInputMapping(
                            target="constructed",
                            kind=FlowMappingKind.OBJECT,
                            fields={"answer": "source.value"},
                        ),
                        FlowInputMapping(
                            target="items",
                            kind=FlowMappingKind.ARRAY,
                            items=("input.payload", "source.value"),
                        ),
                        FlowInputMapping(
                            target="merged",
                            kind=FlowMappingKind.MERGE,
                            sources=("input.payload", "source.value"),
                        ),
                        FlowInputMapping(
                            target="fallback",
                            kind=FlowMappingKind.COALESCE,
                            sources=("source.missing", "source.value"),
                        ),
                        FlowInputMapping(
                            target="document",
                            kind=FlowMappingKind.FILE,
                            source="input.document",
                        ),
                        FlowInputMapping(
                            target="documents",
                            kind=FlowMappingKind.FILE_ARRAY,
                            source="input.documents",
                        ),
                    ),
                    config={"shape": "safe"},
                ),
                FlowNodeDefinition(name="failed", type="echo"),
                FlowNodeDefinition(name="limited", type="echo"),
            ),
            edges=(
                FlowEdgeDefinition(
                    source="source",
                    target="mapper",
                    label="ready",
                    condition=FlowCondition(
                        operator=FlowConditionOperator.ALL,
                        conditions=(
                            FlowCondition(
                                operator=FlowConditionOperator.EQ,
                                selector="source.value.status",
                                value="ready",
                            ),
                            FlowCondition(
                                operator=FlowConditionOperator.NOT,
                                condition=FlowCondition(
                                    operator=FlowConditionOperator.EXISTS,
                                    selector="source.value.blocked",
                                ),
                            ),
                            FlowCondition(
                                operator=FlowConditionOperator.IN,
                                selector="source.value.status",
                                values=("ready", "queued"),
                            ),
                            FlowCondition(
                                operator=FlowConditionOperator.EQ,
                                selector="source.value.status",
                                value_selector="source.value.status",
                            ),
                        ),
                    ),
                    priority=1,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
                FlowEdgeDefinition(
                    source="mapper",
                    target="failed",
                    kind=FlowEdgeKind.ERROR,
                    priority=1,
                ),
                FlowEdgeDefinition(
                    source="mapper",
                    target="limited",
                    default=True,
                ),
            ),
        )

    def _registry(self) -> FlowNodeRegistry:
        def raising_factory(definition: FlowNodeDefinition) -> Node:
            raise AssertionError(f"factory called for {definition.name}")

        return default_flow_node_registry().register(
            "mapper",
            raising_factory,
            metadata=FlowNodeMetadata(
                kind=FlowNodeKind.SELECT,
                supports_ref=True,
                input_contracts=(
                    self._contract("selected", FlowInputType.OBJECT),
                    self._contract("constructed", FlowInputType.OBJECT),
                    self._contract("items", FlowInputType.ARRAY),
                    self._contract("merged", FlowInputType.OBJECT),
                    self._contract("fallback", FlowInputType.OBJECT),
                    self._contract("document", FlowInputType.FILE),
                    self._contract("documents", FlowInputType.FILE_ARRAY),
                    self._contract("optional", FlowInputType.STRING),
                ),
                output_contract=FlowNodeContract(
                    name="result",
                    type=FlowOutputType.OBJECT,
                    metadata={"dynamic": True},
                ),
            ),
        )

    def _contract(
        self,
        name: str,
        type: FlowInputType,
    ) -> FlowNodeContract:
        metadata: Mapping[str, object] = {"dynamic": True}
        if name == "optional":
            metadata = {"optional": True}
        return FlowNodeContract(name=name, type=type, metadata=metadata)


if __name__ == "__main__":
    main()
