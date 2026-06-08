from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallOutcome,
    ToolCallResult,
    ToolDescriptor,
    ToolManagerSettings,
    ToolNameResolution,
    ToolNameResolutionStatus,
)
from avalan.flow import (
    FlowCondition,
    FlowConditionOperator,
    FlowConditionPlan,
    FlowConditionValueType,
    FlowDefinition,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowEntryBehavior,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowJoinPlan,
    FlowJoinPolicy,
    FlowJoinPolicyType,
    FlowMappingKind,
    FlowMappingPlan,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodePlan,
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
    parse_flow_selector,
    tool_flow_node_registry,
)
from avalan.flow.definition import FlowLoopPolicy
from avalan.flow.node import Node
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager


async def plan_flow_adder(a: int, b: int) -> int:
    return a + b


plan_flow_adder.aliases = ["sum"]  # type: ignore[attr-defined]


def _tool_manager() -> ToolManager:
    return ToolManager.create_instance(
        enable_tools=["plan_flow_adder"],
        available_toolsets=[ToolSet(tools=[plan_flow_adder])],
        settings=ToolManagerSettings(),
    )


class FlowExecutionPlanTestCase(TestCase):
    def test_compile_flow_definition_builds_immutable_plan(self) -> None:
        registry = self._registry()
        definition = self._definition()

        result = compile_flow_definition(definition, registry)

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.diagnostics, ())
        assert result.plan is not None
        plan = result.plan
        self.assertEqual(plan.name, "runtime")
        self.assertEqual(plan.version, "2026-06-07")
        self.assertEqual(plan.revision, "rev-1")
        self.assertEqual(plan.entry_node, "source")
        self.assertEqual(plan.inputs[0].name, "payload")
        self.assertEqual(plan.outputs[0].name, "answer")
        self.assertEqual(
            plan.output_selectors["answer"],
            parse_flow_selector("mapper.result"),
        )
        self.assertEqual(
            set(plan.node_map), {"source", "mapper", "failed", "limited"}
        )
        self.assertEqual(set(plan.edges_by_source), {"mapper", "source"})
        self.assertEqual(
            [edge.target for edge in plan.edges_by_source["mapper"]],
            ["failed", "limited"],
        )

        source = plan.node_map["source"]
        mapper = plan.node_map["mapper"]
        self.assertEqual(source.kind, FlowNodeKind.CONSTANT)
        self.assertEqual(mapper.kind, FlowNodeKind.SELECT)
        self.assertEqual(
            mapper.capabilities,
            (FlowNodeCapability.DIRECT_ASYNC,),
        )
        self.assertEqual(mapper.metadata["mode"], "projection")
        self.assertEqual(mapper.config["shape"], "safe")
        self.assertEqual(
            [mapping.kind for mapping in mapper.mappings],
            [
                FlowMappingKind.SELECT,
                FlowMappingKind.OBJECT,
                FlowMappingKind.ARRAY,
                FlowMappingKind.MERGE,
                FlowMappingKind.COALESCE,
                FlowMappingKind.FILE,
                FlowMappingKind.FILE_ARRAY,
            ],
        )
        self.assertEqual(
            mapper.mappings[0].source,
            parse_flow_selector("input.payload"),
        )
        self.assertEqual(
            mapper.mappings[1].fields["answer"],
            parse_flow_selector("source.value"),
        )
        self.assertEqual(
            mapper.mappings[2].items[1],
            parse_flow_selector("source.value"),
        )
        self.assertEqual(
            mapper.mappings[3].sources[0],
            parse_flow_selector("input.payload"),
        )
        self.assertEqual(
            mapper.mappings[4].sources[0],
            parse_flow_selector("source.missing"),
        )
        self.assertEqual(
            mapper.mappings[5].source,
            parse_flow_selector("input.document"),
        )
        assert mapper.join is not None
        self.assertEqual(mapper.join.type, FlowJoinPolicyType.COLLECT)
        self.assertEqual(mapper.join.optional_inputs, ("optional",))
        assert mapper.retry is not None
        self.assertEqual(mapper.retry.max_attempts, 3)
        self.assertEqual(
            mapper.retry.backoff,
            FlowRetryBackoffStrategy.EXPONENTIAL,
        )
        self.assertEqual(mapper.retry.exhausted_route, "failed")
        assert mapper.timeout is not None
        self.assertEqual(mapper.timeout.per_attempt_seconds, 5)
        assert mapper.loop is not None
        self.assertEqual(mapper.loop.max_iterations, 2)
        self.assertEqual(mapper.loop.max_elapsed_seconds, 30)
        self.assertEqual(mapper.loop.limit_route, "limited")
        self.assertEqual(
            mapper.loop.output_selector,
            parse_flow_selector("mapper.result"),
        )
        self.assertEqual(
            mapper.loop.continue_condition.selector,
            parse_flow_selector("mapper.result.more"),
        )
        self.assertEqual(
            mapper.loop.exit_condition.selector,
            parse_flow_selector("mapper.result.done"),
        )

        edge = plan.edges[0]
        self.assertEqual(edge.index, 0)
        self.assertEqual(edge.kind, FlowEdgeKind.SUCCESS)
        self.assertEqual(edge.priority, 1)
        self.assertEqual(edge.label, "ready")
        self.assertEqual(
            edge.routing_policy,
            FlowRouteMatchPolicy.ALL_MATCHING,
        )
        assert edge.condition is not None
        self.assertEqual(edge.condition.operator, FlowConditionOperator.ALL)
        self.assertEqual(
            edge.condition.conditions[0].selector,
            parse_flow_selector("source.value.status"),
        )
        assert edge.condition.conditions[1].condition is not None
        self.assertEqual(
            edge.condition.conditions[1].condition.selector,
            parse_flow_selector("source.value.blocked"),
        )

        with self.assertRaises(FrozenInstanceError):
            mapper.name = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            cast(dict[str, object], mapper.config)["shape"] = "changed"
        with self.assertRaises(TypeError):
            cast(dict[str, object], plan.output_selectors)[
                "answer"
            ] = "changed"

    def test_compile_flow_definition_refuses_invalid_definition(self) -> None:
        result = compile_flow_definition(
            FlowDefinition(
                name="invalid",
                version="2026-06-07",
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
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "env.SECRET"},
                ),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.plan)
        self.assertEqual(result.diagnostics[0].code, "flow.reserved_selector")
        self.assertNotIn("SECRET", str(result.public_diagnostics))

    def test_compile_flow_definition_refuses_legacy_definition(self) -> None:
        result = compile_flow_definition(
            FlowDefinition(
                name="legacy",
                entrypoint="start",
                output_node="finish",
                nodes=(
                    FlowNodeDefinition(name="start", type="input"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(FlowEdgeDefinition(source="start", target="finish"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.plan)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.plan_requires_strict_definition",
        )
        self.assertEqual(
            result.public_diagnostics[0]["category"],
            FlowDiagnosticCategory.EXECUTION.value,
        )

    def test_compile_flow_definition_does_not_build_nodes(self) -> None:
        registry = self._registry()

        result = compile_flow_definition(self._definition(), registry)

        self.assertTrue(result.ok, result.public_diagnostics)

    def test_compile_flow_definition_adds_tool_descriptor_metadata(
        self,
    ) -> None:
        registry = tool_flow_node_registry(_tool_manager())

        result = compile_flow_definition(
            FlowDefinition(
                name="tool-plan",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.JSON,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="calculate"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "calculate.result"},
                ),
                nodes=(
                    FlowNodeDefinition(
                        name="calculate",
                        type="tool",
                        ref="sum",
                        config={"arguments": {"a": "left", "b": "right"}},
                    ),
                ),
            ),
            registry,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        tool_node = result.plan.node_map["calculate"]
        tool_metadata = tool_node.metadata["tool"]
        assert isinstance(tool_metadata, Mapping)
        self.assertEqual(
            tool_metadata["canonical_name"],
            "plan_flow_adder",
        )
        self.assertEqual(tool_metadata["aliases"], ("sum",))
        self.assertIn("parameter_schema", tool_metadata)

    def test_compile_flow_definition_omits_absent_tool_schemas(self) -> None:
        registry = default_flow_node_registry()
        registry.register(
            "tool",
            lambda definition: Node(definition.name),
            metadata=FlowNodeMetadata(
                kind=FlowNodeKind.TOOL,
                supports_ref=True,
                output_contract=FlowNodeContract(
                    name="result",
                    type=FlowOutputType.JSON,
                ),
            ),
        )
        registry.register_tool_resolver(
            "tool",
            _SchemaFreeToolResolver(),
            {"raw_tool": ToolDescriptor(name="raw_tool")},
        )

        result = compile_flow_definition(
            FlowDefinition(
                name="tool-plan",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.JSON,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="raw"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "raw.result"},
                ),
                nodes=(
                    FlowNodeDefinition(
                        name="raw",
                        type="tool",
                        ref="raw_tool",
                    ),
                ),
            ),
            registry,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        tool_metadata = result.plan.node_map["raw"].metadata["tool"]
        assert isinstance(tool_metadata, Mapping)
        self.assertEqual(tool_metadata["canonical_name"], "raw_tool")
        self.assertNotIn("parameter_schema", tool_metadata)
        self.assertNotIn("return_schema", tool_metadata)

    def test_compile_flow_definition_refuses_invalid_tool_node(
        self,
    ) -> None:
        registry = tool_flow_node_registry(_tool_manager())

        result = compile_flow_definition(
            FlowDefinition(
                name="tool-plan",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.JSON,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="calculate"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "calculate.result"},
                ),
                nodes=(
                    FlowNodeDefinition(
                        name="calculate",
                        type="tool",
                        ref="plan_flow_adder",
                    ),
                ),
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.plan)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.missing_argument_binding",
        )

    def test_condition_plan_accepts_literal_values(self) -> None:
        plan = FlowConditionPlan(
            operator=FlowConditionOperator.IN,
            selector=parse_flow_selector("node.output"),
            value={"safe": ["one"]},
            value_selector=parse_flow_selector("other.output"),
            values=({"safe": "two"},),
        )

        self.assertEqual(
            cast(dict[str, object], plan.value)["safe"],
            ("one",),
        )
        self.assertEqual(
            cast(dict[str, object], plan.values[0])["safe"],
            "two",
        )
        with self.assertRaises(TypeError):
            cast(dict[str, object], plan.value)["safe"] = "changed"

    def test_plan_entities_cover_optional_constructor_paths(self) -> None:
        condition = FlowConditionPlan(
            operator=FlowConditionOperator.IS_TYPE,
            selector=parse_flow_selector("node.output"),
            value_type=FlowConditionValueType.OBJECT,
        )
        mapping = FlowMappingPlan(
            target="value",
            kind=FlowMappingKind.SELECT,
        )
        join = FlowJoinPlan(
            type=FlowJoinPolicyType.QUORUM,
            quorum=1,
        )
        node = FlowNodePlan(
            name="node",
            type="echo",
            kind=FlowNodeKind.PASS_THROUGH,
            ref="trusted",
        )

        self.assertEqual(condition.value_type, FlowConditionValueType.OBJECT)
        self.assertEqual(mapping.fields, {})
        self.assertEqual(join.quorum, 1)
        self.assertEqual(node.config, {})
        self.assertEqual(node.metadata, {})
        self.assertEqual(node.ref, "trusted")

    def test_compile_result_validates_diagnostics(self) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.execution.test",
            category=FlowDiagnosticCategory.EXECUTION,
            path="flow",
            message="Flow execution test diagnostic.",
        )

        result = compile_flow_definition(self._definition(), self._registry())
        manual = type(result)(diagnostics=(diagnostic,))

        self.assertFalse(manual.ok)
        self.assertEqual(manual.public_diagnostics[0]["code"], diagnostic.code)

    def _registry(self) -> FlowNodeRegistry:
        def raising_factory(definition: FlowNodeDefinition) -> Node:
            raise AssertionError(f"factory called for {definition.name}")

        return default_flow_node_registry().register(
            "mapper",
            raising_factory,
            metadata=FlowNodeMetadata(
                kind=FlowNodeKind.SELECT,
                input_contracts=(
                    FlowNodeContract(
                        name="selected",
                        type=FlowInputType.OBJECT,
                        metadata={"dynamic": True},
                    ),
                    FlowNodeContract(
                        name="constructed",
                        type=FlowInputType.OBJECT,
                        metadata={"dynamic": True},
                    ),
                    FlowNodeContract(
                        name="items",
                        type=FlowInputType.ARRAY,
                        metadata={"dynamic": True},
                    ),
                    FlowNodeContract(
                        name="merged",
                        type=FlowInputType.OBJECT,
                        metadata={"dynamic": True},
                    ),
                    FlowNodeContract(
                        name="fallback",
                        type=FlowInputType.OBJECT,
                        metadata={"dynamic": True},
                    ),
                    FlowNodeContract(
                        name="document",
                        type=FlowInputType.FILE,
                        metadata={"dynamic": True},
                    ),
                    FlowNodeContract(
                        name="documents",
                        type=FlowInputType.FILE_ARRAY,
                        metadata={"dynamic": True},
                    ),
                    FlowNodeContract(
                        name="optional",
                        type=FlowInputType.STRING,
                        metadata={"optional": True},
                    ),
                ),
                output_contract=FlowNodeContract(
                    name="result",
                    type=FlowOutputType.OBJECT,
                    metadata={"dynamic": True},
                ),
                metadata={"mode": "projection"},
            ),
        )

    def _definition(self) -> FlowDefinition:
        return FlowDefinition(
            name="runtime",
            version="2026-06-07",
            revision="rev-1",
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
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
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="source"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "mapper.result"},
            ),
            nodes=(
                FlowNodeDefinition(
                    name="source",
                    type="constant",
                    config={
                        "value": {
                            "status": "ready",
                            "blocked": False,
                        }
                    },
                ),
                FlowNodeDefinition(
                    name="mapper",
                    type="mapper",
                    join_policy=FlowJoinPolicy(
                        type=FlowJoinPolicyType.COLLECT,
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
                            operator=FlowConditionOperator.EXISTS,
                            selector="mapper.result.done",
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


class _SchemaFreeToolResolver:
    def list_tools(self) -> list[ToolDescriptor]:
        return [ToolDescriptor(name="raw_tool")]

    def resolve_tool_name(
        self,
        name: str,
        *,
        provider_originated: bool = False,
    ) -> ToolNameResolution:
        _ = provider_originated
        return ToolNameResolution(
            requested_name=name,
            status=ToolNameResolutionStatus.EXACT,
            canonical_name="raw_tool",
            candidates=["raw_tool"],
        )

    def validate_tool_call(self, call: ToolCall) -> None:
        _ = call
        return None

    async def execute_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
    ) -> ToolCallOutcome:
        _ = context
        return ToolCallResult(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            call=call,
            result=None,
        )


if __name__ == "__main__":
    main()
