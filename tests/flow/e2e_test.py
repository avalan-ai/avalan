from collections.abc import Mapping
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.event import Event, EventType
from avalan.flow import (
    FlowCondition,
    FlowConditionOperator,
    FlowDefinition,
    FlowDefinitionLoader,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowEdgeState,
    FlowEntryBehavior,
    FlowExecutor,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowInspectionRunState,
    FlowLoopPolicy,
    FlowMappingKind,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeExecutionError,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodePlan,
    FlowNodeRegistry,
    FlowNodeState,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowPlanNodeRunner,
    FlowRetryPolicy,
    FlowReviewState,
    Node,
    compile_flow_source,
    default_flow_node_registry,
    serialize_flow_definition,
)


class FlowE2ETestCase(IsolatedAsyncioTestCase):
    async def test_graph_compile_strict_serialization_runs_equivalently(
        self,
    ) -> None:
        compiled = await compile_flow_source("""
            [flow]
            name = "graph_runtime"
            version = "1"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "result"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            result = "finish.value"

            [graph]
            format = "mermaid"
            source = "inline"
            mode = "executable"
            diagram = '''
            flowchart LR
            start route_1@-->|Private graph label| finish
            start -.-> private_note["Private graph note"]
            '''

            [graph.edges.route_1]
            label = "approved"

            [nodes.start]
            type = "constant"
            value = {answer = "ok"}

            [nodes.finish]
            type = "pass-through"

            [nodes.finish.mapping.value]
            type = "select"
            source = "start.value"
            """)
        invalid = await compile_flow_source("""
            [flow]
            name = "invalid_graph_runtime"
            entrypoint = "start"
            output_node = "finish"

            [graph]
            format = "mermaid"
            source = "inline"
            mode = "executable"
            diagram = '''
            flowchart LR
            start -->|Private graph label| finish
            '''

            [nodes.start]
            type = "constant"
            value = {answer = "ok"}

            [nodes.finish]
            type = "pass-through"
            """)

        self.assertTrue(compiled.ok, compiled.public_diagnostics)
        self.assertTrue(compiled.authoring_graph)
        assert compiled.definition is not None
        assert compiled.canonical_source is not None
        reloaded = await FlowDefinitionLoader().loads_validation_result(
            compiled.canonical_source,
        )
        self.assertTrue(reloaded.ok, reloaded.public_diagnostics)
        self.assertFalse(reloaded.authoring_graph)
        assert reloaded.definition is not None
        self.assertIsNone(compiled.definition.definition_base)
        self.assertEqual(compiled.definition, reloaded.definition)
        self.assertEqual(
            serialize_flow_definition(reloaded.definition),
            compiled.canonical_source,
        )
        self.assertNotIn("[graph]", compiled.canonical_source)
        self.assertNotIn("Private graph", compiled.canonical_source)

        executor = FlowExecutor()
        original_run = await executor.run(compiled.definition)
        strict_run = await executor.run(reloaded.definition)

        self.assertTrue(original_run.ok, original_run.public_diagnostics)
        self.assertTrue(strict_run.ok, strict_run.public_diagnostics)
        self.assertEqual(original_run.outputs, {"result": {"answer": "ok"}})
        self.assertEqual(strict_run.outputs, original_run.outputs)
        self.assertFalse(invalid.ok)
        self.assertIsNone(invalid.definition)
        self.assertEqual(
            [diagnostic["code"] for diagnostic in invalid.public_diagnostics],
            ["flow.graph.unsupported_executable_edge"],
        )
        self.assertNotIn("Private graph label", str(invalid.as_public_dict()))

    async def test_human_review_pauses_and_resumes_medium_risk_routes(
        self,
    ) -> None:
        decisions = {
            "approved": "approved_sink",
            "rejected": "rejected_sink",
            "needs-correction": "correction_sink",
            "expired": "expired_sink",
        }

        for decision, target in decisions.items():
            with self.subTest(decision=decision):
                events: list[Event] = []
                calls: list[str] = []
                definition = _human_review_definition()
                executor = FlowExecutor(
                    registry=_human_review_registry(),
                    runner=_human_review_runner(calls),
                    event_listener=events.append,
                )

                paused = await executor.run(
                    definition,
                    inputs={
                        "application": {
                            "vendor": "Northwind",
                            "risk": "medium",
                            "private_account": "acct-private-57721",
                        },
                    },
                )
                paused_inspection = executor.inspect(paused)
                paused_export = executor.export_trace(paused)

                self.assertTrue(paused.ok, paused.public_diagnostics)
                self.assertEqual(paused.outputs, {})
                self.assertEqual(calls, [])
                self.assertEqual(
                    paused_inspection.state, FlowInspectionRunState.PAUSED
                )
                self.assertEqual(len(paused_inspection.reviews), 1)
                paused_review = paused_inspection.reviews[0]
                self.assertEqual(paused_review.node, "review")
                self.assertEqual(paused_review.state, FlowReviewState.PAUSED)
                self.assertTrue(paused_review.has_pause_token)
                self.assertEqual(
                    paused_review.allowed_decisions,
                    tuple(decisions),
                )
                self.assertEqual(paused_review.timeout_seconds, 600)
                self.assertNotIn("acct-private-57721", str(paused_export))

                resumed = await executor.resume(
                    definition,
                    paused,
                    decisions={
                        "review": {
                            "decision": decision,
                            "comment": "private-review-token",
                        },
                    },
                )
                resumed_inspection = executor.inspect(resumed)
                resumed_export = executor.export_trace(resumed)
                self.assertTrue(resumed.ok, resumed.public_diagnostics)
                assert resumed.result is not None
                review_result = cast(
                    Mapping[str, object],
                    cast(
                        Mapping[str, object],
                        resumed.result.node_outputs["review"],
                    )["result"],
                )

                self.assertEqual(resumed.outputs, {"decision": decision})
                self.assertEqual(calls, [target])
                self.assertEqual(review_result["decision"], decision)
                self.assertEqual(
                    {
                        node.node: node.state
                        for node in resumed_inspection.nodes
                    },
                    {
                        "review": FlowNodeState.SUCCEEDED,
                        "approved_sink": (
                            FlowNodeState.SUCCEEDED
                            if target == "approved_sink"
                            else FlowNodeState.SKIPPED
                        ),
                        "rejected_sink": (
                            FlowNodeState.SUCCEEDED
                            if target == "rejected_sink"
                            else FlowNodeState.SKIPPED
                        ),
                        "correction_sink": (
                            FlowNodeState.SUCCEEDED
                            if target == "correction_sink"
                            else FlowNodeState.SKIPPED
                        ),
                        "expired_sink": (
                            FlowNodeState.SUCCEEDED
                            if target == "expired_sink"
                            else FlowNodeState.SKIPPED
                        ),
                        "timeout_sink": FlowNodeState.SKIPPED,
                    },
                )
                resumed_review = resumed_inspection.reviews[0]
                self.assertEqual(
                    resumed_inspection.state,
                    FlowInspectionRunState.SUCCEEDED,
                )
                self.assertEqual(
                    resumed_review.state,
                    FlowReviewState.SUCCEEDED,
                )
                self.assertFalse(resumed_review.has_pause_token)
                self.assertEqual(
                    resumed_review.allowed_decisions,
                    tuple(decisions),
                )
                self.assertNotIn("private-review-token", str(resumed_export))
                self.assertNotIn("acct-private-57721", str(resumed_export))
                self.assertEqual(
                    _event_payloads(events, EventType.FLOW_NODE_PAUSED)[0][
                        "node"
                    ],
                    "review",
                )
                self.assertEqual(
                    _event_payloads(events, EventType.FLOW_NODE_RESUMED)[0][
                        "node"
                    ],
                    "review",
                )
                self.assertEqual(
                    next(
                        payload
                        for payload in _event_payloads(
                            events,
                            EventType.FLOW_EDGE_ROUTED,
                        )
                        if payload["status"] == "taken"
                    )["target"],
                    target,
                )
                self.assertNotIn("private-review-token", str(events))
                self.assertNotIn("acct-private-57721", str(events))

    async def test_human_review_without_timeout_route_fails_closed(
        self,
    ) -> None:
        calls: list[str] = []
        executor = FlowExecutor(
            registry=_human_review_registry(),
            runner=_human_review_runner(calls),
        )

        result = await executor.run(
            _human_review_definition(include_timeout_route=False),
            inputs={
                "application": {
                    "vendor": "Northwind",
                    "private_account": "acct-private-86420",
                },
            },
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.result)
        self.assertEqual(calls, [])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.missing_human_review_timeout_route"],
        )
        self.assertNotIn("acct-private-86420", str(result.public_diagnostics))

    async def test_fallback_path_exhausts_to_manual_verification(
        self,
    ) -> None:
        events: list[Event] = []
        calls: list[str] = []
        executor = FlowExecutor(
            runner=_fallback_runner(calls),
            event_listener=events.append,
        )

        result = await executor.run(
            _fallback_definition(),
            inputs={
                "vendor": {
                    "name": "Northwind",
                    "private_account": "acct-private-314159",
                },
            },
        )
        inspection = executor.inspect(result)
        exported = executor.export_trace(result)

        self.assertFalse(result.ok)
        self.assertEqual(
            result.outputs,
            {
                "verification": {
                    "status": "manual_verification",
                    "diagnostic_code": "flow.execution.provider_unavailable",
                },
            },
        )
        self.assertEqual(
            calls,
            [
                "validate_input",
                "provider_check",
                "provider_check",
                "provider_check",
                "manual_verification",
            ],
        )
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.provider_unavailable"],
        )
        self.assertEqual(inspection.state, FlowInspectionRunState.FAILED)
        self.assertEqual(
            {
                retry.node: (retry.attempts, retry.exhausted)
                for retry in inspection.retries
            },
            {"provider_check": (3, True)},
        )
        self.assertEqual(
            {node.node: node.state for node in inspection.nodes},
            {
                "validate_input": FlowNodeState.SUCCEEDED,
                "provider_check": FlowNodeState.FAILED,
                "manual_verification": FlowNodeState.SUCCEEDED,
            },
        )
        self.assertEqual(
            {edge.index: edge.state for edge in inspection.edges},
            {
                0: FlowEdgeState.TAKEN,
                1: FlowEdgeState.TAKEN,
            },
        )
        self.assertNotIn("acct-private-314159", str(result.public_diagnostics))
        self.assertNotIn("acct-private-314159", str(exported))
        self.assertEqual(
            _event_payloads(events, EventType.FLOW_NODE_RETRYING)[0][
                "diagnostic_codes"
            ],
            ("flow.execution.provider_unavailable",),
        )
        self.assertEqual(
            _event_payloads(events, EventType.FLOW_NODE_FAILED)[0]["attempts"],
            3,
        )
        self.assertEqual(
            _event_payloads(events, EventType.FLOW_EDGE_ROUTED)[1]["target"],
            "manual_verification",
        )
        self.assertTrue(
            {
                EventType.FLOW_VALIDATION,
                EventType.FLOW_STARTED,
                EventType.FLOW_NODE_STARTED,
                EventType.FLOW_NODE_COMPLETED,
                EventType.FLOW_NODE_RETRYING,
                EventType.FLOW_NODE_FAILED,
                EventType.FLOW_EDGE_ROUTED,
                EventType.FLOW_OUTPUT_SELECTED,
                EventType.FLOW_COMPLETED,
            }.issubset({event.type for event in events})
        )
        self.assertEqual(
            [
                (payload["node"], payload["status"])
                for payload in _event_payloads(
                    events,
                    EventType.FLOW_NODE_STARTED,
                )
            ],
            [
                ("validate_input", "started"),
                ("provider_check", "started"),
                ("manual_verification", "started"),
            ],
        )

    async def test_retry_exhaustion_without_error_route_fails_closed(
        self,
    ) -> None:
        events: list[Event] = []
        calls: list[str] = []
        executor = FlowExecutor(
            runner=_fallback_runner(calls),
            event_listener=events.append,
        )

        result = await executor.run(
            _fallback_definition(with_error_route=False),
            inputs={"vendor": {"private_account": "acct-private-271828"}},
        )
        inspection = executor.inspect(result)

        self.assertFalse(result.ok)
        self.assertEqual(
            calls,
            ["validate_input", "provider_check", "provider_check"],
        )
        self.assertEqual(result.outputs, {})
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            [
                "flow.execution.provider_unavailable",
                "flow.execution.missing_failure_route",
                "flow.execution.missing_output",
            ],
        )
        self.assertEqual(
            {node.node: node.state for node in inspection.nodes},
            {
                "validate_input": FlowNodeState.SUCCEEDED,
                "provider_check": FlowNodeState.FAILED,
                "manual_verification": FlowNodeState.SKIPPED,
            },
        )
        self.assertEqual(
            {
                retry.node: (retry.attempts, retry.exhausted)
                for retry in inspection.retries
            },
            {"provider_check": (2, True)},
        )
        self.assertEqual(
            [
                (payload["node"], payload["status"])
                for payload in _event_payloads(
                    events,
                    EventType.FLOW_NODE_SKIPPED,
                )
            ],
            [("manual_verification", "skipped")],
        )
        self.assertEqual(
            _event_payloads(events, EventType.FLOW_OUTPUT_SELECTED)[0][
                "status"
            ],
            "failed",
        )
        self.assertNotIn("acct-private-271828", str(result.public_diagnostics))
        self.assertNotIn("acct-private-271828", str(events))

    async def test_validate_and_repair_loop_exits_normally(
        self,
    ) -> None:
        events: list[Event] = []
        calls: list[str] = []
        executor = FlowExecutor(
            runner=_repair_loop_runner(calls, exit_after=3),
            event_listener=events.append,
        )

        result = await executor.run(
            _repair_loop_definition(),
            inputs={
                "payload": {
                    "status": "draft",
                    "private_account": "acct-private-loop-123",
                },
            },
        )
        inspection = executor.inspect(result)
        exported = executor.export_trace(result)

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(
            calls,
            ["repair", "repair", "repair", "finished"],
        )
        self.assertEqual(
            result.outputs,
            {"repair": {"status": "valid", "attempts": 3}},
        )
        self.assertEqual(inspection.state, FlowInspectionRunState.SUCCEEDED)
        self.assertEqual(
            {
                node.node: (node.state, node.attempts)
                for node in inspection.nodes
            },
            {
                "repair": (FlowNodeState.SUCCEEDED, 3),
                "finished": (FlowNodeState.SUCCEEDED, 1),
                "manual": (FlowNodeState.SKIPPED, 0),
            },
        )
        self.assertEqual(
            {edge.index: edge.state for edge in inspection.edges},
            {
                0: FlowEdgeState.TAKEN,
                1: FlowEdgeState.PENDING,
            },
        )
        self.assertEqual(len(inspection.loops), 1)
        self.assertEqual(inspection.loops[0].node, "repair")
        self.assertEqual(inspection.loops[0].iterations, 3)
        self.assertEqual(inspection.loops[0].state, FlowNodeState.SUCCEEDED)
        self.assertEqual(inspection.loops[0].max_iterations, 4)
        self.assertEqual(inspection.loops[0].limit_route, "manual")
        self.assertEqual(
            [
                payload["matched"]
                for payload in _event_payloads(
                    events,
                    EventType.FLOW_CONDITION_EVALUATED,
                )
            ],
            [False, True, False, True, True],
        )
        self.assertEqual(
            _event_payloads(events, EventType.FLOW_OUTPUT_SELECTED)[0][
                "output_name"
            ],
            "repair",
        )
        self.assertNotIn("acct-private-loop-123", str(exported))
        self.assertNotIn("repair-private-output", str(exported))
        self.assertNotIn("acct-private-loop-123", str(events))
        self.assertNotIn("repair-private-output", str(events))

    async def test_validate_and_repair_loop_routes_iteration_limit(
        self,
    ) -> None:
        events: list[Event] = []
        calls: list[str] = []
        executor = FlowExecutor(
            runner=_repair_loop_runner(calls, exit_after=None),
            event_listener=events.append,
        )

        result = await executor.run(
            _repair_loop_definition(
                max_iterations=2,
                output_selector="manual.value",
            ),
            inputs={
                "payload": {
                    "status": "invalid",
                    "private_account": "acct-private-loop-987",
                },
            },
        )
        inspection = executor.inspect(result)
        exported = executor.export_trace(result)

        self.assertFalse(result.ok)
        self.assertEqual(calls, ["repair", "repair", "manual"])
        self.assertEqual(
            result.outputs,
            {"repair": {"status": "manual_verification", "attempts": 2}},
        )
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.loop_limit_reached"],
        )
        self.assertEqual(inspection.state, FlowInspectionRunState.FAILED)
        self.assertEqual(
            {
                node.node: (node.state, node.attempts)
                for node in inspection.nodes
            },
            {
                "repair": (FlowNodeState.FAILED, 2),
                "finished": (FlowNodeState.SKIPPED, 0),
                "manual": (FlowNodeState.SUCCEEDED, 1),
            },
        )
        self.assertEqual(
            {edge.index: edge.state for edge in inspection.edges},
            {
                0: FlowEdgeState.PENDING,
                1: FlowEdgeState.TAKEN,
            },
        )
        self.assertEqual(len(inspection.loops), 1)
        self.assertEqual(inspection.loops[0].node, "repair")
        self.assertEqual(inspection.loops[0].iterations, 2)
        self.assertEqual(inspection.loops[0].state, FlowNodeState.FAILED)
        self.assertEqual(inspection.loops[0].max_iterations, 2)
        self.assertEqual(inspection.loops[0].limit_route, "manual")
        self.assertEqual(
            [
                payload["matched"]
                for payload in _event_payloads(
                    events,
                    EventType.FLOW_CONDITION_EVALUATED,
                )
            ],
            [False, True, False, True],
        )
        self.assertEqual(
            _event_payloads(events, EventType.FLOW_NODE_FAILED)[0][
                "diagnostic_codes"
            ],
            ("flow.execution.loop_limit_reached",),
        )
        self.assertEqual(
            next(
                payload
                for payload in _event_payloads(
                    events,
                    EventType.FLOW_EDGE_ROUTED,
                )
                if payload["status"] == "taken"
            )["target"],
            "manual",
        )
        self.assertNotIn(
            "acct-private-loop-987",
            str(result.public_diagnostics),
        )
        self.assertNotIn("acct-private-loop-987", str(exported))
        self.assertNotIn("repair-private-output", str(exported))
        self.assertNotIn("acct-private-loop-987", str(events))
        self.assertNotIn("repair-private-output", str(events))


def _fallback_runner(
    calls: list[str],
) -> FlowPlanNodeRunner:
    async def runner(
        node: FlowNodePlan,
        inputs: Mapping[str, object],
    ) -> object:
        calls.append(node.name)
        if node.name == "provider_check":
            raise FlowNodeExecutionError(
                code="flow.execution.provider_unavailable",
                message="Flow node provider is unavailable.",
                hint="Use the declared fallback route.",
                failure_category="transient",
            )
        if node.name == "manual_verification":
            return {
                "value": {
                    "status": "manual_verification",
                    "diagnostic_code": "flow.execution.provider_unavailable",
                }
            }
        return {"value": inputs["value"]}

    return runner


def _human_review_runner(
    calls: list[str],
) -> FlowPlanNodeRunner:
    async def runner(
        node: FlowNodePlan,
        inputs: Mapping[str, object],
    ) -> object:
        calls.append(node.name)
        return {
            "status": node.name,
            "received": tuple(sorted(inputs)),
        }

    return runner


def _human_review_registry() -> FlowNodeRegistry:
    registry = default_flow_node_registry()
    registry.register(
        "human_review",
        lambda definition: Node(definition.name),
        metadata=FlowNodeMetadata(
            kind=FlowNodeKind.HUMAN_REVIEW,
            async_only=True,
            capabilities=(FlowNodeCapability.DURABLE_PAUSE,),
            input_contract=FlowNodeContract(
                name="payload",
                type=FlowInputType.OBJECT,
            ),
            output_contract=FlowNodeContract(
                name="result",
                type=FlowOutputType.OBJECT,
            ),
        ),
    )
    return registry


def _human_review_definition(
    *,
    include_timeout_route: bool = True,
) -> FlowDefinition:
    decisions = (
        "approved",
        "rejected",
        "needs-correction",
        "expired",
    )
    decision_targets = {
        "approved": "approved_sink",
        "rejected": "rejected_sink",
        "needs-correction": "correction_sink",
        "expired": "expired_sink",
    }
    edges = [
        FlowEdgeDefinition(
            source="review",
            target=target,
            label=decision,
            kind=FlowEdgeKind.RESUME,
            priority=index,
        )
        for index, (decision, target) in enumerate(decision_targets.items())
    ]
    if include_timeout_route:
        edges.append(
            FlowEdgeDefinition(
                source="review",
                target="timeout_sink",
                label="expired",
                kind=FlowEdgeKind.TIMEOUT,
            )
        )
    return FlowDefinition(
        name="human-review-e2e",
        version="2026-06-08",
        revision="r1",
        inputs=(
            FlowInputDefinition(
                name="application",
                type=FlowInputType.OBJECT,
            ),
        ),
        outputs=(
            FlowOutputDefinition(
                name="decision",
                type=FlowOutputType.JSON,
            ),
        ),
        entry_behavior=FlowEntryBehavior(node="review"),
        output_behavior=FlowOutputBehavior(
            outputs={"decision": "review.result.decision"}
        ),
        nodes=(
            FlowNodeDefinition(
                name="review",
                type="human_review",
                mappings=(
                    FlowInputMapping(
                        target="payload",
                        kind=FlowMappingKind.SELECT,
                        source="inputs.application",
                    ),
                ),
                config={
                    "allowed_decisions": decisions,
                    "payload_schema": {
                        "type": "object",
                        "properties": {
                            "vendor": {"type": "string"},
                            "risk": {"type": "string"},
                        },
                        "required": ("vendor", "risk"),
                    },
                    "decision_schema": {
                        "type": "object",
                        "properties": {
                            "decision": {"enum": decisions},
                            "comment": {"type": "string"},
                        },
                        "required": ("decision",),
                    },
                    "timeout_seconds": 600,
                    "audit_metadata": {
                        "risk": "medium",
                        "queue": "ops",
                    },
                },
            ),
            FlowNodeDefinition(name="approved_sink", type="pass-through"),
            FlowNodeDefinition(name="rejected_sink", type="pass-through"),
            FlowNodeDefinition(name="correction_sink", type="pass-through"),
            FlowNodeDefinition(name="expired_sink", type="pass-through"),
            FlowNodeDefinition(name="timeout_sink", type="pass-through"),
        ),
        edges=tuple(edges),
    )


def _fallback_definition(
    *,
    with_error_route: bool = True,
) -> FlowDefinition:
    edges = [
        FlowEdgeDefinition(
            source="validate_input",
            target="provider_check",
        ),
    ]
    if with_error_route:
        edges.append(
            FlowEdgeDefinition(
                source="provider_check",
                target="manual_verification",
                kind=FlowEdgeKind.ERROR,
            )
        )
    return FlowDefinition(
        name="fallback-e2e",
        version="2026-06-08",
        revision="r1",
        inputs=(
            FlowInputDefinition(
                name="vendor",
                type=FlowInputType.OBJECT,
            ),
        ),
        outputs=(
            FlowOutputDefinition(
                name="verification",
                type=FlowOutputType.OBJECT,
            ),
        ),
        entry_behavior=FlowEntryBehavior(node="validate_input"),
        output_behavior=FlowOutputBehavior(
            outputs={"verification": "manual_verification.value"}
        ),
        nodes=(
            FlowNodeDefinition(
                name="validate_input",
                type="pass-through",
                mappings=(
                    FlowInputMapping(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source="inputs.vendor",
                    ),
                ),
            ),
            FlowNodeDefinition(
                name="provider_check",
                type="pass-through",
                retry_policy=FlowRetryPolicy(
                    max_attempts=3 if with_error_route else 2,
                    retryable_categories=("transient",),
                    exhausted_route=(
                        "manual_verification" if with_error_route else None
                    ),
                ),
                mappings=(
                    FlowInputMapping(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source="validate_input.value",
                    ),
                ),
            ),
            FlowNodeDefinition(
                name="manual_verification",
                type="pass-through",
            ),
        ),
        edges=tuple(edges),
    )


def _repair_loop_runner(
    calls: list[str],
    *,
    exit_after: int | None,
) -> FlowPlanNodeRunner:
    async def runner(
        node: FlowNodePlan,
        _: Mapping[str, object],
    ) -> object:
        calls.append(node.name)
        if node.name == "repair":
            attempts = calls.count("repair")
            done = exit_after is not None and attempts >= exit_after
            return {
                "done": done,
                "more": not done,
                "safe": {
                    "status": "valid" if done else "needs_repair",
                    "attempts": attempts,
                },
                "private": "repair-private-output",
            }
        if node.name == "manual":
            return {
                "value": {
                    "status": "manual_verification",
                    "attempts": calls.count("repair"),
                }
            }
        return {"value": {"status": node.name}}

    return runner


def _repair_loop_definition(
    *,
    max_iterations: int = 4,
    output_selector: str = "repair.value",
) -> FlowDefinition:
    return FlowDefinition(
        name="repair-loop-e2e",
        version="2026-06-08",
        revision="r1",
        inputs=(
            FlowInputDefinition(
                name="payload",
                type=FlowInputType.OBJECT,
            ),
        ),
        outputs=(
            FlowOutputDefinition(
                name="repair",
                type=FlowOutputType.OBJECT,
            ),
        ),
        entry_behavior=FlowEntryBehavior(node="repair"),
        output_behavior=FlowOutputBehavior(
            outputs={"repair": output_selector}
        ),
        nodes=(
            FlowNodeDefinition(
                name="repair",
                type="pass-through",
                mappings=(
                    FlowInputMapping(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source="inputs.payload",
                    ),
                ),
                loop_policy=FlowLoopPolicy(
                    max_iterations=max_iterations,
                    continue_condition=FlowCondition(
                        operator=FlowConditionOperator.EQ,
                        selector="repair.value.more",
                        value=True,
                    ),
                    exit_condition=FlowCondition(
                        operator=FlowConditionOperator.EQ,
                        selector="repair.value.done",
                        value=True,
                    ),
                    output_selector="repair.value.safe",
                    limit_route="manual",
                ),
            ),
            FlowNodeDefinition(name="finished", type="pass-through"),
            FlowNodeDefinition(name="manual", type="pass-through"),
        ),
        edges=(
            FlowEdgeDefinition(
                source="repair",
                target="finished",
                kind=FlowEdgeKind.SUCCESS,
            ),
            FlowEdgeDefinition(
                source="repair",
                target="manual",
                kind=FlowEdgeKind.ERROR,
            ),
        ),
    )


def _event_payloads(
    events: list[Event],
    event_type: EventType,
) -> list[Mapping[str, object]]:
    return [
        cast(Mapping[str, object], event.payload)
        for event in events
        if event.type == event_type
    ]


if __name__ == "__main__":
    main()
