from asyncio import CancelledError, sleep
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.flow import (
    FlowConditionOperator,
    FlowConditionPlan,
    FlowConditionValueType,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeKind,
    FlowEdgePlan,
    FlowEdgeState,
    FlowExecutionPlan,
    FlowExecutionTrace,
    FlowInputDefinition,
    FlowInputType,
    FlowJoinPlan,
    FlowJoinPolicyType,
    FlowLoopPlan,
    FlowMappingKind,
    FlowMappingPlan,
    FlowNodeContract,
    FlowNodeExecutionError,
    FlowNodeKind,
    FlowNodePlan,
    FlowNodeState,
    FlowOutputDefinition,
    FlowOutputType,
    FlowPlanExecutionResult,
    FlowRetryBackoffStrategy,
    FlowRetryPlan,
    FlowRouteMatchPolicy,
    FlowRuntimeContext,
    FlowRuntimeEvaluationError,
    FlowTimeoutPlan,
    evaluate_flow_condition_plan,
    evaluate_flow_mappings,
    evaluate_flow_node_mappings,
    evaluate_flow_selector,
    execute_flow_plan,
    parse_flow_selector,
    resolve_flow_selector_value,
)
from avalan.flow.runtime import _retry_delay_seconds


class FlowPlanExecutionTestCase(IsolatedAsyncioTestCase):
    async def test_execute_flow_plan_uses_declared_entry_and_outputs(
        self,
    ) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        async def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            calls.append((node.name, dict(inputs)))
            return {"node": node.name, "inputs": dict(inputs)}

        plan = self._plan(
            entry_node="declared",
            outputs={"answer": "declared.node"},
            nodes=(
                self._node("inferred"),
                self._node("declared", output_contracts=()),
                self._node("terminal"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="inferred",
                    target="terminal",
                    kind=FlowEdgeKind.SUCCESS,
                ),
            ),
        )

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={"payload": {"private": "customer-secret"}},
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs["answer"], "declared")
        self.assertEqual([name for name, _ in calls], ["declared"])
        self.assertEqual(
            self._node_states(result),
            {
                "inferred": FlowNodeState.SKIPPED,
                "declared": FlowNodeState.SUCCEEDED,
                "terminal": FlowNodeState.SKIPPED,
            },
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_execute_flow_plan_uses_exclusive_priority_and_default(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "source":
                return {"status": "ready"}
            return node.name

        ready = self._condition(
            FlowConditionOperator.EQ,
            selector="source.value.status",
            value="ready",
        )
        plan = self._plan(
            entry_node="source",
            outputs={"answer": "low.value"},
            nodes=(
                self._node("source"),
                self._node("fallback", output_contracts=()),
                self._node("high"),
                self._node("low"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="source",
                    target="fallback",
                    kind=FlowEdgeKind.SUCCESS,
                    default=True,
                ),
                FlowEdgePlan(
                    index=1,
                    source="source",
                    target="high",
                    kind=FlowEdgeKind.SUCCESS,
                    condition=ready,
                    priority=10,
                ),
                FlowEdgePlan(
                    index=2,
                    source="source",
                    target="low",
                    kind=FlowEdgeKind.SUCCESS,
                    condition=ready,
                    priority=1,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "low"})
        self.assertEqual(calls, ["source", "low"])
        self.assertEqual(
            self._edge_states(result),
            {
                0: FlowEdgeState.SUPPRESSED,
                1: FlowEdgeState.SUPPRESSED,
                2: FlowEdgeState.TAKEN,
            },
        )

    async def test_execute_flow_plan_uses_all_matching_routes(self) -> None:
        calls: list[str] = []

        async def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "source":
                return {"status": "ready"}
            return {"value": node.name}

        ready = self._condition(
            FlowConditionOperator.EQ,
            selector="source.value.status",
            value="ready",
        )
        plan = self._plan(
            entry_node="source",
            outputs={
                "left": "left.value",
                "right": "right.value",
            },
            nodes=(
                self._node("source"),
                self._node("left"),
                self._node("right"),
                self._node("fallback"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="source",
                    target="left",
                    kind=FlowEdgeKind.SUCCESS,
                    condition=ready,
                    priority=2,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
                FlowEdgePlan(
                    index=1,
                    source="source",
                    target="right",
                    kind=FlowEdgeKind.SUCCESS,
                    condition=ready,
                    priority=1,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
                FlowEdgePlan(
                    index=2,
                    source="source",
                    target="fallback",
                    kind=FlowEdgeKind.SUCCESS,
                    default=True,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"left": "left", "right": "right"})
        self.assertEqual(calls, ["source", "right", "left"])
        self.assertEqual(
            self._edge_states(result),
            {
                0: FlowEdgeState.TAKEN,
                1: FlowEdgeState.TAKEN,
                2: FlowEdgeState.SUPPRESSED,
            },
        )

    async def test_execute_flow_plan_takes_default_when_no_route_matches(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "source":
                return {"status": "blocked"}
            return node.name

        plan = self._plan(
            entry_node="source",
            outputs={"answer": "fallback.result"},
            nodes=(
                self._node("source"),
                self._node("blocked"),
                self._node("fallback", output_contracts=()),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="source",
                    target="blocked",
                    kind=FlowEdgeKind.SUCCESS,
                    condition=self._condition(
                        FlowConditionOperator.EQ,
                        selector="source.value.status",
                        value="ready",
                    ),
                ),
                FlowEdgePlan(
                    index=1,
                    source="source",
                    target="fallback",
                    kind=FlowEdgeKind.SUCCESS,
                    default=True,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "fallback"})
        self.assertEqual(calls, ["source", "fallback"])
        self.assertEqual(
            self._edge_states(result),
            {
                0: FlowEdgeState.SUPPRESSED,
                1: FlowEdgeState.TAKEN,
            },
        )

    async def test_execute_flow_plan_routes_failures_and_finally(
        self,
    ) -> None:
        cases = (
            (
                "error",
                RuntimeError("private provider body"),
                FlowEdgeKind.ERROR,
                "handled",
                FlowNodeState.FAILED,
                "flow.execution.node_failed",
            ),
            (
                "timeout",
                TimeoutError("private timeout body"),
                FlowEdgeKind.TIMEOUT,
                "timed",
                FlowNodeState.FAILED,
                "flow.execution.node_timeout",
            ),
            (
                "cancel",
                CancelledError("private cancellation body"),
                FlowEdgeKind.CANCELLATION,
                "cancelled",
                FlowNodeState.CANCELLED,
                "flow.execution.node_cancelled",
            ),
        )

        for (
            name,
            error,
            edge_kind,
            target,
            node_state,
            code,
        ) in cases:
            with self.subTest(name=name):
                calls: list[str] = []

                def runner(
                    node: FlowNodePlan,
                    _: Mapping[str, object],
                ) -> object:
                    calls.append(node.name)
                    if node.name == "start":
                        raise error
                    return node.name

                plan = self._plan(
                    entry_node="start",
                    outputs={"answer": f"{target}.value"},
                    nodes=(
                        self._node("start"),
                        self._node(target),
                        self._node("cleanup"),
                    ),
                    edges=(
                        FlowEdgePlan(
                            index=0,
                            source="start",
                            target=target,
                            kind=edge_kind,
                        ),
                        FlowEdgePlan(
                            index=1,
                            source="start",
                            target="cleanup",
                            kind=FlowEdgeKind.FINALLY,
                        ),
                    ),
                )

                result = await execute_flow_plan(plan, runner)

                self.assertFalse(result.ok)
                self.assertEqual(result.outputs, {"answer": target})
                self.assertEqual(calls, ["start", target, "cleanup"])
                self.assertEqual(
                    self._node_states(result)["start"], node_state
                )
                self.assertEqual(
                    self._edge_states(result),
                    {0: FlowEdgeState.TAKEN, 1: FlowEdgeState.TAKEN},
                )
                self.assertIn(
                    code,
                    [
                        diagnostic["code"]
                        for diagnostic in result.public_diagnostics
                    ],
                )
                self.assertNotIn("private", str(result.public_diagnostics))

    async def test_execute_flow_plan_reports_missing_failure_route(
        self,
    ) -> None:
        def runner(_: FlowNodePlan, __: Mapping[str, object]) -> object:
            raise ValueError("private model text")

        plan = self._plan(
            entry_node="start",
            outputs={"answer": "start.value"},
            nodes=(self._node("start"),),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {})
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            [
                "flow.execution.node_failed",
                "flow.execution.missing_failure_route",
                "flow.execution.missing_output",
            ],
        )
        self.assertNotIn("private model text", str(result.public_diagnostics))

    async def test_execute_flow_plan_routes_mapping_failure(self) -> None:
        calls: list[str] = []

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            return node.name

        plan = self._plan(
            entry_node="start",
            outputs={"answer": "handled.value"},
            nodes=(
                self._node(
                    "start",
                    mappings=(
                        FlowMappingPlan(
                            target="value",
                            kind=FlowMappingKind.SELECT,
                            source=parse_flow_selector("input.missing"),
                        ),
                    ),
                ),
                self._node("handled"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="handled",
                    kind=FlowEdgeKind.ERROR,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "handled"})
        self.assertEqual(calls, ["handled"])
        self.assertEqual(
            result.public_diagnostics[0]["code"],
            "flow.execution.missing_selector_value",
        )
        self.assertEqual(
            self._node_states(result)["start"], FlowNodeState.FAILED
        )
        self.assertEqual(self._edge_states(result)[0], FlowEdgeState.TAKEN)

    async def test_execute_flow_plan_reports_condition_failure(
        self,
    ) -> None:
        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            if node.name == "source":
                return {"status": "ready"}
            return node.name

        plan = self._plan(
            entry_node="source",
            outputs={"answer": "source.value.status"},
            nodes=(self._node("source"), self._node("target")),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="source",
                    target="target",
                    kind=FlowEdgeKind.SUCCESS,
                    condition=self._condition(
                        FlowConditionOperator.EQ,
                        selector="source.value.missing",
                        value="ready",
                    ),
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "ready"})
        self.assertEqual(self._edge_states(result)[0], FlowEdgeState.FAILED)
        self.assertEqual(
            result.public_diagnostics[0]["code"],
            "flow.condition_missing_value",
        )

    async def test_execute_flow_plan_retries_transient_failure(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            if node.name == "start" and calls.count("start") == 1:
                raise FlowNodeExecutionError(
                    code="flow.execution.transient_node_error",
                    message="Flow node had a transient failure.",
                    hint="Retry the node.",
                    failure_category="transient",
                )
            return node.name

        plan = self._plan(
            entry_node="start",
            outputs={"answer": "finish.value"},
            nodes=(
                self._node(
                    "start",
                    retry=FlowRetryPlan(
                        max_attempts=2,
                        backoff=FlowRetryBackoffStrategy.CONSTANT,
                        initial_delay_seconds=0.001,
                        retryable_categories=("transient",),
                    ),
                ),
                self._node("finish"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="finish",
                    kind=FlowEdgeKind.SUCCESS,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "finish"})
        self.assertEqual(calls, ["start", "start", "finish"])
        self.assertEqual(
            self._node_attempts(result),
            {"start": 2, "finish": 1},
        )
        self.assertEqual(result.public_diagnostics, ())

    async def test_execute_flow_plan_retry_exhaustion_uses_fallback(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            if node.name == "start":
                raise FlowNodeExecutionError(
                    code="flow.execution.provider_unavailable",
                    message="Flow node provider is unavailable.",
                    hint="Use the declared fallback route.",
                    failure_category="transient",
                )
            return node.name

        plan = self._plan(
            entry_node="start",
            outputs={"answer": "fallback.value"},
            nodes=(
                self._node(
                    "start",
                    retry=FlowRetryPlan(
                        max_attempts=2,
                        retryable_categories=("transient",),
                        exhausted_route="fallback",
                    ),
                ),
                self._node("generic"),
                self._node("fallback"),
                self._node("cleanup"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="generic",
                    kind=FlowEdgeKind.ERROR,
                ),
                FlowEdgePlan(
                    index=1,
                    source="start",
                    target="fallback",
                    kind=FlowEdgeKind.ERROR,
                ),
                FlowEdgePlan(
                    index=2,
                    source="start",
                    target="cleanup",
                    kind=FlowEdgeKind.FINALLY,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "fallback"})
        self.assertEqual(calls, ["start", "start", "fallback", "cleanup"])
        self.assertEqual(self._node_attempts(result)["start"], 2)
        self.assertEqual(
            self._edge_states(result),
            {
                0: FlowEdgeState.SUPPRESSED,
                1: FlowEdgeState.TAKEN,
                2: FlowEdgeState.TAKEN,
            },
        )
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.provider_unavailable"],
        )
        self.assertNotIn("private", str(result.public_diagnostics))

    async def test_execute_flow_plan_does_not_retry_validation_by_default(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            if node.name == "start":
                raise FlowNodeExecutionError(
                    code="flow.execution.validation_failed",
                    message="Flow node validation failed.",
                    hint="Route to the validation fallback.",
                    failure_category="validation",
                )
            return node.name

        plan = self._plan(
            entry_node="start",
            outputs={"answer": "handled.value"},
            nodes=(
                self._node(
                    "start",
                    retry=FlowRetryPlan(max_attempts=3),
                ),
                self._node("handled"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="handled",
                    kind=FlowEdgeKind.ERROR,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "handled"})
        self.assertEqual(calls, ["start", "handled"])
        self.assertEqual(self._node_attempts(result)["start"], 1)
        self.assertEqual(
            result.public_diagnostics[0]["code"],
            "flow.execution.validation_failed",
        )

    async def test_execute_flow_plan_does_not_retry_cancellation(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            if node.name == "start":
                raise CancelledError("private cancellation details")
            return node.name

        plan = self._plan(
            entry_node="start",
            outputs={"answer": "cancelled.value"},
            nodes=(
                self._node(
                    "start",
                    retry=FlowRetryPlan(max_attempts=3),
                ),
                self._node("cancelled"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="cancelled",
                    kind=FlowEdgeKind.CANCELLATION,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "cancelled"})
        self.assertEqual(calls, ["start", "cancelled"])
        self.assertEqual(self._node_attempts(result)["start"], 1)
        self.assertEqual(
            result.public_diagnostics[0]["code"],
            "flow.execution.node_cancelled",
        )
        self.assertNotIn("private cancellation details", str(result))

    async def test_execute_flow_plan_respects_non_retryable_category(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            if node.name == "start":
                raise FlowNodeExecutionError(
                    code="flow.execution.transient_blocked",
                    message="Flow node failure is not retryable.",
                    hint="Route to the error handler.",
                    failure_category="transient",
                )
            return node.name

        plan = self._plan(
            entry_node="start",
            outputs={"answer": "handled.value"},
            nodes=(
                self._node(
                    "start",
                    retry=FlowRetryPlan(
                        max_attempts=3,
                        non_retryable_categories=("transient",),
                    ),
                ),
                self._node("handled"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="handled",
                    kind=FlowEdgeKind.ERROR,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "handled"})
        self.assertEqual(calls, ["start", "handled"])
        self.assertEqual(self._node_attempts(result)["start"], 1)
        self.assertEqual(
            result.public_diagnostics[0]["code"],
            "flow.execution.transient_blocked",
        )

    async def test_execute_flow_plan_retries_per_attempt_timeout(
        self,
    ) -> None:
        attempts = 0

        async def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            nonlocal attempts
            if node.name == "start":
                attempts += 1
                if attempts == 1:
                    await sleep(0.02)
                return "ready"
            return node.name

        plan = self._plan(
            entry_node="start",
            outputs={"answer": "start.value"},
            nodes=(
                self._node(
                    "start",
                    retry=FlowRetryPlan(
                        max_attempts=2,
                        retryable_categories=("timeout",),
                    ),
                    timeout=FlowTimeoutPlan(per_attempt_seconds=0.001),
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "ready"})
        self.assertEqual(self._node_attempts(result)["start"], 2)
        self.assertEqual(result.public_diagnostics, ())

    async def test_execute_flow_plan_exits_loop_with_safe_output(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "repair":
                count = len(calls)
                return {
                    "done": count == 3,
                    "more": count < 3,
                    "safe": {"attempts": count},
                    "private": "customer-secret",
                }
            return {"value": node.name}

        result = await execute_flow_plan(self._loop_plan(), runner)

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(calls, ["repair", "repair", "repair", "finished"])
        self.assertEqual(result.outputs, {"answer": {"attempts": 3}})
        self.assertEqual(
            result.node_outputs["repair"]["result"],
            {"attempts": 3},
        )
        self.assertEqual(self._node_attempts(result)["repair"], 3)
        self.assertEqual(
            self._edge_states(result),
            {0: FlowEdgeState.TAKEN, 1: FlowEdgeState.SUPPRESSED},
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_execute_flow_plan_routes_loop_iteration_limit(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "repair":
                return {"done": False, "more": True, "safe": {"ok": False}}
            return {"value": node.name}

        result = await execute_flow_plan(
            self._loop_plan(
                max_iterations=2,
                output_selector="manual.value",
            ),
            runner,
        )

        self.assertFalse(result.ok)
        self.assertEqual(calls, ["repair", "repair", "manual"])
        self.assertEqual(result.outputs, {"answer": "manual"})
        self.assertEqual(
            self._node_states(result)["repair"],
            FlowNodeState.FAILED,
        )
        self.assertEqual(self._node_attempts(result)["repair"], 2)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.execution.loop_limit_reached"],
        )
        self.assertEqual(
            self._edge_states(result),
            {0: FlowEdgeState.SUPPRESSED, 1: FlowEdgeState.TAKEN},
        )

    async def test_execute_flow_plan_routes_loop_elapsed_limit(
        self,
    ) -> None:
        calls: list[str] = []

        async def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "repair":
                await sleep(0.01)
                return {"done": False, "more": True, "safe": {"ok": False}}
            return {"value": node.name}

        result = await execute_flow_plan(
            self._loop_plan(
                max_iterations=None,
                max_elapsed_seconds=0.001,
                output_selector="manual.value",
            ),
            runner,
        )

        self.assertFalse(result.ok)
        self.assertEqual(calls, ["repair", "manual"])
        self.assertEqual(result.outputs, {"answer": "manual"})
        self.assertEqual(self._node_attempts(result)["repair"], 1)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.execution.loop_limit_reached"],
        )

    async def test_execute_flow_plan_fails_closed_when_loop_conditions_miss(
        self,
    ) -> None:
        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            if node.name == "repair":
                return {
                    "done": False,
                    "more": False,
                    "safe": {"private": "customer-secret"},
                }
            return {"value": node.name}

        result = await execute_flow_plan(self._loop_plan(), runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {})
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.execution.loop_condition_unmatched",
                "flow.execution.missing_failure_route",
                "flow.execution.missing_output",
            ],
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_execute_flow_plan_routes_loop_node_failure(
        self,
    ) -> None:
        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            if node.name == "repair":
                raise FlowNodeExecutionError(
                    code="flow.execution.validation_failed",
                    message="Loop node failed.",
                    hint="Inspect the repair node.",
                    failure_category="validation",
                )
            return {"value": node.name}

        result = await execute_flow_plan(self._loop_plan(), runner)

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.execution.validation_failed",
                "flow.execution.missing_failure_route",
                "flow.execution.missing_output",
            ],
        )
        self.assertEqual(self._node_attempts(result)["repair"], 1)

    async def test_execute_flow_plan_reports_loop_output_selector_failure(
        self,
    ) -> None:
        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            if node.name == "repair":
                return {"done": True, "more": False, "safe": "ok"}
            return {"value": node.name}

        result = await execute_flow_plan(
            self._loop_plan(loop_output_selector="repair.result.missing"),
            runner,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.execution.missing_selector_value",
                "flow.execution.missing_failure_route",
                "flow.execution.missing_output",
            ],
        )

    async def test_execute_flow_plan_reports_missing_loop_limit_route(
        self,
    ) -> None:
        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            if node.name == "repair":
                return {"done": False, "more": True, "safe": "ok"}
            return {"value": node.name}

        result = await execute_flow_plan(
            self._loop_plan(
                include_limit_edge=False,
                max_iterations=1,
            ),
            runner,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.execution.loop_limit_reached",
                "flow.execution.missing_failure_route",
                "flow.execution.missing_output",
            ],
        )

    async def test_execute_flow_plan_checks_cancellation_between_nodes(
        self,
    ) -> None:
        checks = 0

        async def check_cancelled() -> None:
            nonlocal checks
            checks += 1

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            return node.name

        plan = self._plan(
            entry_node="start",
            outputs={"answer": "finish.value"},
            nodes=(self._node("start"), self._node("finish")),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="finish",
                    kind=FlowEdgeKind.SUCCESS,
                ),
            ),
        )

        result = await execute_flow_plan(
            plan,
            runner,
            cancellation_checker=check_cancelled,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "finish"})
        self.assertGreaterEqual(checks, 2)

    async def test_execute_flow_plan_runs_join_policies(self) -> None:
        cases = (
            FlowJoinPlan(type=FlowJoinPolicyType.ALL_SUCCESS),
            FlowJoinPlan(type=FlowJoinPolicyType.ALL_DONE),
            FlowJoinPlan(type=FlowJoinPolicyType.ANY_SUCCESS),
            FlowJoinPlan(type=FlowJoinPolicyType.QUORUM, quorum=2),
            FlowJoinPlan(type=FlowJoinPolicyType.FIRST_SUCCESS),
            FlowJoinPlan(type=FlowJoinPolicyType.FAIL_FAST),
            FlowJoinPlan(type=FlowJoinPolicyType.COLLECT),
        )

        for join in cases:
            with self.subTest(join=join.type.value):
                calls: list[tuple[str, dict[str, object]]] = []

                def runner(
                    node: FlowNodePlan,
                    inputs: Mapping[str, object],
                ) -> object:
                    calls.append((node.name, dict(inputs)))
                    if node.name in {"left", "right"}:
                        return {
                            "side": node.name,
                            "shared": node.name,
                        }
                    if node.name == "joined":
                        value = cast(dict[str, object], inputs["value"])
                        return {
                            "value": {
                                **value,
                                "merged": inputs["merged"],
                            }
                        }
                    return {"status": "ready"}

                plan = self._join_plan(join)

                result = await execute_flow_plan(plan, runner)

                self.assertTrue(result.ok, result.public_diagnostics)
                self.assertEqual(
                    result.outputs,
                    {
                        "answer": {
                            "left": {
                                "side": "left",
                                "shared": "left",
                            },
                            "right": {
                                "side": "right",
                                "shared": "right",
                            },
                            "merged": {
                                "side": "right",
                                "shared": "right",
                            },
                        }
                    },
                )
                self.assertEqual(
                    [name for name, _ in calls],
                    ["source", "left", "right", "joined"],
                )
                self.assertEqual(
                    self._node_states(result)["joined"],
                    FlowNodeState.SUCCEEDED,
                )
                self.assertEqual(
                    self._edge_states(result),
                    {
                        0: FlowEdgeState.TAKEN,
                        1: FlowEdgeState.TAKEN,
                        2: FlowEdgeState.TAKEN,
                        3: FlowEdgeState.TAKEN,
                    },
                )

    async def test_execute_flow_plan_waits_for_all_done_join_with_error(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            if node.name == "left":
                raise ValueError("private branch body")
            if node.name == "joined":
                return "joined"
            return node.name

        plan = self._plan(
            entry_node="source",
            outputs={"answer": "joined.value"},
            nodes=(
                self._node("source"),
                self._node("left"),
                self._node("right"),
                self._node(
                    "joined",
                    join=FlowJoinPlan(type=FlowJoinPolicyType.ALL_DONE),
                ),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="source",
                    target="left",
                    kind=FlowEdgeKind.SUCCESS,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
                FlowEdgePlan(
                    index=1,
                    source="source",
                    target="right",
                    kind=FlowEdgeKind.SUCCESS,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
                FlowEdgePlan(
                    index=2,
                    source="left",
                    target="joined",
                    kind=FlowEdgeKind.ERROR,
                ),
                FlowEdgePlan(
                    index=3,
                    source="right",
                    target="joined",
                    kind=FlowEdgeKind.SUCCESS,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "joined"})
        self.assertEqual(calls, ["source", "left", "right", "joined"])
        self.assertEqual(
            self._node_states(result)["joined"],
            FlowNodeState.SUCCEEDED,
        )
        self.assertIn(
            "flow.execution.node_failed",
            str(result.public_diagnostics),
        )
        self.assertNotIn("private branch body", str(result.public_diagnostics))

    async def test_execute_flow_plan_fail_fast_join_does_not_run(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            if node.name == "left":
                raise RuntimeError("private failure payload")
            return node.name

        plan = self._plan(
            entry_node="source",
            outputs={"answer": "joined.value"},
            nodes=(
                self._node("source"),
                self._node("left"),
                self._node("right"),
                self._node(
                    "joined",
                    join=FlowJoinPlan(type=FlowJoinPolicyType.FAIL_FAST),
                ),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="source",
                    target="left",
                    kind=FlowEdgeKind.SUCCESS,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
                FlowEdgePlan(
                    index=1,
                    source="source",
                    target="right",
                    kind=FlowEdgeKind.SUCCESS,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
                FlowEdgePlan(
                    index=2,
                    source="left",
                    target="joined",
                    kind=FlowEdgeKind.ERROR,
                ),
                FlowEdgePlan(
                    index=3,
                    source="right",
                    target="joined",
                    kind=FlowEdgeKind.SUCCESS,
                ),
                FlowEdgePlan(
                    index=4,
                    source="source",
                    target="joined",
                    kind=FlowEdgeKind.SUCCESS,
                    condition=self._condition(
                        FlowConditionOperator.EQ,
                        selector="source.value.missing",
                        value="ready",
                    ),
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(calls, ["source", "left", "right"])
        self.assertEqual(
            self._node_states(result)["joined"],
            FlowNodeState.FAILED,
        )
        self.assertEqual(result.outputs, {})
        self.assertIn(
            "flow.execution.join_failed",
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
        )
        self.assertNotIn(
            "private failure payload",
            str(result.public_diagnostics),
        )

    async def test_execute_flow_plan_honors_concurrency_limit(self) -> None:
        active = 0
        max_active = 0

        async def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            nonlocal active, max_active
            if node.name in {"left", "right"}:
                active += 1
                max_active = max(max_active, active)
                await sleep(0.01)
                active -= 1
            return node.name

        plan = self._plan(
            entry_node="source",
            outputs={"answer": "joined.value"},
            nodes=(
                self._node("source"),
                self._node("left"),
                self._node("right"),
                self._node(
                    "joined",
                    join=FlowJoinPlan(type=FlowJoinPolicyType.ALL_SUCCESS),
                ),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="source",
                    target="left",
                    kind=FlowEdgeKind.SUCCESS,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
                FlowEdgePlan(
                    index=1,
                    source="source",
                    target="right",
                    kind=FlowEdgeKind.SUCCESS,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
                FlowEdgePlan(
                    index=2,
                    source="left",
                    target="joined",
                    kind=FlowEdgeKind.SUCCESS,
                ),
                FlowEdgePlan(
                    index=3,
                    source="right",
                    target="joined",
                    kind=FlowEdgeKind.SUCCESS,
                ),
            ),
        )

        sequential = await execute_flow_plan(
            plan,
            runner,
            concurrency_limit=1,
        )
        self.assertTrue(sequential.ok, sequential.public_diagnostics)
        self.assertEqual(max_active, 1)

        active = 0
        max_active = 0
        concurrent = await execute_flow_plan(
            plan,
            runner,
            concurrency_limit=2,
        )
        self.assertTrue(concurrent.ok, concurrent.public_diagnostics)
        self.assertEqual(max_active, 2)

    async def test_execute_flow_plan_validates_arguments(self) -> None:
        plan = self._plan(
            entry_node="start",
            outputs={"answer": "start.value"},
            nodes=(self._node("start"),),
        )

        def runner(_: FlowNodePlan, __: Mapping[str, object]) -> object:
            return "ok"

        with self.assertRaises(AssertionError):
            await execute_flow_plan(
                "plan",  # type: ignore[arg-type]
                runner,
            )
        with self.assertRaises(AssertionError):
            await execute_flow_plan(plan, "runner")  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await execute_flow_plan(
                plan,
                runner,
                inputs=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            await execute_flow_plan(plan, runner, concurrency_limit=0)
        with self.assertRaises(AssertionError):
            await execute_flow_plan(
                plan,
                runner,
                concurrency_limit=True,  # type: ignore[arg-type]
            )

    def test_flow_plan_execution_result_is_immutable(self) -> None:
        plan = self._plan(
            entry_node="start",
            outputs={"answer": "start.value"},
            nodes=(self._node("start"),),
        )
        diagnostic = FlowDiagnostic(
            code="flow.execution.test",
            category=FlowDiagnosticCategory.EXECUTION,
            path="flow",
            message="Flow execution test diagnostic.",
        )
        raw_outputs = {"answer": {"items": ["one"]}}
        raw_node_outputs = {"start": {"value": {"items": ["one"]}}}

        result = FlowPlanExecutionResult(
            trace=FlowExecutionTrace.from_plan(plan),
            outputs=raw_outputs,
            diagnostics=(diagnostic,),
            node_outputs=raw_node_outputs,
        )
        raw_outputs["answer"]["items"].append("two")
        raw_node_outputs["start"]["value"]["items"].append("two")

        answer = cast(Mapping[str, object], result.outputs["answer"])
        node_value = cast(
            Mapping[str, object],
            result.node_outputs["start"]["value"],
        )
        self.assertFalse(result.ok)
        self.assertEqual(answer["items"], ("one",))
        self.assertEqual(node_value["items"], ("one",))
        self.assertEqual(
            result.public_diagnostics[0]["code"],
            "flow.execution.test",
        )
        with self.assertRaises(TypeError):
            cast(dict[str, object], result.outputs)["other"] = "value"
        with self.assertRaises(FrozenInstanceError):
            result.outputs = {}  # type: ignore[misc]
        with self.assertRaises(AssertionError):
            FlowPlanExecutionResult(
                trace="trace",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowPlanExecutionResult(
                trace=FlowExecutionTrace.from_plan(plan),
                diagnostics=(object(),),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowPlanExecutionResult(
                trace=FlowExecutionTrace.from_plan(plan),
                node_outputs={"": {}},
            )

    def test_retry_delay_seconds_handles_backoff_strategies(self) -> None:
        cases = (
            (
                FlowRetryPlan(max_attempts=2),
                1,
                0,
            ),
            (
                FlowRetryPlan(
                    max_attempts=2,
                    backoff=FlowRetryBackoffStrategy.CONSTANT,
                    initial_delay_seconds=0.5,
                ),
                2,
                0.5,
            ),
            (
                FlowRetryPlan(
                    max_attempts=3,
                    backoff=FlowRetryBackoffStrategy.LINEAR,
                    initial_delay_seconds=0.5,
                    max_delay_seconds=0.75,
                ),
                2,
                0.75,
            ),
            (
                FlowRetryPlan(
                    max_attempts=4,
                    backoff=FlowRetryBackoffStrategy.EXPONENTIAL,
                    initial_delay_seconds=0.5,
                ),
                3,
                2,
            ),
        )

        self.assertEqual(_retry_delay_seconds(None, 1), 0)
        for retry, failed_attempts, expected in cases:
            with self.subTest(backoff=retry.backoff.value):
                self.assertEqual(
                    _retry_delay_seconds(retry, failed_attempts),
                    expected,
                )

    def _plan(
        self,
        *,
        entry_node: str,
        outputs: Mapping[str, str],
        nodes: tuple[FlowNodePlan, ...],
        edges: tuple[FlowEdgePlan, ...] = (),
    ) -> FlowExecutionPlan:
        return FlowExecutionPlan(
            name="runtime",
            version=None,
            revision=None,
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
            ),
            outputs=tuple(
                FlowOutputDefinition(
                    name=name,
                    type=FlowOutputType.JSON,
                )
                for name in outputs
            ),
            entry_node=entry_node,
            output_selectors={
                name: parse_flow_selector(selector)
                for name, selector in outputs.items()
            },
            nodes=nodes,
            edges=edges,
        )

    def _join_plan(self, join: FlowJoinPlan) -> FlowExecutionPlan:
        return self._plan(
            entry_node="source",
            outputs={"answer": "joined.value"},
            nodes=(
                self._node("source"),
                self._node("left"),
                self._node("right"),
                self._node(
                    "joined",
                    join=join,
                    mappings=(
                        FlowMappingPlan(
                            target="value",
                            kind=FlowMappingKind.OBJECT,
                            fields={
                                "left": parse_flow_selector("left.value"),
                                "right": parse_flow_selector("right.value"),
                            },
                        ),
                        FlowMappingPlan(
                            target="merged",
                            kind=FlowMappingKind.MERGE,
                            sources=(
                                parse_flow_selector("left.value"),
                                parse_flow_selector("right.value"),
                            ),
                        ),
                    ),
                ),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="source",
                    target="left",
                    kind=FlowEdgeKind.SUCCESS,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
                FlowEdgePlan(
                    index=1,
                    source="source",
                    target="right",
                    kind=FlowEdgeKind.SUCCESS,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
                FlowEdgePlan(
                    index=2,
                    source="left",
                    target="joined",
                    kind=FlowEdgeKind.SUCCESS,
                ),
                FlowEdgePlan(
                    index=3,
                    source="right",
                    target="joined",
                    kind=FlowEdgeKind.SUCCESS,
                ),
            ),
        )

    def _node(
        self,
        name: str,
        *,
        output_contracts: tuple[FlowNodeContract, ...] | None = None,
        mappings: tuple[FlowMappingPlan, ...] = (),
        join: FlowJoinPlan | None = None,
        retry: FlowRetryPlan | None = None,
        timeout: FlowTimeoutPlan | None = None,
        loop: FlowLoopPlan | None = None,
    ) -> FlowNodePlan:
        return FlowNodePlan(
            name=name,
            type="test",
            kind=FlowNodeKind.PASS_THROUGH,
            mappings=mappings,
            join=join,
            retry=retry,
            timeout=timeout,
            loop=loop,
            output_contracts=(
                (FlowNodeContract(name="value", type=FlowOutputType.JSON),)
                if output_contracts is None
                else output_contracts
            ),
        )

    def _condition(
        self,
        operator: FlowConditionOperator,
        *,
        selector: str,
        value: object,
    ) -> FlowConditionPlan:
        return FlowConditionPlan(
            operator=operator,
            selector=parse_flow_selector(selector),
            value=value,
        )

    def _loop_plan(
        self,
        *,
        max_iterations: int | None = 4,
        max_elapsed_seconds: int | float | None = None,
        output_selector: str = "repair.result",
        loop_output_selector: str = "repair.result.safe",
        include_limit_edge: bool = True,
    ) -> FlowExecutionPlan:
        edges = (
            FlowEdgePlan(
                index=0,
                source="repair",
                target="finished",
                kind=FlowEdgeKind.SUCCESS,
            ),
        )
        if include_limit_edge:
            edges = edges + (
                FlowEdgePlan(
                    index=1,
                    source="repair",
                    target="manual",
                    kind=FlowEdgeKind.SUCCESS,
                ),
            )
        return self._plan(
            entry_node="repair",
            outputs={"answer": output_selector},
            nodes=(
                self._node(
                    "repair",
                    output_contracts=(
                        FlowNodeContract(
                            name="result",
                            type=FlowOutputType.OBJECT,
                        ),
                    ),
                    loop=FlowLoopPlan(
                        max_iterations=max_iterations,
                        max_elapsed_seconds=max_elapsed_seconds,
                        exit_condition=self._condition(
                            FlowConditionOperator.EQ,
                            selector="repair.result.done",
                            value=True,
                        ),
                        continue_condition=self._condition(
                            FlowConditionOperator.EQ,
                            selector="repair.result.more",
                            value=True,
                        ),
                        output_selector=parse_flow_selector(
                            loop_output_selector
                        ),
                        limit_route="manual",
                    ),
                ),
                self._node("finished"),
                self._node("manual"),
            ),
            edges=edges,
        )

    def _node_states(
        self,
        result: FlowPlanExecutionResult,
    ) -> dict[str, FlowNodeState]:
        return {trace.node: trace.state for trace in result.trace.nodes}

    def _edge_states(
        self,
        result: FlowPlanExecutionResult,
    ) -> dict[int, FlowEdgeState]:
        return {trace.index: trace.state for trace in result.trace.edges}

    def _node_attempts(
        self,
        result: FlowPlanExecutionResult,
    ) -> dict[str, int]:
        return {trace.node: trace.attempts for trace in result.trace.nodes}


class FlowRuntimeEvaluationTestCase(TestCase):
    def setUp(self) -> None:
        self.document = {
            "source_kind": "local_path",
            "reference": "/private/customer.pdf",
            "mime_type": "application/pdf",
            "metadata": {"purpose": "review"},
        }
        self.artifact = {
            "source_kind": "artifact",
            "reference": "artifact-1",
            "metadata": {"store": "local"},
        }
        self.context = FlowRuntimeContext(
            inputs={
                "payload": {
                    "customer": {
                        "name": "Ada",
                        "tags": ["vip", "new"],
                    },
                    "city": "Paris",
                    "enabled": True,
                    "expected": "ready",
                    "left": {"shared": "input", "only_input": 1},
                },
                "document": self.document,
                "documents": [self.document, self.artifact],
            },
            node_outputs={
                "prepare": {
                    "result": {
                        "status": "ready",
                        "count": 3,
                        "score": 3.5,
                        "tags": ("alpha", "beta"),
                        "nullable": None,
                        "payload": {"nested": "value"},
                        "right": {"shared": "node", "only_node": 2},
                    },
                },
            },
        )

    def test_evaluate_flow_selector_resolves_nested_runtime_values(
        self,
    ) -> None:
        self.assertEqual(
            evaluate_flow_selector(
                parse_flow_selector("input.payload.customer.tags[1]"),
                self.context,
            ),
            "new",
        )
        self.assertEqual(
            evaluate_flow_selector(
                parse_flow_selector("prepare.result.payload.nested"),
                self.context,
            ),
            "value",
        )

        sentinel = object()
        missing_selectors = (
            "missing.result.value",
            "prepare.result.missing.value",
            "prepare.result.status.name",
            "prepare.result.status[0]",
            "prepare.result.tags[9]",
        )
        for selector in missing_selectors:
            with self.subTest(selector=selector):
                self.assertIs(
                    resolve_flow_selector_value(
                        parse_flow_selector(selector),
                        inputs=self.context.inputs,
                        node_outputs=self.context.node_outputs,
                        missing=sentinel,
                    ),
                    sentinel,
                )

        with self.assertRaises(FlowRuntimeEvaluationError) as raised:
            evaluate_flow_selector(
                parse_flow_selector("input.missing"),
                self.context,
            )

        self.assertEqual(
            raised.exception.code,
            "flow.execution.missing_selector_value",
        )
        self.assertNotIn("customer.pdf", str(raised.exception))

    def test_runtime_context_freezes_values_and_validates_shape(self) -> None:
        raw_payload = {"items": ["one"]}
        context = FlowRuntimeContext(
            inputs={"payload": raw_payload},
            node_outputs={"node": {"result": raw_payload}},
        )
        raw_payload["items"].append("two")
        payload = cast(Mapping[str, object], context.inputs["payload"])
        node_output = cast(
            Mapping[str, object],
            context.node_outputs["node"]["result"],
        )

        self.assertEqual(payload["items"], ("one",))
        self.assertEqual(node_output["items"], ("one",))
        with self.assertRaises(TypeError):
            cast(dict[str, object], context.inputs)["other"] = "value"
        with self.assertRaises(FrozenInstanceError):
            context.inputs = {}  # type: ignore[misc]
        with self.assertRaises(AssertionError):
            FlowRuntimeContext(inputs={1: "bad"})  # type: ignore[dict-item]
        with self.assertRaises(AssertionError):
            FlowRuntimeContext(node_outputs={"": {}})
        with self.assertRaises(AssertionError):
            FlowRuntimeEvaluationError("")
        with self.assertRaises(AssertionError):
            evaluate_flow_selector(
                "input.payload",  # type: ignore[arg-type]
                context,
            )
        with self.assertRaises(AssertionError):
            evaluate_flow_selector(
                parse_flow_selector("input.payload"),
                "context",  # type: ignore[arg-type]
            )

    def test_evaluate_flow_node_mappings_supports_all_mapping_kinds(
        self,
    ) -> None:
        node = FlowNodePlan(
            name="mapper",
            type="select",
            kind=FlowNodeKind.SELECT,
            mappings=(
                FlowMappingPlan(
                    target="selected",
                    kind=FlowMappingKind.SELECT,
                    source=parse_flow_selector("input.payload.customer"),
                ),
                FlowMappingPlan(
                    target="renamed",
                    kind=FlowMappingKind.RENAME,
                    source=parse_flow_selector("prepare.result.status"),
                ),
                FlowMappingPlan(
                    target="constructed",
                    kind=FlowMappingKind.OBJECT,
                    fields={
                        "name": parse_flow_selector(
                            "input.payload.customer.name"
                        ),
                        "status": parse_flow_selector("prepare.result.status"),
                    },
                ),
                FlowMappingPlan(
                    target="items",
                    kind=FlowMappingKind.ARRAY,
                    items=(
                        parse_flow_selector("input.payload.city"),
                        parse_flow_selector("prepare.result.count"),
                    ),
                ),
                FlowMappingPlan(
                    target="merged",
                    kind=FlowMappingKind.MERGE,
                    sources=(
                        parse_flow_selector("input.payload.left"),
                        parse_flow_selector("prepare.result.right"),
                    ),
                ),
                FlowMappingPlan(
                    target="document",
                    kind=FlowMappingKind.FILE,
                    source=parse_flow_selector("input.document"),
                ),
                FlowMappingPlan(
                    target="documents",
                    kind=FlowMappingKind.FILE_ARRAY,
                    source=parse_flow_selector("input.documents"),
                ),
            ),
        )

        result = evaluate_flow_node_mappings(node, self.context)
        selected = cast(Mapping[str, object], result["selected"])
        constructed = cast(Mapping[str, object], result["constructed"])
        merged = cast(Mapping[str, object], result["merged"])
        document = cast(Mapping[str, object], result["document"])
        documents = cast(tuple[object, ...], result["documents"])

        self.assertEqual(selected["name"], "Ada")
        self.assertEqual(result["renamed"], "ready")
        self.assertEqual(constructed, {"name": "Ada", "status": "ready"})
        self.assertEqual(result["items"], ("Paris", 3))
        self.assertEqual(
            merged,
            {"shared": "node", "only_input": 1, "only_node": 2},
        )
        self.assertEqual(document["reference"], "/private/customer.pdf")
        self.assertNotIn("path", document)
        self.assertNotIn("bytes", document)
        self.assertEqual(len(documents), 2)
        self.assertEqual(
            cast(Mapping[str, object], documents[1])["reference"],
            "artifact-1",
        )

    def test_evaluate_flow_mappings_rejects_invalid_runtime_values(
        self,
    ) -> None:
        cases = (
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                    ),
                ),
                "flow.execution.missing_mapping_source",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("input.missing"),
                    ),
                ),
                "flow.execution.missing_selector_value",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.OBJECT,
                    ),
                ),
                "flow.execution.empty_mapping",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.ARRAY,
                    ),
                ),
                "flow.execution.empty_mapping",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.MERGE,
                    ),
                ),
                "flow.execution.empty_mapping",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.MERGE,
                        sources=(
                            parse_flow_selector("prepare.result.status"),
                        ),
                    ),
                ),
                "flow.execution.merge_requires_object",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("input.payload.city"),
                    ),
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("prepare.result.status"),
                    ),
                ),
                "flow.execution.duplicate_mapping_target",
            ),
        )

        for mappings, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(FlowRuntimeEvaluationError) as raised:
                    evaluate_flow_mappings(mappings, self.context)

                self.assertEqual(raised.exception.code, code)
                self.assertNotIn("Paris", str(raised.exception))

        mapping = FlowMappingPlan(
            target="value",
            kind=FlowMappingKind.SELECT,
            source=parse_flow_selector("input.payload.city"),
        )
        object.__setattr__(mapping, "kind", "unsupported")
        with self.assertRaises(FlowRuntimeEvaluationError) as raised:
            evaluate_flow_mappings((mapping,), self.context)
        self.assertEqual(
            raised.exception.code,
            "flow.execution.unsupported_mapping_kind",
        )
        with self.assertRaises(AssertionError):
            evaluate_flow_mappings(
                [mapping],  # type: ignore[arg-type]
                self.context,
            )
        with self.assertRaises(AssertionError):
            evaluate_flow_mappings((object(),), self.context)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            evaluate_flow_mappings(
                (mapping,),
                "context",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            evaluate_flow_node_mappings(
                "node",  # type: ignore[arg-type]
                self.context,
            )

    def test_evaluate_flow_condition_plan_operators(self) -> None:
        cases = (
            (
                FlowConditionOperator.EQ,
                self._condition(
                    FlowConditionOperator.EQ,
                    selector="prepare.result.status",
                    value="ready",
                ),
                self._condition(
                    FlowConditionOperator.EQ,
                    selector="prepare.result.status",
                    value="done",
                ),
            ),
            (
                FlowConditionOperator.NE,
                self._condition(
                    FlowConditionOperator.NE,
                    selector="prepare.result.status",
                    value="done",
                ),
                self._condition(
                    FlowConditionOperator.NE,
                    selector="prepare.result.status",
                    value="ready",
                ),
            ),
            (
                FlowConditionOperator.EXISTS,
                self._condition(
                    FlowConditionOperator.EXISTS,
                    selector="prepare.result.status",
                ),
                self._condition(
                    FlowConditionOperator.EXISTS,
                    selector="prepare.result.missing",
                ),
            ),
            (
                FlowConditionOperator.NOT_EXISTS,
                self._condition(
                    FlowConditionOperator.NOT_EXISTS,
                    selector="prepare.result.missing",
                ),
                self._condition(
                    FlowConditionOperator.NOT_EXISTS,
                    selector="prepare.result.status",
                ),
            ),
            (
                FlowConditionOperator.IS_TYPE,
                self._condition(
                    FlowConditionOperator.IS_TYPE,
                    selector="prepare.result.count",
                    value_type=FlowConditionValueType.INTEGER,
                ),
                self._condition(
                    FlowConditionOperator.IS_TYPE,
                    selector="prepare.result.status",
                    value_type=FlowConditionValueType.INTEGER,
                ),
            ),
            (
                FlowConditionOperator.IN,
                self._condition(
                    FlowConditionOperator.IN,
                    selector="prepare.result.status",
                    values=("ready", "done"),
                ),
                self._condition(
                    FlowConditionOperator.IN,
                    selector="prepare.result.status",
                    values=("done", "failed"),
                ),
            ),
            (
                FlowConditionOperator.NOT_IN,
                self._condition(
                    FlowConditionOperator.NOT_IN,
                    selector="prepare.result.status",
                    values=("done", "failed"),
                ),
                self._condition(
                    FlowConditionOperator.NOT_IN,
                    selector="prepare.result.status",
                    values=("ready", "done"),
                ),
            ),
            (
                FlowConditionOperator.GT,
                self._condition(
                    FlowConditionOperator.GT,
                    selector="prepare.result.count",
                    value=2,
                ),
                self._condition(
                    FlowConditionOperator.GT,
                    selector="prepare.result.count",
                    value=4,
                ),
            ),
            (
                FlowConditionOperator.GTE,
                self._condition(
                    FlowConditionOperator.GTE,
                    selector="prepare.result.count",
                    value=3,
                ),
                self._condition(
                    FlowConditionOperator.GTE,
                    selector="prepare.result.count",
                    value=4,
                ),
            ),
            (
                FlowConditionOperator.LT,
                self._condition(
                    FlowConditionOperator.LT,
                    selector="prepare.result.count",
                    value=4,
                ),
                self._condition(
                    FlowConditionOperator.LT,
                    selector="prepare.result.count",
                    value=2,
                ),
            ),
            (
                FlowConditionOperator.LTE,
                self._condition(
                    FlowConditionOperator.LTE,
                    selector="prepare.result.count",
                    value=3,
                ),
                self._condition(
                    FlowConditionOperator.LTE,
                    selector="prepare.result.count",
                    value=2,
                ),
            ),
            (
                FlowConditionOperator.STARTS_WITH,
                self._condition(
                    FlowConditionOperator.STARTS_WITH,
                    selector="input.payload.city",
                    value="Par",
                ),
                self._condition(
                    FlowConditionOperator.STARTS_WITH,
                    selector="input.payload.city",
                    value="Lon",
                ),
            ),
            (
                FlowConditionOperator.ENDS_WITH,
                self._condition(
                    FlowConditionOperator.ENDS_WITH,
                    selector="input.payload.city",
                    value="ris",
                ),
                self._condition(
                    FlowConditionOperator.ENDS_WITH,
                    selector="input.payload.city",
                    value="don",
                ),
            ),
            (
                FlowConditionOperator.CONTAINS,
                self._condition(
                    FlowConditionOperator.CONTAINS,
                    selector="input.payload.city",
                    value="ari",
                ),
                self._condition(
                    FlowConditionOperator.CONTAINS,
                    selector="input.payload.city",
                    value="zzz",
                ),
            ),
            (
                FlowConditionOperator.IS_NULL,
                self._condition(
                    FlowConditionOperator.IS_NULL,
                    selector="prepare.result.nullable",
                ),
                self._condition(
                    FlowConditionOperator.IS_NULL,
                    selector="prepare.result.status",
                ),
            ),
            (
                FlowConditionOperator.NOT_NULL,
                self._condition(
                    FlowConditionOperator.NOT_NULL,
                    selector="prepare.result.status",
                ),
                self._condition(
                    FlowConditionOperator.NOT_NULL,
                    selector="prepare.result.nullable",
                ),
            ),
            (
                FlowConditionOperator.ALL,
                self._condition(
                    FlowConditionOperator.ALL,
                    conditions=(
                        self._condition(
                            FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="ready",
                        ),
                        self._condition(
                            FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=3,
                        ),
                    ),
                ),
                self._condition(
                    FlowConditionOperator.ALL,
                    conditions=(
                        self._condition(
                            FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="ready",
                        ),
                        self._condition(
                            FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=4,
                        ),
                    ),
                ),
            ),
            (
                FlowConditionOperator.ANY,
                self._condition(
                    FlowConditionOperator.ANY,
                    conditions=(
                        self._condition(
                            FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="done",
                        ),
                        self._condition(
                            FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=3,
                        ),
                    ),
                ),
                self._condition(
                    FlowConditionOperator.ANY,
                    conditions=(
                        self._condition(
                            FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="done",
                        ),
                        self._condition(
                            FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=4,
                        ),
                    ),
                ),
            ),
            (
                FlowConditionOperator.NOT,
                self._condition(
                    FlowConditionOperator.NOT,
                    condition=self._condition(
                        FlowConditionOperator.EQ,
                        selector="prepare.result.status",
                        value="done",
                    ),
                ),
                self._condition(
                    FlowConditionOperator.NOT,
                    condition=self._condition(
                        FlowConditionOperator.EQ,
                        selector="prepare.result.status",
                        value="ready",
                    ),
                ),
            ),
        )

        for operator, true_condition, false_condition in cases:
            with self.subTest(operator=operator.value):
                self.assertTrue(
                    evaluate_flow_condition_plan(true_condition, self.context)
                )
                self.assertFalse(
                    evaluate_flow_condition_plan(false_condition, self.context)
                )

    def test_evaluate_flow_condition_plan_supports_values_and_types(
        self,
    ) -> None:
        value_selector = self._condition(
            FlowConditionOperator.EQ,
            selector="prepare.result.status",
            value_selector="input.payload.expected",
        )
        literal_membership = self._condition(
            FlowConditionOperator.IN,
            selector="prepare.result.status",
            value=("ready", "done"),
        )
        scalar_membership = self._condition(
            FlowConditionOperator.IN,
            selector="prepare.result.status",
            value="ready",
        )
        type_cases = (
            ("prepare.result.status", FlowConditionValueType.STRING),
            ("prepare.result.count", FlowConditionValueType.INTEGER),
            ("prepare.result.score", FlowConditionValueType.NUMBER),
            ("input.payload.enabled", FlowConditionValueType.BOOLEAN),
            ("prepare.result.payload", FlowConditionValueType.OBJECT),
            ("prepare.result.tags", FlowConditionValueType.ARRAY),
            ("prepare.result.nullable", FlowConditionValueType.NULL),
        )

        self.assertTrue(
            evaluate_flow_condition_plan(value_selector, self.context)
        )
        self.assertTrue(
            evaluate_flow_condition_plan(literal_membership, self.context)
        )
        self.assertFalse(
            evaluate_flow_condition_plan(scalar_membership, self.context)
        )
        self.assertTrue(
            evaluate_flow_condition_plan(
                self._condition(
                    FlowConditionOperator.EQ,
                    selector="prepare.result.tags[1]",
                    value="beta",
                ),
                self.context,
            )
        )
        for selector, value_type in type_cases:
            with self.subTest(value_type=value_type.value):
                self.assertTrue(
                    evaluate_flow_condition_plan(
                        self._condition(
                            FlowConditionOperator.IS_TYPE,
                            selector=selector,
                            value_type=value_type,
                        ),
                        self.context,
                    )
                )

    def test_evaluate_flow_condition_plan_reports_missing_values(
        self,
    ) -> None:
        cases = (
            (
                FlowConditionPlan(operator=FlowConditionOperator.EQ),
                "flow.condition_missing_selector",
            ),
            (
                self._condition(
                    FlowConditionOperator.EQ,
                    selector="prepare.result.missing",
                    value="ready",
                ),
                "flow.condition_missing_value",
            ),
            (
                self._condition(
                    FlowConditionOperator.EQ,
                    selector="prepare.result.status",
                    value_selector="prepare.result.missing",
                ),
                "flow.condition_missing_value",
            ),
            (
                FlowConditionPlan(operator=FlowConditionOperator.NOT),
                "flow.condition_missing_child",
            ),
            (
                self._condition(
                    FlowConditionOperator.IS_TYPE,
                    selector="prepare.result.status",
                ),
                "flow.condition_missing_value_type",
            ),
        )

        for condition, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(FlowRuntimeEvaluationError) as raised:
                    evaluate_flow_condition_plan(condition, self.context)

                self.assertEqual(raised.exception.code, code)
                self.assertNotIn("ready", str(raised.exception))

    def test_evaluate_flow_condition_plan_handles_defensive_paths(
        self,
    ) -> None:
        numeric = self._condition(
            FlowConditionOperator.GT,
            selector="prepare.result.status",
            value=3,
        )
        string = self._condition(
            FlowConditionOperator.CONTAINS,
            selector="prepare.result.count",
            value="3",
        )
        unknown = self._condition(
            FlowConditionOperator.EQ,
            selector="prepare.result.status",
            value="ready",
        )
        object.__setattr__(unknown, "operator", "unknown")

        self.assertFalse(evaluate_flow_condition_plan(numeric, self.context))
        self.assertFalse(evaluate_flow_condition_plan(string, self.context))
        with self.assertRaises(FlowRuntimeEvaluationError) as raised:
            evaluate_flow_condition_plan(unknown, self.context)
        self.assertEqual(
            raised.exception.code,
            "flow.condition_unknown_operator",
        )
        with self.assertRaises(AssertionError):
            evaluate_flow_condition_plan(
                "condition",  # type: ignore[arg-type]
                self.context,
            )
        with self.assertRaises(AssertionError):
            evaluate_flow_condition_plan(
                self._condition(
                    FlowConditionOperator.EXISTS,
                    selector="prepare.result.status",
                ),
                "context",  # type: ignore[arg-type]
            )

    def _condition(
        self,
        operator: FlowConditionOperator,
        *,
        selector: str | None = None,
        value: object | None = None,
        value_selector: str | None = None,
        values: tuple[object, ...] = (),
        value_type: FlowConditionValueType | None = None,
        conditions: tuple[FlowConditionPlan, ...] = (),
        condition: FlowConditionPlan | None = None,
    ) -> FlowConditionPlan:
        return FlowConditionPlan(
            operator=operator,
            selector=(
                parse_flow_selector(selector) if selector is not None else None
            ),
            value=value,
            value_selector=(
                parse_flow_selector(value_selector)
                if value_selector is not None
                else None
            ),
            values=values,
            value_type=value_type,
            conditions=conditions,
            condition=condition,
        )


if __name__ == "__main__":
    main()
