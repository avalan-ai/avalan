from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from async_helpers import run_async

from avalan.flow import (
    FlowDefinition,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeDefinition,
    FlowEdgeState,
    FlowEdgeTrace,
    FlowEntryBehavior,
    FlowExecutionTrace,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeDefinition,
    FlowNodeState,
    FlowNodeTrace,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowSourceSpan,
    compile_flow_definition,
)

_async_compile_flow_definition = compile_flow_definition


def compile_flow_definition(*args: object, **kwargs: object) -> object:
    return run_async(_async_compile_flow_definition(*args, **kwargs))


class FlowStateTestCase(TestCase):
    def test_execution_trace_starts_pending_from_plan(self) -> None:
        plan_result = compile_flow_definition(
            FlowDefinition(
                name="stateful",
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
                    outputs={"answer": "finish.result"},
                ),
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(FlowEdgeDefinition(source="start", target="finish"),),
            )
        )

        self.assertTrue(plan_result.ok, plan_result.public_diagnostics)
        assert plan_result.plan is not None
        trace = FlowExecutionTrace.from_plan(plan_result.plan)

        self.assertEqual(
            [node.as_public_dict() for node in trace.nodes],
            [
                {
                    "node": "start",
                    "state": FlowNodeState.PENDING.value,
                    "attempts": 0,
                },
                {
                    "node": "finish",
                    "state": FlowNodeState.PENDING.value,
                    "attempts": 0,
                },
            ],
        )
        self.assertEqual(
            [edge.as_public_dict() for edge in trace.edges],
            [
                {
                    "index": 0,
                    "source": "start",
                    "target": "finish",
                    "state": FlowEdgeState.PENDING.value,
                }
            ],
        )

    def test_execution_trace_updates_are_immutable_and_deterministic(
        self,
    ) -> None:
        trace = FlowExecutionTrace(
            nodes=(
                FlowNodeTrace(node="start"),
                FlowNodeTrace(node="finish"),
            ),
            edges=(FlowEdgeTrace(index=0, source="start", target="finish"),),
        )

        updated = trace.with_node_state(
            "start",
            FlowNodeState.SUCCEEDED,
            attempts=1,
            duration_ms=12.5,
        ).with_edge_state(
            0,
            FlowEdgeState.TAKEN,
            duration_ms=1,
        )

        self.assertEqual(trace.nodes[0].state, FlowNodeState.PENDING)
        self.assertEqual(updated.nodes[0].state, FlowNodeState.SUCCEEDED)
        self.assertEqual(updated.nodes[0].attempts, 1)
        self.assertEqual(updated.nodes[0].duration_ms, 12.5)
        self.assertEqual(updated.edges[0].state, FlowEdgeState.TAKEN)
        self.assertEqual(updated.edges[0].duration_ms, 1)
        with self.assertRaises(FrozenInstanceError):
            updated.nodes[0].state = FlowNodeState.FAILED  # type: ignore[misc]

    def test_all_declared_states_are_public_values(self) -> None:
        self.assertEqual(
            [state.value for state in FlowNodeState],
            [
                "pending",
                "ready",
                "running",
                "succeeded",
                "skipped",
                "failed",
                "retrying",
                "paused",
                "cancelled",
            ],
        )
        self.assertEqual(
            [state.value for state in FlowEdgeState],
            [
                "pending",
                "eligible",
                "taken",
                "suppressed",
                "failed",
            ],
        )

    def test_execution_trace_records_every_runtime_state(self) -> None:
        node_states = tuple(FlowNodeState)
        edge_states = tuple(FlowEdgeState)
        trace = FlowExecutionTrace(
            nodes=tuple(
                FlowNodeTrace(node=f"node_{state.value}")
                for state in node_states
            ),
            edges=tuple(
                FlowEdgeTrace(
                    index=index,
                    source="source",
                    target=f"target_{state.value}",
                )
                for index, state in enumerate(edge_states)
            ),
        )

        for state in node_states:
            trace = trace.with_node_state(
                f"node_{state.value}",
                state,
                attempts=1 if state != FlowNodeState.PENDING else 0,
            )
        for index, state in enumerate(edge_states):
            trace = trace.with_edge_state(index, state)

        public = trace.as_public_dict()
        nodes = cast(tuple[dict[str, object], ...], public["nodes"])
        edges = cast(tuple[dict[str, object], ...], public["edges"])

        self.assertEqual(
            {node["node"]: node["state"] for node in nodes},
            {f"node_{state.value}": state.value for state in FlowNodeState},
        )
        self.assertEqual(
            {edge["index"]: edge["state"] for edge in edges},
            {index: state.value for index, state in enumerate(FlowEdgeState)},
        )

    def test_public_projection_uses_safe_diagnostics(self) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.execution.node_failed",
            category=FlowDiagnosticCategory.EXECUTION,
            path="nodes.start",
            source_span=FlowSourceSpan(
                start_line=1,
                start_column=1,
                source="secret-token",
            ),
            message="Flow node failed.",
            hint="Inspect the sanitized trace.",
        )
        trace = (
            FlowExecutionTrace(
                nodes=(FlowNodeTrace(node="start"),),
                edges=(
                    FlowEdgeTrace(index=0, source="start", target="finish"),
                ),
            )
            .with_node_state(
                "start",
                FlowNodeState.FAILED,
                attempts=2,
                duration_ms=4,
                diagnostics=(diagnostic,),
            )
            .with_edge_state(
                0,
                FlowEdgeState.FAILED,
                duration_ms=1,
                diagnostics=(diagnostic,),
            )
        )

        public = trace.as_public_dict()

        node = cast(tuple[dict[str, object], ...], public["nodes"])[0]
        diagnostics = cast(
            tuple[dict[str, object], ...],
            node["diagnostics"],
        )
        self.assertEqual(diagnostics[0]["category"], "execution")
        edge = cast(tuple[dict[str, object], ...], public["edges"])[0]
        edge_diagnostics = cast(
            tuple[dict[str, object], ...],
            edge["diagnostics"],
        )
        self.assertEqual(edge_diagnostics[0]["category"], "execution")
        self.assertNotIn("secret-token", str(public))
        self.assertIn("duration_ms", node)
        self.assertIn("duration_ms", edge)

    def test_trace_rejects_invalid_nodes_edges_and_updates(self) -> None:
        with self.assertRaises(AssertionError):
            FlowNodeTrace(node="")
        with self.assertRaises(AssertionError):
            FlowNodeTrace(node="start", attempts=-1)
        with self.assertRaises(AssertionError):
            FlowNodeTrace(
                node="start",
                state=cast(FlowNodeState, "running"),
            )
        with self.assertRaises(AssertionError):
            FlowNodeTrace(node="start", duration_ms=-1)
        with self.assertRaises(AssertionError):
            FlowNodeTrace(node="start", attempts=cast(int, True))
        with self.assertRaises(AssertionError):
            FlowEdgeTrace(index=-1, source="start", target="finish")
        with self.assertRaises(AssertionError):
            FlowEdgeTrace(index=0, source="", target="finish")
        with self.assertRaises(AssertionError):
            FlowEdgeTrace(
                index=0,
                source="start",
                target="finish",
                state=cast(FlowEdgeState, "eligible"),
            )
        with self.assertRaises(AssertionError):
            FlowEdgeTrace(
                index=cast(int, True),
                source="start",
                target="finish",
            )
        with self.assertRaises(AssertionError):
            FlowExecutionTrace(
                nodes=(
                    FlowNodeTrace(node="start"),
                    FlowNodeTrace(node="start"),
                ),
            )
        with self.assertRaises(AssertionError):
            FlowExecutionTrace(
                nodes=(FlowNodeTrace(node="start"),),
                edges=(
                    FlowEdgeTrace(index=0, source="start", target="finish"),
                    FlowEdgeTrace(index=0, source="start", target="finish"),
                ),
            )

        trace = FlowExecutionTrace(
            nodes=(FlowNodeTrace(node="start"),),
            edges=(FlowEdgeTrace(index=0, source="start", target="finish"),),
        )
        with self.assertRaises(AssertionError):
            trace.with_node_state("missing", FlowNodeState.READY)
        with self.assertRaises(AssertionError):
            trace.with_edge_state(1, FlowEdgeState.ELIGIBLE)
        with self.assertRaises(AssertionError):
            trace.with_node_state(
                "start",
                FlowNodeState.FAILED,
                diagnostics=(object(),),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    main()
