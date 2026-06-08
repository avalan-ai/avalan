from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import cast
from unittest import TestCase, main

from avalan.flow import (
    FlowArtifactInspection,
    FlowConditionOperator,
    FlowConditionPlan,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeKind,
    FlowEdgePlan,
    FlowEdgeState,
    FlowExecutionPlan,
    FlowExecutionRecord,
    FlowExecutionTrace,
    FlowInputDefinition,
    FlowInputType,
    FlowInspection,
    FlowInspectionRunState,
    FlowLoopInspection,
    FlowLoopPlan,
    FlowNodeAttemptRecord,
    FlowNodeInspection,
    FlowNodeKind,
    FlowNodePlan,
    FlowNodeState,
    FlowNodeTrace,
    FlowOutputDefinition,
    FlowOutputType,
    FlowPlanExecutionResult,
    FlowRetryBackoffStrategy,
    FlowRetryInspection,
    FlowRetryPlan,
    FlowReviewInspection,
    FlowReviewState,
    FlowSourceSpan,
    export_sanitized_flow_trace,
    inspect_flow_record,
    inspect_flow_result,
    parse_flow_selector,
)
from avalan.task import TaskArtifactRef, TaskInputFile


class FlowInspectionTestCase(TestCase):
    def test_result_inspection_exports_safe_runtime_state(self) -> None:
        plan = _plan()
        trace = _trace(plan)
        trace = trace.with_node_state(
            "start",
            FlowNodeState.SUCCEEDED,
            attempts=1,
            duration_ms=1,
        )
        trace = trace.with_node_state(
            "retry",
            FlowNodeState.SUCCEEDED,
            attempts=2,
            duration_ms=2,
        )
        trace = trace.with_node_state(
            "loop",
            FlowNodeState.SUCCEEDED,
            attempts=3,
            duration_ms=3,
        )
        trace = trace.with_node_state(
            "review",
            FlowNodeState.PAUSED,
            attempts=1,
            duration_ms=4,
        )
        trace = trace.with_edge_state(0, FlowEdgeState.TAKEN, duration_ms=1)
        artifact = _artifact_ref("artifact-direct")
        file = TaskInputFile(
            logical_path="/private/report.pdf",
            artifact_ref=artifact,
            metadata={"filename": "private-file-name.pdf"},
        )
        result = FlowPlanExecutionResult(
            trace=trace,
            outputs={
                "answer": {
                    "private": "selected-output",
                    "artifact": artifact,
                    "files": [file],
                }
            },
            node_outputs={"start": {"private": "node-output"}},
            pause_tokens={"review": "private-pause-token"},
        )

        inspection = inspect_flow_result(result, plan=plan)
        exported = export_sanitized_flow_trace(inspection)
        exported_from_result = export_sanitized_flow_trace(result, plan=plan)

        self.assertEqual(inspection.state, FlowInspectionRunState.PAUSED)
        self.assertEqual(exported, exported_from_result)
        self.assertEqual(exported["flow_name"], "inspectable")
        self.assertEqual(exported["flow_version"], "2026-06-08")
        self.assertEqual(exported["flow_revision"], "rev-1")
        self.assertEqual(exported["selected_outputs"], ("answer",))
        retries = cast(tuple[dict[str, object], ...], exported["retries"])
        self.assertEqual(retries[0]["node"], "retry")
        self.assertEqual(retries[0]["attempts"], 2)
        self.assertFalse(retries[0]["exhausted"])
        self.assertEqual(retries[0]["max_attempts"], 2)
        self.assertEqual(retries[0]["exhausted_route"], "fallback")
        loops = cast(tuple[dict[str, object], ...], exported["loops"])
        self.assertEqual(loops[0]["node"], "loop")
        self.assertEqual(loops[0]["iterations"], 3)
        self.assertEqual(loops[0]["limit_route"], "manual")
        reviews = cast(tuple[dict[str, object], ...], exported["reviews"])
        self.assertEqual(reviews[0]["node"], "review")
        self.assertEqual(reviews[0]["state"], "paused")
        self.assertTrue(reviews[0]["has_pause_token"])
        self.assertEqual(reviews[0]["allowed_decisions"], ("approved",))
        artifacts = cast(
            tuple[dict[str, object], ...],
            exported["artifacts"],
        )
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["artifact_id"], "artifact-direct")
        self.assertEqual(artifacts[0]["sha256"], _SHA256)
        rendered = str(exported)
        self.assertNotIn("selected-output", rendered)
        self.assertNotIn("node-output", rendered)
        self.assertNotIn("private-pause-token", rendered)
        self.assertNotIn("private-file-name.pdf", rendered)
        self.assertNotIn("/private/report.pdf", rendered)

    def test_record_inspection_exports_durable_state_without_values(
        self,
    ) -> None:
        plan = _plan()
        diagnostic = _diagnostic()
        trace = _trace(plan)
        trace = trace.with_node_state(
            "start",
            FlowNodeState.SUCCEEDED,
            attempts=1,
        )
        trace = trace.with_node_state(
            "retry",
            FlowNodeState.FAILED,
            attempts=2,
            duration_ms=5,
            diagnostics=(diagnostic,),
        )
        trace = trace.with_node_state(
            "loop",
            FlowNodeState.FAILED,
            attempts=3,
            duration_ms=6,
        )
        trace = trace.with_node_state(
            "review",
            FlowNodeState.SUCCEEDED,
            attempts=1,
        )
        trace = trace.with_node_state(
            "finish",
            FlowNodeState.SKIPPED,
            attempts=0,
        )
        trace = trace.with_edge_state(
            1,
            FlowEdgeState.FAILED,
            diagnostics=(diagnostic,),
        )
        record = FlowExecutionRecord(
            task_run_id="run-1",
            revision=2,
            trace=trace,
            node_attempts=(
                FlowNodeAttemptRecord(
                    node="retry",
                    attempt=1,
                    state=FlowNodeState.FAILED,
                    artifact_refs=(
                        {
                            "artifact_id": "artifact-record",
                            "store": "memory",
                            "storage_key": "private-storage-key",
                            "media_type": "application/json",
                            "size_bytes": 10,
                            "sha256": _SHA256,
                            "metadata": {"filename": "private.json"},
                        },
                    ),
                ),
            ),
            node_outputs={"retry": {"private": "node-output"}},
            selected_outputs={"answer": {"private": "selected-output"}},
            loop_counters={"loop": 3},
            pause_tokens={},
            artifact_refs=(
                {
                    "artifact_id": "artifact-record",
                    "store": "memory",
                },
                {"bad": "shape"},
            ),
            metadata={
                "strict_flow": {
                    "name": "recorded",
                    "version": "2026-06-08",
                    "revision": "rev-record",
                },
                "human_review_audit": {
                    "review": {
                        "state": "resumed",
                        "decision": "approved",
                        "request": {
                            "allowed_decisions": ("approved", "rejected"),
                            "timeout_seconds": 30,
                            "audit_metadata": {"private": "review-note"},
                        },
                    }
                },
            },
            diagnostics=(),
            created_at=datetime(2026, 6, 8, 1, 2, tzinfo=UTC),
            updated_at=datetime(2026, 6, 8, 1, 3, tzinfo=UTC),
        )

        inspection = inspect_flow_record(record, plan=plan)
        exported = export_sanitized_flow_trace(record, plan=plan)
        exported_without_plan = inspect_flow_record(record).as_public_dict()

        self.assertEqual(inspection.state, FlowInspectionRunState.FAILED)
        self.assertEqual(exported["task_run_id"], "run-1")
        self.assertEqual(exported["record_revision"], 2)
        self.assertEqual(exported["created_at"], "2026-06-08T01:02:00+00:00")
        self.assertEqual(exported["updated_at"], "2026-06-08T01:03:00+00:00")
        retries = cast(tuple[dict[str, object], ...], exported["retries"])
        self.assertTrue(retries[0]["exhausted"])
        loops = cast(tuple[dict[str, object], ...], exported["loops"])
        self.assertEqual(loops[0]["iterations"], 3)
        reviews = cast(tuple[dict[str, object], ...], exported["reviews"])
        self.assertEqual(reviews[0]["state"], "resumed")
        self.assertEqual(reviews[0]["decision"], "approved")
        diagnostics = cast(
            tuple[dict[str, object], ...],
            exported["diagnostics"],
        )
        self.assertEqual(diagnostics[0]["category"], "execution")
        self.assertIn("source_span", diagnostics[0])
        artifacts = cast(
            tuple[dict[str, object], ...],
            exported["artifacts"],
        )
        self.assertEqual(artifacts[0]["artifact_id"], "artifact-record")
        self.assertNotIn("storage_key", artifacts[0])
        self.assertNotIn("metadata", artifacts[0])
        self.assertEqual(exported_without_plan["flow_name"], "recorded")
        self.assertEqual(exported_without_plan["flow_revision"], "rev-record")
        rendered = str(exported)
        self.assertNotIn("private-source-token", rendered)
        self.assertNotIn("private-storage-key", rendered)
        self.assertNotIn("private.json", rendered)
        self.assertNotIn("selected-output", rendered)
        self.assertNotIn("node-output", rendered)
        self.assertNotIn("review-note", rendered)

    def test_run_state_derivation_covers_all_public_states(self) -> None:
        self.assertEqual(
            _state_for(FlowNodeState.PENDING),
            FlowInspectionRunState.PENDING,
        )
        self.assertEqual(
            _state_for(FlowNodeState.RUNNING),
            FlowInspectionRunState.RUNNING,
        )
        self.assertEqual(
            _state_for(FlowNodeState.SUCCEEDED),
            FlowInspectionRunState.SUCCEEDED,
        )
        self.assertEqual(
            _state_for(FlowNodeState.FAILED),
            FlowInspectionRunState.FAILED,
        )
        self.assertEqual(
            _state_for(FlowNodeState.PAUSED, pause=True),
            FlowInspectionRunState.PAUSED,
        )
        self.assertEqual(
            _state_for(FlowNodeState.CANCELLED),
            FlowInspectionRunState.CANCELLED,
        )

    def test_review_state_fallbacks_cover_durable_edges(self) -> None:
        expected_states = {
            FlowNodeState.SUCCEEDED: FlowReviewState.SUCCEEDED,
            FlowNodeState.SKIPPED: FlowReviewState.SKIPPED,
            FlowNodeState.FAILED: FlowReviewState.FAILED,
            FlowNodeState.CANCELLED: FlowReviewState.CANCELLED,
            FlowNodeState.READY: FlowReviewState.PENDING,
        }
        plan = _review_plan()
        for node_state, expected in expected_states.items():
            with self.subTest(node_state=node_state):
                trace = FlowExecutionTrace(
                    nodes=(
                        FlowNodeTrace(
                            node="review",
                            state=node_state,
                            attempts=1,
                        ),
                    )
                )
                inspection = inspect_flow_result(
                    FlowPlanExecutionResult(trace=trace),
                    plan=plan,
                )
                self.assertEqual(inspection.reviews[0].state, expected)

        unknown_audit_record = FlowExecutionRecord(
            task_run_id="run-unknown-review",
            revision=1,
            trace=FlowExecutionTrace(nodes=(FlowNodeTrace(node="start"),)),
            metadata={
                "human_review_audit": {
                    "missing": {
                        "request": {
                            "allowed_decisions": "approved",
                            "timeout_seconds": "private-timeout",
                        }
                    }
                }
            },
            created_at=datetime(2026, 6, 8, tzinfo=UTC),
            updated_at=datetime(2026, 6, 8, tzinfo=UTC),
        )
        unknown_review = inspect_flow_record(unknown_audit_record).reviews[0]
        self.assertEqual(unknown_review.state, FlowReviewState.PENDING)
        self.assertEqual(unknown_review.allowed_decisions, ())
        self.assertIsNone(unknown_review.timeout_seconds)

        empty_metadata_record = FlowExecutionRecord(
            task_run_id="run-empty-review",
            revision=1,
            trace=FlowExecutionTrace(nodes=(FlowNodeTrace(node="start"),)),
            created_at=datetime(2026, 6, 8, tzinfo=UTC),
            updated_at=datetime(2026, 6, 8, tzinfo=UTC),
        )
        exported = inspect_flow_record(empty_metadata_record).as_public_dict()
        self.assertNotIn("flow_name", exported)
        self.assertNotIn("reviews", exported)

    def test_models_are_immutable_and_reject_invalid_values(self) -> None:
        node = FlowNodeInspection(
            node="start",
            state=FlowNodeState.SUCCEEDED,
            attempts=1,
        )
        with self.assertRaises(FrozenInstanceError):
            node.node = "changed"  # type: ignore[misc]
        invalid_cases = (
            lambda: FlowNodeInspection(
                node="",
                state=FlowNodeState.PENDING,
                attempts=0,
            ),
            lambda: FlowNodeInspection(
                node="start",
                state=FlowNodeState.PENDING,
                attempts=-1,
            ),
            lambda: FlowRetryInspection(
                node="retry",
                attempts=1,
                exhausted=False,
                max_attempts=0,
            ),
            lambda: FlowLoopInspection(
                node="loop",
                iterations=-1,
                state=FlowNodeState.PENDING,
            ),
            lambda: FlowArtifactInspection(artifact_id="", store="memory"),
            lambda: FlowReviewInspection(
                node="review",
                state=FlowReviewState.PAUSED,
                timeout_seconds=-1,
            ),
            lambda: FlowInspection(
                state=FlowInspectionRunState.PENDING,
                nodes=(object(),),  # type: ignore[arg-type]
            ),
        )
        for build in invalid_cases:
            with self.subTest(build=build):
                with self.assertRaises(AssertionError):
                    build()
        with self.assertRaises(AssertionError):
            inspect_flow_result(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            inspect_flow_record(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            export_sanitized_flow_trace(object())  # type: ignore[arg-type]


def _state_for(
    state: FlowNodeState,
    *,
    pause: bool = False,
) -> FlowInspectionRunState:
    result = FlowPlanExecutionResult(
        trace=FlowExecutionTrace(
            nodes=(FlowNodeTrace(node="node", state=state, attempts=1),)
        ),
        diagnostics=(_diagnostic(),) if state == FlowNodeState.FAILED else (),
        pause_tokens={"node": "pause-token"} if pause else {},
    )
    return inspect_flow_result(result).state


def _trace(plan: FlowExecutionPlan) -> FlowExecutionTrace:
    return FlowExecutionTrace.from_plan(plan)


def _plan() -> FlowExecutionPlan:
    retry = FlowRetryPlan(
        max_attempts=2,
        backoff=FlowRetryBackoffStrategy.NONE,
        retryable_categories=("transient",),
        non_retryable_categories=("validation",),
        exhausted_route="fallback",
    )
    condition = FlowConditionPlan(
        operator=FlowConditionOperator.EXISTS,
        selector=parse_flow_selector("loop.value"),
    )
    loop = FlowLoopPlan(
        max_iterations=3,
        max_elapsed_seconds=9,
        continue_condition=condition,
        exit_condition=condition,
        output_selector=parse_flow_selector("loop.value"),
        limit_route="manual",
    )
    return FlowExecutionPlan(
        name="inspectable",
        version="2026-06-08",
        revision="rev-1",
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.OBJECT),
        ),
        entry_node="start",
        output_selectors={"answer": parse_flow_selector("finish.result")},
        nodes=(
            FlowNodePlan(
                name="start",
                type="echo",
                kind=FlowNodeKind.PASS_THROUGH,
            ),
            FlowNodePlan(
                name="retry",
                type="validation",
                kind=FlowNodeKind.VALIDATION,
                retry=retry,
            ),
            FlowNodePlan(
                name="loop",
                type="validation",
                kind=FlowNodeKind.VALIDATION,
                loop=loop,
            ),
            FlowNodePlan(
                name="review",
                type="human_review",
                kind=FlowNodeKind.HUMAN_REVIEW,
                config={
                    "allowed_decisions": ("approved",),
                    "timeout_seconds": 15,
                },
            ),
            FlowNodePlan(
                name="finish",
                type="echo",
                kind=FlowNodeKind.PASS_THROUGH,
            ),
        ),
        edges=(
            FlowEdgePlan(
                index=0,
                source="start",
                target="retry",
                kind=FlowEdgeKind.SUCCESS,
            ),
            FlowEdgePlan(
                index=1,
                source="retry",
                target="loop",
                kind=FlowEdgeKind.ERROR,
            ),
            FlowEdgePlan(
                index=2,
                source="loop",
                target="review",
                kind=FlowEdgeKind.SUCCESS,
            ),
            FlowEdgePlan(
                index=3,
                source="review",
                target="finish",
                kind=FlowEdgeKind.RESUME,
            ),
        ),
    )


def _review_plan() -> FlowExecutionPlan:
    return FlowExecutionPlan(
        name="review-only",
        version=None,
        revision=None,
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.OBJECT),
        ),
        entry_node="review",
        output_selectors={"answer": parse_flow_selector("review.decision")},
        nodes=(
            FlowNodePlan(
                name="review",
                type="human_review",
                kind=FlowNodeKind.HUMAN_REVIEW,
            ),
        ),
    )


def _diagnostic() -> FlowDiagnostic:
    return FlowDiagnostic(
        code="flow.execution.node_failed",
        category=FlowDiagnosticCategory.EXECUTION,
        path="nodes.retry",
        source_span=FlowSourceSpan(
            start_line=1,
            start_column=1,
            end_line=1,
            end_column=4,
            source="private-source-token",
        ),
        message="Flow node failed.",
        hint="Inspect the safe state.",
    )


def _artifact_ref(artifact_id: str) -> TaskArtifactRef:
    return TaskArtifactRef(
        artifact_id=artifact_id,
        store="memory",
        storage_key="private-storage-key",
        media_type="application/pdf",
        size_bytes=123,
        sha256=_SHA256,
        metadata={"filename": "private-file-name.pdf"},
    )


_SHA256 = "a" * 64


if __name__ == "__main__":
    main()
