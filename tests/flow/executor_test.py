from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.flow import (
    FlowDefinition,
    FlowDiagnosticCategory,
    FlowEdgeKind,
    FlowEdgePlan,
    FlowEntryBehavior,
    FlowExecutionPlan,
    FlowExecutionTrace,
    FlowExecutor,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowInspection,
    FlowInspectionRunState,
    FlowMappingKind,
    FlowMappingPlan,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodePlan,
    FlowNodeState,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowPlanExecutionResult,
    FlowTaskExecutor,
    FlowTaskInspection,
    InMemoryFlowStateStore,
    parse_flow_selector,
)
from avalan.flow.executor import _resume_metadata
from avalan.task import (
    RunMode,
    TaskClientInspection,
    TaskClientOutput,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskInputFile,
    TaskInputType,
    TaskMetadata,
    TaskOutputContract,
    TaskOutputType,
    TaskRunPolicy,
    TaskRunResult,
    UsageTotals,
)
from avalan.task.store import TaskExecutionRequest
from avalan.task.stores import InMemoryTaskStore
from avalan.task.targets.flow import FLOW_RESUME_DECISIONS_METADATA_KEY


class FlowExecutorTestCase(IsolatedAsyncioTestCase):
    async def test_run_executes_definition_and_exports_sanitized_trace(
        self,
    ) -> None:
        executor = FlowExecutor()

        result = await executor.run(
            _definition(),
            inputs={"payload": "private value"},
        )
        inspection = executor.inspect(result)
        exported = executor.export_trace(result)

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "private value"})
        self.assertIsInstance(result.result, FlowPlanExecutionResult)
        self.assertEqual(inspection.state, FlowInspectionRunState.SUCCEEDED)
        self.assertEqual(exported["flow_name"], "sdk-runtime")
        self.assertEqual(exported["selected_outputs"], ("answer",))
        self.assertNotIn("private value", str(exported))

    async def test_run_returns_compile_diagnostics_without_execution(
        self,
    ) -> None:
        calls: list[str] = []

        async def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            return dict(inputs)

        result = await FlowExecutor(runner=runner).run(_invalid_definition())

        self.assertFalse(result.ok)
        self.assertIsNone(result.result)
        self.assertEqual(calls, [])
        self.assertEqual(result.outputs, {})
        self.assertEqual(
            result.public_diagnostics[0]["category"],
            FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION.value,
        )
        with self.assertRaises(AssertionError):
            result.inspect()
        with self.assertRaises(AssertionError):
            FlowExecutor().inspect(result)

    async def test_resume_continues_paused_human_review(self) -> None:
        executor = FlowExecutor()
        plan = _human_review_plan()

        paused = await executor.run(
            plan,
            inputs={"payload": {"note": "private review input"}},
        )
        resumed = await executor.resume(
            plan,
            paused,
            decisions={"review": {"decision": "approved"}},
        )

        self.assertTrue(paused.ok, paused.public_diagnostics)
        self.assertTrue(resumed.ok, resumed.public_diagnostics)
        self.assertEqual(paused.inspect().state, FlowInspectionRunState.PAUSED)
        self.assertEqual(resumed.outputs, {"answer": "approved"})
        assert resumed.result is not None
        states = {node.node: node.state for node in resumed.result.trace.nodes}
        self.assertEqual(states["review"], FlowNodeState.SUCCEEDED)
        self.assertEqual(states["finish"], FlowNodeState.SUCCEEDED)
        exported = resumed.export_sanitized_trace()
        self.assertNotIn("private review input", str(exported))

    async def test_resume_validates_arguments_and_previous_result(
        self,
    ) -> None:
        executor = FlowExecutor()
        plan = _human_review_plan()
        failed_compile = await executor.run(_invalid_definition())

        with self.assertRaises(AssertionError):
            await executor.resume(
                plan,
                failed_compile,
                decisions={"review": {"decision": "approved"}},
            )
        with self.assertRaises(AssertionError):
            await executor.resume(
                plan,
                object(),  # type: ignore[arg-type]
                decisions={"review": {"decision": "approved"}},
            )
        with self.assertRaises(AssertionError):
            await executor.resume(
                plan,
                await executor.run(plan),
                decisions={"review": "approved"},  # type: ignore[dict-item]
            )

    async def test_executor_validates_constructor_and_overrides(
        self,
    ) -> None:
        plan = _plan()
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            return dict(inputs)

        async def cancellation_checker() -> None:
            return None

        def event_listener(_: object) -> None:
            return None

        with self.assertRaises(AssertionError):
            FlowExecutor(concurrency_limit=0)
        with self.assertRaises(AssertionError):
            FlowExecutor(runner=object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            FlowExecutor(registry=object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            FlowExecutor(cancellation_checker=object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            FlowExecutor(event_listener=object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await FlowExecutor().run(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await FlowExecutor().run(
                plan,
                concurrency_limit=True,  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            await FlowExecutor().run(
                plan,
                runner=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            await FlowExecutor().run(plan, concurrency_limit=0)
        with self.assertRaises(AssertionError):
            await FlowExecutor().resume(
                plan,
                FlowPlanExecutionResult(
                    trace=FlowExecutionTrace.from_plan(plan)
                ),
                decisions={"echo": {"decision": "approved"}},
                runner=object(),  # type: ignore[arg-type]
            )

        result = await FlowExecutor(
            runner=runner,
            cancellation_checker=cancellation_checker,
            event_listener=event_listener,
        ).run(plan, inputs={"payload": "safe"}, concurrency_limit=1)

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(calls, ["echo"])

    async def test_task_executor_delegates_run_resume_and_inspection(
        self,
    ) -> None:
        plan = _plan()
        task_client = _FakeTaskClient()
        flow_store = InMemoryFlowStateStore()
        await flow_store.create_flow_execution(
            "run-1",
            trace=plan_trace(plan),
            selected_outputs={"answer": "private selected output"},
            metadata={"strict_flow": {"name": "durable"}},
        )
        executor = FlowTaskExecutor(
            task_client,
            flow_state_store=flow_store,
        )
        definition = _task_definition()

        run = await executor.run(
            definition,
            input_value={"private": "run"},
            metadata={"safe": "metadata"},
        )
        resumed = await executor.resume(
            definition,
            decisions={"review": {"decision": "approved"}},
            input_value={"private": "resume"},
            metadata={"safe": "metadata"},
        )
        inspection = await executor.inspect("run-1", plan=plan)
        exported = await executor.export_trace("run-1", plan=plan)

        self.assertEqual(run.run.run_id, "run-1")
        self.assertEqual(resumed.run.run_id, "run-1")
        self.assertEqual(task_client.run_metadata[0], {"safe": "metadata"})
        self.assertEqual(
            task_client.run_metadata[1][FLOW_RESUME_DECISIONS_METADATA_KEY],
            {"review": {"decision": "approved"}},
        )
        self.assertIsInstance(inspection, FlowTaskInspection)
        self.assertIsInstance(inspection.flow, FlowInspection)
        assert inspection.flow is not None
        self.assertEqual(inspection.flow.flow_name, "sdk-runtime")
        flow_export = cast(Mapping[str, object], exported["flow"])
        self.assertEqual(flow_export["flow_name"], "sdk-runtime")
        self.assertNotIn("private selected output", str(exported))
        public = inspection.as_public_dict()
        self.assertIn("flow", public)

    async def test_task_executor_handles_missing_flow_state(self) -> None:
        task_client = _FakeTaskClient()
        executor = FlowTaskExecutor(
            task_client,
            flow_state_store=InMemoryFlowStateStore(),
        )

        inspection = await executor.inspect("run-1")
        exported = await executor.export_trace("run-1")

        self.assertIsNone(inspection.flow)
        self.assertNotIn("flow", exported)

        no_store_inspection = await FlowTaskExecutor(task_client).inspect(
            "run-1"
        )
        self.assertIsNone(no_store_inspection.flow)

    async def test_task_executor_validates_inputs(self) -> None:
        task_client = _FakeTaskClient()

        with self.assertRaises(AssertionError):
            FlowTaskExecutor(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            FlowTaskExecutor(
                task_client,
                flow_state_store=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            await FlowTaskExecutor(task_client).inspect("")
        with self.assertRaises(AssertionError):
            _resume_metadata(
                {FLOW_RESUME_DECISIONS_METADATA_KEY: {}},
                {"review": {"decision": "approved"}},
            )

    async def test_inspect_export_and_resume_support_result_variants(
        self,
    ) -> None:
        executor = FlowExecutor()
        plan = _human_review_plan()
        paused = await executor.run(plan)
        assert paused.result is not None
        flow_store = InMemoryFlowStateStore()
        record = await flow_store.create_flow_execution(
            "run-record",
            trace=paused.result.trace,
            node_outputs={"review": {"result": {"decision": "approved"}}},
        )

        direct_inspection = executor.inspect(paused.result, plan=plan)
        same_inspection = executor.inspect(direct_inspection)
        record_inspection = executor.inspect(record, plan=plan)
        exported = executor.export_trace(paused.result, plan=plan)
        resumed_from_result = await executor.resume(
            plan,
            paused.result,
            decisions={"review": {"decision": "approved"}},
            inputs={},
            concurrency_limit=1,
        )
        resumed_from_record = await executor.resume(
            plan,
            record,
            decisions={"review": {"decision": "approved"}},
        )
        compile_failed_resume = await executor.resume(
            _invalid_definition(),
            record,
            decisions={},
            inputs={},
        )
        bad_record = await flow_store.create_flow_execution(
            "bad-record",
            trace=paused.result.trace,
            node_outputs={"review": "bad"},
        )

        self.assertIs(same_inspection, direct_inspection)
        self.assertEqual(
            record_inspection.state, FlowInspectionRunState.PAUSED
        )
        self.assertEqual(exported["flow_name"], "review")
        self.assertEqual(resumed_from_result.outputs, {"answer": "approved"})
        self.assertEqual(resumed_from_record.outputs, {"answer": "approved"})
        self.assertFalse(compile_failed_resume.ok)
        self.assertIsNone(compile_failed_resume.result)
        with self.assertRaises(AssertionError):
            executor.inspect(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await executor.resume(
                plan,
                bad_record,
                decisions={"review": {"decision": "approved"}},
            )


class _FakeTaskClient:
    def __init__(self) -> None:
        self.run_metadata: list[Mapping[str, object] | None] = []

    async def run(
        self,
        definition: TaskDefinition,
        *,
        input_value: object = None,
        files: tuple[TaskInputFile, ...] = (),
        idempotency_key: str | None = None,
        metadata: Mapping[str, object] | None = None,
        expires_at: datetime | None = None,
    ) -> TaskRunResult:
        assert isinstance(definition, TaskDefinition)
        _ = input_value, files, idempotency_key, expires_at
        self.run_metadata.append(metadata)
        inspection = await self.inspect("run-1")
        return TaskRunResult(
            run=inspection.run,
            attempt=inspection.attempts[0],
            output={"answer": "safe"},
        )

    async def inspect(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> TaskClientInspection:
        assert run_id == "run-1"
        _ = after_sequence
        ids = iter(("run-1", "attempt-1"))
        store = InMemoryTaskStore(
            clock=lambda: datetime(2026, 6, 8, tzinfo=UTC),
            id_factory=lambda: next(ids),
        )
        definition = _task_definition()
        await store.register_definition(
            definition,
            definition_hash="task-hash",
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id="task-hash")
        )
        attempt = await store.create_attempt(run.run_id)
        return TaskClientInspection(
            run=run,
            attempts=(attempt,),
            output=TaskClientOutput(run_id=run.run_id, state=run.state),
            events=(),
            usage=(),
            usage_totals=UsageTotals(),
            artifacts=(),
        )


def plan_trace(plan: FlowExecutionPlan) -> FlowExecutionTrace:
    return FlowExecutionTrace.from_plan(plan)


def _definition() -> FlowDefinition:
    return FlowDefinition(
        name="sdk-runtime",
        version="2026-06-08",
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.JSON),
        ),
        entry_behavior=FlowEntryBehavior(node="echo"),
        output_behavior=FlowOutputBehavior(outputs={"answer": "echo.value"}),
        nodes=(
            FlowNodeDefinition(
                name="echo",
                type="pass-through",
                mappings=(
                    FlowInputMapping(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source="inputs.payload",
                    ),
                ),
            ),
        ),
    )


def _invalid_definition() -> FlowDefinition:
    return FlowDefinition(
        name="invalid",
        version="2026-06-08",
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.JSON),
        ),
        output_behavior=FlowOutputBehavior(
            outputs={"answer": "missing.value"}
        ),
        nodes=(FlowNodeDefinition(name="echo", type="pass-through"),),
    )


def _plan() -> FlowExecutionPlan:
    return FlowExecutionPlan(
        name="sdk-runtime",
        version="2026-06-08",
        revision=None,
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.JSON),
        ),
        entry_node="echo",
        output_selectors={"answer": parse_flow_selector("echo.value")},
        nodes=(
            FlowNodePlan(
                name="echo",
                type="pass-through",
                kind=FlowNodeKind.PASS_THROUGH,
                mappings=(
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("inputs.payload"),
                    ),
                ),
                output_contracts=(
                    FlowNodeContract(name="value", type=FlowOutputType.JSON),
                ),
            ),
        ),
    )


def _human_review_plan() -> FlowExecutionPlan:
    return FlowExecutionPlan(
        name="review",
        version="2026-06-08",
        revision=None,
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.JSON),
        ),
        entry_node="review",
        output_selectors={"answer": parse_flow_selector("finish.value")},
        nodes=(
            FlowNodePlan(
                name="review",
                type="human_review",
                kind=FlowNodeKind.HUMAN_REVIEW,
                config={"allowed_decisions": ("approved", "rejected")},
                output_contracts=(
                    FlowNodeContract(
                        name="result",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
            ),
            FlowNodePlan(
                name="finish",
                type="pass-through",
                kind=FlowNodeKind.PASS_THROUGH,
                mappings=(
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("review.result.decision"),
                    ),
                ),
                output_contracts=(
                    FlowNodeContract(name="value", type=FlowOutputType.JSON),
                ),
            ),
            FlowNodePlan(
                name="rejected",
                type="pass-through",
                kind=FlowNodeKind.PASS_THROUGH,
                output_contracts=(
                    FlowNodeContract(name="value", type=FlowOutputType.JSON),
                ),
            ),
        ),
        edges=(
            FlowEdgePlan(
                index=0,
                source="review",
                target="finish",
                kind=FlowEdgeKind.RESUME,
                label="approved",
            ),
            FlowEdgePlan(
                index=1,
                source="review",
                target="rejected",
                kind=FlowEdgeKind.RESUME,
                label="rejected",
            ),
        ),
    )


def _task_definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="flow-task", version="1"),
        input=TaskInputContract(type=TaskInputType.OBJECT),
        output=TaskOutputContract(type=TaskOutputType.JSON),
        execution=TaskExecutionTarget.flow("flows/review.toml"),
        run=TaskRunPolicy(mode=RunMode.DIRECT),
    )


if __name__ == "__main__":
    main()
