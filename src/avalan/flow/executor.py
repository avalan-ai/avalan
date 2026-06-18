from ..task.client import TaskClientInspection
from ..task.context import TaskInputFile
from ..task.definition import TaskDefinition
from ..task.runner import TaskRunResult
from ..task.store import TaskSnapshotMetadata, TaskStoreNotFoundError
from ..task.targets.flow_constants import FLOW_RESUME_DECISIONS_METADATA_KEY
from .definition import FlowDefinition, FlowNodeCapability
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
)
from .inspection import (
    FlowInspection,
    export_sanitized_flow_trace,
    inspect_flow_record,
    inspect_flow_result,
)
from .plan import (
    FlowExecutionPlan,
    FlowPlanCompileResult,
    compile_flow_definition,
)
from .registry import FlowNodeRegistry
from .runtime import (
    FlowPlanExecutionResult,
    FlowPlanNodeRunner,
    FlowStreamListener,
    execute_flow_plan,
    flow_node_registry_runner,
)
from .state import FlowExecutionTrace
from .store import FlowExecutionRecord, FlowStateStore

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Protocol, cast


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


FlowCancellationChecker = Callable[[], Awaitable[None]]
FlowExecutionInput = FlowDefinition | FlowExecutionPlan
FlowTaskRunMetadata = Mapping[str, object] | None


class FlowTaskClient(Protocol):
    async def run(
        self,
        definition: TaskDefinition,
        *,
        input_value: object = None,
        files: tuple[TaskInputFile, ...] = (),
        idempotency_key: str | None = None,
        metadata: Mapping[str, object] | None = None,
        expires_at: datetime | None = None,
    ) -> TaskRunResult: ...

    async def inspect(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> TaskClientInspection: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowExecutorRunResult:
    plan: FlowExecutionPlan | None = None
    result: FlowPlanExecutionResult | None = None
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.plan is not None:
            assert isinstance(self.plan, FlowExecutionPlan)
        if self.result is not None:
            assert isinstance(self.result, FlowPlanExecutionResult)
        assert isinstance(self.diagnostics, tuple)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, FlowDiagnostic)

    @property
    def ok(self) -> bool:
        return (
            self.result is not None and self.result.ok and not self.diagnostics
        )

    @property
    def outputs(self) -> Mapping[str, object]:
        if self.result is None:
            return _empty_mapping()
        return self.result.outputs

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        diagnostics = self.diagnostics
        if self.result is not None:
            diagnostics = diagnostics + self.result.diagnostics
        return tuple(diagnostic.as_public_dict() for diagnostic in diagnostics)

    def inspect(self) -> FlowInspection:
        assert self.result is not None, "flow execution result is unavailable"
        return inspect_flow_result(self.result, plan=self.plan)

    def export_sanitized_trace(self) -> TaskSnapshotMetadata:
        return self.inspect().export_sanitized_trace()


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowTaskInspection:
    task: TaskClientInspection
    flow: FlowInspection | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.task, TaskClientInspection)
        if self.flow is not None:
            assert isinstance(self.flow, FlowInspection)

    def as_public_dict(self) -> TaskSnapshotMetadata:
        value: dict[str, object] = {"task": self.task.as_dict()}
        if self.flow is not None:
            value["flow"] = self.flow.as_public_dict()
        return cast(TaskSnapshotMetadata, value)

    def export_sanitized_trace(self) -> TaskSnapshotMetadata:
        value: dict[str, object] = {"task": self.task.as_dict()}
        if self.flow is not None:
            value["flow"] = self.flow.export_sanitized_trace()
        return cast(TaskSnapshotMetadata, value)


class FlowExecutor:
    def __init__(
        self,
        *,
        registry: FlowNodeRegistry | None = None,
        runner: FlowPlanNodeRunner | None = None,
        cancellation_checker: FlowCancellationChecker | None = None,
        event_listener: FlowStreamListener | None = None,
        concurrency_limit: int = 1,
    ) -> None:
        if registry is not None:
            assert isinstance(registry, FlowNodeRegistry)
        if runner is not None:
            assert callable(runner)
        if cancellation_checker is not None:
            assert callable(cancellation_checker)
        if event_listener is not None:
            assert callable(event_listener)
        assert isinstance(concurrency_limit, int)
        assert not isinstance(concurrency_limit, bool)
        assert concurrency_limit > 0
        self._registry = registry
        self._runner = runner
        self._cancellation_checker = cancellation_checker
        self._event_listener = event_listener
        self._concurrency_limit = concurrency_limit

    async def run(
        self,
        flow: FlowExecutionInput,
        *,
        inputs: Mapping[str, object] | None = None,
        runner: FlowPlanNodeRunner | None = None,
        cancellation_checker: FlowCancellationChecker | None = None,
        event_listener: FlowStreamListener | None = None,
        concurrency_limit: int | None = None,
    ) -> FlowExecutorRunResult:
        if inputs is not None:
            assert isinstance(inputs, Mapping)
        if runner is not None:
            assert callable(runner)
        plan_result = await self._compile(flow)
        if not plan_result.ok:
            return FlowExecutorRunResult(
                plan=plan_result.plan,
                diagnostics=plan_result.diagnostics,
            )
        assert plan_result.plan is not None
        direct_diagnostics = _direct_execution_diagnostics(plan_result.plan)
        if direct_diagnostics:
            return FlowExecutorRunResult(
                plan=plan_result.plan,
                diagnostics=direct_diagnostics,
            )
        result = await execute_flow_plan(
            plan_result.plan,
            runner or self._node_runner(),
            inputs=inputs,
            cancellation_checker=(
                cancellation_checker or self._cancellation_checker
            ),
            event_listener=event_listener or self._event_listener,
            concurrency_limit=self._concurrency_limit_value(concurrency_limit),
        )
        return FlowExecutorRunResult(plan=plan_result.plan, result=result)

    async def resume(
        self,
        flow: FlowExecutionInput,
        previous: (
            "FlowExecutorRunResult | FlowPlanExecutionResult | "
            "FlowExecutionRecord"
        ),
        *,
        decisions: Mapping[str, Mapping[str, object]],
        inputs: Mapping[str, object] | None = None,
        runner: FlowPlanNodeRunner | None = None,
        cancellation_checker: FlowCancellationChecker | None = None,
        event_listener: FlowStreamListener | None = None,
        concurrency_limit: int | None = None,
    ) -> FlowExecutorRunResult:
        assert isinstance(decisions, Mapping)
        for node, payload in decisions.items():
            assert isinstance(node, str) and node.strip()
            assert isinstance(payload, Mapping)
        if inputs is not None:
            assert isinstance(inputs, Mapping)
        if runner is not None:
            assert callable(runner)
        plan_result = await self._compile(flow)
        if not plan_result.ok:
            return FlowExecutorRunResult(
                plan=plan_result.plan,
                diagnostics=plan_result.diagnostics,
            )
        assert plan_result.plan is not None
        direct_diagnostics = _direct_execution_diagnostics(plan_result.plan)
        if direct_diagnostics:
            return FlowExecutorRunResult(
                plan=plan_result.plan,
                diagnostics=direct_diagnostics,
            )
        resume_trace, resume_node_outputs = _resume_state(previous)
        result = await execute_flow_plan(
            plan_result.plan,
            runner or self._node_runner(),
            inputs=inputs,
            cancellation_checker=(
                cancellation_checker or self._cancellation_checker
            ),
            event_listener=event_listener or self._event_listener,
            concurrency_limit=self._concurrency_limit_value(concurrency_limit),
            resume_trace=resume_trace,
            resume_node_outputs=resume_node_outputs,
            resume_decisions=decisions,
        )
        return FlowExecutorRunResult(plan=plan_result.plan, result=result)

    def inspect(
        self,
        value: (
            "FlowExecutorRunResult | FlowInspection | "
            "FlowPlanExecutionResult | FlowExecutionRecord"
        ),
        *,
        plan: FlowExecutionPlan | None = None,
    ) -> FlowInspection:
        if isinstance(value, FlowExecutorRunResult):
            return value.inspect()
        if isinstance(value, FlowInspection):
            return value
        if isinstance(value, FlowPlanExecutionResult):
            return inspect_flow_result(value, plan=plan)
        if isinstance(value, FlowExecutionRecord):
            return inspect_flow_record(value, plan=plan)
        raise AssertionError("value must be a flow execution result or record")

    def export_trace(
        self,
        value: (
            "FlowExecutorRunResult | FlowInspection | "
            "FlowPlanExecutionResult | FlowExecutionRecord"
        ),
        *,
        plan: FlowExecutionPlan | None = None,
    ) -> TaskSnapshotMetadata:
        if isinstance(value, FlowExecutorRunResult):
            return value.export_sanitized_trace()
        return export_sanitized_flow_trace(value, plan=plan)

    async def _compile(
        self, flow: FlowExecutionInput
    ) -> FlowPlanCompileResult:
        if isinstance(flow, FlowExecutionPlan):
            return FlowPlanCompileResult(plan=flow)
        if isinstance(flow, FlowDefinition):
            return await compile_flow_definition(flow, self._registry)
        raise AssertionError("flow must be a flow definition or plan")

    def _node_runner(self) -> FlowPlanNodeRunner:
        if self._runner is not None:
            return self._runner
        return flow_node_registry_runner(self._registry)

    def _concurrency_limit_value(self, value: int | None) -> int:
        if value is None:
            return self._concurrency_limit
        assert isinstance(value, int)
        assert not isinstance(value, bool)
        assert value > 0
        return value


class FlowTaskExecutor:
    def __init__(
        self,
        client: FlowTaskClient,
        *,
        flow_state_store: FlowStateStore | None = None,
    ) -> None:
        assert hasattr(client, "run")
        assert hasattr(client, "inspect")
        if flow_state_store is not None:
            assert hasattr(flow_state_store, "get_flow_execution")
        self._client = client
        self._flow_state_store = flow_state_store

    async def run(
        self,
        definition: TaskDefinition,
        *,
        input_value: object = None,
        files: tuple[TaskInputFile, ...] = (),
        idempotency_key: str | None = None,
        metadata: FlowTaskRunMetadata = None,
        expires_at: datetime | None = None,
    ) -> TaskRunResult:
        return await self._client.run(
            definition,
            input_value=input_value,
            files=files,
            idempotency_key=idempotency_key,
            metadata=metadata,
            expires_at=expires_at,
        )

    async def resume(
        self,
        definition: TaskDefinition,
        *,
        decisions: Mapping[str, Mapping[str, object]],
        input_value: object = None,
        files: tuple[TaskInputFile, ...] = (),
        idempotency_key: str | None = None,
        metadata: FlowTaskRunMetadata = None,
        expires_at: datetime | None = None,
    ) -> TaskRunResult:
        assert isinstance(decisions, Mapping)
        return await self.run(
            definition,
            input_value=input_value,
            files=files,
            idempotency_key=idempotency_key,
            metadata=_resume_metadata(metadata, decisions),
            expires_at=expires_at,
        )

    async def inspect(
        self,
        run_id: str,
        *,
        plan: FlowExecutionPlan | None = None,
        after_sequence: int | None = None,
    ) -> FlowTaskInspection:
        assert isinstance(run_id, str) and run_id.strip()
        task = await self._client.inspect(
            run_id,
            after_sequence=after_sequence,
        )
        return FlowTaskInspection(
            task=task,
            flow=await self._flow_inspection(run_id, plan=plan),
        )

    async def export_trace(
        self,
        run_id: str,
        *,
        plan: FlowExecutionPlan | None = None,
        after_sequence: int | None = None,
    ) -> TaskSnapshotMetadata:
        inspection = await self.inspect(
            run_id,
            plan=plan,
            after_sequence=after_sequence,
        )
        return inspection.export_sanitized_trace()

    async def _flow_inspection(
        self,
        run_id: str,
        *,
        plan: FlowExecutionPlan | None,
    ) -> FlowInspection | None:
        if self._flow_state_store is None:
            return None
        try:
            record = await self._flow_state_store.get_flow_execution(run_id)
        except TaskStoreNotFoundError:
            return None
        return inspect_flow_record(record, plan=plan)


def _resume_state(
    previous: (
        FlowExecutorRunResult | FlowPlanExecutionResult | FlowExecutionRecord
    ),
) -> tuple[FlowExecutionTrace, Mapping[str, Mapping[str, object]]]:
    if isinstance(previous, FlowExecutorRunResult):
        assert previous.result is not None, "previous result is unavailable"
        return previous.result.trace, previous.result.node_outputs
    if isinstance(previous, FlowPlanExecutionResult):
        return previous.trace, previous.node_outputs
    if isinstance(previous, FlowExecutionRecord):
        return previous.trace, _nested_node_outputs(previous.node_outputs)
    raise AssertionError("previous must be a flow result or execution record")


def _direct_execution_diagnostics(
    plan: FlowExecutionPlan,
) -> tuple[FlowDiagnostic, ...]:
    assert isinstance(plan, FlowExecutionPlan)
    for node in plan.nodes:
        if FlowNodeCapability.TASK_BACKED in node.capabilities:
            return (
                FlowDiagnostic(
                    code=(
                        "flow.execution.task_backed_node_requires_task_runner"
                    ),
                    category=FlowDiagnosticCategory.EXECUTION,
                    severity=FlowDiagnosticSeverity.ERROR,
                    path=f"nodes.{node.name}.type",
                    message="Task-backed flow nodes require task execution.",
                    hint="Run this flow through a task-backed flow executor.",
                ),
            )
    return ()


def _nested_node_outputs(
    value: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    outputs: dict[str, Mapping[str, object]] = {}
    for node, output in value.items():
        assert isinstance(node, str) and node.strip()
        assert isinstance(output, Mapping)
        outputs[node] = output
    return MappingProxyType(outputs)


def _resume_metadata(
    metadata: FlowTaskRunMetadata,
    decisions: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    assert isinstance(decisions, Mapping)
    value = dict(metadata or {})
    assert FLOW_RESUME_DECISIONS_METADATA_KEY not in value
    value[FLOW_RESUME_DECISIONS_METADATA_KEY] = {
        node: dict(payload) for node, payload in decisions.items()
    }
    return MappingProxyType(value)
