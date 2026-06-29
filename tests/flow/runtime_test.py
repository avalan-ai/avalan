from ast import Attribute, Call, ImportFrom, Name, parse, walk
from asyncio import CancelledError, gather, sleep
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, replace
from json import dumps
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

from async_helpers import run_async

import avalan.flow.executor as flow_executor_module
import avalan.flow.inspection as flow_inspection_module
import avalan.flow.plan as flow_plan_module
import avalan.flow.runtime as flow_runtime_module
from avalan.entities import ToolManagerSettings
from avalan.event import (
    Event,
    EventObservabilityPayload,
    EventPayloadKind,
    EventType,
)
from avalan.flow import (
    FlowConditionOperator,
    FlowConditionPlan,
    FlowConditionValueType,
    FlowDefinition,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeKind,
    FlowEdgePlan,
    FlowEdgeState,
    FlowEntryBehavior,
    FlowExecutionPlan,
    FlowExecutionTrace,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowJoinPlan,
    FlowJoinPolicyType,
    FlowLoopPlan,
    FlowMappingKind,
    FlowMappingPlan,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeExecutionError,
    FlowNodeKind,
    FlowNodePlan,
    FlowNodeRegistryRunner,
    FlowNodeState,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowPlanExecutionResult,
    FlowRetryBackoffStrategy,
    FlowRetryPlan,
    FlowRouteMatchPolicy,
    FlowRuntimeContext,
    FlowRuntimeEvaluationError,
    FlowStreamRecorder,
    FlowStreamSession,
    FlowTimeoutPlan,
    canonical_flow_item,
    compile_flow_definition,
    evaluate_flow_condition_plan,
    evaluate_flow_mappings,
    evaluate_flow_node_mappings,
    evaluate_flow_selector,
    execute_flow_plan,
    flow_node_registry_runner,
    flow_stream_recorder,
    flow_stream_session,
    loads_flow_definition_result,
    parse_flow_selector,
    resolve_flow_selector_value,
    tool_flow_node_registry,
)
from avalan.flow.runtime import (
    _append_condition_event_draft,
    _append_join_event_draft,
    _append_node_event_draft,
    _emit_node_outcome_event,
    _exception_class,
    _flow_event_payload,
    _join_ready,
    _json_schema_adapter,
    _json_schema_adapter_from_module,
    _node_outcome_event_type,
    _node_trace_attempts,
    _NodeRunOutcome,
    _resume_label_matches,
    _retry_delay_seconds,
    _route_from_node,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    stream_observability_payload,
)
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager

_async_compile_flow_definition = compile_flow_definition
_async_loads_flow_definition_result = loads_flow_definition_result
_FLOW_RUNTIME_BOUNDARY_MODULES = {
    "executor.py": flow_executor_module,
    "inspection.py": flow_inspection_module,
    "plan.py": flow_plan_module,
    "runtime.py": flow_runtime_module,
}
_FORBIDDEN_RUNTIME_AUTHORING_NAMES = frozenset(
    {
        "FlowDefinitionLoader",
        "compile_flow_file",
        "compile_flow_graph",
        "compile_flow_source",
        "inspect_flow_graph_file",
        "inspect_flow_graph_source",
        "load_flow_definition",
        "load_flow_definition_result",
        "loads_flow_definition",
        "loads_flow_definition_result",
        "parse_mermaid",
        "parse_mermaid_import",
        "parse_mermaid_view",
        "read_text",
    }
)
_FORBIDDEN_RUNTIME_ASYNC_NAMES = frozenset({"run", "to_thread"})
_FORBIDDEN_RUNTIME_LOOP_METHODS = frozenset({"run_until_complete"})


def compile_flow_definition(*args: object, **kwargs: object) -> object:
    return run_async(_async_compile_flow_definition(*args, **kwargs))


def loads_flow_definition_result(
    *args: object,
    **kwargs: object,
) -> object:
    return run_async(_async_loads_flow_definition_result(*args, **kwargs))


class _FlowEventCollector:
    def __init__(self) -> None:
        self._events: list[Event] = []

    def append(self, item: CanonicalStreamItem) -> None:
        assert isinstance(item, CanonicalStreamItem)
        event_type = item.metadata.get("event_type")
        assert isinstance(event_type, str)
        typed_event_type = _event_type_value(event_type)
        payload = item.data if isinstance(item.data, Mapping) else {}
        self._events.append(
            Event(
                type=typed_event_type,
                payload=payload,
                observability_payload=(
                    EventObservabilityPayload.canonical_stream(
                        stream_observability_payload(item)
                    )
                ),
                started=_optional_float(item.metadata.get("started")),
                finished=_optional_float(item.metadata.get("finished")),
                elapsed=_optional_float(item.metadata.get("elapsed")),
            )
        )

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __getitem__(self, index: int) -> Event:
        return self._events[index]

    def __len__(self) -> int:
        return len(self._events)

    def __eq__(self, other: object) -> bool:
        return self._events == other

    def __str__(self) -> str:
        return str(self._events)


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _event_type_value(value: str) -> EventType | str:
    try:
        return EventType(value)
    except ValueError:
        return value


def _test_flow_item(
    event_type: EventType | str,
    payload: Mapping[str, object],
    *,
    sequence: int = 0,
) -> CanonicalStreamItem:
    return canonical_flow_item(
        stream_session=flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
        ),
        event_type=event_type,
        payload=payload,
        sequence=sequence,
    )


def _runtime_boundary_violations(
    path: Path,
    source: str,
) -> tuple[str, ...]:
    tree = parse(source, filename=str(path))
    violations: list[str] = []
    for node in walk(tree):
        if isinstance(node, ImportFrom):
            violations.extend(
                f"{_node_location(path, node)} imports {alias.name}"
                for alias in node.names
                if alias.name in _FORBIDDEN_RUNTIME_AUTHORING_NAMES
                or (
                    node.module == "asyncio"
                    and alias.name in _FORBIDDEN_RUNTIME_ASYNC_NAMES
                )
            )
            continue
        if isinstance(node, Call):
            call_name = _call_name(node)
            if call_name in _FORBIDDEN_RUNTIME_AUTHORING_NAMES:
                violations.append(
                    f"{_node_location(path, node)} calls {call_name}"
                )
            if _calls_forbidden_async_bridge(node):
                violations.append(
                    f"{_node_location(path, node)} calls {call_name}"
                )
    return tuple(sorted(violations))


def _call_name(node: Call) -> str:
    func = node.func
    if isinstance(func, Name):
        return func.id
    if isinstance(func, Attribute):
        return func.attr
    return ""


def _calls_forbidden_async_bridge(node: Call) -> bool:
    func = node.func
    if isinstance(func, Name):
        return func.id in _FORBIDDEN_RUNTIME_ASYNC_NAMES
    if isinstance(func, Attribute):
        return func.attr in _FORBIDDEN_RUNTIME_LOOP_METHODS or (
            isinstance(func.value, Name)
            and func.value.id == "asyncio"
            and func.attr in _FORBIDDEN_RUNTIME_ASYNC_NAMES
        )
    return False


def _node_location(path: Path, node: object) -> str:
    return f"{path}:{getattr(node, 'lineno', 0)}"


class FlowRuntimeBoundaryTestCase(TestCase):
    def test_runtime_facing_modules_do_not_load_graph_authoring(self) -> None:
        violations = tuple(
            violation
            for path, module in _FLOW_RUNTIME_BOUNDARY_MODULES.items()
            for violation in _runtime_boundary_violations(
                Path(path),
                Path(module.__file__).read_text(encoding="utf-8"),
            )
        )

        self.assertEqual(violations, ())

    def test_runtime_boundary_audit_rejects_authoring_wrappers(self) -> None:
        violations = _runtime_boundary_violations(
            Path("runtime.py"),
            """from asyncio import run, to_thread
from avalan.flow.authoring import compile_flow_source
from avalan.flow.loader import FlowDefinitionLoader

async def bad(path, loop):
    await compile_flow_source("private")
    loader = FlowDefinitionLoader()
    path.read_text()
    asyncio.run(loader.loads("private"))
    loop.run_until_complete(loader.loads("private"))
    return to_thread(path.write_text, "private")
""",
        )

        self.assertEqual(
            violations,
            (
                "runtime.py:1 imports run",
                "runtime.py:1 imports to_thread",
                "runtime.py:10 calls run_until_complete",
                "runtime.py:11 calls to_thread",
                "runtime.py:2 imports compile_flow_source",
                "runtime.py:3 imports FlowDefinitionLoader",
                "runtime.py:6 calls compile_flow_source",
                "runtime.py:7 calls FlowDefinitionLoader",
                "runtime.py:8 calls read_text",
                "runtime.py:9 calls run",
            ),
        )


async def runtime_flow_adder(a: int, b: int) -> int:
    return a + b


def _assert_recorded_duration(
    test_case: TestCase,
    duration_ms: int | float | None,
) -> None:
    test_case.assertIsNotNone(duration_ms)
    assert duration_ms is not None
    test_case.assertGreaterEqual(duration_ms, 0)


class FlowPlanExecutionTestCase(IsolatedAsyncioTestCase):
    async def test_flow_node_registry_runner_caches_native_nodes(self) -> None:
        runner = FlowNodeRegistryRunner()
        node = FlowNodePlan(
            name="echo",
            type="echo",
            kind=FlowNodeKind.PASS_THROUGH,
        )

        self.assertEqual(await runner(node, {"value": "first"}), "first")
        self.assertEqual(await runner(node, {"value": "second"}), "second")
        self.assertEqual(len(runner._nodes), 1)

    async def test_flow_node_registry_runner_validates_arguments(self) -> None:
        with self.assertRaises(AssertionError):
            FlowNodeRegistryRunner(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            flow_node_registry_runner(object())  # type: ignore[arg-type]

    async def test_execute_flow_plan_scopes_nested_events_to_node(
        self,
    ) -> None:
        items: list[CanonicalStreamItem] = []
        node = FlowNodePlan(
            name="agent_node",
            type="agent",
            kind=FlowNodeKind.AGENT,
            output_contracts=(
                FlowNodeContract(name="value", type=FlowOutputType.JSON),
            ),
        )
        plan = self._plan(
            entry_node="agent_node",
            outputs={"answer": "agent_node.value"},
            nodes=(node,),
        )

        async def runner(
            _: FlowNodePlan,
            _inputs: Mapping[str, object],
        ) -> str:
            options = flow_runtime_module._FLOW_EXECUTION_OPTIONS.get()
            assert options is not None
            assert options.event_listener is not None
            assert options.stream_session is not None
            result = options.event_listener(
                canonical_flow_item(
                    stream_session=options.stream_session,
                    event_type=EventType.FLOW_NODE_STARTED,
                    payload={
                        "flow_id": "child-flow",
                        "node": "child_agent",
                        "status": "started",
                    },
                )
            )
            if result is not None:
                await result
            return "ok"

        result = await execute_flow_plan(
            plan,
            runner,
            event_listener=items.append,
        )
        child_items = [
            item
            for item in items
            if item.metadata["event_type"] == EventType.FLOW_NODE_STARTED.value
            and cast(Mapping[str, object], item.data)["node"] == "child_agent"
        ]

        self.assertTrue(result.ok)
        self.assertEqual(len(child_items), 1)
        self.assertEqual(
            cast(Mapping[str, object], child_items[0].data)["flow_node"],
            "agent_node",
        )
        self.assertEqual(
            child_items[0].metadata["parent_node_id"],
            "agent_node",
        )
        self.assertEqual(
            child_items[0].correlation.node_id,
            "child_agent",
        )

    async def test_flow_node_registry_runner_handles_subflows(self) -> None:
        runner = FlowNodeRegistryRunner()
        subflow = self._plan(
            entry_node="echo",
            outputs={"answer": "echo.value"},
            nodes=(
                FlowNodePlan(
                    name="echo",
                    type="echo",
                    kind=FlowNodeKind.PASS_THROUGH,
                    mappings=(
                        FlowMappingPlan(
                            target="value",
                            kind=FlowMappingKind.SELECT,
                            source=parse_flow_selector("inputs.value"),
                        ),
                    ),
                    output_contracts=(
                        FlowNodeContract(
                            name="value",
                            type=FlowOutputType.JSON,
                        ),
                    ),
                ),
            ),
        )
        node = FlowNodePlan(
            name="child",
            type="subflow",
            kind=FlowNodeKind.SUBFLOW,
            output_contracts=(
                FlowNodeContract(name="value", type=FlowOutputType.JSON),
            ),
            metadata={
                "subflow": {
                    "plan": subflow,
                    "output_mapping": {"value": "answer"},
                }
            },
        )
        malformed = FlowNodePlan(
            name="malformed",
            type="subflow",
            kind=FlowNodeKind.SUBFLOW,
        )
        malformed_mapping = FlowNodePlan(
            name="malformed_mapping",
            type="subflow",
            kind=FlowNodeKind.SUBFLOW,
            metadata={
                "subflow": {
                    "plan": subflow,
                    "output_mapping": "answer",
                }
            },
        )
        failing = FlowNodePlan(
            name="failing",
            type="subflow",
            kind=FlowNodeKind.SUBFLOW,
            metadata={
                "subflow": {
                    "plan": self._plan(
                        entry_node="echo",
                        outputs={"answer": "missing.value"},
                        nodes=(self._node("echo"),),
                    ),
                    "output_mapping": {"value": "answer"},
                }
            },
        )

        self.assertEqual(
            await runner(node, {"value": "ready"}),
            {"value": "ready"},
        )
        with self.assertRaises(FlowNodeExecutionError) as malformed_error:
            await runner(malformed, {})
        with self.assertRaises(FlowNodeExecutionError) as mapping_error:
            await runner(malformed_mapping, {})
        with self.assertRaises(FlowNodeExecutionError) as failing_error:
            await runner(failing, {})

        self.assertEqual(
            malformed_error.exception.code,
            "flow.execution.subflow_unavailable",
        )
        self.assertEqual(
            mapping_error.exception.code,
            "flow.execution.subflow_unavailable",
        )
        self.assertEqual(
            failing_error.exception.code,
            "flow.execution.subflow_failed",
        )

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
        durations = {
            trace.node: trace.duration_ms for trace in result.trace.nodes
        }
        _assert_recorded_duration(self, durations["declared"])
        self.assertIsNone(durations["inferred"])
        self.assertIsNone(durations["terminal"])
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_execute_flow_plan_preserves_tool_envelope_outputs(
        self,
    ) -> None:
        envelope: Mapping[str, object] = {
            "status": "diagnostic",
            "call_id": "call-1",
            "canonical_name": "flow_adder",
            "result": None,
            "error": None,
            "diagnostic": {
                "code": "tool.invalid_arguments",
                "retryable": False,
            },
        }
        node = FlowNodePlan(
            name="tool",
            type="tool",
            kind=FlowNodeKind.TOOL,
            config={"output_mode": "envelope"},
            output_contracts=(
                FlowNodeContract(name="result", type=FlowOutputType.JSON),
            ),
        )
        plan = self._plan(
            entry_node="tool",
            outputs={
                "status": "tool.status",
                "diagnostic_code": "tool.diagnostic.code",
                "result": "tool.result",
            },
            nodes=(node,),
        )

        result = await execute_flow_plan(
            plan,
            lambda _node, _inputs: dict(envelope),
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(
            result.outputs,
            {
                "status": "diagnostic",
                "diagnostic_code": "tool.invalid_arguments",
                "result": None,
            },
        )
        self.assertEqual(result.node_outputs["tool"]["status"], "diagnostic")
        self.assertEqual(
            result.node_outputs["tool"]["diagnostic"],
            {"code": "tool.invalid_arguments", "retryable": False},
        )

    async def test_execute_flow_plan_preserves_raw_pipeline_stage_output(
        self,
    ) -> None:
        output = (
            "tool: shell.pipeline\n"
            "status: policy_denied\n"
            "stage_count: 2\n"
            "stage_chain: cat | wc\n"
            "\n"
            "stages:\n"
            "- index: 0\n"
            "  command: cat\n"
            "- index: 1\n"
            "  command: wc\n"
        )
        node = FlowNodePlan(
            name="pipeline",
            type="tool",
            kind=FlowNodeKind.TOOL,
            config={"output_mode": "raw"},
            output_contracts=(
                FlowNodeContract(name="result", type=FlowOutputType.JSON),
            ),
            metadata={"private": "private-shell-root"},
        )
        plan = self._plan(
            entry_node="pipeline",
            outputs={"answer": "pipeline.result"},
            nodes=(node,),
        )

        result = await execute_flow_plan(
            plan,
            lambda _node, _inputs: output,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": output})
        self.assertIn("stage_chain: cat | wc", result.outputs["answer"])
        self.assertNotIn(
            "private-shell-root",
            str(result.trace.as_public_dict()),
        )
        self.assertNotIn(
            "private-shell-root",
            str(result.public_diagnostics),
        )

    async def test_execute_flow_plan_keeps_raw_tool_contract_pruning(
        self,
    ) -> None:
        node = FlowNodePlan(
            name="tool",
            type="tool",
            kind=FlowNodeKind.TOOL,
            config={"output_mode": "raw"},
            output_contracts=(
                FlowNodeContract(name="result", type=FlowOutputType.JSON),
            ),
        )
        plan = self._plan(
            entry_node="tool",
            outputs={"result": "tool.result"},
            nodes=(node,),
        )

        result = await execute_flow_plan(
            plan,
            lambda _node, _inputs: {
                "status": "diagnostic",
                "result": "raw value",
            },
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"result": "raw value"})
        self.assertEqual(result.node_outputs["tool"], {"result": "raw value"})

    async def test_execute_flow_plan_propagates_subflow_cancellation_context(
        self,
    ) -> None:
        events = _FlowEventCollector()
        cancellation_checks = 0
        child_plan = replace(
            self._plan(
                entry_node="first",
                outputs={"answer": "second.value"},
                nodes=(
                    FlowNodePlan(
                        name="first",
                        type="echo",
                        kind=FlowNodeKind.PASS_THROUGH,
                        output_contracts=(
                            FlowNodeContract(
                                name="value",
                                type=FlowOutputType.JSON,
                            ),
                        ),
                    ),
                    FlowNodePlan(
                        name="second",
                        type="echo",
                        kind=FlowNodeKind.PASS_THROUGH,
                        output_contracts=(
                            FlowNodeContract(
                                name="value",
                                type=FlowOutputType.JSON,
                            ),
                        ),
                    ),
                ),
                edges=(
                    FlowEdgePlan(
                        index=0,
                        source="first",
                        target="second",
                        kind=FlowEdgeKind.SUCCESS,
                    ),
                ),
            ),
            name="child-runtime",
        )
        node = FlowNodePlan(
            name="child",
            type="subflow",
            kind=FlowNodeKind.SUBFLOW,
            output_contracts=(
                FlowNodeContract(name="result", type=FlowOutputType.JSON),
            ),
            metadata={
                "subflow": {
                    "plan": child_plan,
                    "output_mapping": {"result": "answer"},
                }
            },
        )
        plan = self._plan(
            entry_node="child",
            outputs={"answer": "child.result"},
            nodes=(node,),
        )

        async def check_cancelled() -> None:
            nonlocal cancellation_checks
            cancellation_checks += 1
            if cancellation_checks >= 5:
                raise CancelledError("stop inside child")

        session = flow_stream_session(
            stream_session_id="parent-flow-session",
            run_id="parent-flow-run",
            turn_id="parent-flow-turn",
            cancellation_checker=check_cancelled,
        )
        result = await execute_flow_plan(
            plan,
            FlowNodeRegistryRunner(),
            event_listener=events.append,
            stream_session=session,
        )
        child_events = [
            event
            for event in events
            if cast(Mapping[str, object], event.payload)["flow_id"]
            == "child-runtime"
        ]
        child_started_nodes = [
            cast(Mapping[str, object], event.payload)["node"]
            for event in child_events
            if event.type == EventType.FLOW_NODE_STARTED
        ]

        self.assertFalse(result.ok)
        self.assertEqual(
            self._node_states(result)["child"], FlowNodeState.CANCELLED
        )
        self.assertIn(
            "flow.execution.node_cancelled",
            [diagnostic.code for diagnostic in result.diagnostics],
        )
        self.assertIn("first", child_started_nodes)
        self.assertNotIn("second", child_started_nodes)
        self.assertIn(
            EventType.FLOW_CANCELLED,
            [event.type for event in child_events],
        )
        self.assertTrue(session.cancelled)
        self.assertTrue(
            all(
                event.observability.data["stream_session_id"]
                == "parent-flow-session"
                for event in child_events
            )
        )
        self.assertGreaterEqual(cancellation_checks, 3)

    async def test_execute_flow_plan_emits_branch_events_safely(self) -> None:
        events = _FlowEventCollector()

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            if node.name == "source":
                return {
                    "status": "ready",
                    "private": "private node output",
                }
            return node.name

        ready = self._condition(
            FlowConditionOperator.EQ,
            selector="source.value.status",
            value="ready",
        )
        skipped = self._condition(
            FlowConditionOperator.EQ,
            selector="source.value.status",
            value="skipped",
        )
        plan = self._plan(
            entry_node="source",
            outputs={"answer": "left.value"},
            nodes=(
                self._node("source"),
                self._node("fallback"),
                self._node("left"),
                self._node("right"),
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
                    target="left",
                    kind=FlowEdgeKind.SUCCESS,
                    condition=ready,
                    priority=0,
                ),
                FlowEdgePlan(
                    index=2,
                    source="source",
                    target="right",
                    kind=FlowEdgeKind.SUCCESS,
                    condition=skipped,
                    priority=1,
                ),
            ),
        )

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={"payload": {"prompt": "private prompt"}},
            event_listener=events.append,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "left"})
        self.assertTrue(
            all(
                cast(Mapping[str, object], event.payload)["flow_id"]
                == "runtime"
                for event in events
            )
        )
        self.assertEqual(
            [event.type for event in events],
            [
                EventType.FLOW_VALIDATION,
                EventType.FLOW_STARTED,
                EventType.FLOW_NODE_STARTED,
                EventType.FLOW_NODE_COMPLETED,
                EventType.FLOW_CONDITION_EVALUATED,
                EventType.FLOW_EDGE_ELIGIBLE,
                EventType.FLOW_CONDITION_EVALUATED,
                EventType.FLOW_EDGE_ELIGIBLE,
                EventType.FLOW_EDGE_ROUTED,
                EventType.FLOW_EDGE_ELIGIBLE,
                EventType.FLOW_EDGE_ROUTED,
                EventType.FLOW_NODE_STARTED,
                EventType.FLOW_NODE_COMPLETED,
                EventType.FLOW_NODE_SKIPPED,
                EventType.FLOW_NODE_SKIPPED,
                EventType.FLOW_OUTPUT_SELECTED,
                EventType.FLOW_COMPLETED,
            ],
        )
        condition_payloads = [
            cast(Mapping[str, object], event.payload)
            for event in events
            if event.type == EventType.FLOW_CONDITION_EVALUATED
        ]
        self.assertEqual(
            [payload["matched"] for payload in condition_payloads],
            [True, False],
        )
        routed_payloads = [
            cast(Mapping[str, object], event.payload)
            for event in events
            if event.type == EventType.FLOW_EDGE_ROUTED
        ]
        self.assertEqual(
            [
                (payload["edge_index"], payload["status"])
                for payload in routed_payloads
            ],
            [(1, "taken"), (0, "suppressed")],
        )
        self.assertNotIn("private prompt", str(events))
        self.assertNotIn("private node output", str(events))

    async def test_execute_flow_plan_projects_flow_events_to_canonical_items(
        self,
    ) -> None:
        events = _FlowEventCollector()

        result = await execute_flow_plan(
            self._plan(
                entry_node="start",
                outputs={"answer": "start.value"},
                nodes=(self._node("start"),),
            ),
            lambda _node, _inputs: "ok",
            event_listener=events.append,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        observability = [event.observability for event in events]
        self.assertTrue(
            all(
                payload.kind is EventPayloadKind.CANONICAL_STREAM
                for payload in observability
            )
        )
        self.assertEqual(
            [payload.data["sequence"] for payload in observability],
            list(range(len(events))),
        )
        self.assertTrue(
            all(
                payload.data["kind"] == StreamItemKind.FLOW_EVENT.value
                for payload in observability
            )
        )
        self.assertTrue(
            all(
                payload.data["channel"] == StreamChannel.FLOW.value
                for payload in observability
            )
        )
        node_started = next(
            event
            for event in events
            if event.type is EventType.FLOW_NODE_STARTED
        )
        self.assertEqual(
            node_started.observability.data["correlation"],
            {"flow_run_id": "runtime", "node_id": "start"},
        )
        self.assertEqual(
            cast(Mapping[str, object], node_started.payload)["status"],
            "started",
        )

    async def test_flow_stream_recorder_keeps_exact_and_coalesced_items(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []
        listener = flow_stream_recorder(delivered.append)

        for sequence, event_type, status in (
            (0, EventType.FLOW_NODE_STARTED, "started"),
            (1, EventType.FLOW_NODE_RETRYING, "retrying"),
            (2, EventType.FLOW_NODE_COMPLETED, "succeeded"),
        ):
            result = listener(
                _test_flow_item(
                    event_type,
                    payload={
                        "flow_id": "flow",
                        "node": "worker",
                        "status": status,
                    },
                    sequence=sequence,
                )
            )
            if result is not None:
                await result

        self.assertEqual(
            [item.sequence for item in listener.items],
            [0, 1, 2],
        )
        self.assertEqual(len(delivered), 3)
        self.assertEqual(len(listener.ui_items), 1)
        assert isinstance(listener.ui_items[0].data, Mapping)
        self.assertEqual(listener.ui_items[0].data["status"], "succeeded")
        self.assertEqual(
            listener.ui_items[0].correlation.node_id,
            "worker",
        )
        self.assertEqual(
            cast(Mapping[str, object], delivered[-1].data)["status"],
            "succeeded",
        )

    async def test_flow_stream_recorder_bounds_exact_history(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []
        listener = flow_stream_recorder(
            delivered.append,
            history_item_limit=2,
        )

        for index in range(4):
            result = listener(
                _test_flow_item(
                    EventType.FLOW_NODE_STARTED,
                    payload={
                        "flow_id": "flow",
                        "node": f"node-{index}",
                        "status": "started",
                    },
                    sequence=index,
                )
            )
            if result is not None:
                await result

        self.assertEqual(
            [item.sequence for item in listener.items],
            [2, 3],
        )
        self.assertEqual(
            [item.sequence for item in listener.ui_items],
            [2, 3],
        )
        self.assertEqual(
            [item.sequence for item in delivered],
            [0, 1, 2, 3],
        )

    async def test_flow_stream_recorder_inserts_out_of_order_items(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []
        listener = flow_stream_recorder(delivered.append)

        for sequence in (2, 0, 1):
            result = listener(
                _test_flow_item(
                    EventType.FLOW_NODE_STARTED,
                    payload={
                        "flow_id": "flow",
                        "node": f"node-{sequence}",
                        "status": "started",
                    },
                    sequence=sequence,
                )
            )
            if result is not None:
                await result

        self.assertEqual(
            [item.sequence for item in delivered],
            [2, 0, 1],
        )
        self.assertEqual(
            [item.sequence for item in listener.items],
            [0, 1, 2],
        )
        self.assertEqual(
            [item.sequence for item in listener.ui_items],
            [0, 1, 2],
        )

    async def test_flow_stream_recorder_rejects_duplicate_retained_sequence(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []
        listener = flow_stream_recorder(delivered.append)

        first = _test_flow_item(
            EventType.FLOW_NODE_STARTED,
            payload={
                "flow_id": "flow",
                "node": "worker",
                "status": "started",
            },
            sequence=0,
        )
        result = listener(first)
        if result is not None:
            await result

        with self.assertRaises(AssertionError):
            listener(
                _test_flow_item(
                    EventType.FLOW_NODE_COMPLETED,
                    payload={
                        "flow_id": "flow",
                        "node": "worker",
                        "status": "succeeded",
                    },
                    sequence=0,
                )
            )

        self.assertEqual(
            [item.sequence for item in listener.items],
            [0],
        )
        self.assertEqual(
            [item.sequence for item in delivered],
            [0, 0],
        )

    async def test_flow_stream_recorder_allows_evicted_sequence_reuse(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []
        listener = flow_stream_recorder(
            delivered.append,
            history_item_limit=2,
        )

        for sequence in (10, 12, 11, 10):
            result = listener(
                _test_flow_item(
                    EventType.FLOW_NODE_STARTED,
                    payload={
                        "flow_id": "flow",
                        "node": f"node-{sequence}",
                        "status": "started",
                    },
                    sequence=sequence,
                )
            )
            if result is not None:
                await result

        self.assertEqual(
            [item.sequence for item in delivered],
            [10, 12, 11, 10],
        )
        self.assertEqual(
            [item.sequence for item in listener.items],
            [11, 12],
        )
        self.assertEqual(
            [item.sequence for item in listener.ui_items],
            [11, 12],
        )

    async def test_flow_stream_recorder_does_not_retain_evicted_ui_item(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []
        listener = flow_stream_recorder(
            delivered.append,
            history_item_limit=2,
        )

        cases = (
            (10, "worker"),
            (12, "other"),
            (9, "worker"),
        )
        for sequence, node in cases:
            result = listener(
                _test_flow_item(
                    EventType.FLOW_NODE_STARTED,
                    payload={
                        "flow_id": "flow",
                        "node": node,
                        "status": "started",
                    },
                    sequence=sequence,
                )
            )
            if result is not None:
                await result

        self.assertEqual(
            [item.sequence for item in delivered],
            [10, 12, 9],
        )
        self.assertEqual(
            [item.sequence for item in listener.items],
            [10, 12],
        )
        self.assertEqual(
            [item.sequence for item in listener.ui_items],
            [10, 12],
        )

    async def test_flow_stream_recorder_prunes_stale_ui_items_over_limit(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []
        retained_items = [
            _test_flow_item(
                EventType.FLOW_NODE_STARTED,
                payload={
                    "flow_id": "flow",
                    "node": "first",
                    "status": "started",
                },
                sequence=0,
            ),
            _test_flow_item(
                EventType.FLOW_NODE_STARTED,
                payload={
                    "flow_id": "flow",
                    "node": "kept",
                    "status": "started",
                },
                sequence=1,
            ),
        ]
        stale_item = _test_flow_item(
            EventType.FLOW_NODE_STARTED,
            payload={
                "flow_id": "flow",
                "node": "stale",
                "status": "started",
            },
            sequence=2,
        )

        def key(
            item: CanonicalStreamItem,
        ) -> tuple[str | None, str | None, str]:
            return (
                item.correlation.flow_run_id,
                item.correlation.node_id,
                "flow_node_progress",
            )

        listener = FlowStreamRecorder(
            delivered.append,
            history_item_limit=2,
            _items=retained_items.copy(),
            _retained_sequences={0, 1},
            _ui_items={
                key(retained_items[0]): retained_items[0],
                key(retained_items[1]): retained_items[1],
                key(stale_item): stale_item,
            },
        )

        result = listener(
            _test_flow_item(
                EventType.FLOW_NODE_STARTED,
                payload={
                    "flow_id": "flow",
                    "node": "new",
                    "status": "started",
                },
                sequence=3,
            )
        )
        if result is not None:
            await result

        self.assertEqual([item.sequence for item in listener.items], [1, 3])
        self.assertEqual([item.sequence for item in listener.ui_items], [2, 3])

    def test_flow_stream_recorder_validates_seeded_retained_sequences(
        self,
    ) -> None:
        retained_item = _test_flow_item(
            EventType.FLOW_NODE_STARTED,
            payload={
                "flow_id": "flow",
                "node": "worker",
                "status": "started",
            },
            sequence=0,
        )

        listener = FlowStreamRecorder(
            lambda _item: None,
            _items=[retained_item],
            _retained_sequences={0},
        )

        self.assertEqual([item.sequence for item in listener.items], [0])
        with self.assertRaises(AssertionError):
            FlowStreamRecorder(
                lambda _item: None,
                _items=[retained_item],
                _retained_sequences={1},
            )

    async def test_flow_stream_recorder_allows_zero_history(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []
        listener = flow_stream_recorder(
            delivered.append,
            history_item_limit=0,
        )

        result = listener(
            _test_flow_item(
                EventType.FLOW_NODE_STARTED,
                payload={
                    "flow_id": "flow",
                    "node": "worker",
                    "status": "started",
                },
            )
        )
        if result is not None:
            await result

        self.assertEqual(listener.items, ())
        self.assertEqual(listener.ui_items, ())
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].sequence, 0)

    async def test_flow_stream_recorder_records_after_delivery(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []

        async def failing_listener(item: CanonicalStreamItem) -> None:
            delivered.append(item)
            await sleep(0)
            raise RuntimeError("listener failed")

        listener = flow_stream_recorder(failing_listener)

        result = listener(
            _test_flow_item(
                EventType.FLOW_NODE_STARTED,
                payload={
                    "flow_id": "flow",
                    "node": "worker",
                    "status": "started",
                },
            )
        )
        assert result is not None
        with self.assertRaisesRegex(RuntimeError, "listener failed"):
            await result

        self.assertEqual(listener.items, ())
        self.assertEqual(listener.ui_items, ())
        self.assertEqual(delivered[0].sequence, 0)

        async def accepting_listener(item: CanonicalStreamItem) -> None:
            delivered.append(item)
            await sleep(0)

        listener.downstream = accepting_listener
        result = listener(
            _test_flow_item(
                EventType.FLOW_NODE_COMPLETED,
                payload={
                    "flow_id": "flow",
                    "node": "worker",
                    "status": "succeeded",
                },
                sequence=1,
            )
        )
        assert result is not None
        await result

        self.assertEqual(
            [item.sequence for item in listener.items],
            [1],
        )
        self.assertEqual(
            [item.sequence for item in delivered],
            [0, 1],
        )
        self.assertEqual(
            cast(Mapping[str, object], listener.ui_items[0].data)["status"],
            "succeeded",
        )

    async def test_flow_stream_recorder_records_concurrent_items(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []

        async def accepting_listener(item: CanonicalStreamItem) -> None:
            delivered.append(item)
            await sleep(0)

        listener = flow_stream_recorder(accepting_listener)

        first = listener(
            _test_flow_item(
                EventType.FLOW_NODE_STARTED,
                payload={
                    "flow_id": "flow",
                    "node": "worker",
                    "status": "started",
                },
            )
        )
        second = listener(
            _test_flow_item(
                EventType.FLOW_NODE_COMPLETED,
                payload={
                    "flow_id": "flow",
                    "node": "worker",
                    "status": "succeeded",
                },
                sequence=1,
            )
        )

        assert first is not None
        assert second is not None
        await gather(first, second)

        self.assertEqual(
            [item.sequence for item in delivered],
            [0, 1],
        )
        self.assertEqual(
            [item.sequence for item in listener.items],
            [0, 1],
        )
        self.assertEqual(
            cast(Mapping[str, object], listener.ui_items[0].data)["status"],
            "succeeded",
        )

    async def test_flow_stream_recorder_skips_sync_failure_record(
        self,
    ) -> None:
        def failing_listener(_item: CanonicalStreamItem) -> None:
            raise RuntimeError("listener failed")

        listener = flow_stream_recorder(failing_listener)

        with self.assertRaisesRegex(RuntimeError, "listener failed"):
            listener(
                _test_flow_item(
                    EventType.FLOW_NODE_STARTED,
                    payload={
                        "flow_id": "flow",
                        "node": "worker",
                        "status": "started",
                    },
                )
            )

        self.assertEqual(listener.items, ())
        self.assertEqual(listener.ui_items, ())

    async def test_flow_stream_recorder_coalesces_edge_route_and_custom_groups(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []
        listener = flow_stream_recorder(delivered.append)

        for sequence, event_type, status in (
            (0, EventType.FLOW_EDGE_ELIGIBLE, "eligible"),
            (1, EventType.FLOW_EDGE_ROUTED, "taken"),
            (2, EventType.FLOW_CONDITION_EVALUATED, "matched"),
            (3, "flow_custom", "observed"),
        ):
            result = listener(
                _test_flow_item(
                    event_type,
                    payload={
                        "flow_id": "flow",
                        "node": "worker",
                        "status": status,
                    },
                    sequence=sequence,
                )
            )
            if result is not None:
                await result

        self.assertEqual(
            [
                cast(Mapping[str, object], item.data)["status"]
                for item in listener.ui_items
            ],
            ["taken", "matched", "observed"],
        )
        self.assertEqual([item.sequence for item in delivered], [0, 1, 2, 3])

    async def test_execute_flow_plan_emits_join_events(self) -> None:
        events = _FlowEventCollector()

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            if node.name in {"left", "right"}:
                return {"value": {"node": node.name}}
            return node.name

        result = await execute_flow_plan(
            self._join_plan(FlowJoinPlan(type=FlowJoinPolicyType.ALL_SUCCESS)),
            runner,
            event_listener=events.append,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        join_payloads = [
            cast(Mapping[str, object], event.payload)
            for event in events
            if event.type == EventType.FLOW_JOIN_READY
        ]
        self.assertEqual(
            [
                (payload["node"], payload["status"])
                for payload in join_payloads
            ],
            [("joined", "waiting"), ("joined", "ready")],
        )

    async def test_execute_flow_plan_emits_retry_and_fallback_events(
        self,
    ) -> None:
        events = _FlowEventCollector()
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
                self._node("fallback"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="fallback",
                    kind=FlowEdgeKind.ERROR,
                ),
            ),
        )

        result = await execute_flow_plan(
            plan,
            runner,
            event_listener=events.append,
        )

        self.assertFalse(result.ok)
        self.assertEqual(calls, ["start", "start", "fallback"])
        retry_payload = next(
            cast(Mapping[str, object], event.payload)
            for event in events
            if event.type == EventType.FLOW_NODE_RETRYING
        )
        failure_payload = next(
            cast(Mapping[str, object], event.payload)
            for event in events
            if event.type == EventType.FLOW_NODE_FAILED
        )
        routed_payload = next(
            cast(Mapping[str, object], event.payload)
            for event in events
            if event.type == EventType.FLOW_EDGE_ROUTED
        )
        self.assertEqual(retry_payload["attempt"], 1)
        self.assertEqual(
            retry_payload["diagnostic_codes"],
            ["flow.execution.provider_unavailable"],
        )
        self.assertEqual(failure_payload["attempts"], 2)
        self.assertEqual(routed_payload["status"], "taken")
        self.assertEqual(routed_payload["edge_kind"], "error")

    async def test_execute_flow_plan_emits_loop_condition_events(
        self,
    ) -> None:
        events = _FlowEventCollector()
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "repair":
                count = len(calls)
                return {
                    "done": count == 2,
                    "more": count < 2,
                    "safe": {"attempts": count},
                    "private": "private repair output",
                }
            return {"value": node.name}

        result = await execute_flow_plan(
            self._loop_plan(),
            runner,
            event_listener=events.append,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(calls, ["repair", "repair", "finished"])
        condition_payloads = [
            cast(Mapping[str, object], event.payload)
            for event in events
            if event.type == EventType.FLOW_CONDITION_EVALUATED
        ]
        self.assertEqual(
            [
                (payload["node"], payload["matched"])
                for payload in condition_payloads
            ],
            [("repair", False), ("repair", True), ("repair", True)],
        )
        self.assertNotIn("private repair output", str(events))

    async def test_execute_flow_plan_emits_pause_and_resume_events(
        self,
    ) -> None:
        pause_events = _FlowEventCollector()
        resume_events = _FlowEventCollector()
        plan = self._human_review_plan()

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            return node.name

        paused = await execute_flow_plan(
            plan,
            runner,
            inputs={"payload": {"summary": "private summary"}},
            event_listener=pause_events.append,
        )
        resumed = await execute_flow_plan(
            plan,
            runner,
            resume_trace=paused.trace,
            resume_node_outputs=paused.node_outputs,
            resume_decisions={"review": {"decision": "approved"}},
            event_listener=resume_events.append,
        )

        self.assertTrue(paused.ok, paused.public_diagnostics)
        self.assertTrue(resumed.ok, resumed.public_diagnostics)
        pause_payload = next(
            cast(Mapping[str, object], event.payload)
            for event in pause_events
            if event.type == EventType.FLOW_NODE_PAUSED
        )
        resume_payload = next(
            cast(Mapping[str, object], event.payload)
            for event in resume_events
            if event.type == EventType.FLOW_NODE_RESUMED
        )
        self.assertEqual(pause_payload["node"], "review")
        self.assertEqual(pause_payload["status"], "paused")
        self.assertEqual(resume_payload["node"], "review")
        self.assertEqual(resume_payload["status"], "resumed")
        self.assertIn(
            EventType.FLOW_COMPLETED, [event.type for event in pause_events]
        )
        self.assertNotIn("private summary", str(pause_events))
        self.assertNotIn("approved", str(resume_events))

    async def test_execute_flow_plan_emits_cancellation_events(self) -> None:
        events = _FlowEventCollector()

        async def cancel() -> None:
            raise CancelledError()

        with self.assertRaises(CancelledError):
            await execute_flow_plan(
                self._plan(
                    entry_node="start",
                    outputs={"answer": "start.value"},
                    nodes=(self._node("start"),),
                ),
                lambda _node, _inputs: "private output",
                cancellation_checker=cancel,
                event_listener=events.append,
            )

        self.assertEqual(
            [event.type for event in events],
            [
                EventType.FLOW_VALIDATION,
                EventType.FLOW_STARTED,
                EventType.FLOW_CANCELLED,
                EventType.FLOW_COMPLETED,
            ],
        )
        self.assertEqual(
            cast(Mapping[str, object], events[-1].payload)["status"],
            "cancelled",
        )
        self.assertEqual(
            events[-2].observability.data["kind"],
            StreamItemKind.FLOW_EVENT.value,
        )
        self.assertEqual(
            events[-1].observability.data["kind"],
            StreamItemKind.FLOW_EVENT.value,
        )
        self.assertNotIn("private output", str(events))

    async def test_execute_flow_plan_uses_stream_session_cancellation(
        self,
    ) -> None:
        events = _FlowEventCollector()
        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
        )
        session.cancel()

        with self.assertRaises(CancelledError):
            await execute_flow_plan(
                self._plan(
                    entry_node="start",
                    outputs={"answer": "start.value"},
                    nodes=(self._node("start"),),
                ),
                lambda _node, _inputs: "private output",
                event_listener=events.append,
                stream_session=session,
            )

        self.assertEqual(
            [event.type for event in events],
            [
                EventType.FLOW_VALIDATION,
                EventType.FLOW_STARTED,
                EventType.FLOW_CANCELLED,
                EventType.FLOW_COMPLETED,
            ],
        )
        self.assertTrue(session.cancelled)
        self.assertTrue(
            all(
                event.observability.data["stream_session_id"] == "flow-session"
                for event in events
            )
        )
        self.assertNotIn(EventType.FLOW_NODE_STARTED, [e.type for e in events])
        self.assertNotIn("private output", str(events))

    async def test_flow_stream_session_marks_legacy_checker_cancellation(
        self,
    ) -> None:
        events = _FlowEventCollector()

        async def cancel() -> None:
            raise CancelledError()

        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
            cancellation_checker=cancel,
        )

        with self.assertRaises(CancelledError):
            await execute_flow_plan(
                self._plan(
                    entry_node="start",
                    outputs={"answer": "start.value"},
                    nodes=(self._node("start"),),
                ),
                lambda _node, _inputs: "private output",
                event_listener=events.append,
                stream_session=session,
            )

        self.assertTrue(session.cancelled)
        self.assertEqual(
            cast(Mapping[str, object], events[-1].payload)["status"],
            "cancelled",
        )

    async def test_flow_stream_session_honors_checker_requested_cancel(
        self,
    ) -> None:
        sessions: dict[str, FlowStreamSession] = {}

        async def cancel() -> None:
            sessions["session"].cancel()

        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
            cancellation_checker=cancel,
        )
        sessions["session"] = session

        with self.assertRaises(CancelledError):
            await session.check_cancelled()
        self.assertTrue(session.cancelled)

    async def test_execute_flow_plan_marks_separate_checker_cancellation(
        self,
    ) -> None:
        events = _FlowEventCollector()
        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
        )

        async def cancel() -> None:
            raise CancelledError()

        with self.assertRaises(CancelledError):
            await execute_flow_plan(
                self._plan(
                    entry_node="start",
                    outputs={"answer": "start.value"},
                    nodes=(self._node("start"),),
                ),
                lambda _node, _inputs: "private output",
                cancellation_checker=cancel,
                event_listener=events.append,
                stream_session=session,
            )

        self.assertTrue(session.cancelled)
        self.assertEqual(
            cast(Mapping[str, object], events[-1].payload)["status"],
            "cancelled",
        )

    def test_flow_stream_session_validates_identity_and_checker(self) -> None:
        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
        )

        self.assertIsInstance(session, FlowStreamSession)
        with self.assertRaises(AssertionError):
            flow_stream_session(
                stream_session_id="",
                run_id="flow-run",
                turn_id="flow-turn",
            )
        with self.assertRaises(AssertionError):
            flow_stream_session(
                stream_session_id="flow-session",
                run_id="flow-run",
                turn_id="flow-turn",
                cancellation_checker=object(),  # type: ignore[arg-type]
            )

    async def test_execute_flow_plan_rejects_bad_stream_session(self) -> None:
        with self.assertRaises(AssertionError):
            await execute_flow_plan(
                self._plan(
                    entry_node="start",
                    outputs={"answer": "start.value"},
                    nodes=(self._node("start"),),
                ),
                lambda _node, _inputs: "private output",
                stream_session=object(),  # type: ignore[arg-type]
            )

    async def test_flow_stream_recorder_rejects_legacy_event_items(
        self,
    ) -> None:
        delivered: list[CanonicalStreamItem] = []
        listener = flow_stream_recorder(delivered.append)

        with self.assertRaises(AssertionError):
            listener(
                Event(  # type: ignore[arg-type]
                    type=EventType.TOKEN_GENERATED,
                    payload={"flow_node": "worker", "count": 1},
                )
            )

        self.assertEqual(delivered, [])

    def test_canonical_flow_item_validates_payload(self) -> None:
        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
        )
        with self.assertRaises(AssertionError):
            canonical_flow_item(
                stream_session=session,
                event_type=EventType.FLOW_STARTED,
                payload="bad",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            canonical_flow_item(
                stream_session=session,
                event_type=EventType.TOKEN_GENERATED,
                payload={},
            )
        with self.assertRaises(AssertionError):
            flow_stream_recorder(
                lambda _item: None,
                history_item_limit=-1,
            )

    def test_canonical_flow_item_normalizes_payloads(self) -> None:
        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
        )
        empty = canonical_flow_item(
            stream_session=session,
            event_type="flow_custom",
            payload={},
            sequence=0,
            started=1.0,
            finished=2.0,
        )
        nested = canonical_flow_item(
            stream_session=session,
            event_type=EventType.FLOW_NODE_STARTED,
            payload={
                "flow_id": "flow",
                "node": "worker",
                "nested": {
                    "object": object(),
                    "values": (1, object()),
                    1: "ignored",
                },
            },
            sequence=1,
        )

        self.assertEqual(empty.data, {})
        self.assertEqual(empty.metadata["event_type"], "flow_custom")
        self.assertEqual(empty.metadata["elapsed"], 1.0)
        self.assertEqual(
            nested.data["nested"],
            {
                "object": {"type": "object"},
                "values": [1, {"type": "object"}],
            },
        )

    def test_canonical_flow_item_duplicates_lifecycle_metadata(self) -> None:
        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
        )
        item = canonical_flow_item(
            stream_session=session,
            event_type=EventType.FLOW_EDGE_ELIGIBLE,
            payload={
                "flow_id": "flow",
                "node": "worker",
                "status": "eligible",
                "state": "routing",
                "attempt": 1,
                "attempts": 2,
                "duration_ms": 3.5,
                "elapsed_ms": 4,
                "route_kind": "success",
                "edge_kind": "success",
                "edge_index": 7,
                "source": "worker",
                "target": "finish",
                "matched": True,
                "eligible": True,
                "ready": False,
                "output_name": "answer",
                "node_count": 2,
                "edge_count": 1,
                "progress": 0.5,
                "progress_percent": 50,
                "diagnostic_codes": ("flow.warning",),
                "private_output": {"secret": "customer"},
            },
            sequence=0,
        )

        expected = {
            "state": "routing",
            "status": "eligible",
            "attempt": 1,
            "attempts": 2,
            "duration_ms": 3.5,
            "elapsed_ms": 4,
            "route_kind": "success",
            "edge_kind": "success",
            "edge_index": 7,
            "source": "worker",
            "target": "finish",
            "matched": True,
            "eligible": True,
            "ready": False,
            "output_name": "answer",
            "node_count": 2,
            "edge_count": 1,
            "progress": 0.5,
            "progress_percent": 50,
        }
        for key, value in expected.items():
            self.assertEqual(item.metadata[key], value)
            self.assertEqual(item.data[key], value)
        self.assertNotIn("diagnostic_codes", item.metadata)
        self.assertNotIn("private_output", item.metadata)
        self.assertEqual(
            item.data["diagnostic_codes"],
            ["flow.warning"],
        )
        aliased = canonical_flow_item(
            stream_session=session,
            event_type=EventType.FLOW_NODE_STARTED,
            payload={
                "flow_id": "flow",
                "node": "worker",
                "status": "started",
            },
            sequence=1,
        )
        self.assertEqual(aliased.metadata["status"], "started")
        self.assertEqual(aliased.metadata["state"], "started")

    def test_canonical_flow_item_ignores_bad_metadata_candidates(self) -> None:
        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
        )
        item = canonical_flow_item(
            stream_session=session,
            event_type=EventType.FLOW_NODE_COMPLETED,
            payload={
                "flow_id": "flow",
                "node": "worker",
                "status": {"nested": "not metadata"},
                "state": ["not metadata"],
                "attempt": True,
                "attempts": -1,
                "duration_ms": float("nan"),
                "elapsed_ms": [1],
                "route_kind": ["success"],
                "edge_kind": {"value": "success"},
                "edge_index": False,
                "source": object(),
                "target": "",
                "matched": "true",
                "eligible": 1,
                "ready": None,
                "output_name": " ",
                "node_count": "2",
                "edge_count": 1.5,
                "progress": float("inf"),
                "progress_percent": -1.0,
            },
            sequence=0,
        )

        for key in (
            "status",
            "state",
            "attempt",
            "attempts",
            "duration_ms",
            "elapsed_ms",
            "route_kind",
            "edge_kind",
            "edge_index",
            "source",
            "target",
            "matched",
            "eligible",
            "ready",
            "output_name",
            "node_count",
            "edge_count",
            "progress",
            "progress_percent",
        ):
            self.assertNotIn(key, item.metadata)
        self.assertEqual(
            item.data["duration_ms"],
            {"type": "float", "value": "nan"},
        )
        dumps(
            {"data": item.data, "metadata": item.metadata},
            allow_nan=False,
        )

    def test_canonical_flow_item_records_parent_sequence(self) -> None:
        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
        )
        item = canonical_flow_item(
            stream_session=session,
            event_type=EventType.FLOW_NODE_STARTED,
            payload={
                "flow_id": "flow",
                "node": "child",
                "status": "started",
            },
            parent_sequence=7,
        )

        self.assertEqual(item.correlation.parent_sequence, 7)

    def test_canonical_flow_item_normalizes_nonfinite_numbers(
        self,
    ) -> None:
        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
        )
        item = canonical_flow_item(
            stream_session=session,
            event_type=EventType.FLOW_NODE_STARTED,
            payload={
                "flow_id": "flow",
                "node": "worker",
                "score": float("nan"),
                "values": (float("inf"), float("-inf"), 1.5),
            },
            sequence=0,
            started=float("inf"),
            finished=float("inf"),
        )

        self.assertEqual(item.data["score"], {"type": "float", "value": "nan"})
        self.assertEqual(
            item.data["values"],
            [
                {"type": "float", "value": "inf"},
                {"type": "float", "value": "-inf"},
                1.5,
            ],
        )
        self.assertEqual(
            item.metadata["started"],
            {"type": "float", "value": "inf"},
        )
        self.assertEqual(
            item.metadata["finished"],
            {"type": "float", "value": "inf"},
        )
        self.assertEqual(
            item.metadata["elapsed"],
            {"type": "float", "value": "nan"},
        )
        dumps(
            {"data": item.data, "metadata": item.metadata},
            allow_nan=False,
        )

    async def test_flow_event_helpers_cover_defensive_paths(self) -> None:
        events = _FlowEventCollector()
        async_events = _FlowEventCollector()
        plan = replace(
            self._plan(
                entry_node="start",
                outputs={"answer": "start.value"},
                nodes=(self._node("start"),),
            ),
            version="v1",
            revision="r1",
        )
        session = flow_stream_session(
            stream_session_id="flow-session",
            run_id="flow-run",
            turn_id="flow-turn",
        )

        self.assertIsNone(_node_outcome_event_type(FlowNodeState.READY))
        await _emit_node_outcome_event(
            events.append,
            session,
            plan,
            _NodeRunOutcome(
                node=self._node("start"),
                state=FlowNodeState.READY,
                route_kind=FlowEdgeKind.SUCCESS,
                attempts=0,
                diagnostics=(),
            ),
        )

        async def listener(item: CanonicalStreamItem) -> None:
            async_events.append(item)
            await sleep(0)

        await _emit_node_outcome_event(
            listener,
            session,
            plan,
            _NodeRunOutcome(
                node=self._node("start"),
                state=FlowNodeState.SUCCEEDED,
                route_kind=FlowEdgeKind.SUCCESS,
                attempts=1,
                diagnostics=(),
            ),
        )
        _append_node_event_draft(
            None,
            EventType.FLOW_NODE_RESUMED,
            node="start",
            status="resumed",
        )
        _append_condition_event_draft(
            None,
            status="unmatched",
            matched=False,
        )
        _append_join_event_draft(
            None,
            node="joined",
            ready=False,
            status="waiting",
        )

        payload = _flow_event_payload(
            plan,
            {
                "nested": {"value": object()},
                "opaque": object(),
            },
        )

        self.assertEqual(events, [])
        self.assertEqual(
            [event.type for event in async_events],
            [EventType.FLOW_NODE_COMPLETED],
        )
        self.assertEqual(payload["flow_id"], "runtime@v1#r1")
        self.assertEqual(payload["flow_version"], "v1")
        self.assertEqual(payload["flow_revision"], "r1")
        self.assertEqual(payload["nested"], {"value": {"type": "object"}})
        self.assertEqual(payload["opaque"], {"type": "object"})

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
        _assert_recorded_duration(self, result.trace.nodes[0].duration_ms)
        _assert_recorded_duration(self, result.trace.edges[0].duration_ms)

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
        _assert_recorded_duration(self, result.trace.nodes[0].duration_ms)
        _assert_recorded_duration(self, result.trace.edges[0].duration_ms)

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

    async def test_execute_flow_plan_retry_exhaustion_prefers_error_edge(
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
                self._node("fallback"),
                self._node("generic"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="fallback",
                    kind=FlowEdgeKind.SUCCESS,
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
                    target="generic",
                    kind=FlowEdgeKind.ERROR,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "fallback"})
        self.assertEqual(calls, ["start", "start", "fallback"])
        self.assertEqual(
            self._edge_states(result),
            {
                0: FlowEdgeState.PENDING,
                1: FlowEdgeState.TAKEN,
                2: FlowEdgeState.SUPPRESSED,
            },
        )
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.provider_unavailable"],
        )

    async def test_execute_flow_plan_retry_exhaustion_requires_error_edge(
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
                self._node("fallback"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="fallback",
                    kind=FlowEdgeKind.SUCCESS,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(calls, ["start", "start"])
        self.assertEqual(
            self._node_states(result),
            {
                "start": FlowNodeState.FAILED,
                "fallback": FlowNodeState.SKIPPED,
            },
        )
        self.assertEqual(self._edge_states(result), {0: FlowEdgeState.PENDING})
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            [
                "flow.execution.provider_unavailable",
                "flow.execution.missing_failure_route",
                "flow.execution.missing_output",
            ],
        )

    async def test_execute_flow_plan_reports_missing_retry_fallback(
        self,
    ) -> None:
        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
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
            outputs={"answer": "start.value"},
            nodes=(
                self._node(
                    "start",
                    retry=FlowRetryPlan(
                        max_attempts=2,
                        retryable_categories=("transient",),
                        exhausted_route="fallback",
                    ),
                ),
                self._node("fallback"),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            [
                "flow.execution.provider_unavailable",
                "flow.execution.missing_failure_route",
                "flow.execution.missing_output",
            ],
        )
        self.assertEqual(
            self._node_states(result),
            {
                "start": FlowNodeState.FAILED,
                "fallback": FlowNodeState.SKIPPED,
            },
        )

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

    async def test_execute_flow_plan_routes_structured_cancellation_error(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            if node.name == "start":
                raise FlowNodeExecutionError(
                    code="flow.execution.worker_cancelled",
                    message="Flow worker cancelled the node.",
                    hint="Route to the cancellation handler.",
                    failure_category="cancellation",
                    route_kind=FlowEdgeKind.CANCELLATION,
                )
            return node.name

        plan = self._plan(
            entry_node="start",
            outputs={"answer": "cancelled.value"},
            nodes=(
                self._node(
                    "start",
                    retry=FlowRetryPlan(
                        max_attempts=3,
                        retryable_categories=("cancellation",),
                    ),
                ),
                self._node("cancelled"),
                self._node("failed"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="cancelled",
                    kind=FlowEdgeKind.CANCELLATION,
                ),
                FlowEdgePlan(
                    index=1,
                    source="start",
                    target="failed",
                    kind=FlowEdgeKind.ERROR,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "cancelled"})
        self.assertEqual(calls, ["start", "cancelled"])
        self.assertEqual(
            self._node_states(result),
            {
                "start": FlowNodeState.CANCELLED,
                "cancelled": FlowNodeState.SUCCEEDED,
                "failed": FlowNodeState.SKIPPED,
            },
        )
        self.assertEqual(self._node_attempts(result)["start"], 1)
        self.assertEqual(
            self._edge_states(result),
            {
                0: FlowEdgeState.TAKEN,
                1: FlowEdgeState.PENDING,
            },
        )
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.worker_cancelled"],
        )

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

    async def test_execute_flow_plan_loop_limit_prefers_error_edge(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "repair":
                return {"done": False, "more": True, "safe": "pending"}
            return node.name

        plan = self._plan(
            entry_node="repair",
            outputs={"answer": "manual.value"},
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
                        max_iterations=1,
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
                            "repair.result.safe"
                        ),
                        limit_route="manual",
                    ),
                ),
                self._node("manual"),
            ),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="repair",
                    target="manual",
                    kind=FlowEdgeKind.SUCCESS,
                ),
                FlowEdgePlan(
                    index=1,
                    source="repair",
                    target="manual",
                    kind=FlowEdgeKind.ERROR,
                ),
            ),
        )

        result = await execute_flow_plan(plan, runner)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "manual"})
        self.assertEqual(calls, ["repair", "manual"])
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.execution.loop_limit_reached"],
        )
        self.assertEqual(
            self._edge_states(result),
            {0: FlowEdgeState.PENDING, 1: FlowEdgeState.TAKEN},
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

    async def test_execute_flow_plan_stops_when_cancelled_between_nodes(
        self,
    ) -> None:
        calls: list[str] = []
        checks = 0

        async def check_cancelled() -> None:
            nonlocal checks
            checks += 1
            if checks == 2:
                raise CancelledError("private stop detail")

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
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

        with self.assertRaises(CancelledError) as raised:
            await execute_flow_plan(
                plan,
                runner,
                cancellation_checker=check_cancelled,
            )

        self.assertEqual(calls, ["start"])
        self.assertEqual(str(raised.exception), "private stop detail")

    async def test_execute_flow_plan_resumes_from_completed_node_outputs(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
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
        trace = FlowExecutionTrace.from_plan(plan).with_node_state(
            "start",
            FlowNodeState.SUCCEEDED,
            attempts=1,
        )

        result = await execute_flow_plan(
            plan,
            runner,
            resume_trace=trace,
            resume_node_outputs={"start": {"value": "ready"}},
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(calls, ["finish"])
        self.assertEqual(result.outputs, {"answer": "finish"})
        self.assertEqual(self._node_attempts(result)["start"], 1)
        self.assertEqual(
            self._edge_states(result),
            {0: FlowEdgeState.TAKEN},
        )

    async def test_execute_flow_plan_pauses_human_review_without_runner(
        self,
    ) -> None:
        calls: list[str] = []

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            return node.name

        result = await execute_flow_plan(
            self._human_review_plan(),
            runner,
            inputs={"payload": {"summary": "safe"}},
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(calls, [])
        self.assertEqual(result.outputs, {})
        self.assertEqual(set(result.pause_tokens), {"review"})
        self.assertTrue(result.pause_tokens["review"])
        self.assertEqual(
            self._node_states(result),
            {
                "review": FlowNodeState.PAUSED,
                "finish": FlowNodeState.PENDING,
                "rejected": FlowNodeState.PENDING,
            },
        )
        self.assertEqual(
            self._edge_states(result),
            {
                0: FlowEdgeState.PENDING,
                1: FlowEdgeState.PENDING,
            },
        )

    async def test_execute_flow_plan_resumes_human_review_decision(
        self,
    ) -> None:
        calls: list[str] = []
        plan = self._human_review_plan(
            decision_schema={
                "type": "object",
                "required": ["decision", "comment"],
                "properties": {
                    "decision": {"enum": ["approved", "rejected"]},
                    "comment": {"type": "string"},
                },
            }
        )
        paused_trace = FlowExecutionTrace.from_plan(plan).with_node_state(
            "review",
            FlowNodeState.PAUSED,
            attempts=1,
        )

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            resume_trace=paused_trace,
            resume_decisions={
                "review": {"decision": "approved", "comment": "ok"}
            },
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(calls, ["finish"])
        self.assertEqual(result.outputs, {"answer": "finish"})
        review_result = cast(
            Mapping[str, object],
            result.node_outputs["review"]["result"],
        )
        self.assertEqual(review_result["decision"], "approved")
        self.assertEqual(
            self._node_states(result),
            {
                "review": FlowNodeState.SUCCEEDED,
                "finish": FlowNodeState.SUCCEEDED,
                "rejected": FlowNodeState.SKIPPED,
            },
        )
        self.assertEqual(
            self._edge_states(result),
            {
                0: FlowEdgeState.TAKEN,
                1: FlowEdgeState.SUPPRESSED,
            },
        )

    async def test_execute_flow_plan_stops_resume_before_routed_node(
        self,
    ) -> None:
        calls: list[str] = []
        events = _FlowEventCollector()
        plan = self._human_review_plan()
        paused_trace = FlowExecutionTrace.from_plan(plan).with_node_state(
            "review",
            FlowNodeState.PAUSED,
            attempts=1,
        )

        async def cancel() -> None:
            raise CancelledError("private resume stop")

        async def listener(event: Event) -> None:
            events.append(event)

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            return node.name

        with self.assertRaises(CancelledError) as raised:
            await execute_flow_plan(
                plan,
                runner,
                resume_trace=paused_trace,
                resume_decisions={
                    "review": {
                        "decision": "approved",
                        "comment": "private-token",
                    }
                },
                cancellation_checker=cancel,
                event_listener=listener,
            )

        self.assertEqual(str(raised.exception), "private resume stop")
        self.assertEqual(calls, [])
        self.assertIn(
            EventType.FLOW_NODE_RESUMED,
            [event.type for event in events],
        )
        self.assertIn(
            EventType.FLOW_CANCELLED,
            [event.type for event in events],
        )
        self.assertNotIn("private-token", str(events))

    async def test_execute_flow_plan_routes_human_review_decision_labels(
        self,
    ) -> None:
        decisions = {
            "approved": "approved_node",
            "rejected": "rejected_node",
            "needs-correction": "correction_node",
            "expired": "expired_node",
            "escalated": "escalated_node",
        }

        for decision, target in decisions.items():
            with self.subTest(decision=decision):
                calls: list[str] = []
                plan = self._plan(
                    entry_node="review",
                    outputs={"answer": f"{target}.value"},
                    nodes=(
                        FlowNodePlan(
                            name="review",
                            type="human_review",
                            kind=FlowNodeKind.HUMAN_REVIEW,
                            config={
                                "allowed_decisions": tuple(decisions),
                            },
                            output_contracts=(
                                FlowNodeContract(
                                    name="result",
                                    type=FlowOutputType.OBJECT,
                                ),
                            ),
                        ),
                    )
                    + tuple(
                        self._node(node_name)
                        for node_name in decisions.values()
                    ),
                    edges=tuple(
                        FlowEdgePlan(
                            index=index,
                            source="review",
                            target=node_name,
                            kind=FlowEdgeKind.RESUME,
                            label=label,
                        )
                        for index, (label, node_name) in enumerate(
                            decisions.items()
                        )
                    ),
                )
                paused_trace = FlowExecutionTrace.from_plan(
                    plan
                ).with_node_state(
                    "review",
                    FlowNodeState.PAUSED,
                    attempts=1,
                )

                def runner(
                    node: FlowNodePlan,
                    _: Mapping[str, object],
                ) -> object:
                    calls.append(node.name)
                    return node.name

                result = await execute_flow_plan(
                    plan,
                    runner,
                    resume_trace=paused_trace,
                    resume_decisions={"review": {"decision": decision}},
                )

                self.assertTrue(result.ok, result.public_diagnostics)
                self.assertEqual(calls, [target])
                self.assertEqual(result.outputs, {"answer": target})

    async def test_execute_flow_plan_rejects_invalid_review_payload_schema(
        self,
    ) -> None:
        calls: list[str] = []
        plan = self._human_review_plan(
            decision_schema={
                "type": "object",
                "required": ["decision", "reviewer"],
                "additionalProperties": False,
                "properties": {
                    "decision": {"enum": ["approved", "rejected"]},
                    "reviewer": {"type": "string"},
                },
            }
        )
        paused_trace = FlowExecutionTrace.from_plan(plan).with_node_state(
            "review",
            FlowNodeState.PAUSED,
            attempts=1,
        )

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            calls.append(node.name)
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            resume_trace=paused_trace,
            resume_decisions={
                "review": {
                    "decision": "approved",
                    "comment": "private-token",
                }
            },
        )

        self.assertFalse(result.ok)
        self.assertEqual(calls, [])
        self.assertIn(
            "flow.execution.invalid_resume_payload",
            [diagnostic.code for diagnostic in result.diagnostics],
        )
        self.assertNotIn("private-token", str(result.public_diagnostics))

    async def test_execute_flow_plan_rejects_invalid_review_schema(
        self,
    ) -> None:
        cases = (
            "schema",
            {"type": object()},
        )

        for decision_schema in cases:
            with self.subTest(schema=type(decision_schema).__name__):
                plan = self._human_review_plan(decision_schema=decision_schema)
                paused_trace = FlowExecutionTrace.from_plan(
                    plan
                ).with_node_state(
                    "review",
                    FlowNodeState.PAUSED,
                    attempts=1,
                )

                def runner(
                    node: FlowNodePlan,
                    _: Mapping[str, object],
                ) -> object:
                    return node.name

                result = await execute_flow_plan(
                    plan,
                    runner,
                    resume_trace=paused_trace,
                    resume_decisions={
                        "review": {
                            "decision": "approved",
                        }
                    },
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    "flow.execution.invalid_resume_schema",
                    [diagnostic.code for diagnostic in result.diagnostics],
                )
                self.assertNotIn("object at", str(result.public_diagnostics))

    async def test_execute_flow_plan_rejects_missing_schema_support(
        self,
    ) -> None:
        plan = self._human_review_plan(
            decision_schema={
                "type": "object",
                "required": ["decision"],
                "properties": {
                    "decision": {"enum": ["approved", "rejected"]},
                },
            }
        )
        paused_trace = FlowExecutionTrace.from_plan(plan).with_node_state(
            "review",
            FlowNodeState.PAUSED,
            attempts=1,
        )

        def runner(node: FlowNodePlan, _: Mapping[str, object]) -> object:
            return node.name

        with patch("avalan.flow.runtime._json_schema_adapter") as adapter:
            adapter.return_value = None
            result = await execute_flow_plan(
                plan,
                runner,
                resume_trace=paused_trace,
                resume_decisions={"review": {"decision": "approved"}},
            )

        self.assertFalse(result.ok)
        self.assertIn(
            "flow.execution.invalid_resume_schema",
            [diagnostic.code for diagnostic in result.diagnostics],
        )

    async def test_execute_flow_plan_rejects_invalid_human_review_resume(
        self,
    ) -> None:
        cases = (
            (
                {"review": {"comment": "private-token"}},
                "flow.execution.invalid_resume_decision",
            ),
            (
                {"review": {"decision": "escalated"}},
                "flow.execution.unknown_resume_decision",
            ),
            (
                {"finish": {"decision": "approved"}},
                "flow.execution.invalid_resume_node",
            ),
            (
                {"review": {"decision": "approved"}},
                "flow.execution.invalid_resume_state",
            ),
            (
                {"missing": {"decision": "approved"}},
                "flow.execution.unknown_resume_node",
            ),
        )
        plan = self._human_review_plan()
        paused_trace = FlowExecutionTrace.from_plan(plan).with_node_state(
            "review",
            FlowNodeState.PAUSED,
            attempts=1,
        )

        for resume_decisions, code in cases:
            with self.subTest(code=code):
                calls: list[str] = []
                resume_trace = (
                    FlowExecutionTrace.from_plan(plan)
                    if code == "flow.execution.invalid_resume_state"
                    else paused_trace
                )

                def runner(
                    node: FlowNodePlan,
                    _: Mapping[str, object],
                ) -> object:
                    calls.append(node.name)
                    return node.name

                result = await execute_flow_plan(
                    plan,
                    runner,
                    resume_trace=resume_trace,
                    resume_decisions=resume_decisions,
                )

                self.assertFalse(result.ok)
                self.assertEqual(calls, [])
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )
                self.assertNotIn(
                    "private-token",
                    str(result.public_diagnostics),
                )

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
        self.assertEqual(sequential.outputs, concurrent.outputs)
        self.assertEqual(
            self._node_states(sequential),
            self._node_states(concurrent),
        )
        self.assertEqual(
            self._edge_states(sequential),
            self._edge_states(concurrent),
        )

    async def test_execute_flow_plan_routes_ignore_sibling_batch_outputs(
        self,
    ) -> None:
        async def run(
            concurrency_limit: int,
        ) -> tuple[FlowPlanExecutionResult, list[str]]:
            calls: list[str] = []

            async def runner(
                node: FlowNodePlan,
                _: Mapping[str, object],
            ) -> object:
                calls.append(node.name)
                if node.name in {"left", "right"}:
                    await sleep(0)
                return node.name

            result = await execute_flow_plan(
                self._plan(
                    entry_node="source",
                    outputs={"answer": "right.value"},
                    nodes=(
                        self._node("source"),
                        self._node("left"),
                        self._node("right"),
                        self._node("chosen"),
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
                            target="chosen",
                            kind=FlowEdgeKind.SUCCESS,
                            condition=self._condition(
                                FlowConditionOperator.EQ,
                                selector="right.value",
                                value="right",
                            ),
                        ),
                    ),
                ),
                runner,
                concurrency_limit=concurrency_limit,
            )
            return result, calls

        sequential, sequential_calls = await run(1)
        concurrent, concurrent_calls = await run(2)

        self.assertFalse(sequential.ok)
        self.assertFalse(concurrent.ok)
        self.assertEqual(sequential.outputs, {"answer": "right"})
        self.assertEqual(sequential.outputs, concurrent.outputs)
        self.assertEqual(sequential_calls, ["source", "left", "right"])
        self.assertEqual(concurrent_calls, ["source", "left", "right"])
        sequential_codes = [
            diagnostic["code"] for diagnostic in sequential.public_diagnostics
        ]
        self.assertEqual(
            sequential_codes,
            ["flow.condition_missing_value"],
        )
        self.assertEqual(
            sequential.public_diagnostics,
            concurrent.public_diagnostics,
        )
        self.assertEqual(
            self._node_states(sequential),
            self._node_states(concurrent),
        )
        self.assertEqual(
            self._node_states(sequential)["chosen"],
            FlowNodeState.SKIPPED,
        )
        self.assertEqual(
            self._edge_states(sequential),
            {
                0: FlowEdgeState.TAKEN,
                1: FlowEdgeState.TAKEN,
                2: FlowEdgeState.FAILED,
            },
        )
        self.assertEqual(
            self._edge_states(sequential),
            self._edge_states(concurrent),
        )

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
        with self.assertRaises(AssertionError):
            await execute_flow_plan(
                plan,
                runner,
                resume_trace=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            await execute_flow_plan(
                plan,
                runner,
                resume_node_outputs=object(),  # type: ignore[arg-type]
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
            (
                FlowRetryPlan(
                    max_attempts=4,
                    backoff=FlowRetryBackoffStrategy.EXPONENTIAL,
                    initial_delay_seconds=0.5,
                    max_delay_seconds=1,
                ),
                3,
                1,
            ),
        )

        self.assertEqual(_retry_delay_seconds(None, 1), 0)
        for retry, failed_attempts, expected in cases:
            with self.subTest(backoff=retry.backoff.value):
                self.assertEqual(
                    _retry_delay_seconds(retry, failed_attempts),
                    expected,
                )
        retry = FlowRetryPlan(
            max_attempts=2,
            backoff=FlowRetryBackoffStrategy.CONSTANT,
        )
        object.__setattr__(retry, "backoff", "unknown")
        with self.assertRaises(FlowRuntimeEvaluationError) as raised:
            _retry_delay_seconds(retry, 1)
        self.assertEqual(
            raised.exception.code,
            "flow.execution.unsupported_retry_backoff",
        )

    def test_route_from_node_does_not_reselect_finally_edges(self) -> None:
        plan = self._plan(
            entry_node="start",
            outputs={"answer": "cleanup.value"},
            nodes=(self._node("start"), self._node("cleanup")),
            edges=(
                FlowEdgePlan(
                    index=0,
                    source="start",
                    target="cleanup",
                    kind=FlowEdgeKind.FINALLY,
                ),
            ),
        )
        routed, trace, diagnostics = _route_from_node(
            plan,
            "start",
            FlowEdgeKind.FINALLY,
            FlowRuntimeContext(),
            FlowExecutionTrace.from_plan(plan),
        )

        self.assertEqual(routed, ("cleanup",))
        self.assertEqual(diagnostics, ())
        self.assertEqual(
            {edge.index: edge.state for edge in trace.edges},
            {0: FlowEdgeState.TAKEN},
        )

    def test_join_ready_handles_defensive_states(self) -> None:
        plan = self._join_plan(FlowJoinPlan(type=FlowJoinPolicyType.ALL_DONE))
        trace = FlowExecutionTrace.from_plan(plan)
        trace = trace.with_node_state("left", FlowNodeState.READY)
        trace = trace.with_edge_state(2, FlowEdgeState.TAKEN)

        ready, diagnostic = _join_ready(plan, "joined", trace)

        self.assertFalse(ready)
        self.assertIsNone(diagnostic)

    def test_join_ready_reports_unknown_policy(self) -> None:
        plan = self._join_plan(FlowJoinPlan(type=FlowJoinPolicyType.ALL_DONE))
        join = plan.node_map["joined"].join
        assert join is not None
        object.__setattr__(join, "type", "unknown")
        trace = FlowExecutionTrace.from_plan(plan)

        ready, diagnostic = _join_ready(plan, "joined", trace)

        self.assertFalse(ready)
        self.assertIsNotNone(diagnostic)
        assert diagnostic is not None
        self.assertEqual(
            diagnostic.code,
            "flow.execution.unsupported_join_policy",
        )

    def test_human_review_resume_helpers_cover_defensive_paths(self) -> None:
        trace = FlowExecutionTrace.from_plan(self._human_review_plan())
        edge = FlowEdgePlan(
            index=0,
            source="review",
            target="finish",
            kind=FlowEdgeKind.RESUME,
            label="approved",
        )

        self.assertEqual(_node_trace_attempts(trace, "missing"), 0)
        self.assertFalse(
            _resume_label_matches(
                edge,
                FlowRuntimeContext(node_outputs={}),
            )
        )
        self.assertTrue(
            _resume_label_matches(
                edge,
                FlowRuntimeContext(
                    node_outputs={"review": {"decision": "approved"}},
                ),
            )
        )

    def test_json_schema_adapter_helpers_cover_defensive_paths(self) -> None:
        empty_module = ModuleType("empty")
        partial_module = ModuleType("partial")
        partial_module.SchemaError = ValueError

        with patch(
            "avalan.flow.runtime.import_module",
            side_effect=ImportError,
        ):
            self.assertIsNone(_json_schema_adapter())
        self.assertIsNone(_json_schema_adapter_from_module(empty_module))
        self.assertIsNone(_exception_class(empty_module, "SchemaError"))
        self.assertIs(
            _exception_class(partial_module, "SchemaError"),
            ValueError,
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

    def _human_review_plan(
        self,
        *,
        decision_schema: object | None = None,
    ) -> FlowExecutionPlan:
        config: dict[str, object] = {
            "allowed_decisions": ("approved", "rejected"),
        }
        if decision_schema is not None:
            config["decision_schema"] = decision_schema
        return self._plan(
            entry_node="review",
            outputs={"answer": "finish.value"},
            nodes=(
                FlowNodePlan(
                    name="review",
                    type="human_review",
                    kind=FlowNodeKind.HUMAN_REVIEW,
                    config=config,
                    mappings=(
                        FlowMappingPlan(
                            target="payload",
                            kind=FlowMappingKind.SELECT,
                            source=parse_flow_selector("inputs.payload"),
                        ),
                    ),
                    output_contracts=(
                        FlowNodeContract(
                            name="result",
                            type=FlowOutputType.OBJECT,
                        ),
                    ),
                ),
                self._node("finish"),
                self._node("rejected"),
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


class FlowRuntimeEndToEndTestCase(IsolatedAsyncioTestCase):
    async def test_strict_tool_definition_runs_with_registry_runner(
        self,
    ) -> None:
        registry = tool_flow_node_registry(
            ToolManager.create_instance(
                enable_tools=["runtime_flow_adder"],
                available_toolsets=[ToolSet(tools=[runtime_flow_adder])],
                settings=ToolManagerSettings(),
            )
        )
        compile_result = compile_flow_definition(
            FlowDefinition(
                name="runtime-tool-registry",
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
                        ref="runtime_flow_adder",
                        mappings=(
                            FlowInputMapping(
                                target="arguments",
                                kind=FlowMappingKind.OBJECT,
                                fields={
                                    "left": "input.payload.left",
                                    "right": "input.payload.right",
                                },
                            ),
                        ),
                        config={"arguments": {"a": "left", "b": "right"}},
                    ),
                ),
            ),
            registry,
        )
        self.assertTrue(compile_result.ok, compile_result.public_diagnostics)
        assert compile_result.plan is not None

        result = await execute_flow_plan(
            compile_result.plan,
            flow_node_registry_runner(registry),
            inputs={"payload": {"left": 2, "right": 5}},
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs["answer"], 7)
        self.assertEqual(result.node_outputs["calculate"]["result"], 7)

    async def test_loaded_native_definition_runs_with_registry_runner(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-native-registry"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "raw"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "joined.value"

            [nodes.raw]
            type = "input"

            [nodes.raw.mapping]
            value = "input.payload"

            [nodes.checked]
            type = "validation"

            [nodes.checked.config]
            value_type = "object"
            required_fields = ["customer", "approved"]

            [nodes.checked.mapping]
            value = "raw.value"

            [nodes.projected]
            type = "select"

            [nodes.projected.mapping.value]
            type = "object"

            [nodes.projected.mapping.value.fields]
            name = "checked.value.customer.name"
            approved = "checked.value.approved"

            [nodes.decide]
            type = "decision"

            [nodes.decide.mapping]
            value = "projected.value"

            [nodes.defaults]
            type = "constant"

            [nodes.defaults.config.value]
            route = "standard"

            [nodes.notice]
            type = "notification"

            [nodes.notice.config]
            channel = "audit"

            [nodes.notice.mapping]
            value = "decide.value"

            [nodes.joined]
            type = "join"

            [nodes.joined.join_policy]
            type = "all_success"

            [nodes.joined.mapping.value]
            type = "object"

            [nodes.joined.mapping.value.fields]
            customer = "notice.value.payload.name"
            approved = "notice.value.payload.approved"
            route = "defaults.value.route"
            status = "notice.value.status"
            channel = "notice.value.channel"

            [[edges]]
            source = "raw"
            target = "checked"

            [[edges]]
            source = "checked"
            target = "projected"
            routing_policy = "all_matching"

            [[edges]]
            source = "checked"
            target = "defaults"
            routing_policy = "all_matching"

            [[edges]]
            source = "projected"
            target = "decide"

            [[edges]]
            source = "decide"
            target = "notice"

            [edges.condition]
            op = "eq"
            selector = "decide.value.approved"
            value = true

            [[edges]]
            source = "notice"
            target = "joined"

            [[edges]]
            source = "defaults"
            target = "joined"
            """)

        result = await execute_flow_plan(
            plan,
            flow_node_registry_runner(),
            inputs={
                "payload": {
                    "customer": {"name": "Ada"},
                    "approved": True,
                    "private": "input-secret",
                },
            },
            concurrency_limit=2,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(
            result.outputs,
            {
                "answer": {
                    "customer": "Ada",
                    "approved": True,
                    "route": "standard",
                    "status": "notified",
                    "channel": "audit",
                },
            },
        )
        self.assertEqual(result.node_outputs["raw"]["value"]["approved"], True)
        self.assertNotIn("input-secret", str(result.public_diagnostics))

    async def test_loaded_native_validation_failure_is_safe(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-native-validation"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "raw"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "checked.value"

            [nodes.raw]
            type = "input"

            [nodes.raw.mapping]
            value = "input.payload"

            [nodes.checked]
            type = "validation"

            [nodes.checked.config]
            value_type = "object"
            required_fields = ["customer"]

            [nodes.checked.mapping]
            value = "raw.value"

            [[edges]]
            source = "raw"
            target = "checked"
            """)

        result = await execute_flow_plan(
            plan,
            flow_node_registry_runner(),
            inputs={"payload": {"private": "input-secret"}},
        )

        self.assertFalse(result.ok)
        self.assertIn(
            "flow.execution.validation_failed",
            [diagnostic.code for diagnostic in result.diagnostics],
        )
        self.assertNotIn("input-secret", str(result.public_diagnostics))

    async def test_loaded_strict_definition_compiles_and_executes(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-e2e"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "joined.value"

            [nodes.start]
            type = "echo"

            [nodes.start.mapping]
            value = "input.payload"

            [nodes.left]
            type = "select"

            [nodes.left.mapping.value]
            type = "object"

            [nodes.left.mapping.value.fields]
            customer = "start.value.customer.name"
            route = "start.value.route"

            [nodes.right]
            type = "select"

            [nodes.right.mapping.value]
            type = "object"

            [nodes.right.mapping.value.fields]
            score = "start.value.score"

            [nodes.joined]
            type = "select"

            [nodes.joined.join_policy]
            type = "all_success"

            [nodes.joined.mapping.value]
            type = "object"

            [nodes.joined.mapping.value.fields]
            customer = "left.value.customer"
            route = "left.value.route"
            score = "right.value.score"

            [[edges]]
            source = "start"
            target = "left"
            routing_policy = "all_matching"

            [[edges]]
            source = "start"
            target = "right"
            routing_policy = "all_matching"

            [[edges]]
            source = "left"
            target = "joined"

            [[edges]]
            source = "right"
            target = "joined"
            """)
        calls: list[tuple[str, dict[str, object]]] = []

        def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            calls.append((node.name, dict(inputs)))
            return inputs["value"]

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={
                "payload": {
                    "customer": {"name": "Ada"},
                    "route": "standard",
                    "score": 98,
                    "private": "customer-secret",
                },
            },
            concurrency_limit=2,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(
            result.outputs,
            {
                "answer": {
                    "customer": "Ada",
                    "route": "standard",
                    "score": 98,
                },
            },
        )
        self.assertEqual(
            {name for name, _ in calls},
            {"start", "left", "right", "joined"},
        )
        self.assertEqual(
            result.node_outputs["joined"]["value"],
            {
                "customer": "Ada",
                "route": "standard",
                "score": 98,
            },
        )
        called_nodes = {name for name, _ in calls}
        for node_trace in result.trace.nodes:
            if node_trace.node in called_nodes:
                _assert_recorded_duration(self, node_trace.duration_ms)
            else:
                self.assertIsNone(node_trace.duration_ms)
        for edge_trace in result.trace.edges:
            _assert_recorded_duration(self, edge_trace.duration_ms)
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_loaded_definition_uses_declared_entry_and_outputs(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-declared-entry"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "declared"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "declared.value.status"

            [nodes.inferred]
            type = "echo"

            [nodes.declared]
            type = "select"

            [nodes.declared.mapping]
            value = "input.payload"

            [nodes.terminal]
            type = "echo"

            [[edges]]
            source = "inferred"
            target = "terminal"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "declared":
                return inputs["value"]
            return {
                "status": "unused",
                "private": "customer-secret",
            }

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={
                "payload": {
                    "status": "ready",
                    "private": "input-secret",
                },
            },
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "ready"})
        self.assertEqual(calls, ["declared"])
        self.assertEqual(
            {node.node: node.state for node in result.trace.nodes},
            {
                "inferred": FlowNodeState.SKIPPED,
                "declared": FlowNodeState.SUCCEEDED,
                "terminal": FlowNodeState.SKIPPED,
            },
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))
        self.assertNotIn("input-secret", str(result.public_diagnostics))

    async def test_loaded_definition_takes_default_route(self) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-default-route"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "review"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "fallback.value"

            [nodes.review]
            type = "select"

            [nodes.review.mapping]
            value = "input.payload"

            [nodes.approved]
            type = "echo"

            [nodes.fallback]
            type = "echo"

            [nodes.cleanup]
            type = "echo"

            [[edges]]
            source = "review"
            target = "approved"
            priority = 1

            [edges.condition]
            op = "eq"
            selector = "review.value.route"
            value = "approved"

            [[edges]]
            source = "review"
            target = "fallback"
            priority = 2
            default = true

            [[edges]]
            source = "review"
            target = "cleanup"
            kind = "finally"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "review":
                return inputs["value"]
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={
                "payload": {
                    "route": "manual",
                    "private": "customer-secret",
                },
            },
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "fallback"})
        self.assertEqual(calls, ["review", "fallback", "cleanup"])
        self.assertEqual(
            {node.node: node.state for node in result.trace.nodes},
            {
                "review": FlowNodeState.SUCCEEDED,
                "approved": FlowNodeState.SKIPPED,
                "fallback": FlowNodeState.SUCCEEDED,
                "cleanup": FlowNodeState.SUCCEEDED,
            },
        )
        self.assertEqual(
            {edge.index: edge.state for edge in result.trace.edges},
            {
                0: FlowEdgeState.SUPPRESSED,
                1: FlowEdgeState.TAKEN,
                2: FlowEdgeState.TAKEN,
            },
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_invalid_loaded_definition_does_not_execute(
        self,
    ) -> None:
        load_result = loads_flow_definition_result("""
            [flow]
            name = "invalid-runtime"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "env.SECRET_TOKEN"

            [nodes.start]
            type = "echo"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            raise AssertionError("runner must not be called")

        self.assertFalse(load_result.ok)
        self.assertIsNone(load_result.definition)
        self.assertIsNone(load_result.flow)
        self.assertEqual(calls, [])
        self.assertIn(
            "flow.reserved_selector",
            {issue.code for issue in load_result.issues},
        )
        self.assertNotIn("SECRET_TOKEN", str(load_result.public_diagnostics))
        self.assertTrue(callable(runner))

    async def test_invalid_loaded_loop_definition_does_not_execute(
        self,
    ) -> None:
        load_result = loads_flow_definition_result("""
            [flow]
            name = "invalid-loop-runtime"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "review"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "review.value"

            [nodes.review]
            type = "select"

            [nodes.review.mapping]
            value = "input.payload"

            [nodes.review.loop_policy]
            output_selector = "review.value.safe"
            limit_route = "manual"

            [nodes.review.loop_policy.continue_condition]
            op = "eq"
            selector = "review.value.more"
            value = true

            [nodes.review.loop_policy.exit_condition]
            op = "eq"
            selector = "review.value.done"
            value = true

            [nodes.manual]
            type = "echo"

            [[edges]]
            source = "review"
            target = "manual"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            raise AssertionError("runner must not be called")

        self.assertFalse(load_result.ok)
        self.assertIsNone(load_result.definition)
        self.assertIsNone(load_result.flow)
        self.assertEqual(calls, [])
        self.assertIn(
            "flow.unbounded_loop",
            {issue.code for issue in load_result.issues},
        )
        self.assertNotIn("payload", str(load_result.public_diagnostics))
        self.assertTrue(callable(runner))

    async def test_loaded_definition_retries_transient_failure(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-retry-success"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "worker"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "finished.value"

            [nodes.worker]
            type = "select"

            [nodes.worker.mapping]
            value = "input.payload"

            [nodes.worker.retry_policy]
            max_attempts = 2
            retryable_categories = ["transient"]

            [nodes.finished]
            type = "echo"

            [nodes.fallback]
            type = "echo"

            [[edges]]
            source = "worker"
            target = "finished"

            [[edges]]
            source = "worker"
            target = "fallback"
            kind = "error"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "worker" and calls.count("worker") == 1:
                raise FlowNodeExecutionError(
                    code="flow.execution.provider_unavailable",
                    message="Flow node provider is unavailable.",
                    hint="Retry the node.",
                    failure_category="transient",
                )
            if node.name == "worker":
                return inputs["value"]
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={
                "payload": {
                    "status": "ready",
                    "private": "customer-secret",
                },
            },
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "finished"})
        self.assertEqual(calls, ["worker", "worker", "finished"])
        self.assertEqual(
            {node.node: node.attempts for node in result.trace.nodes},
            {
                "worker": 2,
                "finished": 1,
                "fallback": 0,
            },
        )
        self.assertEqual(
            {edge.index: edge.state for edge in result.trace.edges},
            {
                0: FlowEdgeState.TAKEN,
                1: FlowEdgeState.PENDING,
            },
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_loaded_definition_stops_before_retry_attempt(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-retry-cancel"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "worker"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "finished.value"

            [nodes.worker]
            type = "select"

            [nodes.worker.mapping]
            value = "input.payload"

            [nodes.worker.retry_policy]
            max_attempts = 3
            retryable_categories = ["transient"]

            [nodes.finished]
            type = "echo"

            [nodes.fallback]
            type = "echo"

            [[edges]]
            source = "worker"
            target = "finished"

            [[edges]]
            source = "worker"
            target = "fallback"
            kind = "error"
            """)
        calls: list[str] = []
        checks = 0

        async def check_cancelled() -> None:
            nonlocal checks
            checks += 1
            if checks == 2:
                raise CancelledError("private retry stop")

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "worker":
                raise FlowNodeExecutionError(
                    code="flow.execution.provider_unavailable",
                    message="Flow node provider is unavailable.",
                    hint="Retry the node.",
                    failure_category="transient",
                )
            return node.name

        with self.assertRaises(CancelledError) as raised:
            await execute_flow_plan(
                plan,
                runner,
                inputs={"payload": {"private": "customer-secret"}},
                cancellation_checker=check_cancelled,
            )

        self.assertEqual(str(raised.exception), "private retry stop")
        self.assertEqual(calls, ["worker"])

    async def test_loaded_definition_routes_retry_exhaustion(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-retry"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "worker"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "fallback.value"

            [nodes.worker]
            type = "select"

            [nodes.worker.mapping]
            value = "input.payload"

            [nodes.worker.retry_policy]
            max_attempts = 2
            retryable_categories = ["transient"]
            exhausted_route = "fallback"

            [nodes.generic]
            type = "echo"

            [nodes.fallback]
            type = "echo"

            [nodes.cleanup]
            type = "echo"

            [[edges]]
            source = "worker"
            target = "generic"
            kind = "error"
            priority = 1

            [[edges]]
            source = "worker"
            target = "fallback"
            kind = "error"
            priority = 2

            [[edges]]
            source = "worker"
            target = "cleanup"
            kind = "finally"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "worker":
                raise FlowNodeExecutionError(
                    code="flow.execution.provider_unavailable",
                    message="Flow node provider is unavailable.",
                    hint="Use the declared fallback route.",
                    failure_category="transient",
                )
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={"payload": {"private": "customer-secret"}},
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "fallback"})
        self.assertEqual(calls, ["worker", "worker", "fallback", "cleanup"])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.provider_unavailable"],
        )
        self.assertEqual(
            {edge.index: edge.state for edge in result.trace.edges},
            {
                0: FlowEdgeState.SUPPRESSED,
                1: FlowEdgeState.TAKEN,
                2: FlowEdgeState.TAKEN,
            },
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_loaded_definition_does_not_retry_validation_failure(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-validation-retry"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "worker"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "fallback.value"

            [nodes.worker]
            type = "select"

            [nodes.worker.mapping]
            value = "input.payload"

            [nodes.worker.retry_policy]
            max_attempts = 3

            [nodes.fallback]
            type = "echo"

            [[edges]]
            source = "worker"
            target = "fallback"
            kind = "error"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "worker":
                raise FlowNodeExecutionError(
                    code="flow.execution.validation_failed",
                    message="Flow node validation failed.",
                    hint="Route to the validation fallback.",
                    failure_category="validation",
                )
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={"payload": {"private": "customer-secret"}},
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "fallback"})
        self.assertEqual(calls, ["worker", "fallback"])
        self.assertEqual(
            {node.node: node.attempts for node in result.trace.nodes},
            {
                "worker": 1,
                "fallback": 1,
            },
        )
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.validation_failed"],
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_loaded_definition_routes_cancellation(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-cancellation"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "worker"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "cancelled.value"

            [nodes.worker]
            type = "select"

            [nodes.worker.mapping]
            value = "input.payload"

            [nodes.cancelled]
            type = "echo"

            [nodes.cleanup]
            type = "echo"

            [[edges]]
            source = "worker"
            target = "cancelled"
            kind = "cancellation"

            [[edges]]
            source = "worker"
            target = "cleanup"
            kind = "finally"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "worker":
                raise CancelledError("private cancellation detail")
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={"payload": {"private": "customer-secret"}},
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "cancelled"})
        self.assertEqual(calls, ["worker", "cancelled", "cleanup"])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.node_cancelled"],
        )
        self.assertEqual(
            {edge.index: edge.state for edge in result.trace.edges},
            {
                0: FlowEdgeState.TAKEN,
                1: FlowEdgeState.TAKEN,
            },
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))
        self.assertNotIn(
            "private cancellation detail",
            str(result.public_diagnostics),
        )

    async def test_loaded_definition_routes_timeout(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-timeout"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "worker"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "timed.value"

            [nodes.worker]
            type = "select"

            [nodes.worker.mapping]
            value = "input.payload"

            [nodes.worker.timeout_policy]
            per_attempt_seconds = 0.001

            [nodes.timed]
            type = "echo"

            [nodes.cleanup]
            type = "echo"

            [[edges]]
            source = "worker"
            target = "timed"
            kind = "timeout"

            [[edges]]
            source = "worker"
            target = "cleanup"
            kind = "finally"
            """)
        calls: list[str] = []

        async def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "worker":
                await sleep(0.02)
                return "private timeout body"
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={"payload": {"private": "customer-secret"}},
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "timed"})
        self.assertEqual(calls, ["worker", "timed", "cleanup"])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.node_timeout"],
        )
        self.assertEqual(
            {edge.index: edge.state for edge in result.trace.edges},
            {
                0: FlowEdgeState.TAKEN,
                1: FlowEdgeState.TAKEN,
            },
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))
        self.assertNotIn(
            "private timeout body", str(result.public_diagnostics)
        )

    async def test_loaded_definition_routes_mapping_failure_before_runner(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-mapping-failure"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "worker"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "fallback.value"

            [nodes.worker]
            type = "select"

            [nodes.worker.mapping]
            value = "input.payload.required"

            [nodes.fallback]
            type = "echo"

            [[edges]]
            source = "worker"
            target = "fallback"
            kind = "error"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={
                "payload": {
                    "private": "customer-secret",
                },
            },
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "fallback"})
        self.assertEqual(calls, ["fallback"])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.missing_selector_value"],
        )
        self.assertEqual(
            {node.node: node.state for node in result.trace.nodes},
            {
                "worker": FlowNodeState.FAILED,
                "fallback": FlowNodeState.SUCCEEDED,
            },
        )
        self.assertEqual(
            {edge.index: edge.state for edge in result.trace.edges},
            {0: FlowEdgeState.TAKEN},
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_loaded_definition_reports_route_condition_failure(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-condition"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "worker"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "worker.value.status"

            [nodes.worker]
            type = "select"

            [nodes.worker.mapping]
            value = "input.payload"

            [nodes.downstream]
            type = "echo"

            [[edges]]
            source = "worker"
            target = "downstream"

            [edges.condition]
            op = "eq"
            selector = "worker.value.missing"
            value = "ready"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            return inputs["value"]

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={
                "payload": {
                    "status": "ready",
                    "private": "customer-secret",
                },
            },
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "ready"})
        self.assertEqual(calls, ["worker"])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.condition_missing_value"],
        )
        self.assertEqual(
            {edge.index: edge.state for edge in result.trace.edges},
            {0: FlowEdgeState.FAILED},
        )
        _assert_recorded_duration(self, result.trace.nodes[0].duration_ms)
        _assert_recorded_duration(self, result.trace.edges[0].duration_ms)
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_loaded_definition_fail_fast_join_blocks_execution(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-fail-fast"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "source"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "joined.value"

            [nodes.source]
            type = "select"

            [nodes.source.mapping]
            value = "input.payload"

            [nodes.left]
            type = "echo"

            [nodes.right]
            type = "echo"

            [nodes.joined]
            type = "echo"

            [nodes.joined.join_policy]
            type = "fail_fast"

            [[edges]]
            source = "source"
            target = "left"
            routing_policy = "all_matching"

            [[edges]]
            source = "source"
            target = "right"
            routing_policy = "all_matching"

            [[edges]]
            source = "left"
            target = "joined"
            kind = "error"

            [[edges]]
            source = "right"
            target = "joined"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "source":
                return inputs["value"]
            if node.name == "left":
                raise RuntimeError("private branch payload")
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={
                "payload": {
                    "status": "ready",
                    "private": "customer-secret",
                },
            },
            concurrency_limit=2,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {})
        self.assertEqual(calls, ["source", "left", "right"])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            [
                "flow.execution.node_failed",
                "flow.execution.join_failed",
                "flow.execution.missing_output",
            ],
        )
        self.assertEqual(
            {node.node: node.state for node in result.trace.nodes},
            {
                "source": FlowNodeState.SUCCEEDED,
                "left": FlowNodeState.FAILED,
                "right": FlowNodeState.SUCCEEDED,
                "joined": FlowNodeState.FAILED,
            },
        )
        self.assertEqual(
            {edge.index: edge.state for edge in result.trace.edges},
            {
                0: FlowEdgeState.TAKEN,
                1: FlowEdgeState.TAKEN,
                2: FlowEdgeState.TAKEN,
                3: FlowEdgeState.TAKEN,
            },
        )
        self.assertNotIn("private branch payload", str(result))
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_loaded_definition_routes_loop_limit(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-loop"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "review"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "manual.value"

            [nodes.review]
            type = "select"

            [nodes.review.mapping]
            value = "input.payload"

            [nodes.review.loop_policy]
            max_iterations = 2
            output_selector = "review.value.safe"
            limit_route = "manual"

            [nodes.review.loop_policy.continue_condition]
            op = "eq"
            selector = "review.value.more"
            value = true

            [nodes.review.loop_policy.exit_condition]
            op = "eq"
            selector = "review.value.done"
            value = true

            [nodes.finished]
            type = "echo"

            [nodes.manual]
            type = "echo"

            [[edges]]
            source = "review"
            target = "finished"
            priority = 1

            [[edges]]
            source = "review"
            target = "manual"
            priority = 2
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "review":
                return {
                    "done": False,
                    "more": True,
                    "safe": "redacted",
                    "private": "customer-secret",
                }
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={"payload": {"private": "input-secret"}},
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "manual"})
        self.assertEqual(calls, ["review", "review", "manual"])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.loop_limit_reached"],
        )
        self.assertEqual(
            {edge.index: edge.state for edge in result.trace.edges},
            {
                0: FlowEdgeState.SUPPRESSED,
                1: FlowEdgeState.TAKEN,
            },
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))
        self.assertNotIn("input-secret", str(result.public_diagnostics))

    async def test_loaded_definition_loop_limit_takes_error_edge(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-loop-limit-route"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "review"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "manual.value"

            [nodes.review]
            type = "select"

            [nodes.review.mapping]
            value = "input.payload"

            [nodes.review.loop_policy]
            max_iterations = 1
            output_selector = "review.value.safe"
            limit_route = "manual"

            [nodes.review.loop_policy.continue_condition]
            op = "eq"
            selector = "review.value.more"
            value = true

            [nodes.review.loop_policy.exit_condition]
            op = "eq"
            selector = "review.value.done"
            value = true

            [nodes.manual]
            type = "echo"

            [nodes.finished]
            type = "echo"

            [[edges]]
            source = "review"
            target = "finished"
            kind = "success"

            [[edges]]
            source = "review"
            target = "manual"
            kind = "error"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "review":
                value = dict(cast(Mapping[str, object], inputs["value"]))
                value["safe"] = "pending"
                return value
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={
                "payload": {
                    "done": False,
                    "more": True,
                    "private": "customer-secret",
                },
            },
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {"answer": "manual"})
        self.assertEqual(calls, ["review", "manual"])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.loop_limit_reached"],
        )
        self.assertEqual(
            {edge.index: edge.state for edge in result.trace.edges},
            {
                0: FlowEdgeState.PENDING,
                1: FlowEdgeState.TAKEN,
            },
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))

    async def test_loaded_definition_stops_between_loop_iterations(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-loop-cancel"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "review"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "review.value"

            [nodes.review]
            type = "select"

            [nodes.review.mapping]
            value = "input.payload"

            [nodes.review.loop_policy]
            max_iterations = 4
            output_selector = "review.value.safe"
            limit_route = "manual"

            [nodes.review.loop_policy.continue_condition]
            op = "eq"
            selector = "review.value.more"
            value = true

            [nodes.review.loop_policy.exit_condition]
            op = "eq"
            selector = "review.value.done"
            value = true

            [nodes.finished]
            type = "echo"

            [nodes.manual]
            type = "echo"

            [[edges]]
            source = "review"
            target = "finished"
            priority = 1

            [[edges]]
            source = "review"
            target = "manual"
            priority = 2
            """)
        calls: list[str] = []
        checks = 0

        async def check_cancelled() -> None:
            nonlocal checks
            checks += 1
            if checks == 2:
                raise CancelledError("private loop stop")

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "review":
                return {
                    "done": False,
                    "more": True,
                    "safe": "pending",
                    "private": "customer-secret",
                }
            return node.name

        with self.assertRaises(CancelledError) as raised:
            await execute_flow_plan(
                plan,
                runner,
                inputs={"payload": {"private": "input-secret"}},
                cancellation_checker=check_cancelled,
            )

        self.assertEqual(str(raised.exception), "private loop stop")
        self.assertEqual(calls, ["review"])

    async def test_loaded_definition_exits_loop_with_declared_output(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-loop-exit"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "review"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "review.value"

            [nodes.review]
            type = "select"

            [nodes.review.mapping]
            value = "input.payload"

            [nodes.review.loop_policy]
            max_iterations = 4
            output_selector = "review.value.safe"
            limit_route = "manual"

            [nodes.review.loop_policy.continue_condition]
            op = "eq"
            selector = "review.value.more"
            value = true

            [nodes.review.loop_policy.exit_condition]
            op = "eq"
            selector = "review.value.done"
            value = true

            [nodes.finished]
            type = "echo"

            [nodes.manual]
            type = "echo"

            [[edges]]
            source = "review"
            target = "finished"
            priority = 1

            [[edges]]
            source = "review"
            target = "manual"
            priority = 2
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            if node.name == "review":
                return {
                    "done": calls.count("review") == 2,
                    "more": calls.count("review") == 1,
                    "safe": "approved",
                    "private": "customer-secret",
                }
            return node.name

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={"payload": {"private": "input-secret"}},
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "approved"})
        self.assertEqual(calls, ["review", "review", "finished"])
        self.assertEqual(result.node_outputs["review"]["value"], "approved")
        self.assertEqual(
            [node.attempts for node in result.trace.nodes],
            [2, 1, 0],
        )
        self.assertEqual(
            {edge.index: edge.state for edge in result.trace.edges},
            {
                0: FlowEdgeState.TAKEN,
                1: FlowEdgeState.SUPPRESSED,
            },
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))
        self.assertNotIn("input-secret", str(result.public_diagnostics))

    async def test_loaded_definition_reports_missing_declared_output(
        self,
    ) -> None:
        plan = self._compile_loaded_definition("""
            [flow]
            name = "runtime-output-missing"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "terminal.missing"

            [nodes.start]
            type = "echo"

            [nodes.terminal]
            type = "echo"

            [[edges]]
            source = "start"
            target = "terminal"
            """)
        calls: list[str] = []

        def runner(
            node: FlowNodePlan,
            _: Mapping[str, object],
        ) -> object:
            calls.append(node.name)
            return {
                "node": node.name,
                "private": "customer-secret",
            }

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={"payload": {"private": "input-secret"}},
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {})
        self.assertEqual(calls, ["start", "terminal"])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.execution.missing_output"],
        )
        self.assertEqual(
            {node.node: node.state for node in result.trace.nodes},
            {
                "start": FlowNodeState.SUCCEEDED,
                "terminal": FlowNodeState.SUCCEEDED,
            },
        )
        self.assertNotIn("customer-secret", str(result.public_diagnostics))
        self.assertNotIn("input-secret", str(result.public_diagnostics))

    async def test_invalid_loaded_definition_rejects_sibling_route_condition(
        self,
    ) -> None:
        load_result = loads_flow_definition_result("""
            [flow]
            name = "runtime-sibling-routing"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "source"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "right.value"

            [nodes.source]
            type = "echo"

            [nodes.left]
            type = "echo"

            [nodes.right]
            type = "echo"

            [nodes.chosen]
            type = "echo"

            [[edges]]
            source = "source"
            target = "left"
            routing_policy = "all_matching"

            [[edges]]
            source = "source"
            target = "right"
            routing_policy = "all_matching"

            [[edges]]
            source = "left"
            target = "chosen"

            [edges.condition]
            op = "eq"
            selector = "right.value"
            value = "right"
            """)

        self.assertFalse(load_result.ok)
        self.assertIsNone(load_result.definition)
        self.assertIsNone(load_result.flow)
        diagnostic_codes = [
            diagnostic["code"] for diagnostic in load_result.public_diagnostics
        ]
        self.assertEqual(
            diagnostic_codes,
            ["flow.bad_reference"],
        )
        self.assertNotIn("right.value", str(load_result.public_diagnostics))

    async def test_invalid_loaded_definition_rejects_retry_success_route(
        self,
    ) -> None:
        load_result = loads_flow_definition_result("""
            [flow]
            name = "runtime-retry-route-kind"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "text"

            [entry]
            type = "node"
            node = "worker"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "fallback.value"

            [nodes.worker]
            type = "echo"

            [nodes.worker.retry_policy]
            max_attempts = 2
            retryable_categories = ["transient"]
            exhausted_route = "fallback"

            [nodes.fallback]
            type = "echo"

            [[edges]]
            source = "worker"
            target = "fallback"
            kind = "success"
            """)

        self.assertFalse(load_result.ok)
        self.assertIsNone(load_result.definition)
        self.assertIsNone(load_result.flow)
        self.assertEqual(
            [
                diagnostic["code"]
                for diagnostic in load_result.public_diagnostics
            ],
            ["flow.missing_retry_exhaustion_route"],
        )
        self.assertNotIn("fallback.value", str(load_result.public_diagnostics))

    def _compile_loaded_definition(self, source: str) -> FlowExecutionPlan:
        load_result = loads_flow_definition_result(source)
        self.assertTrue(load_result.ok, load_result.public_diagnostics)
        assert load_result.definition is not None
        compile_result = compile_flow_definition(load_result.definition)
        self.assertTrue(compile_result.ok, compile_result.public_diagnostics)
        assert compile_result.plan is not None
        return compile_result.plan


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
                    target="fallback",
                    kind=FlowMappingKind.COALESCE,
                    sources=(
                        parse_flow_selector("prepare.result.missing"),
                        parse_flow_selector("input.payload.customer.name"),
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
        self.assertEqual(result["fallback"], "Ada")
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
                        kind=FlowMappingKind.COALESCE,
                    ),
                ),
                "flow.execution.empty_mapping",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.COALESCE,
                        sources=(
                            parse_flow_selector("prepare.result.missing"),
                            parse_flow_selector("input.missing"),
                        ),
                    ),
                ),
                "flow.execution.missing_selector_value",
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
