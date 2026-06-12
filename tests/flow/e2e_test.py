from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

import avalan.flow.executor as flow_executor_module
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
    FlowExecutionPlan,
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
    FlowPlanCompileResult,
    FlowPlanExecutionResult,
    FlowPlanNodeRunner,
    FlowRetryPolicy,
    FlowReviewState,
    InMemoryFlowStateStore,
    Node,
    compile_flow_definition,
    compile_flow_source,
    default_flow_node_registry,
    serialize_flow_definition,
    tool_flow_node_registry,
)
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.shell import (
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellPolicyDenied,
    ShellToolSet,
)


class FlowE2ETestCase(IsolatedAsyncioTestCase):
    async def test_native_strict_toml_loads_validates_compiles_and_runs(
        self,
    ) -> None:
        source = """
            [flow]
            name = "native_strict_runtime"
            version = "1"

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
            answer = "finish.value"

            [nodes.start]
            type = "input"

            [nodes.finish]
            type = "select"

            [nodes.finish.mapping.value]
            type = "object"

            [nodes.finish.mapping.value.fields]
            customer = "input.payload.customer"
            approved = "input.payload.approved"

            [[edges]]
            source = "start"
            target = "finish"
            label = "approved"
            """

        validation = await FlowDefinitionLoader().loads_validation_result(
            source
        )
        loaded = await FlowDefinitionLoader().loads_result(source)

        self.assertTrue(validation.ok, validation.public_diagnostics)
        self.assertFalse(validation.authoring_graph)
        self.assertIsNone(validation.flow)
        self.assertTrue(loaded.ok, loaded.public_diagnostics)
        self.assertFalse(loaded.authoring_graph)
        assert loaded.definition is not None
        assert loaded.flow is not None

        plan = await compile_flow_definition(loaded.definition)
        self.assertTrue(plan.ok, plan.public_diagnostics)
        assert plan.plan is not None
        self.assertEqual(
            [
                (edge.source, edge.target, edge.label)
                for edge in plan.plan.edges
            ],
            [("start", "finish", "approved")],
        )

        result = await FlowExecutor().run(
            loaded.definition,
            inputs={
                "payload": {
                    "customer": "Northwind",
                    "approved": True,
                },
            },
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(
            result.outputs,
            {"answer": {"customer": "Northwind", "approved": True}},
        )
        self.assertFalse(hasattr(loaded.definition, "graph"))
        self.assertNotIn(
            "[graph]", serialize_flow_definition(loaded.definition)
        )

    async def test_shell_tool_node_runs_through_enabled_tool_manager(
        self,
    ) -> None:
        policy = _FlowShellPolicy()
        executor = _FlowShellExecutor("shell flow output\n")
        manager = ToolManager.create_instance(
            enable_tools=["shell.cat"],
            available_toolsets=[
                ShellToolSet(
                    policy=policy,
                    executor=executor,
                    formatter=lambda result: result.stdout,
                )
            ],
        )
        registry = tool_flow_node_registry(manager)

        loaded = await FlowDefinitionLoader(registry).loads_result(
            _shell_cat_flow_source()
        )
        self.assertTrue(loaded.ok, loaded.public_diagnostics)
        assert loaded.definition is not None

        result = await FlowExecutor(registry=registry).run(
            loaded.definition,
            inputs={"payload": {"path": "visible.txt"}},
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"answer": "shell flow output\n"})
        self.assertEqual(
            [request.tool_name for request in policy.requests],
            [
                "shell.cat",
            ],
        )
        self.assertEqual(policy.requests[0].paths[0].path, "visible.txt")
        self.assertEqual(
            [spec.tool_name for spec in executor.specs],
            [
                "shell.cat",
            ],
        )

    async def test_shell_tool_node_returns_policy_denied_result(self) -> None:
        policy = _FlowShellPolicy(
            denial=ShellPolicyDenied(
                ShellExecutionErrorCode.DENIED_PATH,
                "path is denied",
            )
        )
        executor = _FlowShellExecutor("unused")
        manager = ToolManager.create_instance(
            enable_tools=["shell.cat"],
            available_toolsets=[
                ShellToolSet(policy=policy, executor=executor)
            ],
        )
        registry = tool_flow_node_registry(manager)

        loaded = await FlowDefinitionLoader(registry).loads_result(
            _shell_cat_flow_source()
        )
        self.assertTrue(loaded.ok, loaded.public_diagnostics)
        assert loaded.definition is not None

        result = await FlowExecutor(registry=registry).run(
            loaded.definition,
            inputs={"payload": {"path": "denied.txt"}},
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(len(policy.requests), 1)
        self.assertEqual(executor.specs, [])
        assert isinstance(result.outputs["answer"], str)
        self.assertIn("tool: shell.cat", result.outputs["answer"])
        self.assertIn("status: policy_denied", result.outputs["answer"])
        self.assertIn("error_code: denied_path", result.outputs["answer"])
        self.assertIn(
            "error_message: path is denied", result.outputs["answer"]
        )

    async def test_shell_tool_node_returns_command_unavailable_result(
        self,
    ) -> None:
        policy = _FlowShellPolicy(executable=None)
        manager = ToolManager.create_instance(
            enable_tools=["shell.cat"],
            available_toolsets=[ShellToolSet(policy=policy)],
        )
        registry = tool_flow_node_registry(manager)

        loaded = await FlowDefinitionLoader(registry).loads_result(
            _shell_cat_flow_source()
        )
        self.assertTrue(loaded.ok, loaded.public_diagnostics)
        assert loaded.definition is not None

        result = await FlowExecutor(registry=registry).run(
            loaded.definition,
            inputs={"payload": {"path": "missing.txt"}},
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(len(policy.requests), 1)
        assert isinstance(result.outputs["answer"], str)
        self.assertIn("tool: shell.cat", result.outputs["answer"])
        self.assertIn(
            "status: command_unavailable",
            result.outputs["answer"],
        )
        self.assertIn(
            "error_code: command_unavailable",
            result.outputs["answer"],
        )
        self.assertIn(
            "error_message: command is unavailable",
            result.outputs["answer"],
        )

    async def test_shell_tool_node_reports_disabled_tool_diagnostic(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["math"],
            available_toolsets=[
                ToolSet(namespace="math", tools=[]),
                ShellToolSet(
                    policy=_FlowShellPolicy(),
                    executor=_FlowShellExecutor("unused"),
                ),
            ],
        )
        registry = tool_flow_node_registry(manager)

        result = await FlowDefinitionLoader(registry).loads_result(
            _shell_cat_flow_source()
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.definition)
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.tool_disabled"],
        )
        self.assertEqual(
            result.public_diagnostics[0]["path"],
            "nodes.read.ref",
        )

    async def test_shell_tool_node_is_not_available_by_default(self) -> None:
        result = await FlowDefinitionLoader().loads_result(
            _shell_cat_flow_source()
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.definition)
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.unsupported_node_type"],
        )
        self.assertEqual(
            result.public_diagnostics[0]["path"],
            "nodes.read.type",
        )

    async def test_native_strict_toml_rejects_invalid_edges_before_runtime(
        self,
    ) -> None:
        result = await FlowDefinitionLoader().loads_result("""
            [flow]
            name = "invalid_native_strict_runtime"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "input"

            [[edges]]
            source = "start"
            target = "missing_private_node"
            """)

        self.assertFalse(result.ok)
        self.assertFalse(result.authoring_graph)
        self.assertIsNone(result.definition)
        self.assertIsNone(result.flow)
        self.assertEqual(
            [(issue.code, issue.path) for issue in result.issues],
            [("flow.bad_reference", "edges.target")],
        )
        self.assertNotIn(
            "missing_private_node", str(result.public_diagnostics)
        )

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

    async def test_file_graph_loads_validates_compiles_and_runs(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            graph_directory = root / "graphs"
            graph_directory.mkdir()
            graph_path = graph_directory / "customer-token.mmd"
            graph_path.write_text(
                "\n".join(
                    (
                        "flowchart LR",
                        "start route_1@-->|Private graph route| finish",
                        'start -.-> private_note["Private decorative note"]',
                    )
                ),
                encoding="utf-8",
            )
            flow_path = root / "flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "file_graph_runtime"
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
                source = "file"
                mode = "executable"
                path = "graphs/customer-token.mmd"

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
                """,
                encoding="utf-8",
            )

            validation = await FlowDefinitionLoader().load_validation_result(
                flow_path
            )
            loaded = await FlowDefinitionLoader().load_result(flow_path)

        self.assertTrue(validation.ok, validation.public_diagnostics)
        self.assertTrue(validation.authoring_graph)
        self.assertIsNone(validation.flow)
        assert validation.definition is not None
        self.assertTrue(loaded.ok, loaded.public_diagnostics)
        self.assertTrue(loaded.authoring_graph)
        assert loaded.definition is not None
        assert loaded.flow is not None
        self.assertEqual(
            [
                (edge.source, edge.target, edge.label)
                for edge in loaded.definition.edges
            ],
            [("start", "finish", "approved")],
        )
        self.assertEqual(validation.definition.edges, loaded.definition.edges)

        plan = await compile_flow_definition(loaded.definition)
        run = await FlowExecutor().run(loaded.definition)

        self.assertTrue(plan.ok, plan.public_diagnostics)
        self.assertTrue(run.ok, run.public_diagnostics)
        self.assertEqual(run.outputs, {"result": {"answer": "ok"}})
        rendered = str(
            (
                loaded.definition,
                loaded.public_diagnostics,
                serialize_flow_definition(loaded.definition),
            )
        )
        self.assertNotIn("[graph]", rendered)
        self.assertNotIn("Private graph route", rendered)
        self.assertNotIn("Private decorative note", rendered)
        self.assertNotIn("customer-token", rendered)
        self.assertFalse(hasattr(loaded.definition, "graph"))

    async def test_file_graph_without_safe_base_fails_before_node_build(
        self,
    ) -> None:
        build_calls: list[str] = []

        def blocked_factory(definition: FlowNodeDefinition) -> Node:
            build_calls.append(definition.name)
            raise AssertionError("runtime node build should not be reached")

        registry = default_flow_node_registry().register(
            "blocked",
            blocked_factory,
            metadata=FlowNodeMetadata(kind=FlowNodeKind.PASS_THROUGH),
        )
        result = await FlowDefinitionLoader(registry).loads_result("""
            [flow]
            name = "file_graph_without_base"
            entrypoint = "start"
            output_node = "finish"

            [graph]
            format = "mermaid"
            source = "file"
            mode = "executable"
            path = "private/customer-token.mmd"

            [nodes.start]
            type = "blocked"

            [nodes.finish]
            type = "blocked"
            """)

        self.assertFalse(result.ok)
        self.assertTrue(result.authoring_graph)
        self.assertIsNone(result.definition)
        self.assertIsNone(result.flow)
        self.assertEqual(build_calls, [])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.graph.read_failure"],
        )
        self.assertNotIn("customer-token", str(result.public_diagnostics))
        self.assertNotIn("private/", str(result.public_diagnostics))

    async def test_invalid_graph_authoring_stops_before_runtime_build(
        self,
    ) -> None:
        build_calls: list[str] = []

        def blocked_factory(definition: FlowNodeDefinition) -> Node:
            build_calls.append(definition.name)
            raise AssertionError("runtime node build should not be reached")

        registry = default_flow_node_registry().register(
            "blocked",
            blocked_factory,
            metadata=FlowNodeMetadata(kind=FlowNodeKind.PASS_THROUGH),
        )

        result = await FlowDefinitionLoader(registry).loads_result("""
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
            type = "blocked"

            [nodes.finish]
            type = "blocked"
            """)

        self.assertFalse(result.ok)
        self.assertIsNone(result.definition)
        self.assertEqual(build_calls, [])
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.graph.unsupported_executable_edge"],
        )
        self.assertNotIn("Private graph label", str(result.public_diagnostics))

    async def test_executor_boundary_receives_only_strict_flow_objects(
        self,
    ) -> None:
        compiled = await compile_flow_source("""
            [flow]
            name = "graph_runtime_boundary"
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
            start -.-> note["Private decorative note"]
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
        self.assertTrue(compiled.ok, compiled.public_diagnostics)
        assert compiled.definition is not None
        compile_inputs: list[FlowDefinition] = []
        runtime_inputs: list[FlowExecutionPlan] = []
        original_compile = flow_executor_module.compile_flow_definition
        original_execute = flow_executor_module.execute_flow_plan

        async def capturing_compile(
            definition: FlowDefinition,
            registry: FlowNodeRegistry | None = None,
        ) -> FlowPlanCompileResult:
            compile_inputs.append(definition)
            return await original_compile(definition, registry)

        async def capturing_execute(
            plan: FlowExecutionPlan,
            runner: FlowPlanNodeRunner,
            **kwargs: Any,
        ) -> FlowPlanExecutionResult:
            runtime_inputs.append(plan)
            return await original_execute(plan, runner, **kwargs)

        with (
            patch.object(
                flow_executor_module,
                "compile_flow_definition",
                capturing_compile,
            ),
            patch.object(
                flow_executor_module,
                "execute_flow_plan",
                capturing_execute,
            ),
        ):
            result = await FlowExecutor().run(compiled.definition)

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs, {"result": {"answer": "ok"}})
        self.assertEqual(compile_inputs, [compiled.definition])
        self.assertEqual(len(runtime_inputs), 1)
        self.assertIsInstance(runtime_inputs[0], FlowExecutionPlan)
        self.assertEqual(
            [
                (edge.source, edge.target, edge.label)
                for edge in compiled.definition.edges
            ],
            [("start", "finish", "approved")],
        )
        self.assertFalse(hasattr(compiled.definition, "graph"))
        self.assertFalse(hasattr(runtime_inputs[0], "graph"))
        self.assertNotIn("Private graph label", str(compiled.definition))
        self.assertNotIn("Private decorative note", str(runtime_inputs[0]))

    async def test_graph_runtime_trace_events_and_records_are_strict_only(
        self,
    ) -> None:
        compiled = await compile_flow_source(
            """
            [flow]
            name = "graph_runtime_privacy"
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
            start route_1@-->|Private graph route| finish
            private_note d@-->|Private decorative route| private_sink
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
            """,
            source_path="/private/customer/flow.toml",
        )
        invalid = await compile_flow_source(
            """
            [flow]
            name = "graph_runtime_invalid_privacy"
            entrypoint = "start"
            output_node = "finish"

            [graph]
            format = "mermaid"
            source = "inline"
            mode = "executable"
            diagram = '''
            flowchart LR
            start -->|Private invalid graph route| finish
            '''

            [nodes.start]
            type = "constant"
            value = "ok"

            [nodes.finish]
            type = "pass-through"
            """,
            source_path="/private/customer/invalid-flow.toml",
        )

        self.assertTrue(compiled.ok, compiled.public_diagnostics)
        assert compiled.definition is not None
        events: list[Event] = []
        executor = FlowExecutor(event_listener=events.append)
        result = await executor.run(compiled.definition)

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.result is not None
        self.assertEqual(result.outputs, {"result": {"answer": "ok"}})
        store = InMemoryFlowStateStore()
        record = await store.create_flow_execution(
            "run-graph-runtime",
            trace=result.result.trace,
            node_outputs=result.result.node_outputs,
            selected_outputs=result.outputs,
            metadata={
                "strict_flow": {
                    "name": "graph_runtime_privacy",
                    "version": "1",
                }
            },
        )
        record_inspection = executor.inspect(record, plan=result.plan)
        record_export = executor.export_trace(record, plan=result.plan)

        self.assertFalse(invalid.ok)
        self.assertEqual(
            [diagnostic["code"] for diagnostic in invalid.public_diagnostics],
            ["flow.graph.unsupported_executable_edge"],
        )
        self.assertEqual(
            [(edge.source, edge.target) for edge in result.result.trace.edges],
            [("start", "finish")],
        )
        self.assertEqual(
            {
                cast(Mapping[str, object], event.payload)["flow_name"]
                for event in events
                if event.payload is not None
            },
            {"graph_runtime_privacy"},
        )
        _assert_no_authoring_graph_values(
            self,
            (
                tuple(event.payload for event in events),
                result.result.trace.as_public_dict(),
                result.inspect().as_public_dict(),
                result.export_sanitized_trace(),
                record.as_snapshot(),
                record_inspection.as_public_dict(),
                record_export,
                invalid.public_diagnostics,
            ),
        )

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


def _shell_cat_flow_source() -> str:
    return """
        [flow]
        name = "shell_tool_flow"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "json"

        [entry]
        type = "node"
        node = "read"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "read.result"

        [nodes.read]
        type = "tool"
        ref = "shell.cat"

        [nodes.read.mapping.arguments]
        type = "object"

        [nodes.read.mapping.arguments.fields]
        path = "inputs.payload.path"

        [nodes.read.config.arguments]
        path = "path"
        """


class _FlowShellPolicy(ExecutionPolicy):
    def __init__(
        self,
        *,
        denial: ShellPolicyDenied | None = None,
        executable: str | None = "/usr/bin/cat",
    ) -> None:
        self._denial = denial
        self._executable = executable
        self.requests: list[ShellCommandRequest] = []

    async def normalize(
        self,
        request: ShellCommandRequest,
    ) -> ExecutionSpec:
        self.requests.append(request)
        if self._denial is not None:
            raise self._denial
        return ExecutionPolicy().create_execution_spec(
            backend="local",
            tool_name=request.tool_name,
            command=request.command,
            executable=self._executable,
            argv=("cat", "--", "visible.txt"),
            display_argv=("cat", "--", "visible.txt"),
            cwd=".",
            display_cwd=".",
            env={"LC_ALL": "C"},
            stdin=None,
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            resource_class="standard",
            output_plan=None,
            timeout_seconds=1.0,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )


class _FlowShellExecutor:
    def __init__(self, stdout: str) -> None:
        self._stdout = stdout
        self.specs: list[ExecutionSpec] = []

    async def execute(self, spec: ExecutionSpec) -> ExecutionResult:
        self.specs.append(spec)
        return ExecutionResult(
            backend=spec.backend,
            tool_name=spec.tool_name,
            command=spec.command,
            argv=spec.argv,
            display_argv=spec.display_argv,
            cwd=spec.cwd,
            display_cwd=spec.display_cwd,
            status=ShellExecutionStatus.COMPLETED,
            exit_code=0,
            stdout=self._stdout,
            stderr="",
            stdout_media_type=spec.stdout_media_type,
            output_kind=spec.output_kind,
            stdout_bytes=len(self._stdout.encode()),
            stderr_bytes=0,
            stdout_truncated=False,
            stderr_truncated=False,
            timed_out=False,
            cancelled=False,
            duration_ms=1,
            error_code=ShellExecutionErrorCode.COMPLETED,
            metadata=spec.metadata,
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


def _assert_no_authoring_graph_values(
    test_case: FlowE2ETestCase,
    values: tuple[object, ...],
) -> None:
    rendered = str(values)
    for sentinel in (
        "flowchart",
        "route_1",
        "Private graph route",
        "Private decorative route",
        "Private invalid graph route",
        "private_note",
        "private_sink",
        "/private/customer",
        "flow.toml",
        "invalid-flow.toml",
    ):
        test_case.assertNotIn(sentinel, rendered)


if __name__ == "__main__":
    main()
