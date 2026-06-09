from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

from avalan.flow import (
    FLOW_SUBFLOW_NODE_TYPE,
    FlowDefinition,
    FlowEntryBehavior,
    FlowExecutionPlan,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowMappingKind,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodePlan,
    FlowNodeRegistry,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowPlanCompileResult,
    LocalFlowSubflowResolver,
    Node,
    compile_flow_definition,
    execute_flow_plan,
    flow_node_registry_runner,
    parse_flow_selector,
    subflow_node_registry,
    validate_flow_definition,
)
from avalan.flow.registry import FlowNodeConfigurationError


class FlowSubflowValidationTestCase(TestCase):
    def test_validate_flow_definition_accepts_local_subflow(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_child_flow(base / "child.toml")
            registry = subflow_node_registry()

            result = validate_flow_definition(
                _parent_definition(base),
                registry,
            )

            self.assertTrue(result.ok, result.public_diagnostics)
            self.assertTrue(registry.supports_subflow_resolution("subflow"))

    def test_validate_accepts_graph_subflow_without_building_nodes(
        self,
    ) -> None:
        build_calls = 0

        def factory(definition: FlowNodeDefinition) -> Node:
            nonlocal build_calls
            build_calls += 1
            raise AssertionError(definition.name)

        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_graph_child_flow(base / "child.toml")
            registry = subflow_node_registry(
                base_registry=_external_node_registry(factory)
            )

            result = validate_flow_definition(
                _parent_definition(base),
                registry,
            )

            self.assertTrue(result.ok, result.public_diagnostics)
            self.assertEqual(build_calls, 0)

    def test_validate_rejects_bad_graph_subflow_without_building_nodes(
        self,
    ) -> None:
        build_calls = 0

        def factory(definition: FlowNodeDefinition) -> Node:
            nonlocal build_calls
            build_calls += 1
            raise AssertionError(definition.name)

        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_graph_child_flow(base / "child.toml", valid=False)
            registry = subflow_node_registry(
                base_registry=_external_node_registry(factory)
            )

            result = validate_flow_definition(
                _parent_definition(base),
                registry,
            )

            self.assertFalse(result.ok)
            self.assertEqual(
                result.diagnostics[0].code,
                "flow.invalid_subflow",
            )
            self.assertEqual(
                result.diagnostics[0].hint,
                "Reference a valid flow definition; first load issue is "
                "flow.graph.unsupported_executable_edge at graph.edges.",
            )
            self.assertEqual(build_calls, 0)
            self.assertNotIn(
                "Private child route",
                str(result.public_diagnostics),
            )

    def test_validate_rejects_native_subflow_runtime_config(
        self,
    ) -> None:
        build_calls = 0

        def factory(definition: FlowNodeDefinition) -> Node:
            nonlocal build_calls
            build_calls += 1
            raise FlowNodeConfigurationError(
                code="flow.invalid_node_config",
                path=f"nodes.{definition.name}.config",
                message="Flow node configuration is invalid.",
                hint="Use supported runtime configuration.",
            )

        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_child_flow(base / "child.toml", node_type="external")
            registry = subflow_node_registry(
                base_registry=_external_node_registry(factory)
            )

            result = validate_flow_definition(
                _parent_definition(base),
                registry,
            )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.invalid_subflow",
        )
        self.assertEqual(
            result.diagnostics[0].hint,
            "Reference a valid flow definition; first load issue is "
            "flow.invalid_node_config at nodes.start.config.",
        )
        self.assertEqual(build_calls, 1)

    def test_validate_flow_definition_rejects_subflow_without_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_child_flow(base / "child.toml")

            result = validate_flow_definition(_parent_definition(base))

            self.assertFalse(result.ok)
            self.assertEqual(
                result.diagnostics[0].code,
                "flow.unsupported_node_type",
            )

    def test_validate_flow_definition_rejects_untrusted_refs(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            registry = subflow_node_registry()
            cases = (
                ("../child.toml", "flow.path_escape"),
                ("urn:mcp:flow", "flow.path_escape"),
                ("mcp://server/flow.toml", "flow.path_escape"),
                (str(base / "child.toml"), "flow.path_escape"),
                ("child.yaml", "flow.invalid_ref"),
                ("missing.toml", "flow.subflow_not_found"),
            )
            for ref, code in cases:
                with self.subTest(ref=ref):
                    result = validate_flow_definition(
                        _parent_definition(base, ref=ref),
                        registry,
                    )

                    self.assertFalse(result.ok)
                    self.assertEqual(result.diagnostics[0].code, code)
                    self.assertNotIn(ref, str(result.public_diagnostics))

    def test_validate_flow_definition_rejects_ref_outside_trusted_root(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "trusted"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            _write_child_flow(outside / "child.toml")
            registry = subflow_node_registry(
                LocalFlowSubflowResolver(root=root)
            )

            result = validate_flow_definition(
                _parent_definition(outside),
                registry,
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.diagnostics[0].code, "flow.path_escape")

    def test_validate_flow_definition_rejects_symlink_escape(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            outside = Path(directory) / "outside"
            base.mkdir()
            outside.mkdir()
            _write_child_flow(outside / "child.toml")
            (base / "linked.toml").symlink_to(outside / "child.toml")
            registry = subflow_node_registry()

            result = validate_flow_definition(
                _parent_definition(base, ref="linked.toml"),
                registry,
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.diagnostics[0].code, "flow.path_escape")

    def test_validate_flow_definition_accepts_root_backed_subflow(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_child_flow(root / "child.toml")
            registry = subflow_node_registry(
                LocalFlowSubflowResolver(root=root)
            )

            result = validate_flow_definition(
                _parent_definition(None),
                registry,
            )

            self.assertTrue(result.ok, result.public_diagnostics)

    def test_validate_flow_definition_rejects_subflow_without_base(
        self,
    ) -> None:
        registry = subflow_node_registry()

        result = validate_flow_definition(_parent_definition(None), registry)

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.subflow_untrusted_ref",
        )

    def test_validate_flow_definition_rejects_invalid_subflow_file(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "child.toml").write_text("not toml", encoding="utf-8")
            registry = subflow_node_registry()

            result = validate_flow_definition(
                _parent_definition(base),
                registry,
            )

            self.assertFalse(result.ok)
            self.assertEqual(
                result.diagnostics[0].code, "flow.invalid_subflow"
            )

    def test_validate_flow_definition_rejects_subflow_expectation_gaps(
        self,
    ) -> None:
        cases = (
            (
                {
                    "version": "2026-06-08",
                    "output_mapping": {"answer": "answer"},
                },
                "flow.subflow_version_mismatch",
            ),
            (
                {
                    "revision": "rev-2",
                    "output_mapping": {"answer": "answer"},
                },
                "flow.subflow_revision_mismatch",
            ),
            (
                {"version": 7, "output_mapping": {"answer": "answer"}},
                "flow.invalid_subflow_expectation",
            ),
        )

        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_child_flow(base / "child.toml", revision="rev-1")
            registry = subflow_node_registry()
            for config, code in cases:
                with self.subTest(code=code):
                    result = validate_flow_definition(
                        _parent_definition(base, config=config),
                        registry,
                    )

                    self.assertFalse(result.ok)
                    self.assertEqual(result.diagnostics[0].code, code)

    def test_validate_flow_definition_rejects_subflow_input_gaps(
        self,
    ) -> None:
        cases = (
            ((), "flow.missing_subflow_input_mapping"),
            (
                (
                    FlowInputMapping(
                        target="unknown",
                        source="input.payload",
                    ),
                ),
                "flow.unknown_subflow_input",
            ),
            (
                (
                    FlowInputMapping(
                        target="payload",
                        kind=FlowMappingKind.FILE,
                        source="input.payload",
                    ),
                ),
                "flow.incompatible_subflow_input_mapping",
            ),
        )

        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_child_flow(base / "child.toml")
            registry = subflow_node_registry()
            for mappings, code in cases:
                with self.subTest(code=code):
                    result = validate_flow_definition(
                        _parent_definition(base, mappings=mappings),
                        registry,
                    )

                    self.assertFalse(result.ok)
                    self.assertEqual(result.diagnostics[0].code, code)

    def test_validate_flow_definition_accepts_subflow_input_mapping_kinds(
        self,
    ) -> None:
        cases = (
            (FlowInputType.ARRAY, FlowMappingKind.ARRAY),
            (FlowInputType.OBJECT, FlowMappingKind.MERGE),
            (FlowInputType.FILE_ARRAY, FlowMappingKind.FILE_ARRAY),
        )

        with TemporaryDirectory() as directory:
            base = Path(directory)
            registry = subflow_node_registry()
            for input_type, mapping_kind in cases:
                with self.subTest(input_type=input_type):
                    _write_child_flow(
                        base / "child.toml",
                        input_type=input_type,
                    )
                    result = validate_flow_definition(
                        _parent_definition(
                            base,
                            mappings=(
                                FlowInputMapping(
                                    target="payload",
                                    kind=mapping_kind,
                                    source=(
                                        "input.payload"
                                        if mapping_kind
                                        == FlowMappingKind.FILE_ARRAY
                                        else None
                                    ),
                                    sources=(
                                        ("input.payload",)
                                        if mapping_kind
                                        == FlowMappingKind.MERGE
                                        else ()
                                    ),
                                    items=(
                                        ("input.payload",)
                                        if mapping_kind
                                        == FlowMappingKind.ARRAY
                                        else ()
                                    ),
                                ),
                            ),
                        ),
                        registry,
                    )

                    self.assertTrue(result.ok, result.public_diagnostics)

    def test_validate_flow_definition_rejects_subflow_output_gaps(
        self,
    ) -> None:
        cases = (
            ({}, "flow.missing_subflow_output_mapping"),
            ({"answer": "missing"}, "flow.unknown_subflow_output"),
            ({"answer": "answer"}, "flow.missing_subflow_output_mapping"),
            (
                {"answer": "answer", "duplicate": "answer", "audit": "audit"},
                "flow.duplicate_subflow_output_mapping",
            ),
            ({"": "answer"}, "flow.invalid_subflow_output_mapping"),
            ({"answer": ""}, "flow.invalid_subflow_output_mapping"),
        )

        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_child_flow(
                base / "child.toml",
                extra_output=True,
            )
            registry = subflow_node_registry()
            for output_mapping, code in cases:
                with self.subTest(code=code):
                    result = validate_flow_definition(
                        _parent_definition(
                            base,
                            config={"output_mapping": output_mapping},
                        ),
                        registry,
                    )

                    self.assertFalse(result.ok)
                    self.assertEqual(result.diagnostics[0].code, code)

    def test_validate_flow_definition_rejects_uncompilable_subflow(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_child_flow(base / "child.toml")
            registry = subflow_node_registry()

            with patch(
                "avalan.flow.subflow.compile_flow_definition",
                return_value=FlowPlanCompileResult(),
            ):
                result = validate_flow_definition(
                    _parent_definition(base),
                    registry,
                )

            self.assertFalse(result.ok)
            self.assertEqual(
                result.diagnostics[0].code, "flow.invalid_subflow"
            )

    def test_local_subflow_resolver_caches_compiled_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_child_flow(base / "child.toml")
            resolver = LocalFlowSubflowResolver()
            registry = subflow_node_registry(resolver)
            definition = _parent_definition(base)
            node = definition.nodes[0]

            first = resolver.compile_subflow(
                "child.toml",
                parent_definition=definition,
                node=node,
                registry=registry,
            )
            second = resolver.compile_subflow(
                "child.toml",
                parent_definition=definition,
                node=node,
                registry=registry,
            )

            self.assertIs(first, second)

    def test_local_subflow_resolver_rejects_cycles(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_child_flow(base / "child.toml")
            resolver = LocalFlowSubflowResolver()
            registry = subflow_node_registry(resolver)
            definition = _parent_definition(base)
            node = definition.nodes[0]
            path = (base / "child.toml").resolve()
            resolver._compiling.add(path)

            with self.assertRaises(FlowNodeConfigurationError) as raised:
                resolver.compile_subflow(
                    "child.toml",
                    parent_definition=definition,
                    node=node,
                    registry=registry,
                )

            self.assertEqual(raised.exception.code, "flow.subflow_cycle")

    def test_subflow_node_registry_accepts_base_registry(self) -> None:
        registry = subflow_node_registry(
            LocalFlowSubflowResolver(),
            base_registry=FlowNodeRegistry(),
        )

        self.assertTrue(registry.supports(FLOW_SUBFLOW_NODE_TYPE))


class FlowSubflowPlanTestCase(TestCase):
    def test_compile_flow_definition_embeds_subflow_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_child_flow(base / "child.toml")
            registry = subflow_node_registry()

            result = compile_flow_definition(
                _parent_definition(base),
                registry,
            )

            self.assertTrue(result.ok, result.public_diagnostics)
            assert result.plan is not None
            node = result.plan.node_map["child"]
            self.assertEqual(node.kind, FlowNodeKind.SUBFLOW)
            subflow = node.metadata["subflow"]
            self.assertIsInstance(subflow, Mapping)
            assert isinstance(subflow, Mapping)
            self.assertIn("plan", subflow)
            output_mapping = subflow["output_mapping"]
            assert isinstance(output_mapping, Mapping)
            self.assertEqual(output_mapping["answer"], "answer")


class FlowSubflowRuntimeTestCase(IsolatedAsyncioTestCase):
    async def test_subflow_node_placeholder_requires_execution_plan(
        self,
    ) -> None:
        registry = subflow_node_registry()
        node = registry.build(
            FlowNodeDefinition(
                name="child",
                type=FLOW_SUBFLOW_NODE_TYPE,
                ref="child.toml",
                config={"output_mapping": {"answer": "answer"}},
            )
        )

        with self.assertRaises(FlowNodeConfigurationError) as raised:
            await node.execute_async({})

        self.assertEqual(
            raised.exception.code,
            "flow.execution.subflow_requires_plan",
        )

    async def test_execute_flow_plan_runs_subflow_node(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            _write_child_flow(base / "child.toml")
            registry = subflow_node_registry()
            compile_result = compile_flow_definition(
                _parent_definition(base),
                registry,
            )
            self.assertTrue(
                compile_result.ok,
                compile_result.public_diagnostics,
            )
            assert compile_result.plan is not None

            result = await execute_flow_plan(
                compile_result.plan,
                flow_node_registry_runner(registry),
                inputs={"payload": {"value": 7}},
            )

            self.assertTrue(result.ok, result.public_diagnostics)
            self.assertEqual(result.outputs["answer"], 7)
            self.assertEqual(
                result.node_outputs["child"]["result"],
                {"answer": 7},
            )

    async def test_execute_flow_plan_reports_subflow_failures(self) -> None:
        child_plan = FlowNodePlan(
            name="child",
            type=FLOW_SUBFLOW_NODE_TYPE,
            kind=FlowNodeKind.SUBFLOW,
        )

        result = await execute_flow_plan(
            _runtime_parent_plan(child_plan),
            flow_node_registry_runner(subflow_node_registry()),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.subflow_unavailable",
        )

    async def test_execute_flow_plan_reports_invalid_subflow_metadata(
        self,
    ) -> None:
        child_plan = FlowNodePlan(
            name="child",
            type=FLOW_SUBFLOW_NODE_TYPE,
            kind=FlowNodeKind.SUBFLOW,
            metadata={"subflow": {"plan": object(), "output_mapping": {}}},
        )

        result = await execute_flow_plan(
            _runtime_parent_plan(child_plan),
            flow_node_registry_runner(subflow_node_registry()),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.subflow_unavailable",
        )

    async def test_execute_flow_plan_reports_child_plan_failure(self) -> None:
        child_plan = FlowNodePlan(
            name="child",
            type=FLOW_SUBFLOW_NODE_TYPE,
            kind=FlowNodeKind.SUBFLOW,
            output_contracts=(FlowNodeContract(name="result"),),
            metadata={
                "subflow": {
                    "plan": _failing_child_plan(),
                    "output_mapping": {"answer": "answer"},
                }
            },
        )

        result = await execute_flow_plan(
            _runtime_parent_plan(child_plan),
            flow_node_registry_runner(subflow_node_registry()),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.subflow_failed",
        )


def _write_child_flow(
    path: Path,
    *,
    version: str = "2026-06-07",
    revision: str | None = None,
    input_type: FlowInputType = FlowInputType.OBJECT,
    extra_output: bool = False,
    node_type: str = "echo",
) -> None:
    revision_line = f'revision = "{revision}"' if revision is not None else ""
    extra_output_toml = (
        """
        [[outputs]]
        name = "audit"
        type = "json"
        """
        if extra_output
        else ""
    )
    extra_output_mapping = (
        """
        audit = "start.value"
        """
        if extra_output
        else ""
    )
    path.write_text(
        f"""
        [flow]
        name = "child"
        version = "{version}"
        {revision_line}

        [[inputs]]
        name = "payload"
        type = "{input_type.value}"

        [[outputs]]
        name = "answer"
        type = "object"
        {extra_output_toml}

        [entry]
        type = "node"
        node = "start"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "start.value"
        {extra_output_mapping}

        [nodes.start]
        type = "{node_type}"

        [nodes.start.mapping.value]
        type = "select"
        source = "input.payload"
        """,
        encoding="utf-8",
    )


def _write_graph_child_flow(path: Path, *, valid: bool = True) -> None:
    edge = (
        "start route_1@-->|Private child route| finish"
        if valid
        else "start -->|Private child route| finish"
    )
    path.write_text(
        f"""
        [flow]
        name = "child"
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
        answer = "finish.value"

        [graph]
        format = "mermaid"
        source = "inline"
        mode = "executable"
        diagram = '''
        flowchart LR
        {edge}
        '''

        [nodes.start]
        type = "external"

        [nodes.start.mapping.value]
        type = "select"
        source = "input.payload"

        [nodes.finish]
        type = "external"

        [nodes.finish.mapping.value]
        type = "select"
        source = "start.value"
        """,
        encoding="utf-8",
    )


def _external_node_registry(
    factory: Callable[[FlowNodeDefinition], Node],
) -> FlowNodeRegistry:
    return FlowNodeRegistry(
        {"external": factory},
        {
            "external": FlowNodeMetadata(
                kind=FlowNodeKind.PASS_THROUGH,
                input_contract=FlowNodeContract(
                    name="value",
                    type=FlowInputType.OBJECT,
                    metadata={"dynamic": True},
                ),
                output_contract=FlowNodeContract(
                    name="value",
                    type=FlowOutputType.OBJECT,
                    metadata={"dynamic": True},
                ),
            ),
        },
    )


def _parent_definition(
    base: Path | None,
    *,
    ref: str = "child.toml",
    mappings: tuple[FlowInputMapping, ...] | None = None,
    config: dict[str, object] | None = None,
) -> FlowDefinition:
    return FlowDefinition(
        name="parent",
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
        entry_behavior=FlowEntryBehavior(node="child"),
        output_behavior=FlowOutputBehavior(
            outputs={"answer": "child.result.answer"},
        ),
        nodes=(
            FlowNodeDefinition(
                name="child",
                type=FLOW_SUBFLOW_NODE_TYPE,
                ref=ref,
                mappings=(
                    mappings
                    if mappings is not None
                    else (
                        FlowInputMapping(
                            target="payload",
                            source="input.payload",
                        ),
                    )
                ),
                config=(
                    config
                    if config is not None
                    else {
                        "version": "2026-06-07",
                        "output_mapping": {"answer": "answer"},
                    }
                ),
            ),
        ),
        definition_base=base,
    )


def _runtime_parent_plan(child: FlowNodePlan) -> FlowExecutionPlan:
    return FlowExecutionPlan(
        name="parent-runtime",
        version="2026-06-07",
        revision=None,
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
        entry_node="child",
        output_selectors={
            "answer": parse_flow_selector("child.result.answer")
        },
        nodes=(child,),
    )


def _failing_child_plan() -> FlowExecutionPlan:
    return FlowExecutionPlan(
        name="child-runtime",
        version="2026-06-07",
        revision=None,
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
        entry_node="bad",
        output_selectors={"answer": parse_flow_selector("bad.value")},
        nodes=(
            FlowNodePlan(
                name="bad",
                type="missing",
                kind=FlowNodeKind.PASS_THROUGH,
            ),
        ),
    )


if __name__ == "__main__":
    main()
