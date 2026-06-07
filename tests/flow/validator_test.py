from unittest import TestCase, main

from avalan.flow import (
    FlowDefinition,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeDefinition,
    FlowEntryBehavior,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodeRegistry,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowValidationResult,
    validate_flow_definition,
)
from avalan.flow import loader as flow_loader
from avalan.flow.node import Node


class FlowValidatorTestCase(TestCase):
    def _strict_definition(
        self,
        *,
        nodes: tuple[FlowNodeDefinition, ...],
        output_selector: str = "start.result",
    ) -> FlowDefinition:
        return FlowDefinition(
            name="strict",
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
            entry_behavior=FlowEntryBehavior(node=nodes[0].name),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": output_selector},
            ),
            nodes=nodes,
        )

    def test_validate_flow_definition_accepts_valid_definition(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="valid",
                entrypoint="start",
                output_node="finish",
                nodes=(
                    FlowNodeDefinition(name="start", type="input"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(FlowEdgeDefinition(source="start", target="finish"),),
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())
        self.assertEqual(result.public_diagnostics, ())

    def test_validate_flow_definition_accepts_strict_definition(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
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
                    FlowNodeDefinition(name="prepare", type="input"),
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(source="prepare", target="start"),
                    FlowEdgeDefinition(source="start", target="finish"),
                ),
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())

    def test_validate_flow_definition_accepts_revision_identity(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                revision="rev-1",
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
                    outputs={"answer": "start.result"},
                ),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_strict_contract_gaps(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                runtime_limits={"timeout_seconds": 30},
                input=FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
                output=FlowOutputDefinition(
                    name="result",
                    type=FlowOutputType.OBJECT,
                ),
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.STRING,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.JSON,
                    ),
                ),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.missing_identity",
                "flow.scalar_input_alias",
                "flow.scalar_output_alias",
                "flow.duplicate_input",
                "flow.duplicate_output",
                "flow.missing_entry_behavior",
                "flow.missing_output_behavior",
            ],
        )

    def test_validate_flow_definition_rejects_unknown_strict_references(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
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
                    FlowOutputDefinition(
                        name="audit",
                        type=FlowOutputType.JSON,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="missing"),
                output_behavior=FlowOutputBehavior(
                    outputs={
                        "answer": "missing.result",
                        "extra": "start.result",
                    },
                ),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.unknown_entry_node",
                "flow.unknown_output",
                "flow.missing_output_selection",
                "flow.unknown_output_selector_node",
            ],
        )

    def test_validate_flow_definition_rejects_invalid_output_selector(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
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
                    outputs={"answer": "start"},
                ),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.invalid_output_selector"],
        )

    def test_validate_flow_definition_rejects_topology_output_inference(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                version="2026-06-07",
                output_node="finish",
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
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(FlowEdgeDefinition(source="start", target="finish"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.missing_output_behavior"],
        )

    def test_validation_result_requires_diagnostic_tuple(self) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.invalid_entrypoint",
            path="flow.entrypoint",
            category=FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
            message="Flow entrypoint is invalid.",
        )

        result = FlowValidationResult(diagnostics=(diagnostic,))

        self.assertFalse(result.ok)
        self.assertEqual(
            result.public_diagnostics,
            (diagnostic.as_public_dict(),),
        )
        with self.assertRaises(AssertionError):
            FlowValidationResult(
                diagnostics=[diagnostic],  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowValidationResult(
                diagnostics=(object(),),  # type: ignore[arg-type]
            )

    def test_validate_flow_definition_rejects_node_type_cases(self) -> None:
        registry = FlowNodeRegistry(
            {"external": lambda definition: Node(definition.name)},
            {"external": FlowNodeMetadata(supports_ref=True)},
        )
        cases = (
            ("unknown", "flow.unknown_node_type", "nodes.start.type"),
            ("agent", "flow.unsupported_node_type", "nodes.start.type"),
            ("python", "flow.untrusted_callable", "nodes.start.type"),
        )

        for node_type, code, path in cases:
            with self.subTest(node_type=node_type):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="invalid",
                        entrypoint="start",
                        output_node="start",
                        nodes=(
                            FlowNodeDefinition(
                                name="start",
                                type=node_type,
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, code)
                self.assertEqual(result.diagnostics[0].path, path)

    def test_validate_flow_definition_uses_node_metadata_contracts(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"tool": lambda definition: Node(definition.name)},
            {
                "tool": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    supports_ref=True,
                    async_only=True,
                    output_contract=FlowNodeContract(name="result"),
                    capabilities=(FlowNodeCapability.ASYNC_ONLY,),
                    requires_ref=True,
                    required_config_keys=("mode",),
                ),
            },
        )

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(
                        name="start",
                        type="tool",
                        ref="weather",
                        config={"mode": "safe"},
                    ),
                )
            ),
            registry,
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_metadata_gaps(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "custom": lambda definition: Node(definition.name),
                "tool": lambda definition: Node(definition.name),
            },
            {
                "tool": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    supports_ref=True,
                    output_contract=FlowNodeContract(name="result"),
                    requires_ref=True,
                    required_config_keys=("mode",),
                ),
            },
        )

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(name="start", type="custom"),
                    FlowNodeDefinition(name="tool", type="tool"),
                ),
                output_selector="tool.missing",
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.missing_node_kind",
                "flow.missing_ref",
                "flow.missing_node_config",
                "flow.unknown_node_output",
            ],
        )

    def test_validate_flow_definition_rejects_conflicting_node_kind(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"input": lambda definition: Node(definition.name)},
            {"input": FlowNodeMetadata(kind=FlowNodeKind.TOOL)},
        )

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(FlowNodeDefinition(name="start", type="input"),)
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.invalid_node_kind"],
        )

    def test_validate_flow_definition_rejects_duplicate_nodes(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="duplicate",
                entrypoint="start",
                output_node="start",
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="start", type="echo"),
                ),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.duplicate_node")
        self.assertEqual(result.diagnostics[0].path, "nodes.start")

    def test_validate_flow_definition_rejects_missing_legacy_run_fields(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="missing",
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.missing_entrypoint", "flow.missing_output_node"],
        )

    def test_validate_flow_definition_rejects_missing_strict_contracts(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                version="2026-06-07",
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(outputs={}),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.missing_inputs", "flow.missing_outputs"],
        )

    def test_validate_flow_definition_rejects_private_refs(self) -> None:
        registry = FlowNodeRegistry(
            {"external": lambda definition: Node(definition.name)},
            {"external": FlowNodeMetadata(supports_ref=True)},
        )
        cases = ("../secret", "/secret", "https://host/secret", "dir\\secret")

        for ref in cases:
            with self.subTest(ref=ref):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="private_ref",
                        entrypoint="start",
                        output_node="start",
                        nodes=(
                            FlowNodeDefinition(
                                name="start",
                                type="external",
                                ref=ref,
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertEqual(
                    result.diagnostics[0].code, "flow.path_escape"
                )
                self.assertEqual(
                    result.diagnostics[0].category,
                    FlowDiagnosticCategory.PRIVACY,
                )
                self.assertNotIn("secret", str(result.public_diagnostics))

    def test_loader_validates_semantics_before_node_factories(self) -> None:
        calls: list[str] = []

        def factory(definition: FlowNodeDefinition) -> Node:
            calls.append(definition.name)
            return Node(definition.name)

        result = flow_loader.FlowDefinitionLoader(
            FlowNodeRegistry({"counted": factory})
        ).loads_result(
            """
            [flow]
            name = "invalid"
            entrypoint = "start"
            output_node = "missing"

            [nodes.start]
            type = "counted"
            """
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.unknown_output_node")
        self.assertEqual(calls, [])

    def test_loader_maps_validation_diagnostics_to_load_issues(self) -> None:
        result = flow_loader.loads_flow_definition_result("""
            [flow]
            name = "invalid"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "missing"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.unknown_node_type")
        self.assertEqual(result.issues[0].category.value, "unsupported")
        self.assertEqual(
            result.diagnostics[0].category,
            FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
        )


if __name__ == "__main__":
    main()
