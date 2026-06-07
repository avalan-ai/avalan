from unittest import TestCase, main

from avalan.flow import (
    FlowDefinition,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeDefinition,
    FlowNodeDefinition,
    FlowNodeMetadata,
    FlowNodeRegistry,
    FlowValidationResult,
    validate_flow_definition,
)
from avalan.flow import loader as flow_loader
from avalan.flow.node import Node


class FlowValidatorTestCase(TestCase):
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
