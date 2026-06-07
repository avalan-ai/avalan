from unittest import TestCase, main

from avalan.flow import (
    FlowCondition,
    FlowConditionOperator,
    FlowConditionValueType,
    FlowDefinition,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowEntryBehavior,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowJoinPolicy,
    FlowJoinPolicyType,
    FlowMappingKind,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodeRegistry,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowRouteMatchPolicy,
    FlowValidationResult,
    parse_flow_selector,
    validate_flow_definition,
)
from avalan.flow import loader as flow_loader
from avalan.flow import validator as flow_validator
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

    def test_validate_flow_definition_accepts_nested_output_selector(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"schema": lambda definition: Node(definition.name)},
            {
                "schema": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={
                            "type": "object",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(FlowNodeDefinition(name="start", type="schema"),),
                output_selector="start.result.items[0].name",
            ),
            registry,
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_unsafe_output_selectors(
        self,
    ) -> None:
        cases = (
            ("answer", "env.SECRET", "flow.reserved_selector"),
            ("answer", "start.result/{{secret}}", "flow.unsafe_selector"),
        )

        for output, selector, code in cases:
            with self.subTest(selector=selector):
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
                                name=output,
                                type=FlowOutputType.OBJECT,
                            ),
                        ),
                        entry_behavior=FlowEntryBehavior(node="start"),
                        output_behavior=FlowOutputBehavior(
                            outputs={output: selector},
                        ),
                        nodes=(FlowNodeDefinition(name="start", type="echo"),),
                    )
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, code)
                self.assertEqual(
                    result.diagnostics[0].category,
                    FlowDiagnosticCategory.PRIVACY,
                )
                self.assertNotIn("secret", str(result.public_diagnostics))

    def test_validate_flow_definition_rejects_unknown_selector_path(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"schema": lambda definition: Node(definition.name)},
            {
                "schema": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={
                            "type": "object",
                            "properties": {
                                "known": {"type": "string"},
                                "items": {"type": "array"},
                            },
                        },
                    ),
                ),
            },
        )

        cases = (
            "start.result.missing",
            "start.result.known[0]",
            "start.result.known.name",
        )
        for selector in cases:
            with self.subTest(selector=selector):
                result = validate_flow_definition(
                    self._strict_definition(
                        nodes=(
                            FlowNodeDefinition(name="start", type="schema"),
                        ),
                        output_selector=selector,
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertEqual(
                    result.diagnostics[-1].code,
                    "flow.unknown_selector_path",
                )

    def test_validate_flow_definition_accepts_unknown_nested_schema(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"schema": lambda definition: Node(definition.name)},
            {
                "schema": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={
                            "type": ["object", "null"],
                            "properties": {"items": {"type": "array"}},
                        },
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(FlowNodeDefinition(name="start", type="schema"),),
                output_selector="start.result.items[0].name",
            ),
            registry,
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_accepts_open_selector_contracts(
        self,
    ) -> None:
        cases = (
            (
                FlowNodeMetadata(kind=FlowNodeKind.SELECT),
                "start.result.any[0]",
            ),
            (
                FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        metadata={"dynamic": True},
                    ),
                ),
                "start.result.any",
            ),
            (
                FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema_ref="schemas/result.json",
                    ),
                ),
                "start.result.any",
            ),
            (
                FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(name="result"),
                ),
                "start.result.any",
            ),
            (
                FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={"type": "object"},
                    ),
                ),
                "start.result.any",
            ),
            (
                FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={"type": 7},
                    ),
                ),
                "start.result.any",
            ),
        )

        for metadata, selector in cases:
            with self.subTest(selector=selector, metadata=metadata):
                registry = FlowNodeRegistry(
                    {"open": lambda definition: Node(definition.name)},
                    {"open": metadata},
                )
                result = validate_flow_definition(
                    self._strict_definition(
                        nodes=(FlowNodeDefinition(name="start", type="open"),),
                        output_selector=selector,
                    ),
                    registry,
                )

                self.assertTrue(result.ok)

        selector = parse_flow_selector("start.result.any")
        self.assertFalse(
            flow_validator._supports_output_path(  # type: ignore[attr-defined]
                FlowNodeRegistry(
                    {"open": lambda definition: Node(definition.name)},
                    {
                        "open": FlowNodeMetadata(
                            kind=FlowNodeKind.SELECT,
                            output_contract=FlowNodeContract(name="other"),
                        ),
                    },
                ),
                "open",
                selector,
            )
        )

    def test_validate_flow_definition_rejects_agent_file_selector_contracts(
        self,
    ) -> None:
        cases = (
            (
                "render.missing",
                "flow.unknown_node_output",
                FlowNodeContract(name="files"),
            ),
            (
                "render.files.missing",
                "flow.unknown_selector_path",
                FlowNodeContract(
                    name="files",
                    schema={
                        "type": "object",
                        "properties": {"known": {"type": "string"}},
                    },
                ),
            ),
        )

        for selector, code, output_contract in cases:
            with self.subTest(selector=selector):
                registry = FlowNodeRegistry(
                    {
                        "agent": lambda definition: Node(definition.name),
                        "render": lambda definition: Node(definition.name),
                    },
                    {
                        "agent": FlowNodeMetadata(kind=FlowNodeKind.AGENT),
                        "render": FlowNodeMetadata(
                            kind=FlowNodeKind.FILE_CONVERSION,
                            output_contract=output_contract,
                        ),
                    },
                )
                result = validate_flow_definition(
                    FlowDefinition(
                        name="files",
                        entrypoint="render",
                        output_node="agent",
                        nodes=(
                            FlowNodeDefinition(name="render", type="render"),
                            FlowNodeDefinition(
                                name="agent",
                                type="agent",
                                config={"files_input": selector},
                            ),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="render",
                                target="agent",
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, code)

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

    def test_validate_flow_definition_accepts_declarative_mappings(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "document": lambda definition: Node(definition.name),
                "prepare": lambda definition: Node(definition.name),
                "target": lambda definition: Node(definition.name),
            },
            {
                "document": FlowNodeMetadata(
                    kind=FlowNodeKind.FILE_CONVERSION,
                    output_contracts=(
                        FlowNodeContract(
                            name="file",
                            type=FlowOutputType.FILE,
                        ),
                        FlowNodeContract(
                            name="files",
                            type=FlowOutputType.FILE_ARRAY,
                        ),
                    ),
                ),
                "prepare": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="payload",
                        type=FlowOutputType.OBJECT,
                        schema={
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "units": {"type": "string"},
                                "items": {"type": "array"},
                            },
                        },
                    ),
                ),
                "target": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    input_contracts=(
                        FlowNodeContract(
                            name="arguments",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="aliases",
                            type=FlowInputType.ARRAY,
                        ),
                        FlowNodeContract(
                            name="document",
                            type=FlowInputType.FILE,
                        ),
                        FlowNodeContract(
                            name="attachments",
                            type=FlowInputType.FILE_ARRAY,
                        ),
                        FlowNodeContract(
                            name="renamed",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="merged",
                            type=FlowInputType.OBJECT,
                        ),
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            FlowDefinition(
                name="mapped",
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
                entry_behavior=FlowEntryBehavior(node="prepare"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "target.result"},
                ),
                nodes=(
                    FlowNodeDefinition(name="prepare", type="prepare"),
                    FlowNodeDefinition(name="document", type="document"),
                    FlowNodeDefinition(
                        name="target",
                        type="target",
                        join_policy=FlowJoinPolicy(
                            type=FlowJoinPolicyType.ALL_SUCCESS,
                        ),
                        mappings=(
                            FlowInputMapping(
                                target="arguments",
                                kind=FlowMappingKind.OBJECT,
                                fields={
                                    "city": "prepare.payload.city",
                                    "units": "prepare.payload.units",
                                },
                            ),
                            FlowInputMapping(
                                target="aliases",
                                kind=FlowMappingKind.ARRAY,
                                items=(
                                    "prepare.payload.items[0]",
                                    "input.payload.items[0]",
                                ),
                            ),
                            FlowInputMapping(
                                target="document",
                                kind=FlowMappingKind.FILE,
                                source="document.file",
                            ),
                            FlowInputMapping(
                                target="attachments",
                                kind=FlowMappingKind.FILE_ARRAY,
                                source="document.files",
                            ),
                            FlowInputMapping(
                                target="renamed",
                                kind=FlowMappingKind.RENAME,
                                source="input.payload",
                            ),
                            FlowInputMapping(
                                target="merged",
                                kind=FlowMappingKind.MERGE,
                                sources=(
                                    "input.payload",
                                    "prepare.payload",
                                ),
                            ),
                        ),
                    ),
                ),
                edges=(
                    FlowEdgeDefinition(source="prepare", target="target"),
                    FlowEdgeDefinition(source="document", target="target"),
                ),
            ),
            registry,
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_declarative_mappings(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "document": lambda definition: Node(definition.name),
                "prepare": lambda definition: Node(definition.name),
                "target": lambda definition: Node(definition.name),
            },
            {
                "document": FlowNodeMetadata(
                    kind=FlowNodeKind.FILE_CONVERSION,
                    output_contract=FlowNodeContract(
                        name="file",
                        type=FlowOutputType.FILE,
                    ),
                ),
                "prepare": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="payload",
                        type=FlowOutputType.OBJECT,
                        schema={
                            "type": "object",
                            "properties": {"known": {"type": "string"}},
                        },
                    ),
                ),
                "target": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    input_contracts=(
                        FlowNodeContract(
                            name="arguments",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="count",
                            type=FlowInputType.INTEGER,
                        ),
                        FlowNodeContract(
                            name="document",
                            type=FlowInputType.FILE,
                        ),
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            FlowDefinition(
                name="mapped",
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
                entry_behavior=FlowEntryBehavior(node="prepare"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "target.result"},
                ),
                nodes=(
                    FlowNodeDefinition(name="prepare", type="prepare"),
                    FlowNodeDefinition(name="document", type="document"),
                    FlowNodeDefinition(
                        name="target",
                        type="target",
                        mappings=(
                            FlowInputMapping(
                                target="arguments",
                                kind=FlowMappingKind.OBJECT,
                            ),
                            FlowInputMapping(
                                target="arguments",
                                source="missing.payload",
                            ),
                            FlowInputMapping(
                                target="unknown",
                                source="input.payload",
                            ),
                            FlowInputMapping(
                                target="count",
                                source="input.payload",
                            ),
                            FlowInputMapping(
                                target="document",
                                kind=FlowMappingKind.FILE,
                                source="input.payload",
                            ),
                        ),
                    ),
                ),
                edges=(FlowEdgeDefinition(source="prepare", target="target"),),
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.empty_mapping",
                "flow.duplicate_mapping_target",
                "flow.unknown_mapping_source",
                "flow.unknown_mapping_target",
                "flow.incompatible_mapping",
                "flow.incompatible_mapping",
            ],
        )

    def test_validate_flow_definition_rejects_mapping_edge_cases(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "document": lambda definition: Node(definition.name),
                "prepare": lambda definition: Node(definition.name),
                "target": lambda definition: Node(definition.name),
            },
            {
                "document": FlowNodeMetadata(
                    kind=FlowNodeKind.FILE_CONVERSION,
                    output_contracts=(
                        FlowNodeContract(
                            name="file",
                            type=FlowOutputType.FILE,
                        ),
                        FlowNodeContract(
                            name="files",
                            type=FlowOutputType.FILE_ARRAY,
                        ),
                    ),
                ),
                "prepare": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="payload",
                        type=FlowOutputType.OBJECT,
                        schema={
                            "type": "object",
                            "properties": {"known": {"type": "string"}},
                        },
                    ),
                ),
                "target": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    input_contracts=(
                        FlowNodeContract(
                            name="required",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="arguments",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="items",
                            type=FlowInputType.ARRAY,
                        ),
                        FlowNodeContract(
                            name="text",
                            type=FlowInputType.STRING,
                        ),
                        FlowNodeContract(
                            name="files",
                            type=FlowInputType.FILE_ARRAY,
                        ),
                        FlowNodeContract(
                            name="object_bad",
                            type=FlowInputType.STRING,
                        ),
                        FlowNodeContract(
                            name="merge_bad",
                            type=FlowInputType.STRING,
                        ),
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            FlowDefinition(
                name="mapped",
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
                entry_behavior=FlowEntryBehavior(node="prepare"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "target.result"},
                ),
                nodes=(
                    FlowNodeDefinition(name="prepare", type="prepare"),
                    FlowNodeDefinition(name="document", type="document"),
                    FlowNodeDefinition(
                        name="target",
                        type="target",
                        mappings=(
                            FlowInputMapping(
                                target="arguments",
                                kind=FlowMappingKind.MERGE,
                            ),
                            FlowInputMapping(
                                target="items",
                                kind=FlowMappingKind.ARRAY,
                            ),
                            FlowInputMapping(target="text"),
                            FlowInputMapping(
                                target="files",
                                kind=FlowMappingKind.FILE_ARRAY,
                                source="document.file",
                            ),
                            FlowInputMapping(
                                target="object_bad",
                                kind=FlowMappingKind.OBJECT,
                                fields={"payload": "input.payload"},
                            ),
                            FlowInputMapping(
                                target="merge_bad",
                                kind=FlowMappingKind.MERGE,
                                sources=("input.payload",),
                            ),
                            FlowInputMapping(
                                target="arguments",
                                kind=FlowMappingKind.OBJECT,
                                fields={
                                    "unknown_input": "input.missing",
                                    "disconnected": "document.file",
                                    "unknown_output": "prepare.missing",
                                    "unknown_path": "prepare.payload.missing",
                                },
                            ),
                        ),
                    ),
                ),
                edges=(FlowEdgeDefinition(source="prepare", target="target"),),
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.empty_mapping",
                "flow.empty_mapping",
                "flow.missing_mapping_source",
                "flow.bad_reference",
                "flow.incompatible_mapping",
                "flow.incompatible_mapping",
                "flow.duplicate_mapping_target",
                "flow.unknown_mapping_source",
                "flow.bad_reference",
                "flow.unknown_node_output",
                "flow.unknown_selector_path",
                "flow.missing_input_mapping",
            ],
        )

    def test_validate_flow_definition_accepts_declarative_conditions(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="conditioned",
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
                    outputs={"answer": "finish.value"},
                ),
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(
                        source="start",
                        target="finish",
                        condition=FlowCondition(
                            operator=FlowConditionOperator.ALL,
                            conditions=(
                                FlowCondition(
                                    operator=FlowConditionOperator.EQ,
                                    selector="start.value.status",
                                    value_selector="input.payload.expected",
                                ),
                                FlowCondition(
                                    operator=FlowConditionOperator.IS_TYPE,
                                    selector="start.value.score",
                                    value_type=FlowConditionValueType.NUMBER,
                                ),
                                FlowCondition(
                                    operator=FlowConditionOperator.NOT,
                                    condition=FlowCondition(
                                        operator=FlowConditionOperator.EXISTS,
                                        selector="start.value.blocked",
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_declarative_conditions(
        self,
    ) -> None:
        cases = (
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    value="ready",
                ),
                "flow.missing_condition_selector",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                ),
                "flow.missing_condition_value",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.GT,
                    selector="start.value.score",
                    value="3",
                ),
                "flow.invalid_condition_value",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.CONTAINS,
                    selector="start.value.status",
                    value=3,
                ),
                "flow.invalid_condition_value",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.IS_TYPE,
                    selector="start.value.status",
                ),
                "flow.missing_condition_value_type",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.IN,
                    selector="start.value.status",
                ),
                "flow.missing_condition_values",
            ),
            (
                FlowCondition(operator=FlowConditionOperator.ALL),
                "flow.missing_condition_children",
            ),
            (
                FlowCondition(operator=FlowConditionOperator.NOT),
                "flow.missing_condition_child",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="start.value.status",
                    value="ready",
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="start.value.status",
                    value_selector="start.value.other",
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                    value="ready",
                    value_type=FlowConditionValueType.STRING,
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                    value="ready",
                    values=("ready",),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                    value="ready",
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.status",
                        ),
                    ),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.ALL,
                    selector="start.value.status",
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.status",
                        ),
                    ),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.ALL,
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.status",
                        ),
                    ),
                    condition=FlowCondition(
                        operator=FlowConditionOperator.EXISTS,
                        selector="start.value.other",
                    ),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.NOT,
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.status",
                        ),
                    ),
                    condition=FlowCondition(
                        operator=FlowConditionOperator.EXISTS,
                        selector="start.value.other",
                    ),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="input.missing",
                ),
                "flow.unknown_condition_source",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="missing.value",
                ),
                "flow.unknown_condition_source",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="other.value.status",
                ),
                "flow.bad_reference",
            ),
        )

        for condition, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="conditioned",
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
                            outputs={"answer": "finish.value"},
                        ),
                        nodes=(
                            FlowNodeDefinition(name="start", type="echo"),
                            FlowNodeDefinition(name="other", type="echo"),
                            FlowNodeDefinition(name="finish", type="echo"),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="start",
                                target="finish",
                                condition=condition,
                            ),
                        ),
                    )
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_validate_flow_definition_rejects_private_conditions(
        self,
    ) -> None:
        cases = (
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="env.SECRET",
                ),
                "flow.reserved_selector",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                    value="{{secret}}",
                ),
                "flow.unsafe_condition_value",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                    value={"token": "{{secret}}"},
                ),
                "flow.unsafe_condition_value",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.IN,
                    selector="start.value.status",
                    values=(("{{secret}}",),),
                ),
                "flow.unsafe_condition_value",
            ),
        )

        for condition, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="conditioned",
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
                            outputs={"answer": "finish.value"},
                        ),
                        nodes=(
                            FlowNodeDefinition(name="start", type="echo"),
                            FlowNodeDefinition(name="finish", type="echo"),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="start",
                                target="finish",
                                condition=condition,
                            ),
                        ),
                    )
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )
                self.assertIn(
                    FlowDiagnosticCategory.PRIVACY,
                    {
                        diagnostic.category
                        for diagnostic in result.diagnostics
                        if diagnostic.code == code
                    },
                )
                self.assertNotIn("secret", str(result.public_diagnostics))

    def test_validate_flow_definition_rejects_condition_contract_mismatches(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"schema": lambda definition: Node(definition.name)},
            {
                "schema": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={
                            "type": "object",
                            "properties": {"known": {"type": "string"}},
                        },
                    ),
                ),
            },
        )
        cases = (
            ("start.missing", "flow.unknown_node_output"),
            ("start.result.missing", "flow.unknown_selector_path"),
        )

        for selector, code in cases:
            with self.subTest(selector=selector):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="conditioned",
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
                            FlowNodeDefinition(name="start", type="schema"),
                            FlowNodeDefinition(name="finish", type="schema"),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="start",
                                target="finish",
                                condition=FlowCondition(
                                    operator=FlowConditionOperator.EXISTS,
                                    selector=selector,
                                ),
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_validate_flow_definition_accepts_edge_routing_policy(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="routed",
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
                    outputs={"answer": "approved.value"},
                ),
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="approved", type="echo"),
                    FlowNodeDefinition(name="rejected", type="echo"),
                    FlowNodeDefinition(name="fallback", type="echo"),
                    FlowNodeDefinition(name="timeout", type="echo"),
                    FlowNodeDefinition(name="cleanup", type="echo"),
                    FlowNodeDefinition(name="cancel", type="echo"),
                    FlowNodeDefinition(name="pause", type="echo"),
                    FlowNodeDefinition(name="resume", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(
                        source="start",
                        target="approved",
                        condition=FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.approved",
                        ),
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="rejected",
                        condition=FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.rejected",
                        ),
                        priority=1,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="fallback",
                        default=True,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="timeout",
                        kind=FlowEdgeKind.TIMEOUT,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="cleanup",
                        kind=FlowEdgeKind.FINALLY,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="cancel",
                        kind=FlowEdgeKind.CANCELLATION,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="pause",
                        kind=FlowEdgeKind.PAUSE,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="resume",
                        kind=FlowEdgeKind.RESUME,
                    ),
                ),
            )
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_accepts_all_matching_routes(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="routed",
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
                    outputs={"answer": "left.value"},
                ),
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="left", type="echo"),
                    FlowNodeDefinition(name="right", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(
                        source="start",
                        target="left",
                        routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="right",
                        routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                    ),
                ),
            )
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_bad_route_policy(
        self,
    ) -> None:
        cases = (
            (
                (
                    FlowEdgeDefinition(
                        source="start",
                        target="left",
                        default=True,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="right",
                        default=True,
                    ),
                ),
                "flow.duplicate_default_route",
            ),
            (
                (
                    FlowEdgeDefinition(source="start", target="left"),
                    FlowEdgeDefinition(source="start", target="right"),
                ),
                "flow.ambiguous_route",
            ),
            (
                (
                    FlowEdgeDefinition(
                        source="start",
                        target="left",
                        routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                    ),
                    FlowEdgeDefinition(source="start", target="right"),
                ),
                "flow.mixed_routing_policy",
            ),
            (
                (
                    FlowEdgeDefinition(
                        source="start",
                        target="left",
                        priority=-1,
                    ),
                ),
                "flow.invalid_route_priority",
            ),
            (
                (
                    FlowEdgeDefinition(
                        source="start",
                        target="left",
                        condition=FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value",
                        ),
                        default=True,
                    ),
                ),
                "flow.default_route_condition",
            ),
        )

        for edges, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="routed",
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
                            outputs={"answer": "left.value"},
                        ),
                        nodes=(
                            FlowNodeDefinition(name="start", type="echo"),
                            FlowNodeDefinition(name="left", type="echo"),
                            FlowNodeDefinition(name="right", type="echo"),
                        ),
                        edges=edges,
                    )
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_validate_flow_definition_rejects_legacy_edge_policy(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="legacy",
                entrypoint="start",
                output_node="finish",
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(
                        source="start",
                        target="finish",
                        kind=FlowEdgeKind.ERROR,
                    ),
                ),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.unsupported_edge_policy"],
        )

    def test_validate_flow_definition_accepts_join_policies(self) -> None:
        cases = (
            FlowJoinPolicy(type=FlowJoinPolicyType.ALL_SUCCESS),
            FlowJoinPolicy(type=FlowJoinPolicyType.ALL_DONE),
            FlowJoinPolicy(type=FlowJoinPolicyType.ANY_SUCCESS),
            FlowJoinPolicy(type=FlowJoinPolicyType.QUORUM, quorum=2),
            FlowJoinPolicy(type=FlowJoinPolicyType.FIRST_SUCCESS),
            FlowJoinPolicy(type=FlowJoinPolicyType.FAIL_FAST),
            FlowJoinPolicy(type=FlowJoinPolicyType.COLLECT),
        )

        for join_policy in cases:
            with self.subTest(join_policy=join_policy.type.value):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="joined",
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
                        entry_behavior=FlowEntryBehavior(node="left"),
                        output_behavior=FlowOutputBehavior(
                            outputs={"answer": "finish.value"},
                        ),
                        nodes=(
                            FlowNodeDefinition(name="left", type="echo"),
                            FlowNodeDefinition(name="right", type="echo"),
                            FlowNodeDefinition(
                                name="finish",
                                type="echo",
                                join_policy=join_policy,
                            ),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="left",
                                target="finish",
                            ),
                            FlowEdgeDefinition(
                                source="right",
                                target="finish",
                            ),
                        ),
                    )
                )

                self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_missing_join_policy(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="joined",
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
                entry_behavior=FlowEntryBehavior(node="left"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "finish.value"},
                ),
                nodes=(
                    FlowNodeDefinition(name="left", type="echo"),
                    FlowNodeDefinition(name="right", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(source="left", target="finish"),
                    FlowEdgeDefinition(source="right", target="finish"),
                ),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code, "flow.missing_join_policy"
        )
        self.assertEqual(
            result.diagnostics[0].path,
            "nodes.finish.join_policy",
        )

    def test_validate_flow_definition_rejects_join_policy_details(
        self,
    ) -> None:
        cases = (
            (
                FlowJoinPolicy(type=FlowJoinPolicyType.QUORUM),
                "flow.missing_join_quorum",
            ),
            (
                FlowJoinPolicy(type=FlowJoinPolicyType.QUORUM, quorum=0),
                "flow.invalid_join_quorum",
            ),
            (
                FlowJoinPolicy(type=FlowJoinPolicyType.QUORUM, quorum=3),
                "flow.invalid_join_quorum",
            ),
            (
                FlowJoinPolicy(
                    type=FlowJoinPolicyType.ALL_SUCCESS,
                    quorum=1,
                ),
                "flow.unsupported_join_field",
            ),
            (
                FlowJoinPolicy(
                    type=FlowJoinPolicyType.ALL_DONE,
                    optional_inputs=("audit", "audit"),
                ),
                "flow.duplicate_join_optional_input",
            ),
            (
                FlowJoinPolicy(
                    type=FlowJoinPolicyType.ALL_DONE,
                    optional_inputs=("missing",),
                ),
                "flow.unknown_join_optional_input",
            ),
        )

        for join_policy, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    self._join_policy_contract_definition(join_policy),
                    self._join_policy_registry(),
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_validate_flow_definition_uses_join_optional_inputs(
        self,
    ) -> None:
        result = validate_flow_definition(
            self._join_policy_contract_definition(
                FlowJoinPolicy(
                    type=FlowJoinPolicyType.ALL_DONE,
                    optional_inputs=("audit",),
                ),
                include_payload_mapping=True,
            ),
            self._join_policy_registry(),
        )

        self.assertTrue(result.ok)

        missing_required = validate_flow_definition(
            self._join_policy_contract_definition(
                FlowJoinPolicy(
                    type=FlowJoinPolicyType.ALL_DONE,
                    optional_inputs=("audit",),
                ),
            ),
            self._join_policy_registry(),
        )

        self.assertFalse(missing_required.ok)
        self.assertIn(
            "flow.missing_input_mapping",
            [diagnostic.code for diagnostic in missing_required.diagnostics],
        )

    def test_mapping_private_helpers_cover_type_edges(self) -> None:
        registry = FlowNodeRegistry({"open": lambda definition: Node("open")})

        self.assertIsNone(
            flow_validator._flow_input_type(  # type: ignore[attr-defined]
                FlowDefinition(name="flow", nodes=()),
                "missing",
            )
        )
        self.assertIsNone(
            flow_validator._node_output_type(  # type: ignore[attr-defined]
                registry,
                "missing",
                "result",
            )
        )
        self.assertIsNone(
            flow_validator._node_output_type(  # type: ignore[attr-defined]
                registry,
                "open",
                "result",
            )
        )
        self.assertIsNone(
            flow_validator._mapping_type_compatibility(  # type: ignore[attr-defined]
                FlowInputMapping(
                    target="value",
                    source="input.payload",
                ),
                source_type=FlowInputType.OBJECT,
                target_contract=FlowNodeContract(name="value"),
            )
        )
        self.assertIsNone(
            flow_validator._mapping_type_compatibility(  # type: ignore[attr-defined]
                FlowInputMapping(
                    target="value",
                    source="input.payload",
                ),
                source_type=FlowInputType.OBJECT,
                target_contract=FlowNodeContract(name="value", type="custom"),
            )
        )
        self.assertIsNone(
            flow_validator._mapping_type_compatibility(  # type: ignore[attr-defined]
                FlowInputMapping(
                    target="value",
                    kind=FlowMappingKind.FILE_ARRAY,
                    source="input.files",
                ),
                source_type=FlowInputType.FILE_ARRAY,
                target_contract=FlowNodeContract(
                    name="value",
                    type=FlowInputType.FILE_ARRAY,
                ),
            )
        )
        self.assertEqual(
            flow_validator._mapping_type_compatibility(  # type: ignore[attr-defined]
                FlowInputMapping(
                    target="value",
                    kind=FlowMappingKind.FILE_ARRAY,
                    source="input.payload",
                ),
                source_type=FlowInputType.OBJECT,
                target_contract=FlowNodeContract(
                    name="value",
                    type=FlowInputType.FILE_ARRAY,
                ),
            ),
            "flow.incompatible_mapping",
        )
        cases = (
            (FlowInputType.FILE, "file"),
            (FlowInputType.FILE_ARRAY, "file[]"),
            (FlowInputType.ARRAY, "array"),
            (FlowInputType.STRING, "string"),
            (FlowInputType.INTEGER, "integer"),
            (FlowInputType.NUMBER, "number"),
            (FlowInputType.BOOLEAN, "boolean"),
        )
        for input_type, expected in cases:
            with self.subTest(input_type=input_type):
                self.assertEqual(
                    flow_validator._semantic_type(  # type: ignore[attr-defined]
                        input_type,
                    ),
                    expected,
                )

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

    def _join_policy_registry(self) -> FlowNodeRegistry:
        return FlowNodeRegistry(
            {
                "source": lambda definition: Node(definition.name),
                "target": lambda definition: Node(definition.name),
            },
            {
                "source": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="payload",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                "target": FlowNodeMetadata(
                    kind=FlowNodeKind.JOIN,
                    input_contracts=(
                        FlowNodeContract(
                            name="payload",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="audit",
                            type=FlowInputType.OBJECT,
                        ),
                    ),
                    output_contract=FlowNodeContract(
                        name="result",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
            },
        )

    def _join_policy_contract_definition(
        self,
        join_policy: FlowJoinPolicy,
        *,
        include_payload_mapping: bool = False,
    ) -> FlowDefinition:
        mappings: tuple[FlowInputMapping, ...] = ()
        if include_payload_mapping:
            mappings = (
                FlowInputMapping(
                    target="payload",
                    source="left.payload",
                ),
            )
        return FlowDefinition(
            name="joined",
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
            entry_behavior=FlowEntryBehavior(node="left"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "finish.result"},
            ),
            nodes=(
                FlowNodeDefinition(name="left", type="source"),
                FlowNodeDefinition(name="right", type="source"),
                FlowNodeDefinition(
                    name="finish",
                    type="target",
                    join_policy=join_policy,
                    mappings=mappings,
                ),
            ),
            edges=(
                FlowEdgeDefinition(source="left", target="finish"),
                FlowEdgeDefinition(source="right", target="finish"),
            ),
        )


if __name__ == "__main__":
    main()
