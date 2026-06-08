from collections.abc import Iterable, Mapping
from importlib.util import module_from_spec, spec_from_file_location
from json import load
from pathlib import Path
from tomllib import load as load_toml
from types import ModuleType
from typing import cast
from unittest import TestCase, main

from avalan.flow import (
    FlowDefinitionLoader,
    FlowInputType,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodeRegistry,
    FlowOutputType,
    Node,
)
from avalan.task import (
    TaskDefinition,
    TaskFileDescriptor,
    TaskFileSourceKind,
    TaskProviderReferenceKind,
    canonical_json,
    canonical_schema_json,
    load_task_definition,
    validate_task_definition,
    validate_task_input,
    validate_task_output,
)

EXAMPLE_ROOT = Path(__file__).parents[2] / "docs" / "examples" / "tasks"
VALID_EXAMPLES = (
    "minimal_string_agent.task.toml",
    "structured_json.task.toml",
    "file_document.task.toml",
    "file_array_comparison.task.toml",
    "large_direct_file.task.toml",
    "provider_reference_direct.task.toml",
    "poc_extraction/task.toml",
    "poc_extraction/flow_task.toml",
    "poc_extraction/image_flow_task.toml",
    "local_multimodal_media.task.toml",
    "queued_file_task.task.toml",
    "output_artifact.task.toml",
)
INVALID_EXAMPLES = {
    "invalid/path_escape.task.toml": {"execution.path_escape"},
    "invalid/unsafe_privacy.task.toml": {
        "privacy.encryption_key_missing",
        "privacy.raw_retention_required",
    },
    "invalid/unknown_target.task.toml": {"execution.unknown_target"},
    "invalid/invalid_schema.task.toml": {
        "input.invalid_schema",
        "output.invalid_schema",
    },
}


class TaskExamplesTest(TestCase):
    def test_valid_task_examples_load_validate_and_canonicalize(self) -> None:
        for relative_path in VALID_EXAMPLES:
            with self.subTest(example=relative_path):
                definition = load_task_definition(EXAMPLE_ROOT / relative_path)

                self.assertEqual(
                    validate_task_definition(
                        definition,
                        execution_roots=(EXAMPLE_ROOT,),
                    ),
                    (),
                )
                self.assertNotEqual(
                    canonical_json(
                        definition,
                        schema_base_path=EXAMPLE_ROOT / relative_path,
                    ),
                    "",
                )

    def test_structured_json_example_validates_sample_values(self) -> None:
        definition = load_task_definition(
            EXAMPLE_ROOT / "structured_json.task.toml"
        )

        self.assertEqual(
            validate_task_input(
                definition,
                {"question": "What changed?", "priority": 2},
            ),
            (),
        )
        self.assertEqual(
            validate_task_output(
                definition,
                {"answer": "The retry policy changed.", "confidence": 0.8},
            ),
            (),
        )
        self.assertEqual(
            _issue_codes(validate_task_input(definition, {"priority": 2})),
            {"input.invalid_type"},
        )
        self.assertEqual(
            _issue_codes(validate_task_output(definition, {"confidence": 2})),
            {"output.invalid_type"},
        )

    def test_invalid_task_examples_fail_with_documented_codes(self) -> None:
        for relative_path, expected_codes in INVALID_EXAMPLES.items():
            with self.subTest(example=relative_path):
                definition = load_task_definition(EXAMPLE_ROOT / relative_path)

                issues = validate_task_definition(
                    definition,
                    execution_roots=(EXAMPLE_ROOT,),
                )

                self.assertEqual(_issue_codes(issues), expected_codes)
                rendered = " ".join(
                    value
                    for issue in issues
                    for value in issue.as_dict().values()
                )
                self.assertNotIn("../private/agent.toml", rendered)
                self.assertNotIn("OPENAI_API_KEY", rendered)

    def test_sdk_definition_matches_structured_json_toml(self) -> None:
        toml_definition = load_task_definition(
            EXAMPLE_ROOT / "structured_json.task.toml"
        )
        sdk_definition = _load_sdk_module(
            "sdk_definition.py"
        ).build_definition()

        self.assertIsInstance(sdk_definition, TaskDefinition)
        self.assertEqual(
            canonical_json(
                sdk_definition,
                schema_base_path=EXAMPLE_ROOT / "structured_json.task.toml",
            ),
            canonical_json(
                toml_definition,
                schema_base_path=EXAMPLE_ROOT / "structured_json.task.toml",
            ),
        )

    def test_file_input_sdk_examples_build_safe_descriptors(self) -> None:
        module = _load_sdk_module("file_inputs_sdk.py")

        definition = module.build_large_direct_definition()
        local = module.local_document_descriptor("uploads/report.pdf")
        provider_id = module.provider_file_id_descriptor("file_abc123")
        hosted_url = module.hosted_url_descriptor(
            "https://files.example.test/report.pdf"
        )
        object_store = module.object_store_descriptor("gs://bucket/report.pdf")

        self.assertIsInstance(definition, TaskDefinition)
        self.assertIsInstance(local, TaskFileDescriptor)
        self.assertEqual(local.source_kind, TaskFileSourceKind.LOCAL_PATH)
        self.assertEqual(local.conversions[0].name, "text")
        self.assertIsNotNone(provider_id.provider_reference)
        self.assertIsNotNone(hosted_url.provider_reference)
        self.assertIsNotNone(object_store.provider_reference)
        assert provider_id.provider_reference is not None
        assert hosted_url.provider_reference is not None
        assert object_store.provider_reference is not None
        self.assertEqual(
            provider_id.provider_reference.kind,
            TaskProviderReferenceKind.PROVIDER_FILE_ID,
        )
        self.assertEqual(
            hosted_url.provider_reference.kind,
            TaskProviderReferenceKind.HOSTED_URL,
        )
        self.assertEqual(
            object_store.provider_reference.kind,
            TaskProviderReferenceKind.OBJECT_STORE_URI,
        )
        self.assertNotIn(
            "file_abc123",
            str(provider_id.provider_reference.summary()),
        )

    def test_poc_extraction_fixture_contract_and_provenance(self) -> None:
        root = EXAMPLE_ROOT / "poc_extraction"
        task_path = root / "task.toml"
        definition = load_task_definition(task_path)
        flow_definition = load_task_definition(root / "flow_task.toml")
        image_definition = load_task_definition(root / "image_flow_task.toml")
        flow_loader = _poc_flow_loader()
        native_flow = flow_loader.load(root / "flow.toml")
        image_flow = flow_loader.load(root / "image_flow.toml")
        with (root / "invoice.schema.json").open(encoding="utf-8") as file:
            schema = load(file)
        with (root / "agent.toml").open("rb") as file:
            agent_toml = load_toml(file)

        valid_output = _poc_extraction_output()
        invalid_output = _poc_extraction_output()
        line_items = cast(
            list[dict[str, object]],
            invalid_output["line_items"],
        )
        line_items[0]["currency"] = "US"

        self.assertEqual(
            canonical_schema_json(definition.output.schema),
            canonical_schema_json(schema),
        )
        self.assertEqual(
            canonical_schema_json(flow_definition.output.schema),
            canonical_schema_json(schema),
        )
        self.assertEqual(
            canonical_schema_json(image_definition.output.schema),
            canonical_schema_json(schema),
        )
        self.assertEqual(flow_definition.execution.ref, "flow.toml")
        self.assertEqual(image_definition.execution.ref, "image_flow.toml")
        self.assertEqual(
            validate_task_input(
                definition,
                {
                    "source_kind": "local_path",
                    "reference": "./sample.pdf",
                    "mime_type": "application/pdf",
                },
            ),
            (),
        )
        self.assertEqual(
            validate_task_input(
                image_definition,
                {
                    "source_kind": "local_path",
                    "reference": "./sample.pdf",
                    "mime_type": "application/pdf",
                },
            ),
            (),
        )
        self.assertEqual(
            _issue_codes(
                validate_task_input(
                    image_definition,
                    {
                        "source_kind": "local_path",
                        "reference": "./sample.png",
                        "mime_type": "image/png",
                    },
                )
            ),
            {"input.invalid_file"},
        )
        self.assertEqual(validate_task_output(definition, valid_output), ())
        self.assertEqual(
            _issue_codes(validate_task_output(definition, invalid_output)),
            {"output.invalid_type"},
        )
        self.assertEqual(len(native_flow.outputs), 1)
        self.assertEqual(
            native_flow.outputs[0].schema_ref,
            "invoice.schema.json",
        )
        self.assertEqual(len(image_flow.inputs), 1)
        self.assertEqual(image_flow.inputs[0].name, "input")
        self.assertEqual(image_flow.inputs[0].type, FlowInputType.FILE_ARRAY)
        self.assertEqual(image_flow.inputs[0].mime_types, ("application/pdf",))
        self.assertEqual(len(image_flow.outputs), 1)
        self.assertEqual(image_flow.outputs[0].name, "extraction")
        self.assertEqual(image_flow.outputs[0].type, FlowOutputType.OBJECT)
        self.assertEqual(
            image_flow.outputs[0].schema_ref,
            "invoice.schema.json",
        )
        self.assertEqual(
            [(edge.source, edge.target) for edge in image_flow.edges],
            [("render_pages", "extract")],
        )
        image_nodes = image_flow.node_map
        self.assertEqual(image_nodes["render_pages"].type, "file_convert")
        self.assertIsNone(image_nodes["render_pages"].input)
        self.assertEqual(image_nodes["render_pages"].output, "files")
        render_mapping = image_nodes["render_pages"].mappings[0]
        self.assertEqual(render_mapping.target, "files")
        self.assertEqual(render_mapping.kind.value, "file[]")
        self.assertEqual(render_mapping.source, "input.input")
        self.assertEqual(
            dict(image_nodes["render_pages"].config),
            {
                "converter": "pdf_image",
                "format": "png",
                "dpi": 144,
                "pages": "1..",
                "max_pages": 20,
                "max_pixels_per_page": 12000000,
                "max_total_pixels": 120000000,
            },
        )
        self.assertEqual(image_nodes["extract"].type, "agent")
        self.assertEqual(image_nodes["extract"].ref, "agent.toml")
        self.assertEqual(image_nodes["extract"].input, "input")
        self.assertEqual(image_nodes["extract"].output, "extraction")
        extract_mappings = {
            mapping.target: mapping
            for mapping in image_nodes["extract"].mappings
        }
        self.assertEqual(extract_mappings["input"].source, "input.input")
        self.assertEqual(extract_mappings["render_pages"].kind.value, "object")
        self.assertEqual(
            dict(extract_mappings["render_pages"].fields),
            {"files": "render_pages.files"},
        )
        self.assertEqual(
            dict(image_nodes["extract"].config),
            {
                "files_input": "render_pages.files",
                "file_policy": "replace",
            },
        )
        agent = cast(Mapping[str, object], agent_toml["agent"])
        run = cast(Mapping[str, object], agent_toml["run"])
        reasoning = cast(Mapping[str, object], run["reasoning"])
        response_format = cast(Mapping[str, object], run["response_format"])
        self.assertIsInstance(agent["instructions"], str)
        self.assertIsInstance(agent["user"], str)
        self.assertEqual(reasoning["effort"], "high")
        self.assertEqual(response_format["schema_ref"], "invoice.schema.json")

        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in root.rglob("*")
            if path.is_file() and path.suffix != ".pdf"
        )
        self.assertIn('schema_ref = "invoice.schema.json"', combined)
        self.assertIn('type = "agent"', combined)
        self.assertIn('ref = "agent.toml"', combined)
        self.assertIn("instructions =", combined)
        self.assertIn("enable = []", combined)
        self.assertNotIn("system =", combined)
        self.assertNotIn("system instructions", combined)
        self.assertNotIn("run_flow.py", combined)
        self.assertNotIn("Pulumi", combined)
        self.assertNotIn("vdocintel", combined)
        self.assertNotIn("staging", combined.lower())
        self.assertNotIn("LA_Checkmate", combined)
        self.assertNotIn("/".join(("specs", "poc")), combined)

    def test_poc_extraction_docs_record_live_smoke_skip_criteria(
        self,
    ) -> None:
        readme = (EXAMPLE_ROOT / "poc_extraction" / "README.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("outside the default CI path", readme)
        self.assertIn("skipped unless", readme)
        self.assertIn("sanitized non-customer PDF", readme)
        self.assertIn("AZURE_OPENAI_API_KEY", readme)
        self.assertIn("intended deployment", readme)
        self.assertIn("Azure OpenAI", readme)
        self.assertIn("non-empty `line_items` array", readme)
        self.assertIn("network access, live services", readme)
        self.assertNotIn("Pulumi", readme)
        self.assertNotIn("staging", readme.lower())
        self.assertNotIn("LA_Checkmate", readme)
        self.assertNotIn("/".join(("specs", "poc")), readme)


def _issue_codes(issues: Iterable[object]) -> set[str]:
    return {getattr(issue, "code") for issue in issues}


def _poc_flow_loader() -> FlowDefinitionLoader:
    return FlowDefinitionLoader(
        FlowNodeRegistry(
            {
                "agent": _poc_node_factory,
                "file_convert": _poc_node_factory,
            },
            {
                "agent": FlowNodeMetadata(
                    kind=FlowNodeKind.AGENT,
                    supports_ref=True,
                    input_contracts=(
                        FlowNodeContract(name="input", type="any"),
                        FlowNodeContract(
                            name=None,
                            type="object",
                            metadata={"dynamic": True},
                        ),
                    ),
                    output_contract=FlowNodeContract(
                        name="result",
                        type=FlowOutputType.JSON,
                        metadata={"dynamic": True},
                    ),
                ),
                "file_convert": FlowNodeMetadata(
                    kind=FlowNodeKind.FILE_CONVERSION,
                    input_contract=FlowNodeContract(
                        name="files",
                        type=FlowInputType.FILE_ARRAY,
                    ),
                    output_contract=FlowNodeContract(
                        name="files",
                        type=FlowOutputType.FILE_ARRAY,
                    ),
                ),
            },
        )
    )


def _poc_node_factory(definition: FlowNodeDefinition) -> Node:
    return Node(definition.name, func=lambda value: value)


def _poc_extraction_output() -> dict[str, object]:
    return {
        "line_items": [
            {
                "line_number": 1,
                "vendor_name": "Northwind Office Supplies",
                "vendor_address": "42 Market St, Denver, CO 80202",
                "customer_name": "Contoso Research Lab",
                "customer_address": (
                    "100 Example Ave, Suite 1, Denver, CO 80202"
                ),
                "invoice_number": "INV-1001",
                "invoice_date": "01/15/2026",
                "due_date": "02/14/2026",
                "purchase_order": "PO-555100",
                "description": "Document processing services",
                "quantity": "5",
                "unit_price": "25.00",
                "line_amount": "125.00",
                "tax_amount": "0.00",
                "total_amount": "125.00",
                "currency": "USD",
                "notes": "Synthetic invoice fixture",
            }
        ]
    }


def _load_sdk_module(filename: str) -> ModuleType:
    path = EXAMPLE_ROOT / filename
    spec = spec_from_file_location(
        "task_" + filename.replace(".", "_"),
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
