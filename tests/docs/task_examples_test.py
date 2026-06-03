from collections.abc import Iterable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from unittest import TestCase, main

from avalan.task import (
    TaskDefinition,
    canonical_json,
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
        sdk_definition = _load_sdk_module().build_definition()

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


def _issue_codes(issues: Iterable[object]) -> set[str]:
    return {getattr(issue, "code") for issue in issues}


def _load_sdk_module() -> ModuleType:
    path = EXAMPLE_ROOT / "sdk_definition.py"
    spec = spec_from_file_location("task_sdk_definition_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
