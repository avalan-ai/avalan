from pathlib import Path
from typing import cast
from unittest import TestCase, main

from avalan.task import (
    IdempotencyMode,
    ObservabilitySinkType,
    PrivacyAction,
    RetryBackoff,
    RunMode,
    TaskDefinition,
    TaskDefinitionLoader,
    TaskInputType,
    TaskLoadError,
    TaskLoadIssueCategory,
    TaskOutputType,
    TaskTargetType,
    load_task_definition,
    load_task_definition_result,
    loads_task_definition,
    loads_task_definition_result,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


class TaskDefinitionLoaderTest(TestCase):
    def test_load_minimal_toml_definition_with_defaults(self) -> None:
        definition = load_task_definition(FIXTURE_ROOT / "minimal.task.toml")

        self.assertIsInstance(definition, TaskDefinition)
        self.assertEqual(definition.task.name, "person_explainer")
        self.assertEqual(definition.task.version, "1")
        self.assertEqual(definition.input.type, TaskInputType.STRING)
        self.assertIsNone(definition.input.schema)
        self.assertEqual(definition.output.type, TaskOutputType.TEXT)
        self.assertEqual(definition.execution.type, TaskTargetType.AGENT)
        self.assertEqual(
            definition.execution.ref,
            "agents/person_explainer.toml",
        )
        self.assertEqual(definition.run.mode, RunMode.DIRECT)
        self.assertEqual(definition.run.timeout_seconds, 300)
        self.assertEqual(definition.privacy.input, PrivacyAction.HASH)
        self.assertEqual(
            definition.observability.sinks,
            (ObservabilitySinkType.PGSQL,),
        )

    def test_load_full_toml_definition_converts_optional_sections(
        self,
    ) -> None:
        definition = TaskDefinitionLoader().load(
            FIXTURE_ROOT / "full.task.toml"
        )

        self.assertEqual(definition.task.labels, ("finance", "documents"))
        self.assertEqual(definition.task.annotations["owner"], "ops")
        self.assertEqual(definition.input.type, TaskInputType.FILE_ARRAY)
        self.assertEqual(
            definition.input.file_conversions,
            ("markdown", "text"),
        )
        self.assertEqual(definition.input.mime_types, ("application/pdf",))
        self.assertEqual(definition.output.type, TaskOutputType.JSON)
        self.assertIsNotNone(definition.output.schema)
        schema = cast(dict[str, object], definition.output.schema)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(definition.execution.variables["locale"], "en-US")
        self.assertEqual(definition.run.mode, RunMode.QUEUE)
        self.assertEqual(definition.run.queue, "documents")
        self.assertEqual(
            definition.run.idempotency,
            IdempotencyMode.INPUT_AND_FILES_HASH,
        )
        self.assertEqual(definition.retry.backoff, RetryBackoff.EXPONENTIAL)
        self.assertTrue(definition.retry.jitter)
        self.assertEqual(definition.privacy.output, PrivacyAction.ENCRYPT)
        self.assertTrue(definition.artifact.store_bytes)
        self.assertEqual(definition.artifact.retention_days, 30)
        self.assertEqual(definition.limits.total_tokens, 1000)
        self.assertEqual(
            definition.observability.sinks,
            (
                ObservabilitySinkType.PGSQL,
                ObservabilitySinkType.PROMETHEUS,
            ),
        )
        self.assertFalse(definition.observability.trace)

    def test_loads_preserves_logical_relative_refs(self) -> None:
        definition = loads_task_definition(
            """
            [task]
            name = "relative"
            version = "1"

            [input]
            type = "string"

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "../agents/relative.toml"
            """,
            source_path="/tmp/tasks/relative.task.toml",
        )

        self.assertEqual(definition.execution.ref, "../agents/relative.toml")

    def test_loads_preserves_logical_schema_refs(self) -> None:
        definition = loads_task_definition("""
            [task]
            name = "schema_ref"
            version = "1"

            [input]
            type = "object"
            schema_ref = "schemas/input.json"

            [output]
            type = "json"
            schema_ref = "../schemas/output.json"

            [execution]
            type = "agent"
            ref = "agents/schema_ref.toml"
            """)

        self.assertEqual(definition.input.schema_ref, "schemas/input.json")
        self.assertEqual(
            definition.output.schema_ref, "../schemas/output.json"
        )

    def test_malformed_toml_returns_safe_structured_issue(self) -> None:
        result = loads_task_definition_result(
            "[task\nname = 'broken'",
            source_path="/tmp/private/broken.task.toml",
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.definition)
        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        self.assertEqual(issue.code, "task.malformed_toml")
        self.assertEqual(issue.path, "/tmp/private/broken.task.toml")
        self.assertEqual(issue.category, TaskLoadIssueCategory.PARSE)
        self.assertNotIn("broken", issue.message)
        self.assertEqual(
            issue.as_dict()["hint"],
            "Fix the TOML syntax and retry loading.",
        )

    def test_malformed_toml_without_source_uses_generic_path(self) -> None:
        result = loads_task_definition_result("[task")

        self.assertEqual(result.issues[0].path, "toml")

    def test_load_result_wrapper_returns_definition(self) -> None:
        result = load_task_definition_result(
            FIXTURE_ROOT / "minimal.task.toml"
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.definition)

    def test_missing_required_sections_are_aggregated(self) -> None:
        result = loads_task_definition_result("""
            [task]
            name = "missing"
            version = "1"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(
            [issue.path for issue in result.issues],
            ["input", "output", "execution"],
        )
        self.assertEqual(
            {issue.code for issue in result.issues},
            {"task.missing_section"},
        )

    def test_missing_required_fields_are_aggregated(self) -> None:
        result = loads_task_definition_result("""
            [task]
            version = "1"

            [input]

            [output]

            [execution]
            type = "agent"
            """)

        self.assertEqual(
            [issue.path for issue in result.issues],
            ["task.name", "input.type", "output.type", "execution.ref"],
        )
        self.assertEqual(
            {issue.code for issue in result.issues},
            {"task.missing_field"},
        )

    def test_invalid_enum_values_return_specific_issue_codes(self) -> None:
        result = loads_task_definition_result("""
            [task]
            name = "bad"
            version = "1"

            [input]
            type = "message"

            [output]
            type = "yaml"

            [execution]
            type = "workflow"
            ref = "agents/a.toml"

            [privacy]
            output = "plain"

            [observability]
            sinks = ["unknown"]
            """)

        self.assertEqual(
            [issue.code for issue in result.issues],
            [
                "input.invalid_type",
                "output.invalid_type",
                "execution.unknown_target",
                "privacy.unknown_action",
                "observability.unsupported_sink",
            ],
        )

    def test_invalid_values_do_not_leak_raw_user_values(self) -> None:
        source = """
            [task]
            name = "bad"
            version = "1"
            labels = ["safe", 123]

            [input]
            type = "private-message-secret"
            schema_ref = "/Users/person/private.schema.json"

            [output]
            type = "raw-output-secret"

            [execution]
            type = "workflow-secret"
            ref = "/Users/person/private-agent.toml"

            [privacy]
            output = "plain-secret"
            """

        result = loads_task_definition_result(source)
        rendered = " ".join(
            value
            for issue in result.issues
            for value in issue.as_dict().values()
        )

        self.assertFalse(result.ok)
        self.assertNotIn("private-message-secret", rendered)
        self.assertNotIn("raw-output-secret", rendered)
        self.assertNotIn("workflow-secret", rendered)
        self.assertNotIn("plain-secret", rendered)
        self.assertNotIn("/Users/person/private", rendered)

    def test_invalid_scalar_values_return_structured_issues(self) -> None:
        result = loads_task_definition_result("""
            [task]
            name = "bad"
            version = "1"

            [input]
            type = "string"
            required = "yes"

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "agents/a.toml"

            [run]
            timeout_seconds = "fast"
            """)

        self.assertEqual(
            [issue.path for issue in result.issues],
            ["input.required", "run.timeout_seconds"],
        )
        self.assertEqual(
            {issue.code for issue in result.issues},
            {"task.invalid_type"},
        )

    def test_dataclass_validation_failures_return_section_issue(self) -> None:
        result = loads_task_definition_result("""
            [task]
            name = "bad"
            version = "1"

            [input]
            type = "string"

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "agents/a.toml"

            [artifact]
            retention_days = 0
            """)

        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].code, "artifact.invalid_value")
        self.assertEqual(result.issues[0].path, "artifact")

    def test_explicit_zero_policy_values_are_not_defaulted(self) -> None:
        result = loads_task_definition_result("""
            [task]
            name = "bad"
            version = "1"

            [input]
            type = "string"

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "agents/a.toml"

            [run]
            timeout_seconds = 0

            [retry]
            max_attempts = 0
            """)

        self.assertEqual(
            [issue.code for issue in result.issues],
            ["run.invalid_value", "retry.invalid_value"],
        )
        self.assertEqual(
            [issue.path for issue in result.issues],
            ["run", "retry"],
        )

    def test_load_raises_error_with_structured_issues(self) -> None:
        with self.assertRaises(TaskLoadError) as error:
            TaskDefinitionLoader().loads("[input]\ntype = 'string'")

        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["task", "output", "execution"],
        )
        self.assertIn(
            "task.missing_section at task",
            str(error.exception),
        )

    def test_load_path_raises_error_with_structured_issues(self) -> None:
        with self.assertRaises(TaskLoadError) as error:
            TaskDefinitionLoader().load(
                FIXTURE_ROOT / "missing_sections.task.toml"
            )

        self.assertEqual(error.exception.issues[0].path, "input")

    def test_non_table_section_returns_structure_issue(self) -> None:
        result = loads_task_definition_result("""
            task = "invalid"
            """)

        self.assertEqual(result.issues[0].code, "task.invalid_section")
        self.assertEqual(result.issues[0].path, "task")

    def test_non_string_required_values_return_type_issues(self) -> None:
        result = loads_task_definition_result("""
            [task]
            name = 10
            version = "1"

            [input]
            type = 1

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "agents/a.toml"
            """)

        self.assertEqual(
            [issue.path for issue in result.issues],
            ["task.name", "input.type"],
        )

    def test_invalid_optional_shapes_return_type_issues(self) -> None:
        result = loads_task_definition_result("""
            [task]
            name = "bad"
            version = "1"
            labels = "finance"
            annotations = "metadata"

            [input]
            type = "file"
            file_conversions = "markdown"
            schema = "schema"
            schema_ref = 1

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "agents/a.toml"

            [observability]
            sinks = "pgsql"
            """)

        self.assertEqual(
            [issue.path for issue in result.issues],
            [
                "task.labels",
                "task.annotations",
                "input.schema",
                "input.schema_ref",
                "input.file_conversions",
                "observability.sinks",
            ],
        )

    def test_policy_type_errors_short_circuit_each_policy(self) -> None:
        result = loads_task_definition_result("""
            [task]
            name = "bad"
            version = "1"

            [input]
            type = "string"

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "agents/a.toml"

            [retry]
            jitter = "yes"

            [artifact]
            store_bytes = "yes"

            [limits]
            total_tokens = "many"
            """)

        self.assertEqual(
            [issue.path for issue in result.issues],
            ["retry.jitter", "artifact.store_bytes", "limits.total_tokens"],
        )

    def test_deferred_enum_sections_use_generic_enum_issue(self) -> None:
        result = loads_task_definition_result("""
            [task]
            name = "bad"
            version = "1"

            [input]
            type = "string"

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "agents/a.toml"

            [run]
            mode = "eventually"
            """)

        self.assertEqual(result.issues[0].code, "task.invalid_enum")


if __name__ == "__main__":
    main()
