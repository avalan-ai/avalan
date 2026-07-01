from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import TestCase, main

from async_helpers import run_async

from avalan.skill import (
    SkillReadLimits,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
)
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
from avalan.task import loader as task_loader_module

_AsyncTaskDefinitionLoader = TaskDefinitionLoader
_async_load_task_definition = load_task_definition
_async_load_task_definition_result = load_task_definition_result
_async_loads_task_definition = loads_task_definition
_async_loads_task_definition_result = loads_task_definition_result


class TaskDefinitionLoader(_AsyncTaskDefinitionLoader):  # type: ignore[no-redef]
    def load(self, *args: object, **kwargs: object) -> object:
        loader = _AsyncTaskDefinitionLoader(
            encoding=self._encoding,
            skills_settings=self._skills_settings,
        )
        return run_async(loader.load(*args, **kwargs))

    def load_result(self, *args: object, **kwargs: object) -> object:
        loader = _AsyncTaskDefinitionLoader(
            encoding=self._encoding,
            skills_settings=self._skills_settings,
        )
        return run_async(loader.load_result(*args, **kwargs))

    def loads(self, *args: object, **kwargs: object) -> object:
        loader = _AsyncTaskDefinitionLoader(
            encoding=self._encoding,
            skills_settings=self._skills_settings,
        )
        return run_async(loader.loads(*args, **kwargs))

    def loads_result(self, *args: object, **kwargs: object) -> object:
        loader = _AsyncTaskDefinitionLoader(
            encoding=self._encoding,
            skills_settings=self._skills_settings,
        )
        return run_async(loader.loads_result(*args, **kwargs))


def load_task_definition(*args: object, **kwargs: object) -> object:
    return run_async(_async_load_task_definition(*args, **kwargs))


def load_task_definition_result(*args: object, **kwargs: object) -> object:
    return run_async(_async_load_task_definition_result(*args, **kwargs))


def loads_task_definition(*args: object, **kwargs: object) -> object:
    return run_async(_async_loads_task_definition(*args, **kwargs))


def loads_task_definition_result(*args: object, **kwargs: object) -> object:
    return run_async(_async_loads_task_definition_result(*args, **kwargs))


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

    def test_loads_accepts_trusted_skills_settings_for_eligible_targets(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            settings = _trusted_skills(Path(temporary_directory))
            cases = (
                (TaskTargetType.AGENT, "agents/agent.toml"),
                (TaskTargetType.FLOW, "flows/flow.toml"),
                (TaskTargetType.MODEL, "ai://local/model"),
                (TaskTargetType.TASK, "tasks/child.task.toml"),
                (TaskTargetType.TOOL, "skills.read"),
            )

            for target_type, ref in cases:
                with self.subTest(target_type=target_type.value):
                    definition = loads_task_definition(
                        _skills_task_source(target_type, ref),
                        skills_settings=settings,
                    )

                    assert definition.skills is not None
                    self.assertEqual(
                        tuple(
                            source.label
                            for source in definition.skills.sources
                        ),
                        ("workspace-main",),
                    )
                    self.assertEqual(
                        definition.skills.read_limits.max_lines_per_read,
                        50,
                    )
                    self.assertIsNotNone(definition.skills_config)

    def test_loader_constructor_accepts_trusted_skills_settings(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            settings = _trusted_skills(Path(temporary_directory))

            definition = TaskDefinitionLoader(skills_settings=settings).loads(
                _task_source(TaskTargetType.AGENT, "agents/agent.toml")
            )

        assert isinstance(definition, TaskDefinition)
        self.assertIs(definition.skills, settings)

    def test_schema_resolution_preserves_base_for_skills_tasks(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_path = root / "schema.json"
            schema_path.write_text('{"type":"object"}', encoding="utf-8")
            task_path = root / "task.toml"
            task_path.write_text(
                """
                [task]
                name = "skills_schema_task"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "object"
                schema_ref = "schema.json"

                [execution]
                type = "agent"
                ref = "agents/agent.toml"

                [skills]
                source_labels = ["workspace-main"]
                """,
                encoding="utf-8",
            )
            settings = _trusted_skills(root)

            definition = load_task_definition(
                task_path,
                skills_settings=settings,
            )

        assert isinstance(definition, TaskDefinition)
        self.assertEqual(definition.definition_base, task_path)
        self.assertIsNotNone(definition.output.schema)

    def test_loads_inherits_trusted_skills_settings_without_override(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            settings = _trusted_skills(Path(temporary_directory))

            definition = loads_task_definition(
                _task_source(TaskTargetType.AGENT, "agents/agent.toml"),
                skills_settings=settings,
            )

        self.assertIs(definition.skills, settings)
        self.assertIsNone(definition.skills_config)

    def test_skills_section_requires_trusted_settings(self) -> None:
        result = loads_task_definition_result(
            _skills_task_source(TaskTargetType.AGENT, "agents/agent.toml")
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "task.skills_trusted_settings_required",
        )

    def test_skills_section_must_be_table(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            settings = _trusted_skills(Path(temporary_directory))
            issues: list[object] = []

            skills, skills_config = task_loader_module._skills_settings(
                "not-a-table",  # type: ignore[arg-type]
                settings,
                execution_type=TaskTargetType.AGENT,
                issues=issues,  # type: ignore[arg-type]
            )

        self.assertIsNone(skills)
        self.assertIsNone(skills_config)
        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.code, "task.invalid_type")
        self.assertEqual(issue.path, "skills")

    def test_skills_reject_untrusted_authority(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            settings = _trusted_skills(
                Path(temporary_directory),
                authority_kinds=(SkillSourceAuthorityKind.WORKSPACE,),
            )

            result = loads_task_definition_result(
                _task_source(TaskTargetType.AGENT, "agents/agent.toml") + """

                [skills]
                authority_kinds = ["user_local"]
                """,
                skills_settings=settings,
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "skills.policy_denied")
        self.assertEqual(result.issues[0].path, "skills.authority_kinds")

    def test_skills_reject_unsupported_syntax(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            settings = _trusted_skills(Path(temporary_directory))

            result = loads_task_definition_result(
                _task_source(TaskTargetType.AGENT, "agents/agent.toml") + """

                [skills]
                sources = ["workspace-main"]
                """,
                skills_settings=settings,
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "task.invalid_skills_settings")

    def test_skills_reject_unsafe_source_labels(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            settings = _trusted_skills(Path(temporary_directory))

            result = loads_task_definition_result(
                _task_source(TaskTargetType.AGENT, "agents/agent.toml") + """

                [skills]
                source_labels = ["/Users/private/skills"]
                """,
                skills_settings=settings,
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "task.invalid_skills_settings")

    def test_skills_reject_unknown_fields(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            settings = _trusted_skills(Path(temporary_directory))

            result = loads_task_definition_result(
                _task_source(TaskTargetType.AGENT, "agents/agent.toml") + """

                [skills]
                registry = "remote"
                """,
                skills_settings=settings,
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "task.invalid_skills_settings")

    def test_skills_reject_unsupported_targets(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            settings = _trusted_skills(Path(temporary_directory))

            result = loads_task_definition_result(
                _skills_task_source(
                    TaskTargetType.CALLABLE,
                    "tests.task_target:run",
                ),
                skills_settings=settings,
            )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "task.skills_unsupported_target",
        )
        self.assertEqual(result.issues[0].path, "skills")

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

    def test_loads_resolves_schema_refs_relative_to_source(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "schema_ref.task.toml"
            input_schema = root / "schemas" / "input.json"
            output_schema = root / "schemas" / "output.json"
            input_schema.parent.mkdir()
            input_schema.write_text('{"type": "object"}', encoding="utf-8")
            output_schema.write_text('{"type": "array"}', encoding="utf-8")
            definition = loads_task_definition(
                """
                [task]
                name = "schema_ref"
                version = "1"

                [input]
                type = "object"
                schema_ref = "schemas/input.json"

                [output]
                type = "json"
                schema_ref = "schemas/output.json"

                [execution]
                type = "agent"
                ref = "agents/schema_ref.toml"
                """,
                source_path=source_path,
            )

        self.assertIsNone(definition.input.schema_ref)
        self.assertIsNone(definition.output.schema_ref)
        self.assertEqual(definition.input.schema, {"type": "object"})
        self.assertEqual(definition.output.schema, {"type": "array"})

    def test_schema_refs_require_source_path(self) -> None:
        result = loads_task_definition_result("""
            [task]
            name = "schema_ref"
            version = "1"

            [input]
            type = "string"

            [output]
            type = "json"
            schema_ref = "schemas/output.json"

            [execution]
            type = "agent"
            ref = "agents/schema_ref.toml"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "output.invalid_schema")
        self.assertEqual(result.issues[0].path, "output.schema_ref")

    def test_input_schema_ref_failures_are_classified_safely(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = loads_task_definition_result(
                """
                [task]
                name = "schema_ref"
                version = "1"

                [input]
                type = "object"
                schema_ref = "../private/input.json"

                [output]
                type = "json"

                [execution]
                type = "agent"
                ref = "agents/schema_ref.toml"
                """,
                source_path=root / "schema_ref.task.toml",
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "input.invalid_schema")
        self.assertEqual(result.issues[0].path, "input.schema_ref")

    def test_schema_refs_reject_path_escape_safely(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = loads_task_definition_result(
                """
                [task]
                name = "schema_ref"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "json"
                schema_ref = "../private/output.json"

                [execution]
                type = "agent"
                ref = "agents/schema_ref.toml"
                """,
                source_path=root / "schema_ref.task.toml",
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "output.invalid_schema")
        self.assertEqual(result.issues[0].path, "output.schema_ref")
        rendered = " ".join(result.issues[0].as_dict().values())
        self.assertNotIn("private/output", rendered)

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

    def test_load_result_accepts_explicit_encoding_override(self) -> None:
        result = TaskDefinitionLoader().load_result(
            FIXTURE_ROOT / "minimal.task.toml",
            encoding="utf-8",
        )

        self.assertIsNotNone(result.definition)
        assert result.definition is not None
        self.assertEqual(result.definition.task.name, "person_explainer")

    def test_non_table_section_returns_structure_issue(self) -> None:
        result = loads_task_definition_result("""
            task = "invalid"
            """)

        self.assertEqual(result.issues[0].code, "task.invalid_section")
        self.assertEqual(result.issues[0].path, "task")

    def test_execution_container_section_is_rejected_until_task_phase(
        self,
    ) -> None:
        result = loads_task_definition_result("""
            [task]
            name = "container_task"
            version = "1"

            [input]
            type = "string"

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "agents/a.toml"

            [execution.container]
            profile = "workspace-readonly"
            required = true
            """)

        self.assertFalse(result.ok)
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].code, "container.unsupported_syntax")
        self.assertEqual(result.issues[0].path, "execution.container")
        self.assertEqual(result.issues[0].category.value, "unsupported")

    def test_task_toml_isolation_authority_aliases_are_rejected(
        self,
    ) -> None:
        result = loads_task_definition_result("""
            [task]
            name = "sandbox_task"
            version = "1"

            [input]
            type = "string"

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "agents/a.toml"

            [isolation]
            mode = "sandbox"

            [sandboxProfile]
            backend = "bubblewrap"

            [execution.sandboxPolicy]
            network = "full"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(
            {issue.path for issue in result.issues},
            {
                "isolation",
                "sandboxProfile",
                "execution.sandboxPolicy",
            },
        )
        self.assertEqual(
            {issue.code for issue in result.issues},
            {"isolation.unsupported_syntax"},
        )
        self.assertTrue(
            all(
                issue.category is TaskLoadIssueCategory.UNSUPPORTED
                for issue in result.issues
            )
        )

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

    def test_docs_shell_pipeline_task_examples_load(self) -> None:
        root = Path(__file__).resolve().parents[2]
        examples = (
            (
                "pipeline_agent.task.toml",
                TaskTargetType.AGENT,
                "agents/pipeline_reader.toml",
            ),
            (
                "pipeline_flow.task.toml",
                TaskTargetType.FLOW,
                "pipeline_flow.flow.toml",
            ),
        )

        for name, target_type, ref in examples:
            with self.subTest(name=name):
                definition = TaskDefinitionLoader().load(
                    root / "docs" / "examples" / "tasks" / name
                )

                self.assertIsInstance(definition, TaskDefinition)
                self.assertEqual(definition.execution.type, target_type)
                self.assertEqual(definition.execution.ref, ref)
                self.assertEqual(definition.run.mode, RunMode.DIRECT)


def _trusted_skills(
    root: Path,
    *,
    authority_kinds: tuple[SkillSourceAuthorityKind, ...] = (
        SkillSourceAuthorityKind.WORKSPACE,
    ),
) -> TrustedSkillSettings:
    return TrustedSkillSettings(
        authority_kinds=authority_kinds,
        sources=(
            SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
            ),
        ),
        read_limits=SkillReadLimits(max_lines_per_read=100),
    )


def _task_source(target_type: TaskTargetType, ref: str) -> str:
    return f"""
        [task]
        name = "skills_task"
        version = "1"

        [input]
        type = "string"

        [output]
        type = "text"

        [execution]
        type = "{target_type.value}"
        ref = "{ref}"
        """


def _skills_task_source(target_type: TaskTargetType, ref: str) -> str:
    return _task_source(target_type, ref) + """

        [skills]
        source_labels = ["workspace-main"]

        [skills.read_limits]
        max_lines_per_read = 50
        """


if __name__ == "__main__":
    main()
