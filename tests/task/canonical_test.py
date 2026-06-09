from json import loads
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from async_helpers import run_async

from avalan.task import (
    IdempotencyMode,
    PrivacyAction,
    TaskCanonicalizationError,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskRunPolicy,
    canonical_definition,
    canonical_json,
    load_task_definition,
    loads_task_definition,
    spec_hash,
)

_async_canonical_definition = canonical_definition
_async_canonical_json = canonical_json
_async_load_task_definition = load_task_definition
_async_loads_task_definition = loads_task_definition
_async_spec_hash = spec_hash


def canonical_definition(*args: object, **kwargs: object) -> object:
    return run_async(_async_canonical_definition(*args, **kwargs))


def canonical_json(*args: object, **kwargs: object) -> str:
    return run_async(_async_canonical_json(*args, **kwargs))


def load_task_definition(*args: object, **kwargs: object) -> object:
    return run_async(_async_load_task_definition(*args, **kwargs))


def loads_task_definition(*args: object, **kwargs: object) -> object:
    return run_async(_async_loads_task_definition(*args, **kwargs))


def spec_hash(*args: object, **kwargs: object) -> str:
    return run_async(_async_spec_hash(*args, **kwargs))


FIXTURE_ROOT = Path(__file__).parent / "fixtures"


class TaskCanonicalizationTest(TestCase):
    def test_sdk_and_toml_definitions_share_canonical_identity(self) -> None:
        toml_definition = load_task_definition(
            FIXTURE_ROOT / "minimal.task.toml"
        )
        sdk_definition = TaskDefinition(
            task=TaskMetadata(
                name="person_explainer",
                version="1",
                description="Explain who the person named in the input is.",
            ),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent(
                "agents/person_explainer.toml"
            ),
        )

        self.assertEqual(
            canonical_json(toml_definition),
            canonical_json(sdk_definition),
        )
        self.assertEqual(spec_hash(toml_definition), spec_hash(sdk_definition))

    def test_canonical_json_is_compact_sorted_and_contains_defaults(
        self,
    ) -> None:
        definition = load_task_definition(FIXTURE_ROOT / "minimal.task.toml")
        canonical = canonical_json(definition)
        parsed = loads(canonical)

        self.assertTrue(canonical.startswith('{"artifact":'))
        self.assertNotIn(": ", canonical)
        self.assertEqual(parsed["run"]["mode"], "direct")
        self.assertEqual(parsed["run"]["timeout_seconds"], 300)
        self.assertEqual(parsed["privacy"]["input"], "hash")
        self.assertEqual(parsed["privacy"]["token_text"], "drop")
        self.assertEqual(parsed["observability"]["sinks"], ["pgsql"])

    def test_spec_hash_uses_canonical_definition_not_toml_text(self) -> None:
        compact = loads_task_definition("""
            [task]
            name = "same"
            version = "1"

            [input]
            type = "string"

            [output]
            type = "text"

            [execution]
            type = "agent"
            ref = "agents/same.toml"
            """)
        reordered = loads_task_definition("""
            [execution]
            ref = "agents/same.toml"
            type = "agent"

            [output]
            type = "text"

            [input]
            type = "string"

            [task]
            version = "1"
            name = "same"
            """)

        self.assertEqual(spec_hash(compact), spec_hash(reordered))

    def test_schema_ref_and_inline_schema_share_canonical_identity(
        self,
    ) -> None:
        schema = {
            "type": "object",
            "required": ["total", "vendor"],
            "examples": ["one", "two"],
            "properties": {
                "vendor": {"type": "string"},
                "total": {"type": "number"},
            },
        }
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_path = root / "schemas" / "invoice.schema.json"
            schema_path.parent.mkdir()
            schema_path.write_text(
                """
                {
                  "properties": {
                    "total": {"type": "number"},
                    "vendor": {"type": "string"}
                  },
                  "examples": ["one", "two"],
                  "required": ["vendor", "total"],
                  "type": "object"
                }
                """,
                encoding="utf-8",
            )
            referenced = loads_task_definition(
                """
                [task]
                name = "invoice"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "json"
                schema_ref = "schemas/invoice.schema.json"

                [execution]
                type = "agent"
                ref = "agents/invoice.toml"
                """,
                source_path=root / "invoice.task.toml",
            )
            inline = TaskDefinition(
                task=TaskMetadata(name="invoice", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.json(schema=schema),
                execution=TaskExecutionTarget.agent("agents/invoice.toml"),
            )

            self.assertEqual(
                canonical_json(referenced),
                canonical_json(inline),
            )
            self.assertEqual(
                spec_hash(referenced),
                spec_hash(inline),
            )

    def test_sdk_definition_base_resolves_schema_refs_for_hashing(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_path = root / "schemas" / "invoice.schema.json"
            schema_path.parent.mkdir()
            schema_path.write_text(
                """
                {
                  "type": "object",
                  "properties": {
                    "vendor": {"type": "string"},
                    "total": {"type": "number"}
                  },
                  "required": ["total", "vendor"]
                }
                """,
                encoding="utf-8",
            )
            referenced = TaskDefinition(
                task=TaskMetadata(name="sdk_invoice", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.object(
                    schema_ref="schemas/invoice.schema.json"
                ),
                execution=TaskExecutionTarget.agent("agents/invoice.toml"),
                definition_base=root / "invoice.task.toml",
            )
            inline = TaskDefinition(
                task=TaskMetadata(name="sdk_invoice", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.object(
                    schema={
                        "type": "object",
                        "properties": {
                            "total": {"type": "number"},
                            "vendor": {"type": "string"},
                        },
                        "required": ["vendor", "total"],
                    }
                ),
                execution=TaskExecutionTarget.agent("agents/invoice.toml"),
            )

            self.assertEqual(
                canonical_json(referenced), canonical_json(inline)
            )
            self.assertEqual(spec_hash(referenced), spec_hash(inline))

    def test_definition_base_is_not_canonical_content(self) -> None:
        first = TaskDefinition(
            task=TaskMetadata(name="base_hint", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/base_hint.toml"),
            definition_base="/Users/private/tasks/base_hint.task.toml",
        )
        second = TaskDefinition(
            task=TaskMetadata(name="base_hint", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/base_hint.toml"),
            definition_base="/tmp/other/base_hint.task.toml",
        )

        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(spec_hash(first), spec_hash(second))
        self.assertNotIn("/Users/private", canonical_json(first))

    def test_agent_provider_instructions_change_path_aware_identity(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            agent_path = root / "agents" / "extract.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
                [agent]
                instructions = "private stable policy"
                user = "extract"
                """,
                encoding="utf-8",
            )
            definition = loads_task_definition(
                """
                [task]
                name = "extract"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "text"

                [execution]
                type = "agent"
                ref = "agents/extract.toml"
                """,
                source_path=root / "extract.task.toml",
            )

            first = canonical_json(
                definition,
                schema_base_path=root / "extract.task.toml",
            )
            first_hash = spec_hash(
                definition,
                schema_base_path=root / "extract.task.toml",
            )
            agent_path.write_text(
                """
                [agent]
                instructions = "different private policy"
                user = "extract"
                """,
                encoding="utf-8",
            )
            second = canonical_json(
                definition,
                schema_base_path=root / "extract.task.toml",
            )
            second_hash = spec_hash(
                definition,
                schema_base_path=root / "extract.task.toml",
            )

        self.assertNotEqual(first_hash, second_hash)
        self.assertNotEqual(
            loads(first)["execution"]["provider_instructions_sha256"],
            loads(second)["execution"]["provider_instructions_sha256"],
        )
        self.assertNotIn("private stable policy", first)
        self.assertNotIn("different private policy", second)

    def test_agent_provider_instruction_digest_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            no_agent = root / "no-agent.toml"
            no_agent.write_text('[engine]\nuri = "ai://local/model"', "utf-8")
            invalid_instructions = root / "invalid.toml"
            invalid_instructions.write_text(
                "[agent]\ninstructions = 123\n",
                "utf-8",
            )
            cases = (
                TaskDefinition(
                    task=TaskMetadata(name="absolute_agent", version="1"),
                    input=TaskInputContract.string(),
                    output=TaskOutputContract.text(),
                    execution=TaskExecutionTarget.agent(
                        str(root / "private-agent.toml")
                    ),
                ),
                TaskDefinition(
                    task=TaskMetadata(name="missing_agent", version="1"),
                    input=TaskInputContract.string(),
                    output=TaskOutputContract.text(),
                    execution=TaskExecutionTarget.agent("no-agent.toml"),
                ),
                TaskDefinition(
                    task=TaskMetadata(
                        name="invalid_instructions", version="1"
                    ),
                    input=TaskInputContract.string(),
                    output=TaskOutputContract.text(),
                    execution=TaskExecutionTarget.agent("invalid.toml"),
                ),
            )

            for definition in cases:
                with self.subTest(definition=definition.task.name):
                    canonical = canonical_json(
                        definition,
                        schema_base_path=root / "task.toml",
                    )
                    self.assertIsNone(
                        loads(canonical)["execution"][
                            "provider_instructions_sha256"
                        ]
                    )
                    self.assertNotIn("123", canonical)
                    self.assertNotIn("private-agent", canonical)

    def test_machine_specific_values_do_not_enter_canonical_json(self) -> None:
        first = TaskDefinition(
            task=TaskMetadata(
                name="private",
                version="1",
                annotations={
                    "owner": "ops",
                    "api_key": "sk-first",
                    "home": "/Users/mariano/tasks/private.toml",
                },
            ),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.model(
                "ai://env:OPENAI_API_KEY@openai/gpt-4o",
                variables={
                    "database_url": "postgresql://root:secret@localhost/db",
                    "limit": 3,
                    "password": "first",
                },
            ),
            run=TaskRunPolicy(idempotency=IdempotencyMode.INPUT_HASH),
        )
        second = TaskDefinition(
            task=TaskMetadata(
                name="private",
                version="1",
                annotations={
                    "owner": "ops",
                    "api_key": "sk-second",
                    "home": "/Users/other/tasks/private.toml",
                },
            ),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.model(
                "ai://env:ANTHROPIC_API_KEY@openai/gpt-4o",
                variables={
                    "database_url": "postgresql://root:other@localhost/db",
                    "limit": 3,
                    "password": "second",
                },
            ),
            run=TaskRunPolicy(idempotency=IdempotencyMode.INPUT_HASH),
        )

        canonical = canonical_json(first)

        self.assertNotIn("OPENAI_API_KEY", canonical)
        self.assertNotIn("sk-first", canonical)
        self.assertNotIn("/Users/mariano", canonical)
        self.assertNotIn("postgresql://", canonical)
        self.assertNotIn("first", canonical)
        self.assertEqual(spec_hash(first), spec_hash(second))

    def test_secret_like_values_are_redacted_recursively(self) -> None:
        definition = TaskDefinition(
            task=TaskMetadata(
                name="secret_redaction",
                version="1",
                annotations={
                    "Auth-Token": "token-value",
                    "nested": {
                        "private-key": "key-value",
                        "safe": [
                            "postgresql://root:secret@localhost/db",
                            "/Users/person/private.txt",
                            "public",
                        ],
                    },
                },
            ),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent(
                "agents/a.toml",
                variables={
                    "password": "raw-password",
                    "safe_list": [
                        "mysql://root:secret@localhost/db",
                        "/private/tmp/input.txt",
                    ],
                },
            ),
        )

        canonical = canonical_json(definition)

        self.assertIn('"Auth-Token":"<redacted>"', canonical)
        self.assertIn('"private-key":"<redacted>"', canonical)
        self.assertIn('"password":"<redacted>"', canonical)
        self.assertIn('"safe":["<dsn>","<absolute-path>","public"]', canonical)
        self.assertIn('"safe_list":["<dsn>","<absolute-path>"]', canonical)
        self.assertNotIn("token-value", canonical)
        self.assertNotIn("key-value", canonical)
        self.assertNotIn("raw-password", canonical)
        self.assertNotIn("postgresql://", canonical)
        self.assertNotIn("/Users/person", canonical)

    def test_missing_and_invalid_schema_refs_raise_safe_errors(self) -> None:
        definition = TaskDefinition(
            task=TaskMetadata(name="missing_schema", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.json(schema_ref="schemas/missing.json"),
            execution=TaskExecutionTarget.agent("agents/a.toml"),
        )

        with self.assertRaises(TaskCanonicalizationError) as error:
            canonical_definition(definition)

        self.assertIn("output.schema_ref", str(error.exception))
        self.assertNotIn("missing.json", str(error.exception))

        with TemporaryDirectory() as temporary_directory:
            schema_path = Path(temporary_directory) / "schema.json"
            schema_path.write_text("{", encoding="utf-8")
            invalid_json = TaskDefinition(
                task=TaskMetadata(name="invalid_schema", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.json(schema_ref="schema.json"),
                execution=TaskExecutionTarget.agent("agents/a.toml"),
            )

            with self.assertRaisesRegex(
                TaskCanonicalizationError,
                "JSON schema file",
            ):
                canonical_json(invalid_json, schema_base_path=schema_path)

    def test_schema_refs_reject_remote_and_non_object_sources(self) -> None:
        remote = TaskDefinition(
            task=TaskMetadata(name="remote_schema", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.json(
                schema_ref=(
                    "https://example.test/private/schema.json?token=secret"
                )
            ),
            execution=TaskExecutionTarget.agent("agents/a.toml"),
        )

        with self.assertRaises(TaskCanonicalizationError) as error:
            canonical_json(remote)

        self.assertIn(
            "remote schema references are not supported",
            str(error.exception),
        )
        self.assertNotIn("token=secret", str(error.exception))
        self.assertNotIn("private/schema", str(error.exception))

        with TemporaryDirectory() as temporary_directory:
            schema_path = Path(temporary_directory) / "schema.json"
            schema_path.write_text("[]", encoding="utf-8")
            non_object = TaskDefinition(
                task=TaskMetadata(name="array_schema", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.json(schema_ref="schema.json"),
                execution=TaskExecutionTarget.agent("agents/a.toml"),
            )

            with self.assertRaisesRegex(
                TaskCanonicalizationError,
                "JSON object schema",
            ):
                canonical_json(
                    non_object,
                    schema_base_path=schema_path,
                )

    def test_absolute_schema_refs_are_rejected_safely(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            schema_path = Path(temporary_directory) / "schema.json"
            schema_path.write_text(
                '{"type": "object", "properties": {"id": {"type": "string"}}}',
                encoding="utf-8",
            )
            definition = TaskDefinition(
                task=TaskMetadata(name="absolute_schema", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.json(schema_ref=str(schema_path)),
                execution=TaskExecutionTarget.agent("agents/a.toml"),
            )

            with self.assertRaises(TaskCanonicalizationError) as error:
                canonical_json(definition)

            self.assertIn("output.schema_ref", str(error.exception))
            self.assertNotIn(str(schema_path), str(error.exception))

    def test_schema_refs_reject_path_escape_and_external_refs(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_path = root / "schema.json"
            schema_path.write_text(
                '{"$ref": "other.json"}',
                encoding="utf-8",
            )
            external_ref = TaskDefinition(
                task=TaskMetadata(name="external_ref", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.json(schema_ref="schema.json"),
                execution=TaskExecutionTarget.agent("agents/a.toml"),
            )
            path_escape = TaskDefinition(
                task=TaskMetadata(name="path_escape", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.json(schema_ref="../schema.json"),
                execution=TaskExecutionTarget.agent("agents/a.toml"),
            )

            with self.assertRaises(TaskCanonicalizationError) as external:
                canonical_json(external_ref, schema_base_path=root)
            with self.assertRaises(TaskCanonicalizationError) as escape:
                canonical_json(path_escape, schema_base_path=root)

        self.assertIn("external $ref", str(external.exception))
        self.assertIn("output.schema_ref", str(escape.exception))
        self.assertNotIn("other.json", str(external.exception))
        self.assertNotIn("../schema", str(escape.exception))

    def test_metadata_values_are_normalized_or_rejected(self) -> None:
        definition = TaskDefinition(
            task=TaskMetadata(
                name="metadata",
                version="1",
                annotations={
                    "connection": "postgresql://root:secret@localhost/db",
                    "enabled": True,
                    "missing": None,
                    "nested": {
                        "ratio": 1.5,
                        "values": [1, PrivacyAction.HASH],
                    },
                },
            ),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/a.toml"),
        )
        canonical = canonical_json(definition)

        self.assertIn('"connection":"<dsn>"', canonical)
        self.assertIn('"enabled":true', canonical)
        self.assertIn('"missing":null', canonical)
        self.assertIn('"ratio":1.5', canonical)
        self.assertIn('"values":[1,"hash"]', canonical)

        invalid_definition = TaskDefinition(
            task=TaskMetadata(
                name="bad_metadata",
                version="1",
                annotations={"unsupported": object()},
            ),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/a.toml"),
        )

        with self.assertRaisesRegex(
            TaskCanonicalizationError,
            "definition contains a non-JSON value",
        ):
            canonical_json(invalid_definition)

    def test_non_string_schema_keys_are_rejected(self) -> None:
        definition = TaskDefinition(
            task=TaskMetadata(name="bad_schema_key", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.json(schema={"type": "object"}),
            execution=TaskExecutionTarget.agent("agents/a.toml"),
        )
        object.__setattr__(definition.output, "schema", {1: "bad"})

        with self.assertRaisesRegex(
            TaskCanonicalizationError,
            "schema keys must be strings",
        ):
            canonical_json(definition)


if __name__ == "__main__":
    main()
