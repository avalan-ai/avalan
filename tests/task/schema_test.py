from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

from async_helpers import run_async

from avalan.task import (
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskSchemaResolutionError,
    canonical_schema_json,
    resolve_schema_ref,
    resolve_task_definition_schemas,
)

_async_resolve_schema_ref = resolve_schema_ref
_async_resolve_task_definition_schemas = resolve_task_definition_schemas


def resolve_schema_ref(*args: object, **kwargs: object) -> object:
    return run_async(_async_resolve_schema_ref(*args, **kwargs))


def resolve_task_definition_schemas(
    *args: object,
    **kwargs: object,
) -> object:
    return run_async(_async_resolve_task_definition_schemas(*args, **kwargs))


class TaskSchemaResolutionTest(TestCase):
    def test_resolve_task_definition_schemas_returns_original_without_refs(
        self,
    ) -> None:
        definition = TaskDefinition(
            task=TaskMetadata(name="plain", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/plain.toml"),
        )

        self.assertIs(
            resolve_task_definition_schemas(
                definition,
                schema_base_path=None,
            ),
            definition,
        )

    def test_resolve_task_definition_schemas_uses_definition_base(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_path = root / "schemas" / "answer.json"
            schema_path.parent.mkdir()
            schema_path.write_text('{"type": "object"}', encoding="utf-8")
            definition = TaskDefinition(
                task=TaskMetadata(name="sdk_ref", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.object(
                    schema_ref="schemas/answer.json"
                ),
                execution=TaskExecutionTarget.agent("agents/sdk_ref.toml"),
                definition_base=root / "sdk_ref.task.toml",
            )

            resolved = resolve_task_definition_schemas(
                definition,
                schema_base_path=None,
            )

        self.assertIsNone(resolved.definition_base)
        self.assertIsNone(resolved.output.schema_ref)
        self.assertEqual(resolved.output.schema, {"type": "object"})

    def test_resolve_task_definition_schemas_clears_unused_base(self) -> None:
        definition = TaskDefinition(
            task=TaskMetadata(name="plain", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/plain.toml"),
            definition_base="tasks/plain.task.toml",
        )

        resolved = resolve_task_definition_schemas(
            definition,
            schema_base_path=None,
        )

        self.assertIsNone(resolved.definition_base)
        self.assertIsNot(resolved, definition)

    def test_resolve_schema_ref_reports_read_errors_safely(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_path = root / "schema.json"
            schema_path.write_text('{"type": "object"}', encoding="utf-8")

            with patch(
                "avalan.task.schema.Path.read_text",
                side_effect=OSError("private filename"),
            ):
                with self.assertRaises(TaskSchemaResolutionError) as error:
                    resolve_schema_ref(
                        "schema.json",
                        schema_base_path=root,
                        path="output.schema_ref",
                    )

        self.assertIn("output.schema_ref", str(error.exception))
        self.assertNotIn("private filename", str(error.exception))

    def test_schema_refs_reject_nonportable_values(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for schema_ref in ("", "schemas\\answer.json", "schema.json#/x"):
                with self.subTest(schema_ref=schema_ref):
                    with self.assertRaises(TaskSchemaResolutionError):
                        resolve_schema_ref(
                            schema_ref,
                            schema_base_path=root,
                            path="output.schema_ref",
                        )

    def test_schema_base_without_suffix_must_resolve(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing_base = root / "missing"

            with self.assertRaises(TaskSchemaResolutionError) as error:
                resolve_schema_ref(
                    "schema.json",
                    schema_base_path=missing_base,
                    path="output.schema_ref",
                )

        self.assertIn("base path", str(error.exception))
        self.assertNotIn(str(missing_base), str(error.exception))

    def test_schema_ref_rejects_symlink_escape_when_supported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outside = root.parent / "outside-schema.json"
            link = root / "schema.json"
            outside.write_text('{"type": "object"}', encoding="utf-8")
            try:
                try:
                    link.symlink_to(outside)
                except (NotImplementedError, OSError):
                    self.skipTest("symlink creation is unavailable")

                with self.assertRaises(TaskSchemaResolutionError) as error:
                    resolve_schema_ref(
                        "schema.json",
                        schema_base_path=root,
                        path="output.schema_ref",
                    )
            finally:
                link.unlink(missing_ok=True)
                outside.unlink(missing_ok=True)

        self.assertIn("output.schema_ref", str(error.exception))
        self.assertNotIn(str(outside), str(error.exception))

    def test_canonical_schema_json_handles_scalars_and_rejects_objects(
        self,
    ) -> None:
        canonical = canonical_schema_json(
            {
                "type": "number",
                "minimum": 1.5,
                "description": None,
            }
        )

        self.assertIn('"minimum":1.5', canonical)
        self.assertIn('"description":null', canonical)
        with self.assertRaises(TaskSchemaResolutionError):
            canonical_schema_json({"bad": object()})


if __name__ == "__main__":
    main()
