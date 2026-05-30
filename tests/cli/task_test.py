from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import MagicMock

from rich.console import Console

from avalan.cli.commands import task as task_cmds

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "task" / "fixtures"


class CliTaskValidateTestCase(TestCase):
    def setUp(self) -> None:
        self.theme = MagicMock()

    def test_validate_prints_success_for_valid_definition(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_validate(
            Namespace(definition=str(FIXTURE_ROOT / "minimal.task.toml")),
            console,
            self.theme,
        )

        self.assertTrue(result)
        self.assertIn(
            "Task definition is valid: person_explainer 1",
            console.export_text(),
        )

    def test_validate_prints_load_issues(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_validate(
            Namespace(
                definition=str(FIXTURE_ROOT / "missing_sections.task.toml")
            ),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition could not be loaded.", output)
        self.assertIn("task.missing_section", output)
        self.assertIn("input", output)

    def test_validate_prints_validation_issues(self) -> None:
        console = Console(record=True, width=160)
        with TemporaryDirectory() as tmpdir:
            definition = Path(tmpdir) / "bad.task.toml"
            definition.write_text(
                """
                [task]
                name = "bad"
                version = "1"

                [input]
                type = "string"

                [output]
                type = "object"
                schema = {type = "array"}

                [execution]
                type = "flow"
                ref = "flows/private.toml"
                """,
                encoding="utf-8",
            )

            result = task_cmds.task_validate(
                Namespace(definition=str(definition)),
                console,
                self.theme,
            )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition is invalid.", output)
        self.assertIn("output.invalid_schema", output)
        self.assertIn("execution.unsupported_flow", output)
        self.assertNotIn("flows/private.toml", output)

    def test_validate_missing_file_prints_safe_diagnostic(self) -> None:
        console = Console(record=True, width=160)

        result = task_cmds.task_validate(
            Namespace(definition="/tmp/private/missing.task.toml"),
            console,
            self.theme,
        )

        output = console.export_text()
        self.assertFalse(result)
        self.assertIn("Task definition could not be read.", output)
        self.assertIn("file.read", output)
        self.assertNotIn("/tmp/private/missing.task.toml", output)


if __name__ == "__main__":
    main()
