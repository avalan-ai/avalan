from unittest import TestCase, main

from avalan.tool.shell.registry import (
    SHELL_COMMAND_DEFINITIONS,
    SHELL_COMMAND_IDS,
    SHELL_COMMANDS,
    ShellCommandDefinition,
    ShellDependencyGroup,
)


class ShellRegistryTest(TestCase):
    def test_command_ids_are_locked_in_public_order(self) -> None:
        self.assertEqual(
            SHELL_COMMAND_IDS,
            (
                "rg",
                "head",
                "tail",
                "ls",
                "cat",
                "wc",
                "awk",
                "sed",
                "jq",
                "pdftotext",
                "pdftoppm",
                "tesseract",
            ),
        )

    def test_command_definitions_are_complete_and_unique(self) -> None:
        self.assertEqual(len(SHELL_COMMANDS), len(SHELL_COMMAND_IDS))
        self.assertEqual(len(set(SHELL_COMMAND_IDS)), len(SHELL_COMMAND_IDS))
        self.assertEqual(
            set(SHELL_COMMAND_DEFINITIONS),
            set(SHELL_COMMAND_IDS),
        )

    def test_command_definitions_record_dependency_groups(self) -> None:
        groups_by_id = {
            command.logical_id: command.dependency_group
            for command in SHELL_COMMANDS
        }

        self.assertEqual(groups_by_id["rg"], ShellDependencyGroup.CORE)
        self.assertEqual(
            groups_by_id["awk"], ShellDependencyGroup.TEXT_FILTERS
        )
        self.assertEqual(groups_by_id["jq"], ShellDependencyGroup.JSON)
        self.assertEqual(
            groups_by_id["pdftoppm"], ShellDependencyGroup.POPPLER
        )
        self.assertEqual(groups_by_id["tesseract"], ShellDependencyGroup.OCR)

    def test_command_definitions_include_complete_backend_metadata(
        self,
    ) -> None:
        for command in SHELL_COMMANDS:
            with self.subTest(command=command.logical_id):
                self.assertTrue(command.public)
                self.assertEqual(command.executable_name, command.logical_id)
                self.assertGreaterEqual(
                    len(command.container_package_hints),
                    1,
                )
                self.assertTrue(
                    all(command.container_package_hints),
                )

        media_commands = {
            command.logical_id
            for command in SHELL_COMMANDS
            if command.media_risk
        }
        no_double_dash_commands = {
            command.logical_id
            for command in SHELL_COMMANDS
            if not command.supports_double_dash
        }

        self.assertEqual(
            media_commands,
            {"pdftotext", "pdftoppm", "tesseract"},
        )
        self.assertEqual(
            no_double_dash_commands,
            {
                "awk",
                "sed",
                "pdftotext",
                "pdftoppm",
                "tesseract",
            },
        )

    def test_command_definition_rejects_invalid_metadata(self) -> None:
        valid = {
            "logical_id": "rg",
            "executable_name": "rg",
            "dependency_group": ShellDependencyGroup.CORE,
            "container_package_hints": ("ripgrep",),
        }
        invalid_values = {
            "logical_id": "",
            "executable_name": "",
            "dependency_group": "core",
            "container_package_hints": (),
            "public": 1,
            "media_risk": 1,
            "supports_double_dash": 1,
        }
        for field_name, value in invalid_values.items():
            with self.subTest(field_name=field_name):
                kwargs = dict(valid)
                kwargs[field_name] = value
                with self.assertRaises(AssertionError):
                    ShellCommandDefinition(**kwargs)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
