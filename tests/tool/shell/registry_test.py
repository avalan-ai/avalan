from importlib import import_module
from unittest import TestCase, main

from avalan.tool.shell.commands.base import ShellStreamContract
from avalan.tool.shell.entities import ShellOutputKind
from avalan.tool.shell.registry import (
    SHELL_COMMAND_DEFINITIONS,
    SHELL_COMMAND_IDS,
    SHELL_COMMANDS,
    ShellCommandDefinition,
    ShellDependencyGroup,
)


def _build_noop_argv(
    context: object,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    return (), (), None


def _text_output_contract(request: object) -> tuple[str, ShellOutputKind]:
    return "text/plain", ShellOutputKind.TEXT


def _uppercase_output_filter(value: str) -> str:
    return value.upper()


_TEXT_STDIN_CONTRACT = ShellStreamContract(
    media_types=("text/plain",),
    output_kinds=(ShellOutputKind.TEXT,),
)
_JSON_STDIN_CONTRACT = ShellStreamContract(
    media_types=("application/json",),
    output_kinds=(ShellOutputKind.JSON,),
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
                "nl",
                "file",
                "find",
                "wc",
                "awk",
                "sed",
                "jq",
                "pdfinfo",
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

    def test_command_definitions_are_owned_by_command_modules(self) -> None:
        for command_id in SHELL_COMMAND_IDS:
            with self.subTest(command_id=command_id):
                module = import_module(
                    f"avalan.tool.shell.commands.{command_id}"
                )
                self.assertIs(
                    SHELL_COMMAND_DEFINITIONS[command_id],
                    module.COMMAND_DEFINITION,
                )
                self.assertIs(
                    SHELL_COMMAND_DEFINITIONS[command_id].argv_builder,
                    module.build_argv,
                )

    def test_command_specific_output_filters_are_owned_by_modules(
        self,
    ) -> None:
        find_module = import_module("avalan.tool.shell.commands.find")
        rg_module = import_module("avalan.tool.shell.commands.rg")
        ls_module = import_module("avalan.tool.shell.commands.ls")

        self.assertIs(
            SHELL_COMMAND_DEFINITIONS["find"].output_filter,
            find_module.filter_output,
        )
        self.assertIs(
            SHELL_COMMAND_DEFINITIONS["rg"].output_filter,
            rg_module.filter_output,
        )
        self.assertIs(
            SHELL_COMMAND_DEFINITIONS["ls"].output_filter,
            ls_module.filter_output,
        )

    def test_command_definitions_record_dependency_groups(self) -> None:
        groups_by_id = {
            command.logical_id: command.dependency_group
            for command in SHELL_COMMANDS
        }

        self.assertEqual(groups_by_id["rg"], ShellDependencyGroup.CORE)
        self.assertEqual(groups_by_id["nl"], ShellDependencyGroup.CORE)
        self.assertEqual(
            groups_by_id["awk"], ShellDependencyGroup.TEXT_FILTERS
        )
        self.assertEqual(groups_by_id["jq"], ShellDependencyGroup.JSON)
        self.assertEqual(groups_by_id["pdfinfo"], ShellDependencyGroup.POPPLER)
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
        stdin_commands = {
            command.logical_id
            for command in SHELL_COMMANDS
            if command.stdin_contract.supports_stdin
        }
        no_double_dash_commands = {
            command.logical_id
            for command in SHELL_COMMANDS
            if not command.supports_double_dash
        }

        self.assertEqual(stdin_commands, {"awk", "sed", "jq", "wc"})
        for command in SHELL_COMMANDS:
            expected_stdin_contract = ShellStreamContract()
            if command.logical_id in {"awk", "sed", "wc"}:
                expected_stdin_contract = _TEXT_STDIN_CONTRACT
            elif command.logical_id == "jq":
                expected_stdin_contract = _JSON_STDIN_CONTRACT
            with self.subTest(stdin_contract=command.logical_id):
                self.assertEqual(
                    command.stdin_contract,
                    expected_stdin_contract,
                )
        self.assertEqual(
            media_commands,
            {"pdfinfo", "pdftotext", "pdftoppm", "tesseract"},
        )
        self.assertEqual(
            no_double_dash_commands,
            {
                "awk",
                "find",
                "sed",
                "pdfinfo",
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
            "argv_builder": _build_noop_argv,
        }
        invalid_values = {
            "logical_id": "",
            "executable_name": "",
            "dependency_group": "core",
            "container_package_hints": (),
            "argv_builder": None,
            "output_contract": None,
            "stdin_contract": None,
            "output_filter": None,
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

        definition = ShellCommandDefinition(
            **{
                **valid,
                "output_contract": _text_output_contract,
                "stdin_contract": _TEXT_STDIN_CONTRACT,
                "output_filter": _uppercase_output_filter,
            }
        )
        self.assertIs(definition.output_contract, _text_output_contract)
        self.assertEqual(definition.stdin_contract, _TEXT_STDIN_CONTRACT)
        self.assertIs(definition.output_filter, _uppercase_output_filter)

    def test_command_definition_defaults_format_text_output(self) -> None:
        definition = ShellCommandDefinition(
            logical_id="cat",
            executable_name="cat",
            dependency_group=ShellDependencyGroup.CORE,
            container_package_hints=("coreutils",),
            argv_builder=_build_noop_argv,
        )

        self.assertEqual(
            definition.output_contract(object()),
            ("text/plain", ShellOutputKind.TEXT),
        )
        self.assertEqual(definition.stdin_contract, ShellStreamContract())
        self.assertEqual(definition.output_filter("visible"), "visible")
        with self.assertRaises(AssertionError):
            definition.output_filter(object())  # type: ignore[arg-type]

    def test_stream_contract_rejects_invalid_metadata(self) -> None:
        valid = {
            "media_types": ("text/plain",),
            "output_kinds": (ShellOutputKind.TEXT,),
        }
        invalid_values = {
            "media_types": [
                ["text/plain"],
                ("",),
                ("text",),
                ("text/plain", "text/plain"),
            ],
            "output_kinds": [
                [ShellOutputKind.TEXT],
                ("text",),
                (ShellOutputKind.TEXT, ShellOutputKind.TEXT),
            ],
        }
        for field_name, values in invalid_values.items():
            for value in values:
                with self.subTest(field_name=field_name, value=value):
                    kwargs = dict(valid)
                    kwargs[field_name] = value
                    with self.assertRaises(AssertionError):
                        ShellStreamContract(**kwargs)  # type: ignore[arg-type]

        for kwargs in (
            {"media_types": ("text/plain",)},
            {"output_kinds": (ShellOutputKind.TEXT,)},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    ShellStreamContract(**kwargs)  # type: ignore[arg-type]

    def test_command_output_filters_reject_invalid_values(self) -> None:
        for command_id in ("rg", "ls", "find"):
            with self.subTest(command_id=command_id):
                with self.assertRaises(AssertionError):
                    SHELL_COMMAND_DEFINITIONS[command_id].output_filter(
                        object()  # type: ignore[arg-type]
                    )


if __name__ == "__main__":
    main()
