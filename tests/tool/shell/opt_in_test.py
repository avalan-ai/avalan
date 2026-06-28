from collections.abc import Iterator, Sequence
from unittest import TestCase, main

from avalan.tool.shell import ShellToolSettings
from avalan.tool.shell.opt_in import (
    enables_shell_tools,
    normalize_shell_enabled_tools,
    should_append_shell_toolset,
)


class ShellOptInTest(TestCase):
    def test_normalizes_shell_wildcard_only(self) -> None:
        self.assertEqual(
            normalize_shell_enabled_tools(
                [
                    "shell.*",
                    "shell.rg",
                    "shell",
                    "shellx.*",
                ]
            ),
            ["shell", "shell.rg", "shell", "shellx.*"],
        )

    def test_normalize_preserves_none_and_empty_selection(self) -> None:
        self.assertIsNone(normalize_shell_enabled_tools(None))
        self.assertEqual(normalize_shell_enabled_tools([]), [])

    def test_normalize_rejects_invalid_entries(self) -> None:
        for enabled_tools in ([""], ["  "], [object()]):
            with self.subTest(enabled_tools=enabled_tools):
                with self.assertRaises(AssertionError):
                    normalize_shell_enabled_tools(enabled_tools)  # type: ignore[arg-type]

    def test_enables_shell_truth_table(self) -> None:
        cases = (
            (None, False),
            ([], False),
            (["math"], False),
            (["shell"], True),
            (["shell.*"], True),
            (["shell.rg"], True),
            (["shell.pipeline"], True),
            (["shell."], False),
            (["shellx.*"], False),
        )

        for enabled_tools, expected in cases:
            with self.subTest(enabled_tools=enabled_tools):
                self.assertIs(enables_shell_tools(enabled_tools), expected)

    def test_should_append_truth_table(self) -> None:
        settings = ShellToolSettings()
        cases = (
            (None, None, False),
            (settings, None, True),
            (settings, [], True),
            (None, ["shell"], True),
            (None, ["shell.*"], True),
            (None, ["shell.rg"], True),
            (None, ["shell.pipeline"], True),
            (None, ["shellx.*"], False),
        )

        for shell_settings, enabled_tools, expected in cases:
            with self.subTest(
                shell_settings=shell_settings,
                enabled_tools=enabled_tools,
            ):
                self.assertIs(
                    should_append_shell_toolset(
                        shell_settings=shell_settings,
                        enabled_tools=enabled_tools,
                    ),
                    expected,
                )

    def test_normalize_iterates_selection_once(self) -> None:
        enabled_tools = _CountingTools(["shell.*", "math"])

        self.assertEqual(
            normalize_shell_enabled_tools(enabled_tools),
            ["shell", "math"],
        )
        self.assertEqual(enabled_tools.iterations, 1)


class _CountingTools(Sequence[str]):
    def __init__(self, values: list[str]) -> None:
        self._values = values
        self.iterations = 0

    def __iter__(self) -> Iterator[str]:
        self.iterations += 1
        return iter(self._values)

    def __getitem__(self, index: int) -> str:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


if __name__ == "__main__":
    main()
