from collections.abc import Callable
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock

from avalan.cli.theme_registry import (
    DEFAULT_THEME_NAME,
    SUPPORTED_THEME_NAMES,
    UnsupportedThemeError,
    create_theme,
)


def gettext(message: str) -> str:
    return f"translated:{message}"


def ngettext(singular: str, plural: str, n: int) -> str:
    return singular if n == 1 else plural


class ThemeRegistryTestCase(TestCase):
    def test_default_and_supported_names(self) -> None:
        self.assertEqual(DEFAULT_THEME_NAME, "fancy")
        self.assertEqual(SUPPORTED_THEME_NAMES, ("fancy", "basic"))

    def test_create_fancy_theme_uses_translators(self) -> None:
        class LoadedTheme:
            def __init__(
                self,
                gettext_arg: Callable[[str], str],
                ngettext_arg: Callable[[str, str, int], str],
            ) -> None:
                self.gettext = gettext_arg
                self.ngettext = ngettext_arg

        importer = MagicMock(
            return_value=SimpleNamespace(FancyTheme=LoadedTheme)
        )

        theme = create_theme(
            "fancy",
            gettext,
            ngettext,
            module_importer=importer,
        )

        importer.assert_called_once_with("avalan.cli.theme.fancy")
        self.assertIsInstance(theme, LoadedTheme)
        self.assertIs(theme.gettext, gettext)
        self.assertIs(theme.ngettext, ngettext)

    def test_create_basic_theme_uses_translators(self) -> None:
        class LoadedTheme:
            def __init__(
                self,
                gettext_arg: Callable[[str], str],
                ngettext_arg: Callable[[str, str, int], str],
            ) -> None:
                self.gettext = gettext_arg
                self.ngettext = ngettext_arg

        importer = MagicMock(
            return_value=SimpleNamespace(BasicTheme=LoadedTheme)
        )

        theme = create_theme(
            "basic",
            gettext,
            ngettext,
            module_importer=importer,
        )

        importer.assert_called_once_with("avalan.cli.theme.basic")
        self.assertIsInstance(theme, LoadedTheme)
        self.assertIs(theme.gettext, gettext)
        self.assertIs(theme.ngettext, ngettext)

    def test_basic_theme_is_concrete_common_theme(self) -> None:
        from avalan.cli.theme import Theme
        from avalan.cli.theme.basic import BasicTheme
        from avalan.cli.theme.fancy import FancyTheme

        theme = create_theme("basic", gettext, ngettext)

        self.assertIsInstance(theme, BasicTheme)
        self.assertIsInstance(theme, Theme)
        self.assertNotIsInstance(theme, FancyTheme)
        self.assertEqual(theme._("message"), "translated:message")
        self.assertEqual(theme._n("one", "many", 2), "many")

    def test_unsupported_theme_raises_without_importing(self) -> None:
        for name in ("plain", "", "../fancy", "avalan.cli.theme.fancy"):
            with self.subTest(name=name):
                importer = MagicMock()

                with self.assertRaisesRegex(
                    UnsupportedThemeError,
                    "Unsupported theme .* Supported themes: fancy, basic.",
                ) as context:
                    create_theme(
                        name,
                        gettext,
                        ngettext,
                        module_importer=importer,
                    )

                self.assertIsInstance(context.exception, ValueError)
                importer.assert_not_called()

    def test_fancy_name_imports_only_fancy_theme(self) -> None:
        imported_modules: list[str] = []

        class LoadedTheme:
            def __init__(
                self,
                gettext_arg: Callable[[str], str],
                ngettext_arg: Callable[[str, str, int], str],
            ) -> None:
                self.gettext = gettext_arg
                self.ngettext = ngettext_arg

        def importer(module_name: str) -> SimpleNamespace:
            imported_modules.append(module_name)
            if module_name != "avalan.cli.theme.fancy":
                raise AssertionError("Unselected theme was imported.")
            return SimpleNamespace(FancyTheme=LoadedTheme)

        theme = create_theme(
            "fancy",
            gettext,
            ngettext,
            module_importer=importer,
        )

        self.assertIsInstance(theme, LoadedTheme)
        self.assertEqual(imported_modules, ["avalan.cli.theme.fancy"])

    def test_basic_name_imports_only_basic_theme(self) -> None:
        imported_modules: list[str] = []

        class LoadedTheme:
            def __init__(
                self,
                gettext_arg: Callable[[str], str],
                ngettext_arg: Callable[[str, str, int], str],
            ) -> None:
                self.gettext = gettext_arg
                self.ngettext = ngettext_arg

        def importer(module_name: str) -> SimpleNamespace:
            imported_modules.append(module_name)
            if module_name != "avalan.cli.theme.basic":
                raise AssertionError("Unselected theme was imported.")
            return SimpleNamespace(BasicTheme=LoadedTheme)

        theme = create_theme(
            "basic",
            gettext,
            ngettext,
            module_importer=importer,
        )

        self.assertIsInstance(theme, LoadedTheme)
        self.assertEqual(imported_modules, ["avalan.cli.theme.basic"])
