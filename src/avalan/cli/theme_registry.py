from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, cast

if TYPE_CHECKING:
    from .theme import Theme
else:
    Theme = Any


ThemeName: TypeAlias = Literal["fancy", "basic"]

DEFAULT_THEME_NAME: Final[ThemeName] = "fancy"
SUPPORTED_THEME_NAMES: Final[tuple[ThemeName, ...]] = ("fancy", "basic")


class UnsupportedThemeError(ValueError):
    """Raise when a theme name is not supported."""


def create_theme(
    name: str,
    gettext: Callable[[str], str],
    ngettext: Callable[[str, str, int], str],
    *,
    module_importer: Callable[[str], Any] = import_module,
) -> Theme:
    """Create a supported CLI theme by name."""
    match name:
        case "fancy":
            module_name = "avalan.cli.theme.fancy"
            class_name = "FancyTheme"
        case "basic":
            module_name = "avalan.cli.theme.basic"
            class_name = "BasicTheme"
        case _:
            supported = ", ".join(SUPPORTED_THEME_NAMES)
            raise UnsupportedThemeError(
                f"Unsupported theme {name!r}. Supported themes: {supported}."
            )

    module = module_importer(module_name)
    theme_class = cast(type[Theme], getattr(module, class_name))
    return theme_class(gettext, ngettext)
