from importlib import import_module
from sys import modules
from types import ModuleType
from unittest import TestCase

_WATCHED_CLI_MODULES = (
    "avalan.cli.display",
    "avalan.cli.display_reducer",
    "avalan.cli.display_snapshot",
    "avalan.cli.stream_presenter",
    "avalan.cli.theme",
    "avalan.cli.theme.basic",
    "avalan.cli.theme.fancy",
    "avalan.cli.theme_registry",
)
_WATCHED_THEME_MODULES = (
    "avalan.cli.theme",
    "avalan.cli.theme.basic",
    "avalan.cli.theme.fancy",
    "avalan.cli.theme_registry",
)
_WATCHED_PRESENTER_FORBIDDEN_MODULES = (
    *_WATCHED_THEME_MODULES,
    "avalan.cli.commands",
    "avalan.cli.commands.agent",
    "avalan.cli.commands.cache",
    "avalan.cli.commands.deploy",
    "avalan.cli.commands.flow",
    "avalan.cli.commands.memory",
    "avalan.cli.commands.model",
    "avalan.cli.commands.task",
    "avalan.cli.commands.tokenizer",
)


class StreamImportBoundaryTestCase(TestCase):
    def test_model_stream_does_not_import_cli_stream_display_modules(
        self,
    ) -> None:
        self._assert_import_does_not_load_cli_modules("avalan.model.stream")

    def test_server_streaming_does_not_import_cli_stream_display_modules(
        self,
    ) -> None:
        self._assert_import_does_not_load_cli_modules(
            "avalan.server.routers.streaming"
        )

    def test_display_reducer_does_not_import_theme_modules(self) -> None:
        watched = ("avalan.cli.display_reducer", *_WATCHED_THEME_MODULES)
        saved_modules = self._saved_modules_for(watched)

        try:
            for name in watched:
                modules.pop(name, None)

            import_module("avalan.cli.display_reducer")

            for name in _WATCHED_THEME_MODULES:
                self.assertNotIn(name, modules)
        finally:
            for name in watched:
                modules.pop(name, None)
            modules.update(saved_modules)

    def test_stream_presenter_does_not_import_theme_or_command_modules(
        self,
    ) -> None:
        watched = (
            "avalan.cli.stream_presenter",
            *_WATCHED_PRESENTER_FORBIDDEN_MODULES,
        )
        saved_modules = self._saved_modules_for(watched)

        try:
            for name in watched:
                modules.pop(name, None)

            import_module("avalan.cli.stream_presenter")

            for name in _WATCHED_PRESENTER_FORBIDDEN_MODULES:
                self.assertNotIn(name, modules)
        finally:
            for name in watched:
                modules.pop(name, None)
            modules.update(saved_modules)

    def _assert_import_does_not_load_cli_modules(
        self,
        module_name: str,
    ) -> None:
        saved_modules = self._saved_modules(module_name)

        try:
            for name in (module_name, *_WATCHED_CLI_MODULES):
                modules.pop(name, None)

            import_module(module_name)

            for name in _WATCHED_CLI_MODULES:
                self.assertNotIn(name, modules)
        finally:
            for name in (module_name, *_WATCHED_CLI_MODULES):
                modules.pop(name, None)
            modules.update(saved_modules)

    def _saved_modules(self, module_name: str) -> dict[str, ModuleType]:
        watched = (module_name, *_WATCHED_CLI_MODULES)
        return self._saved_modules_for(watched)

    def _saved_modules_for(
        self,
        watched: tuple[str, ...],
    ) -> dict[str, ModuleType]:
        return {
            name: module
            for name in watched
            if (module := modules.get(name)) is not None
        }
