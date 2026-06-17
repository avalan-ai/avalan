import subprocess
import sys
import unittest
from argparse import Namespace
from datetime import datetime
from logging import getLogger
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from avalan.cli.commands import cache as cache_cmds
from avalan.cli.commands import model as model_cmds
from avalan.cli.download import tqdm_rich_progress
from avalan.cli.theme import Theme
from avalan.cli.theme.basic import BasicTheme
from avalan.cli.theme.fancy import FancyTheme
from avalan.entities import Model


def _gettext(message: str) -> str:
    return f"translated:{message}"


def _ngettext(singular: str, plural: str, n: int) -> str:
    return singular if n == 1 else plural


def _model() -> Model:
    now = datetime(2024, 1, 1)
    return Model(
        id="model-id",
        parameters=None,
        parameter_types=None,
        inference=None,
        library_name=None,
        license=None,
        pipeline_tag=None,
        tags=[],
        architectures=None,
        model_type=None,
        auto_model=None,
        processor=None,
        gated=False,
        private=False,
        disabled=False,
        last_downloads=0,
        downloads=0,
        likes=0,
        ranking=None,
        author="author",
        created_at=now,
        updated_at=now,
    )


class BasicThemeTestCase(unittest.TestCase):
    def test_importing_basic_theme_does_not_import_fancy(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "import avalan.cli.theme.basic\n"
                    "print('avalan.cli.theme.fancy' in sys.modules)"
                ),
            ],
            capture_output=True,
            check=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False")

    def test_basic_theme_instantiates_as_concrete_theme(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)

        self.assertIsInstance(theme, Theme)
        self.assertNotIsInstance(theme, FancyTheme)
        self.assertEqual(theme._("message"), "translated:message")
        self.assertEqual(theme._n("one", "many", 2), "many")

    def test_basic_theme_inherits_common_defaults(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)

        self.assertEqual(theme.icons["user_input"], "")
        self.assertEqual(theme.icons["agent_output"], "")
        self.assertEqual(
            theme.ask_access_token(),
            "translated:Enter your Huggingface access token",
        )
        self.assertEqual(theme.events([]), None)
        self.assertEqual(
            theme.flow_run_progress(
                "flowchart LR\n",
                node_states={},
                active_nodes=(),
                message="Flow run started.",
                console_width=80,
            ),
            "Flow run started.\n\nflowchart LR\n",
        )
        self.assertIsInstance(theme.download_progress(), tuple)

    def test_cache_list_command_renders_with_basic_theme(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)
        console = MagicMock()
        hub = MagicMock()
        hub.cache_dir = "/cache"
        hub.cache_scan.return_value = []

        cache_cmds.cache_list(
            Namespace(model=None, summary=False),
            console,
            theme,
            hub,
        )

        console.print.assert_called_once_with(
            "translated:Cache: /cache\nModels: translated:none"
        )

    def test_cache_download_command_uses_basic_progress_template(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)
        console = MagicMock()
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = _model()
        hub.download.return_value = "/models/model-id"

        cache_cmds.cache_download(
            Namespace(
                model="model-id",
                skip_hub_access_check=False,
                workers=3,
                local_dir="/models",
                local_dir_symlinks=True,
            ),
            console,
            theme,
            hub,
        )

        hub.download.assert_called_once()
        download_kwargs = hub.download.call_args.kwargs
        self.assertTrue(
            issubclass(download_kwargs["tqdm_class"], tqdm_rich_progress)
        )
        self.assertEqual(download_kwargs["workers"], 3)
        self.assertEqual(download_kwargs["local_dir"], "/models")
        self.assertTrue(download_kwargs["local_dir_use_symlinks"])
        self.assertEqual(console.print.call_count, 3)

    def test_model_display_command_renders_with_basic_theme(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)
        console = MagicMock()
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = _model()
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri.return_value = SimpleNamespace(is_local=False)
        loaded_model = SimpleNamespace(
            config=SimpleNamespace(model_type="text-generation"),
            tokenizer_config=None,
        )

        with patch.object(model_cmds, "ModelManager", return_value=manager):
            model_cmds.model_display(
                Namespace(
                    model="model-id",
                    skip_hub_access_check=False,
                    summary=False,
                    load=False,
                ),
                console,
                theme,
                hub,
                getLogger(__name__),
                model=loaded_model,
            )

        hub.can_access.assert_called_once_with("model-id")
        hub.model.assert_called_once_with("model-id")
        self.assertEqual(console.print.call_count, 2)
