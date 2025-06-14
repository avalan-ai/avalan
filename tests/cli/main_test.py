import sys
from argparse import ArgumentParser, _SubParsersAction
from types import SimpleNamespace
from pathlib import Path
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.cli.__main__ import CLI, HuggingfaceHub


def _collect_progs(parser: ArgumentParser) -> list[str]:
    progs: list[str] = []
    stack = [parser]
    while stack:
        p = stack.pop()
        progs.append(p.prog)
        for action in p._actions:
            if isinstance(action, _SubParsersAction):
                for sub in action.choices.values():
                    stack.append(sub)
    return progs


class CliInitTestCase(TestCase):
    def test_constructor_creates_parser_and_sets_help_full(self):
        logger = MagicMock()
        with (
            patch.object(sys, "argv", ["prog"]),
            patch(
                "avalan.cli.__main__.getlocale",
                return_value=("en_US", "UTF-8"),
            ),
            patch(
                "avalan.cli.__main__.TransformerModel.get_default_device",
                return_value="cpu",
            ),
            patch.object(
                CLI, "_create_parser", wraps=CLI._create_parser
            ) as cp,
        ):
            cli = CLI(logger)

        locales_path = str(
            Path(sys.modules[CLI.__module__].__file__).resolve().parents[3]
            / "locale"
        )
        cp.assert_called_once_with(
            "cpu",
            HuggingfaceHub.DEFAULT_CACHE_DIR,
            locales_path,
            "en_US",
        )
        args = cli._parser.parse_args(["--help-full"])
        self.assertTrue(hasattr(args, "help_full"))


class CliCallTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        from logging import getLogger

        self.logger = getLogger("cli-test")
        with patch.object(sys, "argv", ["prog"]):
            self.cli = CLI(self.logger)
        self.translator = SimpleNamespace(
            gettext=lambda s: s, ngettext=lambda s, p, n: s if n == 1 else p
        )

    async def test_call_runs_main(self):
        console = MagicMock(export_text=lambda: "")
        with (
            patch.object(sys, "argv", ["prog"]),
            patch(
                "avalan.cli.__main__.translation", return_value=self.translator
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", return_value=console),
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch.object(CLI, "_help") as help_mock,
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
        ):
            await self.cli()
        main_mock.assert_awaited_once()
        help_mock.assert_not_called()

    async def test_call_help_full_outputs_all_commands(self):
        records_console = None

        def create_console(*args, **kwargs):
            nonlocal records_console
            from rich.console import Console

            records_console = Console(record=True)
            return records_console

        with (
            patch.object(sys, "argv", ["prog", "--help-full"]),
            patch(
                "avalan.cli.__main__.translation", return_value=self.translator
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", side_effect=create_console),
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
            patch.object(CLI, "_help", wraps=self.cli._help) as help_mock,
        ):
            await self.cli()

        main_mock.assert_not_called()
        self.assertEqual(
            help_mock.call_count, len(_collect_progs(self.cli._parser))
        )
        output = records_console.export_text()
        for prog in _collect_progs(self.cli._parser):
            self.assertIn(prog, output)
